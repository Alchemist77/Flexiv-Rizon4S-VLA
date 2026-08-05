
from __future__ import annotations

import argparse
import copy
import os
import random
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import CLIPProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
for path in (PROJECT_ROOT, REPO_ROOT):
    if str(path) in sys.path:
        sys.path.remove(str(path))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(1, str(PROJECT_ROOT))

from models.true_vla_policy import TrueVLACfg, TrueVLAPolicy


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RawVLADataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)

        self.images = np.asarray(data["images"]).astype(np.uint8)
        self.states = np.asarray(data["states"]).astype(np.float32)
        self.actions_raw = np.asarray(data["actions"]).astype(np.float32)
        self.text_ids = np.asarray(data["text_ids"]) if "text_ids" in data else None
        self.vocab = None
        if "vocab" in data:
            self.vocab = data["vocab"].item() if getattr(data["vocab"], "shape", None) == () else data["vocab"]

        if "instructions" in data:
            self.instructions = [str(x) for x in data["instructions"]]
        elif self.text_ids is not None and self.vocab is not None:
            self.instructions = self._decode_instructions(self.text_ids, self.vocab)
        else:
            self.instructions = [""] * len(self.images)

        self.action_scale = np.maximum(np.max(np.abs(self.actions_raw), axis=0), 1e-6).astype(np.float32)
        self.actions = (self.actions_raw / self.action_scale).astype(np.float32)

    @staticmethod
    def _normalize_vocab(vocab_obj):
        if isinstance(vocab_obj, dict):
            return {str(k): int(v) for k, v in vocab_obj.items()}
        if isinstance(vocab_obj, np.ndarray):
            return {str(tok): i for i, tok in enumerate(vocab_obj.tolist())}
        if isinstance(vocab_obj, list):
            return {str(tok): i for i, tok in enumerate(vocab_obj)}
        raise TypeError(f"Unsupported vocab type: {type(vocab_obj)}")

    @classmethod
    def _decode_instructions(cls, text_ids: np.ndarray, vocab_obj) -> List[str]:
        vocab = cls._normalize_vocab(vocab_obj)
        inv = {idx: tok for tok, idx in vocab.items()}
        pad_id = vocab.get("<pad>", 0)
        out = []
        for row in np.asarray(text_ids):
            toks = []
            for tid in np.asarray(row).reshape(-1).tolist():
                tid = int(tid)
                if tid == pad_id:
                    continue
                tok = inv.get(tid, "<unk>")
                if tok == "<pad>":
                    continue
                toks.append(tok)
            out.append(" ".join(toks))
        return out

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {
            "image": self.images[idx],
            "state": torch.tensor(self.states[idx], dtype=torch.float32),
            "action": torch.tensor(self.actions[idx], dtype=torch.float32),
            "instruction": self.instructions[idx],
        }


def build_collate_fn(processor: CLIPProcessor):
    def collate(batch):
        images = [b["image"] for b in batch]
        texts = [b["instruction"] for b in batch]
        proc = processor(text=texts, images=images, return_tensors="pt", padding=True)
        states = torch.stack([b["state"] for b in batch], dim=0)
        actions = torch.stack([b["action"] for b in batch], dim=0)
        return {
            "pixel_values": proc["pixel_values"],
            "input_ids": proc["input_ids"],
            "attention_mask": proc["attention_mask"],
            "state": states,
            "action": actions,
            "instruction": texts,
        }
    return collate


def run_epoch(model, loader, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_count = 0

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            state = batch["state"].to(device)
            action = batch["action"].to(device)

            loss = model.loss(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                state=state,
                actions=action,
            )

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            bs = state.size(0)
            total_loss += float(loss.item()) * bs
            total_count += bs

    return total_loss / max(total_count, 1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-path", type=str, required=True)
    p.add_argument("--clip-path", type=str, required=True)
    p.add_argument("--save-path", type=str, default="checkpoints/true_vla_best.pt")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--early-stop-patience", type=int, default=6)
    p.add_argument("--min-delta", type=float, default=1e-5)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--proprio-hidden-dim", type=int, default=256)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--transformer-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max-text-tokens", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--freeze-clip", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset = RawVLADataset(args.dataset_path)
    print(f"images={dataset.images.shape} states={dataset.states.shape} actions={dataset.actions.shape}")
    print(f"action_scale={dataset.action_scale.tolist()}")

    val_size = max(1, int(len(dataset) * args.val_ratio)) if len(dataset) > 1 else 0
    train_size = len(dataset) - val_size
    if val_size > 0:
        train_set, val_set = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
    else:
        train_set, val_set = dataset, []

    processor = CLIPProcessor.from_pretrained(args.clip_path, use_fast=False)
    collate_fn = build_collate_fn(processor)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    ) if val_size > 0 else []

    cfg = TrueVLACfg(
        clip_path=args.clip_path,
        state_dim=int(dataset.states.shape[1]),
        action_dim=int(dataset.actions.shape[1]),
        hidden_dim=args.hidden_dim,
        proprio_hidden_dim=args.proprio_hidden_dim,
        num_heads=args.num_heads,
        transformer_layers=args.transformer_layers,
        dropout=args.dropout,
        freeze_clip=args.freeze_clip,
        max_text_tokens=args.max_text_tokens,
    )
    model = TrueVLAPolicy(cfg).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-5)

    best_val = float("inf")
    best_epoch = -1
    best_state = None
    patience_left = args.early_stop_patience

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    print(model)
    print(f"device={device}")
    print("Start training true VLA 🚀")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer=optimizer)
        val_loss = run_epoch(model, val_loader, device, optimizer=None) if val_loader else train_loss

        print(f"[epoch {epoch:03d}] train_loss={train_loss:.8f} val_loss={val_loss:.8f}")

        improved = val_loss < (best_val - args.min_delta)
        if improved:
            best_val = val_loss
            best_epoch = epoch
            patience_left = args.early_stop_patience
            best_state = copy.deepcopy(model.state_dict())

            torch.save(
                {
                    "model_state_dict": best_state,
                    "state_dim": cfg.state_dim,
                    "action_dim": cfg.action_dim,
                    "action_scale": dataset.action_scale,
                    "clip_path": args.clip_path,
                    "hidden_dim": args.hidden_dim,
                    "proprio_hidden_dim": args.proprio_hidden_dim,
                    "num_heads": args.num_heads,
                    "transformer_layers": args.transformer_layers,
                    "dropout": args.dropout,
                    "freeze_clip": args.freeze_clip,
                    "max_text_tokens": args.max_text_tokens,
                    "model_type": "true_vla_otter_like",
                },
                args.save_path,
            )
            print(f"  -> best model saved to {args.save_path}")
        else:
            patience_left -= 1
            print(f"  -> no improvement, patience_left={patience_left}")
            if patience_left <= 0:
                print("Early stopping triggered.")
                break

    print("=" * 60)
    print(f"Best epoch   : {best_epoch}")
    print(f"Best val loss: {best_val:.8f}")
    print(f"Saved model  : {args.save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
