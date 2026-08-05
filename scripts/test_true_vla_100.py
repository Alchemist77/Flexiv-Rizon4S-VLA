from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
for path in (PROJECT_ROOT, REPO_ROOT):
    if str(path) in sys.path:
        sys.path.remove(str(path))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(1, str(PROJECT_ROOT))

from envs.mujoco_rizon_env_old import MujocoRizonWrapper
from models.true_vla_policy import TrueVLACfg, TrueVLAPolicy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--instruction", type=str, default="auto")
    p.add_argument("--camera-name", type=str, default="cam")
    p.add_argument("--render-size", type=int, default=512)
    p.add_argument("--model-image-size", type=int, default=224)
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--video-dir", type=str, default="videos_true_vla")
    p.add_argument("--print-actions", action="store_true")
    return p.parse_args()


def resize_for_model(img: np.ndarray, model_image_size: int) -> np.ndarray:
    pil = Image.fromarray(img.astype(np.uint8))
    pil = pil.resize((model_image_size, model_image_size), Image.BILINEAR)
    return np.asarray(pil, dtype=np.uint8)


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    cfg = TrueVLACfg(
        clip_path=ckpt["clip_path"],
        state_dim=int(ckpt["state_dim"]),
        action_dim=int(ckpt["action_dim"]),
        hidden_dim=int(ckpt.get("hidden_dim", 256)),
        proprio_hidden_dim=int(ckpt.get("proprio_hidden_dim", 256)),
        num_heads=int(ckpt.get("num_heads", 8)),
        transformer_layers=int(ckpt.get("transformer_layers", 2)),
        dropout=float(ckpt.get("dropout", 0.1)),
        freeze_clip=bool(ckpt.get("freeze_clip", True)),
        max_text_tokens=int(ckpt.get("max_text_tokens", 8)),
    )
    model = TrueVLAPolicy(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    processor = CLIPProcessor.from_pretrained(cfg.clip_path, use_fast=False)
    action_scale = np.asarray(ckpt["action_scale"], dtype=np.float32)
    return model, processor, action_scale, ckpt


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model, processor, action_scale, ckpt = load_model(args.checkpoint, device)

    env = MujocoRizonWrapper(
        env_name="rizon_pickplace",
        seed=args.seed,
        render_mode="rgb_array",
        camera_name=args.camera_name,
        image_size=args.render_size,
        max_episode_steps=args.max_steps,
    )

    if args.save_video:
        os.makedirs(args.video_dir, exist_ok=True)

    for ep in range(args.episodes):
        if args.instruction == "auto":
            img, state, info = env.reset(seed=args.seed + ep)
            instruction = info.get("instruction", "pick and place the blue box")
        else:
            img, state, info = env.reset(seed=args.seed + ep, instruction=args.instruction)
            instruction = args.instruction

        print(f"\n[test] episode={ep+1}/{args.episodes}")
        print(f"[test] instruction='{instruction}'")

        frames = [img.copy()]
        done = False
        step = 0

        while not done and step < args.max_steps:
            img_model = resize_for_model(img, args.model_image_size)

            proc = processor(
                text=[instruction],
                images=[img_model],
                return_tensors="pt",
                padding=True,
            )
            pixel_values = proc["pixel_values"].to(device)
            input_ids = proc["input_ids"].to(device)
            attention_mask = proc["attention_mask"].to(device)
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(device)

            action_t = model.act(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                state=state_t,
            )

            action = action_t.squeeze(0).cpu().numpy().astype(np.float32)
            action = action * action_scale
            action = np.clip(action, -1.0, 1.0)

            if args.print_actions:
                print(f"[step {step:04d}] action={np.round(action, 4).tolist()}")

            img, state, reward, done, info = env.step(action)
            if step % 5 == 0:
                frames.append(img.copy())
            step += 1

            if done:
                print(
                    f"[test] done at step={step} "
                    f"success={info.get('success', 'N/A')} "
                    f"reward={reward:.4f}"
                )

        if args.save_video:
            out_path = os.path.join(args.video_dir, f"true_vla_ep{ep+1:03d}.mp4")
            imageio.mimsave(out_path, frames, fps=20)
            print(f"[test] saved video: {out_path}")

    env.close()
    print("[info] finished")


if __name__ == "__main__":
    main()