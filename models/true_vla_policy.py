
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel


def sincos_position_embedding(seq_len: int, dim: int) -> torch.Tensor:
    pos = torch.arange(seq_len).float()
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    sinusoid_inp = torch.einsum("i,j->ij", pos, inv_freq)
    emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
    if emb.shape[-1] != dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb


class TextAwareVisualExtraction(nn.Module):
    """
    OTTER-like:
      similarity(text_tokens, image_patches)
      -> soft attention over patches
      -> text-aware visual tokens
    """
    def __init__(self, num_img_patches: int, vision_dim: int, temperature: float = 0.07):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(float(temperature)))
        self.register_buffer("pos_emb", sincos_position_embedding(num_img_patches, vision_dim), persistent=False)

    def forward(
        self,
        image_patch_features: torch.Tensor,  # (B, P, D)
        text_features: torch.Tensor,         # (B, T, D)
    ) -> torch.Tensor:
        similarity = torch.einsum("btd,bpd->btp", text_features, image_patch_features)
        attention = F.softmax(similarity / self.temperature.clamp(1e-4, 100.0), dim=-1)
        pe_image_patch_features = image_patch_features + self.pos_emb.unsqueeze(0).to(image_patch_features.dtype)
        text_aware_features = torch.einsum("btp,bpd->btd", attention, pe_image_patch_features)
        return text_aware_features


class ProprioceptionEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


@dataclass
class TrueVLACfg:
    clip_path: str
    state_dim: int
    action_dim: int
    hidden_dim: int = 256
    proprio_hidden_dim: int = 256
    num_heads: int = 8
    transformer_layers: int = 2
    dropout: float = 0.1
    freeze_clip: bool = True
    max_text_tokens: int = 8


class TrueVLAPolicy(nn.Module):
    """
    Practical OTTER-like policy:
      - CLIP vision patch tokens
      - CLIP text token hidden states
      - text-aware visual extraction
      - state MLP
      - small transformer over modality tokens
      - action MLP
    """
    def __init__(self, cfg: TrueVLACfg):
        super().__init__()
        self.cfg = cfg

        self.clip = CLIPModel.from_pretrained(cfg.clip_path)
        if cfg.freeze_clip:
            for p in self.clip.parameters():
                p.requires_grad = False

        self.clip.eval()
        vision_dim = int(self.clip.vision_model.config.hidden_size)
        text_dim = int(self.clip.text_model.config.hidden_size)
        self.text_token_to_vision = nn.Linear(text_dim, vision_dim)


        image_size = int(self.clip.vision_model.config.image_size)
        patch_size = int(self.clip.vision_model.config.patch_size)
        self.num_img_patches = (image_size // patch_size) ** 2

        self.text_aware_visual = TextAwareVisualExtraction(
            num_img_patches=self.num_img_patches,
            vision_dim=vision_dim,
        )

        self.visual_proj = nn.Linear(vision_dim, cfg.hidden_dim)
        self.text_proj = nn.Linear(text_dim, cfg.hidden_dim)
        self.proprio_encoder = ProprioceptionEncoder(
            input_dim=cfg.state_dim,
            hidden_dim=cfg.proprio_hidden_dim,
            output_dim=cfg.hidden_dim,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.modality_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.transformer_layers,
        )

        self.readout = nn.Sequential(
            nn.Linear(cfg.hidden_dim * 4, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(256, cfg.action_dim),
        )

    @staticmethod
    def _valid_text_mask(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # CLIP special tokens: BOS and EOS. We remove BOS and paddings, and usually EOS too.
        valid = attention_mask.bool().clone()
        if valid.size(1) > 0:
            valid[:, 0] = False  # BOS
        eos_pos = attention_mask.sum(dim=1) - 1
        for b in range(valid.size(0)):
            idx = int(eos_pos[b].item())
            if 0 <= idx < valid.size(1):
                valid[b, idx] = False
        return valid

    def encode_modalities(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        state: torch.Tensor,
    ):
        clip_dtype = next(self.clip.parameters()).dtype
        pixel_values = pixel_values.to(dtype=clip_dtype)

        with torch.no_grad() if self.cfg.freeze_clip else torch.enable_grad():
            vision_out = self.clip.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True,
            )
            text_out = self.clip.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True,
            )

        image_tokens = vision_out.last_hidden_state[:, 1:, :].float()     # drop CLS, (B,P,Dv)
        text_hidden = text_out.last_hidden_state.float()                   # (B,L,Dt)
        text_pooled = text_out.pooler_output.float()                       # (B,Dt)

        valid_mask = self._valid_text_mask(input_ids, attention_mask)
        trimmed_text_tokens = []
        max_keep = max(1, self.cfg.max_text_tokens)

        for b in range(text_hidden.size(0)):
            toks = text_hidden[b][valid_mask[b]]
            if toks.size(0) == 0:
                toks = text_hidden[b:b+1, 1:2, :].squeeze(0)
            toks = toks[:max_keep]
            if toks.size(0) < max_keep:
                pad = torch.zeros(max_keep - toks.size(0), toks.size(1), device=toks.device, dtype=toks.dtype)
                toks = torch.cat([toks, pad], dim=0)
            trimmed_text_tokens.append(toks)

        text_tokens = torch.stack(trimmed_text_tokens, dim=0)           # (B,T,Dt)
        text_tokens_for_visual = self.text_token_to_vision(text_tokens) # (B,T,Dv)
        text_aware_visual_tokens = self.text_aware_visual(
            image_tokens, text_tokens_for_visual
        )
        text_aware_visual = text_aware_visual_tokens.mean(dim=1)           # (B,Dv)
        global_visual = image_tokens.mean(dim=1)                           # (B,Dv)
        pooled_text = text_pooled                                          # (B,Dt)
        proprio = self.proprio_encoder(state.float())                      # (B,H)

        visual_token = self.visual_proj(text_aware_visual)                 # (B,H)
        global_visual_token = self.visual_proj(global_visual)              # (B,H)
        text_token = self.text_proj(pooled_text)                           # (B,H)

        tokens = torch.stack(
            [visual_token, global_visual_token, text_token, proprio],
            dim=1,
        )                                                                  # (B,4,H)

        encoded = self.modality_transformer(tokens)                        # (B,4,H)
        flat = encoded.reshape(encoded.size(0), -1)                        # (B,4H)
        return flat

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        fused = self.encode_modalities(pixel_values, input_ids, attention_mask, state)
        return self.readout(fused)

    def loss(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        pred = self.forward(pixel_values, input_ids, attention_mask, state)
        return F.mse_loss(pred, actions)

    @torch.no_grad()
    def act(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(pixel_values, input_ids, attention_mask, state)
