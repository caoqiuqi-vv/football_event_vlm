"""Strictly full-image DINOv3 candidate verifier.

This first verifier ablation deliberately contains neither a crop predictor nor
a local-frame encoder.  It therefore cannot use ROI evidence accidentally.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .set_spotting import NO_EVENT_INDEX, SET_LABELS


class PatchResampler(nn.Module):
    def __init__(self, patch_dim: int, hidden_dim: int, latents_per_frame: int = 32) -> None:
        super().__init__()
        self.input = nn.Linear(patch_dim, hidden_dim)
        self.latents = nn.Parameter(torch.randn(latents_per_frame, hidden_dim) * 0.02)
        self.attention = nn.MultiheadAttention(hidden_dim, 8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, patches: Tensor) -> Tensor:
        batch, frames, _, _ = patches.shape
        values = self.input(patches).flatten(0, 1)
        query = self.latents.unsqueeze(0).expand(values.shape[0], -1, -1)
        output, _ = self.attention(query, values, values, need_weights=False)
        return self.norm(output).reshape(batch, frames * output.shape[1], -1)


class FullImageDinoVerifier(nn.Module):
    """25 global frames plus Stage-1 temporal/audio evidence, without ROI."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        patch_dim: int = 1024,
        candidate_dim: int = 512,
        audio_dim: int = 128,
        hidden_dim: int = 512,
        minimum_height: int = 512,
        minimum_width: int = 896,
        fusion_layers: int = 4,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.minimum_height, self.minimum_width = int(minimum_height), int(minimum_width)
        self.candidate_projection = nn.Linear(candidate_dim, hidden_dim)
        self.shared_projection = nn.LazyLinear(hidden_dim)
        self.audio_projection = nn.Linear(audio_dim, hidden_dim)
        self.global_resampler = PatchResampler(patch_dim, hidden_dim)
        layer = nn.TransformerDecoderLayer(hidden_dim, 8, 4 * hidden_dim, batch_first=True, norm_first=True)
        self.fusion, self.norm = nn.TransformerDecoder(layer, fusion_layers), nn.LayerNorm(hidden_dim)
        self.class_head = nn.Linear(hidden_dim, NO_EVENT_INDEX + 1)
        self.time_head = nn.Linear(hidden_dim, 1)
        self.log_sigma_head = nn.Linear(hidden_dim, 1)
        self.quality_head = nn.Linear(hidden_dim, 1)

    def _patch_tokens(self, frames: Tensor) -> Tensor:
        batch, count, channels, height, width = frames.shape
        if height < self.minimum_height or width < self.minimum_width:
            raise ValueError(f"full-image verifier requires >= {self.minimum_height}x{self.minimum_width}, received {height}x{width}")
        result = self.backbone.forward_features(frames.reshape(batch * count, channels, height, width))
        if isinstance(result, list):
            result = result[0]
        patches = result["x_norm_patchtokens"]
        return patches.reshape(batch, count, patches.shape[1], patches.shape[2])

    def forward(self, global_frames: Tensor, candidate_embedding: Tensor, shared_tokens: Tensor, audio_tokens: Tensor) -> dict[str, Tensor | None]:
        if global_frames.ndim != 5 or global_frames.shape[1] != 25:
            raise ValueError("global_frames must be [batch,25,3,height,width]")
        query = self.candidate_projection(candidate_embedding).unsqueeze(1)
        global_tokens = self.global_resampler(self._patch_tokens(global_frames))
        memory = torch.cat((self.shared_projection(shared_tokens), self.audio_projection(audio_tokens), global_tokens), dim=1)
        hidden = self.norm(self.fusion(query, memory).squeeze(1))
        return {
            "class_logits": self.class_head(hidden),
            "time_delta_sec": 2 * self.time_head(hidden).tanh().squeeze(-1),
            "log_sigma": self.log_sigma_head(hidden).squeeze(-1).clamp(-4, 2),
            "quality_logits": self.quality_head(hidden).squeeze(-1),
            "crop_boxes": None,
        }


class PerClassLogisticFusion(nn.Module):
    def __init__(self, classes: int = len(SET_LABELS)) -> None:
        super().__init__()
        self.weight, self.bias = nn.Parameter(torch.ones(classes, 3)), nn.Parameter(torch.zeros(classes))

    def forward(self, spotter_logits: Tensor, verifier_logits: Tensor, quality_logits: Tensor) -> Tensor:
        return (torch.stack((spotter_logits, verifier_logits, quality_logits), dim=-1) * self.weight).sum(-1) + self.bias
