"""Strict full-image DINOv3 verifier with frame-chunk activation checkpointing.

All 25 full-resolution frames remain evidence.  Chunking only changes how the
independent per-frame ViT passes are scheduled, preventing activation storage
from exceeding a 32 GiB worker when the final DINO blocks are fine-tuned.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .set_spotting import NO_EVENT_INDEX
from .verifier_fullimage import PatchResampler


class CheckpointedFullImageDinoVerifier(nn.Module):
    """25 global frames, no ROI/crop modules, with exact per-frame encoding."""

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
        frame_chunk_size: int = 4,
    ) -> None:
        super().__init__()
        if frame_chunk_size < 1:
            raise ValueError("frame_chunk_size must be positive")
        self.backbone = backbone
        self.minimum_height, self.minimum_width = int(minimum_height), int(minimum_width)
        self.frame_chunk_size = int(frame_chunk_size)
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

    def _features_only(self, frames: Tensor) -> Tensor:
        result = self.backbone.forward_features(frames)
        if isinstance(result, list):
            result = result[0]
        return result["x_norm_patchtokens"]

    def _patch_tokens(self, frames: Tensor) -> Tensor:
        batch, count, channels, height, width = frames.shape
        if height < self.minimum_height or width < self.minimum_width:
            raise ValueError(f"full-image verifier requires >= {self.minimum_height}x{self.minimum_width}, received {height}x{width}")
        flat = frames.reshape(batch * count, channels, height, width)
        trainable_backbone = torch.is_grad_enabled() and any(parameter.requires_grad for parameter in self.backbone.parameters())
        chunks = []
        for value in flat.split(self.frame_chunk_size, dim=0):
            if trainable_backbone:
                chunks.append(checkpoint(self._features_only, value, use_reentrant=False))
            else:
                chunks.append(self._features_only(value))
        patches = torch.cat(chunks, dim=0)
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
