"""High-resolution action examiner with explicit ordered motion evidence.

Unlike the legacy verifier, this model does not merely ask a second head to
reclassify the same DINO vector.  It consumes high-rate, camera-compensated
residual motion and gives every visual/audio token an explicit relative time.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .goal_annotations import LABELS


class PerFramePatchResampler(nn.Module):
    def __init__(self, patch_dim: int, hidden_dim: int, latents: int = 8, heads: int = 8) -> None:
        super().__init__()
        self.input = nn.Linear(patch_dim, hidden_dim)
        self.latents = nn.Parameter(torch.randn(latents, hidden_dim) * 0.02)
        self.attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, patches: Tensor) -> Tensor:
        batch, frames, _patches, _channels = patches.shape
        values = self.input(patches).flatten(0, 1)
        query = self.latents.unsqueeze(0).expand(values.shape[0], -1, -1)
        output, _ = self.attention(query, values, values, need_weights=False)
        return self.norm(output).reshape(batch, frames, output.shape[1], output.shape[2])


class OrderedResidualMotion(nn.Module):
    """Encode 8x8 residual grids and camera translation as an ordered action."""

    def __init__(self, hidden_dim: int, frames: int = 17) -> None:
        super().__init__()
        self.frames = frames
        self.grid = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Flatten(), nn.Linear(64 * 4 * 4, hidden_dim),
        )
        self.camera = nn.Linear(3, hidden_dim)
        self.time = nn.Parameter(torch.randn(frames, hidden_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(hidden_dim, 8, 4 * hidden_dim, batch_first=True, norm_first=True, activation="gelu")
        self.temporal = nn.TransformerEncoder(layer, 3, norm=nn.LayerNorm(hidden_dim))

    def forward(self, residual_grid: Tensor, camera_motion: Tensor) -> Tensor:
        if residual_grid.shape[1:] != (self.frames, 8, 8):
            raise ValueError(f"residual_grid must be [B,{self.frames},8,8]")
        if camera_motion.shape[1:] != (self.frames, 3):
            raise ValueError(f"camera_motion must be [B,{self.frames},3]")
        batch = residual_grid.shape[0]
        grid = self.grid(residual_grid.reshape(batch * self.frames, 1, 8, 8)).reshape(batch, self.frames, -1)
        return self.temporal(grid + self.camera(camera_motion) + self.time.unsqueeze(0))


class GoalMotionExaminer(nn.Module):
    """25 full-image frames + 17-frame ordered residual motion + whistle audio."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        patch_dim: int = 1024,
        candidate_dim: int = 384,
        retriever_memory_dim: int = 384,
        audio_dim: int = 64,
        hidden_dim: int = 512,
        minimum_height: int = 512,
        minimum_width: int = 896,
        frame_chunk_size: int = 2,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.minimum_height = int(minimum_height); self.minimum_width = int(minimum_width)
        self.frame_chunk_size = int(frame_chunk_size)
        self.visual_resampler = PerFramePatchResampler(patch_dim, hidden_dim, latents=8)
        self.visual_time = nn.Parameter(torch.randn(25, hidden_dim) * 0.02)
        self.motion = OrderedResidualMotion(hidden_dim, frames=17)
        self.audio_projection = nn.Linear(audio_dim, hidden_dim)
        self.audio_time_mlp = nn.Sequential(nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
        self.retriever_projection = nn.Linear(retriever_memory_dim, hidden_dim)
        self.candidate_projection = nn.Linear(candidate_dim, hidden_dim)
        self.class_queries = nn.Parameter(torch.randn(len(LABELS), hidden_dim) * 0.02)
        self.query_type = nn.Parameter(torch.randn(2, hidden_dim) * 0.02)
        layer = nn.TransformerDecoderLayer(hidden_dim, 8, 4 * hidden_dim, batch_first=True, norm_first=True, activation="gelu")
        self.fusion = nn.TransformerDecoder(layer, 4, norm=nn.LayerNorm(hidden_dim))
        self.class_evidence = nn.Linear(hidden_dim, 1)
        self.background_head = nn.Linear(hidden_dim, 1)
        self.quality_head = nn.Linear(hidden_dim, 1)
        self.phase_head = nn.Linear(hidden_dim, 4)  # pre/contact/post/no-action
        self.time_residual = nn.Linear(hidden_dim, 1)
        self.uncertainty = nn.Linear(hidden_dim, 1)
        self.whistle_head = nn.Linear(hidden_dim, 1)
        self.confounder_head = nn.Linear(hidden_dim, 7)  # pass/cross/clearance/tackle/interception/cut/other

    def _features_only(self, frames: Tensor) -> Tensor:
        output = self.backbone.forward_features(frames)
        if isinstance(output, list):
            output = output[0]
        return output["x_norm_patchtokens"]

    def _patch_tokens(self, frames: Tensor) -> Tensor:
        batch, count, channels, height, width = frames.shape
        if height < self.minimum_height or width < self.minimum_width:
            raise ValueError(f"examiner requires >= {self.minimum_height}x{self.minimum_width}")
        flat = frames.reshape(batch * count, channels, height, width)
        trainable = torch.is_grad_enabled() and any(parameter.requires_grad for parameter in self.backbone.parameters())
        parts = []
        for chunk in flat.split(self.frame_chunk_size):
            parts.append(checkpoint(self._features_only, chunk, use_reentrant=False) if trainable else self._features_only(chunk))
        patches = torch.cat(parts, dim=0)
        return patches.reshape(batch, count, patches.shape[1], patches.shape[2])

    def forward(
        self,
        global_frames: Tensor,
        residual_grid: Tensor,
        camera_motion: Tensor,
        audio_tokens: Tensor,
        audio_relative_seconds: Tensor,
        candidate_embedding: Tensor,
        retriever_memory: Tensor,
    ) -> dict[str, Tensor]:
        if global_frames.ndim != 5 or global_frames.shape[1] != 25:
            raise ValueError("global_frames must be [B,25,3,H,W]")
        visual = self.visual_resampler(self._patch_tokens(global_frames))
        visual = visual + self.visual_time[None, :, None, :]
        visual = visual.flatten(1, 2)
        motion = self.motion(residual_grid, camera_motion)
        audio = self.audio_projection(audio_tokens) + self.audio_time_mlp(audio_relative_seconds.unsqueeze(-1))
        retriever = self.retriever_projection(retriever_memory)
        memory = torch.cat((retriever, visual, motion, audio), dim=1)
        candidate = self.candidate_projection(candidate_embedding)
        class_queries = candidate[:, None, :] + self.class_queries[None, :, :] + self.query_type[0]
        quality_query = candidate[:, None, :] + self.query_type[1]
        decoded = self.fusion(torch.cat((class_queries, quality_query), dim=1), memory)
        class_hidden, quality_hidden = decoded[:, :len(LABELS)], decoded[:, -1]
        foreground = self.class_evidence(class_hidden).squeeze(-1)
        background = self.background_head(quality_hidden)
        class_logits = torch.cat((foreground, background), dim=-1)
        phase_logits = self.phase_head(motion)
        contact_probability = phase_logits.softmax(dim=-1)[..., 1]
        dense_times = torch.linspace(-1.0, 1.0, motion.shape[1], device=motion.device, dtype=motion.dtype)
        contact_time = (contact_probability * dense_times).sum(dim=1) / contact_probability.sum(dim=1).clamp_min(1e-4)
        time_delta = (contact_time + self.time_residual(quality_hidden).squeeze(-1)).clamp(-2.0, 2.0)
        return {
            "class_logits": class_logits,
            "quality_logit": self.quality_head(quality_hidden).squeeze(-1),
            "time_delta_sec": time_delta,
            "temporal_uncertainty": F.softplus(self.uncertainty(quality_hidden).squeeze(-1)) + 0.05,
            "phase_logits": phase_logits,
            "whistle_logits": self.whistle_head(audio).squeeze(-1),
            "confounder_logits": self.confounder_head(quality_hidden),
            "class_evidence": class_hidden,
        }


def configure_examiner_backbone(backbone: nn.Module, *, warmup: bool, lora_rank: int = 16) -> int:
    """Effect-first DINO tuning: all norms, LoRA early blocks, full last four."""
    from dinov3.train.lora import inject_lora, reset_lora_parameters

    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    if warmup:
        return 0
    injected = inject_lora(backbone, {"rank": lora_rank, "target_last_blocks": 20, "train_norm": True})
    reset_lora_parameters(backbone)
    for block in list(backbone.blocks)[-4:]:
        for parameter in block.parameters():
            parameter.requires_grad_(True)
    return injected

