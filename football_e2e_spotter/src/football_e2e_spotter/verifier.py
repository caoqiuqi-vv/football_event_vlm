"""High-resolution DINOv3 ViT-L/16 candidate verifier.

This module never reads detector/tracker output and never suppresses temporal
candidates. Its differentiable crops are extra evidence beside global frames.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

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


class CandidateCropper(nn.Module):
    """Two candidate-conditioned crop scales; no detector inputs are used."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 6))

    def forward(self, frames: Tensor, candidate: Tensor) -> tuple[Tensor, Tensor]:
        batch, count, channels, height, width = frames.shape
        raw = self.head(candidate).reshape(batch, 2, 3)
        centers, scales = raw[..., :2].tanh() * 0.35, 0.25 + 0.25 * raw[..., 2:3].sigmoid()
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, height, dtype=frames.dtype, device=frames.device), torch.linspace(-1, 1, width, dtype=frames.dtype, device=frames.device), indexing="ij")
        base = torch.stack((xx, yy), dim=-1).reshape(1, 1, height, width, 2)
        crops = []
        for crop_index in range(2):
            grid = centers[:, crop_index].reshape(batch, 1, 1, 1, 2) + base * scales[:, crop_index].reshape(batch, 1, 1, 1, 1)
            grid = grid.expand(batch, count, -1, -1, -1).reshape(batch * count, height, width, 2)
            crop = F.grid_sample(frames.reshape(batch * count, channels, height, width), grid, align_corners=False)
            crops.append(crop.reshape(batch, count, channels, height, width))
        return torch.stack(crops, dim=2), torch.cat((centers, scales), dim=-1)


class DinoVerifier(nn.Module):
    """25-frame global high-resolution DINO verifier with 17-frame local evidence."""

    def __init__(self, backbone: nn.Module, *, patch_dim: int = 1024, candidate_dim: int = 512, audio_dim: int = 128, hidden_dim: int = 512, minimum_height: int = 512, minimum_width: int = 896, fusion_layers: int = 4) -> None:
        super().__init__()
        self.backbone, self.minimum_height, self.minimum_width = backbone, int(minimum_height), int(minimum_width)
        self.candidate_projection, self.shared_projection = nn.Linear(candidate_dim, hidden_dim), nn.LazyLinear(hidden_dim)
        self.audio_projection = nn.Linear(audio_dim, hidden_dim)
        self.global_resampler, self.local_resampler = PatchResampler(patch_dim, hidden_dim), PatchResampler(patch_dim, hidden_dim)
        self.cropper = CandidateCropper(hidden_dim)
        layer = nn.TransformerDecoderLayer(hidden_dim, 8, 4 * hidden_dim, batch_first=True, norm_first=True)
        self.fusion, self.norm = nn.TransformerDecoder(layer, fusion_layers), nn.LayerNorm(hidden_dim)
        self.class_head, self.time_head = nn.Linear(hidden_dim, NO_EVENT_INDEX + 1), nn.Linear(hidden_dim, 1)
        self.log_sigma_head, self.quality_head = nn.Linear(hidden_dim, 1), nn.Linear(hidden_dim, 1)

    def _patch_tokens(self, frames: Tensor) -> Tensor:
        batch, count, channels, height, width = frames.shape
        if height < self.minimum_height or width < self.minimum_width:
            raise ValueError(f"DINO verifier requires >= {self.minimum_height}x{self.minimum_width}, received {height}x{width}")
        result = self.backbone.forward_features(frames.reshape(batch * count, channels, height, width))
        if isinstance(result, list):
            result = result[0]
        patches = result["x_norm_patchtokens"]
        return patches.reshape(batch, count, patches.shape[1], patches.shape[2])

    def forward(self, global_frames: Tensor, candidate_embedding: Tensor, shared_tokens: Tensor, audio_tokens: Tensor) -> dict[str, Tensor]:
        if global_frames.ndim != 5 or global_frames.shape[1] != 25:
            raise ValueError("global_frames must be [batch,25,3,height,width]")
        query = self.candidate_projection(candidate_embedding).unsqueeze(1)
        global_tokens = self.global_resampler(self._patch_tokens(global_frames))
        local_crops, crop_boxes = self.cropper(global_frames[:, :17], query.squeeze(1))
        batch, frames, scales, channels, height, width = local_crops.shape
        local_patches = self._patch_tokens(local_crops.reshape(batch, frames * scales, channels, height, width))
        local_tokens = self.local_resampler(local_patches)
        memory = torch.cat((self.shared_projection(shared_tokens), self.audio_projection(audio_tokens), global_tokens, local_tokens), dim=1)
        hidden = self.norm(self.fusion(query, memory).squeeze(1))
        return {"class_logits": self.class_head(hidden), "time_delta_sec": 2 * self.time_head(hidden).tanh().squeeze(-1), "log_sigma": self.log_sigma_head(hidden).squeeze(-1).clamp(-4, 2), "quality_logits": self.quality_head(hidden).squeeze(-1), "crop_boxes": crop_boxes}


def configure_dinov3_vitl16(backbone: nn.Module, *, warmup: bool, lora_rank: int = 16) -> int:
    """Apply LayerNorm + final-four-block tuning and rank-16 LoRA adaptation."""
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


class PerClassLogisticFusion(nn.Module):
    """Per-class calibrated combination of stage-1, verifier, and quality logits."""

    def __init__(self, classes: int = len(SET_LABELS)) -> None:
        super().__init__()
        self.weight, self.bias = nn.Parameter(torch.ones(classes, 3)), nn.Parameter(torch.zeros(classes))

    def forward(self, spotter_logits: Tensor, verifier_logits: Tensor, quality_logits: Tensor) -> Tensor:
        return (torch.stack((spotter_logits, verifier_logits, quality_logits), dim=-1) * self.weight).sum(-1) + self.bias
