from __future__ import annotations

"""Detection-independent locator over sequential VideoMAE chunk features."""

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class DilatedResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=5,
            padding=2 * int(dilation),
            dilation=int(dilation),
            groups=hidden_dim,
        )
        self.pointwise = nn.Conv1d(hidden_dim, 2 * hidden_dim, kernel_size=1)
        self.output = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, inputs: Tensor) -> Tensor:
        residual = inputs
        hidden = self.norm(inputs).transpose(1, 2)
        hidden = self.depthwise(hidden)
        value, gate = self.pointwise(hidden).chunk(2, dim=1)
        hidden = (value * torch.sigmoid(gate)).transpose(1, 2)
        return residual + self.residual_scale * self.output(hidden)


class CausalShotContext(nn.Module):
    """Past-only shot evidence: local maximum plus exponential traces."""

    def __init__(
        self,
        history_steps: int = 8,
        trace_half_lives: Sequence[float] = (1.0, 2.0, 4.0, 8.0),
    ) -> None:
        super().__init__()
        self.history_steps = max(int(history_steps), 1)
        kernels = []
        positions = torch.arange(self.history_steps, dtype=torch.float32)
        # conv1d is correlation.  The right-most coefficient multiplies the
        # current point after left padding, and earlier coefficients multiply
        # progressively older points.
        age = torch.flip(positions, dims=(0,))
        for half_life in trace_half_lives:
            half_life = max(float(half_life), 1e-3)
            weights = torch.exp(-math.log(2.0) * age / half_life)
            kernels.append(weights / weights.sum().clamp_min(1e-9))
        self.register_buffer(
            "trace_kernels", torch.stack(kernels).unsqueeze(1), persistent=True
        )

    @property
    def output_dim(self) -> int:
        return 1 + int(self.trace_kernels.shape[0])

    def forward(self, shot_probability: Tensor) -> Tensor:
        if shot_probability.ndim != 3 or shot_probability.shape[-1] != 1:
            raise ValueError("shot_probability must be [batch,time,1]")
        values = shot_probability.transpose(1, 2)
        padded = F.pad(values, (self.history_steps - 1, 0))
        past_max = F.max_pool1d(
            padded, kernel_size=self.history_steps, stride=1
        ).transpose(1, 2)
        traces = F.conv1d(padded, self.trace_kernels).transpose(1, 2)
        return torch.cat((past_max, traces), dim=-1)


class SequentialChunkLocator(nn.Module):
    """Anchor-free four-class point locator with football event structure.

    Input steps are stitched VideoMAE tubelets (typically 0.5 seconds each)
    extracted from non-overlapping 4-second chunks.  Six dilated blocks with
    dilations 1..32 and kernel size 5 cover 253 steps, or roughly 126 seconds
    at 2 Hz, without quadratic long-sequence attention.
    """

    LABELS = ("shot", "save", "corner", "freekick")
    FAMILIES = ("shot_chain", "restart")

    def __init__(
        self,
        input_dim: int = 1536,
        hidden_dim: int = 384,
        dilations: Sequence[int] = (1, 2, 4, 8, 16, 32),
        dropout: float = 0.1,
        shot_history_steps: int = 8,
        max_offset_seconds: float = 2.0,
        detach_shot_context: bool = True,
        tubelets_per_chunk: int = 8,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        if not dilations or any(int(value) <= 0 for value in dilations):
            raise ValueError("dilations must contain positive integers")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_offset_seconds = float(max_offset_seconds)
        self.detach_shot_context = bool(detach_shot_context)
        self.tubelets_per_chunk = int(tubelets_per_chunk)
        if self.tubelets_per_chunk <= 0:
            raise ValueError("tubelets_per_chunk must be positive")
        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.chunk_phase_embedding = nn.Embedding(self.tubelets_per_chunk, hidden_dim)
        nn.init.normal_(self.chunk_phase_embedding.weight, std=0.02)
        self.temporal = nn.Sequential(
            *[
                DilatedResidualBlock(hidden_dim, int(dilation), dropout)
                for dilation in dilations
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.family_head = nn.Linear(hidden_dim, len(self.FAMILIES))
        self.shot_head = nn.Linear(hidden_dim, 1)
        self.shot_context = CausalShotContext(history_steps=shot_history_steps)
        self.save_head = nn.Sequential(
            nn.LayerNorm(hidden_dim + self.shot_context.output_dim),
            nn.Linear(hidden_dim + self.shot_context.output_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.restart_subtype_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )
        self.offset_head = nn.Linear(hidden_dim, len(self.LABELS))
        self.family_residual_scale = nn.Parameter(torch.full((len(self.LABELS),), 0.25))

    @property
    def receptive_field_steps(self) -> int:
        return 1 + 4 * sum(
            int(block.depthwise.dilation[0]) for block in self.temporal
        )

    def forward(
        self,
        features: Tensor,
        valid_mask: Tensor | None = None,
        chunk_phase: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if features.ndim != 3 or features.shape[-1] != self.input_dim:
            raise ValueError(
                f"features must be [batch,time,{self.input_dim}], got {tuple(features.shape)}"
            )
        hidden = self.input_projection(features)
        if chunk_phase is None:
            chunk_phase = torch.arange(
                features.shape[1], device=features.device, dtype=torch.long
            ).remainder(self.tubelets_per_chunk).unsqueeze(0).expand(features.shape[0], -1)
        if chunk_phase.shape != features.shape[:2]:
            raise ValueError("chunk_phase must match [batch,time]")
        hidden = hidden + self.chunk_phase_embedding(
            chunk_phase.long().remainder(self.tubelets_per_chunk)
        )
        if valid_mask is not None:
            if valid_mask.shape != features.shape[:2]:
                raise ValueError("valid_mask must match [batch,time]")
            hidden = hidden * valid_mask.to(hidden.dtype).unsqueeze(-1)
        hidden = self.output_norm(self.temporal(hidden))
        family_logits = self.family_head(hidden)
        shot_logits = self.shot_head(hidden)
        shot_probability = torch.sigmoid(shot_logits)
        if self.detach_shot_context:
            shot_probability = shot_probability.detach()
        shot_context = self.shot_context(shot_probability)
        save_logits = self.save_head(torch.cat((hidden, shot_context), dim=-1))
        restart_logits = self.restart_subtype_head(hidden)

        # Family evidence is a bounded residual, not a detection gate.  A bad
        # family score therefore cannot erase a class prediction.
        family_scale = torch.tanh(self.family_residual_scale)
        shot = shot_logits + family_scale[0] * family_logits[..., 0:1]
        save = save_logits + family_scale[1] * family_logits[..., 0:1]
        restart = restart_logits + (
            family_scale[2:].reshape(1, 1, 2) * family_logits[..., 1:2]
        )
        class_logits = torch.cat((shot, save, restart), dim=-1)
        offsets = self.max_offset_seconds * torch.tanh(self.offset_head(hidden))
        if valid_mask is not None:
            invalid = ~valid_mask.bool()
            class_logits = class_logits.masked_fill(invalid.unsqueeze(-1), -20.0)
            family_logits = family_logits.masked_fill(invalid.unsqueeze(-1), -20.0)
            offsets = offsets.masked_fill(invalid.unsqueeze(-1), 0.0)
        return {
            "class_logits": class_logits,
            "family_logits": family_logits,
            "offsets": offsets,
            "shot_context": shot_context,
            "hidden": hidden,
        }
