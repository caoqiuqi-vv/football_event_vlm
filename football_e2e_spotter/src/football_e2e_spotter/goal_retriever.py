"""Long-context multimodal, anchored set retriever without temporal NMS."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .goal_annotations import LABELS


@dataclass(frozen=True)
class RetrieverGeometry:
    context_seconds: int = 180
    core_seconds: int = 60
    feature_fps: float = 1.0
    cell_seconds: float = 2.0
    anchors_per_cell: int = 2

    @property
    def context_steps(self) -> int:
        return int(round(self.context_seconds * self.feature_fps))

    @property
    def core_steps(self) -> int:
        return int(round(self.core_seconds * self.feature_fps))

    @property
    def cells(self) -> int:
        return int(round(self.core_seconds / self.cell_seconds))

    @property
    def slots(self) -> int:
        return self.cells * self.anchors_per_cell

    @property
    def context_left_seconds(self) -> float:
        return 0.5 * (self.context_seconds - self.core_seconds)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, dim: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv1d(dim, dim, 5, padding=2 * dilation, dilation=dilation, groups=dim)
        self.pointwise = nn.Conv1d(dim, 2 * dim, 1)
        self.out = nn.Conv1d(dim, dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        y = self.norm(x).transpose(1, 2)
        y = self.depthwise(y)
        value, gate = self.pointwise(y).chunk(2, dim=1)
        y = self.out(value * gate.sigmoid()).transpose(1, 2)
        return residual + self.dropout(y)


class MultiScaleMemory(nn.Module):
    """Local dilated evidence plus a bidirectional long-range bottleneck."""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.local = nn.ModuleList(ResidualTemporalBlock(dim, dilation, dropout) for dilation in (1, 2, 4, 8))
        self.down = nn.Conv1d(dim, dim, 5, stride=3, padding=2)
        self.gru = nn.GRU(dim, dim // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.up_projection = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU())
        self.final = nn.ModuleList(ResidualTemporalBlock(dim, dilation, dropout) for dilation in (1, 4))

    def forward(self, x: Tensor) -> Tensor:
        for block in self.local:
            x = block(x)
        coarse = self.down(x.transpose(1, 2)).transpose(1, 2)
        coarse, _ = self.gru(coarse)
        coarse = F.interpolate(coarse.transpose(1, 2), size=x.shape[1], mode="linear", align_corners=False).transpose(1, 2)
        x = x + self.up_projection(coarse)
        for block in self.final:
            x = block(x)
        return x


class GoalLongContextRetriever(nn.Module):
    """Predict independent anchored slots from a 180 s multimodal memory.

    Anchors prevent the permutation instability of fully global queries while
    retaining two independent instances in every two-second cell.  Classes use
    one slot-level softmax, but slots are independent: shot/save can coexist.
    """

    def __init__(
        self,
        appearance_dim: int,
        motion_dim: int = 64,
        audio_dim: int = 64,
        hidden_dim: int = 384,
        decoder_layers: int = 4,
        decoder_heads: int = 8,
        dropout: float = 0.15,
        geometry: RetrieverGeometry = RetrieverGeometry(),
    ) -> None:
        super().__init__()
        if hidden_dim % decoder_heads:
            raise ValueError("hidden_dim must be divisible by decoder_heads")
        self.geometry = geometry
        self.hidden_dim = int(hidden_dim)
        self.appearance = nn.Sequential(nn.LayerNorm(appearance_dim), nn.Linear(appearance_dim, hidden_dim))
        self.motion = nn.Sequential(nn.LayerNorm(motion_dim), nn.Linear(motion_dim, hidden_dim))
        self.audio = nn.Sequential(nn.LayerNorm(audio_dim), nn.Linear(audio_dim, hidden_dim))
        self.modality_gate = nn.Parameter(torch.zeros(3, hidden_dim))
        self.fusion = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.position = nn.Parameter(torch.empty(geometry.context_steps, hidden_dim))
        self.memory = MultiScaleMemory(hidden_dim, dropout)
        self.anchor_type = nn.Parameter(torch.empty(geometry.anchors_per_cell, hidden_dim))
        self.anchor_position = nn.Parameter(torch.empty(geometry.cells, hidden_dim))
        decoder_layer = nn.TransformerDecoderLayer(
            hidden_dim, decoder_heads, 4 * hidden_dim, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, decoder_layers, norm=nn.LayerNorm(hidden_dim))
        self.class_head = nn.Linear(hidden_dim, len(LABELS) + 1)
        self.time_head = nn.Linear(hidden_dim, 1)
        self.uncertainty_head = nn.Linear(hidden_dim, 1)
        self.quality_head = nn.Linear(hidden_dim, 1)
        self.duration_head = nn.Linear(hidden_dim, 1)
        self.family_head = nn.Linear(hidden_dim, 2)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.trunc_normal_(self.anchor_type, std=0.02)
        nn.init.trunc_normal_(self.anchor_position, std=0.02)
        nn.init.zeros_(self.class_head.bias)
        with torch.no_grad():
            self.class_head.bias[-1] = 2.0

    def _drop_modalities(self, values: list[Tensor]) -> list[Tensor]:
        if not self.training:
            return values
        batch = values[0].shape[0]
        keep = torch.rand(batch, 3, 1, 1, device=values[0].device) > 0.12
        # Never discard all evidence for one sample.
        empty = ~keep.any(dim=1, keepdim=True)
        keep[:, :1] |= empty
        return [value * keep[:, index] for index, value in enumerate(values)]

    def anchor_centers(self, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        cell = torch.arange(self.geometry.cells, device=device, dtype=dtype)
        centers = (cell + 0.5) * self.geometry.cell_seconds
        return centers[:, None].expand(-1, self.geometry.anchors_per_cell).reshape(-1)

    def forward(self, appearance: Tensor, motion: Tensor, audio: Tensor, valid_mask: Tensor | None = None) -> dict[str, Tensor]:
        if appearance.ndim != 3 or motion.ndim != 3 or audio.ndim != 3:
            raise ValueError("appearance/motion/audio must be [B,T,C]")
        if appearance.shape[:2] != motion.shape[:2] or appearance.shape[:2] != audio.shape[:2]:
            raise ValueError("modal timelines disagree")
        if appearance.shape[1] != self.geometry.context_steps:
            raise ValueError(f"expected {self.geometry.context_steps} context steps")
        streams = self._drop_modalities([self.appearance(appearance), self.motion(motion), self.audio(audio)])
        gates = self.modality_gate.softmax(dim=0)
        fused = sum(stream * gates[index] for index, stream in enumerate(streams))
        memory = self.memory(self.fusion(fused) + self.position.unsqueeze(0))
        left = int(round(self.geometry.context_left_seconds * self.geometry.feature_fps))
        cell_steps = int(round(self.geometry.cell_seconds * self.geometry.feature_fps))
        local_indices = left + torch.arange(self.geometry.cells, device=memory.device) * cell_steps + cell_steps // 2
        local = memory[:, local_indices]
        queries = local[:, :, None, :] + self.anchor_position[None, :, None, :] + self.anchor_type[None, None, :, :]
        queries = queries.reshape(appearance.shape[0], self.geometry.slots, self.hidden_dim)
        padding_mask = None if valid_mask is None else ~valid_mask.bool()
        slots = self.decoder(queries, memory, memory_key_padding_mask=padding_mask)
        anchor_centers = self.anchor_centers(device=slots.device, dtype=slots.dtype)
        time_delta = torch.tanh(self.time_head(slots).squeeze(-1)) * self.geometry.cell_seconds
        return {
            "class_logits": self.class_head(slots),
            "event_time": anchor_centers.unsqueeze(0) + time_delta,
            "time_delta": time_delta,
            "temporal_uncertainty": F.softplus(self.uncertainty_head(slots).squeeze(-1)) + 0.05,
            "quality_logit": self.quality_head(slots).squeeze(-1),
            "region_duration": 5.0 + 15.0 * torch.sigmoid(self.duration_head(slots).squeeze(-1)),
            "family_logits": self.family_head(slots),
            "slot_embedding": slots,
            "memory": memory,
        }

