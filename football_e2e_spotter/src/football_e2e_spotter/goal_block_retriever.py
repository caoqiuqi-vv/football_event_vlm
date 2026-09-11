"""KPI-aligned 20-second block retriever with 180-second context."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .goal_annotations import LABELS
from .goal_retriever import MultiScaleMemory


@dataclass(frozen=True)
class BlockGeometry:
    context_seconds: int = 180
    core_seconds: int = 60
    block_seconds: int = 20
    feature_fps: int = 1

    @property
    def context_steps(self) -> int:
        return self.context_seconds * self.feature_fps

    @property
    def blocks(self) -> int:
        return self.core_seconds // self.block_seconds

    @property
    def context_left(self) -> int:
        return (self.context_seconds - self.core_seconds) // 2


class GoalBlockRetriever(nn.Module):
    """Rank disjoint review blocks; classes are independent sigmoids."""

    def __init__(
        self,
        appearance_dim: int,
        motion_dim: int = 64,
        audio_dim: int = 64,
        hidden_dim: int = 384,
        dropout: float = 0.15,
        geometry: BlockGeometry = BlockGeometry(),
    ) -> None:
        super().__init__()
        self.geometry = geometry
        self.appearance = nn.Sequential(nn.LayerNorm(appearance_dim), nn.Linear(appearance_dim, hidden_dim))
        self.motion = nn.Sequential(nn.LayerNorm(motion_dim), nn.Linear(motion_dim, hidden_dim))
        self.audio = nn.Sequential(nn.LayerNorm(audio_dim), nn.Linear(audio_dim, hidden_dim))
        self.gate = nn.Parameter(torch.zeros(3, hidden_dim))
        self.position = nn.Parameter(torch.randn(geometry.context_steps, hidden_dim) * 0.02)
        self.memory = MultiScaleMemory(hidden_dim, dropout)
        self.class_queries = nn.Parameter(torch.randn(len(LABELS), hidden_dim) * 0.02)
        self.any_query = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        self.block_position = nn.Parameter(torch.randn(geometry.blocks, hidden_dim) * 0.02)
        self.block_attention = nn.MultiheadAttention(hidden_dim, 8, dropout=dropout, batch_first=True)
        self.class_head = nn.Linear(hidden_dim, 1)
        self.any_head = nn.Linear(hidden_dim, 1)
        self.dense_head = nn.Linear(hidden_dim, len(LABELS))
        self.whistle_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

    def _drop(self, streams: list[Tensor]) -> list[Tensor]:
        if not self.training:
            return streams
        batch = streams[0].shape[0]
        keep = torch.rand(batch, 3, 1, 1, device=streams[0].device) > 0.12
        keep[:, :1] |= ~keep.any(dim=1, keepdim=True)
        return [stream * keep[:, index] for index, stream in enumerate(streams)]

    def forward(self, appearance: Tensor, motion: Tensor, audio: Tensor, valid: Tensor | None = None) -> dict[str, Tensor]:
        if appearance.shape[1] != self.geometry.context_steps:
            raise ValueError(f"expected {self.geometry.context_steps} context steps")
        streams = self._drop([self.appearance(appearance), self.motion(motion), self.audio(audio)])
        weights = self.gate.softmax(dim=0)
        fused = sum(stream * weights[index] for index, stream in enumerate(streams)) + self.position.unsqueeze(0)
        memory = self.memory(fused)
        left = self.geometry.context_left
        core = memory[:, left:left + self.geometry.core_seconds]
        block_logits, any_logits = [], []
        for block_index in range(self.geometry.blocks):
            begin = block_index * self.geometry.block_seconds
            tokens = core[:, begin:begin + self.geometry.block_seconds]
            class_query = self.class_queries.unsqueeze(0).expand(appearance.shape[0], -1, -1)
            class_query = class_query + self.block_position[block_index]
            class_hidden, _ = self.block_attention(class_query, tokens, tokens, need_weights=False)
            block_logits.append(self.class_head(class_hidden).squeeze(-1))
            any_query = self.any_query.unsqueeze(0).expand(appearance.shape[0], -1, -1) + self.block_position[block_index]
            any_hidden, _ = self.block_attention(any_query, tokens, tokens, need_weights=False)
            any_logits.append(self.any_head(any_hidden).squeeze(-1).squeeze(-1))
        return {
            "block_logits": torch.stack(block_logits, dim=1),
            "any_logits": torch.stack(any_logits, dim=1),
            "dense_logits": self.dense_head(core),
            "whistle_logits": self.whistle_head(core).squeeze(-1),
            "memory": memory,
        }

