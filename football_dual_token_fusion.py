from __future__ import annotations

from collections.abc import Sequence
import math

import torch
from torch import Tensor, nn


def evenly_spaced_slots(length: int, count: int) -> list[int]:
    if length <= 0 or count <= 0:
        return []
    count = min(int(count), int(length))
    if count == 1:
        return [length // 2]
    return (
        torch.linspace(0, length - 1, count)
        .round()
        .to(torch.long)
        .tolist()
    )


def staggered_local_indices(
    global_indices: Sequence[int],
    overlap_frames: int,
) -> list[int]:
    """Mix exact global anchors with temporal midpoints without extra frames."""
    global_indices = [int(index) for index in global_indices]
    frame_count = len(global_indices)
    if frame_count <= 1:
        return list(global_indices)

    overlap_count = min(max(int(overlap_frames), 0), frame_count)
    midpoint_count = frame_count - overlap_count
    overlap_slots = evenly_spaced_slots(frame_count, overlap_count)
    gap_slots = evenly_spaced_slots(frame_count - 1, midpoint_count)

    local_indices = [global_indices[slot] for slot in overlap_slots]
    local_indices.extend(
        int(round((global_indices[slot] + global_indices[slot + 1]) / 2.0))
        for slot in gap_slots
    )
    return sorted(local_indices)


def staggered_multi_roi_indices(
    global_indices: Sequence[int],
    segment_start: int,
    segment_end: int,
) -> tuple[list[int], list[int]]:
    """Sample two complementary ROI timelines around the global timeline."""
    global_indices = [int(index) for index in global_indices]
    frame_count = len(global_indices)
    if frame_count == 0:
        return [], []
    segment_start = int(segment_start)
    segment_end = max(int(segment_end), segment_start)
    boundaries = torch.linspace(
        float(segment_start), float(segment_end + 1), frame_count + 1
    ).tolist()
    global_set = set(global_indices)
    used_a: set[int] = set()
    used_b: set[int] = set()

    def choose(preferred: float, low: int, high: int, excluded: set[int]) -> int:
        candidates = list(range(low, high + 1))
        available = [index for index in candidates if index not in excluded]
        pool = available if available else candidates
        if not pool:
            return min(max(int(round(preferred)), segment_start), segment_end)
        return min(pool, key=lambda index: (abs(float(index) - preferred), index))

    roi_a: list[int] = []
    roi_b: list[int] = []
    for slot in range(frame_count):
        low = max(int(math.ceil(boundaries[slot])), segment_start)
        high = min(int(math.ceil(boundaries[slot + 1])) - 1, segment_end)
        if high < low:
            low = high = min(
                max(int(round(boundaries[slot])), segment_start), segment_end
            )
        span = max(high - low, 0)
        index_a = choose(low + 0.25 * span, low, high, global_set | used_a)
        index_b = choose(
            low + 0.75 * span,
            low,
            high,
            global_set | used_a | used_b | {index_a},
        )
        roi_a.append(index_a)
        roi_b.append(index_b)
        used_a.add(index_a)
        used_b.add(index_b)
    return roi_a, roi_b


def bounded_asymmetric_residual(
    raw: Tensor,
    *,
    positive_max: float | Tensor,
    negative_max: float | Tensor,
) -> Tensor:
    positive_bound = torch.as_tensor(
        positive_max, device=raw.device, dtype=raw.dtype
    ).clamp_min(1e-6)
    negative_bound = torch.as_tensor(
        negative_max, device=raw.device, dtype=raw.dtype
    ).clamp_min(1e-6)
    return torch.where(
        raw >= 0,
        positive_bound * torch.tanh(raw / positive_bound),
        negative_bound * torch.tanh(raw / negative_bound),
    )


class FullRoiCrossAttentionFusion(nn.Module):
    """Inject valid ROI detail into full-image tokens with zero-start residuals."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        num_layers: int = 1,
        residual_gate_init: float = 0.0,
    ):
        super().__init__()
        self.global_time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.local_time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.local_view_embed = nn.Embedding(2, hidden_dim)
        layer_count = max(int(num_layers), 1)
        self.cross_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    hidden_dim,
                    num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(layer_count)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(layer_count)])
        gate_init = min(max(float(residual_gate_init), -0.95), 0.95)
        self.residual_gates = nn.Parameter(
            torch.full((layer_count,), torch.atanh(torch.tensor(gate_init)).item())
        )

    @staticmethod
    def _normalize_times(times: Tensor) -> Tensor:
        start = times.amin(dim=1, keepdim=True)
        duration = (times.amax(dim=1, keepdim=True) - start).clamp_min(1e-6)
        return (times - start) / duration

    def forward(
        self,
        global_tokens: Tensor,
        local_tokens: Tensor,
        global_times: Tensor,
        local_times: Tensor,
        local_quality: Tensor,
        local_view_ids: Tensor | None = None,
    ) -> dict[str, Tensor]:
        batch_size, global_frames, _ = global_tokens.shape
        local_frames = local_tokens.shape[1]
        if global_times.shape != (batch_size, global_frames):
            raise ValueError(
                f"global_times shape={tuple(global_times.shape)} must be "
                f"{(batch_size, global_frames)}"
            )
        if local_times.shape != (batch_size, local_frames):
            raise ValueError(
                f"local_times shape={tuple(local_times.shape)} must be "
                f"{(batch_size, local_frames)}"
            )
        if local_quality.shape != (batch_size, local_frames):
            raise ValueError(
                f"local_quality shape={tuple(local_quality.shape)} must be "
                f"{(batch_size, local_frames)}"
            )

        if local_view_ids is None:
            local_view_ids = torch.zeros(
                batch_size,
                local_frames,
                dtype=torch.long,
                device=local_tokens.device,
            )
        elif local_view_ids.shape != (batch_size, local_frames):
            raise ValueError(
                f"local_view_ids shape={tuple(local_view_ids.shape)} must be "
                f"{(batch_size, local_frames)}"
            )

        start = global_times.amin(dim=1, keepdim=True)
        duration = (global_times.amax(dim=1, keepdim=True) - start).clamp_min(1e-6)
        normalized_global_times = (global_times - start) / duration
        normalized_local_times = (local_times - start) / duration
        global_time_features = self.global_time_embed(
            normalized_global_times.unsqueeze(-1).to(global_tokens.dtype)
        )
        local_time_features = self.local_time_embed(
            normalized_local_times.unsqueeze(-1).to(local_tokens.dtype)
        )
        query_tokens = global_tokens + global_time_features
        local_view_features = self.local_view_embed(
            local_view_ids.to(device=local_tokens.device, dtype=torch.long).clamp(0, 1)
        ).to(local_tokens.dtype)
        key_value_tokens = (
            local_tokens + local_time_features + local_view_features
        ) * local_quality.unsqueeze(-1)
        key_padding_mask = local_quality <= 0
        all_invalid = key_padding_mask.all(dim=1)
        if bool(all_invalid.any()):
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_invalid, 0] = False
            key_value_tokens = key_value_tokens.clone()
            key_value_tokens[all_invalid, 0] = 0.0

        attention: Tensor | None = None
        fused = global_tokens
        for index, (layer, norm) in enumerate(zip(self.cross_layers, self.norms)):
            attended, weights = layer(
                query_tokens,
                key_value_tokens,
                key_value_tokens,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )
            fused = fused + torch.tanh(self.residual_gates[index]) * norm(attended)
            query_tokens = fused + global_time_features
            attention = weights.mean(dim=1)
        valid = torch.ones(batch_size, global_frames, dtype=torch.bool, device=global_tokens.device)
        return {
            "tokens": fused,
            "attention": attention,
            "valid": valid,
            "residual_gates": torch.tanh(self.residual_gates),
        }


class DualViewTokenFusionTransformer(nn.Module):
    """Fuse timestamped global/ROI tokens and expose label-wise attention."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        num_labels: int,
    ):
        super().__init__()
        self.view_embed = nn.Parameter(torch.zeros(1, 2, hidden_dim))
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.class_queries = nn.Parameter(
            torch.zeros(1, num_labels, hidden_dim)
        )
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.query_norm = nn.LayerNorm(hidden_dim)
        nn.init.trunc_normal_(self.view_embed, std=0.02)
        nn.init.trunc_normal_(self.class_queries, std=0.02)

    @staticmethod
    def _normalized_times(global_times: Tensor, local_times: Tensor) -> Tensor:
        times = torch.cat([global_times, local_times], dim=1)
        start = times.amin(dim=1, keepdim=True)
        duration = (times.amax(dim=1, keepdim=True) - start).clamp_min(1e-6)
        return (times - start) / duration

    def forward(
        self,
        global_tokens: Tensor,
        local_tokens: Tensor,
        global_times: Tensor,
        local_times: Tensor,
        local_quality: Tensor,
    ) -> dict[str, Tensor]:
        batch_size, global_frames, hidden_dim = global_tokens.shape
        local_frames = local_tokens.shape[1]
        if global_times.shape != (batch_size, global_frames):
            raise ValueError(
                f"global_times shape={tuple(global_times.shape)} must be "
                f"{(batch_size, global_frames)}"
            )
        if local_times.shape != (batch_size, local_frames):
            raise ValueError(
                f"local_times shape={tuple(local_times.shape)} must be "
                f"{(batch_size, local_frames)}"
            )
        if local_quality.shape != (batch_size, local_frames):
            raise ValueError(
                f"local_quality shape={tuple(local_quality.shape)} must be "
                f"{(batch_size, local_frames)}"
            )

        tokens = torch.cat(
            [global_tokens, local_tokens * local_quality.unsqueeze(-1)],
            dim=1,
        )
        times = torch.cat([global_times, local_times], dim=1)
        token_valid = torch.cat(
            [
                torch.ones(
                    batch_size, global_frames, dtype=torch.bool, device=tokens.device
                ),
                local_quality > 0,
            ],
            dim=1,
        )
        normalized_times = self._normalized_times(global_times, local_times)
        view_ids = torch.cat(
            [
                torch.zeros(
                    batch_size,
                    global_frames,
                    dtype=torch.long,
                    device=tokens.device,
                ),
                torch.ones(
                    batch_size,
                    local_frames,
                    dtype=torch.long,
                    device=tokens.device,
                ),
            ],
            dim=1,
        )
        order = times.argsort(dim=1, stable=True)
        gather_tokens = order.unsqueeze(-1).expand(-1, -1, hidden_dim)
        tokens = tokens.gather(1, gather_tokens)
        normalized_times = normalized_times.gather(1, order)
        sorted_times = times.gather(1, order)
        sorted_views = view_ids.gather(1, order)
        sorted_valid = token_valid.gather(1, order)

        view_features = self.view_embed.expand(batch_size, -1, -1)
        view_features = view_features.gather(
            1,
            sorted_views.unsqueeze(-1).expand(-1, -1, hidden_dim),
        )
        tokens = (
            tokens
            + view_features
            + self.time_embed(normalized_times.unsqueeze(-1).to(tokens.dtype))
        )
        padding_mask = ~sorted_valid
        encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)
        queries = self.class_queries.expand(batch_size, -1, -1)
        attended, weights = self.cross_attn(
            queries,
            encoded,
            encoded,
            need_weights=True,
            key_padding_mask=padding_mask,
            average_attn_weights=False,
        )
        query_features = self.query_norm(attended)
        attention = weights.mean(dim=1).transpose(1, 2)
        return {
            "query_features": query_features,
            "temporal": query_features.mean(dim=1),
            "attention": attention,
            "encoded_tokens": encoded,
            "times": sorted_times,
            "view_ids": sorted_views,
            "order": order,
            "valid": sorted_valid,
        }
