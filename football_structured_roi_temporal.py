from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from football_sparsemax_roi_residual import ROIOnlyResidualAttention
from train_football_events import TemporalConditionedSpatialAttention


class StructuredROIOnlyResidualAttention(ROIOnlyResidualAttention):
    """Keep crop-aligned spatial tokens instead of pooling an ROI to one vector."""

    def __init__(
        self,
        *args,
        patch_grid: tuple[int, int],
        spatial_grid: tuple[int, int] = (3, 5),
        bin_bandwidth: float = 0.55,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.patch_grid = tuple(map(int, patch_grid))
        self.spatial_grid = tuple(map(int, spatial_grid))
        self.bin_bandwidth = float(bin_bandwidth)
        if min(*self.patch_grid, *self.spatial_grid) <= 0:
            raise ValueError("patch_grid and spatial_grid must be positive")
        if self.bin_bandwidth <= 0:
            raise ValueError("bin_bandwidth must be positive")
        patch_h, patch_w = self.patch_grid
        ys = torch.linspace(0.0, 1.0, patch_h)
        xs = torch.linspace(0.0, 1.0, patch_w)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        self.register_buffer(
            "patch_coordinates",
            torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1),
            persistent=False,
        )
        grid_h, grid_w = self.spatial_grid
        bin_ys = torch.linspace(-1.0, 1.0, grid_h)
        bin_xs = torch.linspace(-1.0, 1.0, grid_w)
        bin_yy, bin_xx = torch.meshgrid(bin_ys, bin_xs, indexing="ij")
        self.register_buffer(
            "bin_centers",
            torch.stack([bin_xx.reshape(-1), bin_yy.reshape(-1)], dim=-1),
            persistent=False,
        )

    def _forward_fp32(
        self, global_frame_tokens: Tensor, patch_tokens: Tensor
    ) -> dict[str, Tensor]:
        result = super()._forward_fp32(global_frame_tokens, patch_tokens)
        attention = result.get("attention_maps")
        if attention is None:
            raise RuntimeError(
                "structured ROI attention requires return_attention_maps=true"
            )
        if patch_tokens.shape[2] != self.patch_coordinates.shape[0]:
            raise ValueError(
                f"patch count={patch_tokens.shape[2]} does not match "
                f"configured grid={self.patch_grid}"
            )
        values = self.patch_value(self.patch_norm(patch_tokens))
        coordinates = self.patch_coordinates.to(
            device=attention.device, dtype=attention.dtype
        )
        mean = torch.einsum("btcqn,nd->btcqd", attention, coordinates)
        centered = (
            coordinates.reshape(1, 1, 1, 1, -1, 2)
            - mean.unsqueeze(-2)
        )
        variance = (
            attention.unsqueeze(-1) * centered.square()
        ).sum(dim=-2)
        min_scale = coordinates.new_tensor(
            [
                1.5 / max(self.patch_grid[1] - 1, 1),
                1.5 / max(self.patch_grid[0] - 1, 1),
            ]
        )
        scale = variance.clamp_min(min_scale.square()).sqrt()
        normalized_coordinates = centered / (2.0 * scale.unsqueeze(-2))
        bin_centers = self.bin_centers.to(
            device=attention.device, dtype=attention.dtype
        )
        squared_distance = (
            normalized_coordinates.unsqueeze(-3)
            - bin_centers.reshape(1, 1, 1, 1, -1, 1, 2)
        ).square().sum(dim=-1)
        kernels = torch.exp(
            -0.5 * squared_distance / (self.bin_bandwidth ** 2)
        )
        weights = attention.unsqueeze(-2) * kernels
        mass = weights.sum(dim=-1)
        structured_tokens = torch.einsum(
            "btcqsn,btnh->btcqsh", weights, values
        ) / mass.clamp_min(1e-6).unsqueeze(-1)
        relative_mass = mass / mass.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        token_scale = (
            relative_mass * float(self.bin_centers.shape[0])
        ).clamp_min(1e-8).sqrt().clamp(max=2.0)
        structured_tokens = structured_tokens * token_scale.unsqueeze(-1)
        result["structured_roi_tokens"] = structured_tokens
        result["structured_roi_mass"] = relative_mass
        result["structured_roi_geometry"] = torch.cat([mean, scale], dim=-1)
        return result


def upgrade_structured_roi_attention(
    module: TemporalConditionedSpatialAttention,
    *,
    patch_grid: tuple[int, int],
    spatial_grid: tuple[int, int] = (3, 5),
    sparsemax_temperature: float = 4.0,
    bin_bandwidth: float = 0.55,
) -> StructuredROIOnlyResidualAttention:
    upgraded = StructuredROIOnlyResidualAttention(
        patch_dim=module.patch_norm.normalized_shape[0],
        hidden_dim=module.context_norm.normalized_shape[0],
        attention_dim=module.attention_dim,
        num_labels=module.num_labels,
        queries_per_class=module.queries_per_class,
        context_layers=len(module.context_encoder.layers),
        temporal_layers=len(module.temporal_encoder.layers),
        num_heads=module.context_encoder.layers[0].self_attn.num_heads,
        dropout=float(module.region_adapter[3].p),
        max_frames=module.context_pos_embed.shape[1],
        gate_init=0.25,
        dynamic_query_scale_init=float(module.context_query_scale.detach()),
        return_attention_maps=True,
        sparsemax_temperature=sparsemax_temperature,
        patch_grid=patch_grid,
        spatial_grid=spatial_grid,
        bin_bandwidth=bin_bandwidth,
    )
    upgraded.load_state_dict(module.state_dict(), strict=False)
    return upgraded


class StructuredROITemporalFusion(nn.Module):
    """Top-k ROI spatial-token reasoning with bounded global corrections."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_labels: int,
        num_frames: int,
        queries_per_class: int,
        spatial_grid: tuple[int, int] = (3, 5),
        token_dim: int = 256,
        topk_frames: int = 4,
        exploration_frames: int = 2,
        temporal_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        gate_init: float = 0.10,
        max_clip_delta: float = 1.0,
        max_frame_delta: float = 0.75,
    ):
        super().__init__()
        if token_dim % num_heads != 0:
            raise ValueError("structured ROI token_dim must divide num_heads")
        self.hidden_dim = int(hidden_dim)
        self.num_labels = int(num_labels)
        self.num_frames = int(num_frames)
        self.queries_per_class = int(queries_per_class)
        self.spatial_grid = tuple(map(int, spatial_grid))
        self.num_bins = self.spatial_grid[0] * self.spatial_grid[1]
        self.token_dim = int(token_dim)
        self.topk_frames = min(max(int(topk_frames), 1), self.num_frames)
        self.exploration_frames = min(
            max(int(exploration_frames), 0),
            self.num_frames - self.topk_frames,
        )
        self.max_clip_delta = float(max_clip_delta)
        self.max_frame_delta = float(max_frame_delta)
        self.local_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, token_dim)
        )
        self.global_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, token_dim)
        )
        self.geometry_projection = nn.Sequential(
            nn.Linear(4, token_dim), nn.GELU(), nn.Linear(token_dim, token_dim)
        )
        self.class_tokens = nn.Parameter(
            torch.zeros(1, num_labels, 1, token_dim)
        )
        self.query_embedding = nn.Parameter(
            torch.zeros(1, 1, 1, queries_per_class, 1, token_dim)
        )
        self.bin_embedding = nn.Parameter(
            torch.zeros(1, 1, 1, 1, self.num_bins, token_dim)
        )
        self.frame_embedding = nn.Parameter(
            torch.zeros(1, num_frames, 1, 1, 1, token_dim)
        )
        self.selection_embedding = nn.Parameter(torch.zeros(2, token_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=num_heads,
            dim_feedforward=token_dim * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            layer, num_layers=max(int(temporal_layers), 1)
        )
        self.temporal_norm = nn.LayerNorm(token_dim)
        decision_dim = token_dim * 2 + 5
        decision_hidden = max(token_dim // 2, 64)
        self.clip_delta_head = nn.Sequential(
            nn.LayerNorm(decision_dim),
            nn.Linear(decision_dim, decision_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(decision_hidden, 1),
        )
        self.clip_gate_head = nn.Sequential(
            nn.LayerNorm(decision_dim),
            nn.Linear(decision_dim, decision_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(decision_hidden, 1),
        )
        self.frame_delta_head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, decision_hidden),
            nn.GELU(),
            nn.Linear(decision_hidden, 1),
        )
        self.frame_gate_head = nn.Sequential(
            nn.LayerNorm(token_dim + 2),
            nn.Linear(token_dim + 2, decision_hidden),
            nn.GELU(),
            nn.Linear(decision_hidden, 1),
        )
        for parameter in (
            self.class_tokens,
            self.query_embedding,
            self.bin_embedding,
            self.frame_embedding,
            self.selection_embedding,
        ):
            nn.init.trunc_normal_(parameter, std=0.02)
        gate_init = min(max(float(gate_init), 1e-4), 1.0 - 1e-4)
        gate_bias = torch.logit(torch.tensor(gate_init)).item()
        for head in (self.clip_delta_head, self.frame_delta_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        for head in (self.clip_gate_head, self.frame_gate_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.constant_(head[-1].bias, gate_bias)

    def _exploration_indices(self, topk: Tensor, frames: int) -> Tensor:
        bsz, classes, _ = topk.shape
        count = self.exploration_frames
        if count == 0:
            return topk.new_empty((bsz, classes, 0))
        selected_mask = torch.zeros(
            bsz, classes, frames, device=topk.device, dtype=torch.bool
        )
        selected_mask.scatter_(2, topk, True)
        if self.training:
            priorities = torch.rand(bsz, classes, frames, device=topk.device)
        else:
            anchors = torch.linspace(
                0, frames - 1, count + 2, device=topk.device
            )[1:-1]
            positions = torch.arange(
                frames, device=topk.device, dtype=anchors.dtype
            )
            priorities = -torch.stack(
                [(positions - anchor).abs() for anchor in anchors], dim=0
            ).amin(dim=0)
            priorities = priorities.reshape(1, 1, frames).expand(
                bsz, classes, -1
            )
        priorities = priorities.masked_fill(selected_mask, -1e9)
        return torch.topk(priorities, k=count, dim=-1).indices

    @staticmethod
    def _gather_frames(values: Tensor, indices: Tensor) -> Tensor:
        ordered = values.permute(0, 2, 1, *range(3, values.ndim))
        tail = ordered.shape[3:]
        gather_index = indices.reshape(
            *indices.shape, *([1] * len(tail))
        ).expand(*indices.shape, *tail)
        return ordered.gather(2, gather_index)

    def forward(
        self,
        global_outputs: dict[str, Tensor],
        spatial_outputs: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        tokens = spatial_outputs.get("structured_roi_tokens")
        geometry = spatial_outputs.get("structured_roi_geometry")
        if tokens is None or geometry is None:
            raise ValueError("structured ROI fusion requires tokens and geometry")
        bsz, frames, classes, queries, bins, hidden = tokens.shape
        expected = (
            self.num_labels,
            self.queries_per_class,
            self.num_bins,
            self.hidden_dim,
        )
        if (classes, queries, bins, hidden) != expected:
            raise ValueError(
                f"structured ROI token shape={tuple(tokens.shape)} expected "
                f"B,F,{expected}"
            )
        spatial_frame_logits = spatial_outputs["frame_event_logits"]
        ranking = spatial_frame_logits.detach().permute(0, 2, 1)
        topk = torch.topk(ranking, k=min(self.topk_frames, frames), dim=-1).indices
        exploration = self._exploration_indices(topk, frames)
        selected = torch.cat([topk, exploration], dim=-1)
        selected_tokens = self._gather_frames(tokens, selected)
        selected_geometry = self._gather_frames(geometry, selected)
        local = self.local_projection(selected_tokens)
        local = local + self.geometry_projection(
            selected_geometry
        ).unsqueeze(-2)
        frame_position = self.frame_embedding[:, :frames].expand(
            bsz, -1, classes, -1, -1, -1
        )
        selected_position = self._gather_frames(frame_position, selected)
        local = (
            local
            + selected_position
            + self.query_embedding
            + self.bin_embedding
        )
        kinds = torch.cat(
            [
                selected.new_zeros(topk.shape),
                selected.new_ones(exploration.shape),
            ],
            dim=-1,
        )
        local = local + self.selection_embedding[kinds].unsqueeze(-2).unsqueeze(-2)
        sequence_tokens = local.reshape(
            bsz, classes, selected.shape[-1] * queries * bins, self.token_dim
        )
        global_temporal = global_outputs["temporal"].detach()
        if global_temporal.ndim == 2:
            global_temporal = global_temporal.unsqueeze(1).expand(
                -1, classes, -1
            )
        global_context = self.global_projection(global_temporal)
        class_token = self.class_tokens.expand(bsz, -1, -1, -1)
        class_token = class_token + global_context.unsqueeze(2)
        sequence = torch.cat([class_token, sequence_tokens], dim=2).reshape(
            bsz * classes, 1 + sequence_tokens.shape[2], self.token_dim
        )
        class_features = self.temporal_norm(
            self.temporal_encoder(sequence)[:, 0]
        ).reshape(bsz, classes, self.token_dim)
        global_logits = global_outputs["logits"].detach()
        topk_prob = torch.sigmoid(ranking.gather(2, topk)).mean(dim=-1)
        entropy_quality = 1.0 - spatial_outputs[
            "attention_entropy"
        ].mean(dim=(1, 3))
        global_probability = torch.sigmoid(global_logits)
        global_uncertainty = 1.0 - (2.0 * global_probability - 1.0).abs()
        local_probability = torch.sigmoid(spatial_outputs["clip_logits"])
        scalar = torch.stack(
            [
                global_logits,
                global_probability,
                global_uncertainty,
                local_probability,
                0.5 * (topk_prob + entropy_quality),
            ],
            dim=-1,
        )
        decision = torch.cat([class_features, global_context, scalar], dim=-1)
        raw_delta = self.max_clip_delta * torch.tanh(
            self.clip_delta_head(decision).squeeze(-1)
        )
        clip_gate = torch.sigmoid(
            self.clip_gate_head(decision).squeeze(-1)
        )
        clip_correction = clip_gate * raw_delta
        logits = global_logits + clip_correction
        all_local = self.local_projection(tokens).mean(dim=(3, 4))
        global_frame_logits = global_outputs["frame_event_logits"].detach()
        frame_scalar = torch.stack(
            [
                torch.sigmoid(global_frame_logits),
                torch.sigmoid(spatial_frame_logits),
            ],
            dim=-1,
        )
        raw_frame_delta = self.max_frame_delta * torch.tanh(
            self.frame_delta_head(all_local).squeeze(-1)
        )
        frame_gate = torch.sigmoid(
            self.frame_gate_head(
                torch.cat([all_local, frame_scalar], dim=-1)
            ).squeeze(-1)
        )
        frame_correction = frame_gate * raw_frame_delta
        fused_frame_logits = global_frame_logits + frame_correction
        feature_delta = frame_correction.unsqueeze(-1).expand(
            -1, -1, -1, self.hidden_dim
        ) / math.sqrt(float(self.hidden_dim))
        return logits, fused_frame_logits, feature_delta, clip_correction

