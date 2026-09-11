"""Object-conditioned motion reasoning for football event recognition.

The detector is used only as a training teacher.  At inference this module
predicts ball/goal/person evidence directly from frozen or fine-tuned DINO
patch tokens.  It preserves the strong full-image anchor with zero-initialized
bounded residuals at both frame and clip level.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn


OBJECT_NAMES = ("ball", "goal", "person")


def _zero_last_linear(module: nn.Module) -> None:
    linears = [item for item in module.modules() if isinstance(item, nn.Linear)]
    if not linears:
        raise ValueError("residual module has no Linear layer to initialize")
    nn.init.zeros_(linears[-1].weight)
    if linears[-1].bias is not None:
        nn.init.zeros_(linears[-1].bias)


def _safe_time_derivative(values: Tensor, frame_times: Tensor) -> Tensor:
    """Finite differences with a zero first sample and robust time deltas."""
    if values.ndim < 3 or frame_times.ndim != 2:
        raise ValueError("motion derivative expects values [B,T,...], times [B,T]")
    if values.shape[:2] != frame_times.shape:
        raise ValueError("motion derivative values/times shape mismatch")
    if values.shape[1] <= 1:
        return torch.zeros_like(values)
    delta = values[:, 1:] - values[:, :-1]
    dt = (frame_times[:, 1:] - frame_times[:, :-1]).clamp_min(1e-3)
    while dt.ndim < delta.ndim:
        dt = dt.unsqueeze(-1)
    derivative = delta / dt
    return torch.cat((torch.zeros_like(derivative[:, :1]), derivative), dim=1)


def _gather_sparse_tokens(
    patch_tokens: Tensor,
    scores: Tensor,
    *,
    ratio: float,
    temperature: float,
) -> tuple[Tensor, Tensor]:
    patches = int(scores.shape[2])
    topk = min(max(int(math.ceil(patches * float(ratio))), 1), patches)
    values, indices = scores.topk(topk, dim=2)
    weights_float = F.softmax(
        values.float() / max(float(temperature), 1e-3), dim=2
    )
    pool_weights = weights_float.to(patch_tokens.dtype)
    gather_index = indices.unsqueeze(-1).expand(
        -1, -1, -1, patch_tokens.shape[-1]
    )
    selected = patch_tokens.gather(2, gather_index)
    pooled = (selected * pool_weights.unsqueeze(-1)).sum(dim=2)
    dense = torch.zeros_like(scores).scatter(
        2, indices, weights_float.to(scores.dtype)
    )
    return pooled, dense


class ObjectTokenCrossAttentionFusion(nn.Module):
    """Condition frozen event semantics on detached ball/goal evidence.

    Event gradients update this fusion module and the shared anchor path, but
    never reshape detector heatmaps into a classification shortcut.
    """

    def __init__(
        self,
        *,
        object_dim: int,
        hidden_dim: int,
        num_labels: int,
        num_heads: int,
        dropout: float,
        max_delta: float,
        gate_init: float,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("object-token fusion hidden_dim must divide num_heads")
        self.num_labels = int(num_labels)
        self.max_delta = max(float(max_delta), 1e-4)
        self.object_projection = nn.Sequential(
            nn.LayerNorm(object_dim),
            nn.Linear(object_dim, hidden_dim),
        )
        self.visibility_projection = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
        )
        self.relation_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.class_queries = nn.Parameter(
            torch.empty(1, self.num_labels, hidden_dim)
        )
        nn.init.trunc_normal_(self.class_queries, std=0.02)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        fusion_dim = hidden_dim * 4 + 2
        self.delta_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.gate_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, max(hidden_dim // 2, 64)),
            nn.GELU(),
            nn.Linear(max(hidden_dim // 2, 64), 1),
        )
        _zero_last_linear(self.delta_head)
        probability = min(max(float(gate_init), 1e-4), 1.0 - 1e-4)
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(
            self.gate_head[-1].bias,
            math.log(probability / (1.0 - probability)),
        )

    def forward(
        self,
        global_event_token: Tensor,
        object_tokens: Tensor,
        relation_tokens: Tensor,
        visibility: Tensor,
    ) -> dict[str, Tensor]:
        if global_event_token.ndim == 2:
            global_tokens = global_event_token.unsqueeze(1).expand(
                -1, self.num_labels, -1
            )
        elif (
            global_event_token.ndim == 3
            and global_event_token.shape[1] == self.num_labels
        ):
            global_tokens = global_event_token
        else:
            raise ValueError(
                "global event token must be [B,H] or [B,C,H]"
            )
        if object_tokens.ndim != 4 or object_tokens.shape[2] != 2:
            raise ValueError("object tokens must be [B,T,2,D] for ball/goal")
        if visibility.shape != object_tokens.shape[:3]:
            raise ValueError("object visibility must match [B,T,2]")
        if relation_tokens.shape[:2] != object_tokens.shape[:2]:
            raise ValueError("relation tokens must match object time axis")

        # This boundary is deliberate: event labels learn how to consume
        # detector evidence, not how to redraw detector heatmaps.
        detached_objects = object_tokens.detach()
        detached_visibility = visibility.detach().clamp(0.0, 1.0)
        detached_relations = relation_tokens.detach()
        object_evidence = self.object_projection(detached_objects)
        object_evidence = object_evidence + self.visibility_projection(
            detached_visibility.unsqueeze(-1)
        )
        batch, frames, objects, hidden = object_evidence.shape
        object_evidence = object_evidence.reshape(
            batch, frames * objects, hidden
        )
        relation_evidence = self.relation_projection(detached_relations)
        evidence = torch.cat((object_evidence, relation_evidence), dim=1)
        queries = global_tokens + self.class_queries
        attended, attention = self.cross_attention(
            queries,
            evidence,
            evidence,
            need_weights=True,
            average_attn_weights=False,
        )
        attended = self.output_norm(attended)
        visibility_summary = detached_visibility.amax(dim=1)
        fusion = torch.cat(
            (
                global_tokens,
                attended,
                global_tokens * attended,
                (global_tokens - attended).abs(),
                visibility_summary.unsqueeze(1).expand(
                    -1, self.num_labels, -1
                ),
            ),
            dim=-1,
        )
        raw_delta = self.delta_head(fusion).squeeze(-1)
        gate = self.gate_head(fusion).squeeze(-1).sigmoid()
        correction = gate * self.max_delta * torch.tanh(raw_delta)
        return {
            "correction": correction,
            "raw_delta": raw_delta,
            "gate": gate,
            "attention": attention.mean(dim=1),
            "attended_tokens": attended,
        }


class ObjectMotionEvidenceAdapter(nn.Module):
    """High-frame-rate ball/goal/person relation adapter.

    Inputs are DINO patch tokens from a short dense temporal window.  The
    adapter predicts object heatmaps and visibility, retains explicit spatial
    coordinates, builds camera-relative motion features, and returns bounded
    class-specific frame and clip residuals.
    """

    def __init__(
        self,
        *,
        patch_dim: int,
        hidden_dim: int,
        num_labels: int,
        num_heads: int = 8,
        temporal_layers: int = 2,
        dropout: float = 0.1,
        topk_ratios: Sequence[float] = (0.01, 0.05, 0.08),
        attention_temperature: float = 0.5,
        residual_max_delta: float = 1.0,
        frame_residual_max_delta: float | None = None,
        clip_residual_max_delta: float | None = None,
        gate_init: float = 0.10,
        gate_max: float = 1.0,
        gate_floor: float = 0.25,
        relation_delta: float = 1.0,
        fusion_delta: float = 0.5,
        evidence_gate_floor: float = 0.0,
        class_evidence_floors: Sequence[float] | None = None,
        detach_detector_for_event: bool = True,
        ball_layer_weights: Sequence[float] = (0.15, 0.25, 0.25, 0.35),
        ball_topk: int = 4,
        ball_temperature: float = 0.25,
        heatmap_upsample_factor: int = 1,
        ball_fpn_dim: int = 64,
        object_cross_attention_enabled: bool = False,
        object_cross_attention_max_delta: float = 0.35,
        object_cross_attention_gate_init: float = 0.25,
    ) -> None:
        super().__init__()
        if len(topk_ratios) != len(OBJECT_NAMES):
            raise ValueError(
                f"topk_ratios must have {len(OBJECT_NAMES)} values"
            )
        if hidden_dim % num_heads:
            raise ValueError("object motion hidden_dim must be divisible by num_heads")
        self.patch_dim = int(patch_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_labels = int(num_labels)
        self.topk_ratios = tuple(
            min(max(float(value), 1e-4), 1.0) for value in topk_ratios
        )
        self.attention_temperature = max(float(attention_temperature), 1e-3)
        legacy_max_delta = max(float(residual_max_delta), 1e-3)
        self.frame_residual_max_delta = max(
            float(
                legacy_max_delta
                if frame_residual_max_delta is None
                else frame_residual_max_delta
            ),
            1e-4,
        )
        self.clip_residual_max_delta = max(
            float(
                legacy_max_delta
                if clip_residual_max_delta is None
                else clip_residual_max_delta
            ),
            1e-4,
        )
        # Keep the legacy attribute for checkpoint/runtime diagnostics.
        self.residual_max_delta = max(
            self.frame_residual_max_delta, self.clip_residual_max_delta
        )
        self.gate_max = min(max(float(gate_max), 1e-4), 1.0)
        self.gate_floor = min(max(float(gate_floor), 0.0), self.gate_max)
        self.relation_delta = max(float(relation_delta), 1e-4)
        self.fusion_delta = max(float(fusion_delta), 0.0)
        self.evidence_gate_floor = min(
            max(float(evidence_gate_floor), 0.0), 1.0
        )
        if class_evidence_floors is None:
            class_evidence_floors = (self.evidence_gate_floor,) * self.num_labels
        if len(class_evidence_floors) != self.num_labels:
            raise ValueError("class_evidence_floors must match num_labels")
        floors = torch.as_tensor(tuple(class_evidence_floors), dtype=torch.float32)
        if ((floors < 0.0) | (floors > 1.0)).any():
            raise ValueError("class_evidence_floors must be in [0, 1]")
        self.register_buffer("class_evidence_floors", floors, persistent=True)
        self.detach_detector_for_event = bool(detach_detector_for_event)
        self.ball_topk = max(int(ball_topk), 1)
        self.ball_temperature = max(float(ball_temperature), 1e-3)
        self.heatmap_upsample_factor = int(heatmap_upsample_factor)
        if self.heatmap_upsample_factor not in (1, 2):
            raise ValueError("heatmap_upsample_factor must be 1 or 2")
        initial_ball_weights = torch.as_tensor(tuple(ball_layer_weights), dtype=torch.float32)
        if initial_ball_weights.numel() != 4 or (initial_ball_weights <= 0).any():
            raise ValueError("ball_layer_weights must contain four positive values")
        initial_ball_weights = initial_ball_weights / initial_ball_weights.sum()
        if float(initial_ball_weights.max()) > 0.7:
            raise ValueError("initial ball layer weight may not exceed 0.7")
        # w_i = 0.1 + 0.6*softmax(z_i) is convex and guarantees max(w_i)<=0.7.
        initial_simplex = ((initial_ball_weights - 0.1) / 0.6).clamp_min(1e-6)
        self.ball_layer_mix_logits = nn.Parameter(initial_simplex.log())
        self.ball_layer_norms = nn.ModuleList([nn.LayerNorm(patch_dim) for _ in range(4)])
        self.ball_layer_heads = nn.ModuleList([nn.Linear(patch_dim, 1) for _ in range(4)])
        self.ball_fpn_laterals: nn.ModuleList | None = None
        self.ball_fpn_refine: nn.Module | None = None
        self.ball_fpn_head: nn.Module | None = None
        if self.heatmap_upsample_factor == 2:
            fpn_dim = max(int(ball_fpn_dim), 16)
            self.ball_fpn_laterals = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(patch_dim),
                        nn.Linear(patch_dim, fpn_dim),
                    )
                    for _ in range(4)
                ]
            )
            self.ball_fpn_refine = nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1),
                nn.GELU(),
                nn.ConvTranspose2d(
                    fpn_dim, fpn_dim, kernel_size=2, stride=2
                ),
                nn.GELU(),
            )
            self.ball_fpn_head = nn.Conv2d(fpn_dim, 1, 1)
            # A newly introduced stride-8 branch must preserve a trained
            # patch-16 detector at initialization. It predicts a residual over
            # the upsampled low-resolution logits instead of random logits.
            nn.init.zeros_(self.ball_fpn_head.weight)
            nn.init.zeros_(self.ball_fpn_head.bias)

        self.heatmap_head = nn.Sequential(
            nn.LayerNorm(patch_dim), nn.Linear(patch_dim, len(OBJECT_NAMES))
        )
        self.presence_heads = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, 1))
                for _ in OBJECT_NAMES
            ]
        )
        self.no_object_tokens = nn.Parameter(
            torch.empty(len(OBJECT_NAMES), patch_dim)
        )
        nn.init.trunc_normal_(self.no_object_tokens, std=0.02)

        # ball, goal, nearest-to-ball person, nearest-to-goal person
        visual_dim = patch_dim * 4
        geometry_dim = 45
        self.relation_projection = nn.Sequential(
            nn.LayerNorm(visual_dim + geometry_dim),
            nn.Linear(visual_dim + geometry_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.time_projection = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(
            encoder_layer, num_layers=max(int(temporal_layers), 1)
        )
        self.temporal_norm = nn.LayerNorm(hidden_dim)

        self.frame_residual_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_labels),
        )
        self.class_queries = nn.Parameter(torch.empty(num_labels, hidden_dim))
        nn.init.trunc_normal_(self.class_queries, std=0.02)
        self.clip_residual_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.clip_gate_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1)
        )
        _zero_last_linear(self.frame_residual_head)
        _zero_last_linear(self.clip_residual_head)
        gate_probability = min(max(float(gate_init), 1e-4), 1.0 - 1e-4)
        gate_bias = math.log(gate_probability / (1.0 - gate_probability))
        nn.init.zeros_(self.clip_gate_head[-1].weight)
        nn.init.constant_(self.clip_gate_head[-1].bias, gate_bias)
        self.object_cross_attention_enabled = bool(
            object_cross_attention_enabled
        )
        self.object_cross_attention: ObjectTokenCrossAttentionFusion | None = None
        if self.object_cross_attention_enabled:
            self.object_cross_attention = ObjectTokenCrossAttentionFusion(
                object_dim=patch_dim,
                hidden_dim=hidden_dim,
                num_labels=num_labels,
                num_heads=num_heads,
                dropout=dropout,
                max_delta=object_cross_attention_max_delta,
                gate_init=object_cross_attention_gate_init,
            )

    @staticmethod
    def _grid(
        grid_h: int, grid_w: int, *, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, grid_h, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, grid_w, device=device, dtype=dtype),
            indexing="ij",
        )
        return torch.stack((xx, yy), dim=-1).reshape(grid_h * grid_w, 2)

    @staticmethod
    def _weighted_position(weights: Tensor, grid: Tensor) -> tuple[Tensor, Tensor]:
        normalized = weights / weights.sum(dim=2, keepdim=True).clamp_min(1e-6)
        center = torch.einsum("btp,pd->btd", normalized, grid)
        delta = grid.view(1, 1, grid.shape[0], 2) - center.unsqueeze(2)
        variance = (normalized.unsqueeze(-1) * delta.square()).sum(dim=2)
        return center, variance.clamp_min(1e-6).sqrt()

    def forward(
        self,
        patch_tokens: Tensor,
        frame_times: Tensor,
        *,
        grid_h: int,
        grid_w: int,
        ball_patch_layers: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if patch_tokens.ndim != 4:
            raise ValueError("object motion adapter expects [B,T,P,D] patch tokens")
        if frame_times.ndim != 2 or frame_times.shape != patch_tokens.shape[:2]:
            raise ValueError("object motion frame_times must match [B,T]")
        if int(grid_h) * int(grid_w) != patch_tokens.shape[2]:
            raise ValueError(
                f"object motion grid {(grid_h, grid_w)} does not match "
                f"patches={patch_tokens.shape[2]}"
            )

        if ball_patch_layers is None:
            ball_patch_layers = patch_tokens.unsqueeze(2).expand(-1, -1, 4, -1, -1)
        if ball_patch_layers.shape[:2] != patch_tokens.shape[:2] or ball_patch_layers.shape[2] != 4:
            raise ValueError("ball_patch_layers must be [B,T,4,P,D]")
        ball_layer_weights = 0.1 + 0.6 * self.ball_layer_mix_logits.softmax(dim=0)
        ball_features = torch.einsum("l,btlpd->btpd", ball_layer_weights, ball_patch_layers)
        live_ball_layer_logits = torch.stack(
            [
                head(norm(ball_patch_layers[:, :, index])).squeeze(-1)
                for index, (norm, head) in enumerate(zip(self.ball_layer_norms, self.ball_layer_heads))
            ],
            dim=2,
        )
        readout_ball_layer_logits = torch.stack(
            [
                head(norm(ball_patch_layers[:, :, index].detach())).squeeze(-1)
                for index, (norm, head) in enumerate(zip(self.ball_layer_norms, self.ball_layer_heads))
            ],
            dim=2,
        )
        lowres_ball_logits = torch.einsum(
            "l,btlp->btp", ball_layer_weights, live_ball_layer_logits
        )
        ball_readout_logits = torch.einsum("l,btlp->btp", ball_layer_weights, readout_ball_layer_logits)
        goal_person_lowres_logits = self.heatmap_head(
            patch_tokens.detach()
        )[..., 1:]
        ball_detection_features = ball_features
        if self.heatmap_upsample_factor == 2:
            if (
                self.ball_fpn_laterals is None
                or self.ball_fpn_refine is None
                or self.ball_fpn_head is None
            ):
                raise RuntimeError("ball FPN modules are missing")
            batch, frames, _layers, patches, _dim = ball_patch_layers.shape
            lateral_maps = []
            for layer_index, lateral in enumerate(self.ball_fpn_laterals):
                projected = lateral(ball_patch_layers[:, :, layer_index])
                lateral_maps.append(
                    projected.reshape(
                        batch * frames, grid_h, grid_w, -1
                    ).permute(0, 3, 1, 2)
                )
            fused_map = sum(
                weight * value
                for weight, value in zip(ball_layer_weights, lateral_maps)
            )
            refined = self.ball_fpn_refine(fused_map)
            highres_h, highres_w = refined.shape[-2:]
            base_ball_logits = F.interpolate(
                lowres_ball_logits.reshape(batch * frames, 1, grid_h, grid_w),
                size=(highres_h, highres_w),
                mode="bilinear",
                align_corners=False,
            )
            ball_logits = (
                base_ball_logits + self.ball_fpn_head(refined)
            ).reshape(batch, frames, highres_h * highres_w)
            ball_detection_features = refined.permute(0, 2, 3, 1).reshape(
                batch, frames, highres_h * highres_w, refined.shape[1]
            )
            goal_person_logits = F.interpolate(
                goal_person_lowres_logits.permute(0, 1, 3, 2).reshape(
                    batch * frames, 2, grid_h, grid_w
                ),
                size=(highres_h, highres_w),
                mode="bilinear",
                align_corners=False,
            ).reshape(batch, frames, 2, highres_h * highres_w).permute(
                0, 1, 3, 2
            )
            heatmap_logits = torch.cat(
                (ball_logits.unsqueeze(-1), goal_person_logits), dim=-1
            )
        else:
            ball_logits = lowres_ball_logits
            heatmap_logits = torch.cat(
                (
                    ball_readout_logits.unsqueeze(-1),
                    goal_person_lowres_logits,
                ),
                dim=-1,
            )
        routing_logits = torch.cat(
            (lowres_ball_logits.unsqueeze(-1), goal_person_lowres_logits),
            dim=-1,
        )
        probabilities = routing_logits.sigmoid()

        ball_k = min(self.ball_topk, int(ball_logits.shape[2]))
        top_values, top_indices = ball_logits.topk(ball_k, dim=2)
        top_weights = (
            top_values.float() / self.ball_temperature
        ).softmax(dim=2).to(ball_logits.dtype)
        ball_sparse_detection = torch.zeros_like(ball_logits).scatter(
            2, top_indices, top_weights
        )
        ball_entropy = -(
            top_weights * top_weights.clamp_min(1e-8).log()
        ).sum(dim=2) / max(math.log(max(ball_k, 2)), 1.0)
        if self.heatmap_upsample_factor == 2:
            ball_sparse = F.avg_pool2d(
                ball_sparse_detection.reshape(
                    ball_sparse_detection.shape[0]
                    * ball_sparse_detection.shape[1],
                    1,
                    grid_h * 2,
                    grid_w * 2,
                ),
                kernel_size=2,
                stride=2,
            ).reshape(
                ball_sparse_detection.shape[0],
                ball_sparse_detection.shape[1],
                grid_h * grid_w,
            ) * 4.0
        else:
            ball_sparse = ball_sparse_detection
        ball_pooled = torch.einsum(
            "btp,btpd->btd", ball_sparse, ball_features
        )

        object_tokens: list[Tensor] = []
        raw_object_tokens: list[Tensor] = []
        sparse_maps: list[Tensor] = [ball_sparse]
        presence_logits: list[Tensor] = []
        for object_index in range(len(OBJECT_NAMES)):
            if object_index == 0:
                pooled, sparse = ball_pooled, ball_sparse
            else:
                scores = routing_logits[..., object_index]
                pooled, sparse = _gather_sparse_tokens(
                    patch_tokens.detach(),
                    scores,
                    ratio=self.topk_ratios[object_index],
                    temperature=self.attention_temperature,
                )
                sparse_maps.append(sparse)
            object_probability = probabilities[..., object_index]
            stats = torch.stack(
                (
                    object_probability.amax(dim=2),
                    object_probability.mean(dim=2),
                    (object_probability * sparse).sum(dim=2),
                ),
                dim=-1,
            ).detach()
            current_presence_logit = self.presence_heads[object_index](stats).squeeze(-1)
            presence_logits.append(current_presence_logit)
            gate = current_presence_logit.sigmoid().unsqueeze(-1)
            no_object = self.no_object_tokens[object_index].view(1, 1, -1)
            raw_object_tokens.append(pooled)
            object_tokens.append(gate * pooled + (1.0 - gate) * no_object)

        presence_logits_tensor = torch.stack(presence_logits, dim=-1)
        presence = presence_logits_tensor.sigmoid()
        grid = self._grid(
            int(grid_h), int(grid_w),
            device=patch_tokens.device,
            dtype=patch_tokens.dtype,
        )
        centers: list[Tensor] = []
        spreads: list[Tensor] = []
        for object_index in range(len(OBJECT_NAMES)):
            current_sparse = (
                ball_sparse_detection
                if object_index == 0
                else sparse_maps[object_index]
            )
            current_grid = (
                self._grid(
                    int(grid_h) * self.heatmap_upsample_factor,
                    int(grid_w) * self.heatmap_upsample_factor,
                    device=patch_tokens.device,
                    dtype=patch_tokens.dtype,
                )
                if object_index == 0
                else grid
            )
            center, spread = self._weighted_position(
                current_sparse, current_grid
            )
            centers.append(center)
            spreads.append(spread)
        centers_tensor = torch.stack(centers, dim=2)
        spreads_tensor = torch.stack(spreads, dim=2)

        ball_center = centers_tensor[:, :, 0]
        goal_center = centers_tensor[:, :, 1]
        person_probability = probabilities[..., 2]
        distance_to_ball = (
            grid.view(1, 1, -1, 2) - ball_center.unsqueeze(2)
        ).square().sum(dim=-1)
        distance_to_goal = (
            grid.view(1, 1, -1, 2) - goal_center.unsqueeze(2)
        ).square().sum(dim=-1)
        person_ball_weights = person_probability * torch.exp(-distance_to_ball / 0.20)
        person_goal_weights = person_probability * torch.exp(-distance_to_goal / 0.35)
        person_ball_center, person_ball_spread = self._weighted_position(
            person_ball_weights, grid
        )
        person_goal_center, person_goal_spread = self._weighted_position(
            person_goal_weights, grid
        )
        person_ball_token = torch.einsum(
            "btp,btpd->btd",
            person_ball_weights
            / person_ball_weights.sum(dim=2, keepdim=True).clamp_min(1e-6),
            patch_tokens,
        )
        person_goal_token = torch.einsum(
            "btp,btpd->btd",
            person_goal_weights
            / person_goal_weights.sum(dim=2, keepdim=True).clamp_min(1e-6),
            patch_tokens,
        )
        person_gate = presence[..., 2:3]
        no_person = self.no_object_tokens[2].view(1, 1, -1)
        person_ball_token = person_gate * person_ball_token + (1.0 - person_gate) * no_person
        person_goal_token = person_gate * person_goal_token + (1.0 - person_gate) * no_person

        velocity = _safe_time_derivative(centers_tensor, frame_times)
        acceleration = _safe_time_derivative(velocity, frame_times)
        ball_goal_delta = ball_center - goal_center
        ball_person_delta = ball_center - person_ball_center
        camera_relative_velocity = velocity[:, :, 0] - velocity[:, :, 1]
        ball_person_velocity = velocity[:, :, 0] - _safe_time_derivative(
            person_ball_center, frame_times
        )
        geometry = torch.cat(
            (
                centers_tensor.flatten(2),                 # 6
                spreads_tensor.flatten(2),                 # 6
                person_ball_center,                        # 2
                person_goal_center,                        # 2
                person_ball_spread,                        # 2
                person_goal_spread,                        # 2
                ball_goal_delta,                           # 2
                ball_person_delta,                         # 2
                velocity.flatten(2),                       # 6
                acceleration[:, :, 0],                     # 2
                camera_relative_velocity,                  # 2
                ball_person_velocity,                      # 2
                ball_goal_delta.norm(dim=-1, keepdim=True),# 1
                ball_person_delta.norm(dim=-1, keepdim=True),# 1
                presence,                                  # 3
                spreads_tensor.mean(dim=-1),               # 3
                ball_entropy.unsqueeze(-1),                  # 1
            ),
            dim=-1,
        )
        if geometry.shape[-1] != 45:
            raise RuntimeError(f"object motion geometry dim={geometry.shape[-1]} expected=45")
        visual = torch.cat(
            (
                object_tokens[0],
                object_tokens[1],
                person_ball_token,
                person_goal_token,
            ),
            dim=-1,
        )
        # Detection is a teacher-distilled task. Event BCE must not reshape its
        # heatmaps into an easier classification shortcut. The relation branch
        # consumes detector evidence but, by default, cannot send event
        # gradients back into the detector/presence heads.
        if self.detach_detector_for_event:
            visual = visual.detach()
            geometry = geometry.detach()
        temporal_input = self.relation_projection(
            torch.cat((visual, geometry), dim=-1)
        )
        relative_time = frame_times - frame_times.mean(dim=1, keepdim=True)
        duration = (
            frame_times.amax(dim=1, keepdim=True)
            - frame_times.amin(dim=1, keepdim=True)
        ).clamp_min(1e-3)
        normalized_time = relative_time / duration
        time_features = torch.stack(
            (normalized_time, normalized_time.square(), torch.sin(math.pi * normalized_time)),
            dim=-1,
        )
        temporal_tokens = self.temporal(
            temporal_input + self.time_projection(time_features.to(temporal_input.dtype))
        )
        temporal_tokens = self.temporal_norm(temporal_tokens)

        raw_frame_residual = self.frame_residual_head(temporal_tokens)
        ball_evidence = presence[..., 0].detach()
        evidence_floors = self.class_evidence_floors.to(
            device=ball_evidence.device, dtype=ball_evidence.dtype
        ).view(1, 1, -1)
        frame_evidence_gate = evidence_floors + (
            1.0 - evidence_floors
        ) * ball_evidence.unsqueeze(-1)
        scale = math.sqrt(max(self.hidden_dim, 1))
        class_attention = F.softmax(torch.einsum("ch,bth->bct", self.class_queries, temporal_tokens) / scale, dim=-1)
        class_tokens = torch.einsum("bct,bth->bch", class_attention, temporal_tokens)
        raw_clip_residual = self.clip_residual_head(class_tokens).squeeze(-1)
        learned_clip_gate = self.clip_gate_head(class_tokens).squeeze(-1).sigmoid()
        clip_gate = self.gate_floor + (self.gate_max - self.gate_floor) * learned_clip_gate
        learned_frame_gate = learned_clip_gate.unsqueeze(1).expand(-1, temporal_tokens.shape[1], -1)
        base_frame_gate = clip_gate.unsqueeze(1).expand_as(raw_frame_residual)
        clip_evidence_gate = torch.einsum(
            "bct,btc->bc", class_attention, frame_evidence_gate
        ).clamp(0.0, 1.0)
        frame_gate = base_frame_gate * frame_evidence_gate
        clip_gate = clip_gate * clip_evidence_gate
        frame_residual = frame_gate * min(self.relation_delta, self.frame_residual_max_delta) * torch.tanh(raw_frame_residual)
        clip_residual = clip_gate * min(self.relation_delta, self.clip_residual_max_delta) * torch.tanh(raw_clip_residual)
        return {
            "heatmap_logits": heatmap_logits,
            "ball_lora_logits": ball_logits,
            "ball_student_features": ball_detection_features,
            "ball_student_center": centers_tensor[:, :, 0],
            "ball_student_presence": presence[:, :, 0],
            "ball_student_entropy": ball_entropy,
            "ball_layer_weights": ball_layer_weights,
            "presence_logits": presence_logits_tensor,
            "presence": presence,
            "object_tokens": torch.stack(raw_object_tokens[:2], dim=2),
            "object_visibility": probabilities[..., :2].amax(dim=2),
            "centers": centers_tensor,
            "spreads": spreads_tensor,
            "velocity": velocity,
            "acceleration": acceleration,
            "sparse_attention": torch.stack(sparse_maps, dim=-1),
            "class_attention": class_attention,
            "frame_evidence_gate": frame_evidence_gate,
            "clip_evidence_gate": clip_evidence_gate,
            "learned_frame_gate": learned_frame_gate,
            "learned_clip_gate": learned_clip_gate,
            "frame_gate": frame_gate,
            "clip_gate": clip_gate,
            "raw_frame_residual": raw_frame_residual,
            "frame_residual": frame_residual,
            "raw_clip_residual": raw_clip_residual,
            "clip_residual": clip_residual,
            "temporal_tokens": temporal_tokens,
        }

    def fuse_global_event(
        self,
        global_event_token: Tensor,
        motion_outputs: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        if not self.object_cross_attention_enabled or self.object_cross_attention is None:
            raise RuntimeError("object-token cross-attention fusion is disabled")
        return self.object_cross_attention(
            global_event_token,
            motion_outputs["object_tokens"],
            motion_outputs["temporal_tokens"],
            motion_outputs["object_visibility"],
        )


def interpolate_motion_residual(
    residual: Tensor,
    source_times: Tensor,
    target_times: Tensor,
) -> Tensor:
    """Piecewise-linear residual interpolation with zero outside motion span."""
    if residual.ndim != 3 or source_times.ndim != 2 or target_times.ndim != 2:
        raise ValueError("motion interpolation expects [B,T,C], [B,T], [B,S]")
    if residual.shape[:2] != source_times.shape:
        raise ValueError("motion interpolation source shapes do not match")
    outputs: list[Tensor] = []
    for batch_index in range(residual.shape[0]):
        source = source_times[batch_index]
        target = target_times[batch_index]
        right = torch.searchsorted(source.contiguous(), target.contiguous()).clamp(
            1, source.shape[0] - 1
        )
        left = right - 1
        left_time = source[left]
        right_time = source[right]
        alpha = ((target - left_time) / (right_time - left_time).clamp_min(1e-6)).unsqueeze(-1)
        interpolated = residual[batch_index, left] * (1.0 - alpha) + residual[
            batch_index, right
        ] * alpha
        inside = (target >= source[0]) & (target <= source[-1])
        outputs.append(interpolated * inside.unsqueeze(-1).to(interpolated.dtype))
    return torch.stack(outputs, dim=0)
