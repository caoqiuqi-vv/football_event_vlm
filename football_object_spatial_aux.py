"""Training-only ball/goal spatial distillation for football event models.

The detector is a teacher, never an inference dependency.  A compact aligned
index supplies ball/goal boxes for sampled source frames.  DINO patch tokens
learn two heatmaps and a small zero-initialized residual event branch consumes
the heatmap-pooled object tokens.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


OBJECT_NAMES = ("ball", "goal")


class ObjectTeacherTargetProvider:
    """Create patch-aligned soft targets from compact detector indices.

    The accepted v2 payload is the existing robust index schema.  New formal
    dual-teacher indices may use version=3 and the same tensor fields while
    adding provenance metadata.  Missing index frames are ignored rather than
    treated as hard negatives.
    """

    def __init__(
        self,
        index_root: str | Path,
        *,
        image_size: Sequence[int],
        patch_size: int = 16,
        ball_class_id: int = 1,
        goal_class_id: int = 2,
        ball_confidence: float = 0.10,
        goal_confidence: float = 0.20,
        max_frame_gap_sec: float = 0.65,
        ball_sigma_patches: float = 1.25,
        goal_dilation_patches: float = 0.5,
        max_cached_videos: int = 2,
    ) -> None:
        self.index_root = Path(index_root).expanduser()
        self.image_height, self.image_width = (int(v) for v in image_size)
        self.patch_size = int(patch_size)
        if self.image_height % self.patch_size or self.image_width % self.patch_size:
            raise ValueError(
                "object teacher requires image dimensions divisible by patch_size: "
                f"image={image_size} patch={patch_size}"
            )
        self.grid_h = self.image_height // self.patch_size
        self.grid_w = self.image_width // self.patch_size
        self.patch_count = self.grid_h * self.grid_w
        self.class_ids = (int(ball_class_id), int(goal_class_id))
        self.confidence_floors = (float(ball_confidence), float(goal_confidence))
        self.max_frame_gap_sec = max(float(max_frame_gap_sec), 0.0)
        self.ball_sigma_patches = max(float(ball_sigma_patches), 0.25)
        self.goal_dilation_patches = max(float(goal_dilation_patches), 0.0)
        self.max_cached_videos = max(int(max_cached_videos), 1)
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        yy, xx = torch.meshgrid(
            torch.arange(self.grid_h, dtype=torch.float32) + 0.5,
            torch.arange(self.grid_w, dtype=torch.float32) + 0.5,
            indexing="ij",
        )
        self._grid_x = xx
        self._grid_y = yy

    def has_video(self, video_id: str) -> bool:
        return (self.index_root / f"{video_id}.pt").is_file()

    def _load(self, video_id: str) -> dict[str, Any] | None:
        cached = self._cache.pop(video_id, None)
        if cached is not None:
            self._cache[video_id] = cached
            return cached
        path = self.index_root / f"{video_id}.pt"
        if not path.is_file():
            return None
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or int(payload.get("version", 0)) not in (2, 3):
            raise ValueError(f"unsupported object teacher index: {path}")
        self._cache[video_id] = payload
        while len(self._cache) > self.max_cached_videos:
            self._cache.popitem(last=False)
        return payload

    def empty(self, frames: int) -> tuple[Tensor, Tensor]:
        shape = (int(frames), self.patch_count, len(OBJECT_NAMES))
        return torch.zeros(shape, dtype=torch.float32), torch.zeros(shape, dtype=torch.float32)

    def _ball_target(self, box: Tensor, confidence: float, width: float, height: float) -> Tensor:
        cx = float((box[0] + box[2]) * 0.5) / max(width, 1.0) * self.grid_w
        cy = float((box[1] + box[3]) * 0.5) / max(height, 1.0) * self.grid_h
        distance2 = (self._grid_x - cx).square() + (self._grid_y - cy).square()
        return float(confidence) * torch.exp(
            -0.5 * distance2 / (self.ball_sigma_patches**2)
        )

    def _goal_target(self, box: Tensor, confidence: float, width: float, height: float) -> Tensor:
        x1 = float(box[0]) / max(width, 1.0) * self.grid_w - self.goal_dilation_patches
        y1 = float(box[1]) / max(height, 1.0) * self.grid_h - self.goal_dilation_patches
        x2 = float(box[2]) / max(width, 1.0) * self.grid_w + self.goal_dilation_patches
        y2 = float(box[3]) / max(height, 1.0) * self.grid_h + self.goal_dilation_patches
        # A soft one-patch boundary is less brittle than a binary resized box.
        dx = torch.maximum(torch.maximum(x1 - self._grid_x, self._grid_x - x2), torch.zeros_like(self._grid_x))
        dy = torch.maximum(torch.maximum(y1 - self._grid_y, self._grid_y - y2), torch.zeros_like(self._grid_y))
        distance = torch.sqrt(dx.square() + dy.square())
        return float(confidence) * (1.0 - distance).clamp(0.0, 1.0)

    def targets(self, video_id: str, absolute_times: Sequence[float]) -> tuple[Tensor, Tensor]:
        payload = self._load(str(video_id))
        targets, masks = self.empty(len(absolute_times))
        if payload is None:
            return targets, masks
        fps = float(payload.get("fps", 0.0) or 0.0)
        width = float(payload.get("image_size", {}).get("width", 0) or 0)
        height = float(payload.get("image_size", {}).get("height", 0) or 0)
        frame_ids = torch.as_tensor(payload.get("frame_ids", []), dtype=torch.int64)
        offsets = torch.as_tensor(payload.get("frame_offsets", []), dtype=torch.int64)
        classes = torch.as_tensor(payload.get("classes", []), dtype=torch.int64)
        confidences = torch.as_tensor(payload.get("confidences", []), dtype=torch.float32)
        boxes = torch.as_tensor(payload.get("boxes", []), dtype=torch.float32).reshape(-1, 4)
        if fps <= 0 or width <= 0 or height <= 0 or not len(frame_ids) or len(offsets) != len(frame_ids) + 1:
            return targets, masks
        max_gap_frames = max(int(round(self.max_frame_gap_sec * fps)), 1)
        for time_index, absolute_time in enumerate(absolute_times):
            wanted = int(round(float(absolute_time) * fps))
            insertion = int(torch.searchsorted(frame_ids, torch.tensor(wanted)).item())
            candidates = [idx for idx in (insertion - 1, insertion) if 0 <= idx < len(frame_ids)]
            if not candidates:
                continue
            nearest = min(candidates, key=lambda idx: abs(int(frame_ids[idx]) - wanted))
            if abs(int(frame_ids[nearest]) - wanted) > max_gap_frames:
                continue
            # The detector ran close enough to this sampled frame.  It is valid
            # supervision, but negatives are deliberately weak in the loss.
            masks[time_index] = 1.0
            left, right = int(offsets[nearest]), int(offsets[nearest + 1])
            for object_index, (class_id, floor) in enumerate(
                zip(self.class_ids, self.confidence_floors)
            ):
                selected = torch.nonzero(
                    (classes[left:right] == class_id)
                    & (confidences[left:right] >= floor),
                    as_tuple=False,
                ).flatten()
                heatmap = torch.zeros((self.grid_h, self.grid_w), dtype=torch.float32)
                for relative_index in selected.tolist():
                    item_index = left + int(relative_index)
                    if object_index == 0:
                        contribution = self._ball_target(
                            boxes[item_index], float(confidences[item_index]), width, height
                        )
                    else:
                        contribution = self._goal_target(
                            boxes[item_index], float(confidences[item_index]), width, height
                        )
                    heatmap = torch.maximum(heatmap, contribution)
                targets[time_index, :, object_index] = heatmap.flatten().clamp(0.0, 1.0)
        return targets, masks


class ObjectSpatialAuxHead(nn.Module):
    """Heatmap-supervised object tokens plus an anchor-safe event residual."""

    def __init__(
        self,
        *,
        patch_dim: int,
        hidden_dim: int,
        num_labels: int,
        temporal_layers: int = 1,
        dropout: float = 0.1,
        topk_ratio: float = 0.03,
        temperature: float = 0.7,
        residual_max_delta: float = 1.0,
        relation_aux_enabled: bool = False,
        relation_targets: int = 10,
        relation_topk_candidates: int = 3,
        relation_candidate_nms_radius: int = 2,
        relation_grid_size: Sequence[int] = (),
        relation_context_radii: Sequence[float] = (1.5, 4.0, 8.0),
    ) -> None:
        super().__init__()
        self.topk_ratio = min(max(float(topk_ratio), 1e-4), 1.0)
        self.temperature = max(float(temperature), 1e-3)
        self.residual_max_delta = max(float(residual_max_delta), 1e-3)
        self.heatmap_head = nn.Sequential(
            nn.LayerNorm(patch_dim), nn.Linear(patch_dim, len(OBJECT_NAMES))
        )
        self.relation_aux_enabled = bool(relation_aux_enabled)
        self.relation_topk_candidates = max(int(relation_topk_candidates), 1)
        self.relation_candidate_nms_radius = max(int(relation_candidate_nms_radius), 0)
        self.relation_context_radii = tuple(
            max(float(radius), 0.25) for radius in relation_context_radii
        )
        if not self.relation_context_radii:
            raise ValueError("relation_context_radii must not be empty")
        grid_size = tuple(int(value) for value in relation_grid_size)
        if self.relation_aux_enabled and (
            len(grid_size) != 2 or min(grid_size) <= 0
        ):
            raise ValueError(
                "relation auxiliary head requires relation_grid_size=[height,width]"
            )
        self.relation_grid_h = grid_size[0] if grid_size else 0
        self.relation_grid_w = grid_size[1] if grid_size else 0
        self.relation_head: nn.Module | None = None
        self.context_predictor: nn.Module | None = None
        if self.relation_aux_enabled:
            relation_hidden = max(hidden_dim // 2, 128)
            self.relation_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, relation_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(relation_hidden, int(relation_targets)),
            )
            self.context_predictor = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, relation_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(
                    relation_hidden,
                    len(self.relation_context_radii) * int(patch_dim),
                ),
            )
        relation_dim = patch_dim * 3 + len(OBJECT_NAMES) * 3
        self.relation_proj = nn.Sequential(
            nn.LayerNorm(relation_dim),
            nn.Linear(relation_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        gru_hidden = max(hidden_dim // 2, 1)
        self.temporal = nn.GRU(
            hidden_dim,
            gru_hidden,
            num_layers=max(int(temporal_layers), 1),
            dropout=dropout if int(temporal_layers) > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        temporal_dim = gru_hidden * 2
        self.class_queries = nn.Parameter(torch.empty(num_labels, temporal_dim))
        nn.init.trunc_normal_(self.class_queries, std=0.02)
        self.class_key = nn.Linear(temporal_dim, temporal_dim, bias=False)
        self.residual_head = nn.Sequential(
            nn.LayerNorm(temporal_dim),
            nn.Linear(temporal_dim, max(temporal_dim // 2, 64)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(temporal_dim // 2, 64), 1),
        )
        # Exact identity to the anchor at initialization.
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def predict_heatmaps(self, patch_tokens: Tensor) -> Tensor:
        if patch_tokens.ndim != 4:
            raise ValueError("object spatial aux expects [batch, frames, patches, dim]")
        return self.heatmap_head(patch_tokens)

    def predict_representation_aux(
        self, frame_tokens: Tensor, patch_tokens: Tensor
    ) -> dict[str, Tensor]:
        """Predict training-only targets without altering event logits."""

        heatmap_logits = self.predict_heatmaps(patch_tokens)
        result = {"heatmap_logits": heatmap_logits}
        if not self.relation_aux_enabled:
            return result
        if self.relation_head is None or self.context_predictor is None:
            raise RuntimeError("relation auxiliary modules are missing")
        if frame_tokens.ndim != 3 or frame_tokens.shape[:2] != patch_tokens.shape[:2]:
            raise ValueError(
                "relation auxiliary frame tokens must be [batch,frames,hidden] "
                "and align with patch tokens"
            )
        batch, frames, patches, patch_dim = patch_tokens.shape
        expected_patches = self.relation_grid_h * self.relation_grid_w
        if patches != expected_patches:
            raise ValueError(
                f"relation patch count={patches} must match grid "
                f"{self.relation_grid_h}x{self.relation_grid_w}"
            )

        candidate_count = min(self.relation_topk_candidates, patches)
        ball_scores = heatmap_logits[..., 0]
        nms_radius = self.relation_candidate_nms_radius
        if nms_radius > 0:
            score_maps = ball_scores.float().reshape(
                batch * frames, 1, self.relation_grid_h, self.relation_grid_w
            )
            local_max = F.max_pool2d(
                score_maps, kernel_size=2 * nms_radius + 1,
                stride=1, padding=nms_radius,
            ).reshape(batch, frames, patches)
            candidate_scores = ball_scores.float().masked_fill(
                ball_scores.float() < local_max, float("-inf")
            )
        else:
            candidate_scores = ball_scores.float()
        candidate_values, candidate_indices = torch.topk(
            candidate_scores, k=candidate_count, dim=2
        )
        candidate_weights = F.softmax(
            candidate_values / self.temperature, dim=2
        )
        candidate_x = (
            candidate_indices.remainder(self.relation_grid_w).float() + 0.5
        )
        candidate_y = (
            torch.div(
                candidate_indices,
                self.relation_grid_w,
                rounding_mode="floor",
            ).float()
            + 0.5
        )
        grid_y, grid_x = torch.meshgrid(
            torch.arange(
                self.relation_grid_h,
                device=patch_tokens.device,
                dtype=torch.float32,
            )
            + 0.5,
            torch.arange(
                self.relation_grid_w,
                device=patch_tokens.device,
                dtype=torch.float32,
            )
            + 0.5,
            indexing="ij",
        )
        flat_x = grid_x.flatten().reshape(1, 1, 1, patches)
        flat_y = grid_y.flatten().reshape(1, 1, 1, patches)
        distance2 = (
            flat_x - candidate_x.unsqueeze(-1)
        ).square() + (
            flat_y - candidate_y.unsqueeze(-1)
        ).square()
        context_targets = []
        for radius in self.relation_context_radii:
            spatial_weights = torch.exp(-0.5 * distance2 / (radius * radius))
            spatial_weights = spatial_weights / spatial_weights.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-8)
            candidate_context = torch.einsum(
                "btkp,btpd->btkd",
                spatial_weights.to(patch_tokens.dtype),
                patch_tokens,
            )
            context_targets.append(
                torch.einsum(
                    "btk,btkd->btd",
                    candidate_weights.to(candidate_context.dtype),
                    candidate_context,
                )
            )
        stacked_targets = torch.stack(context_targets, dim=2)
        context_predictions = self.context_predictor(frame_tokens).reshape(
            batch, frames, len(self.relation_context_radii), patch_dim
        )
        entropy = -(
            candidate_weights.clamp_min(1e-8)
            * candidate_weights.clamp_min(1e-8).log()
        ).sum(dim=2) / max(math.log(max(candidate_count, 2)), 1.0)
        result.update(
            {
                "relation_logits": self.relation_head(frame_tokens),
                "context_predictions": context_predictions,
                "context_targets": stacked_targets,
                "candidate_entropy": entropy,
                "candidate_indices": candidate_indices,
                "candidate_weights": candidate_weights,
            }
        )
        return result

    def forward(self, patch_tokens: Tensor) -> dict[str, Tensor]:
        heatmap_logits = self.predict_heatmaps(patch_tokens)
        bsz, frames, patches, _ = heatmap_logits.shape
        topk = min(max(int(math.ceil(patches * self.topk_ratio)), 1), patches)
        per_object_tokens: list[Tensor] = []
        per_object_stats: list[Tensor] = []
        attention_maps: list[Tensor] = []
        for object_index in range(len(OBJECT_NAMES)):
            scores = heatmap_logits[..., object_index]
            values, indices = scores.topk(topk, dim=2)
            # CUDA autocast may promote softmax to fp32 while ``scores`` and
            # its zero-initialized attention buffer remain bf16. Keep the
            # normalization stable in fp32, then cast back before scatter,
            # which requires matching source and destination dtypes.
            sparse_weights = F.softmax(
                values.float() / self.temperature, dim=2
            ).to(scores.dtype)
            gather_index = indices.unsqueeze(-1).expand(-1, -1, -1, patch_tokens.shape[-1])
            selected_tokens = patch_tokens.gather(2, gather_index)
            pooled = (selected_tokens * sparse_weights.unsqueeze(-1)).sum(dim=2)
            per_object_tokens.append(pooled)
            dense_attention = torch.zeros_like(scores).scatter(2, indices, sparse_weights)
            attention_maps.append(dense_attention)
            probabilities = scores.sigmoid()
            entropy = -(
                dense_attention.clamp_min(1e-8) * dense_attention.clamp_min(1e-8).log()
            ).sum(dim=2) / max(math.log(max(topk, 2)), 1.0)
            per_object_stats.append(
                torch.stack(
                    [probabilities.amax(dim=2), probabilities.mean(dim=2), entropy],
                    dim=-1,
                )
            )
        ball_token, goal_token = per_object_tokens
        relation = torch.cat(
            [
                ball_token,
                goal_token,
                ball_token - goal_token,
                *per_object_stats,
            ],
            dim=-1,
        )
        temporal_input = self.relation_proj(relation)
        temporal_tokens, _ = self.temporal(temporal_input)
        keys = self.class_key(temporal_tokens)
        scale = math.sqrt(max(keys.shape[-1], 1))
        class_attention = torch.einsum("ch,bth->bct", self.class_queries, keys) / scale
        class_attention = F.softmax(class_attention, dim=-1)
        class_tokens = torch.einsum("bct,bth->bch", class_attention, temporal_tokens)
        raw_residual = self.residual_head(class_tokens).squeeze(-1)
        residual = self.residual_max_delta * torch.tanh(raw_residual)
        return {
            "heatmap_logits": heatmap_logits,
            "attention_maps": torch.stack(attention_maps, dim=-1),
            "object_tokens": torch.stack(per_object_tokens, dim=2),
            "class_attention": class_attention,
            "raw_residual": raw_residual,
            "residual": residual,
        }


def object_teacher_heatmap_loss(
    outputs: dict[str, Tensor],
    batch: dict[str, Any],
    cfg: Any,
    device: torch.device,
) -> tuple[Tensor, dict[str, float]]:
    logits = outputs.get("object_heatmap_logits")
    if logits is None:
        raise ValueError("object teacher loss requires outputs['object_heatmap_logits']")
    targets = batch["object_heatmap_targets"].to(device, non_blocking=True).to(logits.dtype)
    masks = batch["object_heatmap_masks"].to(device, non_blocking=True).to(logits.dtype)
    if targets.shape != logits.shape or masks.shape != logits.shape:
        raise ValueError(
            f"object heatmap shape mismatch logits={tuple(logits.shape)} "
            f"targets={tuple(targets.shape)} masks={tuple(masks.shape)}"
        )
    aux_cfg = cfg.model.get("object_spatial_aux", {})
    positive_threshold = float(aux_cfg.get("positive_threshold", 0.05))
    negative_weights = torch.tensor(
        aux_cfg.get("negative_weights", [0.02, 0.08]),
        device=device,
        dtype=logits.dtype,
    ).view(1, 1, 1, -1)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    positive = (targets >= positive_threshold).to(logits.dtype) * masks
    negative = (targets < positive_threshold).to(logits.dtype) * masks
    positive_loss = (bce * positive).sum() / positive.sum().clamp_min(1.0)
    # Keep the coefficients as genuine weak-negative weights.  Dividing by
    # their weighted count would cancel them and make detector misses strong.
    negative_loss = (bce * negative * negative_weights).sum() / (
        negative.sum().clamp_min(1.0)
    )
    loss = positive_loss + negative_loss
    with torch.no_grad():
        top_indices = logits.argmax(dim=2, keepdim=True)
        top_hits = targets.gather(2, top_indices).squeeze(2) >= positive_threshold
        object_present = targets.amax(dim=2) >= positive_threshold
        hit_rate = (
            (top_hits & object_present).float().sum()
            / object_present.float().sum().clamp_min(1.0)
        )
    components = {
        "object_heatmap_loss": float(loss.detach().cpu()),
        "object_heatmap_positive_loss": float(positive_loss.detach().cpu()),
        "object_heatmap_negative_loss": float(negative_loss.detach().cpu()),
        "object_heatmap_top1_hit": float(hit_rate.detach().cpu()),
        "object_teacher_valid_fraction": float(masks.mean().detach().cpu()),
        "object_residual_abs_mean": float(
            outputs.get("object_spatial_residual", logits.new_zeros(())).abs().mean().detach().cpu()
        ),
    }
    for object_index, object_name in enumerate(OBJECT_NAMES):
        channel_mask = masks[..., object_index]
        channel_present = object_present[..., object_index]
        channel_valid_frames = channel_mask.amax(dim=2) > 0
        channel_denominator = (channel_present & channel_valid_frames).sum().clamp_min(1)
        components[f"object_{object_name}_teacher_valid_fraction"] = float(
            channel_mask.mean().detach().cpu()
        )
        components[f"object_{object_name}_positive_frame_fraction"] = float(
            (channel_present & channel_valid_frames).float().mean().detach().cpu()
        )
        components[f"object_{object_name}_top1_hit"] = float(
            (
                top_hits[..., object_index]
                & channel_present
                & channel_valid_frames
            ).sum().div(channel_denominator).detach().cpu()
        )
    return loss, components
