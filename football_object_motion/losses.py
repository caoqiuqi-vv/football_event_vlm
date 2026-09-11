"""Losses and diagnostics for the object-motion evidence adapter."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .model import OBJECT_NAMES


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def _weighted_object_mean(values: Tensor, weights: Tensor) -> Tensor:
    if values.ndim != 1 or weights.ndim != 1 or values.shape != weights.shape:
        raise ValueError("object loss values/weights must be matching vectors")
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def _pairwise_residual_ranking(
    residual: Tensor,
    targets: Tensor,
    masks: Tensor,
    *,
    margin: float,
    temperature: float,
) -> tuple[Tensor, int]:
    """Rank positive residual evidence above same-batch negatives per class."""
    losses: list[Tensor] = []
    pair_count = 0
    scale = max(float(temperature), 1e-3)
    for class_index in range(residual.shape[1]):
        valid = masks[:, class_index] > 0
        positive = residual[:, class_index][
            valid & (targets[:, class_index] > 0.5)
        ]
        negative = residual[:, class_index][
            valid & (targets[:, class_index] <= 0.5)
        ]
        if positive.numel() == 0 or negative.numel() == 0:
            continue
        differences = positive[:, None] - negative[None, :]
        losses.append(
            (
                F.softplus((float(margin) - differences) / scale) * scale
            ).mean()
        )
        pair_count += int(differences.numel())
    if not losses:
        return residual.sum() * 0.0, 0
    return torch.stack(losses).mean(), pair_count


def balanced_relation_bce(logits: Tensor, targets: Tensor, masks: Tensor) -> Tensor:
    """Per-class equal positive/negative BCE on ungated relation logits."""
    raw = F.binary_cross_entropy_with_logits(logits, targets.to(logits.dtype), reduction="none")
    valid = masks.to(logits.dtype)
    positive = valid * (targets > 0.5).to(logits.dtype)
    negative = valid * (targets <= 0.5).to(logits.dtype)
    terms = []
    for class_index in range(logits.shape[1]):
        terms.append(0.5 * (_masked_mean(raw[:, class_index], positive[:, class_index]) + _masked_mean(raw[:, class_index], negative[:, class_index])))
    return torch.stack(terms).mean()


def validate_same_video_pairs(
    metas: list[dict[str, Any]],
    targets: Tensor,
    masks: Tensor | None = None,
    *,
    min_gap_sec: float = 5.0,
    skip_unusable_supervision: bool = False,
) -> int:
    """Validate adjacent reviewed positive/negative pairs and class masks."""
    if len(metas) != targets.shape[0] or len(metas) % 2:
        raise ValueError("relation pairing requires an even metadata-aligned batch")
    valid_masks = torch.ones_like(targets) if masks is None else masks
    pairs = 0
    for index in range(0, len(metas), 2):
        positive, negative = metas[index], metas[index + 1]
        class_index = int(positive.get("relation_class_index", -1))
        metadata_legal = (
            class_index >= 0 and class_index < targets.shape[1]
            and positive.get("pair_id") == negative.get("pair_id")
            and positive.get("pair_role") == "positive"
            and negative.get("pair_role") == "negative"
            and positive.get("relation_class_index") == negative.get("relation_class_index")
            and (positive.get("source"), positive.get("video_id")) == (negative.get("source"), negative.get("video_id"))
            and bool(negative.get("reviewed_negative", False))
            and bool(negative.get("full_clean_window", False))
            and bool(negative.get("review_manifest"))
            and float(negative.get("nearest_same_class_gt_gap", -1.0)) >= min_gap_sec
        )
        if not metadata_legal:
            raise ValueError(f"illegal relation pair at rows {index},{index + 1}")
        supervision_usable = bool(
            targets[index, class_index] > 0.5
            and targets[index + 1, class_index] <= 0.5
            and valid_masks[index, class_index] > 0
            and valid_masks[index + 1, class_index] > 0
        )
        if not supervision_usable:
            if skip_unusable_supervision:
                continue
            raise ValueError(
                f"illegal relation supervision at rows {index},{index + 1}"
            )
        pairs += 1
    return pairs



class AdjacentPairResidualQueue:
    """One-item detached queue; epoch order flips which side gets rank gradient."""

    def __init__(self) -> None:
        self.pending: tuple[Tensor, Tensor, Tensor, dict[str, Any]] | None = None

    def reset(self) -> None:
        self.pending = None

    def consume(
        self,
        residual: Tensor,
        targets: Tensor,
        masks: Tensor,
        meta: dict[str, Any],
        *,
        margin: float,
        temperature: float,
        min_gap_sec: float,
    ) -> tuple[Tensor, int]:
        if residual.shape[0] != 1:
            raise ValueError("adjacent pair queue expects batch size one")
        current = (residual, targets, masks, meta)
        if self.pending is None:
            self.pending = (
                residual.detach(), targets.detach(), masks.detach(), dict(meta)
            )
            return residual.sum() * 0.0, 0
        prior = self.pending
        self.pending = None
        if prior[3].get("pair_id") != meta.get("pair_id"):
            raise ValueError("pair queue attempted to cross pair_id boundary")
        rows = [prior, current]
        if rows[0][3].get("pair_role") == "negative":
            rows.reverse()
        positive, negative = rows
        stacked_targets = torch.cat((positive[1], negative[1]), dim=0)
        stacked_masks = torch.cat((positive[2], negative[2]), dim=0)
        valid_pairs = validate_same_video_pairs(
            [positive[3], negative[3]], stacked_targets, stacked_masks,
            min_gap_sec=min_gap_sec,
            skip_unusable_supervision=True,
        )
        if valid_pairs == 0:
            return residual.sum() * 0.0, 0
        class_index = int(positive[3]["relation_class_index"])
        difference = positive[0][0, class_index] - negative[0][0, class_index]
        scale = max(float(temperature), 1e-3)
        return F.softplus((float(margin) - difference) / scale) * scale, 1


_PAIR_RESIDUAL_QUEUES: dict[str, AdjacentPairResidualQueue] = {}


def reset_pair_residual_queues() -> None:
    _PAIR_RESIDUAL_QUEUES.clear()


def _queued_pairwise_ranking(
    residual: Tensor,
    targets: Tensor,
    masks: Tensor,
    metas: list[dict[str, Any]],
    *,
    margin: float,
    temperature: float,
    min_gap_sec: float,
) -> tuple[Tensor, int]:
    if residual.shape[0] != 1 or len(metas) != 1:
        raise ValueError("pair queue requires one metadata-aligned sample")
    key = f"{residual.device.type}:{residual.device.index}"
    queue = _PAIR_RESIDUAL_QUEUES.setdefault(key, AdjacentPairResidualQueue())
    return queue.consume(
        residual, targets, masks, metas[0], margin=margin,
        temperature=temperature, min_gap_sec=min_gap_sec,
    )

def _validated_pairwise_ranking(residual: Tensor, targets: Tensor, masks: Tensor, metas: list[dict[str, Any]], *, margin: float, temperature: float, min_gap_sec: float) -> tuple[Tensor, int]:
    count = validate_same_video_pairs(metas, targets, masks, min_gap_sec=min_gap_sec)
    scale = max(float(temperature), 1e-3)
    losses = []
    for index in range(0, len(metas), 2):
        class_index = int(metas[index]["relation_class_index"])
        difference = residual[index, class_index] - residual[index + 1, class_index]
        losses.append(F.softplus((float(margin) - difference) / scale) * scale)
    return torch.stack(losses).mean(), count


def bidirectional_guard_loss(final: Tensor, anchor: Tensor, targets: Tensor, masks: Tensor, metas: list[dict[str, Any]], *, negative_margin: float = 0.15) -> tuple[Tensor, int, int]:
    """No positive suppression; only reviewed full-clean negatives may be capped."""
    valid = masks.to(final.dtype)
    positive = valid * (targets > 0.5).to(final.dtype)
    up = F.relu(anchor.detach() - final) * positive
    reviewed = final.new_tensor([bool(meta.get("reviewed_negative", False)) and bool(meta.get("full_clean_window", False)) for meta in metas]).unsqueeze(1)
    negative = valid * (targets <= 0.5).to(final.dtype) * reviewed
    down = F.relu(final - anchor.detach() - negative_margin) * negative
    return _masked_mean(up, positive) + _masked_mean(down, negative), int((up > 0).sum()), int((down > 0).sum())


def _distribution_quality_mask(target_mass: Tensor, masks: Tensor) -> Tensor:
    return (target_mass > 0).to(masks.dtype) * masks.amax(dim=2)


def _motion_quality_pair_mask(coordinate_masks: Tensor, motion_quality: Tensor | None) -> Tensor:
    result = coordinate_masks.clone()
    if result.shape[1] > 1:
        result[:, 1:] = result[:, 1:] * coordinate_masks[:, :-1]
        result[:, 0] = 0.0
    if motion_quality is not None:
        result = result * motion_quality.to(result.device, result.dtype)
    return result


def _centers_from_heatmaps(targets: Tensor, grid_h: int, grid_w: int) -> Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, grid_h, device=targets.device, dtype=targets.dtype),
        torch.linspace(-1.0, 1.0, grid_w, device=targets.device, dtype=targets.dtype),
        indexing="ij",
    )
    grid = torch.stack((xx, yy), dim=-1).reshape(grid_h * grid_w, 2)
    weights = targets / targets.sum(dim=2, keepdim=True).clamp_min(1e-6)
    return torch.einsum("btpo,pd->btod", weights, grid)



def ball_feature_preserve_mask(
    ball_targets: Tensor,
    *,
    grid_h: int,
    grid_w: int,
    dilation_patches: int = 2,
) -> Tensor:
    """Mask non-ball patches after excluding a spatially dilated ball region."""
    if ball_targets.shape[-1] != int(grid_h) * int(grid_w):
        raise ValueError("ball preserve grid does not match patch count")
    flat = (ball_targets > 0).to(ball_targets.dtype)
    shaped = flat.reshape(-1, 1, int(grid_h), int(grid_w))
    radius = max(int(dilation_patches), 0)
    if radius:
        shaped = F.max_pool2d(shaped, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return (1.0 - shaped).reshape_as(ball_targets)


def _ball_lora_losses(
    outputs: dict[str, Tensor],
    targets: Tensor,
    masks: Tensor,
    coordinate_targets: Tensor,
    motion_quality: Tensor | None,
    object_cfg: Any,
) -> tuple[dict[str, Tensor], float]:
    live_logits = outputs.get("object_motion_ball_lora_logits")
    features = outputs.get("object_motion_ball_student_features")
    centers = outputs.get("object_motion_ball_student_center")
    if not (torch.is_tensor(live_logits) and torch.is_tensor(features) and torch.is_tensor(centers)):
        zero = targets.sum() * 0.0
        return {"ball_strong": zero, "ball_center": zero, "ball_contrastive": zero, "ball_track": zero, "ball_preserve": zero}, 1.0
    ball_targets = targets[..., 0]
    ball_masks = masks[..., 0]
    strong = ((ball_masks.amax(dim=2) >= 0.999) & (ball_targets.sum(dim=2) > 0)).to(live_logits.dtype)
    localization = F.binary_cross_entropy_with_logits(live_logits, ball_targets, reduction="none")
    positive = (ball_targets >= float(object_cfg.get("positive_threshold", 0.05))).to(localization.dtype)
    negative = 1.0 - positive
    positive_loss = (localization * positive).sum(dim=2) / positive.sum(dim=2).clamp_min(1.0)
    negative_loss = (localization * negative).sum(dim=2) / negative.sum(dim=2).clamp_min(1.0)
    ball_strong = _masked_mean(
        positive_loss + float(object_cfg.get("ball_negative_loss_weight", 0.05)) * negative_loss,
        strong,
    )

    normalized = F.normalize(features.float(), dim=-1).to(features.dtype)
    target_weights = ball_targets / ball_targets.sum(dim=2, keepdim=True).clamp_min(1e-6)
    positive_feature = F.normalize(torch.einsum("btp,btpd->btd", target_weights, normalized).float(), dim=-1).to(features.dtype)
    similarity = torch.einsum("btpd,btd->btp", normalized, positive_feature)
    image_size = object_cfg.get("image_size", [512, 896])
    patch_size = int(
        object_cfg.get(
            "heatmap_patch_size",
            object_cfg.get("patch_size", 16),
        )
    )
    grid_h, grid_w = int(image_size[0]) // patch_size, int(image_size[1]) // patch_size
    nonball = ball_feature_preserve_mask(ball_targets, grid_h=grid_h, grid_w=grid_w)
    positive_similarity = (similarity * target_weights).sum(dim=2)
    negative_similarity = (similarity * nonball).sum(dim=2) / nonball.sum(dim=2).clamp_min(1.0)
    contrastive = _masked_mean(F.relu(0.2 + negative_similarity - positive_similarity), strong)

    teacher_center = coordinate_targets[..., 0, :2]
    ball_center = _masked_mean(
        F.smooth_l1_loss(
            centers, teacher_center, reduction="none", beta=0.05
        ).mean(dim=-1),
        strong,
    )
    predicted_delta = torch.zeros_like(centers)
    teacher_delta = torch.zeros_like(teacher_center)
    track_mask = torch.zeros_like(strong)
    if centers.shape[1] > 1:
        predicted_delta[:, 1:] = centers[:, 1:] - centers[:, :-1]
        teacher_delta[:, 1:] = teacher_center[:, 1:] - teacher_center[:, :-1]
        track_mask[:, 1:] = strong[:, 1:] * strong[:, :-1]
    if motion_quality is not None:
        track_mask = track_mask * motion_quality[..., 0].to(track_mask.dtype)
    track = _masked_mean(
        F.smooth_l1_loss(predicted_delta, teacher_delta, reduction="none", beta=0.05).mean(dim=-1),
        track_mask,
    )

    live_preserve = outputs.get("object_motion_ball_preserve_features")
    reference = outputs.get("object_motion_ball_preserve_reference")
    indices = outputs.get("object_motion_ball_preserve_indices")
    preserve = live_logits.sum() * 0.0
    background_cosine = 1.0
    if torch.is_tensor(live_preserve) and torch.is_tensor(reference) and torch.is_tensor(indices):
        selected_targets = ball_targets.index_select(1, indices.to(ball_targets.device))
        preserve_patch_size = int(object_cfg.get("patch_size", 16))
        preserve_grid_h = int(image_size[0]) // preserve_patch_size
        preserve_grid_w = int(image_size[1]) // preserve_patch_size
        if selected_targets.shape[2] != live_preserve.shape[2]:
            selected_targets = F.interpolate(
                selected_targets.reshape(
                    -1, 1, grid_h, grid_w
                ),
                size=(preserve_grid_h, preserve_grid_w),
                mode="area",
            ).reshape(
                selected_targets.shape[0],
                selected_targets.shape[1],
                preserve_grid_h * preserve_grid_w,
            )
        preserve_mask = ball_feature_preserve_mask(
            selected_targets,
            grid_h=preserve_grid_h,
            grid_w=preserve_grid_w,
        )
        cosine = F.cosine_similarity(live_preserve.float(), reference.float(), dim=-1).to(live_logits.dtype)
        preserve = _masked_mean(1.0 - cosine, preserve_mask)
        background_cosine = float(_masked_mean(cosine.detach(), preserve_mask).cpu())
    return {
        "ball_strong": ball_strong,
        "ball_center": ball_center,
        "ball_contrastive": contrastive,
        "ball_track": track,
        "ball_preserve": preserve,
    }, background_cosine


def trusted_no_evidence_mask(targets: Tensor, masks: Tensor) -> Tensor:
    """Only explicit absence in both ball/goal channels justifies suppression.

    Missing pseudo-labels have mask zero and must never become event negatives.
    Person is not required because several recipes deliberately omit it.
    """
    return ((targets[..., :2] < 0.5) & (masks[..., :2] >= 0.999)).all(dim=-1)


def object_motion_auxiliary_loss(
    outputs: dict[str, Tensor],
    batch: dict[str, Any],
    cfg: Any,
    device: torch.device,
    *,
    clip_targets: Tensor,
    clip_label_masks: Tensor,
) -> tuple[Tensor, dict[str, float]]:
    """Compute object, motion, frame and dense-ranking surrogate losses."""
    required = (
        "object_motion_heatmap_logits",
        "object_motion_presence_logits",
        "object_motion_centers",
        "object_motion_velocity",
        "object_motion_frame_logits",
        "object_motion_clip_residual",
    )
    missing = [key for key in required if key not in outputs]
    if missing:
        raise ValueError(f"object motion outputs missing keys: {missing}")
    targets = batch["object_motion_heatmap_targets"].to(
        device, non_blocking=True
    ).to(outputs["object_motion_heatmap_logits"].dtype)
    masks = batch["object_motion_heatmap_masks"].to(
        device, non_blocking=True
    ).to(targets.dtype)
    presence_targets = batch["object_motion_presence_targets"].to(
        device, non_blocking=True
    ).to(targets.dtype)
    presence_masks = batch["object_motion_presence_masks"].to(
        device, non_blocking=True
    ).to(targets.dtype)
    coordinate_targets = batch["object_motion_coordinate_targets"].to(
        device, non_blocking=True
    ).to(targets.dtype)
    coordinate_masks = batch["object_motion_coordinate_masks"].to(
        device, non_blocking=True
    ).to(targets.dtype)
    frame_times = batch["object_motion_times"].to(
        device, non_blocking=True
    ).to(targets.dtype)
    logits = outputs["object_motion_heatmap_logits"]
    if logits.shape != targets.shape or masks.shape != targets.shape:
        raise ValueError(
            f"object motion heatmap shape mismatch logits={tuple(logits.shape)} "
            f"targets={tuple(targets.shape)} masks={tuple(masks.shape)}"
        )
    object_cfg = cfg.model.get("object_motion", {})
    train_cfg = cfg.train
    positive_threshold = float(object_cfg.get("positive_threshold", 0.05))
    negative_weights = logits.new_tensor(
        object_cfg.get("negative_weights", [0.02, 0.08, 0.04])
    )
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    positive = (targets >= positive_threshold).to(logits.dtype) * masks
    negative = (targets < positive_threshold).to(logits.dtype) * masks
    object_weights = logits.new_tensor(
        object_cfg.get("object_loss_weights", [2.0, 1.0, 0.25])
    )
    if object_weights.numel() != len(OBJECT_NAMES):
        raise ValueError("model.object_motion.object_loss_weights must have 3 values")
    heatmap_positive_by_object = torch.stack(
        [
            _masked_mean(bce[..., index], positive[..., index])
            for index in range(len(OBJECT_NAMES))
        ]
    )
    heatmap_negative_by_object = torch.stack(
        [
            negative_weights[index]
            * _masked_mean(bce[..., index], negative[..., index])
            for index in range(len(OBJECT_NAMES))
        ]
    )
    heatmap_positive = _weighted_object_mean(
        heatmap_positive_by_object, object_weights
    )
    heatmap_negative = _weighted_object_mean(
        heatmap_negative_by_object, object_weights
    )
    heatmap_loss = _weighted_object_mean(
        heatmap_positive_by_object + heatmap_negative_by_object,
        object_weights,
    )

    target_mass = targets.sum(dim=2)
    target_distribution = targets / target_mass.unsqueeze(2).clamp_min(1e-6)
    distribution_matrix = -(
        target_distribution
        * F.log_softmax(logits.float(), dim=2).to(logits.dtype)
    ).sum(dim=2) / max(math.log(max(int(logits.shape[2]), 2)), 1.0)
    distribution_mask = _distribution_quality_mask(target_mass, masks).to(logits.dtype)
    distribution_loss_by_object = torch.stack(
        [
            _masked_mean(
                distribution_matrix[..., index],
                distribution_mask[..., index],
            )
            for index in range(len(OBJECT_NAMES))
        ]
    )
    distribution_loss = _weighted_object_mean(
        distribution_loss_by_object, object_weights
    )

    presence_logits = outputs["object_motion_presence_logits"]
    presence_bce = F.binary_cross_entropy_with_logits(
        presence_logits, presence_targets, reduction="none"
    )
    presence_negative_weights = presence_logits.new_tensor(
        object_cfg.get("presence_negative_weights", [0.10, 0.20, 0.50])
    )
    presence_positive = (presence_targets > 0.5).to(targets.dtype) * presence_masks
    presence_negative = (presence_targets <= 0.5).to(targets.dtype) * presence_masks
    presence_loss_by_object = torch.stack(
        [
            _masked_mean(
                presence_bce[..., index], presence_positive[..., index]
            )
            + presence_negative_weights[index]
            * _masked_mean(
                presence_bce[..., index], presence_negative[..., index]
            )
            for index in range(len(OBJECT_NAMES))
        ]
    )
    presence_loss = _weighted_object_mean(
        presence_loss_by_object, object_weights
    )

    centers = outputs["object_motion_centers"]
    centers_for_generic_loss = torch.cat((centers[:, :, :1].detach(), centers[:, :, 1:]), dim=2)
    center_targets = coordinate_targets[..., :2]
    # Person supervision is a union heatmap and has no single semantically
    # correct coordinate.  Its coordinate mask is therefore zero in Teacher.
    coordinate_loss = _masked_mean(
        F.smooth_l1_loss(centers_for_generic_loss, center_targets, reduction="none").mean(dim=-1),
        coordinate_masks,
    )

    teacher_centers = center_targets
    teacher_displacement = torch.zeros_like(teacher_centers)
    predicted_displacement = torch.zeros_like(centers_for_generic_loss)
    if frame_times.shape[1] > 1:
        # Supervise normalized adjacent-frame displacement instead of dividing
        # detector coordinate jitter by ~0.125 s. The adapter still exposes
        # physical per-second velocity as a relation feature, while this loss
        # remains bounded and robust to small Teacher-box fluctuations.
        teacher_displacement[:, 1:] = (
            teacher_centers[:, 1:] - teacher_centers[:, :-1]
        )
        predicted_displacement[:, 1:] = centers_for_generic_loss[:, 1:] - centers_for_generic_loss[:, :-1]
    raw_motion_quality = batch.get("object_motion_motion_quality")
    motion_quality = raw_motion_quality.to(device, non_blocking=True) if torch.is_tensor(raw_motion_quality) else None
    pair_mask = _motion_quality_pair_mask(coordinate_masks, motion_quality)
    motion_loss = _masked_mean(
        F.smooth_l1_loss(
            predicted_displacement,
            teacher_displacement,
            reduction="none",
            beta=0.05,
        ).mean(dim=-1),
        pair_mask,
    )

    frame_targets = batch["object_motion_frame_targets"].to(
        device, non_blocking=True
    ).to(targets.dtype)
    frame_masks = batch["object_motion_frame_target_masks"].to(
        device, non_blocking=True
    ).to(targets.dtype)
    frame_logits = outputs["object_motion_frame_logits"]
    frame_bce = F.binary_cross_entropy_with_logits(
        frame_logits, frame_targets, reduction="none"
    )
    frame_positive = (
        (frame_targets >= positive_threshold).to(frame_bce.dtype) * frame_masks
    )
    frame_negative = (
        (frame_targets < positive_threshold).to(frame_bce.dtype) * frame_masks
    )
    frame_positive_loss = _masked_mean(frame_bce, frame_positive)
    frame_negative_loss = _masked_mean(frame_bce, frame_negative)
    frame_loss = 0.5 * (frame_positive_loss + frame_negative_loss)

    residual = outputs["object_motion_clip_residual"]
    valid = clip_label_masks.to(residual.dtype)
    ranking_residual = outputs.get(
        "object_motion_adapter_clip_residual", residual
    )
    ranking_target = str(object_cfg.get("event_ranking_target", "residual"))
    if ranking_target == "final":
        ranking_residual = outputs["logits"]
    elif ranking_target != "residual":
        raise ValueError("event_ranking_target must be residual or final")
    relation_logits = outputs.get("object_motion_raw_clip_residual_logits", ranking_residual)
    relation_loss = balanced_relation_bce(relation_logits, clip_targets, clip_label_masks)
    if bool(object_cfg.get("require_true_pairs", False)) and batch.get("meta", [{}])[0].get("pair_id"):
        ranking_function = _queued_pairwise_ranking if ranking_residual.shape[0] == 1 else _validated_pairwise_ranking
        dense_rank_loss, dense_rank_pair_count = ranking_function(
            ranking_residual, clip_targets, clip_label_masks, batch.get("meta", []),
            margin=float(object_cfg.get("pairwise_rank_margin", 0.05)),
            temperature=float(object_cfg.get("pairwise_rank_temperature", 0.10)),
            min_gap_sec=float(object_cfg.get("pair_min_gap_sec", 5.0)),
        )
    elif bool(object_cfg.get("require_true_pairs", False)):
        dense_rank_loss, dense_rank_pair_count = ranking_residual.sum() * 0.0, 0
    else:
        dense_rank_loss, dense_rank_pair_count = _pairwise_residual_ranking(
            ranking_residual, clip_targets, clip_label_masks,
            margin=float(object_cfg.get("pairwise_rank_margin", 0.05)),
            temperature=float(object_cfg.get("pairwise_rank_temperature", 0.10)),
        )
    anchor_logits = outputs.get("retention_reference_logits", outputs.get("logits", relation_logits).detach())
    final_logits = outputs.get("logits", anchor_logits)
    guard_loss, upward_violations, downward_violations = bidirectional_guard_loss(final_logits, anchor_logits, clip_targets, clip_label_masks, batch.get("meta", [{} for _ in range(final_logits.shape[0])]), negative_margin=float(object_cfg.get("negative_guard_margin", 0.15)))

    no_evidence = trusted_no_evidence_mask(presence_targets, presence_masks).to(residual.dtype)
    frame_residual = outputs["object_motion_frame_residual"]
    no_evidence_loss = _masked_mean(
        frame_residual.square().mean(dim=-1), no_evidence
    )

    adapter_clip_residual = outputs.get(
        "object_motion_adapter_clip_residual", residual
    )
    adapter_frame_residual = outputs.get(
        "object_motion_adapter_frame_residual", frame_residual
    )
    residual_energy_loss = (
        _masked_mean(adapter_clip_residual.square(), valid)
        + _masked_mean(
            adapter_frame_residual.square(),
            frame_masks,
        )
    )
    learned_clip_gate = outputs.get(
        "object_motion_learned_clip_gate", outputs["object_motion_clip_gate"]
    )
    learned_frame_gate = outputs.get(
        "object_motion_learned_frame_gate", outputs["object_motion_frame_gate"]
    )
    gate_budget = min(
        max(float(object_cfg.get("learned_gate_budget", 0.25)), 0.0), 1.0
    )
    gate_budget_loss = (
        F.relu(learned_clip_gate - gate_budget).square().mean()
        + F.relu(learned_frame_gate - gate_budget).square().mean()
    )
    raw_clip_residual = outputs.get("object_motion_raw_clip_residual_logits")
    raw_frame_residual = outputs.get("object_motion_raw_frame_residual")
    saturation_threshold = min(
        max(float(object_cfg.get("residual_saturation_threshold", 0.85)), 0.0),
        0.999,
    )
    saturation_terms: list[Tensor] = []
    saturation_fractions: list[Tensor] = []
    for raw_value in (raw_clip_residual, raw_frame_residual):
        if not torch.is_tensor(raw_value):
            continue
        normalized = raw_value.tanh().abs()
        saturation_terms.append(
            F.relu(normalized - saturation_threshold).square().mean()
        )
        saturation_fractions.append(
            (normalized >= saturation_threshold).to(normalized.dtype).mean()
        )
    if saturation_terms:
        saturation_loss = torch.stack(saturation_terms).mean()
        saturation_fraction = torch.stack(saturation_fractions).mean()
    else:
        saturation_loss = residual.sum() * 0.0
        saturation_fraction = residual.detach().sum() * 0.0

    ball_lora_losses, background_cosine = _ball_lora_losses(
        outputs,
        targets,
        masks,
        coordinate_targets,
        motion_quality,
        object_cfg,
    )

    weights = {
        "heatmap": float(train_cfg.get("object_motion_heatmap_loss_weight", 0.5) or 0.0),
        "distribution": float(train_cfg.get("object_motion_distribution_loss_weight", 0.0) or 0.0),
        "presence": float(train_cfg.get("object_motion_presence_loss_weight", 0.25) or 0.0),
        "coordinate": float(train_cfg.get("object_motion_coordinate_loss_weight", 0.25) or 0.0),
        "motion": float(train_cfg.get("object_motion_consistency_loss_weight", 0.20) or 0.0),
        "frame": float(train_cfg.get("object_motion_frame_loss_weight", 0.50) or 0.0),
        "ranking": float(train_cfg.get("object_motion_dense_rank_loss_weight", 0.15) or 0.0),
        "relation": float(train_cfg.get("object_motion_relation_loss_weight", 1.0) or 0.0),
        "guard": float(train_cfg.get("object_motion_guard_loss_weight", 1.0) or 0.0),
        "no_evidence": float(train_cfg.get("object_motion_no_evidence_loss_weight", 0.05) or 0.0),
        "residual_energy": float(train_cfg.get("object_motion_residual_energy_loss_weight", 0.0) or 0.0),
        "gate_budget": float(train_cfg.get("object_motion_gate_budget_loss_weight", 0.0) or 0.0),
        "saturation": float(train_cfg.get("object_motion_saturation_loss_weight", 0.0) or 0.0),
        "ball_strong": float(train_cfg.get("object_motion_ball_strong_loss_weight", 1.0) or 0.0),
        "ball_center": float(train_cfg.get("object_motion_ball_center_loss_weight", 0.5) or 0.0),
        "ball_contrastive": float(train_cfg.get("object_motion_ball_contrastive_loss_weight", 0.2) or 0.0),
        "ball_track": float(train_cfg.get("object_motion_ball_track_loss_weight", 0.1) or 0.0),
        "ball_preserve": float(train_cfg.get("object_motion_ball_preserve_loss_weight", 0.05) or 0.0),
    }
    losses = {
        "heatmap": heatmap_loss,
        "distribution": distribution_loss,
        "presence": presence_loss,
        "coordinate": coordinate_loss,
        "motion": motion_loss,
        "frame": frame_loss,
        "ranking": dense_rank_loss,
        "relation": relation_loss,
        "guard": guard_loss,
        "no_evidence": no_evidence_loss,
        "residual_energy": residual_energy_loss,
        "gate_budget": gate_budget_loss,
        "saturation": saturation_loss,
        **ball_lora_losses,
    }
    total = sum(weights[name] * value for name, value in losses.items())

    components: dict[str, float] = {
        "object_motion_aux_loss": float(total.detach().cpu()),
        "object_motion_heatmap_loss": float(heatmap_loss.detach().cpu()),
        "object_motion_heatmap_positive_loss": float(heatmap_positive.detach().cpu()),
        "object_motion_heatmap_negative_loss": float(heatmap_negative.detach().cpu()),
        "object_motion_distribution_loss": float(distribution_loss.detach().cpu()),
        "object_motion_presence_loss": float(presence_loss.detach().cpu()),
        "object_motion_coordinate_loss": float(coordinate_loss.detach().cpu()),
        "object_motion_consistency_loss": float(motion_loss.detach().cpu()),
        "object_motion_frame_loss": float(frame_loss.detach().cpu()),
        "object_motion_frame_positive_loss": float(frame_positive_loss.detach().cpu()),
        "object_motion_frame_negative_loss": float(frame_negative_loss.detach().cpu()),
        "object_motion_dense_rank_loss": float(dense_rank_loss.detach().cpu()),
        "object_motion_dense_rank_pairs": float(dense_rank_pair_count),
        "object_motion_relation_loss": float(relation_loss.detach().cpu()),
        "object_motion_upward_violations": float(upward_violations),
        "object_motion_downward_violations": float(downward_violations),
        "object_motion_no_evidence_loss": float(no_evidence_loss.detach().cpu()),
        "object_motion_residual_energy_loss": float(residual_energy_loss.detach().cpu()),
        "object_motion_gate_budget_loss": float(gate_budget_loss.detach().cpu()),
        "object_motion_saturation_loss": float(saturation_loss.detach().cpu()),
        "object_motion_ball_strong_loss": float(ball_lora_losses["ball_strong"].detach().cpu()),
        "object_motion_ball_center_loss": float(ball_lora_losses["ball_center"].detach().cpu()),
        "object_motion_ball_contrastive_loss": float(ball_lora_losses["ball_contrastive"].detach().cpu()),
        "object_motion_ball_track_loss": float(ball_lora_losses["ball_track"].detach().cpu()),
        "object_motion_ball_preserve_loss": float(ball_lora_losses["ball_preserve"].detach().cpu()),
        "object_motion_ball_background_cosine": background_cosine,
        "object_motion_ball_entropy_mean": float(
            outputs.get("object_motion_ball_student_entropy", targets.new_zeros(())).detach().float().mean().cpu()
        ),
        "object_motion_ball_layer_weight_max": float(
            outputs.get("object_motion_ball_layer_weights", targets.new_zeros(())).detach().float().max().cpu()
        ),
        "object_motion_residual_saturation_fraction": float(
            saturation_fraction.detach().cpu()
        ),
        "object_motion_clip_residual_abs": float(residual.abs().mean().detach().cpu()),
        "object_motion_frame_residual_abs": float(frame_residual.abs().mean().detach().cpu()),
        "object_motion_clip_gate_mean": float(outputs["object_motion_clip_gate"].mean().detach().cpu()),
        "object_motion_frame_gate_mean": float(outputs["object_motion_frame_gate"].mean().detach().cpu()),
        "object_motion_learned_clip_gate_mean": float(
            learned_clip_gate.mean().detach().cpu()
        ),
        "object_motion_learned_frame_gate_mean": float(
            learned_frame_gate.mean().detach().cpu()
        ),
        "object_motion_ball_evidence_gate_mean": float(
            outputs.get(
                "object_motion_frame_evidence_gate",
                torch.ones_like(learned_frame_gate[..., 0]),
            ).mean().detach().cpu()
        ),
        "object_motion_anchor_local_coverage": float(
            outputs.get(
                "object_motion_anchor_local_coverage",
                torch.ones_like(residual),
            ).mean().detach().cpu()
        ),
        "object_motion_teacher_valid_fraction": float(batch.get("object_motion_teacher_filled", masks).float().mean().detach().cpu()),
        "object_motion_event_residual_scale": float(
            outputs.get(
                "object_motion_event_residual_scale",
                residual.new_tensor(1.0),
            ).detach().cpu()
        ),
        "object_motion_shared_anchor_delta_abs": float(
            outputs.get(
                "object_motion_shared_anchor_delta",
                torch.zeros_like(residual),
            ).detach().abs().mean().cpu()
        ),
        "object_motion_object_token_gate_mean": float(
            outputs.get(
                "object_motion_object_token_gate",
                torch.zeros_like(residual),
            ).detach().mean().cpu()
        ),
    }
    with torch.no_grad():
        for object_index, object_name in enumerate(OBJECT_NAMES):
            components[f"object_motion_{object_name}_heatmap_loss"] = float(
                (
                    heatmap_positive_by_object[object_index]
                    + heatmap_negative_by_object[object_index]
                ).detach().cpu()
            )
            components[f"object_motion_{object_name}_presence_loss"] = float(
                presence_loss_by_object[object_index].detach().cpu()
            )
            components[f"object_motion_{object_name}_distribution_loss"] = float(
                distribution_loss_by_object[object_index].detach().cpu()
            )
            object_present = presence_targets[..., object_index] > 0.5
            top_index = logits[..., object_index].argmax(dim=2, keepdim=True)
            hit = targets[..., object_index].gather(2, top_index).squeeze(2) >= positive_threshold
            components[f"object_motion_{object_name}_top1_hit"] = float(
                (hit & object_present).float().sum().div(
                    object_present.float().sum().clamp_min(1.0)
                ).cpu()
            )
            prediction = presence_logits[..., object_index].sigmoid() >= 0.5
            presence_valid = presence_masks[..., object_index] > 0
            components[f"object_motion_{object_name}_presence_accuracy"] = float(
                ((prediction == object_present) * presence_valid).float().sum().div(
                    presence_valid.float().sum().clamp_min(1.0)
                ).cpu()
            )
            presence_tp = (prediction & object_present & presence_valid).float().sum()
            presence_fp = (
                prediction & ~object_present & presence_valid
            ).float().sum()
            presence_fn = (
                ~prediction & object_present & presence_valid
            ).float().sum()
            components[f"object_motion_{object_name}_presence_precision"] = float(
                presence_tp.div((presence_tp + presence_fp).clamp_min(1.0)).cpu()
            )
            components[f"object_motion_{object_name}_presence_recall"] = float(
                presence_tp.div((presence_tp + presence_fn).clamp_min(1.0)).cpu()
            )
    return total, components
