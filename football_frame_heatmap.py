"""Frame-heatmap aggregation policies for sparse event supervision."""

from __future__ import annotations

import torch
from torch import Tensor


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def positive_core_context_heatmap_loss(
    heatmap_matrix: Tensor,
    frame_targets: Tensor,
    frame_masks: Tensor,
    clip_targets: Tensor,
    clip_label_masks: Tensor,
    *,
    core_min_target: float = 0.5,
    context_max_target: float = 0.1,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Equalize event-core/context inside each trusted positive clip/class.

    Positive slots are normalized independently over core and context frames,
    then the available sides are averaged 1:1. Gaussian shoulders are ignored.
    Negative clip/classes retain ordinary masked focal/BCE aggregation. The final
    positive-vs-negative mixture is weighted by trusted clip/class slot counts,
    not by the number of frames in each side.
    """
    if not 0.0 <= context_max_target < core_min_target <= 1.0:
        raise ValueError(
            "heatmap thresholds require 0 <= context_max < core_min <= 1"
        )
    if heatmap_matrix.shape != frame_targets.shape or frame_masks.shape != frame_targets.shape:
        raise ValueError("heatmap matrix/targets/masks must share [B,T,C] shape")
    positive_slots = clip_label_masks * (clip_targets > 0.5).to(heatmap_matrix.dtype)
    negative_slots = clip_label_masks * (clip_targets <= 0.5).to(heatmap_matrix.dtype)
    positive_frame_mask = frame_masks * (positive_slots > 0).unsqueeze(1)
    core_mask = positive_frame_mask * (frame_targets >= core_min_target).to(
        heatmap_matrix.dtype
    )
    context_mask = positive_frame_mask * (frame_targets <= context_max_target).to(
        heatmap_matrix.dtype
    )
    negative_frame_mask = frame_masks * (negative_slots > 0).unsqueeze(1)

    core_sum = (heatmap_matrix * core_mask).sum(dim=1)
    core_count = core_mask.sum(dim=1)
    context_sum = (heatmap_matrix * context_mask).sum(dim=1)
    context_count = context_mask.sum(dim=1)
    core_slot_loss = core_sum / core_count.clamp_min(1.0)
    context_slot_loss = context_sum / context_count.clamp_min(1.0)
    has_core = (core_count > 0).to(heatmap_matrix.dtype)
    has_context = (context_count > 0).to(heatmap_matrix.dtype)
    available_sides = has_core + has_context
    positive_slot_loss = (
        core_slot_loss * has_core + context_slot_loss * has_context
    ) / available_sides.clamp_min(1.0)
    usable_positive_slots = positive_slots * (available_sides > 0).to(
        heatmap_matrix.dtype
    )
    positive_loss = _masked_mean(positive_slot_loss, usable_positive_slots)
    negative_loss = _masked_mean(heatmap_matrix, negative_frame_mask)

    positive_count = usable_positive_slots.sum()
    negative_count = negative_slots.sum()
    total = (
        positive_loss * positive_count + negative_loss * negative_count
    ) / (positive_count + negative_count).clamp_min(1.0)
    diagnostics = {
        "frame_heatmap_core_loss": _masked_mean(heatmap_matrix, core_mask),
        "frame_heatmap_context_loss": _masked_mean(heatmap_matrix, context_mask),
        "frame_heatmap_core_slots": core_mask.sum().detach(),
        "frame_heatmap_context_slots": context_mask.sum().detach(),
        "frame_heatmap_balanced_pos_slots": positive_count.detach(),
        "frame_heatmap_balanced_neg_slots": negative_count.detach(),
        "frame_heatmap_balanced_pos_loss": positive_loss,
        "frame_heatmap_balanced_neg_loss": negative_loss,
    }
    return total, diagnostics


__all__ = ["positive_core_context_heatmap_loss"]
