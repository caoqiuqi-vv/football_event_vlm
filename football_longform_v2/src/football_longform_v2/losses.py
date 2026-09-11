from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def sigmoid_focal_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    valid_mask: Tensor | None = None,
    alpha: float = 0.25,
    gamma: float = 2.0,
    density_balanced: bool = False,
) -> Tensor:
    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have identical shapes")
    probability = logits.sigmoid()
    if density_balanced:
        positive = -targets * (1.0 - probability).pow(gamma) * F.logsigmoid(logits)
        negative = -(1.0 - targets) * probability.pow(gamma) * F.logsigmoid(-logits)
        if valid_mask is not None:
            if valid_mask.ndim == logits.ndim - 1:
                valid_mask = valid_mask.unsqueeze(-1)
            weight = valid_mask.to(logits.dtype).expand_as(logits)
        else:
            weight = torch.ones_like(logits)
        positive = positive * weight
        negative = negative * weight
        reduce_dims = tuple(range(logits.ndim - 1))
        raw_positive_mass = (targets * weight).sum(dim=reduce_dims)
        raw_negative_mass = ((1.0 - targets) * weight).sum(dim=reduce_dims)
        positive_by_channel = positive.sum(dim=reduce_dims) / raw_positive_mass.clamp_min(1.0)
        negative_by_channel = negative.sum(dim=reduce_dims) / raw_negative_mass.clamp_min(1.0)
        positive_active = raw_positive_mass > 0
        negative_active = raw_negative_mass > 0
        positive_loss = (
            positive_by_channel[positive_active].mean()
            if positive_active.any() else positive.sum() * 0.0
        )
        negative_loss = (
            negative_by_channel[negative_active].mean()
            if negative_active.any() else negative.sum() * 0.0
        )
        effective_alpha = 0.5 if alpha < 0 else alpha
        return effective_alpha * positive_loss + (1.0 - effective_alpha) * negative_loss
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probability * targets + (1.0 - probability) * (1.0 - targets)
    loss = bce * (1.0 - p_t).pow(gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss
    if valid_mask is not None:
        if valid_mask.ndim == logits.ndim - 1:
            valid_mask = valid_mask.unsqueeze(-1)
        weight = valid_mask.to(loss.dtype).expand_as(loss)
        return (loss * weight).sum() / weight.sum().clamp_min(1.0)
    return loss.mean()


def locator_loss(
    outputs: dict[str, Tensor],
    heatmap_targets: Tensor,
    *,
    offset_targets: Tensor | None = None,
    valid_mask: Tensor | None = None,
    alpha: float = 0.25,
    gamma: float = 2.0,
    offset_weight: float = 0.5,
    density_balanced: bool = False,
    class_targets: Tensor | None = None,
    class_valid_mask: Tensor | None = None,
    class_weight: float = 0.5,
    state_targets: Tensor | None = None,
    state_valid_mask: Tensor | None = None,
    state_weight: float = 0.1,
) -> dict[str, Tensor]:
    heatmap = sigmoid_focal_loss(
        outputs["rgb_logits"],
        heatmap_targets,
        valid_mask=valid_mask,
        alpha=alpha,
        gamma=gamma,
        density_balanced=density_balanced,
    )
    offset = heatmap.new_zeros(())
    if offset_targets is not None:
        support = heatmap_targets >= 0.5
        if valid_mask is not None:
            if valid_mask.ndim == support.ndim - 1:
                valid_mask = valid_mask.unsqueeze(-1)
            support = support & valid_mask.bool().expand_as(support)
        if support.any():
            offset = F.smooth_l1_loss(
                outputs["offsets"][support], offset_targets[support], reduction="mean"
            )
    class_heatmap = heatmap.new_zeros(())
    if class_targets is not None:
        if "class_logits" not in outputs:
            raise KeyError("class_targets were provided but model has no class_logits")
        class_heatmap = sigmoid_focal_loss(
            outputs["class_logits"], class_targets, valid_mask=class_valid_mask,
            alpha=alpha, gamma=gamma, density_balanced=density_balanced,
        )
    state_heatmap = heatmap.new_zeros(())
    if state_targets is not None:
        if "state_logits" not in outputs:
            raise KeyError("state_targets were provided but model has no state_logits")
        state_heatmap = sigmoid_focal_loss(
            outputs["state_logits"], state_targets, valid_mask=state_valid_mask,
            alpha=alpha, gamma=gamma, density_balanced=density_balanced,
        )
    total = (
        heatmap + float(offset_weight) * offset + float(class_weight) * class_heatmap
        + float(state_weight) * state_heatmap
    )
    return {
        "loss": total, "heatmap_loss": heatmap, "offset_loss": offset,
        "class_heatmap_loss": class_heatmap, "state_heatmap_loss": state_heatmap,
    }


def sequential_chunk_locator_loss(
    outputs: dict[str, Tensor],
    *,
    class_targets: Tensor,
    family_targets: Tensor,
    offset_targets: Tensor,
    class_valid_mask: Tensor,
    family_valid_mask: Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
    family_weight: float = 0.3,
    offset_weight: float = 0.2,
    hierarchy_weight: float = 0.05,
) -> dict[str, Tensor]:
    """Structured loss for the five-class sequential chunk locator."""
    class_heatmap = sigmoid_focal_loss(
        outputs["class_logits"],
        class_targets,
        valid_mask=class_valid_mask,
        alpha=alpha,
        gamma=gamma,
        density_balanced=True,
    )
    family_heatmap = sigmoid_focal_loss(
        outputs["family_logits"],
        family_targets,
        valid_mask=family_valid_mask,
        alpha=alpha,
        gamma=gamma,
        density_balanced=True,
    )
    offset = class_heatmap.new_zeros(())
    support = (class_targets >= 0.5) & class_valid_mask.bool()
    if support.any():
        offset = F.smooth_l1_loss(
            outputs["offsets"][support], offset_targets[support], reduction="mean"
        )

    class_probability = outputs["class_logits"].sigmoid()
    family_probability = outputs["family_logits"].sigmoid()
    parent_index = torch.tensor((0, 0, 1, 1, 1), device=class_probability.device)
    parent_probability = family_probability.index_select(-1, parent_index)
    hierarchy_error = F.relu(class_probability - parent_probability).square()
    hierarchy_weight_mask = class_valid_mask.to(hierarchy_error.dtype)
    hierarchy = (hierarchy_error * hierarchy_weight_mask).sum() / hierarchy_weight_mask.sum().clamp_min(1.0)

    total = (
        class_heatmap
        + float(family_weight) * family_heatmap
        + float(offset_weight) * offset
        + float(hierarchy_weight) * hierarchy
    )
    return {
        "loss": total,
        "class_heatmap_loss": class_heatmap,
        "family_heatmap_loss": family_heatmap,
        "offset_loss": offset,
        "hierarchy_loss": hierarchy,
    }
