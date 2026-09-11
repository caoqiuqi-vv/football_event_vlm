from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def density_balanced_focal(
    logits: Tensor,
    targets: Tensor,
    valid: Tensor,
    *,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> Tensor:
    """Normalize positive and background mass independently for every class."""
    targets = targets.to(logits.dtype).clamp(0.0, 1.0)
    valid_weight = valid.to(logits.dtype)
    probability = logits.sigmoid()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    positive = valid_weight * targets
    negative = valid_weight * (1.0 - targets)
    positive_loss = bce * (1.0 - probability).pow(gamma) * positive
    negative_loss = bce * probability.pow(gamma) * negative
    positive_term = positive_loss.sum(dim=(0, 1)) / positive.sum(dim=(0, 1)).clamp_min(1.0)
    negative_term = negative_loss.sum(dim=(0, 1)) / negative.sum(dim=(0, 1)).clamp_min(1.0)
    has_positive = positive.sum(dim=(0, 1)) > 0
    class_loss = torch.where(
        has_positive,
        float(alpha) * positive_term + (1.0 - float(alpha)) * negative_term,
        negative_term,
    )
    return class_loss.mean()


def soft_center_contrastive(
    embeddings: Tensor,
    targets: Tensor,
    valid: Tensor,
    *,
    temperature: float = 0.1,
) -> Tensor:
    """Compact positive modes and separate class centers without mining negatives."""
    centers = []
    compactness = embeddings.new_zeros(())
    active = 0
    for class_index in range(targets.shape[-1]):
        weights = targets[..., class_index] * valid[..., class_index].to(targets.dtype)
        selected = weights >= 0.5
        if int(selected.sum()) < 2:
            continue
        features = embeddings[selected]
        sample_weights = weights[selected]
        center = F.normalize(
            (features * sample_weights[:, None]).sum(dim=0)
            / sample_weights.sum().clamp_min(1e-9), dim=0,
        )
        compactness = compactness + (
            (1.0 - features @ center) * sample_weights
        ).sum() / sample_weights.sum().clamp_min(1e-9)
        centers.append(center)
        active += 1
    if active == 0:
        return embeddings.sum() * 0.0
    compactness = compactness / active
    if len(centers) < 2:
        return compactness
    centers_tensor = torch.stack(centers)
    similarity = centers_tensor @ centers_tensor.transpose(0, 1)
    off_diagonal = ~torch.eye(len(centers), dtype=torch.bool, device=similarity.device)
    separation = F.relu(
        (similarity[off_diagonal] - 0.2) / max(float(temperature), 1e-3)
    ).mean()
    return compactness + 0.05 * separation


def spotting_loss(
    outputs: dict[str, Tensor],
    *,
    class_targets: Tensor,
    family_targets: Tensor,
    offset_targets: Tensor,
    class_valid: Tensor,
    family_valid: Tensor,
    positive_alpha: float,
    focal_gamma: float,
    family_weight: float,
    offset_weight: float,
    contrastive_weight: float,
    contrastive_temperature: float,
    hierarchy_weight: float,
) -> dict[str, Tensor]:
    class_loss = density_balanced_focal(
        outputs["class_logits"], class_targets, class_valid,
        alpha=positive_alpha, gamma=focal_gamma,
    )
    family_loss = density_balanced_focal(
        outputs["family_logits"], family_targets, family_valid,
        alpha=positive_alpha, gamma=focal_gamma,
    )
    offset_mask = class_valid & (class_targets >= 0.5)
    if offset_mask.any():
        offset_loss = F.smooth_l1_loss(
            outputs["offsets"][offset_mask], offset_targets[offset_mask], reduction="mean"
        )
    else:
        offset_loss = outputs["offsets"].sum() * 0.0
    contrastive_loss = soft_center_contrastive(
        outputs["embeddings"], class_targets, class_valid,
        temperature=contrastive_temperature,
    )
    class_probability = outputs["class_logits"].sigmoid()
    family_probability = outputs["family_logits"].sigmoid()
    expected_family = torch.stack((
        class_probability[..., :2].amax(dim=-1),
        class_probability[..., 2:].amax(dim=-1),
    ), dim=-1)
    hierarchy_mask = family_valid.to(expected_family.dtype)
    hierarchy_loss = (
        F.relu(expected_family - family_probability - 0.15).square() * hierarchy_mask
    ).sum() / hierarchy_mask.sum().clamp_min(1.0)
    total = (
        class_loss
        + float(family_weight) * family_loss
        + float(offset_weight) * offset_loss
        + float(contrastive_weight) * contrastive_loss
        + float(hierarchy_weight) * hierarchy_loss
    )
    return {
        "loss": total,
        "class_loss": class_loss,
        "family_loss": family_loss,
        "offset_loss": offset_loss,
        "contrastive_loss": contrastive_loss,
        "hierarchy_loss": hierarchy_loss,
    }
