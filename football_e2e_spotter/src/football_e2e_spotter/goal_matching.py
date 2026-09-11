"""Hungarian supervision for independent no-NMS retriever slots."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor
import torch.nn.functional as F

from .goal_annotations import FAMILY_MAP, LABELS


LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
FAMILY_TO_INDEX = {"shot_chain": 0, "restart": 1}


def hungarian_indices(
    class_logits: Tensor,
    event_times: Tensor,
    uncertainty: Tensor,
    target_labels: Tensor,
    target_times: Tensor,
) -> tuple[Tensor, Tensor]:
    """Match one GT to at most one slot; never merge neighboring instances."""
    if target_labels.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=class_logits.device)
        return empty, empty
    probability = class_logits.softmax(dim=-1)
    class_cost = -probability[:, target_labels]
    scale = uncertainty[:, None].clamp_min(0.05)
    time_error = (event_times[:, None] - target_times[None, :]).abs()
    time_cost = time_error / scale + scale.log()
    # A modest anchor-distance term prevents far-away slots stealing a GT from
    # a locally plausible slot early in training.
    cost = class_cost + 0.35 * time_cost + 0.02 * time_error
    rows, columns = linear_sum_assignment(cost.detach().float().cpu().numpy())
    return (
        torch.as_tensor(rows, dtype=torch.long, device=class_logits.device),
        torch.as_tensor(columns, dtype=torch.long, device=class_logits.device),
    )


def focal_cross_entropy(logits: Tensor, targets: Tensor, *, gamma: float = 2.0) -> Tensor:
    log_probability = F.log_softmax(logits, dim=-1)
    log_pt = log_probability.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    pt = log_pt.exp()
    return (-(1.0 - pt).pow(gamma) * log_pt).mean()


def retriever_loss(outputs: dict[str, Tensor], targets: Sequence[dict[str, Tensor]]) -> tuple[Tensor, dict[str, float]]:
    class_logits = outputs["class_logits"]
    batch_size, slots, class_count = class_logits.shape
    no_event = class_count - 1
    class_targets = torch.full((batch_size, slots), no_event, dtype=torch.long, device=class_logits.device)
    quality_targets = torch.zeros((batch_size, slots), dtype=class_logits.dtype, device=class_logits.device)
    matched_slot: list[Tensor] = []
    matched_target: list[Tensor] = []
    matched_batch: list[Tensor] = []
    for batch_index, target in enumerate(targets):
        labels = target["labels"].to(class_logits.device)
        times = target["times"].to(class_logits.device)
        rows, columns = hungarian_indices(
            class_logits[batch_index], outputs["event_time"][batch_index],
            outputs["temporal_uncertainty"][batch_index], labels, times,
        )
        if rows.numel():
            class_targets[batch_index, rows] = labels[columns]
            quality_targets[batch_index, rows] = 1.0
            matched_slot.append(rows)
            matched_target.append(columns)
            matched_batch.append(torch.full_like(rows, batch_index))

    classification = focal_cross_entropy(class_logits.reshape(-1, class_count), class_targets.reshape(-1))
    quality = F.binary_cross_entropy_with_logits(outputs["quality_logit"], quality_targets)
    if matched_slot:
        slot_indices = torch.cat(matched_slot)
        target_indices = torch.cat(matched_target)
        batch_indices = torch.cat(matched_batch)
        predicted_time = outputs["event_time"][batch_indices, slot_indices]
        sigma = outputs["temporal_uncertainty"][batch_indices, slot_indices].clamp(0.05, 10.0)
        true_time = torch.cat([targets[int(batch)]["times"][target].reshape(1) for batch, target in zip(batch_indices.cpu(), target_indices.cpu())]).to(predicted_time.device)
        error = F.smooth_l1_loss(predicted_time, true_time, reduction="none", beta=1.0)
        temporal = (error / sigma + sigma.log()).mean()
        matched_labels = class_targets[batch_indices, slot_indices]
        family_targets = torch.as_tensor([
            FAMILY_TO_INDEX[FAMILY_MAP[LABELS[int(label)]]] for label in matched_labels.cpu()
        ], device=class_logits.device)
        family = F.cross_entropy(outputs["family_logits"][batch_indices, slot_indices], family_targets)
        positive_score = outputs["quality_logit"][batch_indices, slot_indices]
        negative_score = outputs["quality_logit"].masked_fill(quality_targets.bool(), -1e4).amax(dim=1)
        ranking = F.softplus(0.5 + negative_score[batch_indices] - positive_score).mean()
    else:
        temporal = class_logits.sum() * 0.0
        family = temporal
        ranking = temporal
    total = classification + 0.65 * temporal + 0.35 * quality + 0.15 * family + 0.15 * ranking
    stats = {
        "loss": float(total.detach()), "classification": float(classification.detach()),
        "temporal": float(temporal.detach()), "quality": float(quality.detach()),
        "family": float(family.detach()), "ranking": float(ranking.detach()),
        "matched": float(sum(value.numel() for value in matched_slot)),
    }
    return total, stats

