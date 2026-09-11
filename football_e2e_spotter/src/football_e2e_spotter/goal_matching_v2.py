"""Recall-preserving v2 loss for the goal retriever."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor
import torch.nn.functional as F

from .goal_annotations import FAMILY_MAP, LABELS
from .goal_matching import hungarian_indices


FAMILY_TO_INDEX = {"shot_chain": 0, "restart": 1}


def focal_cross_entropy(logits: Tensor, targets: Tensor, gamma: float = 2.0, no_event_weight: float = 0.15) -> Tensor:
    log_probability = F.log_softmax(logits, dim=-1)
    log_pt = log_probability.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    pt = log_pt.exp()
    loss = -(1.0 - pt).pow(gamma) * log_pt
    weights = torch.where(targets == logits.shape[-1] - 1, no_event_weight, 1.0)
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def retriever_loss(outputs: dict[str, Tensor], targets: Sequence[dict[str, Tensor]]) -> tuple[Tensor, dict[str, float]]:
    logits = outputs["class_logits"]
    batch_size, slots, class_count = logits.shape
    no_event = class_count - 1
    class_targets = torch.full((batch_size, slots), no_event, dtype=torch.long, device=logits.device)
    quality_targets = torch.zeros((batch_size, slots), dtype=logits.dtype, device=logits.device)
    matches: list[tuple[int, Tensor, Tensor]] = []
    for batch_index, target in enumerate(targets):
        labels = target["labels"].to(logits.device)
        times = target["times"].to(logits.device)
        rows, columns = hungarian_indices(
            logits[batch_index], outputs["event_time"][batch_index],
            outputs["temporal_uncertainty"][batch_index], labels, times,
        )
        if rows.numel():
            class_targets[batch_index, rows] = labels[columns]
            quality_targets[batch_index, rows] = 1.0
            matches.append((batch_index, rows, columns))
    classification = focal_cross_entropy(logits.reshape(-1, class_count), class_targets.reshape(-1))
    quality = F.binary_cross_entropy_with_logits(outputs["quality_logit"], quality_targets)
    temporal_terms, family_logits, family_targets, positive_scores, negative_scores = [], [], [], [], []
    for batch_index, rows, columns in matches:
        predicted_time = outputs["event_time"][batch_index, rows]
        true_time = targets[batch_index]["times"].to(logits.device)[columns]
        sigma = outputs["temporal_uncertainty"][batch_index, rows].clamp(0.05, 10.0)
        error = F.smooth_l1_loss(predicted_time, true_time, reduction="none", beta=1.0)
        temporal_terms.append(error / sigma + sigma.log())
        labels = class_targets[batch_index, rows]
        family_logits.append(outputs["family_logits"][batch_index, rows])
        family_targets.extend(FAMILY_TO_INDEX[FAMILY_MAP[LABELS[int(label)]]] for label in labels.detach().cpu())
        positive_scores.append(outputs["quality_logit"][batch_index, rows])
        negative = outputs["quality_logit"][batch_index].masked_fill(quality_targets[batch_index].bool(), -1e4).amax()
        negative_scores.append(negative.expand_as(rows))
    if temporal_terms:
        temporal = torch.cat(temporal_terms).mean()
        family = F.cross_entropy(torch.cat(family_logits), torch.as_tensor(family_targets, device=logits.device))
        ranking = F.softplus(0.5 + torch.cat(negative_scores) - torch.cat(positive_scores)).mean()
    else:
        temporal = logits.sum() * 0.0; family = temporal; ranking = temporal
    # A fixed 12 s review contract prevents an unconstrained duration head from
    # gaming the workload metric and gives DDP a gradient for every head.
    duration = F.smooth_l1_loss(outputs["region_duration"], torch.full_like(outputs["region_duration"], 12.0), beta=1.0)
    total = classification + 0.65 * temporal + 0.35 * quality + 0.15 * family + 0.15 * ranking + 0.03 * duration
    stats = {
        "loss": float(total.detach()), "classification": float(classification.detach()),
        "temporal": float(temporal.detach()), "quality": float(quality.detach()),
        "family": float(family.detach()), "ranking": float(ranking.detach()),
        "duration": float(duration.detach()), "matched": float(sum(rows.numel() for _batch, rows, _columns in matches)),
    }
    return total, stats

