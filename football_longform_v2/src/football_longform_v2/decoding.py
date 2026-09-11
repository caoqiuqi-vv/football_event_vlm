from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class Proposal:
    batch_index: int
    family: str
    timestamp: float
    score: float
    timeline_index: int


def decode_proposals(
    logits: Tensor,
    timestamps: Tensor,
    families: tuple[str, ...],
    *,
    threshold: float,
    nms_radius_seconds: dict[str, float],
    max_per_minute: dict[str, float],
) -> list[list[Proposal]]:
    if logits.ndim != 3 or timestamps.shape != logits.shape[:2]:
        raise ValueError("expected logits [batch,time,family] and timestamps [batch,time]")
    if logits.shape[-1] != len(families):
        raise ValueError("family count mismatch")
    probability = logits.sigmoid()
    results: list[list[Proposal]] = []
    for batch_index in range(logits.shape[0]):
        batch_results: list[Proposal] = []
        times = timestamps[batch_index]
        if times.numel() <= 1:
            step_seconds = 1.0
        else:
            step_seconds = float(torch.median(times[1:] - times[:-1]).clamp_min(1e-6))
        duration_minutes = max(float(times[-1] - times[0]) / 60.0, step_seconds / 60.0)
        for family_index, family in enumerate(families):
            scores = probability[batch_index, :, family_index]
            radius_steps = max(1, int(round(nms_radius_seconds[family] / step_seconds)))
            kernel = 2 * radius_steps + 1
            pooled = F.max_pool1d(
                scores.reshape(1, 1, -1), kernel, stride=1, padding=radius_steps
            ).reshape(-1)
            candidate_indices = torch.nonzero(
                (scores >= float(threshold)) & (scores >= pooled), as_tuple=False
            ).flatten()
            limit = max(1, int(math.ceil(duration_minutes * max_per_minute[family])))
            if candidate_indices.numel() > limit:
                order = torch.topk(scores[candidate_indices], k=limit).indices
                candidate_indices = candidate_indices[order]
            for index in candidate_indices.tolist():
                batch_results.append(
                    Proposal(
                        batch_index=batch_index,
                        family=family,
                        timestamp=float(times[index]),
                        score=float(scores[index]),
                        timeline_index=int(index),
                    )
                )
        results.append(sorted(batch_results, key=lambda item: item.timestamp))
    return results



def decode_family_conditioned_classes(
    family_logits: Tensor,
    class_logits: Tensor,
    timestamps: Tensor,
    families: tuple[str, ...],
    labels: tuple[str, ...],
    *,
    label_to_family: dict[str, str],
    family_nms_radius_seconds: dict[str, float],
    class_nms_radius_seconds: dict[str, float],
    local_search_radius_seconds: dict[str, float],
    max_class_per_minute: dict[str, float],
) -> dict[str, list[Proposal]]:
    """Classify high-recall family peaks while allowing class-specific local timestamps."""
    if family_logits.ndim != 2 or class_logits.ndim != 2 or timestamps.ndim != 1:
        raise ValueError("expected family and class logits [time,channel] plus timestamps [time]")
    if timestamps.numel() == 0:
        raise ValueError("cannot decode an empty timeline")
    if family_logits.shape[0] != timestamps.numel() or class_logits.shape[0] != timestamps.numel():
        raise ValueError("family, class and timestamp lengths disagree")
    family_candidates = decode_proposals(
        family_logits.unsqueeze(0), timestamps.unsqueeze(0), families, threshold=0.0,
        nms_radius_seconds=family_nms_radius_seconds,
        max_per_minute={family: 1000.0 for family in families},
    )[0]
    class_probability = class_logits.sigmoid()
    duration_minutes = max(
        float(timestamps[-1] - timestamps[0]) / 60.0 if timestamps.numel() > 1 else 0.0,
        1.0 / 60.0,
    )
    result: dict[str, list[Proposal]] = {}
    for label_index, label in enumerate(labels):
        parent = label_to_family[label]
        raw: list[Proposal] = []
        radius = float(local_search_radius_seconds[label])
        for parent_proposal in family_candidates:
            if parent_proposal.family != parent:
                continue
            left = int(torch.searchsorted(
                timestamps, timestamps.new_tensor(parent_proposal.timestamp - radius), right=False
            ))
            right = int(torch.searchsorted(
                timestamps, timestamps.new_tensor(parent_proposal.timestamp + radius), right=True
            ))
            left = max(left, 0)
            right = min(max(right, left + 1), timestamps.numel())
            local_scores = class_probability[left:right, label_index]
            local_index = left + int(local_scores.argmax())
            class_score = float(class_probability[local_index, label_index])
            combined_score = math.sqrt(max(parent_proposal.score * class_score, 0.0))
            raw.append(Proposal(
                batch_index=0, family=label, timestamp=float(timestamps[local_index]),
                score=combined_score, timeline_index=local_index,
            ))
        kept: list[Proposal] = []
        nms_radius = float(class_nms_radius_seconds[label])
        for candidate in sorted(raw, key=lambda item: item.score, reverse=True):
            if all(abs(candidate.timestamp - other.timestamp) > nms_radius for other in kept):
                kept.append(candidate)
        limit = max(1, int(math.ceil(duration_minutes * max_class_per_minute[label])))
        result[label] = sorted(kept[:limit], key=lambda item: item.timestamp)
    return result
