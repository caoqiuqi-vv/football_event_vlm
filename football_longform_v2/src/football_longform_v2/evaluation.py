from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import torch
from torch import Tensor

from .annotations import Event, build_family_targets, build_label_targets, family_event_times
from .decoding import Proposal, decode_proposals
from .feature_store import AlignedTimeline


@dataclass(frozen=True)
class ContinuousOutput:
    logits: Tensor
    offsets: Tensor
    class_logits: Tensor | None
    coverage: Tensor


@torch.inference_mode()
def infer_continuous_timeline(
    model: torch.nn.Module,
    timeline: AlignedTimeline,
    *,
    device: torch.device,
    core_steps: int,
    context_steps: int,
) -> ContinuousOutput:
    """Infer a long timeline in overlapped blocks while emitting every point once."""
    if core_steps <= 0 or context_steps < 0:
        raise ValueError("core_steps must be positive and context_steps non-negative")
    length = timeline.timestamps.numel()
    if length == 0:
        raise ValueError("cannot infer an empty timeline")
    logits_parts: list[Tensor] = []
    offset_parts: list[Tensor] = []
    class_logits_parts: list[Tensor] = []
    coverage = torch.zeros(length, dtype=torch.int32)
    model.eval()
    for core_start in range(0, length, core_steps):
        core_stop = min(core_start + core_steps, length)
        input_start = max(0, core_start - context_steps)
        input_stop = min(length, core_stop + context_steps)
        outputs = model(
            timeline.context[input_start:input_stop].unsqueeze(0).to(device, non_blocking=True),
            timeline.motion[input_start:input_stop].unsqueeze(0).to(device, non_blocking=True),
            timeline.timestamps[input_start:input_stop].unsqueeze(0).to(device, non_blocking=True),
            context_valid=timeline.context_valid[input_start:input_stop].unsqueeze(0).to(device, non_blocking=True),
            motion_valid=timeline.motion_valid[input_start:input_stop].unsqueeze(0).to(device, non_blocking=True),
        )
        local_start = core_start - input_start
        local_stop = local_start + (core_stop - core_start)
        logits_parts.append(outputs["rgb_logits"][0, local_start:local_stop].float().cpu())
        offset_parts.append(outputs["offsets"][0, local_start:local_stop].float().cpu())
        if "class_logits" in outputs:
            class_logits_parts.append(
                outputs["class_logits"][0, local_start:local_stop].float().cpu()
            )
        coverage[core_start:core_stop] += 1
    if not torch.equal(coverage, torch.ones_like(coverage)):
        raise RuntimeError("continuous inference did not emit each timeline point exactly once")
    return ContinuousOutput(
        logits=torch.cat(logits_parts, dim=0),
        offsets=torch.cat(offset_parts, dim=0),
        class_logits=torch.cat(class_logits_parts, dim=0) if class_logits_parts else None,
        coverage=coverage,
    )


def average_precision(
    scores: Tensor, targets: Tensor, *, positive_count: int | None = None
) -> float | None:
    """Binary average precision without a sklearn dependency; None means no positives."""
    scores = scores.detach().flatten().float().cpu()
    targets = targets.detach().flatten().bool().cpu()
    denominator = int(targets.sum()) if positive_count is None else int(positive_count)
    if denominator == 0:
        return None
    order = torch.argsort(scores, descending=True, stable=True)
    ranked = targets[order].to(torch.float32)
    precision = ranked.cumsum(0) / torch.arange(1, ranked.numel() + 1, dtype=torch.float32)
    return float((precision * ranked).sum() / denominator)


def operating_point_at_recall(
    scores: Iterable[float], labels: Iterable[bool], *, positive_count: int,
    target_recall: float, total_minutes: float | None = None,
) -> dict | None:
    """Highest score threshold reaching a recall target under fixed 1:1 matches."""
    if positive_count <= 0:
        return None
    score_tensor = torch.tensor(list(scores), dtype=torch.float32)
    label_tensor = torch.tensor(list(labels), dtype=torch.bool)
    if score_tensor.numel() != label_tensor.numel():
        raise ValueError("scores and labels must have equal lengths")
    if score_tensor.numel() == 0:
        return {
            "target_recall": float(target_recall), "target_achieved": False,
            "threshold": None, "recall": 0.0, "precision": None,
            "tp": 0, "fp": 0, "proposals": 0, "fp_per_minute": 0.0,
        }
    order = torch.argsort(score_tensor, descending=True, stable=True)
    sorted_scores = score_tensor[order]
    sorted_labels = label_tensor[order]
    cumulative_tp = sorted_labels.to(torch.int64).cumsum(0)
    required_tp = int(math.ceil(float(target_recall) * positive_count - 1e-12))
    reaching = torch.nonzero(cumulative_tp >= required_tp, as_tuple=False).flatten()
    cutoff = int(reaching[0]) if reaching.numel() else score_tensor.numel() - 1
    threshold = float(sorted_scores[cutoff])
    selected = score_tensor >= threshold
    tp = int(label_tensor[selected].sum())
    proposals = int(selected.sum())
    fp = proposals - tp
    recall = tp / positive_count
    return {
        "target_recall": float(target_recall),
        "target_achieved": recall >= float(target_recall),
        "threshold": threshold,
        "recall": recall,
        "precision": tp / proposals if proposals else None,
        "tp": tp, "fp": fp, "proposals": proposals,
        "fp_per_minute": (fp / total_minutes) if total_minutes else None,
    }


def events_for_label(events: Iterable[Event], label: str) -> list[float]:
    return [event.timestamp for event in events if event.label == label]


def events_for_family(events: Iterable[Event], family: str) -> list[float]:
    return family_event_times(tuple(events), family)


def scored_point_matches(
    proposals: Iterable[Proposal], targets: Iterable[float], *, tolerance_seconds: float
) -> tuple[list[float], list[bool], int]:
    """Score-sorted, one-to-one event matching for point AP and recall."""
    unmatched = list(targets)
    scores: list[float] = []
    labels: list[bool] = []
    for proposal in sorted(proposals, key=lambda item: item.score, reverse=True):
        scores.append(proposal.score)
        if not unmatched:
            labels.append(False)
            continue
        distances = [abs(proposal.timestamp - target) for target in unmatched]
        best_index = min(range(len(unmatched)), key=lambda index: distances[index])
        if distances[best_index] <= tolerance_seconds:
            labels.append(True)
            unmatched.pop(best_index)
        else:
            labels.append(False)
    return scores, labels, len(unmatched)


def family_proposals(
    logits: Tensor,
    timestamps: Tensor,
    families: tuple[str, ...],
    *,
    threshold: float,
    nms_radius_seconds: dict[str, float],
    max_per_minute: dict[str, float],
) -> dict[str, list[Proposal]]:
    decoded = decode_proposals(
        logits.unsqueeze(0), timestamps.unsqueeze(0), families,
        threshold=threshold, nms_radius_seconds=nms_radius_seconds,
        max_per_minute=max_per_minute,
    )[0]
    return {family: [proposal for proposal in decoded if proposal.family == family] for family in families}


def dense_label_targets(
    timestamps: Tensor, events: tuple[Event, ...], labels: tuple[str, ...]
) -> Tensor:
    return build_label_targets(timestamps, events, labels)


def dense_family_targets(
    timestamps: Tensor, events: tuple[Event, ...], families: tuple[str, ...]
) -> Tensor:
    targets, _ = build_family_targets(timestamps, events, families)
    return targets
