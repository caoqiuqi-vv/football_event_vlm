"""Paired-window consistency losses for football online-simulation training.

The loss is deliberately asymmetric: a strongly supervised central window is a
stop-gradient teacher for an edge window containing the same annotated event.
Rows are paired only through explicit event-level metadata, never merely because
they came from the same long chunk (which may contain multiple events).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def consistency_warmup_factor(
    cfg: Any, *, epoch: int, step: int, steps_per_epoch: int
) -> float:
    """Return a deterministic [0, 1] ramp for the auxiliary objective."""
    start_epoch = max(int(_cfg_get(cfg, "start_epoch", 1)), 1)
    if epoch < start_epoch:
        return 0.0
    warmup_steps = int(_cfg_get(cfg, "warmup_steps", 0) or 0)
    if warmup_steps <= 0:
        warmup_epochs = max(float(_cfg_get(cfg, "warmup_epochs", 0.0) or 0.0), 0.0)
        warmup_steps = int(round(warmup_epochs * max(int(steps_per_epoch), 1)))
    if warmup_steps <= 0:
        return 1.0
    active_step = (
        (int(epoch) - start_epoch) * max(int(steps_per_epoch), 1)
        + max(int(step) - 1, 0)
    )
    return min(max(float(active_step + 1) / float(warmup_steps), 0.0), 1.0)


def _pair_class_mask(meta: Mapping[str, Any], num_labels: int, device: torch.device) -> Tensor:
    raw = tuple(meta.get("online_pair_class_mask", ()) or ())
    if not raw:
        return torch.ones(num_labels, dtype=torch.bool, device=device)
    if len(raw) != num_labels:
        raise ValueError(
            "online_pair_class_mask must contain one value per class: "
            f"expected={num_labels}, got={len(raw)}"
        )
    return torch.as_tensor(raw, dtype=torch.bool, device=device)


def _absolute_time_interpolate(
    source_times: Tensor, source_values: Tensor, query_times: Tensor
) -> Tensor:
    """Linearly interpolate ``[T,C]`` values at absolute query times."""
    if source_times.ndim != 1 or query_times.ndim != 1 or source_values.ndim != 2:
        raise ValueError("absolute-time interpolation expects [T], [T,C], [Q]")
    if source_times.numel() != source_values.shape[0]:
        raise ValueError("source time/value lengths differ")
    if source_times.numel() < 2:
        return source_values[:1].expand(query_times.numel(), -1)
    # Dataset frame times are monotonic. Fail loudly rather than silently align
    # a corrupt/decreasing decode timeline.
    if not bool(torch.all(source_times[1:] >= source_times[:-1])):
        raise ValueError("frame_times must be monotonically non-decreasing")
    right = torch.searchsorted(source_times.contiguous(), query_times.contiguous())
    right = right.clamp(1, source_times.numel() - 1)
    left = right - 1
    left_t = source_times[left]
    right_t = source_times[right]
    alpha = ((query_times - left_t) / (right_t - left_t).clamp_min(1e-6)).unsqueeze(-1)
    return source_values[left] + alpha * (source_values[right] - source_values[left])


def online_pair_consistency_loss(
    outputs: Mapping[str, Tensor],
    batch: Mapping[str, Any],
    cfg: Any,
    *,
    epoch: int,
    step: int,
    steps_per_epoch: int,
) -> tuple[Tensor, dict[str, float]]:
    """Compute central-teacher -> edge-student clip and curve consistency.

    A low constant solution is not an optimum of the combined training target:
    central rows retain their ordinary positive BCE/frame supervision, the
    central target is detached, and the teacher target can be lower-bounded by
    ``teacher_logit_floor`` (default 0).  Only positive, trusted class slots are
    paired.
    """
    logits = outputs["logits"]
    zero = logits.new_zeros(())
    enabled = bool(_cfg_get(cfg, "enabled", False))
    clip_weight = float(_cfg_get(cfg, "clip_weight", 0.0) or 0.0)
    curve_weight = float(_cfg_get(cfg, "response_weight", 0.0) or 0.0)
    warmup = consistency_warmup_factor(
        cfg, epoch=epoch, step=step, steps_per_epoch=steps_per_epoch
    )
    metrics = {
        "online_pair_consistency_loss": 0.0,
        "online_pair_clip_consistency_loss": 0.0,
        "online_pair_response_consistency_loss": 0.0,
        "online_pair_consistency_pairs": 0.0,
        "online_pair_consistency_slots": 0.0,
        "online_pair_response_points": 0.0,
        "online_pair_consistency_warmup": float(warmup),
        "online_pair_teacher_logit": 0.0,
        "online_pair_edge_logit": 0.0,
        "online_pair_clip_abs_gap": 0.0,
        "online_pair_teacher_floor_fraction": 0.0,
    }
    if not enabled or warmup <= 0 or (clip_weight <= 0 and curve_weight <= 0):
        return zero, metrics

    metas: Sequence[Mapping[str, Any]] = batch.get("meta", ())
    targets = batch.get("targets")
    label_masks = batch.get("label_masks")
    if not torch.is_tensor(targets) or not torch.is_tensor(label_masks):
        raise ValueError("online pair consistency requires targets and label_masks")
    targets = targets.to(device=logits.device)
    label_masks = label_masks.to(device=logits.device)
    if len(metas) != logits.shape[0]:
        raise ValueError("online pair metadata batch size differs from logits")

    groups: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: {"central": [], "edge": []}
    )
    metadata_rows = 0
    strict = bool(_cfg_get(cfg, "strict_metadata", True))
    for row_index, meta in enumerate(metas):
        pair_id = str(meta.get("online_pair_id", "") or "")
        role = str(meta.get("online_pair_role", "") or "").strip().lower()
        if strict and bool(pair_id) != bool(role in ("central", "edge")):
            raise ValueError(
                "online pair rows must provide both online_pair_id and a valid "
                "online_pair_role; unrelated background rows may leave both empty"
            )
        if pair_id and role in ("central", "edge"):
            groups[pair_id][role].append(row_index)
            metadata_rows += 1
    if (
        strict
        and bool(_cfg_get(cfg, "require_pair_each_batch", False))
        and metadata_rows == 0
    ):
        raise ValueError(
            "online_pair_consistency requires at least one explicit central/edge "
            "pair in every training batch"
        )

    clip_losses: list[Tensor] = []
    teacher_values: list[Tensor] = []
    edge_values: list[Tensor] = []
    floor_hits = 0
    clip_slots = 0
    pair_count = 0
    teacher_floor = float(_cfg_get(cfg, "teacher_logit_floor", 0.0))
    beta = max(float(_cfg_get(cfg, "smooth_l1_beta", 0.5) or 0.5), 1e-6)

    curve = outputs.get("response_curve_logits")
    if curve is None and bool(_cfg_get(cfg, "allow_frame_logits_fallback", True)):
        curve = outputs.get("frame_event_logits")
    frame_times = batch.get("frame_times")
    if (
        curve_weight > 0
        and bool(_cfg_get(cfg, "require_response_curve", True))
        and (not torch.is_tensor(curve) or not torch.is_tensor(frame_times))
    ):
        raise ValueError(
            "response consistency requires response_curve_logits and absolute frame_times"
        )
    response_losses: list[Tensor] = []
    response_points = 0
    support_radius = max(float(_cfg_get(cfg, "response_support_radius_sec", 2.0)), 0.0)

    for rows in groups.values():
        if not rows["central"] or not rows["edge"]:
            continue
        pair_count += 1
        for edge_index in rows["edge"]:
            edge_meta = metas[edge_index]
            edge_class_mask = _pair_class_mask(edge_meta, logits.shape[1], logits.device)
            for label_index in range(logits.shape[1]):
                if not bool(edge_class_mask[label_index]):
                    continue
                valid_central = [
                    index
                    for index in rows["central"]
                    if targets[index, label_index] > 0.5
                    and label_masks[index, label_index] > 0
                    and bool(_pair_class_mask(metas[index], logits.shape[1], logits.device)[label_index])
                ]
                if (
                    not valid_central
                    or targets[edge_index, label_index] <= 0.5
                    or label_masks[edge_index, label_index] <= 0
                ):
                    continue
                # The strongest central view is the most reliable teacher for
                # this event class; selection is detached and carries no gradient.
                central_index = max(
                    valid_central,
                    key=lambda index: float(logits[index, label_index].detach()),
                )
                raw_teacher = logits[central_index, label_index].detach()
                teacher = raw_teacher.clamp_min(teacher_floor)
                edge = logits[edge_index, label_index]
                clip_losses.append(F.smooth_l1_loss(edge, teacher, beta=beta))
                teacher_values.append(raw_teacher)
                edge_values.append(edge.detach())
                floor_hits += int(float(raw_teacher) < teacher_floor)
                clip_slots += 1

                if (
                    curve_weight <= 0
                    or not torch.is_tensor(curve)
                    or not torch.is_tensor(frame_times)
                ):
                    continue
                central_times = frame_times[central_index].to(curve.device).float()
                edge_times = frame_times[edge_index].to(curve.device).float()
                overlap = (edge_times >= central_times.min()) & (
                    edge_times <= central_times.max()
                )
                anchor = float(
                    edge_meta.get(
                        "online_pair_anchor_time",
                        edge_meta.get("anchor_time", -1.0),
                    )
                )
                if support_radius > 0 and anchor >= 0:
                    overlap = overlap & ((edge_times - anchor).abs() <= support_radius)
                if not bool(overlap.any()):
                    continue
                query = edge_times[overlap]
                teacher_curve = _absolute_time_interpolate(
                    central_times,
                    curve[central_index].detach().float(),
                    query,
                )[:, label_index]
                student_curve = curve[edge_index, overlap, label_index].float()
                response_losses.append(
                    F.smooth_l1_loss(student_curve, teacher_curve, beta=beta)
                )
                response_points += int(query.numel())

    if strict and metadata_rows and pair_count == 0:
        raise ValueError(
            "online_pair_consistency received pair metadata but no central/edge "
            "pair was colocated in this batch"
        )
    clip_loss = torch.stack(clip_losses).mean() if clip_losses else zero
    response_loss = torch.stack(response_losses).mean() if response_losses else zero
    total = float(warmup) * (
        clip_weight * clip_loss + curve_weight * response_loss
    )
    metrics.update(
        {
            "online_pair_consistency_loss": float(total.detach().cpu()),
            "online_pair_clip_consistency_loss": float(clip_loss.detach().cpu()),
            "online_pair_response_consistency_loss": float(response_loss.detach().cpu()),
            "online_pair_consistency_pairs": float(pair_count),
            "online_pair_consistency_slots": float(clip_slots),
            "online_pair_response_points": float(response_points),
            "online_pair_teacher_floor_fraction": float(floor_hits / max(clip_slots, 1)),
        }
    )
    if teacher_values:
        teacher_tensor = torch.stack(teacher_values).float()
        edge_tensor = torch.stack(edge_values).float()
        metrics["online_pair_teacher_logit"] = float(teacher_tensor.mean().cpu())
        metrics["online_pair_edge_logit"] = float(edge_tensor.mean().cpu())
        metrics["online_pair_clip_abs_gap"] = float(
            (teacher_tensor - edge_tensor).abs().mean().cpu()
        )
    return total, metrics


__all__ = [
    "consistency_warmup_factor",
    "online_pair_consistency_loss",
]
