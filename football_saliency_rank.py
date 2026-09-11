"""Bag-level saliency ranking + cross-window consistency (E1.3.2).

Replaces the frame-level null-mass binary supervision, whose semantics do not
hold for football: "not an event frame" does not mean "no local evidence"
(passes, pressing, keeper setup, crowd staging are meaningful context), and
the manual event time itself carries annotation noise.  Instead:

  1. bag_saliency_rank_loss - event windows must carry stronger Top-K patch
     evidence than paired confirmed clean-negative windows.  MIL over the
     whole event support span (shoulders included), never locking the manual
     anchor.  Per class, independently.
  2. cross_window_saliency_consistency_loss - the same absolute video moment
     seen in two overlapping windows must produce the same class saliency.
     This supervision does not depend on the manual anchor's precision.

The monotone no-evidence gate stays in the model as a diagnostic only; it
receives no gradient from these losses (saliency is its input, not its
output), so the main task can no longer shortcut through it.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor

_NEG_INF = float("-inf")


def _get(cfg: Any, key: str, default: Any) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _masked_topk_mean(values: Tensor, span: Tensor, k: int) -> Tensor:
    """[B, T] values, [B, T] bool span -> [B] mean of strongest k in-span values.

    Rows with fewer than k in-span frames fall back to the full in-span mean
    (the pooled value is still the strongest-evidence summary available).
    """
    k = max(int(k), 1)
    masked = values.masked_fill(~span, _NEG_INF)
    count = span.sum(dim=1)
    enough = count >= k
    pooled = masked.topk(k, dim=1).values.mean(dim=1)
    fallback = values.masked_fill(~span, 0.0).sum(dim=1) / count.clamp_min(1.0)
    return torch.where(enough, pooled, fallback)


def bag_saliency_rank_loss(
    saliency: Tensor,             # [B, T, C] per-frame per-class log1p evidence strength
    targets: Tensor,              # [B, T, C] frame targets
    masks: Tensor,                # [B, T, C] frame masks
    clean_negative_rows: Tensor,  # [B] bool: confirmed clean-negative rows
    *,
    labels: tuple[str, ...] | list[str],
    margin: float = 0.10,
    topk: int = 3,
    support_threshold: float = 0.1,
    pairs_per_positive: int = 2,
) -> tuple[Tensor, dict[str, float]]:
    """Rank event-window Top-K saliency above clean-negative Top-K + margin.

    Event side: every frame with target >= support_threshold belongs to the
    event support span (shoulders included, anchor not locked); pool the
    strongest ``topk`` frames.  Negative side: clean-negative rows pool their
    strongest ``topk`` frames.  Hinge = softplus(margin + neg - pos), with
    ``pairs_per_positive`` sampled clean negatives per positive window, per
    class independently (no cross-class mixing).
    """
    if saliency.ndim != 3:
        raise ValueError(f"saliency must be [B,T,C], got {tuple(saliency.shape)}")
    if not torch.is_tensor(targets) or not torch.is_tensor(masks):
        raise ValueError("bag saliency rank requires frame targets and masks")
    if targets.shape != saliency.shape or masks.shape != saliency.shape:
        raise ValueError(
            "frame targets/masks must match saliency [B,T,C]: "
            f"saliency={tuple(saliency.shape)} targets={tuple(targets.shape)} "
            f"masks={tuple(masks.shape)}"
        )
    device = saliency.device
    targets_f = targets.to(device=device, dtype=saliency.dtype)
    masks_f = masks.to(device=device, dtype=saliency.dtype)
    clean = clean_negative_rows.to(device=device, dtype=torch.bool)
    if clean.ndim == 1:
        if clean.shape[0] != saliency.shape[0]:
            raise ValueError("row-level clean mask must be [B]")
    elif clean.ndim == 2:
        if tuple(clean.shape) != (saliency.shape[0], saliency.shape[-1]):
            raise ValueError(
                "per-class clean mask must be [B,C]: "
                f"mask={tuple(clean.shape)} saliency={tuple(saliency.shape)}"
            )
    else:
        raise ValueError("clean-negative mask must be [B] or [B,C]")
    losses: list[Tensor] = []
    pos_pooled: list[Tensor] = []
    neg_pooled: list[Tensor] = []
    gaps: list[Tensor] = []
    violations: list[Tensor] = []
    per_class_gaps: list[Tensor | None] = [None] * saliency.shape[-1]
    pairs = 0
    active_classes = 0
    pair_count = max(int(pairs_per_positive), 1)
    for class_index in range(saliency.shape[-1]):
        class_saliency = saliency[:, :, class_index]
        class_targets = targets_f[:, :, class_index]
        class_masks = masks_f[:, :, class_index]
        support = (class_masks > 0) & (class_targets >= support_threshold)
        background = (class_masks > 0) & (class_targets <= 0.5)
        class_clean = clean if clean.ndim == 1 else clean[:, class_index]
        pos_has = support.any(dim=1)
        neg_has = class_clean & background.any(dim=1)
        if not bool(pos_has.any()) or not bool(neg_has.any()):
            continue
        pos_scores = _masked_topk_mean(class_saliency, support, topk)
        neg_scores = _masked_topk_mean(class_saliency, background, topk)
        # Rows without an event support span (clean negatives, context rows)
        # fall back to 0/full-mean in _masked_topk_mean; keep only true
        # positive / clean-negative rows.
        pos_scores = pos_scores[pos_has & pos_scores.isfinite()]
        neg_scores = neg_scores[neg_has & neg_scores.isfinite()]
        if pos_scores.numel() == 0 or neg_scores.numel() == 0:
            continue
        sampled = torch.randint(
            neg_scores.numel(), (pos_scores.numel(), pair_count), device=device
        )
        sampled_neg = neg_scores[sampled]
        class_gaps = pos_scores.reshape(-1, 1) - sampled_neg
        class_loss = F.softplus(class_gaps.new_tensor(float(margin)) - class_gaps)
        losses.append(class_loss.mean())
        pos_pooled.append(pos_scores.mean())
        neg_pooled.append(sampled_neg.mean())
        gaps.append(class_gaps.mean())
        per_class_gaps[class_index] = class_gaps.mean()
        violations.append((class_gaps < float(margin)).float().mean())
        pairs += pos_scores.numel() * pair_count
        active_classes += 1
    zero = saliency.new_zeros(())
    loss = torch.stack(losses).mean() if losses else zero
    diagnostics = {
        "class_evidence_saliency_rank_loss": float(loss.detach().cpu()),
        "class_evidence_saliency_rank_active_classes": float(active_classes),
        "class_evidence_saliency_rank_pairs": float(pairs),
        "class_evidence_saliency_rank_gap": float(
            torch.stack(gaps).mean().detach().cpu() if gaps else 0.0
        ),
        "class_evidence_saliency_rank_violation_fraction": float(
            torch.stack(violations).mean().detach().cpu() if violations else 0.0
        ),
        "class_evidence_saliency_rank_positive_pooled": float(
            torch.stack(pos_pooled).mean().detach().cpu() if pos_pooled else 0.0
        ),
        "class_evidence_saliency_rank_negative_pooled": float(
            torch.stack(neg_pooled).mean().detach().cpu() if neg_pooled else 0.0
        ),
    }
    for class_index, label in enumerate(labels):
        if class_index >= saliency.shape[-1]:
            continue
        class_gap = per_class_gaps[class_index]
        diagnostics[f"class_evidence_saliency_rank_gap_{label}"] = float(
            class_gap.detach().cpu() if class_gap is not None else 0.0
        )
    return loss, diagnostics


def cross_window_saliency_consistency_loss(
    saliency: Tensor,                # [B, T, C]
    meta: list[Mapping[str, Any]],
    *,
    clip_duration: float,
    num_frames: int,
    tolerance_sec: float = 0.6,
    labels: tuple[str, ...] | list[str],
) -> tuple[Tensor, dict[str, float]]:
    """Constrain class saliency at the same absolute moment across windows.

    Two rows of the same video whose windows overlap cover the same absolute
    moments (the online-simulation sampler deliberately creates such pairs);
    their saliency there must agree.  Uses smooth-L1 over every matched
    (row_i, frame_ti, row_j, frame_tj, class) triple, normalized by the total
    number of matched frames.  Zero matched pairs -> zero loss (diagnostic
    reports pairs=0).
    """
    bsz = saliency.shape[0]
    device = saliency.device
    starts: list[float] = []
    videos: list[str] = []
    pair_ids: list[str] = []
    pair_roles: list[str] = []
    for row_meta in meta:
        starts.append(float(row_meta.get("sampled_clip_start", 0.0)))
        videos.append(str(row_meta.get("video_path", "")))
        pair_ids.append(str(row_meta.get("online_pair_id", "")))
        pair_roles.append(str(row_meta.get("online_pair_role", "")))
    frame_step = float(clip_duration) / max(int(num_frames) - 1, 1)
    frame_times = (
        torch.arange(int(num_frames), device=device).float() * frame_step
    )  # [T]
    losses: list[Tensor] = []
    diffs: list[Tensor] = []
    matched_frames = 0
    for i in range(bsz):
        for j in range(i + 1, bsz):
            if videos[i] != videos[j] or not videos[i]:
                continue
            if (
                not pair_ids[i]
                or pair_ids[i] != pair_ids[j]
                or {pair_roles[i], pair_roles[j]} != {"central", "edge"}
            ):
                continue
            window_delta = abs(starts[i] - starts[j])
            if window_delta >= float(clip_duration):
                continue
            # Absolute frame time difference matrix [T, T].
            time_i = (frame_times + starts[i]).unsqueeze(1)  # [T, 1]
            time_j = (frame_times + starts[j]).unsqueeze(0)  # [1, T]
            tdiff = time_i - time_j
            match = tdiff.abs() <= float(tolerance_sec)
            if not bool(match.any()):
                continue
            si = saliency[i].unsqueeze(1).expand(-1, int(num_frames), -1)  # [T, T, C]
            sj = saliency[j].unsqueeze(0).expand(int(num_frames), -1, -1)  # [T, T, C]
            diff = si - sj  # [T, T, C]
            matched = match.unsqueeze(-1).expand_as(diff)  # [T, T, C]
            if not bool(matched.any()):
                continue
            masked_diff = diff[matched]
            smooth = masked_diff.abs().mean()
            losses.append(smooth)
            diffs.append(masked_diff.abs().mean())
            matched_frames += int(matched.sum() // saliency.shape[-1])
    zero = saliency.new_zeros(())
    loss = torch.stack(losses).mean() if losses else zero
    diagnostics = {
        "class_evidence_cross_window_loss": float(loss.detach().cpu()),
        "class_evidence_cross_window_matched_frames": float(matched_frames),
        "class_evidence_cross_window_gap": float(
            torch.stack(diffs).mean().detach().cpu() if diffs else 0.0
        ),
    }
    return loss, diagnostics


def signed_causal_local_evidence_loss(
    kept_logits: Tensor,
    counterfactual_logits: Tensor,
    targets: Tensor,
    label_masks: Tensor,
    clean_negative_mask: Tensor,
    *,
    labels: tuple[str, ...] | list[str],
    positive_margin: float = 0.15,
    negative_margin: float = 0.10,
    negative_weight: float = 0.5,
    temperature: float = 0.5,
) -> tuple[Tensor, dict[str, float]]:
    """Make local evidence support positives and refute trusted negatives.

    The counterfactual keeps the identical global frame context while replacing
    sample-specific patch evidence by zero local evidence. Positive clips should
    score higher with their local evidence; per-class confirmed clean negatives
    should score lower, so detailed negative evidence remains useful.
    """
    if kept_logits.shape != counterfactual_logits.shape:
        raise ValueError("kept/counterfactual logits must have identical [B,C] shape")
    if targets.shape != kept_logits.shape or label_masks.shape != kept_logits.shape:
        raise ValueError("targets and label masks must match logits [B,C]")
    clean = clean_negative_mask.to(device=kept_logits.device, dtype=torch.bool)
    if clean.ndim == 1:
        clean = clean.unsqueeze(1).expand_as(kept_logits)
    if tuple(clean.shape) != tuple(kept_logits.shape):
        raise ValueError("signed causal clean-negative mask must be [B] or [B,C]")
    masks = label_masks.to(device=kept_logits.device) > 0
    targets_f = targets.to(device=kept_logits.device)
    gap = kept_logits - counterfactual_logits.detach()
    tau = max(float(temperature), 1e-6)
    neg_weight = max(float(negative_weight), 0.0)
    class_losses: list[Tensor] = []
    positive_gaps: list[Tensor] = []
    negative_gaps: list[Tensor] = []
    per_class_positive: list[Tensor | None] = [None] * kept_logits.shape[1]
    per_class_negative: list[Tensor | None] = [None] * kept_logits.shape[1]
    positive_violations: list[Tensor] = []
    negative_violations: list[Tensor] = []
    positive_slots = 0
    negative_slots = 0
    for class_index in range(kept_logits.shape[1]):
        positive = masks[:, class_index] & (targets_f[:, class_index] > 0.5)
        negative = (
            masks[:, class_index]
            & clean[:, class_index]
            & (targets_f[:, class_index] <= 0.5)
        )
        sides: list[Tensor] = []
        side_weights: list[float] = []
        if bool(positive.any()):
            pos_gap = gap[positive, class_index]
            pos_loss = tau * F.softplus(
                (pos_gap.new_tensor(float(positive_margin)) - pos_gap) / tau
            ).mean()
            sides.append(pos_loss)
            side_weights.append(1.0)
            positive_gaps.append(pos_gap.detach())
            per_class_positive[class_index] = pos_gap.detach().mean()
            positive_violations.append((pos_gap.detach() < float(positive_margin)).float())
            positive_slots += int(pos_gap.numel())
        if bool(negative.any()) and neg_weight > 0:
            # Negative evidence is useful when it lowers the event logit:
            # counterfactual - full >= negative_margin.
            neg_gap = -gap[negative, class_index]
            neg_loss = tau * F.softplus(
                (neg_gap.new_tensor(float(negative_margin)) - neg_gap) / tau
            ).mean()
            sides.append(neg_loss)
            side_weights.append(neg_weight)
            negative_gaps.append(neg_gap.detach())
            per_class_negative[class_index] = neg_gap.detach().mean()
            negative_violations.append((neg_gap.detach() < float(negative_margin)).float())
            negative_slots += int(neg_gap.numel())
        if sides:
            weights = kept_logits.new_tensor(side_weights)
            class_losses.append((torch.stack(sides) * weights).sum() / weights.sum())
    zero = kept_logits.sum() * 0.0
    loss = torch.stack(class_losses).mean() if class_losses else zero
    diagnostics = {
        "class_evidence_signed_causal_loss": float(loss.detach().cpu()),
        "class_evidence_signed_causal_active_classes": float(len(class_losses)),
        "class_evidence_signed_causal_positive_slots": float(positive_slots),
        "class_evidence_signed_causal_negative_slots": float(negative_slots),
        "class_evidence_signed_causal_positive_gap": float(
            torch.cat(positive_gaps).mean().cpu() if positive_gaps else 0.0
        ),
        "class_evidence_signed_causal_negative_gap": float(
            torch.cat(negative_gaps).mean().cpu() if negative_gaps else 0.0
        ),
        "class_evidence_signed_causal_positive_violation": float(
            torch.cat(positive_violations).mean().cpu() if positive_violations else 0.0
        ),
        "class_evidence_signed_causal_negative_violation": float(
            torch.cat(negative_violations).mean().cpu() if negative_violations else 0.0
        ),
    }
    for class_index, label in enumerate(labels):
        if class_index >= kept_logits.shape[1]:
            continue
        pos_gap = per_class_positive[class_index]
        neg_gap = per_class_negative[class_index]
        diagnostics[f"class_evidence_signed_causal_positive_gap_{label}"] = float(
            pos_gap.cpu() if pos_gap is not None else 0.0
        )
        diagnostics[f"class_evidence_signed_causal_negative_gap_{label}"] = float(
            neg_gap.cpu() if neg_gap is not None else 0.0
        )
    return loss, diagnostics


__all__ = [
    "bag_saliency_rank_loss",
    "cross_window_saliency_consistency_loss",
    "signed_causal_local_evidence_loss",
]
