"""Anti-shortcut supervision for class-conditioned no-evidence diagnostics."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor


def _get(cfg: Any, key: str, default: Any) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def balanced_no_evidence_loss(
    outputs: Mapping[str, Tensor],
    batch: Mapping[str, Any],
    device: torch.device,
    cfg: Any,
    *,
    labels: tuple[str, ...] | list[str],
) -> tuple[Tensor, dict[str, float]]:
    """Balance event/background router supervision independently per class.

    Gaussian shoulder targets between ``background_max_target`` and
    ``event_min_target`` are deliberately ignored. They represent uncertain
    context rather than reliable evidence-present/evidence-absent frames.
    """
    mass = outputs.get("class_evidence_no_evidence_mass")
    frame_targets = batch.get("frame_targets")
    frame_masks = batch.get("frame_target_masks")
    if not torch.is_tensor(mass):
        raise ValueError("balanced no-evidence loss requires no_evidence_mass")
    if not torch.is_tensor(frame_targets) or not torch.is_tensor(frame_masks):
        raise ValueError("balanced no-evidence loss requires frame targets/masks")
    targets = frame_targets.to(device, non_blocking=True).to(mass.dtype)
    masks = frame_masks.to(device, non_blocking=True).to(mass.dtype)
    label_masks = batch.get("label_masks")
    if torch.is_tensor(label_masks):
        masks = masks * label_masks.to(
            device, non_blocking=True
        ).to(mass.dtype).unsqueeze(1)
    event_min = float(_get(cfg, "event_min_target", 0.5))
    background_max = float(_get(cfg, "background_max_target", 0.1))
    if not 0.0 <= background_max < event_min <= 1.0:
        raise ValueError(
            "no-evidence thresholds require 0 <= background_max < event_min <= 1"
        )
    # Symmetric hinge loss (stable): event frames want null mass <= hinge
    # margin, background frames want null mass >= 1 - hinge margin.
    # BCE's equal-weight event/background average is an UNSTABLE balance:
    # d(loss)/d(bias) ~ 2*mean(mass) - 1, so the gate drifts into the
    # saturated regime (mass ~ 0.1) where sigmoid' ~ 0.08 compresses a
    # saliency gap of 0.02 into a null-mass gap of ~0.004 that never grows.
    # A symmetric hinge has +-1 gradients, a stable equilibrium, and drives
    # the shared patch scorer to produce stronger, more concentrated
    # event saliency (the gate's feedback path).
    hinge_event = float(_get(cfg, "no_evidence_hinge_event_margin", 0.4))
    hinge_background = float(_get(cfg, "no_evidence_hinge_background_margin", 0.4))
    mass_float = mass.float()
    event_loss_matrix = F.relu(mass_float - hinge_event)
    background_loss_matrix = F.relu((1.0 - hinge_background) - mass_float)
    matrix = torch.where(
        targets.float() >= event_min,
        event_loss_matrix,
        background_loss_matrix,
    ).to(mass.dtype)
    event_mask = masks * (targets >= event_min).to(mass.dtype)
    background_mask = masks * (targets <= background_max).to(mass.dtype)

    class_losses: list[Tensor] = []
    event_losses: list[Tensor] = []
    background_losses: list[Tensor] = []
    active_classes = 0
    require_both_sides = bool(
        _get(cfg, "require_both_sides_per_class", False)
    )
    global_ddp_balance = bool(_get(cfg, "global_ddp_balance", False))
    distributed = global_ddp_balance and dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if distributed else 1
    for class_index in range(mass.shape[-1]):
        class_event_mask = event_mask[:, :, class_index]
        class_background_mask = background_mask[:, :, class_index]
        event_count = class_event_mask.sum()
        background_count = class_background_mask.sum()
        global_event_count = event_count.detach().clone()
        global_background_count = background_count.detach().clone()
        if distributed:
            dist.all_reduce(global_event_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(global_background_count, op=dist.ReduceOp.SUM)
        if require_both_sides and not (
            bool(global_event_count > 0) and bool(global_background_count > 0)
        ):
            continue
        sides: list[Tensor] = []
        if bool(global_event_count > 0):
            if distributed:
                value = (
                    (matrix[:, :, class_index] * class_event_mask).sum()
                    / global_event_count.clamp_min(1.0)
                    * float(world_size)
                )
            else:
                value = _masked_mean(matrix[:, :, class_index], class_event_mask)
            sides.append(value)
            event_losses.append(value)
        if bool(global_background_count > 0):
            if distributed:
                value = (
                    (matrix[:, :, class_index] * class_background_mask).sum()
                    / global_background_count.clamp_min(1.0)
                    * float(world_size)
                )
            else:
                value = _masked_mean(
                    matrix[:, :, class_index], class_background_mask
                )
            sides.append(value)
            background_losses.append(value)
        if sides:
            class_losses.append(torch.stack(sides).mean())
            active_classes += 1
    zero = mass.new_zeros(())
    loss = torch.stack(class_losses).mean() if class_losses else zero
    event_loss = torch.stack(event_losses).mean() if event_losses else zero
    background_loss = (
        torch.stack(background_losses).mean() if background_losses else zero
    )
    valid = event_mask + background_mask
    event_mass_numerator = (mass * event_mask).sum()
    event_valid_slots = event_mask.sum()
    background_mass_numerator = (mass * background_mask).sum()
    background_valid_slots = background_mask.sum()
    diagnostics = {
        "class_evidence_no_evidence_loss": float(loss.detach().cpu()),
        "class_evidence_no_evidence_event_loss": float(event_loss.detach().cpu()),
        "class_evidence_no_evidence_background_loss": float(
            background_loss.detach().cpu()
        ),
        "class_evidence_no_evidence_active_classes": float(active_classes),
        "class_evidence_no_evidence_ignored_fraction": float(
            ((masks > 0) & (valid <= 0)).sum().detach().cpu()
            / max(float((masks > 0).sum().detach().cpu()), 1.0)
        ),
        "class_evidence_no_evidence_mass": float(
            _masked_mean(mass, valid.clamp_max(1.0)).detach().cpu()
        ),
        "class_evidence_no_evidence_event_mass": float(
            _masked_mean(mass, event_mask).detach().cpu()
        ),
        "class_evidence_no_evidence_background_mass": float(
            _masked_mean(mass, background_mask).detach().cpu()
        ),
        "class_evidence_no_evidence_event_mass_numerator": float(
            event_mass_numerator.detach().cpu()
        ),
        "class_evidence_no_evidence_event_mass_valid_slots": float(
            event_valid_slots.detach().cpu()
        ),
        "class_evidence_no_evidence_background_mass_numerator": float(
            background_mass_numerator.detach().cpu()
        ),
        "class_evidence_no_evidence_background_mass_valid_slots": float(
            background_valid_slots.detach().cpu()
        ),
    }
    for class_index, label in enumerate(labels):
        diagnostics[f"class_evidence_no_evidence_event_mass_{label}"] = float(
            _masked_mean(
                mass[:, :, class_index], event_mask[:, :, class_index]
            ).detach().cpu()
        )
        diagnostics[f"class_evidence_no_evidence_background_mass_{label}"] = float(
            _masked_mean(
                mass[:, :, class_index], background_mask[:, :, class_index]
            ).detach().cpu()
        )
        for kind, class_mask in (
            ("event", event_mask[:, :, class_index]),
            ("background", background_mask[:, :, class_index]),
        ):
            stem = f"class_evidence_no_evidence_{kind}_mass_{label}"
            diagnostics[f"{stem}_numerator"] = float(
                (mass[:, :, class_index] * class_mask).sum().detach().cpu()
            )
            diagnostics[f"{stem}_valid_slots"] = float(
                class_mask.sum().detach().cpu()
            )
    content_logits = outputs.get("class_evidence_no_evidence_content_logits")
    if torch.is_tensor(content_logits):
        diagnostics["class_evidence_no_evidence_content_logit_abs"] = float(
            content_logits.detach().float().abs().mean().cpu()
        )
        diagnostics["class_evidence_no_evidence_content_logit_std"] = float(
            content_logits.detach().float().std(unbiased=False).cpu()
        )
    # Monotone saliency gate monitoring (plan gate): report saliency strength
    # on event vs background frames and the null-mass separation. The gate
    # passes when event saliency exceeds background saliency and
    # background null mass - event null mass grows (both move through the
    # shared patch scorer, so they are the observable training signal).
    saliency = outputs.get("class_evidence_no_evidence_saliency")
    if torch.is_tensor(saliency):
        saliency = saliency.to(mass.dtype)
        saliency_event = _masked_mean(
            saliency.mean(dim=-1), event_mask
        ).detach().cpu()
        saliency_background = _masked_mean(
            saliency.mean(dim=-1), background_mask
        ).detach().cpu()
        diagnostics["class_evidence_no_evidence_event_saliency"] = float(
            saliency_event
        )
        diagnostics["class_evidence_no_evidence_background_saliency"] = float(
            saliency_background
        )
        diagnostics["class_evidence_no_evidence_saliency_gap"] = float(
            saliency_event - saliency_background
        )
        diagnostics["class_evidence_no_evidence_null_mass_gap"] = float(
            _masked_mean(mass, background_mask).detach().cpu()
            - _masked_mean(mass, event_mask).detach().cpu()
        )
    return loss, diagnostics


__all__ = ["balanced_no_evidence_loss"]
