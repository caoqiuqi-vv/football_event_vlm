"""Contracts and gradient diagnostics for detection-assisted representation learning.

Mechanism A is deliberately narrower than the historical object-motion paths:
the detector supplies a training-only localization loss, while event logits are
computed without reading detector predictions.  Both losses may update the
same DINO LoRA parameters, which is the causal mechanism under test.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor, nn


_MECHANISM_KEYS = {
    "enabled",
    "variant",
    "gradient_diagnostics",
}
_DIAGNOSTIC_KEYS = {
    "enabled",
    "exit_after_max_measurements",
    "interval_steps",
    "max_measurements_per_epoch",
    "require_nonzero",
}


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _enabled(mapping: Mapping[str, Any], key: str) -> bool:
    value = mapping.get(key, {})
    return bool(value.get("enabled", False)) if isinstance(value, Mapping) else False


def validate_mechanism_a_config(cfg: Any) -> None:
    """Fail closed when a Mechanism-A run mixes in another causal mechanism."""

    model_cfg = _mapping(cfg.get("model", {}), "model")
    mechanism_cfg = _mapping(model_cfg.get("mechanism_a", {}), "model.mechanism_a")
    if not bool(mechanism_cfg.get("enabled", False)):
        return

    unknown = sorted(set(mechanism_cfg) - _MECHANISM_KEYS)
    if unknown:
        raise ValueError(f"unknown model.mechanism_a keys: {unknown}")
    diagnostics_cfg = _mapping(
        mechanism_cfg.get("gradient_diagnostics", {}),
        "model.mechanism_a.gradient_diagnostics",
    )
    unknown_diagnostics = sorted(set(diagnostics_cfg) - _DIAGNOSTIC_KEYS)
    if unknown_diagnostics:
        raise ValueError(
            "unknown model.mechanism_a.gradient_diagnostics keys: "
            f"{unknown_diagnostics}"
        )

    variant = str(mechanism_cfg.get("variant", "")).strip().lower()
    if variant not in {"control", "object_aux", "relation_aux"}:
        raise ValueError(
            "model.mechanism_a.variant must be control, object_aux, or relation_aux"
        )
    if str(model_cfg.get("controlled_online_train_scope", "")) != "mechanism_a":
        raise ValueError(
            "Mechanism A requires model.controlled_online_train_scope=mechanism_a"
        )
    if str(model_cfg.get("view_fusion", "single")) != "single":
        raise ValueError("Mechanism A requires model.view_fusion=single")
    if bool(model_cfg.get("freeze_loaded_backbone", False)):
        raise ValueError(
            "Mechanism A must update shared LoRA; freeze_loaded_backbone must be false"
        )
    lora_cfg = _mapping(model_cfg.get("lora", {}), "model.lora")
    if not bool(lora_cfg.get("enabled", False)):
        raise ValueError("Mechanism A requires model.lora.enabled=true")
    if bool(lora_cfg.get("train_norm", False)):
        raise ValueError(
            "Mechanism A v1 keeps train_norm=false so the shared parameter set is exact"
        )

    aux_cfg = _mapping(
        model_cfg.get("object_spatial_aux", {}), "model.object_spatial_aux"
    )
    if not bool(aux_cfg.get("enabled", False)):
        raise ValueError("Mechanism A requires model.object_spatial_aux.enabled=true")
    if not bool(aux_cfg.get("representation_only", False)):
        raise ValueError(
            "Mechanism A requires object_spatial_aux.representation_only=true"
        )
    teacher_format = str(aux_cfg.get("teacher_format", ""))
    expected_teacher_format = (
        "tracked_ball_goal_relation_v1"
        if variant == "relation_aux"
        else "tracked_ball_npz_v1"
    )
    if teacher_format != expected_teacher_format:
        raise ValueError(
            f"Mechanism A {variant} requires object_spatial_aux.teacher_format="
            f"{expected_teacher_format}"
        )
    relation_cfg = _mapping(
        model_cfg.get("ball_goal_relation_aux", {}),
        "model.ball_goal_relation_aux",
    )
    relation_enabled = bool(relation_cfg.get("enabled", False))
    if (variant == "relation_aux") != relation_enabled:
        raise ValueError(
            "ball_goal_relation_aux.enabled must be true only for relation_aux"
        )
    if _enabled(aux_cfg, "online_missing_teacher"):
        raise ValueError("Mechanism A requires an offline, frozen object teacher")

    forbidden_model_paths = (
        "spatial_attention",
        "spatial_token_pooling",
        "class_evidence",
        "highres_glimpse",
        "temporal_difference",
        "response_curve_primary",
    )
    active_model_paths = [
        path for path in forbidden_model_paths if _enabled(model_cfg, path)
    ]
    if active_model_paths:
        raise ValueError(
            "Mechanism A cannot mix other evidence/fusion mechanisms: "
            f"{active_model_paths}"
        )

    video_cfg = _mapping(cfg.get("video", {}), "video")
    if float(video_cfg.get("hflip_prob", 0.0) or 0.0) != 0.0:
        raise ValueError("Mechanism A teacher alignment requires video.hflip_prob=0")
    cache_cfg = _mapping(cfg.get("cache", {}), "cache")
    if bool(cache_cfg.get("enabled", False)):
        raise ValueError("Mechanism A requires online DINO features; cache.enabled=false")

    train_cfg = _mapping(cfg.get("train", {}), "train")
    object_weight = float(train_cfg.get("object_heatmap_loss_weight", 0.0) or 0.0)
    relation_weight = float(
        train_cfg.get("ball_goal_relation_loss_weight", 0.0) or 0.0
    )
    if variant == "control" and (object_weight != 0.0 or relation_weight != 0.0):
        raise ValueError(
            "Mechanism A control requires object and relation loss weights=0"
        )
    if variant == "object_aux" and (object_weight <= 0.0 or relation_weight != 0.0):
        raise ValueError(
            "Mechanism A object_aux requires object weight>0 and relation weight=0"
        )
    if variant == "relation_aux" and (
        object_weight <= 0.0 or relation_weight <= 0.0
    ):
        raise ValueError(
            "Mechanism A relation_aux requires object and relation loss weights>0"
        )
    forbidden_losses = (
        "frame_det_loss_weight",
        "frame_tail_rank_loss_weight",
        "positive_retention_loss_weight",
        "positive_low_tail_bce_mining_loss_weight",
        "positive_low_tail_logit_margin_loss_weight",
        "clean_negative_rank_loss_weight",
        "clean_positive_logit_margin_loss_weight",
        "hard_positive_tail_margin_loss_weight",
        "tail_separation_loss_weight",
        "negative_teacher_guard_loss_weight",
        "online_global_tail_rank_loss_weight",
        "global_action_aux_loss_weight",
        "temporal_branch_aux_loss_weight",
        "temporal_gate_quality_loss_weight",
        "highres_causal_shuffle_loss_weight",
        "highres_crop_entropy_loss_weight",
        "highres_local_clip_loss_weight",
        "roi_feature_loss_weight",
        "save_given_shot_loss_weight",
        "save_shot_cohort_loss_weight",
        "set_piece_subtype_loss_weight",
        "raw_context_span_loss_weight",
        "class_evidence_cross_window_weight",
        "class_evidence_saliency_rank_weight",
        "global_conditioned_fn_weight",
        "global_conditioned_fp_weight",
        "hard_negative_rank_loss_weight",
        "online_hard_negative_loss_weight",
        "online_hard_negative_rank_loss_weight",
        "roi_quality_loss_weight",
        "local_loss_weight",
        "spatial_clip_loss_weight",
        "spatial_temporal_localization_loss_weight",
        "structured_frame_temporal_localization_loss_weight",
        "spatial_counterfactual_loss_weight",
        "global_conditioned_correction_loss_weight",
        "class_evidence_counterfactual_loss_weight",
        "class_evidence_no_evidence_loss_weight",
        "class_evidence_signed_causal_loss_weight",
        "fixed_teacher_logit_guard_weight",
        "ema_positive_retention_weight",
    )
    active_losses = [
        key
        for key in forbidden_losses
        if float(train_cfg.get(key, 0.0) or 0.0) != 0.0
    ]
    if active_losses:
        raise ValueError(
            "Mechanism A only permits clip, object heatmap, and relation losses; "
            f"disable {active_losses}"
        )
    if variant == "control" and bool(diagnostics_cfg.get("enabled", False)):
        raise ValueError(
            "gradient alignment diagnostics require the object_aux variant"
        )
    if _enabled(train_cfg, "online_pair_consistency"):
        raise ValueError("Mechanism A v1 forbids train.online_pair_consistency")
    if train_cfg.get("curriculum_stages", []):
        raise ValueError("Mechanism A v1 forbids train.curriculum_stages")
    long_video_cfg = _mapping(
        _mapping(cfg.get("data", {}), "data").get("long_video", {}),
        "data.long_video",
    )
    if _enabled(long_video_cfg, "online_simulation"):
        raise ValueError("Mechanism A v1 forbids data.long_video.online_simulation")
    if _enabled(long_video_cfg, "hard_negative"):
        raise ValueError("Mechanism A v1 forbids hard-negative sampling")


def shared_lora_named_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    """Return the exact shared representation parameters used by Mechanism A."""

    module = model.module if hasattr(model, "module") else model
    result: list[tuple[str, nn.Parameter]] = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad or not name.startswith("backbone."):
            continue
        if name.endswith((".lora_a", ".lora_b")):
            result.append((name, parameter))
    return result


def should_measure_gradient_alignment(
    mechanism_cfg: Mapping[str, Any], *, step: int, measured: int
) -> bool:
    diagnostics_cfg = _mapping(
        mechanism_cfg.get("gradient_diagnostics", {}),
        "model.mechanism_a.gradient_diagnostics",
    )
    if not bool(diagnostics_cfg.get("enabled", False)):
        return False
    interval = max(int(diagnostics_cfg.get("interval_steps", 200)), 1)
    maximum = max(int(diagnostics_cfg.get("max_measurements_per_epoch", 4)), 1)
    return measured < maximum and (int(step) == 1 or int(step) % interval == 0)


def gradient_alignment(
    event_loss: Tensor,
    object_loss: Tensor,
    named_parameters: Sequence[tuple[str, nn.Parameter]],
    *,
    object_weight: float,
    require_nonzero: bool = True,
) -> dict[str, float]:
    """Measure event/object gradient magnitude and cosine on shared LoRA.

    ``autograd.grad`` leaves ``parameter.grad`` untouched.  The caller can run
    the ordinary weighted backward afterwards on the same retained graph.
    """

    parameters = [parameter for _, parameter in named_parameters]
    if not parameters:
        raise RuntimeError("Mechanism A found no trainable shared LoRA parameters")
    event_grads = torch.autograd.grad(
        event_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    object_grads = torch.autograd.grad(
        object_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    device = event_loss.device
    dot = torch.zeros((), dtype=torch.float64, device=device)
    event_sq = torch.zeros_like(dot)
    object_sq = torch.zeros_like(dot)
    shared_tensors = 0
    for event_grad, object_grad in zip(event_grads, object_grads):
        if event_grad is not None:
            current = event_grad.detach().double()
            event_sq = event_sq + current.square().sum()
        if object_grad is not None:
            current = object_grad.detach().double()
            object_sq = object_sq + current.square().sum()
        if event_grad is not None and object_grad is not None:
            dot = dot + (
                event_grad.detach().double() * object_grad.detach().double()
            ).sum()
            shared_tensors += 1
    event_norm = math.sqrt(float(event_sq.cpu()))
    object_norm = math.sqrt(float(object_sq.cpu()))
    if require_nonzero and (event_norm <= 0.0 or object_norm <= 0.0):
        raise RuntimeError(
            "Mechanism A gradient route is inactive: "
            f"event_norm={event_norm:.6g} object_norm={object_norm:.6g}"
        )
    denominator = max(event_norm * object_norm, 1e-30)
    cosine = float(dot.cpu()) / denominator
    raw_ratio = object_norm / max(event_norm, 1e-30)
    return {
        "mechanism_a_event_grad_norm": event_norm,
        "mechanism_a_object_grad_norm": object_norm,
        "mechanism_a_grad_cosine": max(min(cosine, 1.0), -1.0),
        "mechanism_a_object_to_event_grad_ratio": raw_ratio,
        "mechanism_a_weighted_object_to_event_grad_ratio": (
            abs(float(object_weight)) * raw_ratio
        ),
        "mechanism_a_shared_parameter_tensors": float(len(parameters)),
        "mechanism_a_shared_gradient_tensors": float(shared_tensors),
    }
