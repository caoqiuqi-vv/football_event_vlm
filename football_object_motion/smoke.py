#!/usr/bin/env python
"""One-real-batch GPU preflight for the object-motion experiment."""

from __future__ import annotations

import argparse
import json
import time
import math
from pathlib import Path

import torch
import torch.nn.functional as F

import train_football_events as base
from football_object_motion.train import (
    build_motion_optimizer,
    forward_motion_batch,
    install_hooks,
    make_motion_loader,
    make_motion_model,
    object_motion_loss_hook,
    object_motion_teacher_from_config,
    prepare_motion_datasets,
)
from football_object_motion.losses import reset_pair_residual_queues, validate_same_video_pairs
from football_object_motion.smoke_contract import (
    RESOLVED_CRITICAL_PARAMETERS,
    SCHEMA,
    atomic_write_json,
    build_contract,
)


def adapter_grad_norms(model: torch.nn.Module) -> dict[str, float]:
    prefixes = {
        "ball_lora": ("backbone.",),
        "ball_head": ("object_motion_adapter.ball_layer_",),
        "heatmap": ("object_motion_adapter.heatmap_head.",),
        "presence": ("object_motion_adapter.presence_heads.",),
        "relation": ("object_motion_adapter.relation_projection.",),
        "frame_residual": ("object_motion_adapter.frame_residual_head.",),
        "clip_residual": ("object_motion_adapter.clip_residual_head.",),
    }
    result = {name: 0.0 for name in prefixes}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        for group, candidates in prefixes.items():
            matches = any(name.startswith(prefix) for prefix in candidates)
            if group == "ball_lora":
                matches = matches and name.endswith(("ball_lora_a", "ball_lora_b"))
            if matches:
                result[group] += float(parameter.grad.detach().float().square().sum().cpu())
    return {name: math.sqrt(value) for name, value in result.items()}


def detector_event_grad_norm(
    model: torch.nn.Module, outputs: dict[str, torch.Tensor]
) -> float:
    """Measure event-path gradients into ball/detector parameters only."""
    detector_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if (
            "ball_lora_" in name
            or name.startswith("object_motion_adapter.ball_layer_")
            or name.startswith("object_motion_adapter.heatmap_head.")
            or name.startswith("object_motion_adapter.presence_heads.")
        )
    ]
    event_objective = (
        outputs["object_motion_raw_clip_residual_logits"].sum()
        + outputs["object_motion_raw_frame_residual"].sum()
    )
    gradients = torch.autograd.grad(event_objective, detector_parameters, retain_graph=True, allow_unused=True)
    square = sum(
        float(gradient.detach().float().square().sum().cpu())
        for gradient in gradients
        if gradient is not None
    )
    return math.sqrt(square)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--contract-hash", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    contract = build_contract(repo, Path(args.config), Path(args.checkpoint))
    if contract["contract_hash"] != args.contract_hash:
        raise ValueError("watchdog contract hash does not match resolved smoke contract")
    if args.batch_size != 1:
        raise ValueError("pair-queue smoke requires batch size 1")
    overrides = [
        f"model.init_checkpoint={args.checkpoint}",
        "model.init_checkpoint_strict=false",
        "model.object_spatial_aux.enabled=false",
        "model.controlled_online_train_scope=all",
        "model.freeze_backbone=true",
        "model.finetune_last_blocks=0",
        "model.backbone_frame_chunk_size=1",
        "model.object_motion.enabled=true",
        "model.object_motion.ball_lora_rank=4",
        "model.object_motion.ball_lora_alpha=4.0",
        "model.object_motion.ball_lora_last_blocks=8",
        "model.object_motion.ball_feature_layers=[11,17,20,23]",
        "model.object_motion.ball_layer_weights=[0.15,0.25,0.25,0.35]",
        "model.object_motion.ball_topk=4",
        "model.object_motion.ball_temperature=0.25",
        "model.object_motion.backbone_frame_chunk_size=1",
        "model.object_motion.checkpoint_trainable_blocks=true",
        "model.object_motion.frames_per_segment=11",
        "model.object_motion.duration_sec=10.0",
        "model.object_motion.image_size=[720,1280]",
        "model.object_motion.teacher_image_size=[720,1280]",
        "model.object_motion.patch_size=16",
        "model.object_motion.hidden_dim=512",
        "model.object_motion.num_heads=8",
        "model.object_motion.temporal_layers=2",
        "model.object_motion.dropout=0.1",
        "model.object_motion.topk_ratios=[0.01,0.05,0.08]",
        "model.object_motion.temperature=0.5",
        "model.object_motion.residual_max_delta=0.25",
        "model.object_motion.frame_residual_max_delta=0.15",
        "model.object_motion.clip_residual_max_delta=0.25",
        "model.object_motion.gate_init=0.10",
        "model.object_motion.gate_max=1.0",
        "model.object_motion.gate_floor=0.25",
        "model.object_motion.relation_delta=1.0",
        "model.object_motion.fusion_delta=0.5",
        "model.object_motion.evidence_gate_floor=0.02",
        "model.object_motion.class_evidence_floors=[0.05,0.05,0.35]",
        "model.object_motion.detach_detector_for_event=true",
        "model.object_motion.learned_gate_budget=1.0",
        "model.object_motion.residual_saturation_threshold=0.85",
        "model.object_motion.pairwise_rank_margin=0.05",
        "model.object_motion.pairwise_rank_temperature=0.10",
        "model.object_motion.require_true_pairs=true",
        "model.object_motion.pair_min_gap_sec=5.0",
        "model.object_motion.negative_guard_margin=0.15",
        "model.object_motion.reviewed_negative_manifests=[outputs/football_hard_negatives/reviewed_mid_score_v2_shot_save_setpiece.json,outputs/football_hard_negatives/other_action_reviewed_train_shot_save_score_filtered.json]",
        "model.object_motion.positive_threshold=0.05",
        "model.object_motion.object_loss_weights=[2.0,1.0,0.25]",
        "model.object_motion.negative_weights=[0.50,0.25,0.05]",
        "model.object_motion.presence_negative_weights=[1.0,1.0,0.25]",
        "model.object_motion.online_teacher.enabled=true",
        "model.object_motion.online_teacher.ball_checkpoint=/mnt/data_7t/qiuqi/code/soccer/onlysoccer_1920_11s.pt",
        "model.object_motion.online_teacher.scene_checkpoint=/home/new_users/qiuqi/code/det_and_track/checkpoints/yolo11m_person_goal_fieldLine_1920_7.22.pt",
        "model.object_motion.online_teacher.ball_confidence=0.10",
        "model.object_motion.online_teacher.goal_confidence=0.25",
        "model.object_motion.online_teacher.person_confidence=0.25",
        "model.object_motion.online_teacher.input_size=[1088,1920]",
        "model.object_motion.online_teacher.batch_size=8",
        "model.object_motion.online_teacher.half=true",
        "video.positive_window_strategy=anchor_range_jitter",
        "video.positive_anchor_min_sec=0.5",
        "video.positive_anchor_max_sec=9.5",
        "video.temporal_jitter_sec=0.0",
        "data.long_video.negative_ratio=5.0",
        "data.long_video.hard_negative.enabled=true",
        "data.long_video.hard_negative.manifests=[outputs/football_hard_negatives/reviewed_mid_score_v2_shot_save_setpiece.json,outputs/football_hard_negatives/other_action_reviewed_train_shot_save_score_filtered.json]",
        "data.long_video.negative_ratio_by_split.train=5.0",
        "data.long_video.online_simulation.enabled=true",
        "data.long_video.online_simulation.window_stride_sec=5.0",
        "data.long_video.online_simulation.primary_positive_min_sec=3.0",
        "data.long_video.online_simulation.primary_positive_max_sec=7.0",
        "data.long_video.online_simulation.grid_clip_loss_weight=0.35",
        "data.long_video.online_simulation.grid_frame_loss_weight=0.50",
        "data.long_video.online_simulation.edge_clip_loss_weight=0.35",
        "data.long_video.online_simulation.edge_frame_loss_weight=0.50",
        "data.long_video.online_simulation.event_sampling_mode=cover_all_once",
        "data.long_video.online_simulation.clean_background_windows_per_epoch=2400",
        "data.long_video.online_simulation.near_event_ignore_sec=2.0",
        "data.long_video.online_simulation.near_event_anchor_gap_min_sec=5.0",
        "data.long_video.online_simulation.near_event_anchor_gap_max_sec=15.0",
        "data.long_video.split_files.val=[configs/football/splits/object_motion_dense_sentinel3.txt]",
        "train.object_heatmap_loss_weight=1.0",
        "train.object_motion_heatmap_loss_weight=0.50",
        "train.object_motion_distribution_loss_weight=0.50",
        "train.object_motion_presence_loss_weight=0.25",
        "train.object_motion_coordinate_loss_weight=0.25",
        "train.object_motion_consistency_loss_weight=0.20",
        "train.object_motion_frame_loss_weight=0.10",
        "train.object_motion_dense_rank_loss_weight=0.15",
        "train.object_motion_no_evidence_loss_weight=0.10",
        "train.object_motion_residual_energy_loss_weight=0.0",
        "train.object_motion_gate_budget_loss_weight=0.0",
        "train.object_motion_saturation_loss_weight=0.02",
        "train.object_motion_relation_loss_weight=1.0",
        "train.object_motion_guard_loss_weight=1.0",
        "train.object_motion_ball_strong_loss_weight=1.0",
        "train.object_motion_ball_center_loss_weight=0.50",
        "train.object_motion_ball_contrastive_loss_weight=0.20",
        "train.object_motion_ball_track_loss_weight=0.10",
        "train.object_motion_ball_preserve_loss_weight=0.05",
        "train.ball_lora_lr=1.0e-6",
        "train.ball_lora_weight_decay=0.0",
        "train.ball_lora_grad_clip=0.10",
        "train.object_motion_head_lr=2.0e-5",
        "train.object_motion_head_weight_decay=0.05",
        "train.object_motion_head_grad_clip=1.0",
        "train.grad_clip_norm=1.0",
        "train.object_motion_event_residual_scale=0.0",
        "train.positive_retention_loss_weight=1.0",
        "train.positive_retention_margin=0.0",
        "data.num_workers=0",
        "train.frame_det_loss_weight=0.0",
        "eval.online_validation.enabled=false",
    ]
    install_hooks()
    cfg = base.load_config(args.config, overrides)
    cfg["device"] = "cuda:0"
    cfg["gpu_ids"] = [0]
    base.configure_runtime_threads(cfg)
    base.configure_label_schema(cfg)
    base.seed_everything(int(cfg.get("seed", 42)), True)
    device = torch.device("cuda:0")

    train_dataset, _val_dataset, _train_records, _val_records = (
        prepare_motion_datasets(cfg, use_cache=False)
    )
    train_loader = make_motion_loader(
        train_dataset, cfg, is_train=True, batch_size=1, distributed=False
    )
    iterator = iter(train_loader)
    batches = [next(iterator), next(iterator)]
    combined_targets = torch.cat([batch["targets"] for batch in batches], dim=0)
    combined_masks = torch.cat([batch["label_masks"] for batch in batches], dim=0)
    combined_meta = [batch["meta"][0] for batch in batches]
    pair_count = validate_same_video_pairs(
        combined_meta, combined_targets, combined_masks, min_gap_sec=5.0
    )
    positive_meta, negative_meta = combined_meta
    class_index = int(positive_meta["relation_class_index"])
    masks_valid = bool(
        combined_masks[0, class_index] > 0 and combined_masks[1, class_index] > 0
    )
    coverage_min = min(float(meta.get("object_motion_coverage", 0.0)) for meta in combined_meta)
    first_pair = {
        "legal": pair_count > 0,
        "pair_id": str(positive_meta["pair_id"]),
        "roles": [str(positive_meta["pair_role"]), str(negative_meta["pair_role"])],
        "source": str(positive_meta["source"]),
        "video_id": str(positive_meta["video_id"]),
        "class_index": class_index,
        "same_pair_id": positive_meta["pair_id"] == negative_meta["pair_id"],
        "same_video": (positive_meta["source"], positive_meta["video_id"]) == (
            negative_meta["source"], negative_meta["video_id"]
        ),
        "masks_valid": masks_valid,
        "coverage_min": coverage_min,
        "negative_reviewed": bool(negative_meta.get("reviewed_negative", False)),
        "negative_full_clean": bool(negative_meta.get("full_clean_window", False)),
        "negative_gt_gap_sec": float(negative_meta["nearest_same_class_gt_gap"]),
    }
    if not masks_valid or coverage_min < 0.95:
        raise RuntimeError(f"invalid first pair smoke evidence: {first_pair}")

    model = make_motion_model(cfg, use_cached_features=False, device=device)
    model.train()
    teacher = object_motion_teacher_from_config(cfg, device)
    if teacher is None:
        raise RuntimeError("online object-motion Teacher was not created")
    teacher_metrics = [teacher.fill_missing(batch) for batch in batches]
    optimizer = build_motion_optimizer(model, cfg)
    reset_pair_residual_queues()
    step_results = []
    for step_index, (alpha, batch) in enumerate(zip((0.0, 0.10), batches), start=1):
        cfg.train["object_motion_event_residual_scale"] = alpha
        optimizer.zero_grad(set_to_none=True)
        with base.autocast_context(device, True, "bf16"):
            outputs = forward_motion_batch(model, batch, device, return_aux=True)
            if not isinstance(outputs, dict):
                raise RuntimeError("motion model did not return auxiliary outputs")
            anchor = outputs["retention_reference_logits"]
            anchor_error = float((outputs["logits"] - anchor).abs().max().detach().cpu())
            identity_bitwise = bool(torch.equal(outputs["logits"], anchor))
            if step_index == 1 and (not identity_bitwise or anchor_error != 0.0):
                raise RuntimeError(f"alpha-zero adapter changed anchor logits by {anchor_error}")
            targets = batch["targets"].to(device)
            masks = batch["label_masks"].to(device)
            clip_loss = (
                F.binary_cross_entropy_with_logits(outputs["logits"], targets, reduction="none")
                * masks
            ).sum() / masks.sum().clamp_min(1.0)
            auxiliary_loss, components = object_motion_loss_hook(outputs, batch, cfg, device)
            total_loss = clip_loss + auxiliary_loss
        event_detector_grad = detector_event_grad_norm(model, outputs)
        if event_detector_grad != 0.0:
            raise RuntimeError(f"event loss reached ball/detector parameters: {event_detector_grad}")
        total_loss.backward()
        gradients = adapter_grad_norms(model)
        strong_ball_frames = int(
            (batch["object_motion_heatmap_masks"][..., 0].amax(dim=2) >= 0.999).sum()
        )
        if not math.isfinite(float(total_loss.detach().cpu())):
            raise RuntimeError("non-finite smoke loss")
        if step_index == 1:
            if strong_ball_frames <= 0 or gradients["ball_lora"] <= 1e-6:
                raise RuntimeError(
                    f"step1 lacks strong-ball LoRA evidence frames={strong_ball_frames} gradients={gradients}"
                )
            if gradients["clip_residual"] <= 0.0:
                raise RuntimeError(f"step1 missing residual-head gradients: {gradients}")
            if gradients["relation"] != 0.0:
                raise RuntimeError(
                    "zero-initialized residual head unexpectedly passed step1 "
                    f"gradients into relation projection: {gradients}"
                )
        elif gradients["frame_residual"] <= 1e-5 or gradients["relation"] <= 1e-5:
            raise RuntimeError(f"step2 residual/relation gradients too small: {gradients}")
        illegal_gradients = [
            name
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
            and not name.startswith("object_motion_adapter.")
            and "ball_lora_" not in name
        ]
        if illegal_gradients:
            raise RuntimeError(f"anchor/legacy path received gradients: {illegal_gradients[:5]}")
        optimizer.step()
        with torch.no_grad(), base.autocast_context(device, True, "bf16"):
            post_outputs = forward_motion_batch(model, batch, device, return_aux=True)
        effective_residual = {
            "clip_abs_mean": float(post_outputs["object_motion_clip_residual"].abs().mean().float().cpu()),
            "frame_abs_mean": float(post_outputs["object_motion_frame_residual"].abs().mean().float().cpu()),
            "clip_abs_max": float(post_outputs["object_motion_clip_residual"].abs().max().float().cpu()),
            "frame_abs_max": float(post_outputs["object_motion_frame_residual"].abs().max().float().cpu()),
        }
        if step_index == 2 and not all(
            math.isfinite(value) and value > 0.0
            for value in (effective_residual["clip_abs_mean"], effective_residual["frame_abs_mean"])
        ):
            raise RuntimeError(f"step2 effective residual is invalid: {effective_residual}")
        step_results.append(
            {
                "step": step_index,
                "alpha": alpha,
                "total_loss": float(total_loss.detach().cpu()),
                "clip_loss": float(clip_loss.detach().cpu()),
                "strong_ball_frames": strong_ball_frames,
                "adapter_grad_norms": gradients,
                "detector_event_grad_norm": event_detector_grad,
                "identity_bitwise": identity_bitwise,
                "anchor_identity_max_abs_error": anchor_error,
                "effective_residual": effective_residual,
                "components": components,
            }
        )

    payload = {
        "state": "passed",
        "schema": SCHEMA,
        "experiment_status": "teacher-supervised exploratory",
        "generated_at_unix": time.time(),
        "contract_hash": contract["contract_hash"],
        "contract": contract,
        "steps_completed": 2,
        "steps": step_results,
        "pair_count": len(train_dataset.relation_pair_indices),
        "first_pair": first_pair,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "batch_size": args.batch_size,
        "teacher": teacher_metrics,
        "peak_cuda_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
