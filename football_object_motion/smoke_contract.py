"""Contract hashing and artifact validation for the causal v4 GPU smoke gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "object_motion_v4_two_step_smoke_v1"

# These are the production launcher values that determine pair construction,
# decoded input shape and the two curriculum stages exercised by the smoke.
RESOLVED_CRITICAL_PARAMETERS: dict[str, Any] = {
    "model.object_motion.frames_per_segment": 11,
    "model.object_motion.duration_sec": 10.0,
    "model.object_motion.image_size": [720, 1280],
    "model.object_motion.teacher_image_size": [720, 1280],
    "model.object_motion.patch_size": 16,
    "model.object_motion.hidden_dim": 512,
    "model.object_motion.num_heads": 8,
    "model.object_motion.temporal_layers": 2,
    "model.object_motion.detach_detector_for_event": True,
    "model.object_motion.class_evidence_floors": [0.05, 0.05, 0.35],
    "model.object_motion.ball_lora_rank": 4,
    "model.object_motion.ball_lora_alpha": 4.0,
    "model.object_motion.ball_lora_last_blocks": 8,
    "model.object_motion.ball_feature_layers": [11, 17, 20, 23],
    "model.object_motion.ball_layer_weights": [0.15, 0.25, 0.25, 0.35],
    "model.object_motion.ball_topk": 4,
    "model.object_motion.ball_temperature": 0.25,
    "model.object_motion.checkpoint_trainable_blocks": True,
    "model.object_motion.online_teacher.batch_size": 8,
    "model.object_motion.require_true_pairs": True,
    "model.object_motion.pair_min_gap_sec": 5.0,
    "model.object_motion.reviewed_negative_manifests": [
        "outputs/football_hard_negatives/reviewed_mid_score_v2_shot_save_setpiece.json",
        "outputs/football_hard_negatives/other_action_reviewed_train_shot_save_score_filtered.json",
    ],
    "video.positive_window_strategy": "anchor_range_jitter",
    "video.positive_anchor_min_sec": 0.5,
    "video.positive_anchor_max_sec": 9.5,
    "video.temporal_jitter_sec": 0.0,
    "data.long_video.negative_ratio": 5.0,
    "data.long_video.negative_ratio_by_split.train": 5.0,
    "data.long_video.hard_negative.enabled": True,
    "data.long_video.online_simulation.window_stride_sec": 5.0,
    "data.long_video.online_simulation.primary_positive_min_sec": 3.0,
    "data.long_video.online_simulation.primary_positive_max_sec": 7.0,
    "data.long_video.online_simulation.clean_background_windows_per_epoch": 2400,
    "data.long_video.online_simulation.near_event_ignore_sec": 2.0,
    "data.long_video.online_simulation.near_event_anchor_gap_min_sec": 5.0,
    "data.long_video.online_simulation.near_event_anchor_gap_max_sec": 15.0,
    "train.per_gpu_batch_size": 1,
    "train.object_motion_frame_loss_weight": 0.10,
    "train.object_motion_dense_rank_loss_weight": 0.15,
    "train.object_motion_relation_loss_weight": 1.0,
    "train.object_motion_guard_loss_weight": 1.0,
    "train.ball_lora_lr": 1e-6,
    "train.ball_lora_weight_decay": 0.0,
    "train.ball_lora_grad_clip": 0.1,
    "train.object_motion_head_lr": 2e-5,
    "train.object_motion_head_weight_decay": 0.05,
    "train.object_motion_head_grad_clip": 1.0,
    "train.object_motion_ball_contrastive_loss_weight": 0.2,
    "train.object_motion_ball_center_loss_weight": 0.5,
    "train.object_motion_ball_track_loss_weight": 0.1,
    "train.object_motion_ball_preserve_loss_weight": 0.05,
    "smoke.step1_alpha": 0.0,
    "smoke.step2_alpha": 0.10,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_fingerprint(path: Path, *, large: bool = False) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    result: dict[str, Any] = {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if not large:
        result["sha256"] = _sha256(resolved)
    return result


def build_contract(
    repo: Path,
    config: Path,
    checkpoint: Path,
    *,
    parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    repo = repo.resolve()
    manifest_paths = [repo / value for value in RESOLVED_CRITICAL_PARAMETERS["model.object_motion.reviewed_negative_manifests"]]
    sources = [
        repo / "scripts/run_object_motion_adapter_v4.sh",
        repo / "scripts/run_object_motion_adapter_v3.sh",
        repo / "scripts/run_object_motion_smoke_after_epoch2.sh",
        repo / "football_object_motion/smoke.py",
        repo / "football_object_motion/ball_backbone.py",
        repo / "football_object_motion/model.py",
        repo / "football_object_motion/train.py",
        repo / "football_object_motion/losses.py",
        repo / "football_object_motion/teacher.py",
        config,
        *manifest_paths,
    ]
    files = [_file_fingerprint(path) for path in sources]
    files.append(_file_fingerprint(checkpoint, large=True))
    payload = {
        "schema": "object_motion_v4_contract_v1",
        "files": files,
        "resolved_parameters": dict(parameters or RESOLVED_CRITICAL_PARAMETERS),
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload["contract_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def validate_smoke_artifact(
    payload: Mapping[str, Any],
    *,
    expected_contract_hash: str,
    smoke_started_at: float,
    now: float | None = None,
    max_age_sec: float = 86400.0,
) -> tuple[bool, str]:
    current = float(time.time() if now is None else now)
    try:
        generated = float(payload["generated_at_unix"])
        if payload.get("state") != "passed" or payload.get("schema") != SCHEMA:
            return False, "state/schema mismatch"
        if payload.get("contract_hash") != expected_contract_hash:
            return False, "contract hash mismatch"
        if int(payload.get("steps_completed", 0)) != 2:
            return False, "two steps were not completed"
        if generated < float(smoke_started_at) or generated > current:
            return False, "artifact was not generated by this watchdog run"
        if current - generated > float(max_age_sec):
            return False, "artifact is stale"
        if int(payload.get("pair_count", 0)) <= 0:
            return False, "pair_count is zero"
        pair = payload.get("first_pair", {})
        if not isinstance(pair, Mapping):
            return False, "first_pair is missing"
        if not (
            pair.get("legal") is True
            and pair.get("same_pair_id") is True
            and pair.get("same_video") is True
            and pair.get("masks_valid") is True
            and float(pair.get("coverage_min", 0.0)) >= 0.95
        ):
            return False, "first pair metadata/mask/coverage is invalid"
        steps = payload.get("steps", [])
        if not isinstance(steps, Sequence) or len(steps) != 2:
            return False, "step evidence is missing"
        step1, step2 = steps
        grad1 = step1.get("adapter_grad_norms", {})
        grad2 = step2.get("adapter_grad_norms", {})
        if not (
            step1.get("alpha") == 0.0
            and step1.get("identity_bitwise") is True
            and float(step1.get("anchor_identity_max_abs_error", 1.0)) == 0.0
            and float(step1.get("detector_event_grad_norm", 1.0)) == 0.0
            and float(grad1.get("clip_residual", 0.0)) > 0.0
            and float(grad1.get("relation", -1.0)) == 0.0
            and float(grad1.get("ball_lora", 0.0)) > 1e-6
        ):
            return False, "step1 identity/gradient evidence is invalid"
        if payload.get("experiment_status") != "teacher-supervised exploratory":
            return False, "experiment status is not exploratory"
        peak_memory = float(payload.get("peak_cuda_memory_gib", float("inf")))
        if not math.isfinite(peak_memory) or peak_memory >= 28.0:
            return False, "smoke peak CUDA memory is not below 28 GiB"
        residual = step2.get("effective_residual", {})
        residual_values = [float(residual.get("clip_abs_mean", 0.0)), float(residual.get("frame_abs_mean", 0.0))]
        if not (
            abs(float(step2.get("alpha", -1.0)) - 0.10) < 1e-9
            and float(grad2.get("frame_residual", 0.0)) > 1e-5
            and float(grad2.get("relation", 0.0)) > 1e-5
            and all(math.isfinite(value) and value > 0.0 for value in residual_values)
        ):
            return False, "step2 gradient/residual evidence is invalid"
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        return False, f"malformed artifact: {error}"
    return True, "ok"


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    contract_parser = subparsers.add_parser("contract")
    for target in (contract_parser,):
        target.add_argument("--repo", required=True, type=Path)
        target.add_argument("--config", required=True, type=Path)
        target.add_argument("--checkpoint", required=True, type=Path)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--artifact", required=True, type=Path)
    validate_parser.add_argument("--contract-hash", required=True)
    validate_parser.add_argument("--started-at", required=True, type=float)
    validate_parser.add_argument("--max-age-sec", type=float, default=86400.0)
    args = parser.parse_args()
    if args.command == "contract":
        print(json.dumps(build_contract(args.repo, args.config, args.checkpoint), ensure_ascii=False))
        return
    payload = json.loads(args.artifact.read_text())
    valid, reason = validate_smoke_artifact(
        payload,
        expected_contract_hash=args.contract_hash,
        smoke_started_at=args.started_at,
        max_age_sec=args.max_age_sec,
    )
    print(reason)
    raise SystemExit(0 if valid else 1)


if __name__ == "__main__":
    main()
