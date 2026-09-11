#!/usr/bin/env python
"""Recall-constrained, NMS-free evaluation of a clip-teacher candidate reranker.

The experiment deliberately keeps the model family small: Stage-1 alone, a
complete event-centred clip model alone, and standardized logit blends.  All
fusion choices and thresholds are selected on calibration videos and then
applied unchanged to an optional external test set.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    positive = value >= 0.0
    result = np.empty_like(value)
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    result[~positive] = exp_value / (1.0 + exp_value)
    return result


def logit(value: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=np.float64), eps, 1.0 - eps)
    return np.log(value / (1.0 - value))


def robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    q25, q75 = np.quantile(values, (0.25, 0.75))
    return median, max(float(q75 - q25) / 1.349, 1e-3)


def parse_ids(value: str) -> set[str]:
    return {item.strip() for item in value.replace(";", ",").split(",") if item.strip()}


def load_teacher_shards(path: Path, rows: int) -> dict[str, Any]:
    paths = sorted(path.glob("teacher_rank*_of_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no teacher shards in {path}")
    logits: np.ndarray | None = None
    centres: np.ndarray | None = None
    filled = np.zeros(rows, dtype=bool)
    reference: dict[str, Any] = {}
    for shard_path in paths:
        shard = np.load(shard_path, allow_pickle=False)
        indices = shard["indices"].astype(np.int64)
        values = shard["teacher_logits"].astype(np.float32)
        shard_centres = shard["actual_window_centres"].astype(np.float32)
        offsets = shard["offsets"].astype(np.float32)
        labels = [str(value) for value in shard["labels"]]
        checkpoint = str(shard["checkpoint"])
        if values.ndim != 3 or values.shape[:2] != shard_centres.shape:
            raise ValueError(f"invalid teacher shard shape in {shard_path}: {values.shape}")
        current = {"offsets": offsets, "labels": labels, "checkpoint": checkpoint}
        if not reference:
            reference = current
            logits = np.empty((rows, values.shape[1], values.shape[2]), dtype=np.float32)
            centres = np.empty((rows, values.shape[1]), dtype=np.float32)
        elif (
            labels != reference["labels"]
            or checkpoint != reference["checkpoint"]
            or not np.array_equal(offsets, reference["offsets"])
        ):
            raise ValueError(f"inconsistent teacher shard metadata: {shard_path}")
        if filled[indices].any():
            raise ValueError(f"duplicate teacher rows in {shard_path}")
        assert logits is not None and centres is not None
        logits[indices] = values
        centres[indices] = shard_centres
        filled[indices] = True
    if logits is None or centres is None or not filled.all():
        raise ValueError(f"teacher shards missing rows: {np.where(~filled)[0][:10].tolist()}")
    return {**reference, "logits": logits, "actual_window_centres": centres}


def load_dataset(
    cache_path: Path,
    metadata_path: Path,
    teacher_dir: Path,
    excluded_video_ids: set[str],
) -> dict[str, Any]:
    cache = np.load(cache_path, allow_pickle=True)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if len(metadata) != len(cache["video_ids"]):
        raise ValueError("cache and metadata row counts differ")
    labels = [str(value) for value in cache["labels"]]
    shot_index = labels.index("shot")
    video_ids = cache["video_ids"].astype(str)
    candidate_times = cache["candidate_times"][:, shot_index].astype(np.float64)
    baseline_logits = logit(cache["online_probs"][:, shot_index])
    ends = cache["clip_ends"].astype(np.float64)
    durations = {
        video_id: float(ends[video_ids == video_id].max())
        for video_id in sorted(set(video_ids.tolist()))
        if video_id not in excluded_video_ids
    }
    gt: dict[str, list[float]] = {}
    for index, video_id in enumerate(video_ids):
        if video_id in excluded_video_ids or video_id in gt:
            continue
        gt[video_id] = sorted(
            float(value) for value in metadata[index]["online_gt_anchors"][shot_index]
        )
    keep = np.asarray([value not in excluded_video_ids for value in video_ids], dtype=bool)
    teacher = load_teacher_shards(teacher_dir, len(video_ids))
    teacher_labels = teacher["labels"]
    teacher_shot = teacher_labels.index("shot")
    teacher_save = teacher_labels.index("save") if "save" in teacher_labels else None
    offsets = teacher["offsets"]
    centre_index = int(np.argmin(np.abs(offsets)))
    shot_grid = teacher["logits"][:, :, teacher_shot]
    variants = {
        "teacher_shot_center": shot_grid[:, centre_index],
        "teacher_shot_mean": shot_grid.mean(axis=1),
        "teacher_shot_max": shot_grid.max(axis=1),
    }
    if teacher_save is not None:
        variants["teacher_shot_or_save_max"] = np.maximum(
            shot_grid, teacher["logits"][:, :, teacher_save]
        ).max(axis=1)
    return {
        "video_ids": video_ids[keep],
        "candidate_times": candidate_times[keep],
        "baseline_logits": baseline_logits[keep],
        "teacher_variants": {key: value[keep] for key, value in variants.items()},
        "gt": gt,
        "durations": durations,
        "teacher_metadata": {
            "checkpoint": teacher["checkpoint"],
            "labels": teacher_labels,
            "offsets_sec": offsets.tolist(),
            "centre_index": centre_index,
        },
    }


def predictions(data: dict[str, Any], scores: np.ndarray) -> list[dict[str, Any]]:
    return [
        {"video_id": str(video_id), "time_sec": float(time_sec), "score": float(score), "row": index}
        for index, (video_id, time_sec, score) in enumerate(
            zip(data["video_ids"], data["candidate_times"], scores)
        )
    ]


def one_to_one(
    rows: Sequence[dict[str, Any]],
    gt: dict[str, list[float]],
    threshold: float,
    tolerance: float,
) -> dict[str, Any]:
    selected: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if float(row["score"]) >= threshold:
            selected.setdefault(str(row["video_id"]), []).append(row)
    tp = fp = fn = 0
    per_video: dict[str, dict[str, Any]] = {}
    for video_id, targets in gt.items():
        available = list(targets)
        video_tp = video_fp = 0
        for row in sorted(selected.get(video_id, ()), key=lambda item: float(item["score"]), reverse=True):
            eligible = [
                index for index, target in enumerate(available)
                if abs(float(row["time_sec"]) - target) <= tolerance
            ]
            if eligible:
                index = min(eligible, key=lambda value: abs(float(row["time_sec"]) - available[value]))
                available.pop(index)
                tp += 1
                video_tp += 1
            else:
                fp += 1
                video_fp += 1
        video_fn = len(available)
        fn += video_fn
        precision = video_tp / max(video_tp + video_fp, 1)
        recall = video_tp / max(video_tp + video_fn, 1)
        per_video[video_id] = {
            "tp": video_tp, "fp": video_fp, "fn": video_fn,
            "precision": precision, "recall": recall,
        }
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    macro_precision = float(np.mean([value["precision"] for value in per_video.values()]))
    macro_recall = float(np.mean([value["recall"] for value in per_video.values()]))
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "macro_precision": macro_precision, "macro_recall": macro_recall,
        "threshold": float(threshold), "per_video": per_video,
    }


def choose_threshold(
    rows: Sequence[dict[str, Any]],
    gt: dict[str, list[float]],
    recall_floor: float,
    tolerance: float,
) -> tuple[float, dict[str, Any]]:
    values = sorted({float(row["score"]) for row in rows}, reverse=True)
    if not values:
        raise ValueError("no candidate scores")
    evaluated = [one_to_one(rows, gt, value, tolerance) for value in values]
    ceiling = max(float(value["recall"]) for value in evaluated)
    feasible = [value for value in evaluated if float(value["recall"]) + 1e-12 >= recall_floor]
    pool = feasible or [value for value in evaluated if float(value["recall"]) + 1e-12 >= ceiling]
    best = max(
        pool,
        key=lambda value: (
            float(value["precision"]), float(value["f1"]),
            float(value["macro_recall"]), float(value["threshold"]),
        ),
    )
    best["candidate_recall_ceiling"] = ceiling
    best["requested_recall_floor"] = recall_floor
    best["recall_floor_reachable"] = bool(ceiling + 1e-12 >= recall_floor)
    return float(best["threshold"]), best


def merged_duration(intervals: Sequence[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    start, end = ordered[0]
    total = 0.0
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def workload(
    rows: Sequence[dict[str, Any]],
    durations: dict[str, float],
    threshold: float,
    review_sec: float,
) -> dict[str, Any]:
    selected = 0
    union = 0.0
    half = 0.5 * review_sec
    for video_id, duration in durations.items():
        intervals = []
        for row in rows:
            if row["video_id"] != video_id or float(row["score"]) < threshold:
                continue
            selected += 1
            time_sec = float(row["time_sec"])
            intervals.append((max(0.0, time_sec - half), min(duration, time_sec + half)))
        union += merged_duration(intervals)
    total = sum(durations.values())
    return {
        "selected_slots": selected,
        "review_union_minutes": union / 60.0,
        "total_video_minutes": total / 60.0,
        "participation_ratio": union / max(total, 1e-12),
    }


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "per_video"}


def evaluate_scores(
    data: dict[str, Any],
    scores: np.ndarray,
    threshold: float | None,
    recall_floor: float,
    tolerance: float,
    review_sec: float,
) -> tuple[float, dict[str, Any]]:
    rows = predictions(data, scores)
    if threshold is None:
        threshold, metrics = choose_threshold(rows, data["gt"], recall_floor, tolerance)
    else:
        metrics = one_to_one(rows, data["gt"], threshold, tolerance)
    return threshold, {
        "metrics": compact_metrics(metrics),
        "workload": workload(rows, data["durations"], threshold, review_sec),
        "per_video": metrics["per_video"],
    }


def fusion_grid(alpha_values: Iterable[float]) -> list[dict[str, Any]]:
    return [
        {"teacher_variant": variant, "alpha": float(alpha)}
        for variant in (
            "teacher_shot_center", "teacher_shot_mean", "teacher_shot_max",
            "teacher_shot_or_save_max",
        )
        for alpha in alpha_values
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--teacher-shard-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--external-cache", type=Path)
    parser.add_argument("--external-metadata", type=Path)
    parser.add_argument("--external-teacher-shard-dir", type=Path)
    parser.add_argument("--recall-floor", type=float, default=0.90)
    parser.add_argument("--tolerance-sec", type=float, default=3.0)
    parser.add_argument("--review-sec", type=float, default=10.0)
    parser.add_argument("--alpha-grid", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--excluded-video-ids", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    excluded = parse_ids(args.excluded_video_ids)
    calibration = load_dataset(args.cache, args.metadata, args.teacher_shard_dir, excluded)
    base_loc, base_scale = robust_location_scale(calibration["baseline_logits"])
    teacher_stats = {
        key: robust_location_scale(value)
        for key, value in calibration["teacher_variants"].items()
    }
    alpha_values = [float(item) for item in args.alpha_grid.split(",") if item.strip()]
    calibration_results: dict[str, Any] = {}
    thresholds: dict[str, float] = {}

    threshold, result = evaluate_scores(
        calibration, calibration["baseline_logits"], None,
        args.recall_floor, args.tolerance_sec, args.review_sec,
    )
    thresholds["stage1"] = threshold
    calibration_results["stage1"] = result
    for key, values in calibration["teacher_variants"].items():
        threshold, result = evaluate_scores(
            calibration, values, None,
            args.recall_floor, args.tolerance_sec, args.review_sec,
        )
        thresholds[key] = threshold
        calibration_results[key] = result

    base_z = (calibration["baseline_logits"] - base_loc) / base_scale
    fusion_candidates: list[dict[str, Any]] = []
    for spec in fusion_grid(alpha_values):
        variant = spec["teacher_variant"]
        if variant not in calibration["teacher_variants"]:
            continue
        teacher_loc, teacher_scale = teacher_stats[variant]
        teacher_z = (calibration["teacher_variants"][variant] - teacher_loc) / teacher_scale
        alpha = float(spec["alpha"])
        score = (1.0 - alpha) * base_z + alpha * teacher_z
        threshold, result = evaluate_scores(
            calibration, score, None,
            args.recall_floor, args.tolerance_sec, args.review_sec,
        )
        fusion_candidates.append({**spec, "threshold": threshold, **result})
    selected_fusion = max(
        fusion_candidates,
        key=lambda item: (
            float(item["metrics"]["recall"]) + 1e-12 >= args.recall_floor,
            float(item["metrics"]["precision"]),
            -float(item["workload"]["participation_ratio"]),
            float(item["metrics"]["macro_recall"]),
        ),
    )

    report: dict[str, Any] = {
        "protocol": {
            "selection_split": "calibration",
            "external_threshold_retuning": False,
            "no_temporal_nms": True,
            "no_candidate_merge": True,
            "one_to_one_tolerance_sec": args.tolerance_sec,
            "recall_floor": args.recall_floor,
            "review_sec_per_candidate": args.review_sec,
            "excluded_video_ids_predeclared": sorted(excluded),
            "fusion_family": "standardized_logit_convex_blend",
        },
        "teacher": calibration["teacher_metadata"],
        "calibration": {
            "videos": len(calibration["gt"]),
            "shot_gt": sum(len(value) for value in calibration["gt"].values()),
            "normalization": {
                "stage1": {"location": base_loc, "scale": base_scale},
                "teacher": {
                    key: {"location": value[0], "scale": value[1]}
                    for key, value in teacher_stats.items()
                },
            },
            "single_sources": calibration_results,
            "fusion_grid": fusion_candidates,
            "selected_fusion": selected_fusion,
        },
    }

    if args.external_cache is not None:
        if args.external_metadata is None or args.external_teacher_shard_dir is None:
            raise ValueError("external metadata and teacher shard dir are required")
        external = load_dataset(
            args.external_cache, args.external_metadata,
            args.external_teacher_shard_dir, excluded,
        )
        overlap = sorted(set(calibration["gt"]) & set(external["gt"]))
        if overlap:
            raise ValueError(f"calibration/external video overlap: {overlap}")
        external_results: dict[str, Any] = {}
        _, external_results["stage1"] = evaluate_scores(
            external, external["baseline_logits"], thresholds["stage1"],
            args.recall_floor, args.tolerance_sec, args.review_sec,
        )
        for key, values in external["teacher_variants"].items():
            _, external_results[key] = evaluate_scores(
                external, values, thresholds[key],
                args.recall_floor, args.tolerance_sec, args.review_sec,
            )
        variant = selected_fusion["teacher_variant"]
        alpha = float(selected_fusion["alpha"])
        teacher_loc, teacher_scale = teacher_stats[variant]
        external_base_z = (external["baseline_logits"] - base_loc) / base_scale
        external_teacher_z = (external["teacher_variants"][variant] - teacher_loc) / teacher_scale
        external_fusion = (1.0 - alpha) * external_base_z + alpha * external_teacher_z
        _, selected_external = evaluate_scores(
            external, external_fusion, float(selected_fusion["threshold"]),
            args.recall_floor, args.tolerance_sec, args.review_sec,
        )
        report["external_test"] = {
            "videos": len(external["gt"]),
            "shot_gt": sum(len(value) for value in external["gt"].values()),
            "single_sources_fixed_calibration_thresholds": external_results,
            "selected_fusion_fixed": {
                "teacher_variant": variant,
                "alpha": alpha,
                "threshold": float(selected_fusion["threshold"]),
                **selected_external,
            },
        }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
