#!/usr/bin/env python
"""Calibrate/apply an NMS-free football event review policy.

This is the deployable bridge between dense model candidates and a fast human
correction loop.  It deliberately keeps every event candidate independent.
Only *display intervals* are grouped to avoid asking a reviewer to watch the
same seconds repeatedly; grouping never changes prediction counts or event
matching.

Calibration chooses, per class:
  * a fixed Stage-1/complete-clip-teacher score definition;
  * a low boundary satisfying a recall constraint (and, for shot, a
    video-bootstrap lower-tail constraint when reachable);
  * a high boundary for high-precision auto-accept.

The emitted policy contains every normalization statistic and threshold, so an
external split can be evaluated without score or threshold refitting.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


DEFAULT_LABELS = ("shot", "save", "set_piece")


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    positive = values >= 0
    out = np.empty_like(values)
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    return out


def logit(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), eps, 1.0 - eps)
    return np.log(values / (1.0 - values))


def robust_stats(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    location = float(np.median(values))
    q25, q75 = np.quantile(values, (0.25, 0.75))
    return location, max(float(q75 - q25) / 1.349, 1e-3)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_teacher_shards(path: Path, row_count: int) -> dict[str, Any]:
    paths = sorted(path.glob("teacher_rank*_of_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no complete clip teacher shards found in {path}")
    logits: np.ndarray | None = None
    filled = np.zeros(row_count, dtype=bool)
    reference: dict[str, Any] | None = None
    for shard_path in paths:
        shard = np.load(shard_path, allow_pickle=False)
        indices = shard["indices"].astype(np.int64)
        shard_logits = shard["teacher_logits"].astype(np.float32)
        current = {
            "labels": [str(value) for value in shard["labels"]],
            "offsets_sec": shard["offsets"].astype(float).tolist(),
            "checkpoint": str(shard["checkpoint"]),
        }
        if reference is None:
            reference = current
            logits = np.empty((row_count, *shard_logits.shape[1:]), dtype=np.float32)
        elif current != reference:
            raise ValueError(f"incompatible teacher shard: {shard_path}")
        if filled[indices].any():
            raise ValueError(f"duplicate teacher rows in {shard_path}")
        assert logits is not None
        logits[indices] = shard_logits
        filled[indices] = True
    if logits is None or reference is None or not filled.all():
        missing = np.flatnonzero(~filled)[:20].tolist()
        raise ValueError(f"teacher shards do not cover all cache rows; missing={missing}")
    return {**reference, "logits": logits}


def load_dense(cache_path: Path, metadata_path: Path) -> dict[str, Any]:
    cache = np.load(cache_path, allow_pickle=True)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    row_count = len(cache["video_ids"])
    if len(metadata) != row_count:
        raise ValueError("dense cache and metadata row counts disagree")
    labels = [str(value) for value in cache["labels"]]
    video_ids = cache["video_ids"].astype(str)
    ends = cache["clip_ends"].astype(np.float64)
    durations = {
        video_id: float(np.max(ends[video_ids == video_id]))
        for video_id in sorted(set(video_ids.tolist()))
    }
    gt: dict[str, dict[str, list[float]]] = {}
    first_row: dict[str, int] = {}
    for row, video_id in enumerate(video_ids):
        first_row.setdefault(video_id, row)
    for video_id, row in first_row.items():
        anchors = metadata[row]["online_gt_anchors"]
        gt[video_id] = {
            label: sorted(float(value) for value in anchors[index])
            for index, label in enumerate(labels)
        }
    return {
        "cache": cache,
        "labels": labels,
        "video_ids": video_ids,
        "candidate_times": cache["candidate_times"].astype(np.float64),
        "stage1_logits": logit(cache["online_probs"]),
        "clip_starts": cache["clip_starts"].astype(np.float64),
        "clip_ends": ends,
        "gt": gt,
        "durations": durations,
        "metadata": metadata,
    }


def class_gt(data: dict[str, Any], label: str) -> dict[str, list[float]]:
    return {video_id: labels.get(label, []) for video_id, labels in data["gt"].items()}


def candidate_rows(
    data: dict[str, Any], label: str, scores: np.ndarray,
    *, source: str = "visual",
) -> list[dict[str, Any]]:
    label_index = data["labels"].index(label)
    return [
        {
            "candidate_id": f"{source}:{row}:{label}",
            "row": row,
            "video_id": str(data["video_ids"][row]),
            "label": label,
            "time_sec": float(data["candidate_times"][row, label_index]),
            "score": float(scores[row]),
            "source": source,
        }
        for row in range(len(scores))
    ]


def match_selected(
    rows: Sequence[dict[str, Any]], gt: dict[str, list[float]], tolerance_sec: float,
) -> dict[str, Any]:
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_video[str(row["video_id"])].append(row)
    tp = fp = fn = 0
    per_video: dict[str, dict[str, Any]] = {}
    matched_rows: set[str] = set()
    for video_id, targets in gt.items():
        available = list(targets)
        video_tp = video_fp = 0
        for row in sorted(
            by_video.get(video_id, ()),
            key=lambda item: (-float(item["score"]), float(item["time_sec"]), str(item["candidate_id"])),
        ):
            eligible = [
                index for index, target in enumerate(available)
                if abs(float(row["time_sec"]) - target) <= tolerance_sec
            ]
            if eligible:
                best = min(eligible, key=lambda index: abs(float(row["time_sec"]) - available[index]))
                available.pop(best)
                tp += 1
                video_tp += 1
                matched_rows.add(str(row["candidate_id"]))
            else:
                fp += 1
                video_fp += 1
        video_fn = len(available)
        fn += video_fn
        support = video_tp + video_fn
        per_video[video_id] = {
            "tp": video_tp,
            "fp": video_fp,
            "fn": video_fn,
            "support": support,
            "precision": video_tp / max(video_tp + video_fp, 1),
            "recall": video_tp / max(support, 1),
        }
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "macro_recall": float(np.mean([item["recall"] for item in per_video.values()])),
        "macro_precision": float(np.mean([item["precision"] for item in per_video.values()])),
        "per_video": per_video,
        "matched_candidate_ids": matched_rows,
    }


def threshold_grid(scores: np.ndarray, size: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, max(size, 3))
    values = np.unique(np.quantile(scores, quantiles))
    return values[::-1]


def bootstrap_recall_p05(
    per_video: dict[str, dict[str, Any]], *, samples: int, seed: int,
) -> float:
    if samples <= 0:
        return float("nan")
    rows = list(per_video.values())
    tp = np.asarray([row["tp"] for row in rows], dtype=np.float64)
    support = np.asarray([row["support"] for row in rows], dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(rows), size=(samples, len(rows)))
    recall = tp[indices].sum(axis=1) / np.maximum(support[indices].sum(axis=1), 1.0)
    return float(np.quantile(recall, 0.05))


def choose_low_boundary(
    rows: Sequence[dict[str, Any]], gt: dict[str, list[float]], *,
    recall_target: float, bootstrap_floor: float, tolerance_sec: float,
    grid_size: int, bootstrap_samples: int, seed: int,
) -> tuple[float, dict[str, Any]]:
    scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
    evaluated: list[dict[str, Any]] = []
    for threshold in threshold_grid(scores, grid_size):
        metrics = match_selected(
            [row for row in rows if float(row["score"]) >= threshold], gt, tolerance_sec,
        )
        p05 = bootstrap_recall_p05(
            metrics["per_video"], samples=bootstrap_samples, seed=seed,
        )
        evaluated.append({"threshold": float(threshold), "metrics": metrics, "bootstrap_recall_p05": p05})
    ceiling = max(float(item["metrics"]["recall"]) for item in evaluated)
    robust = [
        item for item in evaluated
        if float(item["metrics"]["recall"]) + 1e-12 >= recall_target
        and float(item["bootstrap_recall_p05"]) + 1e-12 >= bootstrap_floor
    ]
    point = [
        item for item in evaluated
        if float(item["metrics"]["recall"]) + 1e-12 >= recall_target
    ]
    pool = robust or point or [
        item for item in evaluated
        if float(item["metrics"]["recall"]) + 1e-12 >= ceiling
    ]
    chosen = max(
        pool,
        key=lambda item: (
            float(item["metrics"]["precision"]),
            float(item["metrics"]["macro_recall"]),
            float(item["threshold"]),
        ),
    )
    summary = compact_metrics(chosen["metrics"])
    summary.update({
        "threshold": chosen["threshold"],
        "bootstrap_recall_p05": chosen["bootstrap_recall_p05"],
        "recall_target": recall_target,
        "bootstrap_floor": bootstrap_floor,
        "candidate_recall_ceiling": ceiling,
        "point_gate_pass": bool(summary["recall"] + 1e-12 >= recall_target),
        "bootstrap_gate_pass": bool(chosen["bootstrap_recall_p05"] + 1e-12 >= bootstrap_floor),
    })
    return float(chosen["threshold"]), summary


def choose_auto_boundary(
    rows: Sequence[dict[str, Any]], gt: dict[str, list[float]], low: float, *,
    precision_floor: float, tolerance_sec: float, grid_size: int,
) -> tuple[float, dict[str, Any]]:
    eligible = [row for row in rows if float(row["score"]) >= low]
    if not eligible:
        return math.inf, {"enabled": False, "reason": "no_selected_candidates"}
    scores = np.asarray([float(row["score"]) for row in eligible])
    feasible: list[tuple[float, dict[str, Any]]] = []
    for threshold in threshold_grid(scores, grid_size):
        metrics = match_selected(
            [row for row in eligible if float(row["score"]) >= threshold], gt, tolerance_sec,
        )
        if metrics["tp"] > 0 and metrics["precision"] + 1e-12 >= precision_floor:
            feasible.append((float(threshold), metrics))
    if not feasible:
        return math.inf, {
            "enabled": False,
            "reason": "precision_floor_unreachable",
            "precision_floor": precision_floor,
        }
    threshold, metrics = max(
        feasible,
        key=lambda item: (item[1]["tp"], item[1]["precision"], -item[0]),
    )
    return threshold, {
        "enabled": True,
        "threshold": threshold,
        "precision_floor": precision_floor,
        **compact_metrics(metrics),
    }


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in metrics.items()
        if key not in {"per_video", "matched_candidate_ids"}
    }


def teacher_variants(
    teacher: dict[str, Any], stage_labels: Sequence[str], label: str,
) -> dict[str, np.ndarray]:
    if label not in teacher["labels"]:
        return {}
    class_index = teacher["labels"].index(label)
    grid = teacher["logits"][:, :, class_index].astype(np.float64)
    offsets = np.asarray(teacher["offsets_sec"], dtype=np.float64)
    center = int(np.argmin(np.abs(offsets)))
    variants = {
        "center": grid[:, center],
        "mean": grid.mean(axis=1),
        "max": grid.max(axis=1),
    }
    if label == "shot" and "save" in teacher["labels"]:
        save_index = teacher["labels"].index("save")
        variants["shot_or_save_max"] = np.maximum(
            grid, teacher["logits"][:, :, save_index]
        ).max(axis=1)
    return variants


def build_calibration_score_candidates(
    data: dict[str, Any], teacher: dict[str, Any] | None, label: str,
    alphas: Iterable[float],
) -> list[dict[str, Any]]:
    index = data["labels"].index(label)
    baseline = data["stage1_logits"][:, index]
    base_location, base_scale = robust_stats(baseline)
    candidates = [{
        "name": "stage1",
        "scores": baseline,
        "definition": {
            "kind": "stage1",
            "stage1_location": base_location,
            "stage1_scale": base_scale,
        },
    }]
    if teacher is None:
        return candidates
    baseline_z = (baseline - base_location) / base_scale
    for variant_name, values in teacher_variants(teacher, data["labels"], label).items():
        teacher_location, teacher_scale = robust_stats(values)
        teacher_z = (values - teacher_location) / teacher_scale
        candidates.append({
            "name": f"teacher_{variant_name}",
            "scores": values,
            "definition": {
                "kind": "teacher",
                "teacher_variant": variant_name,
                "teacher_location": teacher_location,
                "teacher_scale": teacher_scale,
            },
        })
        for alpha in alphas:
            alpha = float(alpha)
            if alpha <= 0.0 or alpha >= 1.0:
                continue
            candidates.append({
                "name": f"blend_{variant_name}_a{alpha:g}",
                "scores": (1.0 - alpha) * baseline_z + alpha * teacher_z,
                "definition": {
                    "kind": "blend",
                    "teacher_variant": variant_name,
                    "alpha": alpha,
                    "stage1_location": base_location,
                    "stage1_scale": base_scale,
                    "teacher_location": teacher_location,
                    "teacher_scale": teacher_scale,
                },
            })
    return candidates


def score_from_definition(
    data: dict[str, Any], teacher: dict[str, Any] | None,
    label: str, definition: dict[str, Any],
) -> np.ndarray:
    index = data["labels"].index(label)
    baseline = data["stage1_logits"][:, index]
    kind = definition["kind"]
    if kind == "stage1":
        return baseline
    if teacher is None:
        raise ValueError(f"policy for {label} requires complete clip teacher features")
    variants = teacher_variants(teacher, data["labels"], label)
    teacher_score = variants[definition["teacher_variant"]]
    if kind == "teacher":
        return teacher_score
    if kind == "blend":
        baseline_z = (
            baseline - float(definition["stage1_location"])
        ) / float(definition["stage1_scale"])
        teacher_z = (
            teacher_score - float(definition["teacher_location"])
        ) / float(definition["teacher_scale"])
        alpha = float(definition["alpha"])
        return (1.0 - alpha) * baseline_z + alpha * teacher_z
    raise ValueError(f"unknown score definition kind: {kind}")


def intervals_union_duration(intervals: Sequence[tuple[float, float]]) -> float:
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


def make_review_segments(
    review_rows: Sequence[dict[str, Any]], durations: dict[str, float], *,
    review_sec: float, max_segment_sec: float,
) -> list[dict[str, Any]]:
    half = 0.5 * review_sec
    by_video: dict[str, list[tuple[float, float, dict[str, Any]]]] = defaultdict(list)
    for row in review_rows:
        video_id = str(row["video_id"])
        time_sec = float(row["time_sec"])
        by_video[video_id].append((
            max(0.0, time_sec - half),
            min(float(durations[video_id]), time_sec + half),
            row,
        ))
    result: list[dict[str, Any]] = []
    for video_id, items in sorted(by_video.items()):
        current: list[tuple[float, float, dict[str, Any]]] = []
        current_start = current_end = 0.0
        groups: list[list[tuple[float, float, dict[str, Any]]]] = []
        for start, end, row in sorted(items, key=lambda item: (item[0], item[1])):
            can_merge = (
                current
                and start <= current_end
                and max(current_end, end) - current_start <= max_segment_sec
            )
            if not current or can_merge:
                if not current:
                    current_start, current_end = start, end
                else:
                    current_end = max(current_end, end)
                current.append((start, end, row))
            else:
                groups.append(current)
                current = [(start, end, row)]
                current_start, current_end = start, end
        if current:
            groups.append(current)
        for video_index, group in enumerate(groups):
            start = min(item[0] for item in group)
            end = max(item[1] for item in group)
            candidates = [item[2] for item in group]
            result.append({
                "segment_id": f"{video_id}:{video_index:05d}",
                "video_id": video_id,
                "segment_index": video_index,
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "duration_sec": round(end - start, 3),
                "candidate_count": len(candidates),
                "candidate_ids": ";".join(str(item["candidate_id"]) for item in candidates),
                "predicted_labels": ";".join(sorted({str(item["label"]) for item in candidates})),
                "sources": ";".join(sorted({str(item["source"]) for item in candidates})),
                "max_score": max(float(item["score"]) for item in candidates),
            })
    return result


def package_split(
    data: dict[str, Any], teacher: dict[str, Any] | None,
    policy: dict[str, Any], output: Path, *, split_name: str,
    review_sec: float, max_segment_sec: float, tolerance_sec: float,
) -> dict[str, Any]:
    prediction_rows: list[dict[str, Any]] = []
    selected_by_label: dict[str, list[dict[str, Any]]] = {}
    review_rows: list[dict[str, Any]] = []
    metrics_by_label: dict[str, Any] = {}
    for label, item in policy["labels"].items():
        scores = score_from_definition(data, teacher, label, item["score_definition"])
        rows = candidate_rows(data, label, scores)
        low = float(item["review_boundary"])
        auto = float(item["auto_accept_boundary"])
        selected: list[dict[str, Any]] = []
        for row in rows:
            score = float(row["score"])
            if score >= auto:
                decision = "auto_accept"
            elif score >= low:
                decision = "review"
            else:
                decision = "reject"
            row["decision"] = decision
            row["review_boundary"] = low
            row["auto_accept_boundary"] = auto
            if decision != "reject":
                selected.append(row)
                prediction_rows.append(row)
            if decision == "review":
                review_rows.append(row)
        selected_by_label[label] = selected
        metrics = match_selected(selected, class_gt(data, label), tolerance_sec)
        metrics_by_label[label] = {
            **compact_metrics(metrics),
            "per_video": metrics["per_video"],
            "selected_candidates": len(selected),
            "auto_accept_candidates": sum(row["decision"] == "auto_accept" for row in selected),
            "review_candidates": sum(row["decision"] == "review" for row in selected),
        }

    segments = make_review_segments(
        review_rows, data["durations"], review_sec=review_sec,
        max_segment_sec=max_segment_sec,
    )
    intervals_by_video: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for segment in segments:
        intervals_by_video[str(segment["video_id"])].append(
            (float(segment["start_sec"]), float(segment["end_sec"]))
        )
    union_sec = sum(intervals_union_duration(value) for value in intervals_by_video.values())
    total_sec = sum(data["durations"].values())
    dense_windows = len(data["video_ids"])
    workload = {
        "dense_windows": dense_windows,
        "review_candidates": len(review_rows),
        "review_candidate_ratio_vs_dense_windows": len(review_rows) / max(dense_windows, 1),
        "review_segments": len(segments),
        "review_segment_ratio_vs_dense_windows": len(segments) / max(dense_windows, 1),
        "review_union_minutes": union_sec / 60.0,
        "total_video_minutes": total_sec / 60.0,
        "review_time_ratio": union_sec / max(total_sec, 1e-12),
        "review_segment_count_gate_lt_35pct": len(segments) / max(dense_windows, 1) < 0.35,
        "review_candidate_count_gate_lt_35pct": len(review_rows) / max(dense_windows, 1) < 0.35,
        "review_time_gate_lt_35pct": union_sec / max(total_sec, 1e-12) < 0.35,
        "display_interval_grouping_only": True,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "predictions_selected.csv", prediction_rows)
    write_csv(output / "review_segments.csv", segments)
    annotation_rows = [
        {
            **segment,
            "review_status": "",
            "correct_labels": "",
            "correct_event_times_sec": "",
            "confounder": "",
            "notes": "",
        }
        for segment in segments
    ]
    write_csv(output / "review_annotations.csv", annotation_rows)
    report = {
        "split": split_name,
        "protocol": {
            "no_temporal_nms": True,
            "no_prediction_merge": True,
            "one_to_one_tolerance_sec": tolerance_sec,
            "display_interval_grouping_is_not_prediction_postprocessing": True,
        },
        "videos": len(data["durations"]),
        "metrics": metrics_by_label,
        "workload": workload,
        "artifacts": {
            "predictions": str(output / "predictions_selected.csv"),
            "review_segments": str(output / "review_segments.csv"),
            "review_annotations": str(output / "review_annotations.csv"),
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def calibrate_policy(
    data: dict[str, Any], teacher: dict[str, Any] | None, *,
    labels: Sequence[str], recall_targets: dict[str, float],
    bootstrap_floors: dict[str, float], auto_precision_floor: float,
    tolerance_sec: float, grid_size: int, bootstrap_samples: int,
    alphas: Sequence[float], seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    policy_labels: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    for class_offset, label in enumerate(labels):
        if label not in data["labels"]:
            continue
        source_results: list[dict[str, Any]] = []
        candidates = build_calibration_score_candidates(data, teacher, label, alphas)
        for candidate in candidates:
            rows = candidate_rows(data, label, candidate["scores"])
            threshold, metrics = choose_low_boundary(
                rows, class_gt(data, label), recall_target=recall_targets[label],
                bootstrap_floor=bootstrap_floors[label], tolerance_sec=tolerance_sec,
                grid_size=grid_size, bootstrap_samples=bootstrap_samples,
                seed=seed + class_offset,
            )
            source_results.append({
                "name": candidate["name"],
                "score_definition": candidate["definition"],
                "review_boundary": threshold,
                "metrics": metrics,
                "scores": candidate["scores"],
            })
        robust = [
            item for item in source_results
            if item["metrics"]["point_gate_pass"] and item["metrics"]["bootstrap_gate_pass"]
        ]
        point = [item for item in source_results if item["metrics"]["point_gate_pass"]]
        pool = robust or point or source_results
        chosen = max(
            pool,
            key=lambda item: (
                float(item["metrics"]["precision"]),
                -float(item["metrics"]["fp"]),
                float(item["metrics"]["macro_recall"]),
            ),
        )
        chosen_rows = candidate_rows(data, label, chosen["scores"])
        auto_boundary, auto_metrics = choose_auto_boundary(
            chosen_rows, class_gt(data, label), float(chosen["review_boundary"]),
            precision_floor=auto_precision_floor, tolerance_sec=tolerance_sec,
            grid_size=grid_size,
        )
        policy_labels[label] = {
            "score_name": chosen["name"],
            "score_definition": chosen["score_definition"],
            "review_boundary": float(chosen["review_boundary"]),
            "auto_accept_boundary": float(auto_boundary),
            "calibration_metrics": chosen["metrics"],
            "auto_accept_calibration": auto_metrics,
        }
        diagnostics[label] = [
            {
                "name": item["name"],
                "score_definition": item["score_definition"],
                "review_boundary": item["review_boundary"],
                "metrics": item["metrics"],
            }
            for item in source_results
        ]
    policy = {
        "schema": "football_goal_oriented_review_policy.v1",
        "labels": policy_labels,
        "selection_split": "calibration18",
        "external_threshold_retuning": False,
        "no_temporal_nms": True,
        "tolerance_sec": tolerance_sec,
        "recall_targets": recall_targets,
        "bootstrap_recall_floors": bootstrap_floors,
        "auto_accept_precision_floor": auto_precision_floor,
        "teacher": None if teacher is None else {
            key: teacher[key] for key in ("checkpoint", "labels", "offsets_sec")
        },
    }
    return policy, diagnostics


def parse_mapping(raw: str, labels: Sequence[str], default: float) -> dict[str, float]:
    result = {label: default for label in labels}
    for item in raw.split(","):
        if not item.strip():
            continue
        label, value = item.split("=", 1)
        if label.strip() not in result:
            raise ValueError(f"unknown label in mapping: {label}")
        result[label.strip()] = float(value)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--teacher-shard-dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--policy", type=Path, help="Apply an existing policy instead of calibrating")
    parser.add_argument("--split-name", default="calibration18")
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--recall-targets", default="shot=0.93,save=0.88,set_piece=0.88")
    parser.add_argument("--bootstrap-floors", default="shot=0.90,save=0.85,set_piece=0.85")
    parser.add_argument("--auto-precision-floor", type=float, default=0.80)
    parser.add_argument("--tolerance-sec", type=float, default=3.0)
    parser.add_argument("--review-sec", type=float, default=10.0)
    parser.add_argument("--max-review-segment-sec", type=float, default=20.0)
    parser.add_argument("--threshold-grid-size", type=int, default=401)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--alpha-grid", default="0.15,0.25,0.35,0.5,0.65,0.75,0.85")
    parser.add_argument("--seed", type=int, default=20260830)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = [item.strip() for item in args.labels.split(",") if item.strip()]
    data = load_dense(args.cache, args.metadata)
    teacher = (
        load_teacher_shards(args.teacher_shard_dir, len(data["video_ids"]))
        if args.teacher_shard_dir is not None else None
    )
    if args.policy is None:
        recall_targets = parse_mapping(args.recall_targets, labels, 0.85)
        bootstrap_floors = parse_mapping(args.bootstrap_floors, labels, 0.85)
        alphas = [float(item) for item in args.alpha_grid.split(",") if item.strip()]
        policy, diagnostics = calibrate_policy(
            data, teacher, labels=labels, recall_targets=recall_targets,
            bootstrap_floors=bootstrap_floors,
            auto_precision_floor=args.auto_precision_floor,
            tolerance_sec=args.tolerance_sec, grid_size=args.threshold_grid_size,
            bootstrap_samples=args.bootstrap_samples, alphas=alphas, seed=args.seed,
        )
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "policy.json").write_text(
            json.dumps(policy, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (args.output / "calibration_diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    else:
        policy = json.loads(args.policy.read_text(encoding="utf-8"))
        if policy.get("schema") != "football_goal_oriented_review_policy.v1":
            raise ValueError("unsupported policy schema")
    report = package_split(
        data, teacher, policy, args.output, split_name=args.split_name,
        review_sec=args.review_sec, max_segment_sec=args.max_review_segment_sec,
        tolerance_sec=args.tolerance_sec,
    )
    summary = {
        "policy": str(args.output / "policy.json") if args.policy is None else str(args.policy),
        "report": str(args.output / "report.json"),
        "metrics": {
            label: {
                key: value for key, value in metrics.items()
                if key in {"precision", "recall", "f1", "tp", "fp", "fn", "selected_candidates", "review_candidates"}
            }
            for label, metrics in report["metrics"].items()
        },
        "workload": report["workload"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
