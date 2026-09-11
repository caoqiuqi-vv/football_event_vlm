"""Video-adaptive, label-free score calibration for long-context retrieval."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .goal_annotations import LABELS, load_goal_annotations
from .goal_retriever_eval import choose_threshold, intervals_union_seconds


def robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    median = float(np.median(values)) if len(values) else 0.0
    mad = float(np.median(np.abs(values - median))) if len(values) else 1.0
    return median, max(1.4826 * mad, 0.25)


def logits(scores: np.ndarray) -> np.ndarray:
    values = np.clip(scores.astype(np.float64), 1e-7, 1.0 - 1e-7)
    return np.log(values / (1.0 - values))


def adaptive_predictions(predictions: list[dict], alpha: float, global_median: float, global_scale: float) -> list[dict]:
    output = []
    by_video: dict[str, list[int]] = {}
    for index, row in enumerate(predictions):
        by_video.setdefault(str(row["video_id"]), []).append(index)
    raw_logits = logits(np.asarray([row["score"] for row in predictions], dtype=np.float64))
    global_z = (raw_logits - global_median) / global_scale
    adaptive = global_z.copy()
    for indices in by_video.values():
        indices_array = np.asarray(indices, dtype=np.int64)
        values = raw_logits[indices_array]
        median, scale = robust_location_scale(values)
        video_z = (values - median) / scale
        order = np.argsort(np.argsort(values, kind="stable"), kind="stable")
        percentile = (order + 0.5) / max(len(order), 1)
        rank_logit = np.log(np.clip(percentile, 1e-4, 1 - 1e-4) / np.clip(1 - percentile, 1e-4, 1))
        local = 0.7 * video_z + 0.3 * rank_logit
        adaptive[indices_array] = (1.0 - alpha) * global_z[indices_array] + alpha * local
    for row, score in zip(predictions, adaptive):
        output.append({**row, "raw_score": float(row["score"]), "score": float(score)})
    return output


def evaluate_predictions_adaptive(
    predictions: list[dict],
    *,
    manifest: str | Path,
    split: str,
    durations: dict[str, float],
    recall_floors: dict[str, float] | None = None,
) -> dict:
    recall_floors = recall_floors or {label: (0.97 if label == "shot" else 0.94) for label in LABELS}
    payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    items = {str(item["media_id"]): item for item in payload[split] if str(item["media_id"]) in durations}
    gt = {label: {} for label in LABELS}
    for video_id, item in items.items():
        events = load_goal_annotations(item["annotation_path"])
        for label in LABELS:
            gt[label][video_id] = [event.timestamp for event in events if event.accepted and event.label == label]
    selected, retained, searches = {}, [], {}
    for label in LABELS:
        class_predictions = [row for row in predictions if row["label"] == label]
        raw_logit = logits(np.asarray([row["score"] for row in class_predictions], dtype=np.float64))
        global_median, global_scale = robust_location_scale(raw_logit)
        rows = []
        transformed_by_alpha = {}
        for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
            transformed = adaptive_predictions(class_predictions, alpha, global_median, global_scale)
            transformed_by_alpha[alpha] = transformed
            metric = choose_threshold(transformed, gt[label], recall_floors[label])
            rows.append({**metric, "adaptive_alpha": alpha, "global_logit_median": global_median, "global_logit_scale": global_scale})
        chosen = max(rows, key=lambda row: (
            bool(row["recall_floor_reachable"]), float(row["precision"]),
            float(row["recall"]), float(row["threshold"]), -abs(float(row["adaptive_alpha"]) - 0.5),
        ))
        selected[label] = chosen; searches[label] = rows
        retained.extend(row for row in transformed_by_alpha[chosen["adaptive_alpha"]] if row["score"] >= chosen["threshold"])
    intervals = {video_id: [] for video_id in durations}
    for row in retained:
        half = 0.5 * float(row["duration"]); duration = durations[row["video_id"]]
        intervals[row["video_id"]].append((max(0.0, row["time"] - half), min(duration, row["time"] + half)))
    reviewed = {video_id: intervals_union_seconds(values) for video_id, values in intervals.items()}
    total_duration = sum(durations.values()); review_seconds = sum(reviewed.values())
    return {
        "protocol": {
            "no_nms": True, "one_to_one_tolerance_sec": 3.0, "review_interval_union": True,
            "video_adaptation_uses_labels": False, "adaptive_alpha_selected_on_calibration_only": True,
        },
        "classes": selected, "adaptive_search": searches,
        "retained_candidates": len(retained), "review_seconds": review_seconds,
        "total_video_seconds": total_duration, "review_ratio": review_seconds / max(total_duration, 1e-9),
        "per_video_review_ratio": {video_id: reviewed[video_id] / max(duration, 1e-9) for video_id, duration in durations.items()},
        "gate_pass": all(selected[label]["recall_floor_reachable"] for label in LABELS) and review_seconds / max(total_duration, 1e-9) < 0.35,
    }

