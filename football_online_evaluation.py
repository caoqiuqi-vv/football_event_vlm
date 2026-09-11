"""Fast event-level threshold tuning for fixed-stride football scans.

The expensive model forward is performed once.  Greedy temporal NMS is also
computed once per video/class because lowering a threshold only exposes a
prefix of the score-sorted candidate list.  Threshold search then operates on
the cached NMS peaks and strict one-to-one event matches.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np


def _nms(candidates: list[tuple[float, float]], radius: float) -> list[tuple[float, float]]:
    kept: list[tuple[float, float]] = []
    for score, timestamp in sorted(candidates, key=lambda item: item[0], reverse=True):
        if any(abs(timestamp - other_time) <= radius for _, other_time in kept):
            continue
        kept.append((score, timestamp))
    return kept


def _metrics(
    video_peaks: dict[tuple[str, str], list[tuple[float, float]]],
    video_gt: dict[tuple[str, str], set[float]],
    threshold: float,
    tolerance: float,
) -> dict[str, float | int]:
    tp = fp = fn = predictions = 0
    for key in set(video_peaks) | set(video_gt):
        unmatched = set(video_gt.get(key, set()))
        selected = [item for item in video_peaks.get(key, ()) if item[0] >= threshold]
        predictions += len(selected)
        for _score, timestamp in selected:
            eligible = [value for value in unmatched if abs(timestamp - value) <= tolerance]
            if eligible:
                unmatched.remove(min(eligible, key=lambda value: abs(timestamp - value)))
                tp += 1
            else:
                fp += 1
        fn += len(unmatched)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "predictions_after_nms": predictions,
    }


def tune_online_event_thresholds(
    probs: np.ndarray,
    candidate_times: np.ndarray,
    metas: Sequence[dict[str, Any]],
    labels: Sequence[str],
    objectives: Sequence[str],
    min_recalls: Sequence[float | None],
    masks: np.ndarray | None = None,
    *,
    nms_radius_sec: float,
    tolerance_sec: float,
    fbeta_beta: float = 1.0,
    max_candidates: int = 401,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Tune thresholds on strict event metrics, never window labels."""
    if probs.shape != candidate_times.shape or probs.shape[0] != len(metas):
        raise ValueError("online threshold tuning inputs have inconsistent shapes")
    if probs.shape[1] != len(labels):
        raise ValueError("online threshold tuning class count mismatch")
    if len(objectives) != len(labels) or len(min_recalls) != len(labels):
        raise ValueError("online threshold policy must have one item per class")
    if masks is not None and masks.shape != probs.shape:
        raise ValueError("online threshold masks must match probabilities")

    beta_sq = max(float(fbeta_beta), 1e-6) ** 2
    thresholds = np.full(len(labels), 0.5, dtype=np.float32)
    diagnostics: dict[str, Any] = {}
    for class_index, label in enumerate(labels):
        raw_candidates: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
        video_gt: dict[tuple[str, str], set[float]] = defaultdict(set)
        video_complete: dict[tuple[str, str], bool] = {}
        for row_index, meta in enumerate(metas):
            key = (str(meta.get("source", "")), str(meta.get("video_id", "")))
            video_complete.setdefault(key, True)
            if masks is not None and float(masks[row_index, class_index]) <= 0.5:
                video_complete[key] = False
        complete_videos = {key for key, complete in video_complete.items() if complete}
        partial_videos = set(video_complete) - complete_videos
        for row_index, meta in enumerate(metas):
            key = (str(meta.get("source", "")), str(meta.get("video_id", "")))
            if key not in complete_videos:
                continue
            raw_candidates[key].append(
                (float(probs[row_index, class_index]), float(candidate_times[row_index, class_index]))
            )
            anchors = meta.get("online_gt_anchors", ()) or ()
            if len(anchors) == len(labels):
                video_gt[key].update(round(float(value), 4) for value in anchors[class_index])
        video_peaks = {
            key: _nms(values, float(nms_radius_sec))
            for key, values in raw_candidates.items()
        }
        support = int(sum(len(values) for values in video_gt.values()))
        if support == 0:
            thresholds[class_index] = 1.0
            diagnostics[label] = {
                "reason": "no_ground_truth",
                "threshold": 1.0,
                "support": 0,
            }
            continue
        unique_scores = np.asarray(
            sorted({score for values in video_peaks.values() for score, _ in values}, reverse=True),
            dtype=np.float64,
        )
        if unique_scores.size == 0:
            diagnostics[label] = {"reason": "no_predictions"}
            continue
        cap = max(int(max_candidates), 3)
        if unique_scores.size > cap:
            indices = np.unique(np.linspace(0, unique_scores.size - 1, cap).round().astype(int))
            search_scores = unique_scores[indices]
        else:
            search_scores = unique_scores
        candidates = [
            (float(threshold), _metrics(video_peaks, video_gt, float(threshold), float(tolerance_sec)))
            for threshold in search_scores
        ]
        max_recall = max(float(item[1]["recall"]) for item in candidates)
        floor = min_recalls[class_index]
        eligible = [item for item in candidates if floor is None or item[1]["recall"] + 1e-12 >= float(floor)]
        floor_reachable = bool(eligible)
        if not eligible:
            eligible = [item for item in candidates if abs(float(item[1]["recall"]) - max_recall) <= 1e-12]
        objective = str(objectives[class_index]).strip().lower()

        def key(item: tuple[float, dict[str, float | int]]) -> tuple[float, ...]:
            threshold, metric = item
            precision = float(metric["precision"])
            recall = float(metric["recall"])
            f1 = float(metric["f1"])
            fbeta = (1.0 + beta_sq) * precision * recall / max(beta_sq * precision + recall, 1e-12)
            if objective == "precision":
                return precision, f1, recall, threshold
            if objective == "fbeta":
                return fbeta, recall, precision, threshold
            return f1, recall, precision, threshold

        selected_threshold, selected = max(eligible, key=key)
        thresholds[class_index] = float(selected_threshold)
        diagnostics[label] = {
            **selected,
            "threshold": float(selected_threshold),
            "objective": objective,
            "min_recall": None if floor is None else float(floor),
            "recall_floor_reachable": floor_reachable,
            "candidate_recall_ceiling": max_recall,
            "threshold_candidates": int(len(search_scores)),
            "nms_peaks": int(sum(len(values) for values in video_peaks.values())),
            "support": support,
            "complete_video_count": len(complete_videos),
            "partial_video_count": len(partial_videos),
            "partial_video_ids": sorted(key[1] for key in partial_videos),
        }
    return thresholds, diagnostics
