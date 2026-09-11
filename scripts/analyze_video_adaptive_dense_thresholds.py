#!/usr/bin/env python
"""Compare fixed, per-video oracle, and leakage-free adaptive dense thresholds.

The metrics intentionally use the same window-overlap protocol as the dense
evaluation scripts.  Oracle thresholds read each video's GT and are only an
upper bound.  The adaptive protocol calibrates a score against the video's
median logit and learns both the calibration strength and threshold on all
other videos (leave-one-video-out).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

from analyze_dense_pr_ceiling import (
    VideoLabelScores,
    combine_metrics,
    evaluate_data,
    finalize_counts,
    load_run_scores,
    thresholds_from_summaries,
)


def _metric(item: VideoLabelScores, scores: list[float], threshold: float) -> dict[str, Any]:
    tp = fp = 0
    matched: set[int] = set()
    for score, matches in zip(scores, item.matched_gt_indices):
        if score < threshold:
            continue
        if matches:
            tp += 1
            matched.update(matches)
        else:
            fp += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = len(matched) / item.num_gt if item.num_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": item.num_gt - len(matched),
        "num_pred": tp + fp,
        "num_gt": item.num_gt,
        "num_matched_gt": len(matched),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _logit(probability: float) -> float:
    p = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _adaptive_scores(item: VideoLabelScores, alpha: float) -> tuple[list[float], float]:
    logits = [_logit(score) for score in item.scores]
    location = statistics.median(logits) if logits else 0.0
    return [score - alpha * location for score in logits], location


def _aggregate(
    rows: Iterable[tuple[VideoLabelScores, list[float]]], threshold: float
) -> dict[str, Any]:
    return combine_metrics(_metric(item, scores, threshold) for item, scores in rows)


def _candidate_thresholds(score_lists: Iterable[list[float]]) -> list[float]:
    values = sorted({value for scores in score_lists for value in scores}, reverse=True)
    if not values:
        return [float("inf")]
    return [values[0] + 1e-9, *values]


def _select(
    rows: list[tuple[VideoLabelScores, list[float]]], recall_floor: float
) -> tuple[float, dict[str, Any], bool]:
    # Exact incremental sweep.  The previous implementation recomputed every
    # video for every unique score (O(N^2)); sorting once preserves identical
    # threshold candidates and tie-breaking in O(N log N).
    ranked: list[tuple[float, int, tuple[int, ...]]] = []
    num_gt = 0
    for row_index, (item, scores) in enumerate(rows):
        num_gt += int(item.num_gt)
        ranked.extend(
            (float(score), row_index, matches)
            for score, matches in zip(scores, item.matched_gt_indices)
        )
    ranked.sort(key=lambda entry: entry[0], reverse=True)
    initial_threshold = ranked[0][0] + 1e-9 if ranked else float("inf")
    evaluated: list[tuple[float, dict[str, Any]]] = [
        (
            initial_threshold,
            finalize_counts(tp=0, fp=0, num_gt=num_gt, num_matched_gt=0),
        )
    ]
    tp = fp = 0
    matched_gt: set[tuple[int, int]] = set()
    offset = 0
    while offset < len(ranked):
        threshold = ranked[offset][0]
        end = offset
        while end < len(ranked) and ranked[end][0] == threshold:
            _, row_index, matches = ranked[end]
            if matches:
                tp += 1
                matched_gt.update((row_index, gt_index) for gt_index in matches)
            else:
                fp += 1
            end += 1
        evaluated.append(
            (
                threshold,
                finalize_counts(
                    tp=tp,
                    fp=fp,
                    num_gt=num_gt,
                    num_matched_gt=len(matched_gt),
                ),
            )
        )
        offset = end
    feasible = [row for row in evaluated if row[1]["recall"] + 1e-12 >= recall_floor]
    if feasible:
        threshold, metric = max(
            feasible,
            key=lambda row: (
                row[1]["precision"], row[1]["recall"], row[1]["f1"],
                -row[1]["num_pred"], row[0],
            ),
        )
        return threshold, metric, True
    threshold, metric = max(
        evaluated,
        key=lambda row: (
            row[1]["recall"], row[1]["precision"], row[1]["f1"],
            -row[1]["num_pred"], row[0],
        ),
    )
    return threshold, metric, False


def _oracle(item: VideoLabelScores, recall_floor: float) -> dict[str, Any]:
    if item.num_gt == 0:
        threshold = max(item.scores, default=1.0) + 1e-9
        return {"threshold": threshold, "recall_floor_feasible": True, **evaluate_data(item, threshold)}
    threshold, metric, feasible = _select([(item, list(item.scores))], recall_floor)
    return {"threshold": threshold, "recall_floor_feasible": feasible, **metric}


def _parse_floors(raw: str, labels: list[str]) -> dict[str, float]:
    result = {"shot": 0.90, "save": 0.85, "set_piece": 0.85}
    for item in raw.split(","):
        if not item.strip():
            continue
        label, value = item.split("=", 1)
        result[label.strip()] = float(value)
    return {label: result.get(label, 0.85) for label in labels}


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir)
    labels = [item.strip() for item in args.labels.split(",") if item.strip()]
    requested_ids = [
        line.strip() for line in Path(args.video_id_file).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ] if args.video_id_file else None
    data, video_ids = load_run_scores(
        run_dir, labels=labels, branch="fused", tolerance_sec=args.match_tolerance_sec,
        exclusions=set(), video_ids=requested_ids,
    )
    floors = _parse_floors(args.recall_floors, labels)
    checkpoint_thresholds = thresholds_from_summaries(run_dir, video_ids, labels)
    by_label = {label: [item for item in data if item.label == label] for label in labels}
    alphas = [args.alpha_min + index * args.alpha_step
              for index in range(round((args.alpha_max - args.alpha_min) / args.alpha_step) + 1)]
    output: dict[str, Any] = {
        "protocol": "window_overlap_many_predictions_per_gt",
        "match_tolerance_sec": args.match_tolerance_sec,
        "warning": "per_video_oracle reads held-out GT and is an upper bound, not a deployable test metric",
        "video_ids": video_ids,
        "recall_floors": floors,
        "per_class": {},
    }
    for label in labels:
        items = by_label[label]
        fixed_rows = [evaluate_data(item, checkpoint_thresholds[label]) for item in items]
        oracle_details = [
            {"video_id": item.video_id, **_oracle(item, floors[label])} for item in items
        ]
        folds: list[dict[str, Any]] = []
        for held_out in items:
            train = [item for item in items if item.video_id != held_out.video_id]
            best: tuple[tuple[float, ...], float, float, dict[str, Any], bool] | None = None
            for alpha in alphas:
                train_rows = [(item, _adaptive_scores(item, alpha)[0]) for item in train]
                threshold, metric, feasible = _select(train_rows, floors[label])
                key = (
                    float(feasible), metric["precision"], metric["recall"], metric["f1"],
                    -metric["num_pred"], -abs(alpha),
                )
                if best is None or key > best[0]:
                    best = (key, alpha, threshold, metric, feasible)
            assert best is not None
            _, alpha, threshold, train_metric, feasible = best
            held_scores, held_location = _adaptive_scores(held_out, alpha)
            folds.append({
                "held_out_video_id": held_out.video_id,
                "alpha": alpha,
                "video_median_logit": held_location,
                "adaptive_threshold": threshold,
                "train_recall_floor_feasible": feasible,
                "train_metric": train_metric,
                "test_metric": _metric(held_out, held_scores, threshold),
            })
        output["per_class"][label] = {
            "fixed_checkpoint_threshold": checkpoint_thresholds[label],
            "fixed_global": combine_metrics(fixed_rows),
            "per_video_oracle": {
                "aggregate": combine_metrics(oracle_details),
                "details": oracle_details,
            },
            "adaptive_median_logit_loov": {
                "description": "score=logit(prob)-alpha*median_video_logit; alpha and threshold learned on N-1 videos",
                "aggregate": combine_metrics(fold["test_metric"] for fold in folds),
                "folds": folds,
            },
        }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--video-id-file", default="")
    parser.add_argument("--labels", default="shot,save,set_piece")
    parser.add_argument("--match-tolerance-sec", type=float, default=3.0)
    parser.add_argument("--recall-floors", default="shot=0.90,save=0.85,set_piece=0.85")
    parser.add_argument("--alpha-min", type=float, default=-0.5)
    parser.add_argument("--alpha-max", type=float, default=1.5)
    parser.add_argument("--alpha-step", type=float, default=0.1)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        label: {
            strategy: values["aggregate"] if isinstance(values, dict) and "aggregate" in values else values
            for strategy, values in result.items()
            if strategy in {"fixed_global", "per_video_oracle", "adaptive_median_logit_loov"}
        }
        for label, result in report["per_class"].items()
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
