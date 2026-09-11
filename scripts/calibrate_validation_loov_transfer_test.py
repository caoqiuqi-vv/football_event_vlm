#!/usr/bin/env python3
"""Select video-adaptive thresholds on validation and freeze them for test.

The calibration set is used twice, for two distinct purposes:

1. Leave-one-video-out predictions select the median-logit adaptation strength
   ``alpha`` without evaluating that choice in-sample.
2. After alpha is fixed, all calibration videos fit one final score threshold.

The resulting (alpha, threshold) pair is frozen.  A test video's labels are
never used to select either value; only its unlabeled score median is used at
inference time.  Test GT is read solely to report the final metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

from analyze_dense_pr_ceiling import VideoLabelScores, combine_metrics, load_run_scores
from analyze_video_adaptive_dense_thresholds import (
    _adaptive_scores,
    _metric,
    _parse_floors,
    _select,
)


def _read_ids(path: str) -> list[str]:
    result = [
        line.strip()
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(result) != len(set(result)):
        raise ValueError(f"Duplicate video ids in {path}")
    return result


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _per_video_metrics(
    items: Sequence[VideoLabelScores], alpha: float, threshold: float
) -> list[dict[str, Any]]:
    result = []
    for item in items:
        scores, median_logit = _adaptive_scores(item, alpha)
        result.append(
            {
                "video_id": item.video_id,
                "video_median_logit": median_logit,
                "effective_probability_threshold": _sigmoid(
                    threshold + alpha * median_logit
                ),
                **_metric(item, scores, threshold),
            }
        )
    return result


def _alpha_loov(
    items: Sequence[VideoLabelScores],
    alphas: Sequence[float],
    recall_floor: float,
) -> tuple[float, list[dict[str, Any]], dict[str, Any], bool, list[dict[str, Any]]]:
    """Select one alpha using pooled held-out predictions on validation."""
    candidates: list[dict[str, Any]] = []
    for alpha in alphas:
        folds: list[dict[str, Any]] = []
        for held_out in items:
            train = [item for item in items if item.video_id != held_out.video_id]
            train_rows = [(item, _adaptive_scores(item, alpha)[0]) for item in train]
            threshold, train_metric, train_feasible = _select(train_rows, recall_floor)
            held_scores, held_location = _adaptive_scores(held_out, alpha)
            folds.append(
                {
                    "held_out_video_id": held_out.video_id,
                    "threshold_fit_on_n_minus_1": threshold,
                    "held_out_video_median_logit": held_location,
                    "train_recall_floor_feasible": train_feasible,
                    "train_metric": train_metric,
                    "held_out_metric": _metric(held_out, held_scores, threshold),
                }
            )
        aggregate = combine_metrics(fold["held_out_metric"] for fold in folds)
        candidates.append(
            {
                "alpha": alpha,
                "recall_floor_feasible": aggregate["recall"] + 1e-12 >= recall_floor,
                "aggregate": aggregate,
                "folds": folds,
            }
        )

    feasible = [row for row in candidates if row["recall_floor_feasible"]]
    if feasible:
        best = max(
            feasible,
            key=lambda row: (
                row["aggregate"]["precision"],
                row["aggregate"]["recall"],
                row["aggregate"]["f1"],
                -row["aggregate"]["num_pred"],
                -abs(row["alpha"]),
            ),
        )
        selection_feasible = True
    else:
        best = max(
            candidates,
            key=lambda row: (
                row["aggregate"]["recall"],
                row["aggregate"]["precision"],
                row["aggregate"]["f1"],
                -row["aggregate"]["num_pred"],
                -abs(row["alpha"]),
            ),
        )
        selection_feasible = False
    return (
        float(best["alpha"]),
        best["folds"],
        best["aggregate"],
        selection_feasible,
        [
            {
                "alpha": row["alpha"],
                "recall_floor_feasible": row["recall_floor_feasible"],
                "aggregate": row["aggregate"],
            }
            for row in candidates
        ],
    )


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    labels = [value.strip() for value in args.labels.split(",") if value.strip()]
    calibration_ids = _read_ids(args.calibration_video_id_file)
    test_ids = _read_ids(args.test_video_id_file)
    overlap = sorted(set(calibration_ids) & set(test_ids))
    if overlap:
        raise ValueError(f"Calibration/test leakage: {overlap}")

    calibration_data, loaded_calibration_ids = load_run_scores(
        Path(args.calibration_run_dir),
        labels=labels,
        branch="fused",
        tolerance_sec=args.match_tolerance_sec,
        exclusions=set(),
        video_ids=calibration_ids,
    )
    test_data, loaded_test_ids = load_run_scores(
        Path(args.test_run_dir),
        labels=labels,
        branch="fused",
        tolerance_sec=args.match_tolerance_sec,
        exclusions=set(),
        video_ids=test_ids,
    )
    if loaded_calibration_ids != calibration_ids:
        missing = [value for value in calibration_ids if value not in loaded_calibration_ids]
        raise FileNotFoundError(f"Incomplete calibration dense cache; missing={missing}")
    if loaded_test_ids != test_ids:
        missing = [value for value in test_ids if value not in loaded_test_ids]
        raise FileNotFoundError(f"Incomplete test dense cache; missing={missing}")

    floors = _parse_floors(args.recall_floors, labels)
    alphas = [
        args.alpha_min + index * args.alpha_step
        for index in range(
            round((args.alpha_max - args.alpha_min) / args.alpha_step) + 1
        )
    ]
    calibration_by_label = {
        label: [item for item in calibration_data if item.label == label]
        for label in labels
    }
    test_by_label = {
        label: [item for item in test_data if item.label == label] for label in labels
    }

    report: dict[str, Any] = {
        "protocol": "validation_loov_select_alpha_then_refit_one_threshold_and_freeze_for_test",
        "metric_protocol": "window_overlap_many_predictions_per_gt_unique_gt_recall",
        "match_tolerance_sec": args.match_tolerance_sec,
        "test_label_usage": "evaluation_only",
        "inference_adaptation": "unlabeled per-video median logit only",
        "calibration": {
            "run_dir": str(Path(args.calibration_run_dir).resolve()),
            "video_id_file": str(Path(args.calibration_video_id_file).resolve()),
            "video_id_file_sha256": _sha256(args.calibration_video_id_file),
            "video_ids": loaded_calibration_ids,
        },
        "test": {
            "run_dir": str(Path(args.test_run_dir).resolve()),
            "video_id_file": str(Path(args.test_video_id_file).resolve()),
            "video_id_file_sha256": _sha256(args.test_video_id_file),
            "video_ids": loaded_test_ids,
        },
        "calibration_test_overlap": overlap,
        "recall_floors": floors,
        "per_class": {},
    }

    for label in labels:
        calibration_items = calibration_by_label[label]
        test_items = test_by_label[label]
        alpha, folds, loov_metric, alpha_feasible, alpha_sweep = _alpha_loov(
            calibration_items, alphas, floors[label]
        )
        calibration_rows = [
            (item, _adaptive_scores(item, alpha)[0]) for item in calibration_items
        ]
        threshold, calibration_refit_metric, refit_feasible = _select(
            calibration_rows, floors[label]
        )
        test_details = _per_video_metrics(test_items, alpha, threshold)
        report["per_class"][label] = {
            "selected_alpha": alpha,
            "frozen_adaptive_logit_threshold": threshold,
            "zero_median_probability_threshold": _sigmoid(threshold),
            "validation_loov_selection": {
                "recall_floor_feasible": alpha_feasible,
                "aggregate": loov_metric,
                "folds": folds,
                "alpha_sweep": alpha_sweep,
            },
            "validation_all_refit": {
                "recall_floor_feasible": refit_feasible,
                "aggregate": calibration_refit_metric,
            },
            "test_frozen_transfer": {
                "aggregate": combine_metrics(test_details),
                "per_video": test_details,
            },
        }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-run-dir", required=True)
    parser.add_argument("--calibration-video-id-file", required=True)
    parser.add_argument("--test-run-dir", required=True)
    parser.add_argument("--test-video-id-file", required=True)
    parser.add_argument("--labels", default="shot,save,set_piece")
    parser.add_argument("--match-tolerance-sec", type=float, default=3.0)
    parser.add_argument(
        "--recall-floors", default="shot=0.90,save=0.85,set_piece=0.85"
    )
    parser.add_argument("--alpha-min", type=float, default=-0.5)
    parser.add_argument("--alpha-max", type=float, default=1.5)
    parser.add_argument("--alpha-step", type=float, default=0.1)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = analyze(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                label: {
                    "alpha": row["selected_alpha"],
                    "threshold": row["frozen_adaptive_logit_threshold"],
                    "validation_loov": row["validation_loov_selection"]["aggregate"],
                    "validation_refit": row["validation_all_refit"]["aggregate"],
                    "test": row["test_frozen_transfer"]["aggregate"],
                }
                for label, row in report["per_class"].items()
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
