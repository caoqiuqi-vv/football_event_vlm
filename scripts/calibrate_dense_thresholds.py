#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_dense_pr_ceiling import (  # noqa: E402
    aggregate_at_threshold,
    best_row,
    exact_curve,
    load_run_scores,
    thresholds_from_summaries,
)


def parse_labels(raw: str) -> list[str]:
    labels = [item.strip() for item in raw.split(",") if item.strip()]
    if not labels:
        raise ValueError("At least one label is required")
    if len(set(labels)) != len(labels):
        raise ValueError(f"Duplicate labels: {labels}")
    return labels


def parse_video_ids(path: str) -> list[str] | None:
    if not path:
        return None
    video_ids = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not video_ids:
        raise ValueError(f"No video ids found in {path}")
    return video_ids


def calibrate(args: argparse.Namespace) -> dict[str, Any]:
    labels = parse_labels(args.labels)
    requested_ids = parse_video_ids(args.video_id_file)
    baseline_run = Path(args.baseline_run)
    candidate_run = Path(args.candidate_run)
    baseline_thresholds = thresholds_from_summaries(
        baseline_run,
        requested_ids
        or sorted(
            path.name
            for path in baseline_run.iterdir()
            if path.is_dir() and (path / "summary.json").is_file()
        ),
        labels,
    )

    per_class: dict[str, Any] = {}
    candidate_thresholds: dict[str, float] = {}
    common_ids: list[str] | None = None
    for label in labels:
        baseline_data, baseline_ids = load_run_scores(
            baseline_run,
            labels=[label],
            branch=args.baseline_branch,
            tolerance_sec=args.match_tolerance_sec,
            exclusions=set(),
            video_ids=requested_ids,
        )
        candidate_data, candidate_ids = load_run_scores(
            candidate_run,
            labels=[label],
            branch=args.candidate_branch,
            tolerance_sec=args.match_tolerance_sec,
            exclusions=set(),
            video_ids=requested_ids,
        )
        if baseline_ids != candidate_ids:
            raise ValueError(
                f"Video mismatch for {label}: baseline={baseline_ids} candidate={candidate_ids}"
            )
        if common_ids is None:
            common_ids = baseline_ids
        elif common_ids != baseline_ids:
            raise ValueError(f"Inconsistent video ids for label={label}")

        baseline_threshold = float(baseline_thresholds[label])
        baseline_metrics = aggregate_at_threshold(
            baseline_data, baseline_threshold
        )
        recall_target = min(
            1.0,
            float(baseline_metrics["recall"])
            + float(args.recall_margin_pp) / 100.0,
        )
        selected = best_row(
            exact_curve(candidate_data),
            recall_floor=recall_target,
            objective="precision",
        )
        if selected is None:
            raise RuntimeError(
                f"Candidate cannot reach recall={recall_target:.6f} for {label}"
            )
        threshold = float(selected["threshold"])
        candidate_metrics = aggregate_at_threshold(candidate_data, threshold)
        candidate_thresholds[label] = threshold
        per_class[label] = {
            "baseline_threshold": baseline_threshold,
            "candidate_threshold": threshold,
            "recall_target": recall_target,
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
            "precision_delta": float(candidate_metrics["precision"])
            - float(baseline_metrics["precision"]),
            "recall_delta": float(candidate_metrics["recall"])
            - float(baseline_metrics["recall"]),
            "recall_guard_pass": float(candidate_metrics["recall"])
            + 1e-12
            >= recall_target,
        }

    return {
        "protocol": "window_overlap_dense_calibration",
        "selection": "max_precision_at_baseline_recall",
        "baseline_run": str(baseline_run),
        "candidate_run": str(candidate_run),
        "baseline_branch": args.baseline_branch,
        "candidate_branch": args.candidate_branch,
        "match_tolerance_sec": float(args.match_tolerance_sec),
        "recall_margin_pp": float(args.recall_margin_pp),
        "video_ids": common_ids or [],
        "candidate_thresholds": candidate_thresholds,
        "per_class": per_class,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate candidate dense-window thresholds at each baseline class recall."
        )
    )
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--candidate-run", required=True)
    parser.add_argument("--video-id-file", default="")
    parser.add_argument("--labels", default="shot,save,set_piece")
    parser.add_argument("--baseline-branch", default="fused")
    parser.add_argument("--candidate-branch", default="fused")
    parser.add_argument("--match-tolerance-sec", type=float, default=5.0)
    parser.add_argument(
        "--recall-margin-pp",
        type=float,
        default=0.0,
        help="Require candidate recall above baseline by this many percentage points.",
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = calibrate(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for label, item in report["per_class"].items():
        print(
            f"{label}: threshold={item['candidate_threshold']:.9g} "
            f"P={item['candidate']['precision']:.4f} "
            f"R={item['candidate']['recall']:.4f} "
            f"dP={100.0 * item['precision_delta']:+.2f}pp "
            f"dR={100.0 * item['recall_delta']:+.2f}pp",
            flush=True,
        )
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
