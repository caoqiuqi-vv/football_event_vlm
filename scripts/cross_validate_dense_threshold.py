#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from analyze_dense_pr_ceiling import (
    aggregate_at_threshold,
    best_row,
    combine_metrics,
    evaluate_data,
    exact_curve,
    load_run_scores,
)


def parse_video_ids(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def requested_video_ids(raw: str, path: str) -> list[str]:
    if raw and path:
        raise ValueError("Use only one of --video-ids and --video-id-file")
    if path:
        return [
            line.strip()
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    return parse_video_ids(raw)


def index_by_video(data: list[Any]) -> dict[str, Any]:
    result = {item.video_id: item for item in data}
    if len(result) != len(data):
        raise ValueError("Expected exactly one score record per video and label")
    return result


def cross_validate(args: argparse.Namespace) -> dict[str, Any]:
    requested_ids = requested_video_ids(args.video_ids, args.video_id_file)
    selected_ids = requested_ids or None
    baseline_data, baseline_ids = load_run_scores(
        Path(args.baseline_run),
        labels=[args.label],
        branch=args.baseline_branch,
        tolerance_sec=args.match_tolerance_sec,
        exclusions=set(),
        video_ids=selected_ids,
    )
    candidate_data, candidate_ids = load_run_scores(
        Path(args.candidate_run),
        labels=[args.label],
        branch=args.candidate_branch,
        tolerance_sec=args.match_tolerance_sec,
        exclusions=set(),
        video_ids=selected_ids,
    )
    video_ids = requested_ids or baseline_ids
    if set(video_ids) != set(baseline_ids) or set(video_ids) != set(candidate_ids):
        raise ValueError(
            "Baseline and candidate must contain the same requested videos: "
            f"requested={video_ids} baseline={baseline_ids} candidate={candidate_ids}"
        )
    if len(video_ids) < 3:
        raise ValueError("Leave-one-video-out validation requires at least three videos")

    baseline_by_video = index_by_video(baseline_data)
    candidate_by_video = index_by_video(candidate_data)
    folds: list[dict[str, Any]] = []
    for held_out in video_ids:
        train_ids = [video_id for video_id in video_ids if video_id != held_out]
        baseline_train = [baseline_by_video[video_id] for video_id in train_ids]
        candidate_train = [candidate_by_video[video_id] for video_id in train_ids]
        baseline_train_metrics = aggregate_at_threshold(
            baseline_train, args.baseline_threshold
        )
        selected = best_row(
            exact_curve(candidate_train),
            recall_floor=float(baseline_train_metrics["recall"]),
            objective="precision",
        )
        if selected is None:
            raise RuntimeError(f"No candidate threshold found for held_out={held_out}")
        threshold = float(selected["threshold"])
        baseline_test = evaluate_data(
            baseline_by_video[held_out], args.baseline_threshold
        )
        candidate_test = evaluate_data(candidate_by_video[held_out], threshold)
        folds.append(
            {
                "held_out_video_id": held_out,
                "train_video_ids": train_ids,
                "baseline_train": baseline_train_metrics,
                "selected_candidate_train": selected,
                "selected_threshold": threshold,
                "baseline_test": baseline_test,
                "candidate_test": candidate_test,
                "test_precision_delta": float(candidate_test["precision"])
                - float(baseline_test["precision"]),
                "test_recall_delta": float(candidate_test["recall"])
                - float(baseline_test["recall"]),
            }
        )

    baseline_aggregate = combine_metrics(fold["baseline_test"] for fold in folds)
    candidate_aggregate = combine_metrics(fold["candidate_test"] for fold in folds)
    return {
        "protocol": "window_overlap_leave_one_video_out",
        "label": args.label,
        "match_tolerance_sec": args.match_tolerance_sec,
        "baseline": {
            "run_dir": args.baseline_run,
            "branch": args.baseline_branch,
            "threshold": args.baseline_threshold,
            "aggregate": baseline_aggregate,
        },
        "candidate": {
            "run_dir": args.candidate_run,
            "branch": args.candidate_branch,
            "aggregate": candidate_aggregate,
        },
        "aggregate_precision_delta": float(candidate_aggregate["precision"])
        - float(baseline_aggregate["precision"]),
        "aggregate_recall_delta": float(candidate_aggregate["recall"])
        - float(baseline_aggregate["recall"]),
        "recall_guard_pass": float(candidate_aggregate["recall"])
        + float(args.recall_tolerance_pp) / 100.0
        >= float(baseline_aggregate["recall"]),
        "folds": folds,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a candidate threshold on N-1 dense-eval videos at the baseline "
            "recall, then test it on the held-out video."
        )
    )
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--candidate-run", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--baseline-threshold", type=float, required=True)
    parser.add_argument("--baseline-branch", default="fused", choices=("fused", "global", "local"))
    parser.add_argument("--candidate-branch", default="fused", choices=("fused", "global", "local"))
    parser.add_argument("--video-ids", default="")
    parser.add_argument("--video-id-file", default="")
    parser.add_argument("--match-tolerance-sec", type=float, default=5.0)
    parser.add_argument("--recall-tolerance-pp", type=float, default=1.0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = cross_validate(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
