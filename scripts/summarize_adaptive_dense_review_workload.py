#!/usr/bin/env python
"""Compute deduplicated review workload for LOOV adaptive dense thresholds."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


def read_ids(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def logit(probability: float) -> float:
    p = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def merge_intervals(
    rows: list[dict[str, Any]], merge_gap_sec: float
) -> list[dict[str, Any]]:
    rows.sort(key=lambda row: (row["start_sec"], row["end_sec"]))
    merged: list[dict[str, Any]] = []
    for row in rows:
        if not merged or row["start_sec"] > merged[-1]["end_sec"] + merge_gap_sec:
            merged.append(
                {
                    "start_sec": row["start_sec"],
                    "end_sec": row["end_sec"],
                    "labels": set(row["labels"]),
                    "raw_windows": 1,
                }
            )
            continue
        merged[-1]["end_sec"] = max(merged[-1]["end_sec"], row["end_sec"])
        merged[-1]["labels"].update(row["labels"])
        merged[-1]["raw_windows"] += 1
    return merged


def summarize_mode(
    run_dir: Path,
    video_ids: list[str],
    labels: list[str],
    fold_params: dict[str, dict[str, dict[str, float]]],
    *,
    merge_gap_sec: float,
    match_tolerance_sec: float,
    manual_overhead_sec: float,
) -> dict[str, Any]:
    total_duration = 0.0
    total_union = 0.0
    total_raw = 0
    total_segments = 0
    confirmed = 0
    raw_by_label = {label: 0 for label in labels}
    details: list[dict[str, Any]] = []
    for video_id in video_ids:
        video_dir = run_dir / video_id
        summary = json.loads((video_dir / "summary.json").read_text())
        gt = json.loads((video_dir / "gt_events.json").read_text())
        with (video_dir / "window_predictions.csv").open(newline="") as handle:
            windows = list(csv.DictReader(handle))
        locations = {
            label: statistics.median(
                [logit(float(row[f"prob_{label}"])) for row in windows]
            )
            if windows
            else 0.0
            for label in labels
        }
        candidates: list[dict[str, Any]] = []
        for row in windows:
            active: set[str] = set()
            for label in labels:
                params = fold_params[label][video_id]
                score = (
                    logit(float(row[f"prob_{label}"]))
                    - params["alpha"] * locations[label]
                )
                if score >= params["threshold"]:
                    active.add(label)
                    raw_by_label[label] += 1
            if active:
                candidates.append(
                    {
                        "start_sec": float(row["start_sec"]),
                        "end_sec": float(row["end_sec"]),
                        "labels": active,
                    }
                )
        segments = merge_intervals(candidates, merge_gap_sec)
        video_confirmed = 0
        for segment in segments:
            hit = any(
                event.get("label") in segment["labels"]
                and segment["start_sec"] - match_tolerance_sec
                <= float(event["time_sec"])
                <= segment["end_sec"] + match_tolerance_sec
                for event in gt
            )
            video_confirmed += int(hit)
        duration = float(summary["duration_sec"])
        union_sec = sum(row["end_sec"] - row["start_sec"] for row in segments)
        total_duration += duration
        total_union += union_sec
        total_raw += len(candidates)
        total_segments += len(segments)
        confirmed += video_confirmed
        details.append(
            {
                "video_id": video_id,
                "duration_sec": duration,
                "raw_positive_windows": len(candidates),
                "merged_review_segments": len(segments),
                "confirmed_segments": video_confirmed,
                "rejected_segments": len(segments) - video_confirmed,
                "review_union_sec": union_sec,
                "review_union_ratio": union_sec / duration if duration else 0.0,
            }
        )
    rejected = total_segments - confirmed
    manual_sec = total_union + manual_overhead_sec * total_segments
    return {
        "labels": labels,
        "raw_positive_windows": total_raw,
        "raw_positive_windows_by_label": raw_by_label,
        "merged_review_segments": total_segments,
        "confirmed_segments": confirmed,
        "rejected_segments": rejected,
        "rejected_segment_ratio": rejected / total_segments if total_segments else 0.0,
        "review_union_sec": total_union,
        "review_union_ratio": total_union / total_duration if total_duration else 0.0,
        "manual_time_with_overhead_sec": manual_sec,
        "manual_time_ratio_with_overhead": (
            manual_sec / total_duration if total_duration else 0.0
        ),
        "segments_per_video_hour": (
            total_segments / (total_duration / 3600.0) if total_duration else 0.0
        ),
        "total_duration_sec": total_duration,
        "per_video": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--adaptive-report", type=Path, required=True)
    parser.add_argument("--video-id-file", type=Path, required=True)
    parser.add_argument("--labels", default="shot,save,set_piece")
    parser.add_argument("--match-tolerance-sec", type=float, default=3.0)
    parser.add_argument("--merge-gap-sec", type=float, default=0.0)
    parser.add_argument("--manual-overhead-sec", type=float, default=3.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.adaptive_report.read_text())
    video_ids = read_ids(args.video_id_file)
    all_labels = [value.strip() for value in args.labels.split(",") if value.strip()]
    fold_params: dict[str, dict[str, dict[str, float]]] = {}
    for label in all_labels:
        folds = report["per_class"][label]["adaptive_median_logit_loov"]["folds"]
        fold_params[label] = {
            str(fold["held_out_video_id"]): {
                "alpha": float(fold["alpha"]),
                "threshold": float(fold["adaptive_threshold"]),
            }
            for fold in folds
        }

    payload = {
        "protocol": "adaptive_median_logit_loov_cross_class_interval_union",
        "run_dir": str(args.run_dir),
        "adaptive_report": str(args.adaptive_report),
        "match_tolerance_sec": args.match_tolerance_sec,
        "merge_gap_sec": args.merge_gap_sec,
        "all_classes": summarize_mode(
            args.run_dir,
            video_ids,
            all_labels,
            fold_params,
            merge_gap_sec=args.merge_gap_sec,
            match_tolerance_sec=args.match_tolerance_sec,
            manual_overhead_sec=args.manual_overhead_sec,
        ),
        "shot_only": summarize_mode(
            args.run_dir,
            video_ids,
            ["shot"],
            fold_params,
            merge_gap_sec=args.merge_gap_sec,
            match_tolerance_sec=args.match_tolerance_sec,
            manual_overhead_sec=args.manual_overhead_sec,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "all_classes": payload["all_classes"],
                "shot_only": payload["shot_only"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
