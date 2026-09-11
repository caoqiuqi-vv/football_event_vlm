#!/usr/bin/env python3
"""Split window-overlap false positives by distance to the nearest GT event."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_BINS_SEC = (5.0, 15.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol-json",
        type=Path,
        help="Protocol JSON providing labels, video IDs, thresholds, and tolerance.",
    )
    parser.add_argument("--video-ids", default="")
    parser.add_argument("--labels", default="")
    parser.add_argument("--thresholds", default="")
    parser.add_argument("--match-tolerance-sec", type=float)
    parser.add_argument(
        "--fp-gap-bins-sec",
        default=",".join(str(value) for value in DEFAULT_BINS_SEC),
        help="Comma-separated upper bounds after the tolerance-expanded window.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_thresholds(value: str, labels: list[str]) -> dict[str, float]:
    entries = split_csv(value)
    if len(entries) != len(labels):
        raise ValueError(
            f"Expected {len(labels)} thresholds for labels={labels}, got {entries}"
        )
    return dict(zip(labels, (float(item) for item in entries)))


def window_gap_sec(
    start_sec: float,
    end_sec: float,
    gt_time_sec: float,
    tolerance_sec: float,
) -> float:
    """Distance beyond the tolerance-expanded prediction window."""
    expanded_start = start_sec - tolerance_sec
    expanded_end = end_sec + tolerance_sec
    if gt_time_sec < expanded_start:
        return expanded_start - gt_time_sec
    if gt_time_sec > expanded_end:
        return gt_time_sec - expanded_end
    return 0.0


def gap_bin_name(gap_sec: float, bounds: tuple[float, ...]) -> str:
    lower = 0.0
    for upper in bounds:
        if gap_sec <= upper:
            return f"gap_{lower:g}_{upper:g}s"
        lower = upper
    return f"gap_gt_{bounds[-1]:g}s" if bounds else "gap_all"


def summarize_counts(counts: dict[str, int]) -> dict[str, Any]:
    tp = int(counts.get("tp", 0))
    fp = int(counts.get("fp", 0))
    num_pred = tp + fp
    result: dict[str, Any] = {
        "tp": tp,
        "fp": fp,
        "num_pred": num_pred,
        "precision": tp / num_pred if num_pred else 0.0,
    }
    for key, count in sorted(counts.items()):
        if not key.startswith("gap_"):
            continue
        result[key] = int(count)
        result[f"{key}_fraction_of_fp"] = count / fp if fp else 0.0
    return result


def load_protocol(args: argparse.Namespace) -> tuple[list[str], list[str], dict[str, float], float]:
    protocol_path = args.protocol_json or (
        args.run_dir / "protocol_comparison_checkpoint_thr.json"
    )
    protocol: dict[str, Any] = {}
    if protocol_path.is_file():
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

    labels = split_csv(args.labels) or list(protocol.get("labels", []))
    video_ids = split_csv(args.video_ids) or [
        str(item) for item in protocol.get("video_ids", [])
    ]
    if not labels or not video_ids:
        raise ValueError("labels and video IDs must come from arguments or protocol JSON")

    if args.thresholds:
        thresholds = parse_thresholds(args.thresholds, labels)
    else:
        thresholds = {
            label: float(protocol.get("thresholds", {})[label]) for label in labels
        }
    tolerance = (
        float(args.match_tolerance_sec)
        if args.match_tolerance_sec is not None
        else float(protocol.get("match_tolerance_sec", 5.0))
    )
    return labels, video_ids, thresholds, tolerance


def analyze(
    run_dir: Path,
    labels: list[str],
    video_ids: list[str],
    thresholds: dict[str, float],
    tolerance_sec: float,
    gap_bounds: tuple[float, ...],
) -> dict[str, Any]:
    total_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_video: list[dict[str, Any]] = []

    for video_id in video_ids:
        video_dir = run_dir / video_id
        rows = read_csv(video_dir / "window_predictions.csv")
        gt_rows = read_csv(video_dir / "gt_events.csv")
        gt_times = {
            label: [
                float(row["time_sec"])
                for row in gt_rows
                if row.get("label") == label
            ]
            for label in labels
        }
        for label in labels:
            counts: dict[str, int] = defaultdict(int)
            for row in rows:
                score = float(row[f"prob_{label}"])
                if score < thresholds[label]:
                    continue
                start_sec = float(row["start_sec"])
                end_sec = float(row["end_sec"])
                gaps = [
                    window_gap_sec(start_sec, end_sec, time_sec, tolerance_sec)
                    for time_sec in gt_times[label]
                ]
                nearest_gap = min(gaps, default=math.inf)
                if nearest_gap <= 0.0:
                    counts["tp"] += 1
                    total_counts[label]["tp"] += 1
                    continue
                counts["fp"] += 1
                total_counts[label]["fp"] += 1
                bin_name = (
                    "gap_no_gt"
                    if not gaps
                    else gap_bin_name(nearest_gap, gap_bounds)
                )
                counts[bin_name] += 1
                total_counts[label][bin_name] += 1
            per_video.append(
                {
                    "video_id": video_id,
                    "label": label,
                    **summarize_counts(counts),
                }
            )

    per_class = {
        label: summarize_counts(total_counts[label]) for label in labels
    }
    all_counts: dict[str, int] = defaultdict(int)
    for counts in total_counts.values():
        for key, value in counts.items():
            all_counts[key] += value
    return {
        "run_dir": str(run_dir),
        "protocol": "window_overlap",
        "match_tolerance_sec": tolerance_sec,
        "thresholds": thresholds,
        "fp_gap_bins_sec": list(gap_bounds),
        "per_class": per_class,
        "micro": summarize_counts(all_counts),
        "per_video": per_video,
    }


def main() -> None:
    args = parse_args()
    labels, video_ids, thresholds, tolerance = load_protocol(args)
    gap_bounds = tuple(sorted(float(item) for item in split_csv(args.fp_gap_bins_sec)))
    result = analyze(
        args.run_dir,
        labels,
        video_ids,
        thresholds,
        tolerance,
        gap_bounds,
    )
    output = args.output or args.run_dir / "window_overlap_fp_distance.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["per_class"], ensure_ascii=False, indent=2))
    print(f"saved={output}")


if __name__ == "__main__":
    main()
