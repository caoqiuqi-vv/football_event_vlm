#!/usr/bin/env python
"""Split a shared set-piece score into corner/freekick recall diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


SUBTYPE_LABELS = {
    "corner": {"corner", "角球"},
    "freekick": {"freekick", "free_kick", "任意球"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--adaptive-report", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--tolerance-sec", type=float, default=3.0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def time_sec(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "0").strip()
    if ":" not in text:
        return float(text)
    parts = [float(item) for item in text.split(":")]
    result = 0.0
    for part in parts:
        result = result * 60.0 + part
    return result


def logit(probability: float) -> float:
    probability = min(max(float(probability), 1e-8), 1.0 - 1e-8)
    return math.log(probability / (1.0 - probability))


def load_gt(path: Path) -> dict[str, list[float]]:
    payload = json.loads(path.read_text())
    rows = payload.get("data", payload) if isinstance(payload, dict) else payload
    result = {subtype: [] for subtype in SUBTYPE_LABELS}
    for row in rows:
        if not isinstance(row, dict) or row.get("label_correct") is False:
            continue
        label = str(row.get("label", row.get("event_label", ""))).strip().lower()
        for subtype, aliases in SUBTYPE_LABELS.items():
            if label in aliases:
                raw_time = row.get(
                    "timestamp", row.get("startTime", row.get("time_sec", 0.0))
                )
                result[subtype].append(time_sec(raw_time))
    for values in result.values():
        values.sort()
    return result


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    report = json.loads(Path(args.adaptive_report).read_text())
    folds = report["per_class"]["set_piece"]["adaptive_median_logit_loov"]["folds"]
    fold_by_video = {str(row["held_out_video_id"]): row for row in folds}
    totals = {
        subtype: {"gt": 0, "window_matched": 0, "point_matched": 0}
        for subtype in SUBTYPE_LABELS
    }
    per_video: list[dict[str, Any]] = []
    tolerance = float(args.tolerance_sec)
    for video_id, fold in fold_by_video.items():
        with (run_dir / video_id / "window_predictions.csv").open(newline="") as file:
            windows = list(csv.DictReader(file))
        median_logit = float(fold["video_median_logit"])
        alpha = float(fold["alpha"])
        threshold = float(fold["adaptive_threshold"])
        selected: list[tuple[float, float, float, float]] = []
        for row in windows:
            probability = float(row["prob_set_piece"])
            adjusted_score = logit(probability) - alpha * median_logit
            if adjusted_score >= threshold:
                start = float(row["start_sec"])
                end = float(row["end_sec"])
                selected.append((adjusted_score, start, end, 0.5 * (start + end)))
        gt_by_subtype = load_gt(Path(args.gt_dir) / f"{video_id}.json")
        video_row: dict[str, Any] = {
            "video_id": video_id,
            "selected_windows": len(selected),
            "subtypes": {},
        }
        for subtype, gt_times in gt_by_subtype.items():
            window_matched = sum(
                any(start - tolerance <= gt <= end + tolerance for _, start, end, _ in selected)
                for gt in gt_times
            )
            unmatched = set(range(len(gt_times)))
            point_matched = 0
            for score, _, _, center in sorted(selected, reverse=True):
                candidates = [
                    index
                    for index in unmatched
                    if abs(center - gt_times[index]) <= tolerance
                ]
                if not candidates:
                    continue
                best = min(candidates, key=lambda index: abs(center - gt_times[index]))
                unmatched.remove(best)
                point_matched += 1
            totals[subtype]["gt"] += len(gt_times)
            totals[subtype]["window_matched"] += window_matched
            totals[subtype]["point_matched"] += point_matched
            video_row["subtypes"][subtype] = {
                "gt": len(gt_times),
                "window_matched": window_matched,
                "window_recall": window_matched / len(gt_times) if gt_times else None,
                "point_matched": point_matched,
                "point_recall": point_matched / len(gt_times) if gt_times else None,
            }
        per_video.append(video_row)
    aggregate = {}
    for subtype, counts in totals.items():
        aggregate[subtype] = {
            **counts,
            "window_recall": counts["window_matched"] / counts["gt"] if counts["gt"] else 0.0,
            "point_recall": counts["point_matched"] / counts["gt"] if counts["gt"] else 0.0,
        }
    result = {
        "model_score": "shared set_piece head",
        "threshold_protocol": "adaptive_median_logit_loov",
        "tolerance_sec": tolerance,
        "aggregate": aggregate,
        "per_video": per_video,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
