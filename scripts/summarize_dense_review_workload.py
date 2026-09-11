#!/usr/bin/env python
"""Summarize deduplicated human-review workload from dense window predictions."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_ids(path: str, run_dir: Path) -> list[str]:
    if path:
        return [line.strip() for line in Path(path).read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
    config = run_dir / "run_config.json"
    if config.exists():
        return [str(value) for value in json.loads(config.read_text()).get("video_ids", [])]
    return sorted(item.name for item in run_dir.iterdir() if item.is_dir())


def _merge(rows: list[dict[str, Any]], merge_gap: float) -> list[dict[str, Any]]:
    rows.sort(key=lambda row: (row["start_sec"], row["end_sec"]))
    merged: list[dict[str, Any]] = []
    for row in rows:
        if not merged or row["start_sec"] > merged[-1]["end_sec"] + merge_gap:
            merged.append({
                "start_sec": row["start_sec"], "end_sec": row["end_sec"],
                "labels": set(row["labels"]), "raw_windows": 1,
            })
        else:
            merged[-1]["end_sec"] = max(merged[-1]["end_sec"], row["end_sec"])
            merged[-1]["labels"].update(row["labels"])
            merged[-1]["raw_windows"] += 1
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--video-id-file", default="")
    parser.add_argument("--labels", default="shot,save,set_piece")
    parser.add_argument("--match-tolerance-sec", type=float, default=3.0)
    parser.add_argument("--merge-gap-sec", type=float, default=0.0)
    parser.add_argument("--manual-overhead-sec", type=float, default=3.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    labels = [value.strip() for value in args.labels.split(",") if value.strip()]
    video_ids = _read_ids(args.video_id_file, run_dir)
    total_duration = total_union = 0.0
    total_raw = total_segments = confirmed = rejected = 0
    raw_by_label = {label: 0 for label in labels}
    subtype_counts: dict[str, dict[str, int]] = {}
    details: list[dict[str, Any]] = []
    for video_id in video_ids:
        video_dir = run_dir / video_id
        summary = json.loads((video_dir / "summary.json").read_text())
        thresholds = {label: float(summary["thresholds"][label]) for label in labels}
        with (video_dir / "window_predictions.csv").open(newline="") as handle:
            windows = list(csv.DictReader(handle))
        gt = json.loads((video_dir / "gt_events.json").read_text())
        candidate_rows: list[dict[str, Any]] = []
        positive_intervals = {label: [] for label in labels}
        for window in windows:
            active = {
                label for label in labels
                if float(window[f"prob_{label}"]) >= thresholds[label]
            }
            if not active:
                continue
            start, end = float(window["start_sec"]), float(window["end_sec"])
            candidate_rows.append({"start_sec": start, "end_sec": end, "labels": active})
            for label in active:
                raw_by_label[label] += 1
                positive_intervals[label].append((start, end))
        segments = _merge(candidate_rows, args.merge_gap_sec)
        video_confirmed = 0
        for segment in segments:
            is_confirmed = any(
                event.get("label") in segment["labels"]
                and segment["start_sec"] - args.match_tolerance_sec
                <= float(event["time_sec"])
                <= segment["end_sec"] + args.match_tolerance_sec
                for event in gt
            )
            segment["confirmed"] = is_confirmed
            segment["labels"] = sorted(segment["labels"])
            video_confirmed += int(is_confirmed)
        for event in gt:
            if event.get("label") != "set_piece":
                continue
            subtype = str(event.get("raw_label") or "unknown")
            entry = subtype_counts.setdefault(subtype, {"num_gt": 0, "covered_gt": 0})
            entry["num_gt"] += 1
            time_sec = float(event["time_sec"])
            if any(start - args.match_tolerance_sec <= time_sec <= end + args.match_tolerance_sec
                   for start, end in positive_intervals["set_piece"]):
                entry["covered_gt"] += 1
        duration = float(summary["duration_sec"])
        union = sum(segment["end_sec"] - segment["start_sec"] for segment in segments)
        total_duration += duration
        total_union += union
        total_raw += len(candidate_rows)
        total_segments += len(segments)
        confirmed += video_confirmed
        rejected += len(segments) - video_confirmed
        details.append({
            "video_id": video_id, "duration_sec": duration,
            "raw_positive_windows": len(candidate_rows),
            "merged_review_segments": len(segments),
            "confirmed_segments": video_confirmed,
            "rejected_segments": len(segments) - video_confirmed,
            "review_union_sec": union,
            "segments": segments,
        })
    for value in subtype_counts.values():
        value["recall"] = value["covered_gt"] / value["num_gt"] if value["num_gt"] else 0.0
    manual_time = total_union + args.manual_overhead_sec * total_segments
    report = {
        "protocol": "cross_class_interval_union_at_checkpoint_thresholds",
        "match_tolerance_sec": args.match_tolerance_sec,
        "merge_gap_sec": args.merge_gap_sec,
        "summary": {
            "num_videos": len(video_ids), "total_duration_sec": total_duration,
            "raw_positive_windows": total_raw, "raw_positive_windows_by_label": raw_by_label,
            "merged_review_segments": total_segments,
            "confirmed_segments": confirmed, "rejected_segments": rejected,
            "rejected_segment_ratio": rejected / total_segments if total_segments else 0.0,
            "review_union_sec": total_union,
            "review_union_ratio": total_union / total_duration if total_duration else 0.0,
            "manual_time_with_overhead_sec": manual_time,
            "manual_time_ratio": manual_time / total_duration if total_duration else 0.0,
            "segments_per_video_hour": total_segments / (total_duration / 3600.0) if total_duration else 0.0,
        },
        "set_piece_subtype_window_coverage": subtype_counts,
        "per_video": details,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"summary": report["summary"],
                      "set_piece_subtype_window_coverage": subtype_counts},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
