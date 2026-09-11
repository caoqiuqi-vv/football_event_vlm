#!/usr/bin/env python3
"""Summarize visual, whistle-fused, and human-review coverage on final test18 GT."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


TARGETS = ("shot", "save", "corner", "free_kick", "kickoff")
SUBTYPE_KEYS = {"corner": "corner", "free_kick": "freekick", "kickoff": "kickoff"}


def merge_intervals(rows: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(rows):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def covered(time_sec: float, intervals: Iterable[tuple[float, float]], tolerance: float) -> bool:
    return any(start - tolerance <= time_sec <= end + tolerance for start, end in intervals)


def load_final_events(path: Path) -> tuple[dict[str, dict[str, list[float]]], dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    events: dict[str, dict[str, list[float]]] = {}
    durations: dict[str, float] = {}
    for video_id, video in payload["videos"].items():
        per_label = {label: [] for label in TARGETS}
        for row in video.get("events", []):
            label = str(row.get("semantic_label") or row.get("label") or "")
            if label in per_label:
                per_label[label].append(float(row["time_sec"]))
        events[video_id] = per_label
        durations[video_id] = float(video.get("source", {}).get("duration_sec", 0.0))
    return events, durations


def load_review_intervals(path: Path) -> tuple[
    dict[str, list[tuple[float, float]]], dict[str, list[tuple[float, float]]], dict[str, Any]
]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    all_by_video: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    candidate_by_video: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    for video in manifest["videos"]:
        video_id = str(video["video_id"])
        for event in video.get("events", []):
            segment_id = str(event["segment_id"])
            interval = (float(event["start_sec"]), float(event["end_sec"]))
            all_by_video[video_id][segment_id] = interval
            sources = set(event.get("review_sources") or [])
            if event.get("task_source"):
                sources.add(str(event["task_source"]))
            if "candidate" in sources:
                candidate_by_video[video_id][segment_id] = interval
    return (
        {video_id: merge_intervals(rows.values()) for video_id, rows in all_by_video.items()},
        {video_id: merge_intervals(rows.values()) for video_id, rows in candidate_by_video.items()},
        manifest.get("summary", {}),
    )


def interval_coverage(
    truth: dict[str, dict[str, list[float]]],
    intervals: dict[str, list[tuple[float, float]]],
    tolerance: float,
) -> dict[str, dict[str, float | int]]:
    output = {}
    for label in TARGETS:
        total = matched = 0
        for video_id, by_label in truth.items():
            times = by_label[label]
            total += len(times)
            matched += sum(covered(time_sec, intervals.get(video_id, []), tolerance) for time_sec in times)
        output[label] = {
            "gt": total,
            "matched": matched,
            "missed": total - matched,
            "recall": matched / total if total else 0.0,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-labels", type=Path, required=True)
    parser.add_argument("--adaptive-report", type=Path, required=True)
    parser.add_argument("--whistle-report", type=Path, required=True)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    truth, durations = load_final_events(args.final_labels)
    adaptive = json.loads(args.adaptive_report.read_text(encoding="utf-8"))
    whistle = json.loads(args.whistle_report.read_text(encoding="utf-8"))
    whistle_loov = next(
        row for row in whistle["protocols"] if row["name"] == "test18_leave_one_video_out"
    )
    all_intervals, candidate_intervals, queue_summary = load_review_intervals(args.review_manifest)

    visual: dict[str, dict[str, float | int]] = {}
    fused: dict[str, dict[str, float | int]] = {}
    for label in ("shot", "save"):
        metric = adaptive["per_class"][label]["adaptive_median_logit_loov"]["aggregate"]
        row = {
            "gt": int(metric["num_gt"]),
            "matched": int(metric["num_matched_gt"]),
            "missed": int(metric["fn"]),
            "recall": float(metric["recall"]),
            "candidate_window_precision": float(metric["precision"]),
            "candidate_windows": int(metric["num_pred"]),
        }
        visual[label] = row
        fused[label] = dict(row)
    for label, subtype in SUBTYPE_KEYS.items():
        metric = whistle_loov["aggregate"][subtype]
        visual[label] = {
            "gt": int(metric["gt"]),
            "matched": int(metric["visual_window_matched"]),
            "missed": int(metric["gt"] - metric["visual_window_matched"]),
            "recall": float(metric["visual_window_recall"]),
        }
        fused[label] = {
            "gt": int(metric["gt"]),
            "matched": int(metric["fused_window_matched"]),
            "missed": int(metric["gt"] - metric["fused_window_matched"]),
            "recall": float(metric["fused_window_recall"]),
            "additional_matched_by_whistle": int(
                metric["fused_window_matched"] - metric["visual_window_matched"]
            ),
        }

    total_duration = sum(durations.values())
    queue_union_sec = sum(sum(end - start for start, end in rows) for rows in all_intervals.values())
    candidate_union_sec = sum(
        sum(end - start for start, end in rows) for rows in candidate_intervals.values()
    )
    report = {
        "schema_version": "football_test18_pipeline_recall_v1",
        "sources": {
            "final_labels": str(args.final_labels.resolve()),
            "adaptive_report": str(args.adaptive_report.resolve()),
            "whistle_report": str(args.whistle_report.resolve()),
            "review_manifest": str(args.review_manifest.resolve()),
        },
        "protocol": {
            "visual": "LOOV adaptive threshold; GT covered by selected 10-second dense window with ±3s matching tolerance",
            "whistle": "LOOV threshold; whistle is a review-only subtype-agnostic interval [-5s,+10s], never an automatic subtype prediction",
            "review_strict": "final GT timestamp lies inside an actual 8775 playback interval",
            "review_tolerance3": "diagnostic: final GT lies inside an actual 8775 interval expanded by ±3s",
            "same_class_one_to_one_point_nms": False,
        },
        "gt_counts": dict(sorted(Counter(
            label for by_label in truth.values() for label, times in by_label.items() for _ in times
        ).items())),
        "visual_dense": visual,
        "visual_plus_whistle_review": fused,
        "human_review_queue": {
            "manifest_summary": queue_summary,
            "all_ai_plus_original_gt": {
                "strict": interval_coverage(truth, all_intervals, 0.0),
                "tolerance3": interval_coverage(truth, all_intervals, 3.0),
                "union_sec": queue_union_sec,
                "ratio_of_video": queue_union_sec / total_duration if total_duration else 0.0,
            },
            "ai_candidate_segments_only": {
                "strict": interval_coverage(truth, candidate_intervals, 0.0),
                "tolerance3": interval_coverage(truth, candidate_intervals, 3.0),
                "union_sec": candidate_union_sec,
                "ratio_of_video": candidate_union_sec / total_duration if total_duration else 0.0,
            },
            "total_video_sec": total_duration,
        },
        "whistle_workload": whistle_loov["workload"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
