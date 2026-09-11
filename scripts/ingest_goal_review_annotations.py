#!/usr/bin/env python
"""Validate review CSV and emit reusable event/contrast-set annotations."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


STATUSES = {"confirmed", "corrected", "rejected", "uncertain"}
CONFOUNDERS = {
    "shot", "save", "set_piece", "cross", "long_pass", "clearance",
    "ordinary_pass", "restart_preparation", "camera_motion", "background",
    "uncertain", "",
}


def split_values(raw: str) -> list[str]:
    return [item.strip() for item in raw.replace(",", ";").split(";") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    with args.annotations.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    errors: list[str] = []
    events_by_video: dict[str, list[dict[str, object]]] = defaultdict(list)
    contrast: list[dict[str, object]] = []
    counts: dict[str, int] = defaultdict(int)
    for row_number, row in enumerate(rows, start=2):
        status = row.get("review_status", "").strip()
        if not status:
            if args.allow_incomplete:
                continue
            errors.append(f"row {row_number}: missing review_status")
            continue
        if status not in STATUSES:
            errors.append(f"row {row_number}: invalid review_status={status!r}")
            continue
        confounder = row.get("confounder", "").strip()
        if confounder not in CONFOUNDERS:
            errors.append(f"row {row_number}: invalid confounder={confounder!r}")
        labels = split_values(row.get("correct_labels", ""))
        raw_times = split_values(row.get("correct_event_times_sec", ""))
        try:
            times = [float(value) for value in raw_times]
        except ValueError:
            errors.append(f"row {row_number}: invalid corrected time")
            continue
        if status in {"confirmed", "corrected"}:
            if not labels or not times:
                errors.append(f"row {row_number}: confirmed/corrected requires labels and times")
                continue
            if len(labels) == 1 and len(times) > 1:
                labels = labels * len(times)
            if len(labels) != len(times):
                errors.append(f"row {row_number}: labels/times length mismatch")
                continue
            for label, time_sec in zip(labels, times):
                events_by_video[row["video_id"]].append({
                    "label": label,
                    "time_sec": time_sec,
                    "source": "goal_oriented_human_review",
                    "segment_id": row["segment_id"],
                })
        contrast.append({
            "segment_id": row["segment_id"],
            "video_id": row["video_id"],
            "start_sec": float(row["start_sec"]),
            "end_sec": float(row["end_sec"]),
            "candidate_ids": row.get("candidate_ids", ""),
            "predicted_labels": row.get("predicted_labels", ""),
            "review_status": status,
            "correct_labels": ";".join(labels),
            "correct_event_times_sec": ";".join(str(value) for value in times),
            "confounder": confounder or ("uncertain" if status == "uncertain" else "background" if status == "rejected" else ""),
            "loss_mask": 0 if status == "uncertain" else 1,
            "notes": row.get("notes", ""),
        })
        counts[status] += 1
    if errors:
        raise ValueError("annotation validation failed:\n" + "\n".join(errors[:50]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "contrast_set.jsonl").open("w", encoding="utf-8") as handle:
        for row in contrast:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    for video_id, events in events_by_video.items():
        events.sort(key=lambda item: (float(item["time_sec"]), str(item["label"])))
        (args.output_dir / f"{video_id}.events.patch.json").write_text(
            json.dumps({"video_id": video_id, "events": events}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    summary = {
        "schema": "football_goal_review_feedback.v1",
        "review_rows": len(rows),
        "ingested_rows": len(contrast),
        "status_counts": dict(sorted(counts.items())),
        "corrected_events": sum(len(events) for events in events_by_video.values()),
        "videos_with_event_patches": len(events_by_video),
        "uncertain_loss_is_masked": True,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
