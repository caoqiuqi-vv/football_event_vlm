#!/usr/bin/env python
"""Build sparse detector-teacher intervals from the actual event training records."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_football_events as football


def merge_intervals(rows: list[tuple[float, float]], gap_sec: float) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(rows):
        if merged and start <= merged[-1][1] + gap_sec:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", action="append", default=["train"])
    parser.add_argument("--teacher-fps", type=float, default=2.0)
    parser.add_argument("--merge-gap-sec", type=float, default=0.65)
    parser.add_argument("--extra-margin-sec", type=float, default=0.75)
    parser.add_argument(
        "--positive-only",
        action="store_true",
        help="Exclude random/background records; missing Teacher frames stay masked.",
    )
    args = parser.parse_args()

    cfg = football.load_config(args.config, [])
    football.configure_label_schema(cfg)
    intervals: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    video_paths: dict[tuple[str, str], str] = {}
    record_counts: dict[str, int] = {}
    clip_duration = float(cfg.video.get("clip_duration", 10.0))
    train_jitter = float(cfg.video.get("temporal_jitter_sec", 0.0))
    for split in dict.fromkeys(args.split):
        records, _ = football.load_long_video_records(cfg, split)
        record_counts[split] = len(records)
        for record in records:
            if args.positive_only and bool(record.is_negative):
                continue
            key = (record.source, record.video_id)
            jitter = train_jitter if split == "train" else 0.0
            start = max(
                0.0,
                float(record.base_clip_start) - jitter - args.extra_margin_sec,
            )
            end = min(
                float(record.video_duration),
                float(record.base_clip_start)
                + clip_duration
                + jitter
                + args.extra_margin_sec,
            )
            intervals[key].append((start, end))
            video_paths[key] = record.video_path

    videos = []
    union_seconds = 0.0
    for (source, video_id), rows in sorted(intervals.items()):
        merged = merge_intervals(rows, args.merge_gap_sec)
        duration = sum(end - start for start, end in merged)
        union_seconds += duration
        videos.append(
            {
                "source": source,
                "video_id": video_id,
                "video_path": video_paths[(source, video_id)],
                "intervals": [
                    {"start_sec": start, "end_sec": end} for start, end in merged
                ],
                "union_seconds": duration,
                "teacher_frames": int(round(duration * args.teacher_fps)),
            }
        )
    payload = {
        "schema_version": 1,
        "config": str(Path(args.config).resolve()),
        "splits": list(dict.fromkeys(args.split)),
        "record_counts": record_counts,
        "teacher_fps": args.teacher_fps,
        "merge_gap_sec": args.merge_gap_sec,
        "extra_margin_sec": args.extra_margin_sec,
        "positive_only": bool(args.positive_only),
        "video_count": len(videos),
        "interval_count": sum(len(item["intervals"]) for item in videos),
        "union_seconds": union_seconds,
        "union_hours": union_seconds / 3600.0,
        "teacher_frames": int(round(union_seconds * args.teacher_fps)),
        "videos": videos,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in payload.items() if key != "videos"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
