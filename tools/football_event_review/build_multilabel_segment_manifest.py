#!/usr/bin/env python3
"""Build review UI input from the recall-preserving multi-label Segment protocol."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from build_review_manifest import DEFAULT_LABELS, find_video, read_csv, as_float


def literal(value: str, expected: type):
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, expected):
        raise ValueError(f"expected {expected.__name__}, got {type(parsed).__name__}: {value}")
    return parsed


def event_id(video_id: str, segment_index: int, label: str, start: float, end: float) -> str:
    digest = hashlib.sha1(f"{video_id}|{segment_index}|{label}|{start:.3f}|{end:.3f}".encode()).hexdigest()[:14]
    return f"{video_id}_segment{segment_index:05d}_{label}_{digest}"


def build_timeline(eval_dir: Path, labels: tuple[str, ...]) -> tuple[list[dict], dict[tuple[int, str], float]]:
    frame_scores: dict[tuple[int, str], float] = {}
    for row in read_csv(eval_dir / "frame_event_window_scores.csv"):
        if row.get("branch", "global") != "global":
            continue
        key = (int(row["window_index"]), row["label"])
        frame_scores[key] = max(frame_scores.get(key, 0.0), as_float(row.get("max_frame_prob")))
    timeline = []
    for row in read_csv(eval_dir / "window_predictions.csv"):
        index = int(row["index"])
        timeline.append({
            "index": index, "start_sec": as_float(row["start_sec"]), "end_sec": as_float(row["end_sec"]),
            "dino": {label: as_float(row.get(f"prob_{label}")) for label in labels},
            "frame_detection": {label: frame_scores.get((index, label), 0.0) for label in labels},
            "roi": {"valid": as_float(row.get("roi_valid")) > .5, "confidence": as_float(row.get("roi_confidence")), "mode": row.get("roi_proposal_mode", "")},
        })
    return timeline, frame_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--review-segments", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--tolerance-sec", type=float, default=2.0)
    parser.add_argument("--evaluation-filter", choices=("all", "fp", "matched"), default="all")
    args = parser.parse_args()
    labels = tuple(item.strip() for item in args.labels.split(",") if item.strip())
    segment_rows = read_csv(args.review_segments)
    by_video: dict[str, list[dict[str, str]]] = {}
    for row in segment_rows:
        by_video.setdefault(str(row["video_id"]), []).append(row)

    videos = []
    missing = []
    for video_id, rows in sorted(by_video.items()):
        video_path = find_video(args.video_root, video_id)
        if video_path is None:
            missing.append(video_id)
            continue
        eval_dir = args.run_dir / video_id
        timeline, frame_scores = build_timeline(eval_dir, labels)
        gt = read_csv(eval_dir / "gt_events.csv")
        events = []
        for row in rows:
            start, end = as_float(row["start_sec"]), as_float(row["end_sec"])
            segment_index = int(row["segment_index"])
            active_labels = [str(item) for item in literal(row["labels"], list)]
            scores = {str(k): float(v) for k, v in literal(row["scores"], dict).items()}
            window_indices = [int(item) for item in literal(row["source_window_indices"], list)]
            for label in active_labels:
                matches = [
                    as_float(item["time_sec"]) for item in gt
                    if item.get("label") == label and start - args.tolerance_sec <= as_float(item["time_sec"]) <= end + args.tolerance_sec
                ]
                evaluation = "matched" if matches else "fp"
                if args.evaluation_filter != "all" and evaluation != args.evaluation_filter:
                    continue
                score = scores[label]
                representative = max(
                    (item for item in timeline if item["index"] in window_indices),
                    key=lambda item: (item["dino"].get(label, 0.0), -item["start_sec"]),
                    default={"start_sec": start, "end_sec": end},
                )
                time_sec = (representative["start_sec"] + representative["end_sec"]) / 2
                events.append({
                    "id": event_id(video_id, segment_index, label, start, end),
                    "segment_id": f"{video_id}_segment{segment_index:05d}",
                    "segment_index": segment_index, "segment_labels": active_labels,
                    "video_id": video_id, "label": label, "time_sec": time_sec,
                    "start_sec": start, "end_sec": end, "support_start_sec": start, "support_end_sec": end,
                    "score": score, "dino_scores": scores,
                    "frame_detection_scores": {
                        item: max([frame_scores.get((index, item), 0.0) for index in window_indices] or [0.0])
                        for item in labels
                    },
                    "window_indices": window_indices, "merged_predictions": len(window_indices),
                    "evaluation_status": evaluation, "matching_gt_times": matches,
                    "match_tolerance_sec": args.tolerance_sec,
                    "review_protocol": "multilabel_review_segment",
                })
        videos.append({
            "video_id": video_id, "video_path": str(video_path),
            "duration_sec": max((item["end_sec"] for item in timeline), default=0.0),
            "events": sorted(events, key=lambda item: (item["start_sec"], item["label"])),
            "timeline": timeline,
        })

    manifest = {
        "schema_version": 2, "created_at": datetime.now(timezone.utc).isoformat(),
        "labels": labels,
        "review_labels": ["shot", "save", "set_piece"],
        "source": {
            "run_dir": str(args.run_dir.resolve()), "video_root": str(args.video_root.resolve()),
            "review_segments": str(args.review_segments.resolve()),
            "protocol": "multilabel_review_segment", "match_tolerance_sec": args.tolerance_sec,
            "evaluation_filter": args.evaluation_filter,
            "definition": "one queue item per (Segment,label); labels from the same Segment share playback bounds",
        },
        "videos": videos,
        "summary": {
            "num_videos": len(videos), "num_review_segments": len(segment_rows),
            "num_segment_label_decisions": sum(len(video["events"]) for video in videos),
            "missing_videos": missing,
            "events_by_evaluation": {
                status: sum(item["evaluation_status"] == status for video in videos for item in video["events"])
                for status in ("fp", "matched")
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
