#!/usr/bin/env python3
"""Augment a multi-label review manifest with recall-oriented whistle evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def whistle_id(video_id: str, time_sec: float) -> str:
    digest = hashlib.sha1(f"{video_id}|whistle|{time_sec:.3f}".encode()).hexdigest()[:14]
    return f"{video_id}_whistle_set_piece_{digest}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--whistle-candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--standalone-min-score", type=float, default=0.65)
    args = parser.parse_args()

    manifest = json.loads(args.base_manifest.read_text(encoding="utf-8"))
    videos = {str(item["video_id"]): item for item in manifest["videos"]}
    rows = read_csv(args.whistle_candidates)
    flagged_segments: set[tuple[str, int]] = set()
    added = 0
    skipped = 0

    for row in rows:
        video_id = str(row["video_id"])
        video = videos.get(video_id)
        if video is None:
            continue
        signal = {
            "time_sec": float(row["time_sec"]),
            "score": float(row["score"]),
            "view_start_sec": float(row["start_sec"]),
            "view_end_sec": float(row["end_sec"]),
        }
        if row["ui_action"] == "flag_existing_segment" and row.get("existing_segment_index", ""):
            segment_index = int(row["existing_segment_index"])
            flagged_segments.add((video_id, segment_index))
            for event in video["events"]:
                if int(event.get("segment_index", -1)) == segment_index:
                    event.setdefault("whistle_signals", []).append(signal)
                    event["whistle_score"] = max(float(event.get("whistle_score", 0.0)), signal["score"])
                    event["review_source"] = "dino_frame_whistle"
            continue

        if row["ui_action"] != "new_review_segment" or signal["score"] < args.standalone_min_score:
            skipped += int(row["ui_action"] == "new_review_segment")
            continue

        timeline = [
            item for item in video.get("timeline", [])
            if float(item["end_sec"]) >= signal["view_start_sec"]
            and float(item["start_sec"]) <= signal["view_end_sec"]
        ]
        labels = list(manifest.get("labels", ["shot", "save", "set_piece"]))
        dino = {label: max([float(item.get("dino", {}).get(label, 0.0)) for item in timeline] or [0.0]) for label in labels}
        frame = {label: max([float(item.get("frame_detection", {}).get(label, 0.0)) for item in timeline] or [0.0]) for label in labels}
        event = {
            "id": whistle_id(video_id, signal["time_sec"]),
            "segment_id": f"{video_id}_whistle_{signal['time_sec']:.3f}",
            "segment_index": -1,
            "segment_labels": ["set_piece"],
            "video_id": video_id,
            "label": "set_piece",
            "time_sec": signal["time_sec"],
            "start_sec": signal["view_start_sec"],
            "end_sec": signal["view_end_sec"],
            "support_start_sec": signal["view_start_sec"],
            "support_end_sec": signal["view_end_sec"],
            "score": dino.get("set_piece", 0.0),
            "dino_scores": dino,
            "frame_detection_scores": frame,
            "whistle_signals": [signal],
            "whistle_score": signal["score"],
            "window_indices": [int(item["index"]) for item in timeline],
            "merged_predictions": len(timeline),
            "evaluation_status": "whistle_rescue",
            "review_source": "whistle_rescue",
            "review_protocol": "recall_first_whistle_rescue",
        }
        video["events"].append(event)
        added += 1

    for video in manifest["videos"]:
        video["events"].sort(
            key=lambda item: (
                float(item["start_sec"]),
                1 if item.get("review_source") == "whistle_rescue" else 0,
                str(item["label"]),
            )
        )

    manifest["schema_version"] = max(3, int(manifest.get("schema_version", 1)))
    manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    manifest.setdefault("source", {}).update({
        "queue_strategy": "recall_first_d7_frame_plus_whistle",
        "whistle_candidates": str(args.whistle_candidates.resolve()),
        "whistle_standalone_min_score": args.standalone_min_score,
        "whistle_existing_segment_policy": "flag_all_without_extra_playback",
    })
    manifest["summary"].update({
        "whistle_flagged_review_segments": len(flagged_segments),
        "whistle_standalone_segments_added": added,
        "whistle_standalone_segments_filtered": skipped,
        "num_segment_label_decisions_with_whistle": sum(len(video["events"]) for video in manifest["videos"]),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
