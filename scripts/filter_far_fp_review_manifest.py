#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter a football review manifest to far FP / no-annotation candidates.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--far-sec", type=float, default=15.0)
    parser.add_argument("--max-events", type=int, default=100)
    parser.add_argument("--max-per-video-label", type=int, default=4)
    parser.add_argument("--calibration-video-id-file", type=Path)
    parser.add_argument("--holdout-video-id-file", type=Path)
    return parser.parse_args()


def parse_time(raw: Any) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip()
    if not text:
        return None
    if ":" in text:
        parts = [float(x) for x in text.split(":")]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
    try:
        return float(text)
    except ValueError:
        return None


def annotation_times(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        rows = data.get("events", data.get("annotations", []))
    else:
        rows = data
    out = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        t = parse_time(row.get("time_sec", row.get("timestamp", row.get("start_sec"))))
        if t is None:
            continue
        out.append({"time_sec": t, "label": row.get("label", row.get("event_type", "")), "id": row.get("id", "")})
    return out


def read_ids(path: Path | None) -> set[str]:
    if not path or not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip() and not line.strip().startswith("#")}


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.input.read_text())
    cal_ids = read_ids(args.calibration_video_id_file)
    holdout_ids = read_ids(args.holdout_video_id_file)
    ann_cache: dict[str, list[dict[str, Any]]] = {}
    candidates = []
    for video in manifest.get("videos", []):
        video_id = str(video.get("video_id"))
        anns = ann_cache.setdefault(video_id, annotation_times(args.annotations_dir / f"{video_id}.json"))
        split = "holdout6" if video_id in holdout_ids else "cal29" if video_id in cal_ids else "unknown"
        for event in video.get("events", []):
            t = float(event.get("time_sec", 0.0) or 0.0)
            nearest = min((abs(t - float(ann["time_sec"])) for ann in anns), default=1e9)
            if nearest <= args.far_sec:
                continue
            item = dict(event)
            item["nearest_annotation_distance_sec"] = nearest if nearest < 1e8 else None
            item["review_split"] = split
            candidates.append((video_id, item))
    candidates.sort(key=lambda pair: (pair[1].get("review_split") != "holdout6", pair[1].get("label", ""), -float(pair[1].get("score", 0.0) or 0.0), pair[0], float(pair[1].get("time_sec", 0.0) or 0.0)))
    selected_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counts = Counter()
    for video_id, event in candidates:
        key = (video_id, event.get("label", ""))
        if counts[key] >= args.max_per_video_label:
            continue
        selected_by_video[video_id].append(event)
        counts[key] += 1
        if sum(len(v) for v in selected_by_video.values()) >= args.max_events:
            break
    new_videos = []
    for video in manifest.get("videos", []):
        video_id = str(video.get("video_id"))
        events = selected_by_video.get(video_id, [])
        if not events:
            continue
        out_video = dict(video)
        out_video["events"] = events
        new_videos.append(out_video)
    manifest["videos"] = new_videos
    manifest.setdefault("source", {})["far_fp_filter"] = {
        "far_sec": args.far_sec,
        "max_events": args.max_events,
        "max_per_video_label": args.max_per_video_label,
        "annotations_dir": str(args.annotations_dir),
        "note": "Candidates are FP predictions farther than far_sec from any annotation label in the repaired annotation file.",
    }
    manifest["summary"] = {
        "num_videos": len(new_videos),
        "num_events": sum(len(v.get("events", [])) for v in new_videos),
        "events_by_label": dict(Counter(e.get("label", "") for v in new_videos for e in v.get("events", []))),
        "events_by_split": dict(Counter(e.get("review_split", "unknown") for v in new_videos for e in v.get("events", []))),
        "num_candidates_before_cap": len(candidates),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
