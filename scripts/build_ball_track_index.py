#!/usr/bin/env python
"""Convert JSONL tracked-ball pseudo-labels into per-video numeric indices."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np


SOURCE_CODES = {
    "unknown": 0,
    "track_observed": 1,
    "raw_repair": 2,
    "track_interpolated": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--exclude-video-ids", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_excluded(path: Path | None) -> set[str]:
    if path is None:
        return set()
    values = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if value and not value.startswith("#"):
            values.add(value)
    return values


def convert_video(source: Path, destination: Path) -> dict[str, object]:
    timestamps: list[float] = []
    boxes: list[list[float]] = []
    confidences: list[float] = []
    quality_weights: list[float] = []
    flags: list[int] = []
    source_codes: list[int] = []
    source_counts: Counter[str] = Counter()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            timestamp = float(row["timestamp_sec"])
            box = [float(value) for value in row["bbox_xyxy_norm"]]
            if len(box) != 4 or not np.isfinite(box).all():
                raise ValueError(f"{source}:{line_number}: invalid normalized box")
            if min(box) < -1e-4 or max(box) > 1.0001 or box[2] < box[0] or box[3] < box[1]:
                raise ValueError(f"{source}:{line_number}: out-of-range normalized box {box}")
            source_name = str(row.get("source", "unknown"))
            timestamps.append(timestamp)
            boxes.append(box)
            confidences.append(float(row.get("confidence", 0.0)))
            quality_weights.append(float(row.get("quality_weight", 0.0)))
            flag = int(bool(row.get("usable_for_heatmap", False)))
            flag |= int(bool(row.get("usable_for_motion", False))) << 1
            flags.append(flag)
            source_codes.append(SOURCE_CODES.get(source_name, 0))
            source_counts[source_name] += 1
    timestamp_array = np.asarray(timestamps, dtype=np.float32)
    if len(timestamp_array) and np.any(np.diff(timestamp_array) < 0):
        raise ValueError(f"{source}: timestamps are not monotonic")
    arrays = {
        "timestamp_sec": timestamp_array,
        "bbox_xyxy_norm": np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        "confidence": np.asarray(confidences, dtype=np.float16),
        "quality_weight": np.asarray(quality_weights, dtype=np.float16),
        "flags": np.asarray(flags, dtype=np.uint8),
        "source_code": np.asarray(source_codes, dtype=np.uint8),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.npz")
    np.savez(temporary, **arrays)
    temporary.replace(destination)
    return {
        "rows": len(timestamp_array),
        "heatmap_usable": int(np.count_nonzero(arrays["flags"] & 1)),
        "motion_usable": int(np.count_nonzero(arrays["flags"] & 2)),
        "source_counts": dict(sorted(source_counts.items())),
        "start_sec": float(timestamp_array[0]) if len(timestamp_array) else None,
        "end_sec": float(timestamp_array[-1]) if len(timestamp_array) else None,
    }


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    excluded = load_excluded(args.exclude_video_ids)
    sources = sorted(input_root.glob("*/ball_pseudolabels.jsonl"))
    if not sources:
        raise SystemExit(f"no ball_pseudolabels.jsonl files under {input_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    videos: dict[str, dict[str, object]] = {}
    skipped_excluded: list[str] = []
    for position, source in enumerate(sources, start=1):
        video_id = source.parent.name
        if video_id in excluded:
            skipped_excluded.append(video_id)
            continue
        destination = output_root / f"{video_id}.npz"
        if destination.is_file() and not args.overwrite:
            with np.load(destination, allow_pickle=False) as archive:
                result = {
                    "rows": int(len(archive["timestamp_sec"])),
                    "heatmap_usable": int(np.count_nonzero(archive["flags"] & 1)),
                    "motion_usable": int(np.count_nonzero(archive["flags"] & 2)),
                    "source_counts": {},
                }
        else:
            result = convert_video(source, destination)
        videos[video_id] = result
        if position % 10 == 0 or position == len(sources):
            print(f"indexed {position}/{len(sources)} videos", flush=True)
    aggregate = {
        "videos": len(videos),
        "rows": sum(int(value["rows"]) for value in videos.values()),
        "heatmap_usable": sum(int(value["heatmap_usable"]) for value in videos.values()),
        "motion_usable": sum(int(value["motion_usable"]) for value in videos.values()),
    }
    manifest = {
        "schema": "offline-tracked-ball-index-v1",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "excluded_video_ids": sorted(excluded),
        "skipped_excluded": skipped_excluded,
        "aggregate": aggregate,
        "videos": videos,
        "elapsed_sec": time.time() - started,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), **aggregate}, ensure_ascii=False))


if __name__ == "__main__":
    main()
