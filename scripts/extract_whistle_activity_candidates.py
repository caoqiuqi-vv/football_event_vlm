#!/usr/bin/env python
"""Convert per-video audio features into whistle activity proposals.

The operation groups contiguous acoustic activity, not football event
predictions.  It therefore does not suppress or merge visual event slots.  A
proposal is always routed to human review unless a later calibrated multimodal
model confirms it.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def smooth(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or len(values) == 0:
        return values.astype(np.float64, copy=True)
    kernel = np.ones(2 * radius + 1, dtype=np.float64)
    numerator = np.convolve(values, kernel, mode="same")
    denominator = np.convolve(np.ones_like(values), kernel, mode="same")
    return numerator / np.maximum(denominator, 1.0)


def connected_activity(
    times: np.ndarray, scores: np.ndarray, *, threshold: float, max_gap_sec: float,
) -> list[dict[str, float]]:
    active = np.flatnonzero(scores >= threshold)
    if len(active) == 0:
        return []
    groups: list[list[int]] = [[int(active[0])]]
    for index in active[1:]:
        index = int(index)
        if float(times[index] - times[groups[-1][-1]]) <= max_gap_sec:
            groups[-1].append(index)
        else:
            groups.append([index])
    rows: list[dict[str, float]] = []
    for group in groups:
        values = scores[group]
        peak_local = int(np.argmax(values))
        peak_index = group[peak_local]
        rows.append({
            "start_sec": float(times[group[0]]),
            "end_sec": float(times[group[-1]]),
            "peak_time_sec": float(times[peak_index]),
            "peak_score": float(scores[peak_index]),
            "mean_score": float(np.mean(values)),
            "activity_points": len(group),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-index-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--video-ids", type=Path)
    parser.add_argument("--minimum-score", type=float, default=-0.25)
    parser.add_argument("--smooth-radius", type=int, default=1)
    parser.add_argument("--max-gap-sec", type=float, default=0.45)
    args = parser.parse_args()
    if args.video_ids:
        video_ids = [
            line.strip() for line in args.video_ids.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    else:
        video_ids = sorted(path.stem for path in args.audio_index_dir.glob("*.npz"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "schema": "football_whistle_activity.v1",
        "visual_event_slots_unchanged": True,
        "temporal_nms": False,
        "grouping": "connected acoustic activity only",
        "minimum_score": args.minimum_score,
        "smooth_radius": args.smooth_radius,
        "max_gap_sec": args.max_gap_sec,
        "videos": {},
    }
    for video_id in video_ids:
        path = args.audio_index_dir / f"{video_id}.npz"
        if not path.is_file():
            summary["videos"][video_id] = {"audio_valid": False, "candidates": 0}
            continue
        data = np.load(path, allow_pickle=False)
        times = data["times"].astype(np.float64)
        features = data["feats"].astype(np.float64)
        valid = len(times) > 0 and features.shape == (len(times), 5)
        rows: list[dict[str, object]] = []
        if valid:
            # Narrow-band peakiness is primary; high-frequency ratio supports it
            # while a large broadband energy flux is slightly downweighted.
            score = smooth(
                features[:, 0] + 0.35 * features[:, 2] - 0.10 * features[:, 3],
                args.smooth_radius,
            )
            for index, row in enumerate(connected_activity(
                times, score, threshold=args.minimum_score,
                max_gap_sec=args.max_gap_sec,
            )):
                rows.append({"video_id": video_id, "activity_id": index, **row})
        output = args.output_dir / f"{video_id}_whistles.csv"
        fields = [
            "video_id", "activity_id", "start_sec", "end_sec",
            "peak_time_sec", "peak_score", "mean_score", "activity_points",
        ]
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        summary["videos"][video_id] = {"audio_valid": bool(valid), "candidates": len(rows)}
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "videos": len(video_ids),
        "valid": sum(bool(item["audio_valid"]) for item in summary["videos"].values()),
        "candidates": sum(int(item["candidates"]) for item in summary["videos"].values()),
        "output": str(args.output_dir),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
