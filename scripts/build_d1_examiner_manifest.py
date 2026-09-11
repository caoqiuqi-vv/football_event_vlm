#!/usr/bin/env python
"""Build the D1 candidate examiner feature-extraction manifest (CPU only).

Spec: docs/football_tail_first_fullchain_plan_20260910.md §5 (D1 候选点高帧率核验器).

For every candidate in the V2 candidate list
(outputs/football_candidate_verifiers/v2_full166_20260910/candidates.csv) of the
protocol splits (train / val15 / test18; the 4 "unused" videos are skipped, same
row filter as scripts/fit_football_verifier_v2.py), we record one manifest row:

  candidate_idx   global int id (used as npz key by the extractor)
  video_id / split / label / window_index
  peak_time_sec   candidate peak time = window center (V2 时间轴口径)
  target          0/1 ('' for is_ignored rows; the ignore band itself is dropped
                  from the manifest columns — target=='' is the same information)
  priority_score  peak_score (extraction priority)
  video_path      resolved raw 720p video path

Sampling contract shared with the extractor: 48 frames at 8 fps covering
peak_time ± 3s; frame times are the 48 bin centers of [-3s, +3s), i.e.
t_i = peak_time + (i - 23.5) / 8 for i in 0..47 (symmetric around the peak).

Outputs:
  manifest_20260911.parquet   the manifest
  manifest_stats.json         split/label counts, frame totals, storage and
                              extraction-cost estimates, missing-video list
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ("shot", "save", "set_piece")
PROTOCOL_SPLITS = ("train", "val15", "test18")

WINDOW_HALF_SEC = 3.0
SAMPLE_FPS = 8.0
NUM_FRAMES = 48
EMBED_DIM = 1024  # DINOv3 ViT-L/16

VIDEO_ROOTS = (
    Path("/mnt/data_16t/football/raw_video_720P"),
    Path("/mnt/data_16t/football/raw_video_hq_720P"),
)


def resolve_video_path(video_id: str) -> Path | None:
    for root in VIDEO_ROOTS:
        path = root / f"{video_id}.mp4"
        if path.is_file():
            return path
    return None


def sample_offsets_sec() -> np.ndarray:
    # 48 bin centers of [-3s, +3s): symmetric around the peak, spacing 1/8 s.
    return (np.arange(NUM_FRAMES, dtype=np.float64) - (NUM_FRAMES - 1) / 2.0) / SAMPLE_FPS


def union_seconds(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    total = 0.0
    cur_lo, cur_hi = intervals[0]
    for lo, hi in intervals[1:]:
        if lo <= cur_hi:
            cur_hi = max(cur_hi, hi)
        else:
            total += cur_hi - cur_lo
            cur_lo, cur_hi = lo, hi
    total += cur_hi - cur_lo
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates-csv",
        type=Path,
        default=Path("outputs/football_candidate_verifiers/v2_full166_20260910/candidates.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/football_d1_examiner"))
    parser.add_argument("--manifest-name", default="manifest_20260911")
    parser.add_argument(
        "--probe-videos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="open each referenced video with cv2 to record fps/frame_count/duration",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    skipped: dict[str, int] = {}
    with args.candidates_csv.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            split = raw["split"]
            if split not in PROTOCOL_SPLITS:
                skipped[split] = skipped.get(split, 0) + 1
                continue
            target_raw = raw.get("target", "")
            rows.append(
                {
                    "video_id": raw["video_id"],
                    "split": split,
                    "label": raw["label"],
                    "window_index": int(float(raw["window_index"])),
                    "peak_time_sec": float(raw["peak_time_sec"]),
                    "target": -1 if target_raw in ("", None) else int(target_raw),
                    "priority_score": float(raw["peak_score"]),
                }
            )
    if not rows:
        raise ValueError(f"no protocol-split candidates found in {args.candidates_csv}")

    frame = pd.DataFrame(rows)
    frame.insert(0, "candidate_idx", np.arange(len(frame), dtype=np.int64))

    # --- resolve video paths -------------------------------------------------
    video_ids = sorted(frame["video_id"].unique())
    path_map: dict[str, str] = {}
    missing: list[str] = []
    for video_id in video_ids:
        path = resolve_video_path(video_id)
        if path is None:
            missing.append(video_id)
        else:
            path_map[video_id] = str(path)
    frame["video_path"] = frame["video_id"].map(path_map)
    n_missing_rows = int(frame["video_path"].isna().sum())

    # --- probe video metadata (fps / frames / duration) ----------------------
    probe: dict[str, dict[str, float]] = {}
    if args.probe_videos:
        import cv2

        for video_id in video_ids:
            path = path_map.get(video_id)
            if path is None:
                continue
            cap = cv2.VideoCapture(path)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
            probe[video_id] = {
                "fps": fps,
                "frame_count": frame_count,
                "duration_sec": frame_count / fps if fps > 0 else 0.0,
            }

    # --- per-video candidate-window union coverage (extraction cost driver) --
    offsets = sample_offsets_sec()
    union_by_video: dict[str, float] = {}
    for video_id, group in frame.groupby("video_id"):
        intervals = [
            (float(t + offsets[0]), float(t + offsets[-1]))
            for t in group["peak_time_sec"].to_numpy()
        ]
        union_by_video[video_id] = union_seconds(intervals)

    # --- write manifest ------------------------------------------------------
    manifest_path = args.output_dir / f"{args.manifest_name}.parquet"
    frame.to_parquet(manifest_path, index=False)

    # --- stats ---------------------------------------------------------------
    split_label_counts = {
        split: {label: int(((frame["split"] == split) & (frame["label"] == label)).sum()) for label in LABELS}
        for split in PROTOCOL_SPLITS
    }
    split_target_counts = {
        split: {
            "pos": int(((frame["split"] == split) & (frame["target"] == 1)).sum()),
            "neg": int(((frame["split"] == split) & (frame["target"] == 0)).sum()),
            "ignored": int(((frame["split"] == split) & (frame["target"] == -1)).sum()),
        }
        for split in PROTOCOL_SPLITS
    }
    n_candidates = len(frame)
    total_frames = n_candidates * NUM_FRAMES
    bytes_fp16 = 2
    storage = {
        "cls_only_1024d": {
            "bytes_per_candidate": NUM_FRAMES * EMBED_DIM * bytes_fp16,
            "total_gib": round(total_frames * EMBED_DIM * bytes_fp16 / 2**30, 2),
        },
        "cls_patchmean_2048d": {
            "bytes_per_candidate": NUM_FRAMES * 2 * EMBED_DIM * bytes_fp16,
            "total_gib": round(total_frames * 2 * EMBED_DIM * bytes_fp16 / 2**30, 2),
        },
    }
    total_union_sec = sum(union_by_video.values())
    total_duration_sec = sum(p["duration_sec"] for p in probe.values())
    stats = {
        "created": "2026-09-11",
        "candidates_csv": str(args.candidates_csv),
        "manifest_path": str(manifest_path),
        "sampling": {
            "num_frames": NUM_FRAMES,
            "sample_fps": SAMPLE_FPS,
            "window_half_sec": WINDOW_HALF_SEC,
            "frame_time_formula": "t_i = peak_time_sec + (i - 23.5) / 8, i in 0..47",
        },
        "candidates_total": n_candidates,
        "candidates_per_split_label": split_label_counts,
        "candidates_per_split_target": split_target_counts,
        "skipped_rows_by_split": skipped,
        "videos": {
            split: int((frame.loc[frame["split"] == split, "video_id"]).nunique())
            for split in PROTOCOL_SPLITS
        },
        "videos_total": len(video_ids),
        "missing_videos": missing,
        "missing_candidate_rows": n_missing_rows,
        "total_candidate_frames": total_frames,
        "storage_estimate_fp16": storage,
        "extraction_cost_estimate": {
            "union_window_seconds_all_videos": round(total_union_sec, 1),
            "probed_video_duration_seconds": round(total_duration_sec, 1),
            "union_coverage_frac": (
                round(total_union_sec / total_duration_sec, 4) if total_duration_sec else None
            ),
            "unique_frames_at_8fps_deduped": int(round(total_union_sec * SAMPLE_FPS)),
            "note": (
                "all peak_times sit on the 5 s window grid, which is a multiple of the "
                "1/8 s sampling grid, so overlapping candidate windows dedupe exactly "
                "on frame indices; unique frame count bounds the DINO forward count"
            ),
        },
        "video_probe": probe,
        "union_window_sec_by_video": {k: round(v, 1) for k, v in union_by_video.items()},
    }
    stats_path = args.output_dir / "manifest_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"manifest rows: {n_candidates} -> {manifest_path}")
    print(f"videos: {len(video_ids)} (missing: {len(missing)})")
    print(f"per split x label: {json.dumps(split_label_counts)}")
    print(f"per split target: {json.dumps(split_target_counts)}")
    print(f"total frames (no dedup): {total_frames:,}")
    print(
        "fp16 storage: cls_only {0:.2f} GiB | cls+patchmean {1:.2f} GiB".format(
            storage["cls_only_1024d"]["total_gib"], storage["cls_patchmean_2048d"]["total_gib"]
        )
    )
    print(
        f"union coverage: {total_union_sec:.0f}s of {total_duration_sec:.0f}s "
        f"({(total_union_sec / total_duration_sec * 100) if total_duration_sec else 0:.1f}%) -> "
        f"~{int(round(total_union_sec * SAMPLE_FPS)):,} unique frames at 8fps"
    )
    print(f"stats -> {stats_path}")


if __name__ == "__main__":
    main()
