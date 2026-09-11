#!/usr/bin/env python
"""Build a per-video geometric evidence index from detection+track results.

Reads ``detection_and_track_result/<video_id>/compact_detection_tracks.json.gz``
and writes ``<output_dir>/<video_id>.npz`` with:

  frame_ids : int32 [N]   original source frame ids (stride 2)
  fps       : float32 scalar (source fps; frame time = frame_id / fps)
  feats     : float32 [N, 19]

Feature layout (all normalized, 0 when absent):
  0 ball_present   1 ball_x   2 ball_y   3 ball_conf
  4 ball_vx        5 ball_vy  6 ball_speed  7 ball_accel
  8 goal_present   9 goal_cx 10 goal_cy  11 goal_w  12 goal_h
  13 ball_goal_dist  14 ball_goal_vel_cos
  15 keeper_dist   16 crowd_count  17 crowd_goal_count
  18 evidence_valid (1 whenever the frame had any detection row)

Coordinates are normalized by the detection image size, so they transfer
directly to any letterbox-free resize of the same video.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

CLS_PERSON, CLS_BALL, CLS_GOAL = 0, 1, 2
FEAT_DIM = 19


def build_video(path: Path) -> tuple[np.ndarray, float, np.ndarray]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    fps = float(data["metadata"]["sampling"]["source_fps"])
    width = float(data["metadata"]["image_size"]["width"])
    height = float(data["metadata"]["image_size"]["height"])
    diag = float(np.hypot(width, height))
    frames = data.get("frames", [])
    n = len(frames)
    frame_ids = np.asarray([int(f["f"]) for f in frames], dtype=np.int32)
    feats = np.zeros((n, FEAT_DIM), dtype=np.float32)
    for row, frame in enumerate(frames):
        objs = frame.get("o", [])
        if not objs:
            continue
        feats[row, 18] = 1.0
        ball = None
        goal = None
        persons = []
        for obj in objs:
            cls = int(obj[0])
            conf = float(obj[1])
            x1, y1, x2, y2 = (float(v) for v in obj[2:6])
            if cls == CLS_BALL and (ball is None or conf > ball[1]):
                ball = (obj, conf, x1, y1, x2, y2)
            elif cls == CLS_GOAL and (goal is None or conf > goal[1]):
                goal = (obj, conf, x1, y1, x2, y2)
            elif cls == CLS_PERSON:
                persons.append((0.5 * (x1 + x2), 0.5 * (y1 + y2), y2 - y1))
        if ball is not None:
            _, conf, x1, y1, x2, y2 = ball
            feats[row, 0] = 1.0
            feats[row, 1] = (0.5 * (x1 + x2)) / width
            feats[row, 2] = (0.5 * (y1 + y2)) / height
            feats[row, 3] = conf
        if goal is not None:
            _, conf, x1, y1, x2, y2 = goal
            gcx, gcy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
            feats[row, 8] = 1.0
            feats[row, 9] = gcx / width
            feats[row, 10] = gcy / height
            feats[row, 11] = (x2 - x1) / width
            feats[row, 12] = (y2 - y1) / height
            feats[row, 16] = min(len(persons) / 30.0, 1.0)
            if persons:
                dists = [
                    float(np.hypot(px - gcx, py - gcy)) / diag
                    for px, py, _ in persons
                ]
                feats[row, 15] = float(min(dists))
                # crowd near goal: centers inside the goal box expanded 1.5x
                ex1, ex2 = x1 - 0.25 * (x2 - x1), x2 + 0.25 * (x2 - x1)
                ey1, ey2 = y1 - 0.25 * (y2 - y1), y2 + 0.5 * (y2 - y1)
                near = sum(1 for px, py, _ in persons if ex1 <= px <= ex2 and ey1 <= py <= ey2)
                feats[row, 17] = min(near / 8.0, 1.0)
        elif persons:
            feats[row, 16] = min(len(persons) / 30.0, 1.0)
        if ball is not None and goal is not None:
            bx, by = feats[row, 1] * width, feats[row, 2] * height
            gcx, gcy = feats[row, 9] * width, feats[row, 10] * height
            feats[row, 13] = float(np.hypot(bx - gcx, by - gcy)) / diag
    # ball kinematics from the presence-gated position series
    if n >= 3:
        t = frame_ids.astype(np.float64) / fps
        present = feats[:, 0] > 0
        pos = feats[:, 1:3].astype(np.float64).copy()
        # interpolate short gaps (<0.5s) for smoother derivatives
        for dim in range(2):
            series = pos[:, dim].copy()
            series[~present] = np.nan
            valid = np.isfinite(series)
            if valid.sum() >= 2:
                interp = np.interp(t, t[valid], series[valid])
                gap_filled = series
                # only fill gaps shorter than 0.5s
                idx = np.arange(n)
                prev_valid = np.maximum.accumulate(np.where(valid, idx, -1))
                next_valid = np.minimum.accumulate(
                    np.where(valid[::-1], idx[::-1], n)[::-1]
                )
                gap_len = np.zeros(n)
                in_gap = ~valid & (prev_valid >= 0) & (next_valid < n)
                gap_len[in_gap] = t[next_valid[in_gap]] - t[prev_valid[in_gap]]
                filled = np.where(in_gap & (gap_len <= 0.5), interp, series)
                pos[:, dim] = np.where(np.isfinite(filled), filled, 0.0)
        dt = np.gradient(t)
        vel = np.gradient(pos, axis=0) / np.maximum(dt[:, None], 1e-3)
        accel = np.gradient(vel, axis=0) / np.maximum(dt[:, None], 1e-3)
        vel[~present] = 0.0
        accel[~present] = 0.0
        speed = np.linalg.norm(vel, axis=1)
        feats[:, 4] = np.clip(vel[:, 0] / 0.5, -1, 1)   # ~half field width/s
        feats[:, 5] = np.clip(vel[:, 1] / 0.5, -1, 1)
        feats[:, 6] = np.clip(speed / 0.35, 0, 1)
        feats[:, 7] = np.clip(np.linalg.norm(accel, axis=1) / 0.5, 0, 1)
        # cosine between ball velocity and ball->goal direction
        has_both = present & (feats[:, 8] > 0)
        if has_both.any():
            bg = np.stack(
                [feats[:, 9] - feats[:, 1], feats[:, 10] - feats[:, 2]], axis=1
            )
            bg_norm = np.linalg.norm(bg, axis=1, keepdims=True)
            sp = speed[has_both]
            denom = np.maximum(bg_norm[has_both, 0] * sp, 1e-6)
            feats[has_both, 14] = (
                (bg[has_both, 0] * vel[has_both, 0] + bg[has_both, 1] * vel[has_both, 1])
                / denom
            )
    return frame_ids, fps, feats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--det-root", type=Path,
                        default=Path("/mnt/data_16t/football/detection_and_track_result"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video-ids", type=Path, default=None,
                        help="optional text file; default: all videos under det-root")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.video_ids:
        video_ids = [l.strip() for l in args.video_ids.read_text().splitlines() if l.strip()]
    else:
        video_ids = sorted(
            p.name for p in args.det_root.iterdir()
            if (p / "compact_detection_tracks.json.gz").exists()
        )
    done = skipped = 0
    for vid in video_ids:
        src = args.det_root / vid / "compact_detection_tracks.json.gz"
        dst = args.output_dir / f"{vid}.npz"
        if not src.exists():
            skipped += 1
            continue
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            done += 1
            continue
        frame_ids, fps, feats = build_video(src)
        np.savez_compressed(dst, frame_ids=frame_ids, fps=np.float32(fps), feats=feats)
        done += 1
    print(f"evidence index: wrote/kept {done}, missing source {skipped}, "
          f"dim={FEAT_DIM}, dir={args.output_dir}")


if __name__ == "__main__":
    main()
