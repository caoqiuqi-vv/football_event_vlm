#!/usr/bin/env python
"""Watch dual-Teacher shards, build v3 indices, then launch guarded training."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def completed_count(path: Path, allowed_video_ids: set[str] | None = None) -> int:
    if not path.is_file():
        return 0
    completed = json.loads(path.read_text()).get("completed", [])
    if allowed_video_ids is None:
        return len(completed)
    return sum(str(key).partition(":")[0] in allowed_video_ids for key in completed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--ball-root", required=True)
    parser.add_argument("--goal-root", required=True)
    parser.add_argument("--index-root", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--ball-shards", type=int, default=3)
    parser.add_argument("--goal-shards", type=int, default=1)
    parser.add_argument("--train-gpus", default="0,1,6")
    parser.add_argument(
        "--ball-checkpoint",
        default="/mnt/data_7t/qiuqi/code/soccer/onlysoccer_1920_11s.pt",
    )
    parser.add_argument(
        "--goal-checkpoint",
        default="/home/new_users/qiuqi/code/det_and_track/checkpoints/yolo11m_person_goal_fieldLine_1920_7.22.pt",
    )
    parser.add_argument("--poll-sec", type=float, default=30.0)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text())
    expected_ball = [0 for _ in range(args.ball_shards)]
    expected_goal = [0 for _ in range(args.goal_shards)]
    ball_video_ids = [set() for _ in range(args.ball_shards)]
    goal_video_ids = [set() for _ in range(args.goal_shards)]
    for video_index, video in enumerate(manifest["videos"]):
        count = len(video["intervals"])
        video_id = str(video["video_id"])
        expected_ball[video_index % args.ball_shards] += count
        expected_goal[video_index % args.goal_shards] += count
        ball_video_ids[video_index % args.ball_shards].add(video_id)
        goal_video_ids[video_index % args.goal_shards].add(video_id)
    ball_root, goal_root = Path(args.ball_root), Path(args.goal_root)
    status_path = Path(args.index_root).parent / "dual_teacher_watchdog_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        actual_ball = [
            completed_count(ball_root / f"shard_{index:02d}_state.json", ball_video_ids[index])
            for index in range(args.ball_shards)
        ]
        actual_goal = [
            completed_count(goal_root / f"shard_{index:02d}_state.json", goal_video_ids[index])
            for index in range(args.goal_shards)
        ]
        status = {
            "state": "waiting_teachers",
            "ball": {"completed": actual_ball, "expected": expected_ball},
            "goal": {"completed": actual_goal, "expected": expected_goal},
            "updated_at_unix": time.time(),
        }
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(status, ensure_ascii=False), flush=True)
        if actual_ball == expected_ball and actual_goal == expected_goal:
            break
        time.sleep(max(min(args.poll_sec, 60.0), 5.0))

    status["state"] = "building_indices"
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    subprocess.run(
        [
            sys.executable,
            "scripts/build_football_dual_teacher_indices.py",
            "--sampling-manifest",
            str(manifest_path),
            "--ball-root",
            str(ball_root.resolve()),
            "--goal-root",
            str(goal_root.resolve()),
            "--output-root",
            str(Path(args.index_root).resolve()),
            "--ball-checkpoint",
            str(Path(args.ball_checkpoint).resolve()),
            "--goal-checkpoint",
            str(Path(args.goal_checkpoint).resolve()),
            "--ball-teacher-name",
            "YOLO onlysoccer_1920_11s",
            "--ball-confidence",
            "0.10",
            "--goal-confidence",
            "0.25",
        ],
        check=True,
    )
    index_summary = json.loads((Path(args.index_root) / "summary.json").read_text())
    if int(index_summary.get("video_count", 0)) != int(manifest["video_count"]):
        raise RuntimeError(f"incomplete dual Teacher index: {index_summary}")
    train_gpus = [value.strip() for value in args.train_gpus.split(",") if value.strip()]
    if not train_gpus:
        raise ValueError("--train-gpus must contain at least one GPU")
    status.update({
        "state": "training",
        "index_summary": index_summary,
        "training_started_at_unix": time.time(),
        "train_gpus": train_gpus,
    })
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(train_gpus)
    train_command = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={len(train_gpus)}",
        "train_football_events.py",
        "--config",
        str(Path(args.train_config).resolve()),
    ]
    try:
        subprocess.run(train_command, env=env, check=True)
    except Exception as error:
        status.update({
            "state": "failed",
            "failure_stage": "training",
            "failed_at_unix": time.time(),
            "error": f"{type(error).__name__}: {error}",
            "train_command": train_command,
        })
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
        raise
    status["state"] = "complete"
    status["completed_at_unix"] = time.time()
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
