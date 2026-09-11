#!/usr/bin/env python3
"""Watch and restart a resumable per-video dense evaluation tmux session."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--video-id-file", type=Path, required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--launch-script", type=Path, required=True)
    parser.add_argument("--poll-sec", type=float, default=60.0)
    parser.add_argument("--max-restarts", type=int, default=20)
    return parser.parse_args()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def session_alive(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def completed(ids: list[str], run_dir: Path) -> list[str]:
    required = ("summary.json", "metrics.json", "window_predictions.csv", "gt_events.json")
    return [
        video_id
        for video_id in ids
        if all((run_dir / video_id / filename).is_file() for filename in required)
    ]


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    ids = [
        line.strip()
        for line in args.video_id_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    log_path = run_dir / "dense_watchdog.jsonl"
    marker = run_dir / "DENSE_COMPLETE.json"
    restarts = 0
    last_count = -1
    last_progress_at = time.monotonic()
    while True:
        done = completed(ids, run_dir)
        if len(done) != last_count:
            last_count = len(done)
            last_progress_at = time.monotonic()
            append_jsonl(log_path, {
                "time": now(), "event": "progress", "completed": len(done),
                "total": len(ids), "remaining": len(ids) - len(done),
            })
        if len(done) == len(ids) and (run_dir / "summary_metrics.json").is_file():
            payload = {
                "time": now(), "status": "complete", "videos": len(ids),
                "restarts": restarts,
            }
            marker.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            append_jsonl(log_path, {**payload, "event": "complete"})
            return
        if not session_alive(args.session):
            if restarts >= args.max_restarts:
                append_jsonl(log_path, {
                    "time": now(), "event": "blocked", "reason": "max_restarts",
                    "completed": len(done), "total": len(ids),
                })
                raise SystemExit(2)
            restarts += 1
            result = subprocess.run(
                [
                    "tmux", "new-session", "-d", "-s", args.session,
                    str(args.launch_script.resolve()),
                ],
                check=False,
            )
            append_jsonl(log_path, {
                "time": now(), "event": "restart", "restart": restarts,
                "returncode": result.returncode, "completed": len(done),
            })
        elif time.monotonic() - last_progress_at > 45 * 60:
            append_jsonl(log_path, {
                "time": now(), "event": "stale_warning", "completed": len(done),
                "minutes_without_completed_video": 45,
            })
            last_progress_at = time.monotonic()
        time.sleep(max(5.0, args.poll_sec))


if __name__ == "__main__":
    main()
