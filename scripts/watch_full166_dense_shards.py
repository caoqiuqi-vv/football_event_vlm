#!/usr/bin/env python3
"""Restart dense shards and finalize the unified run after every video exists."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


REQUIRED = ("summary.json", "metrics.json", "window_predictions.csv", "gt_events.json")


def alive(name: str) -> bool:
    return subprocess.run(["tmux", "has-session", "-t", name], stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL, check=False).returncode == 0


def completed(run_dir: Path, ids: list[str]) -> int:
    return sum(all((run_dir / video_id / name).is_file() for name in REQUIRED) for video_id in ids)


def append(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def start(session: str, runner: Path, gpu: int, id_file: Path) -> None:
    subprocess.run([
        "tmux", "new-session", "-d", "-s", session,
        str(runner), str(gpu), str(id_file),
    ], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,4,6")
    parser.add_argument("--poll-sec", type=float, default=60)
    args = parser.parse_args()
    run_dir, runner = args.run_dir.resolve(), args.runner.resolve()
    all_ids = [x.strip() for x in (run_dir / "all_video_ids.txt").read_text().splitlines() if x.strip()]
    state_path = run_dir / "shards" / "watchdog.jsonl"
    restarts = {int(g): 0 for g in args.gpus.split(",")}
    last_total = -1
    while True:
        total_done = completed(run_dir, all_ids)
        if total_done != last_total:
            last_total = total_done
            append(state_path, {"time": datetime.now(timezone.utc).isoformat(),
                                "event": "progress", "completed": total_done,
                                "total": len(all_ids), "remaining": len(all_ids) - total_done})
        if total_done == len(all_ids):
            break
        for gpu in restarts:
            id_file = run_dir / "shards" / f"gpu{gpu}_video_ids.txt"
            ids = [x.strip() for x in id_file.read_text().splitlines() if x.strip()]
            session = f"football_full166_dense_gpu{gpu}_20260902"
            if completed(run_dir, ids) < len(ids) and not alive(session):
                restarts[gpu] += 1
                if restarts[gpu] > 20:
                    raise RuntimeError(f"GPU {gpu} shard exceeded restart limit")
                start(session, runner, gpu, id_file)
                append(state_path, {"time": datetime.now(timezone.utc).isoformat(),
                                    "event": "restart", "gpu": gpu,
                                    "restart": restarts[gpu]})
        time.sleep(max(5, args.poll_sec))

    # Aggregate directly from the validated per-video results. Re-running the
    # inference wrapper is unsafe when compatible cached videos are symlinked.
    from evaluate_football_model import aggregate_frame_event_outputs, summarize

    labels = ["shot", "save", "set_piece"]
    summary = summarize(run_dir, all_ids, labels)
    done_ids = {
        row["video_id"] for row in summary["per_video"]
        if row.get("status") == "done"
    }
    if len(done_ids) != len(all_ids):
        raise RuntimeError(f"final aggregation incomplete: {len(done_ids)} != {len(all_ids)}")
    aggregate_frame_event_outputs(run_dir, all_ids, labels, topk=5)
    marker = {"time": datetime.now(timezone.utc).isoformat(), "status": "complete",
              "videos": len(all_ids), "gpus": list(restarts), "restarts": restarts}
    (run_dir / "DENSE_COMPLETE.json").write_text(json.dumps(marker, ensure_ascii=False, indent=2) + "\n")
    append(state_path, {**marker, "event": "complete"})


if __name__ == "__main__":
    main()
