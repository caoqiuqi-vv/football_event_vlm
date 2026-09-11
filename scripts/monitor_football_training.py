#!/usr/bin/env python
"""Persist health/checkpoint state for a long football training process."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

import yaml


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def gpu_snapshot() -> list[dict[str, int]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, timeout=10)
    except Exception:
        return []
    rows = []
    for line in output.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == 4:
            rows.append(
                {
                    "index": int(values[0]),
                    "memory_used_mb": int(values[1]),
                    "memory_free_mb": int(values[2]),
                    "utilization_percent": int(values[3]),
                }
            )
    return rows


def checkpoint_epochs(output_dir: Path) -> list[int]:
    epochs = []
    for path in output_dir.glob("*.pt"):
        match = re.search(r"epoch[_-]?(\d+)", path.name)
        if match:
            epochs.append(int(match.group(1)))
    return sorted(set(epochs))


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--poll-sec", type=float, default=60.0)
    parser.add_argument("--gpu-index", action="append", type=int, default=[])
    args = parser.parse_args()

    config_path = args.output_dir / "config.yaml"
    expected_epochs = 0
    if config_path.is_file():
        config = yaml.safe_load(config_path.read_text()) or {}
        expected_epochs = int(config.get("train", {}).get("epochs", 0) or 0)
    started = time.time()
    while True:
        alive = process_alive(args.pid)
        epochs = checkpoint_epochs(args.output_dir)
        checkpoint_files = sorted(
            path.name for path in args.output_dir.glob("*.pt")
        )
        max_epoch = max(epochs, default=0)
        if alive:
            state = "training"
        elif expected_epochs and max_epoch >= expected_epochs:
            state = "complete"
        else:
            state = "failed"
        snapshot = gpu_snapshot()
        if args.gpu_index:
            wanted = set(args.gpu_index)
            snapshot = [row for row in snapshot if row["index"] in wanted]
        payload = {
            "state": state,
            "pid": args.pid,
            "process_alive": alive,
            "output_dir": str(args.output_dir.resolve()),
            "expected_epochs": expected_epochs,
            "checkpoint_epochs": epochs,
            "checkpoint_files": checkpoint_files,
            "max_checkpoint_epoch": max_epoch,
            "elapsed_seconds": time.time() - started,
            "gpu": snapshot,
            "updated_at_unix": time.time(),
        }
        atomic_json(args.status, payload)
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        if not alive:
            raise SystemExit(0 if state == "complete" else 1)
        time.sleep(max(args.poll_sec, 10.0))


if __name__ == "__main__":
    main()

