#!/usr/bin/env python3
"""Restart-safe watchdog for the NMS-free two-stage experiment.

The stages are intentionally sequential: cache -> OOF spotter candidates ->
Verifier -> calibration.  State is persisted after every successful command,
and a failed stage is never silently skipped or replaced by NMS.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, help="JSON with ordered {name, command} stages")
    parser.add_argument("--state", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    if not plan.get("no_temporal_nms", False):
        raise RuntimeError("refusing a plan without explicit no_temporal_nms=true")
    stages = plan.get("stages", [])
    if not stages:
        raise ValueError("plan has no stages")
    state_path, log_path = Path(args.state), Path(args.log)
    state_path.parent.mkdir(parents=True, exist_ok=True); log_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = json.loads(state_path.read_text()) if state_path.stat().st_size else {"no_temporal_nms": True, "completed": [], "started_at": now()}
        if state.get("no_temporal_nms") is not True:
            raise RuntimeError("existing state violates no-NMS contract")
        completed = set(state.get("completed", []))
        with log_path.open("a") as log:
            for stage in stages:
                name, command = stage["name"], stage["command"]
                if name in completed:
                    continue
                log.write(f"[{now()}] start {name}: {command}\n"); log.flush()
                if args.dry_run:
                    result = 0
                else:
                    env = os.environ.copy(); env.update({"CUDA_VISIBLE_DEVICES": args.gpu, "PYTHONUNBUFFERED": "1"})
                    result = subprocess.run(command, shell=True, cwd=plan.get("cwd"), env=env, stdout=log, stderr=subprocess.STDOUT).returncode
                if result:
                    state.update({"failed_stage": name, "failed_at": now(), "returncode": result})
                    state_path.write_text(json.dumps(state, indent=2) + "\n")
                    raise SystemExit(result)
                completed.add(name); state.update({"completed": sorted(completed), "updated_at": now()})
                state_path.write_text(json.dumps(state, indent=2) + "\n")
                log.write(f"[{now()}] complete {name}\n"); log.flush()
        state.update({"completed_at": now(), "status": "complete"})
        state_path.write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    main()
