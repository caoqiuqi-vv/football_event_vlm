from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
for path in (PROJECT_ROOT / "src", WORKSPACE_ROOT, PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_official_feature_store import atomic_json, canonical_items, resolve  # noqa: E402
from football_longform_v2.config import load_config  # noqa: E402
from football_longform_v2.schema import read_video_ids  # noqa: E402


@contextmanager
def coordinator_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another coordinator owns {path}") from error
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def expected_worker_paths(feature_root: Path, run_id: str, num_shards: int) -> list[Path]:
    return [feature_root / "manifests" / run_id / f"worker-{index}.json" for index in range(num_shards)]


def worker_terminal_error(
    paths: list[Path], *, run_id: str, num_shards: int
) -> str | None:
    for index, path in enumerate(paths):
        if not path.is_file():
            continue
        try:
            ledger = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            return f"cannot read worker-{index} ledger: {error}"
        if ledger.get("run_id") != run_id or ledger.get("num_shards") != num_shards:
            return f"worker-{index} ledger run contract mismatch"
        if ledger.get("shard_index") != index:
            return f"worker-{index} ledger shard index mismatch"
        if int(ledger.get("failed", 0)):
            return f"worker-{index} reports failed={ledger.get('failed')}"
    return None


def run_aggregate(config: str, run_id: str, video_root: str) -> None:
    command = [
        sys.executable, str(PROJECT_ROOT / "scripts/aggregate_official_feature_store.py"),
        "--config", config, "--run-id", run_id, "--video-root", video_root,
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh only the official LF-A0 feature manifest.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--video-root", default="/mnt/data_16t/football/raw_video_720P")
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--worker-pids", type=int, nargs="+", required=True)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    if args.num_shards < 1 or len(args.worker_pids) != args.num_shards:
        raise ValueError("worker PID count must equal --num-shards")
    if not 0.0 < args.poll_seconds <= 60.0:
        raise ValueError("--poll-seconds must be in (0, 60]")

    config_argument = str(Path(args.config).resolve())
    config = load_config(config_argument)
    root = Path(config["_project_root"])
    feature_root = resolve(root, config["paths"]["feature_store"])
    selected = ("train", "calibration")
    split_ids = {
        split: read_video_ids(resolve(root, config["paths"][f"{split}_ids"]))
        for split in selected
    }
    total = len(canonical_items(split_ids, selected))
    ledgers = expected_worker_paths(feature_root, args.run_id, args.num_shards)
    state_path = root / "experiments/lf_a0_official_fullscale/feature_coordinator_state.json"
    lock_path = feature_root / ".coordinators" / f"{args.run_id}.lock"
    with coordinator_lock(lock_path):
        while True:
            run_aggregate(config_argument, args.run_id, args.video_root)
            main_manifest = json.loads((feature_root / "manifest.json").read_text(encoding="utf-8"))
            valid = int(main_manifest.get("cached_valid", 0))
            terminal_error = worker_terminal_error(
                ledgers, run_id=args.run_id, num_shards=args.num_shards
            )
            state = {
                "run_id": args.run_id, "updated_unix": time.time(), "total": total,
                "cached_valid": valid, "worker_ledgers": [str(path) for path in ledgers],
                "worker_pids": args.worker_pids,
            }
            if terminal_error:
                atomic_json(state_path, {**state, "status": "failed", "reason": terminal_error})
                raise SystemExit(2)
            if valid == total:
                atomic_json(state_path, {**state, "status": "done"})
                return
            alive = [pid_alive(pid) for pid in args.worker_pids]
            if not any(alive):
                atomic_json(state_path, {
                    **state, "status": "failed",
                    "reason": "all workers disappeared before cache completion",
                    "worker_alive": alive,
                })
                raise SystemExit(2)
            atomic_json(state_path, {**state, "status": "running", "worker_alive": alive})
            print(f"coordinator cached_valid={valid}/{total} worker_alive={alive}", flush=True)
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
