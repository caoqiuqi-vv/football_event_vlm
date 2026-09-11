#!/usr/bin/env python
"""Watchdog: train feature bank -> calibration bank -> six-GPU retriever training."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "football_longform_v2/experiments/lf_a0_official_fullscale/canonical_manifest.json"
PIXELS = Path("/mnt/data_16t/football/set_spotter_4fps_288x512")
FEATURES = Path("/mnt/data_16t/football/goal_feature_bank_dino_v1")
CHECKPOINT = ROOT / "outputs/football_events/vitl16_d7c_fullimage_latest_split_dino_pretrained_lora_ddp_e8_20260828/best.pt"
EXPERIMENT = ROOT / "football_e2e_spotter/experiments/goal_retriever_v1"
OUTPUT = ROOT / "outputs/football_goal_retriever/longctx_dino_motion_audio_v1_20260830"
GPUS = (0, 1, 2, 4, 5, 7)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def manifest_ids(split: str) -> list[str]:
    return [str(item["media_id"]) for item in json.loads(MANIFEST.read_text(encoding="utf-8"))[split]]


def ready(root: Path, split: str, ids: list[str], filename: str) -> list[str]:
    return [video_id for video_id in ids if (root / split / video_id / filename).is_file()]


def update(stage: str, **values) -> None:
    payload = {"schema": "football.goal_retriever_watchdog.v1", "stage": stage, "updated_at": time.time(), **values}
    atomic_json(EXPERIMENT / "watchdog_status.json", payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def wait_for(stage: str, predicate, description: str, timeout_hours: float = 48.0) -> None:
    deadline = time.time() + timeout_hours * 3600
    while True:
        value = predicate()
        if value:
            update(stage, condition=description, ready=True)
            return
        if time.time() >= deadline:
            raise TimeoutError(f"timeout waiting for {description}")
        update(stage, condition=description, ready=False)
        time.sleep(30)


def run_calibration_features() -> None:
    processes = []
    for shard, gpu in enumerate(GPUS):
        log_path = EXPERIMENT / f"feature_calibration_shard{shard}.log"
        handle = log_path.open("a", encoding="utf-8")
        command = [
            sys.executable, str(ROOT / "football_e2e_spotter/scripts/build_goal_feature_bank_mmap.py"),
            "--manifest", str(MANIFEST), "--pixel-root", str(PIXELS), "--output-root", str(FEATURES),
            "--split", "calibration", "--checkpoint", str(CHECKPOINT),
            "--gpu", str(gpu), "--shard-index", str(shard), "--num-shards", str(len(GPUS)),
            "--chunk-frames", "12",
        ]
        process = subprocess.Popen(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append((process, handle, log_path))
    update("calibration_features_running", pids=[process.pid for process, _handle, _path in processes])
    failures = []
    for process, handle, log_path in processes:
        code = process.wait(); handle.close()
        if code:
            failures.append({"pid": process.pid, "exit_code": code, "log": str(log_path)})
    if failures:
        raise RuntimeError(f"calibration feature workers failed: {failures}")


def run_training() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    log_path = EXPERIMENT / "train_retriever.log"
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in GPUS)
    command = [
        "torchrun", "--standalone", f"--nproc_per_node={len(GPUS)}",
        str(ROOT / "football_e2e_spotter/scripts/train_goal_retriever.py"),
        "--manifest", str(MANIFEST), "--feature-root", str(FEATURES), "--output", str(OUTPUT),
        "--epochs", "20", "--batch-size", "12", "--eval-batch-size", "32", "--workers", "0",
        "--lr", "0.0002", "--hidden-dim", "384", "--shot-recall-floor", "0.97", "--other-recall-floor", "0.94",
    ]
    with log_path.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        update("retriever_training", pid=process.pid, log=str(log_path), output=str(OUTPUT))
        code = process.wait()
    if code:
        raise RuntimeError(f"retriever training failed with {code}; see {log_path}")


def main() -> None:
    EXPERIMENT.mkdir(parents=True, exist_ok=True)
    train_ids = manifest_ids("train"); calibration_ids = manifest_ids("calibration")
    wait_for(
        "waiting_train_features",
        lambda: len(ready(FEATURES, "train", train_ids, "metadata.json")) == len(train_ids),
        f"{len(train_ids)} train feature banks",
    )
    wait_for(
        "waiting_calibration_pixels",
        lambda: len(ready(PIXELS, "calibration", calibration_ids, "metadata.json")) == len(calibration_ids),
        f"{len(calibration_ids)} calibration pixel/audio caches",
    )
    if len(ready(FEATURES, "calibration", calibration_ids, "metadata.json")) != len(calibration_ids):
        run_calibration_features()
    if len(ready(FEATURES, "calibration", calibration_ids, "metadata.json")) != len(calibration_ids):
        raise RuntimeError("calibration feature bank incomplete after workers exited")
    run_training()
    best = torch_load = None
    report_paths = sorted(OUTPUT.glob("calibration_epoch*.json"))
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in report_paths]
    if reports:
        best = max(reports, key=lambda row: (
            bool(row["gate_pass"]), sum(value["recall_floor_reachable"] for value in row["classes"].values()),
            -float(row["review_ratio"]), sum(value["recall"] for value in row["classes"].values()),
        ))
    update("retriever_complete", output=str(OUTPUT), best_calibration=best)


if __name__ == "__main__":
    main()

