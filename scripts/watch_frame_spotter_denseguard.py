#!/usr/bin/env python
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
import subprocess
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/football/dinov3_vitl16_fullimage_frame_spotter_denseguard_from_best_20260830.yaml"
OUTPUT = ROOT / "outputs/football_events/vitl16_d7c_fullimage_frame_spotter_denseguard_from_best_e10_20260830"
LAST = OUTPUT / "last.pt"
RECOVERY = OUTPUT / "pre_eval_recovery.pt"
LOG = OUTPUT / "watchdog.log"
TRAIN_LOG = OUTPUT / "watchdog_train_console.log"
TARGET_EPOCH = 10
MAX_RESTARTS = 3
GPU_IDS = (0, 1, 2, 4, 5, 7)


def log(message: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    with LOG.open("a") as handle:
        handle.write(f"[{stamp}] {message}\n")


def resume_checkpoint() -> Path | None:
    if LAST.is_file():
        return LAST
    if RECOVERY.is_file():
        return RECOVERY
    return None


def checkpoint_epoch() -> int:
    checkpoint = resume_checkpoint()
    if checkpoint is None:
        return 0
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        return int(state.get("epoch", 0) or 0)
    except Exception as exc:
        log(f"checkpoint_read_failed error={type(exc).__name__}:{exc}")
        return 0


def training_running() -> bool:
    result = subprocess.run(
        ["pgrep", "-af", f"train_football_events.py --config {CONFIG.relative_to(ROOT)}"],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    return bool(result.stdout.strip())


def gpus_available() -> bool:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=False,
    )
    if result.returncode != 0:
        return False
    used = {}
    for line in result.stdout.splitlines():
        index, memory = (part.strip() for part in line.split(",", 1))
        used[int(index)] = int(memory)
    return all(used.get(index, 10**9) < 1024 for index in GPU_IDS)


def launch(resume: bool) -> int:
    command = [
        str(Path(os.sys.executable).with_name("torchrun")),
        "--standalone", "--nproc-per-node=6",
        "train_football_events.py", "--config", str(CONFIG.relative_to(ROOT)),
    ]
    checkpoint = resume_checkpoint()
    if resume and checkpoint is not None:
        command.extend([
            "train.resume.enabled=true",
            f"train.resume.checkpoint={checkpoint}",
            "train.resume.strict=true",
            "train.resume.load_optimizer=true",
            "train.resume.reset_optimizer_lr=true",
            "train.resume.load_scheduler=false",
            "train.resume.reset_scheduler=true",
            "train.resume.load_scaler=true",
            "eval.per_gpu_batch_size=4",
        ])
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,4,5,7"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    with TRAIN_LOG.open("a") as handle:
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    log(f"launched pid={process.pid} resume={resume} epoch={checkpoint_epoch()}")
    return int(process.pid)


def main() -> None:
    log(f"watchdog_started pid={os.getpid()} target_epoch={TARGET_EPOCH}")
    restarts = 0
    while True:
        epoch = checkpoint_epoch()
        if epoch >= TARGET_EPOCH:
            log(f"complete epoch={epoch}")
            return
        if training_running():
            time.sleep(30)
            continue
        if restarts >= MAX_RESTARTS:
            log(f"blocked max_restarts={MAX_RESTARTS} epoch={epoch}")
            return
        if not gpus_available():
            log(f"waiting_gpus epoch={epoch}")
            time.sleep(30)
            continue
        launch(resume=resume_checkpoint() is not None)
        restarts += 1
        time.sleep(60)


if __name__ == "__main__":
    main()
