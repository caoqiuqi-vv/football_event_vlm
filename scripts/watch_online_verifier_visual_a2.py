#!/usr/bin/env python
"""Queue A2 visual verifier after the active Stage-1 experiment finishes."""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
STAGE1 = ROOT / "outputs/football_events/vitl16_d7c_fullimage_frame_spotter_denseguard_from_best_e10_20260830"
CAL_METRICS = STAGE1 / "cal18_finalbest_densegrid_metrics.json"
CACHE = CAL_METRICS.with_suffix(".predictions.npz")
METADATA = CAL_METRICS.with_suffix(".predictions.meta.json")
VISUAL = ROOT / "outputs/football_candidate_verifiers/online_shot_visual_a2_finalbest_cal18_features_20260830"
TEST_METRICS = STAGE1 / "test18_finalbest_densegrid_metrics.json"
TEST_CACHE = TEST_METRICS.with_suffix(".predictions.npz")
TEST_METADATA = TEST_METRICS.with_suffix(".predictions.meta.json")
TEST_VISUAL = ROOT / "outputs/football_candidate_verifiers/online_shot_visual_a2_finalbest_test18_features_20260830"
A2 = ROOT / "outputs/football_candidate_verifiers/online_shot_residual_set_a2_visual_finalbest_cal18_external18_e5_20260830"
A1_EXTERNAL = ROOT / "outputs/football_candidate_verifiers/online_shot_residual_set_a1_finalbest_cal18_external18_e5_20260830"
A1 = A1_EXTERNAL
LOG = VISUAL / "watchdog.log"
GPU_IDS = (0, 1, 2, 4, 5, 7)
TARGET_EPOCH = 10
_CHECKPOINT_EPOCH_CACHE: dict[Path, tuple[int, int, int]] = {}


def log(message: str) -> None:
    VISUAL.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    with LOG.open("a") as handle:
        handle.write(f"[{stamp}] {message}\n")


def checkpoint_epoch(path: Path) -> int:
    if not path.is_file():
        return 0
    stat = path.stat()
    cached = _CHECKPOINT_EPOCH_CACHE.get(path)
    signature = (stat.st_mtime_ns, stat.st_size)
    if cached is not None and cached[:2] == signature:
        return cached[2]
    try:
        epoch = int(
            torch.load(path, map_location="cpu", weights_only=False).get("epoch", 0) or 0
        )
        _CHECKPOINT_EPOCH_CACHE[path] = (signature[0], signature[1], epoch)
        return epoch
    except Exception as exc:
        log(f"checkpoint_read_failed path={path} error={type(exc).__name__}:{exc}")
        return 0


def stage1_running() -> bool:
    result = subprocess.run(
        ["pgrep", "-af", "train_football_events.py --config configs/football/dinov3_vitl16_fullimage_frame_spotter_denseguard_from_best_20260830.yaml"],
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
        index, memory = (item.strip() for item in line.split(",", 1))
        used[int(index)] = int(memory)
    return all(used.get(gpu, 10**9) < 1024 for gpu in GPU_IDS)


def launch_extractors(checkpoint: Path, cache: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    processes = []
    for rank, gpu in enumerate(GPU_IDS):
        done = output / f"visual_rank{rank:02d}.done.json"
        if done.is_file():
            continue
        command = [
            sys.executable, "scripts/extract_online_verifier_visual_features.py",
            "--cache", str(cache),
            "--checkpoint", str(checkpoint),
            "--video-root", "/mnt/data_16t/football/raw_video_720P",
            "--output-dir", str(output),
            "--rank", str(rank), "--world-size", str(len(GPU_IDS)),
            "--device", "cuda:0", "--candidate-batch-size", "1",
        ]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        handle = (output / f"extract_rank{rank:02d}.log").open("a")
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        processes.append((rank, process, handle))
        log(f"extractor_started output={output.name} rank={rank} gpu={gpu} pid={process.pid}")
    failures = []
    for rank, process, handle in processes:
        status = process.wait()
        handle.close()
        log(f"extractor_finished rank={rank} status={status}")
        if status != 0:
            failures.append((rank, status))
    if failures:
        raise RuntimeError(f"visual extraction failed: {failures}")


def run_cal18_dense(checkpoint: Path) -> None:
    if CACHE.is_file() and METADATA.is_file():
        log("cal18_dense_cache_exists")
        return
    command = [
        str(Path(sys.executable).with_name("torchrun")),
        "--standalone", "--nproc-per-node=6", "train_football_events.py",
        "--config", "configs/football/dinov3_vitl16_fullimage_frame_spotter_denseguard_from_best_20260830.yaml",
        "--eval-only", "--eval-output", str(CAL_METRICS),
        f"model.init_checkpoint={checkpoint}",
        "eval.per_gpu_batch_size=4",
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, GPU_IDS))
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    with (VISUAL / "cal18_dense_eval.log").open("a") as handle:
        status = subprocess.run(
            command, cwd=ROOT, env=env, stdout=handle,
            stderr=subprocess.STDOUT, check=False,
        ).returncode
    if status != 0:
        raise RuntimeError(f"cal18 dense eval exited with status {status}")
    log("cal18_dense_cache_complete")


def run_test18_dense(checkpoint: Path) -> None:
    if TEST_CACHE.is_file() and TEST_METADATA.is_file():
        log("test18_dense_cache_exists")
        return
    command = [
        str(Path(sys.executable).with_name("torchrun")),
        "--standalone", "--nproc-per-node=6", "train_football_events.py",
        "--config", "configs/football/dinov3_vitl16_fullimage_frame_spotter_denseguard_from_best_20260830.yaml",
        "--eval-only", "--eval-output", str(TEST_METRICS),
        f"model.init_checkpoint={checkpoint}",
        "data.long_video.split_files.val=[configs/football/splits/thirdparty18_test_long15_val_no_pn_train/thirdparty18_test_video_ids.txt]",
        "eval.per_gpu_batch_size=4",
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, GPU_IDS))
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    with (VISUAL / "test18_dense_eval.log").open("a") as handle:
        status = subprocess.run(
            command, cwd=ROOT, env=env, stdout=handle,
            stderr=subprocess.STDOUT, check=False,
        ).returncode
    if status != 0:
        raise RuntimeError(f"test18 dense eval exited with status {status}")
    log("test18_dense_cache_complete")


def run_a1_external() -> None:
    command = [
        sys.executable, "scripts/run_online_shot_set_verifier_experiment.py",
        "--cache", str(CACHE), "--metadata", str(METADATA),
        "--external-cache", str(TEST_CACHE),
        "--external-metadata", str(TEST_METADATA),
        "--output", str(A1_EXTERNAL), "--folds", "5", "--epochs", "5",
        "--batch-size", "64", "--threshold-grid-size", "201", "--device", "cpu",
    ]
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONPATH="football_e2e_spotter/src")
    with (VISUAL / "a1_external.log").open("a") as handle:
        status = subprocess.run(
            command, cwd=ROOT, env=env, stdout=handle,
            stderr=subprocess.STDOUT, check=False,
        ).returncode
    if status != 0:
        raise RuntimeError(f"A1 external eval exited with status {status}")
    log("a1_external_complete")


def run_a2() -> None:
    command = [
        sys.executable, "scripts/run_online_shot_set_verifier_experiment.py",
        "--cache", str(CACHE), "--metadata", str(METADATA),
        "--visual-shard-dir", str(VISUAL), "--output", str(A2),
        "--external-cache", str(TEST_CACHE),
        "--external-metadata", str(TEST_METADATA),
        "--external-visual-shard-dir", str(TEST_VISUAL),
        "--folds", "5", "--epochs", "5", "--batch-size", "32",
        "--threshold-grid-size", "201", "--device", "cpu",
    ]
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONPATH="football_e2e_spotter/src")
    with (VISUAL / "a2_oof.log").open("a") as handle:
        status = subprocess.run(
            command, cwd=ROOT, env=env, stdout=handle,
            stderr=subprocess.STDOUT, check=False,
        ).returncode
    if status != 0:
        raise RuntimeError(f"A2 OOF exited with status {status}")
    a2 = json.loads((A2 / "report.json").read_text())
    comparison = {"a2": a2["set_verifier_oof"]}
    if (A1 / "report.json").is_file():
        a1 = json.loads((A1 / "report.json").read_text())
        comparison["a1"] = a1["set_verifier_oof"]
    (A2 / "a1_a2_comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n"
    )
    log("a2_oof_complete")


def main() -> None:
    log(f"watchdog_started pid={os.getpid()}")
    while True:
        best = STAGE1 / "best.pt"
        last = STAGE1 / "last.pt"
        epoch = max(checkpoint_epoch(best), checkpoint_epoch(last))
        if epoch >= TARGET_EPOCH and not stage1_running():
            checkpoint = best if best.is_file() else last
            break
        log(f"waiting_stage1 epoch={epoch} running={stage1_running()}")
        time.sleep(60)
    while not gpus_available():
        log("waiting_gpus")
        time.sleep(60)
    log(f"stage1_ready checkpoint={checkpoint}")
    run_cal18_dense(checkpoint)
    run_test18_dense(checkpoint)
    launch_extractors(checkpoint, CACHE, VISUAL)
    launch_extractors(checkpoint, TEST_CACHE, TEST_VISUAL)
    run_a1_external()
    run_a2()
    log("watchdog_complete")


if __name__ == "__main__":
    main()
