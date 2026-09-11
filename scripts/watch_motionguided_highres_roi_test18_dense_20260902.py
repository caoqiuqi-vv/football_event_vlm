#!/usr/bin/env python3
"""Launch branch-verified test18 dense evaluation after full166 releases GPUs."""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/home/new_users/qiuqi/code/dinov3-main")
FULL166_DONE = ROOT / "outputs/football_full_review/full166_fromlast_e8_best_dense_s5_20260902/DENSE_COMPLETE.json"
FULL166_PROGRESS = ROOT / "outputs/football_full_review/full166_fromlast_e8_best_dense_s5_20260902/shards/watchdog.jsonl"
RUN_NAME = "vitl16_d7c_motionguided_highres_roi_residual_best_test18_dense_branchactive_20260902"
RUN_DIR = ROOT / "outputs/football_eval_runs" / RUN_NAME
TEST_IDS = ROOT / "configs/football/splits/thirdparty18_test_long15_val_no_pn_train/thirdparty18_test_video_ids.txt"
RUNNER = ROOT / "scripts/run_motionguided_highres_roi_test18_dense_shard_20260902.sh"
GPUS = (6,)
REQUIRED = ("summary.json", "metrics.json", "window_predictions.csv", "gt_events.json")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def complete(video_id: str) -> bool:
    return all((RUN_DIR / video_id / name).is_file() for name in REQUIRED)


def alive(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def dependency_released() -> bool:
    if FULL166_DONE.is_file():
        return True
    if not FULL166_PROGRESS.is_file():
        return False
    for line in reversed(FULL166_PROGRESS.read_text().splitlines()):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") == "progress":
            return int(event.get("completed", 0)) >= int(event.get("total", 166))
    return False


def run_checked(command: list[str], log_path: Path) -> None:
    with log_path.open("a", encoding="utf-8") as handle:
        subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)


def main() -> None:
    RUN_DIR.joinpath("shards").mkdir(parents=True, exist_ok=True)
    status_path = RUN_DIR / "watchdog_status.json"
    ids = [line.strip() for line in TEST_IDS.read_text().splitlines() if line.strip()]
    if len(ids) != 18:
        raise RuntimeError(f"expected 18 test videos, got {len(ids)}")
    RUN_DIR.joinpath("all_video_ids.txt").write_text("\n".join(ids) + "\n")
    shard_files: dict[int, Path] = {}
    for offset, gpu in enumerate(GPUS):
        path = RUN_DIR / "shards" / f"gpu{gpu}_video_ids.txt"
        path.write_text("\n".join(ids[offset::len(GPUS)]) + "\n")
        shard_files[gpu] = path

    status_path.write_text(json.dumps({
        "status": "waiting_for_full166",
        "time": utc_now(),
        "dependency": str(FULL166_DONE),
        "checkpoint": "motionguided_highres_roi_residual/best.pt",
        "gpus": list(GPUS),
        "videos": len(ids),
    }, ensure_ascii=False, indent=2) + "\n")
    while not dependency_released():
        time.sleep(30)

    restarts = {gpu: 0 for gpu in GPUS}
    while not all(complete(video_id) for video_id in ids):
        for gpu, shard_file in shard_files.items():
            shard_ids = [x.strip() for x in shard_file.read_text().splitlines() if x.strip()]
            if all(complete(video_id) for video_id in shard_ids):
                continue
            session = f"football_motionguided_highres_test18_gpu{gpu}_20260902"
            if not alive(session):
                restarts[gpu] += 1
                if restarts[gpu] > 8:
                    raise RuntimeError(f"GPU {gpu} exceeded restart limit")
                subprocess.run([
                    "tmux", "new-session", "-d", "-s", session,
                    str(RUNNER), str(gpu), str(shard_file),
                ], cwd=ROOT, check=True)
        done = sum(complete(video_id) for video_id in ids)
        status_path.write_text(json.dumps({
            "status": "dense_running", "time": utc_now(), "completed": done,
            "total": len(ids), "restarts": restarts,
        }, ensure_ascii=False, indent=2) + "\n")
        time.sleep(30)

    log = RUN_DIR / "postprocess.log"
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    with log.open("a", encoding="utf-8") as handle:
        subprocess.run([
            str(RUNNER), "0", str(RUN_DIR / "all_video_ids.txt")
        ], cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT, check=True)

    inactive = []
    max_deltas = []
    for video_id in ids:
        summary = json.loads((RUN_DIR / video_id / "summary.json").read_text())
        diagnostic = summary.get("score_diagnostics", {}).get("highres_glimpse", {})
        if diagnostic.get("active") is not True:
            inactive.append(video_id)
        max_deltas.append(float(diagnostic.get("max_abs_logit_delta", 0.0)))
    if inactive:
        raise RuntimeError(f"high-res ROI branch inactive for videos: {inactive}")

    branch_comparison = RUN_DIR / "highres_anchor_vs_final_fixed_thresholds.json"
    run_checked([
        "python", "scripts/analyze_highres_residual_branch.py",
        "--run-dir", str(RUN_DIR), "--video-id-file", str(TEST_IDS),
        "--labels", "shot,save,set_piece", "--match-tolerance-sec", "3",
        "--output", str(branch_comparison),
    ], log)

    adaptive = RUN_DIR / "video_adaptive_thresholds_loov_tol3.json"
    workload = RUN_DIR / "adaptive_manual_workload_tol3.json"
    run_checked([
        "python", "scripts/analyze_video_adaptive_dense_thresholds.py",
        "--run-dir", str(RUN_DIR), "--video-id-file", str(TEST_IDS),
        "--labels", "shot,save,set_piece", "--match-tolerance-sec", "3",
        "--recall-floors", "shot=0.90,save=0.85,set_piece=0.85",
        "--output", str(adaptive),
    ], log)
    run_checked([
        "python", "scripts/summarize_adaptive_dense_review_workload.py",
        "--run-dir", str(RUN_DIR), "--adaptive-report", str(adaptive),
        "--video-id-file", str(TEST_IDS), "--labels", "shot,save,set_piece",
        "--match-tolerance-sec", "3", "--merge-gap-sec", "0",
        "--manual-overhead-sec", "3", "--output", str(workload),
    ], log)
    status_path.write_text(json.dumps({
        "status": "complete", "time": utc_now(), "completed": len(ids),
        "branch_active_videos": len(ids), "max_abs_logit_delta_min": min(max_deltas),
        "max_abs_logit_delta_max": max(max_deltas), "adaptive_report": str(adaptive),
        "workload_report": str(workload), "branch_comparison": str(branch_comparison),
    }, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
