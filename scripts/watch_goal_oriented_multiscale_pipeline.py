#!/usr/bin/env python
"""Recoverable watchdog for the goal-oriented football review pipeline."""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GPU_IDS = (0, 1, 2, 4, 5, 7)
VIDEO_ROOT = Path("/mnt/data_16t/football/raw_video_720P")
CAL_IDS = Path("/mnt/data_7t/qiuqi/dinov3-main/configs/football_longform_v2/experiments/lf_a0_official_fullscale/canonical_calibration_media_ids.txt")
TEST_IDS = ROOT / "configs/football/splits/thirdparty18_test_long15_val_no_pn_train/thirdparty18_test_video_ids.txt"
STAGE1 = ROOT / "outputs/football_events/vitl16_d7c_fullimage_frame_spotter_denseguard_from_best_e10_20260830"
STAGE1_CONFIG = ROOT / "configs/football/dinov3_vitl16_fullimage_frame_spotter_denseguard_from_best_20260830.yaml"
STAGE1_CHECKPOINT = STAGE1 / "best.pt"
CAL_CACHE = STAGE1 / "val15_online_ema_epoch_002.npz"
CAL_META = STAGE1 / "val15_online_ema_epoch_002.meta.json"
TEST_METRICS = STAGE1 / "test18_best_epoch2_densegrid_metrics.json"
TEST_CACHE = TEST_METRICS.with_suffix(".predictions.npz")
TEST_META = TEST_METRICS.with_suffix(".predictions.meta.json")
TEACHER_CHECKPOINT = ROOT / "outputs/football_events/vitl16_d7c_fullimage_latest_split_dino_pretrained_lora_ddp_e8_20260828/best.pt"
CAL_TEACHER = ROOT / "outputs/football_candidate_verifiers/clip_teacher_512_center_cal18_epoch2_20260830"
TEST_TEACHER = ROOT / "outputs/football_candidate_verifiers/clip_teacher_512_center_test18_epoch2_20260830"
CAL_AUDIO = ROOT / "outputs/football_audio_features/cal18_5hz_20260830"
TEST_AUDIO = ROOT / "outputs/football_audio_features/test18_5hz_20260830"
CAL_WHISTLE = ROOT / "outputs/football_audio_features/cal18_whistle_activity_20260830"
TEST_WHISTLE = ROOT / "outputs/football_audio_features/test18_whistle_activity_20260830"
CAL_OUTPUT = ROOT / "outputs/football_goal_pipeline/multimodal_cal18_epoch2_20260830"
TEST_OUTPUT = ROOT / "outputs/football_goal_pipeline/multimodal_test18_epoch2_20260830"
ROOT_OUTPUT = ROOT / "outputs/football_goal_pipeline"
LOG = ROOT_OUTPUT / "watchdog_20260830.log"
STATE = ROOT_OUTPUT / "watchdog_20260830.state.json"


def log(message: str) -> None:
    ROOT_OUTPUT.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    line = f"[{stamp}] {message}"
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def record(stage: str, status: str, **extra: object) -> None:
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.is_file() else {"stages": {}}
    state["stages"][stage] = {
        "status": status,
        "updated_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        **extra,
    }
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(command: list[str], *, log_path: Path, env: dict[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"run {' '.join(command)}")
    with log_path.open("a", encoding="utf-8") as handle:
        status = subprocess.run(
            command, cwd=ROOT, env=env, stdout=handle,
            stderr=subprocess.STDOUT, check=False,
        ).returncode
    if status != 0:
        raise RuntimeError(f"command failed status={status}; see {log_path}")


def process_running(pattern: str) -> bool:
    result = subprocess.run(
        ["pgrep", "-af", pattern], text=True, capture_output=True, check=False,
    )
    lines = [line for line in result.stdout.splitlines() if str(os.getpid()) not in line]
    return bool(lines)


def wait_for_external(pattern: str, description: str) -> None:
    while process_running(pattern):
        log(f"waiting {description}")
        time.sleep(30)


def ids(path: Path) -> list[str]:
    return [
        line.strip() for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def teacher_complete(path: Path) -> bool:
    done = sorted(path.glob("teacher_rank*.done.json"))
    shards = sorted(path.glob("teacher_rank*_of_*.npz"))
    return len(done) == len(GPU_IDS) and len(shards) == len(GPU_IDS)


def launch_teacher(cache: Path, output: Path) -> None:
    if teacher_complete(output):
        return
    output.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[int, subprocess.Popen[bytes], object]] = []
    for rank, gpu in enumerate(GPU_IDS):
        done = output / f"teacher_rank{rank:02d}.done.json"
        if done.is_file():
            continue
        command = [
            sys.executable, "scripts/extract_candidate_clip_teacher_scores.py",
            "--cache", str(cache), "--checkpoint", str(TEACHER_CHECKPOINT),
            "--video-root", str(VIDEO_ROOT), "--output-dir", str(output),
            "--rank", str(rank), "--world-size", str(len(GPU_IDS)),
            "--device", "cuda:0", "--offsets-sec", "0", "--clip-sec", "10",
            "--batch-size", "4", "--num-workers", "2",
        ]
        env = dict(os.environ)
        env.update(
            CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1",
            PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        )
        handle = (output / f"rank{rank}.log").open("a", encoding="utf-8")
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=handle,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
        processes.append((rank, process, handle))
        log(f"teacher rank={rank} gpu={gpu} pid={process.pid}")
    failures = []
    for rank, process, handle in processes:
        status = process.wait()
        handle.close()
        if status:
            failures.append((rank, status))
    if failures or not teacher_complete(output):
        raise RuntimeError(f"teacher extraction incomplete: failures={failures}")


def build_audio(id_file: Path, output: Path) -> None:
    expected = ids(id_file)
    if all((output / f"{video_id}.npz").is_file() for video_id in expected):
        return
    output.mkdir(parents=True, exist_ok=True)
    run([
        sys.executable, "scripts/build_audio_feature_index.py",
        "--video-dir", str(VIDEO_ROOT), "--video-ids", str(id_file),
        "--output-dir", str(output),
    ], log_path=output / "build.log")


def build_whistles(id_file: Path, audio: Path, output: Path) -> None:
    if (output / "summary.json").is_file():
        return
    run([
        sys.executable, "scripts/extract_whistle_activity_candidates.py",
        "--audio-index-dir", str(audio), "--output-dir", str(output),
        "--video-ids", str(id_file),
    ], log_path=output / "build.log")


def calibrate() -> dict[str, object]:
    if not (CAL_OUTPUT / "policy.json").is_file():
        run([
            sys.executable, "-m", "scripts.goal_oriented_multimodal_policy",
            "--cache", str(CAL_CACHE), "--metadata", str(CAL_META),
            "--teacher-shard-dir", str(CAL_TEACHER),
            "--whistle-dir", str(CAL_WHISTLE), "--output", str(CAL_OUTPUT),
            "--split-name", "calibration18",
        ], log_path=CAL_OUTPUT / "calibrate.log")
    return json.loads((CAL_OUTPUT / "policy.json").read_text(encoding="utf-8"))


def teacher_required(policy: dict[str, object]) -> bool:
    labels = policy["labels"]
    return any(item["score_definition"]["kind"] != "stage1" for item in labels.values())


def test_dense() -> None:
    if TEST_CACHE.is_file() and TEST_META.is_file():
        return
    command = [
        str(Path(sys.executable).with_name("torchrun")),
        "--standalone", f"--nproc-per-node={len(GPU_IDS)}",
        "train_football_events.py", "--config", str(STAGE1_CONFIG.relative_to(ROOT)),
        "--eval-only", "--eval-output", str(TEST_METRICS),
        f"model.init_checkpoint={STAGE1_CHECKPOINT}",
        f"data.long_video.split_files.val=[{TEST_IDS.relative_to(ROOT)}]",
        "eval.per_gpu_batch_size=4",
    ]
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES=",".join(map(str, GPU_IDS)),
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
    )
    run(command, log_path=ROOT_OUTPUT / "test18_dense.log", env=env)


def apply_test(policy: dict[str, object]) -> dict[str, object]:
    if not (TEST_OUTPUT / "report.json").is_file():
        command = [
            sys.executable, "-m", "scripts.goal_oriented_multimodal_policy",
            "--cache", str(TEST_CACHE), "--metadata", str(TEST_META),
            "--whistle-dir", str(TEST_WHISTLE), "--output", str(TEST_OUTPUT),
            "--policy", str(CAL_OUTPUT / "policy.json"), "--split-name", "test18",
        ]
        if teacher_required(policy):
            command.extend(["--teacher-shard-dir", str(TEST_TEACHER)])
        run(command, log_path=TEST_OUTPUT / "apply.log")
    return json.loads((TEST_OUTPUT / "report.json").read_text(encoding="utf-8"))


def final_summary(cal_report: dict[str, object], test_report: dict[str, object], policy: dict[str, object]) -> None:
    baseline_path = ROOT / "outputs/football_goal_pipeline/stage1_only_cal18_epoch2_20260830/report.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.is_file() else None
    summary = {
        "schema": "football_goal_oriented_pipeline_result.v1",
        "status": "complete",
        "policy": str(CAL_OUTPUT / "policy.json"),
        "calibration": cal_report,
        "external_test": test_report,
        "stage1_only_calibration_baseline": baseline,
        "teacher_used": teacher_required(policy),
        "no_temporal_nms": True,
        "primary_workload_kpi": "review_interval_union_duration / original_video_duration",
        "human_feedback_command": (
            "python scripts/ingest_goal_review_annotations.py --annotations "
            f"{TEST_OUTPUT / 'review_annotations.csv'} --output-dir "
            f"{TEST_OUTPUT / 'review_feedback'}"
        ),
    }
    (ROOT_OUTPUT / "final_result_20260830.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    try:
        record("cal_teacher", "running")
        if not teacher_complete(CAL_TEACHER):
            wait_for_external(
                "extract_candidate_clip_teacher_scores.py.*clip_teacher_512_center_cal18_epoch2_20260830",
                "externally launched calibration teacher",
            )
        launch_teacher(CAL_CACHE, CAL_TEACHER)
        record("cal_teacher", "complete")

        record("cal_audio", "running")
        wait_for_external("build_audio_feature_index.py.*cal18_5hz_20260830", "calibration audio")
        build_audio(CAL_IDS, CAL_AUDIO)
        build_whistles(CAL_IDS, CAL_AUDIO, CAL_WHISTLE)
        record("cal_audio", "complete")

        record("calibrate", "running")
        policy = calibrate()
        cal_report = json.loads((CAL_OUTPUT / "report.json").read_text(encoding="utf-8"))
        record("calibrate", "complete", teacher_used=teacher_required(policy))

        record("test_dense", "running")
        test_dense()
        record("test_dense", "complete")

        record("test_audio", "running")
        build_audio(TEST_IDS, TEST_AUDIO)
        build_whistles(TEST_IDS, TEST_AUDIO, TEST_WHISTLE)
        record("test_audio", "complete")

        if teacher_required(policy):
            record("test_teacher", "running")
            launch_teacher(TEST_CACHE, TEST_TEACHER)
            record("test_teacher", "complete")
        else:
            record("test_teacher", "skipped", reason="calibration selected Stage-1 scores")

        record("test_apply", "running")
        test_report = apply_test(policy)
        record("test_apply", "complete")
        final_summary(cal_report, test_report, policy)
        record("pipeline", "complete")
        log("pipeline complete")
    except Exception as exc:
        record("pipeline", "failed", error=f"{type(exc).__name__}: {exc}")
        log(f"pipeline failed: {type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
