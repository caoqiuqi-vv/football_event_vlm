from __future__ import annotations

"""Wait for A1, then select, cache, and train A2 without leaving GPUs idle."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
SCRIPTS = PROJECT_ROOT / "scripts"


def process_is_expected(pid: int, expected_fragment: str) -> bool:
    command_path = Path(f"/proc/{pid}/cmdline")
    if not command_path.is_file():
        return False
    try:
        command = command_path.read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return False
    return expected_fragment in command


def run_logged(command: list[str], log, *, env: dict[str, str] | None = None) -> None:
    log.write(json.dumps({"event": "command", "command": command}) + "\n")
    log.flush()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"command failed returncode={completed.returncode}: {command}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--expected-command-fragment", default="lf_a1_videomae_clean5.yaml")
    parser.add_argument(
        "--experiment",
        default="football_longform_v2/experiments/lf_a1_videomae_clean5",
    )
    parser.add_argument(
        "--canonical-config",
        default="football_longform_v2/configs/lf_a0_official_fullscale.yaml",
    )
    parser.add_argument(
        "--a1-config",
        default="football_longform_v2/configs/lf_a1_videomae_clean5.yaml",
    )
    parser.add_argument(
        "--a2-config",
        default="football_longform_v2/configs/lf_a2_sequential_locator.yaml",
    )
    parser.add_argument(
        "--store-root",
        default="/mnt/data_16t/football/feature_store_v2/videomae_k710_a1_best_4s_v1",
    )
    parser.add_argument("--gpus", default="0,6,7")
    parser.add_argument("--extract-batch-size", type=int, default=24)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument(
        "--state",
        default="football_longform_v2/experiments/lf_a2_sequential_locator/continuation_state.json",
    )
    args = parser.parse_args()
    experiment = (REPO_ROOT / args.experiment).resolve()
    state_path = (REPO_ROOT / args.state).resolve()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = state_path.with_suffix(".log")
    state = {
        "schema": "football_longform_v2.a1_to_a2_continuation.v1",
        "wait_pid": args.wait_pid,
        "expected_command_fragment": args.expected_command_fragment,
        "stage": "waiting_for_a1",
        "fixed_thirdparty18_opened": False,
    }

    def save_state() -> None:
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    save_state()
    with log_path.open("a", encoding="utf-8") as log:
        try:
            while process_is_expected(args.wait_pid, args.expected_command_fragment):
                time.sleep(max(args.poll_seconds, 1.0))
            state["stage"] = "selecting_a1_checkpoint"
            save_state()
            run_logged([
                sys.executable,
                str(SCRIPTS / "select_a1_supported_checkpoint.py"),
                "--experiment", str(experiment),
            ], log)
            selection_path = experiment / "supported_checkpoint_selection.json"
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            checkpoint = Path(selection["best"]["checkpoint"]).resolve()
            state.update({
                "stage": "extracting_chunk_store",
                "selected_epoch": int(selection["best"]["epoch"]),
                "selected_checkpoint": str(checkpoint),
                "selection_tuple": selection["best"]["selection_tuple"],
            })
            save_state()
            run_logged([
                sys.executable,
                str(SCRIPTS / "coordinate_videomae_chunk_store.py"),
                "--canonical-config", str((REPO_ROOT / args.canonical_config).resolve()),
                "--a1-config", str((REPO_ROOT / args.a1_config).resolve()),
                "--checkpoint", str(checkpoint),
                "--output-root", str(Path(args.store_root).expanduser().resolve()),
                "--gpus", args.gpus,
                "--batch-size", str(args.extract_batch_size),
                "--chunk-seconds", "4.0",
                "--expected-train", "135",
                "--expected-calibration", "18",
            ], log)
            state["stage"] = "training_a2"
            save_state()
            visible = tuple(int(item) for item in args.gpus.split(",") if item.strip())
            relative_ids = ",".join(str(index) for index in range(len(visible)))
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(item) for item in visible)
            run_logged([
                sys.executable,
                str(SCRIPTS / "train_chunk_locator.py"),
                "--config", str((REPO_ROOT / args.a2_config).resolve()),
                "--device", "cuda:0",
                "--device-ids", relative_ids,
            ], log, env=env)
            state["stage"] = "complete"
            state["completed_at_unix"] = time.time()
            save_state()
        except Exception as error:
            state["stage"] = "failed"
            state["error"] = repr(error)
            save_state()
            log.write(f"failure={error!r}\n")
            log.flush()
            raise


if __name__ == "__main__":
    main()
