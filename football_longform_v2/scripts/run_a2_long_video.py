from __future__ import annotations

"""End-to-end unlabeled long-video entrypoint: cache once, then locate events."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
SCRIPTS = PROJECT_ROOT / "scripts"


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    completed = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
    if completed.returncode:
        raise RuntimeError(f"command failed returncode={completed.returncode}: {command}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--a1-config", required=True)
    parser.add_argument("--a1-checkpoint", required=True)
    parser.add_argument("--a2-checkpoint", required=True)
    parser.add_argument("--operating-points", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args()
    video = Path(args.video).expanduser().resolve()
    work_root = Path(args.work_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    run([
        sys.executable,
        str(SCRIPTS / "build_videomae_chunk_store.py"),
        "--a1-config", str(Path(args.a1_config).expanduser().resolve()),
        "--checkpoint", str(Path(args.a1_checkpoint).expanduser().resolve()),
        "--split", "inference",
        "--video", str(video),
        "--video-id", args.video_id,
        "--output-root", str(work_root),
        "--device", "cuda:0",
        "--batch-size", str(args.batch_size),
        "--chunk-seconds", "4.0",
    ], env=env)
    timeline = work_root / "inference" / args.video_id
    run([
        sys.executable,
        str(SCRIPTS / "infer_chunk_locator.py"),
        "--checkpoint", str(Path(args.a2_checkpoint).expanduser().resolve()),
        "--operating-points", str(Path(args.operating_points).expanduser().resolve()),
        "--timeline-dir", str(timeline),
        "--video-id", args.video_id,
        "--output", str(output),
        "--device", "cuda:0",
    ], env=env)
    print(json.dumps({
        "video": str(video), "video_id": args.video_id,
        "timeline": str(timeline), "output": str(output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
