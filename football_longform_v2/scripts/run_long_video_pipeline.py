from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RGB-only football long-video event inference."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--operating-points", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--context-batch-size", type=int, default=32)
    parser.add_argument("--reuse-cache", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    timeline = Path(args.cache_root).expanduser().resolve() / args.video_id / "timeline.npz"
    if timeline.exists() and not args.reuse_cache:
        raise FileExistsError(
            f"timeline exists; pass --reuse-cache to validate and reuse: {timeline}"
        )
    if not timeline.exists():
        build_command = [
            sys.executable, str(PROJECT_ROOT / "scripts/build_rgb_timeline.py"),
            "--config", str(config_path), "--video-id", args.video_id,
            "--video-root", str(Path(args.video_root).expanduser().resolve()),
            "--device", args.device, "--context-batch-size", str(args.context_batch_size),
            "--output", str(timeline),
        ]
        print("launch=" + " ".join(build_command), flush=True)
        subprocess.run(build_command, cwd=PROJECT_ROOT, check=True)
    infer_command = [
        sys.executable, str(PROJECT_ROOT / "scripts/infer_cached_long_video.py"),
        "--config", str(config_path), "--operating-points",
        str(Path(args.operating_points).expanduser().resolve()),
        "--timeline", str(timeline), "--video-id", args.video_id,
        "--device", args.device, "--output", str(Path(args.output).expanduser().resolve()),
    ]
    if args.checkpoint:
        infer_command.extend([
            "--checkpoint", str(Path(args.checkpoint).expanduser().resolve())
        ])
    print("launch=" + " ".join(infer_command), flush=True)
    subprocess.run(infer_command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
