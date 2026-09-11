#!/usr/bin/env python3
"""Build seek-friendly 720P MP4 proxies for the football review UI."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
from pathlib import Path


def encode(
    video: dict,
    output_dir: Path,
    gpu: int,
    height: int | None = None,
    cq: int = 23,
    video_bitrate: str = "3M",
    maxrate: str = "5M",
    bufsize: str = "10M",
) -> tuple[str, Path]:
    video_id = str(video["video_id"])
    source = Path(video["video_path"])
    target = output_dir / f"{video_id}.mp4"
    temporary = output_dir / f"{video_id}.tmp.mp4"
    if target.exists() and target.stat().st_size > 1_000_000:
        print(f"SKIP {video_id} {target}", flush=True)
        return video_id, target
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-hwaccel", "cuda", "-hwaccel_device", str(gpu),
        "-i", str(source),
        "-map", "0:v:0", "-map", "0:a:0?",
        "-c:v", "h264_nvenc", "-gpu", str(gpu), "-preset", "p4", "-tune", "hq",
        "-rc", "vbr", "-cq", str(cq), "-b:v", video_bitrate, "-maxrate", maxrate, "-bufsize", bufsize,
        "-g", "60", "-keyint_min", "60", "-forced-idr", "1",
        "-c:a", "copy", "-movflags", "+faststart", str(temporary),
    ]
    if height is not None:
        # low-res proxy: decode on GPU, scale in software, NVENC re-uploads.
        # -vf must come after -i <source> (it is an output option).
        idx = command.index(str(source)) + 1
        command[idx:idx] = ["-vf", f"scale=-2:{height}"]
    print(f"START {video_id} gpu={gpu} height={height}", flush=True)
    subprocess.run(command, check=True)
    temporary.replace(target)
    print(f"DONE {video_id} gpu={gpu} height={height} bytes={target.stat().st_size}", flush=True)
    return video_id, target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--gpus", default="2,3")
    parser.add_argument("--height", type=int, default=None,
                        help="Scale to this height (e.g. 360 for a tunnel-friendly low-res proxy). "
                             "Default: keep source resolution.")
    parser.add_argument("--cq", type=int, default=23)
    parser.add_argument("--video-bitrate", default="3M")
    parser.add_argument("--maxrate", default="5M")
    parser.add_argument("--bufsize", default="10M")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gpus = [int(item) for item in args.gpus.split(",") if item.strip()]
    videos = list(manifest["videos"])
    results: dict[str, Path] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {
            executor.submit(encode, video, args.output_dir, gpus[index % len(gpus)],
                            args.height, args.cq, args.video_bitrate, args.maxrate, args.bufsize): video
            for index, video in enumerate(videos)
        }
        for future in concurrent.futures.as_completed(futures):
            video_id, path = future.result()
            results[video_id] = path
    for video in videos:
        video["source_video_path"] = video["video_path"]
        video["video_path"] = str(results[str(video["video_id"])])
        video["streaming_proxy"] = {"faststart": True, "keyframe_interval_sec": 2.0}
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"MANIFEST {args.output_manifest}", flush=True)


if __name__ == "__main__":
    main()
