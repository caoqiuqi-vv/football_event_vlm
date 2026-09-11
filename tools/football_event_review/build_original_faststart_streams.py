#!/usr/bin/env python3
"""Build original-quality, seek-friendly MP4 streams without re-encoding."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import fcntl
import os
import shutil

from mp4_layout import is_faststart
import subprocess
from fractions import Fraction
from pathlib import Path


def probe(path: Path) -> dict:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "format=duration,size:stream=codec_name,width,height,avg_frame_rate",
        "-of", "json", str(path),
    ]
    return json.loads(subprocess.check_output(command, text=True))


def verify_stream_copy(video_id: str, source_probe: dict, target_probe: dict) -> None:
    src_stream = source_probe["streams"][0]
    dst_stream = target_probe["streams"][0]
    for key in ("codec_name", "width", "height"):
        if dst_stream.get(key) != src_stream.get(key):
            raise RuntimeError(f"{video_id}: stream-copy verification failed for {key}")
    src_fps = float(Fraction(src_stream["avg_frame_rate"]))
    dst_fps = float(Fraction(dst_stream["avg_frame_rate"]))
    if abs(src_fps - dst_fps) > 0.01:
        raise RuntimeError(
            f"{video_id}: frame rate changed from {src_fps:.6f} to {dst_fps:.6f}"
        )
    src_duration = float(source_probe["format"]["duration"])
    dst_duration = float(target_probe["format"]["duration"])
    if abs(src_duration - dst_duration) > 0.1:
        raise RuntimeError(f"{video_id}: duration changed by {dst_duration-src_duration:.3f}s")


def remux(video: dict, output_dir: Path) -> dict:
    video_id = str(video["video_id"])
    if Path(video_id).name != video_id or video_id in {".", ".."}:
        raise ValueError("Unsafe video id")
    source = Path(video["video_path"]).resolve()
    target = output_dir / f"{video_id}.mp4"
    if source == target.resolve():
        raise ValueError("Source and output must be different paths")
    temporary = output_dir / f".{video_id}.part.mp4"
    marker = output_dir / f"{video_id}.source.json"
    st = source.stat()
    signature = {"path": str(source), "size": st.st_size, "mtime_ns": st.st_mtime_ns}
    source_probe = probe(source)
    if target.is_file() and marker.is_file():
        try:
            if json.loads(marker.read_text()) == signature and is_faststart(target):
                verify_stream_copy(video_id, source_probe, probe(target))
                print(f"SKIP {video_id}", flush=True)
                return {"video_id": video_id, "path": str(target), "status": "ready"}
        except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError):
            pass
    # ffmpeg writes a private temporary file; readers see only validated copies.
    # Keep a filesystem reserve and fail before overwriting any serving file.
    if shutil.disk_usage(output_dir).free < st.st_size + 5 * 1024**3:
        raise RuntimeError("Insufficient free space for media copy plus 5 GiB reserve")
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", str(source), "-map", "0:v:0", "-map", "0:a:0?",
        "-c", "copy", "-movflags", "+faststart", str(temporary),
    ]
    print(f"START {video_id} bytes={st.st_size}", flush=True)
    subprocess.run(command, check=True)
    verify_stream_copy(video_id, source_probe, probe(temporary))
    if not is_faststart(temporary):
        raise RuntimeError(f"{video_id}: moov was not moved before mdat")
    current = source.stat()
    if (current.st_size, current.st_mtime_ns) != (st.st_size, st.st_mtime_ns):
        raise RuntimeError(f"{video_id}: source changed during copy")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(target)
    marker_tmp = marker.with_suffix(".tmp")
    marker_tmp.write_text(json.dumps(signature) + "\n")
    marker_tmp.replace(marker)
    print(f"DONE {video_id} bytes={target.stat().st_size}", flush=True)
    return {"video_id": video_id, "path": str(target), "status": "ready"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    preparation_lock = (args.output_dir / ".prepare.lock").open("a")
    fcntl.flock(preparation_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    videos = list(manifest["videos"])
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = [executor.submit(remux, video, args.output_dir) for video in videos]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as error:
                failures.append(repr(error))
                print(f"ERROR {error!r}", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} remux jobs failed: {failures}")
    print(f"ALL_READY videos={len(videos)} output_dir={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
