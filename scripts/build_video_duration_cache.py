#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".webm")


def read_ids(paths: list[Path]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            video_id = line.strip()
            if not video_id or video_id.startswith("#") or video_id in seen:
                continue
            seen.add(video_id)
            values.append(video_id)
    return values


def find_video(roots: list[Path], video_id: str) -> Path | None:
    for root in roots:
        for ext in EXTENSIONS:
            candidate = root / f"{video_id}{ext}"
            if candidate.is_file():
                return candidate
    return None


def probe(path: Path, timeout_sec: float) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", str(path),
        ],
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=True,
    )
    return float(result.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id-file", type=Path, action="append", required=True)
    parser.add_argument("--video-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-sec", type=float, default=20.0)
    args = parser.parse_args()
    ids = read_ids(args.video_id_file)
    found = {video_id: find_video(args.video_root, video_id) for video_id in ids}
    durations: dict[str, float] = {}
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as pool:
        futures = {
            pool.submit(probe, path, args.timeout_sec): video_id
            for video_id, path in found.items()
            if path is not None
        }
        for future in as_completed(futures):
            video_id = futures[future]
            try:
                duration = float(future.result())
                if duration > 0:
                    durations[video_id] = duration
                else:
                    errors[video_id] = f"non-positive duration {duration}"
            except Exception as exc:
                errors[video_id] = f"{type(exc).__name__}: {exc}"
    for video_id, path in found.items():
        if path is None:
            errors[video_id] = "video not found"
    payload = {
        "protocol": "ffprobe_video_duration_cache_v1",
        "video_roots": [str(path) for path in args.video_root],
        "requested": len(ids),
        "resolved": len(durations),
        "durations_sec": dict(sorted(durations.items())),
        "errors": dict(sorted(errors.items())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "requested": len(ids), "resolved": len(durations), "errors": len(errors)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
