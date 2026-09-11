#!/usr/bin/env python
"""Render bounded review clips from a goal-oriented review manifest."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def render(row: dict[str, str], video_root: Path, output: Path, crf: int) -> dict[str, object]:
    video_id = row["video_id"]
    source = video_root / f"{video_id}.mp4"
    destination = output / video_id / f"{row['segment_index']}.mp4"
    destination.parent.mkdir(parents=True, exist_ok=True)
    start = float(row["start_sec"])
    duration = max(float(row["end_sec"]) - start, 0.05)
    if destination.is_file() and destination.stat().st_size > 1024:
        return {"segment_id": row["segment_id"], "status": "exists", "path": str(destination)}
    command = [
        "ffmpeg", "-nostdin", "-v", "error", "-ss", f"{start:.3f}",
        "-i", str(source), "-t", f"{duration:.3f}", "-map", "0:v:0",
        "-map", "0:a?", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", str(crf), "-c:a", "aac", "-movflags", "+faststart", "-y",
        str(destination),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    return {
        "segment_id": row["segment_id"],
        "status": "ok" if result.returncode == 0 else "failed",
        "path": str(destination),
        "error": result.stderr[-500:] if result.returncode else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-segments", required=True, type=Path)
    parser.add_argument("--video-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--crf", type=int, default=24)
    args = parser.parse_args()
    with args.review_segments.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(args.workers, 1)) as executor:
        futures = [executor.submit(render, row, args.video_root, args.output_dir, args.crf) for row in rows]
        for future in as_completed(futures):
            item = future.result()
            results.append(item)
            if item["status"] == "failed":
                print(json.dumps(item, ensure_ascii=False), flush=True)
    results.sort(key=lambda item: str(item["segment_id"]))
    summary = {
        "segments": len(rows),
        "ok": sum(item["status"] in {"ok", "exists"} for item in results),
        "failed": sum(item["status"] == "failed" for item in results),
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "render_results.json").write_text(
        json.dumps({"summary": summary, "items": results}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
