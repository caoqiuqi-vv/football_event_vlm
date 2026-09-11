#!/usr/bin/env python3
"""Extract one representative gameplay clip per long video for team-colour review.

The clip is centred near the first quartile of known GT events so it is likely to
contain active play and both teams.  The original-resolution stream is copied
without re-encoding; team_cluster later samples upper-body player crops from it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--video-id-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clip-sec", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    return parser.parse_args()


def choose_interval(run_dir: Path, video_id: str, clip_sec: float) -> dict:
    summary = json.loads((run_dir / video_id / "summary.json").read_text(encoding="utf-8"))
    duration = float(summary["duration_sec"])
    events = json.loads((run_dir / video_id / "gt_events.json").read_text(encoding="utf-8"))
    times = sorted(
        float(item["time_sec"])
        for item in events
        if 20.0 <= float(item["time_sec"]) <= max(20.0, duration - 10.0)
    )
    if times:
        anchor = times[min(len(times) - 1, len(times) // 4)]
        reason = "gt_event_q25"
    else:
        anchor = min(90.0, duration * 0.2)
        reason = "duration_20pct_fallback"
    start = max(0.0, min(max(0.0, duration - clip_sec), anchor - clip_sec / 3.0))
    end = min(duration, start + clip_sec)
    return {
        "video_id": video_id,
        "source_video": str(Path(summary["video_path"]).resolve()),
        "duration_sec": round(duration, 3),
        "sample_start_sec": round(start, 3),
        "sample_end_sec": round(end, 3),
        "anchor_sec": round(anchor, 3),
        "selection_reason": reason,
    }


def extract(row: dict, output_dir: Path, ffmpeg: str) -> dict:
    output = output_dir / f"{row['video_id']}.mp4"
    if output.is_file() and output.stat().st_size > 1_000_000:
        return {**row, "clip_path": str(output.resolve()), "status": "existing"}
    temporary = output.with_suffix(".mp4.tmp")
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{row['sample_start_sec']:.3f}", "-i", row["source_video"],
        "-t", f"{row['sample_end_sec'] - row['sample_start_sec']:.3f}",
        "-map", "0:v:0", "-an", "-c:v", "copy", "-movflags", "+faststart",
        "-f", "mp4", str(temporary),
    ]
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed for {row['video_id']}: {result.stderr[-500:]}")
    temporary.replace(output)
    return {**row, "clip_path": str(output.resolve()), "status": "extracted"}


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    video_ids = [x.strip() for x in args.video_id_file.read_text().splitlines() if x.strip()]
    rows = [choose_interval(run_dir, video_id, args.clip_sec) for video_id in video_ids]
    completed = []
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(extract, row, output_dir, args.ffmpeg): row for row in rows}
        for future in as_completed(futures):
            row = futures[future]
            try:
                completed.append(future.result())
            except Exception as error:  # retain per-video failure for retry
                failures.append({**row, "status": "failed", "error": str(error)})
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_policy": "q25_gt_event_minus_one_third_clip_else_duration_20pct",
        "colour_region_policy": "upper_body_primary",
        "videos": sorted(completed + failures, key=lambda item: item["video_id"]),
        "summary": {"requested": len(rows), "ready": len(completed), "failed": len(failures)},
    }
    temporary = output_dir / "team_calibration_clip_plan.json.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_dir / "team_calibration_clip_plan.json")
    print(json.dumps(payload["summary"], ensure_ascii=False))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
