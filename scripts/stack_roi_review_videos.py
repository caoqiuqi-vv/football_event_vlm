#!/usr/bin/env python3
"""Stack matching fixed and dynamic ROI review videos for manual comparison."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-root", type=Path, required=True)
    parser.add_argument("--dynamic-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fixed_review_root = args.fixed_root / "videos" / "review"
    dynamic_review_root = args.dynamic_root / "videos" / "review"
    video_output_root = args.output_dir / "videos"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fixed_paths = sorted(fixed_review_root.rglob("*.mp4"))
    rows: list[dict[str, str]] = []
    counts = {"ok": 0, "skipped": 0, "missing_dynamic": 0, "error": 0}
    for fixed_path in fixed_paths:
        relative_path = fixed_path.relative_to(fixed_review_root)
        dynamic_path = dynamic_review_root / relative_path
        output_path = video_output_root / relative_path
        row = {
            "relative_path": str(relative_path),
            "fixed_path": str(fixed_path),
            "dynamic_path": str(dynamic_path),
            "comparison_path": str(output_path),
            "status": "",
            "error": "",
        }
        if not dynamic_path.is_file():
            row["status"] = "missing_dynamic"
            counts["missing_dynamic"] += 1
            rows.append(row)
            continue
        if output_path.is_file() and not args.force:
            row["status"] = "skipped"
            counts["skipped"] += 1
            rows.append(row)
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            args.ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(fixed_path),
            "-i",
            str(dynamic_path),
            "-filter_complex",
            "[0:v][1:v]hstack=inputs=2[v]",
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-preset",
            args.preset,
            "-crf",
            str(args.crf),
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            row["status"] = "ok"
            counts["ok"] += 1
        else:
            row["status"] = "error"
            row["error"] = result.stderr.strip().replace("\n", " ")
            counts["error"] += 1
        rows.append(row)

    manifest_path = args.output_dir / "comparison_manifest.csv"
    fields = [
        "relative_path",
        "fixed_path",
        "dynamic_path",
        "comparison_path",
        "status",
        "error",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"paired={len(rows)} ok={counts['ok']} skipped={counts['skipped']} "
        f"missing_dynamic={counts['missing_dynamic']} errors={counts['error']} "
        f"manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
