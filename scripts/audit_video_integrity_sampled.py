#!/usr/bin/env python3
"""Low-impact video integrity audit using ffprobe plus sampled short decodes."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-sec", type=float, default=1.0)
    parser.add_argument("--timeout-sec", type=float, default=20.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def run(command: list[str], timeout: float) -> tuple[int, str, str, float]:
    started = time.monotonic()
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
        return result.returncode, result.stdout, result.stderr, time.monotonic() - started
    except subprocess.TimeoutExpired as exc:
        stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        return 124, "", f"timeout after {timeout:.1f}s {stderr}".strip(), time.monotonic() - started


def probe(path: Path, timeout: float) -> tuple[dict, str, float]:
    rc, stdout, stderr, elapsed = run([
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration,size:stream=index,codec_type,codec_name,width,height,avg_frame_rate,duration,nb_frames",
        "-of", "json", str(path),
    ], timeout)
    if rc != 0:
        return {}, stderr.strip() or f"ffprobe rc={rc}", elapsed
    try:
        return json.loads(stdout), stderr.strip(), elapsed
    except json.JSONDecodeError as exc:
        return {}, f"invalid ffprobe json: {exc}", elapsed


def fraction(value: str) -> float:
    try:
        if "/" in value:
            lhs, rhs = value.split("/", 1)
            return float(lhs) / float(rhs) if float(rhs) else 0.0
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def sampled_times(duration: float) -> list[float]:
    raw = [2.0, duration * 0.25, duration * 0.5, duration * 0.75,
           max(duration - 60.0, 0.0), max(duration - 15.0, 0.0), max(duration - 3.0, 0.0)]
    values = sorted({round(min(max(value, 0.0), max(duration - 0.5, 0.0)), 3) for value in raw})
    return values


def decode_sample(path: Path, start: float, sample_sec: float, timeout: float) -> tuple[bool, str, float]:
    rc, _, stderr, elapsed = run([
        "ffmpeg", "-nostdin", "-v", "error", "-xerror", "-ss", f"{start:.3f}",
        "-i", str(path), "-map", "0:v:0", "-t", f"{sample_sec:.3f}",
        "-an", "-sn", "-dn", "-f", "null", "-",
    ], timeout)
    return rc == 0, stderr.strip(), elapsed


def save(output: Path, payload: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(output)
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["video_id", "status", "duration_sec", "width", "height", "fps", "size_bytes", "failed_points", "errors"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in payload["results"]:
            writer.writerow({key: item.get(key, "") for key in fields})


def main() -> None:
    args = parse_args()
    root = Path(args.video_dir)
    output = Path(args.output)
    videos = sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES)
    previous = {}
    if args.resume and output.is_file():
        old = json.loads(output.read_text(encoding="utf-8"))
        previous = {item["path"]: item for item in old.get("results", [])}
    results = []
    for index, path in enumerate(videos, 1):
        stat = path.stat()
        cached = previous.get(str(path))
        if cached and cached.get("size_bytes") == stat.st_size and cached.get("mtime_ns") == stat.st_mtime_ns:
            item = cached
        else:
            metadata, probe_error, probe_elapsed = probe(path, args.timeout_sec)
            streams = [item for item in metadata.get("streams", []) if item.get("codec_type") == "video"]
            stream = streams[0] if streams else {}
            format_info = metadata.get("format", {})
            try:
                duration = float(format_info.get("duration") or stream.get("duration") or 0.0)
            except (TypeError, ValueError):
                duration = 0.0
            width = int(stream.get("width") or 0)
            height = int(stream.get("height") or 0)
            fps = fraction(str(stream.get("avg_frame_rate") or "0"))
            errors = [probe_error] if probe_error else []
            checks = []
            if not streams or duration <= 0 or width <= 0 or height <= 0 or fps <= 0:
                status = "fail"
                errors.append("invalid video metadata")
            else:
                for point in sampled_times(duration):
                    ok, error, elapsed = decode_sample(path, point, args.sample_sec, args.timeout_sec)
                    checks.append({"start_sec": point, "ok": ok, "elapsed_sec": elapsed, "error": error[:1000]})
                    if not ok:
                        errors.append(f"t={point:.3f}: {error or 'decode failed'}")
                status = "pass" if not errors else "suspicious"
            item = {
                "video_id": path.stem,
                "path": str(path),
                "status": status,
                "duration_sec": duration,
                "width": width,
                "height": height,
                "fps": fps,
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "probe_elapsed_sec": probe_elapsed,
                "checks": checks,
                "failed_points": sum(not check["ok"] for check in checks),
                "errors": " | ".join(errors)[:4000],
            }
        results.append(item)
        counts = {name: sum(row["status"] == name for row in results) for name in ("pass", "suspicious", "fail")}
        payload = {
            "protocol": "ffprobe_plus_7point_short_decode_v1",
            "video_dir": str(root),
            "sample_sec": args.sample_sec,
            "total_videos": len(videos),
            "completed": len(results),
            "counts": counts,
            "results": results,
        }
        save(output, payload)
        print(f"[{index}/{len(videos)}] {path.name} status={item['status']} counts={counts}", flush=True)

if __name__ == "__main__":
    main()
