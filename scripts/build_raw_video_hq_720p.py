#!/usr/bin/env python
from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_SOURCE_ROOTS = (
    "/mnt/data/Datasets/Datasets/Football/xbotgo_football_data_0608/videos",
    "/mnt/data/Datasets/Datasets/Football/xbotgo_football_data_0608/may/Xbotgo/videos",
    "/mnt/data/Datasets/Datasets/Football/may/Xbotgo/videos",
    "/mnt/data/Datasets/Datasets/Football/may/human_data/Xbotgo/videos",
)


def run_json(cmd: list[str]) -> dict:
    return json.loads(subprocess.check_output(cmd, text=True))


def ffprobe_video(path: Path) -> dict:
    data = run_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,codec_name,pix_fmt,avg_frame_rate,nb_frames,duration",
            "-show_entries",
            "format=size,bit_rate,duration",
            "-of",
            "json",
            str(path),
        ]
    )
    stream = data.get("streams", [{}])[0]
    fmt = data.get("format", {})
    return {"stream": stream, "format": fmt}


def is_valid_video(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 0:
        return False
    try:
        info = ffprobe_video(path)
    except Exception:
        return False
    stream = info.get("stream", {})
    return int(stream.get("width") or 0) > 0 and int(stream.get("height") or 0) > 0


def discover_sources(source_roots: list[Path], video_ids: set[str] | None) -> list[Path]:
    found: dict[str, Path] = {}
    for root in source_roots:
        if not root.exists():
            continue
        for path in sorted(root.glob("*.mp4")):
            video_id = path.stem
            if video_ids is not None and video_id not in video_ids:
                continue
            # Keep first source-root priority.
            found.setdefault(video_id, path)
    return [found[key] for key in sorted(found)]


def build_ffmpeg_cmd(src: Path, dst_tmp: Path, crf: int, preset: str, audio: str) -> list[str]:
    # Scale only if height is above 720; preserve source when already <=720.
    scale_expr = "scale='if(gt(ih,720),-2,iw)':'if(gt(ih,720),720,ih)':flags=lanczos"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-y",
        "-i",
        str(src),
        "-map",
        "0:v:0",
        "-vf",
        scale_expr,
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-profile:v",
        "high",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]
    if audio == "copy":
        cmd += ["-map", "0:a?", "-c:a", "copy"]
    elif audio == "aac":
        cmd += ["-map", "0:a?", "-c:a", "aac", "-b:a", "128k"]
    elif audio == "drop":
        cmd += ["-an"]
    else:
        raise ValueError(f"Unsupported audio mode: {audio}")
    cmd.append(str(dst_tmp))
    return cmd


def transcode_one(src: Path, output_dir: Path, crf: int, preset: str, audio: str, overwrite: bool) -> dict:
    dst = output_dir / src.name
    tmp = output_dir / f".{src.stem}.tmp.mp4"
    if dst.exists() and not overwrite and is_valid_video(dst):
        return {"video_id": src.stem, "status": "skip_exists", "src": str(src), "dst": str(dst)}
    tmp.unlink(missing_ok=True)
    cmd = build_ffmpeg_cmd(src, tmp, crf, preset, audio)
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        return {
            "video_id": src.stem,
            "status": "failed",
            "src": str(src),
            "dst": str(dst),
            "returncode": proc.returncode,
            "stderr_tail": proc.stderr[-4000:],
        }
    if not is_valid_video(tmp):
        tmp.unlink(missing_ok=True)
        return {"video_id": src.stem, "status": "failed_invalid_output", "src": str(src), "dst": str(dst)}
    tmp.replace(dst)
    src_info = ffprobe_video(src)
    dst_info = ffprobe_video(dst)
    return {
        "video_id": src.stem,
        "status": "done",
        "src": str(src),
        "dst": str(dst),
        "src_stream": src_info["stream"],
        "src_format": src_info["format"],
        "dst_stream": dst_info["stream"],
        "dst_format": dst_info["format"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build high-quality 720P football videos from original sources.")
    parser.add_argument("--source-root", action="append", default=[], help="Source directory containing full-video mp4 files. Can be repeated.")
    parser.add_argument("--output-dir", default="/mnt/data_16t/football/raw_video_hq_720P")
    parser.add_argument("--video-ids", default="", help="Comma-separated video IDs. Empty means all discovered videos.")
    parser.add_argument("--video-id-file", default="", help="Optional text file with one video ID per line.")
    parser.add_argument("--crf", type=int, default=18, help="H.264 CRF. Lower is higher quality/larger. Recommended: 18-20.")
    parser.add_argument("--preset", default="medium", help="x264 preset: veryfast, faster, fast, medium, slow...")
    parser.add_argument("--audio", choices=["drop", "copy", "aac"], default="drop", help="Audio handling. Model does not need audio, so default drops it.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel ffmpeg jobs. Use 1-2 to avoid saturating disk/CPU.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--manifest-name", default="manifest.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_roots = [Path(item) for item in (args.source_root or list(DEFAULT_SOURCE_ROOTS))]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_ids: set[str] | None = None
    ids: list[str] = []
    if args.video_ids.strip():
        ids.extend(item.strip() for item in args.video_ids.split(",") if item.strip())
    if args.video_id_file:
        ids.extend(
            line.strip()
            for line in Path(args.video_id_file).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    if ids:
        video_ids = set(ids)
    sources = discover_sources(source_roots, video_ids)
    if not sources:
        raise SystemExit(f"No source videos found. source_roots={[str(p) for p in source_roots]} video_ids={sorted(video_ids or [])}")
    print(f"source_roots={[str(p) for p in source_roots]}", flush=True)
    print(f"output_dir={output_dir}", flush=True)
    print(f"videos={len(sources)} crf={args.crf} preset={args.preset} audio={args.audio} workers={args.workers}", flush=True)
    manifest = output_dir / args.manifest_name
    with manifest.open("a", encoding="utf-8") as mf:
        with futures.ThreadPoolExecutor(max_workers=max(args.workers, 1)) as executor:
            future_to_src = {
                executor.submit(transcode_one, src, output_dir, args.crf, args.preset, args.audio, args.overwrite): src
                for src in sources
            }
            for index, future in enumerate(futures.as_completed(future_to_src), start=1):
                result = future.result()
                mf.write(json.dumps(result, ensure_ascii=False) + "\n")
                mf.flush()
                print(f"[{index}/{len(sources)}] {result['video_id']} {result['status']}", flush=True)
                if result["status"].startswith("failed"):
                    print(json.dumps(result, ensure_ascii=False, indent=2), file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
