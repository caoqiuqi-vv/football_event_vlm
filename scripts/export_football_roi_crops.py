#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from football_detection_aware import RobustClipCropper
from train_football_events import get_video_duration, parse_image_size

VIDEO_EXTENSIONS = (".mp4", ".MP4", ".mov", ".MOV", ".mkv", ".MKV", ".avi", ".AVI")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export representative still-image ROI crops for robust detection-aware windows.")
    parser.add_argument("--index-root", default="outputs/football_roi_indices/robust_v2")
    parser.add_argument("--video-root", default="/mnt/data_16t/football/raw_video_720P")
    parser.add_argument("--video-id-file", default="configs/football/splits/detector_aware_seed42_58videos/val_video_ids.txt")
    parser.add_argument("--video-ids", default="", help="Optional comma-separated override.")
    parser.add_argument("--output-dir", default="outputs/football_roi_crops/robust_v2")
    parser.add_argument("--clip-sec", type=float, default=10.0)
    parser.add_argument("--stride-sec", type=float, default=10.0)
    parser.add_argument("--input-size", default="384,640", help="ROI classifier input height,width; controls integer-multiple ROI fitting.")
    parser.add_argument("--max-windows-per-video", type=int, default=0)
    parser.add_argument("--save-invalid-full-frame", action="store_true", help="Save full-frame middle images when ROI is invalid.")
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--padding", type=float, default=0.0)
    parser.add_argument("--min-area-ratio", type=float, default=0.0)
    parser.add_argument("--max-area-ratio", type=float, default=0.35)
    parser.add_argument("--min-roi-confidence", type=float, default=0.40)
    return parser.parse_args()


def read_video_ids(path: str) -> list[str]:
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip() and not line.startswith("#")]


def find_video(root: Path, video_id: str) -> Path:
    for extension in VIDEO_EXTENSIONS:
        path = root / f"{video_id}{extension}"
        if path.exists():
            return path
    matches = sorted(path for path in root.glob(f"{video_id}*") if path.suffix in VIDEO_EXTENSIONS)
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Missing video_id={video_id} under {root}")


def video_geometry(path: Path) -> tuple[int, int, float, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise FileNotFoundError(f"Could not open {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    if width <= 0 or height <= 0 or fps <= 0 or frames <= 0:
        raise RuntimeError(f"Invalid video metadata for {path}: {width}x{height} fps={fps} frames={frames}")
    return width, height, fps, frames


def build_windows(duration: float, clip_sec: float, stride_sec: float) -> list[tuple[float, float]]:
    max_start = max(duration - clip_sec, 0.0)
    starts: list[float] = []
    start = 0.0
    while start <= max_start + 1e-6:
        starts.append(start)
        start += stride_sec
    if not starts:
        starts = [0.0]
    if starts[-1] < max_start - 1e-6:
        starts.append(max_start)
    return [(start, min(start + clip_sec, duration)) for start in starts]


def read_middle_frame(video_path: Path, start_sec: float, end_sec: float) -> Any | None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        return None
    middle_sec = (float(start_sec) + float(end_sec)) * 0.5
    capture.set(cv2.CAP_PROP_POS_MSEC, middle_sec * 1000.0)
    ok, frame = capture.read()
    capture.release()
    return frame if ok else None


def safe_crop(frame: Any, bbox: tuple[int, int, int, int] | None) -> Any:
    if bbox is None:
        return frame
    height, width = frame.shape[:2]
    x1 = max(0, min(int(bbox[0]), width - 1))
    y1 = max(0, min(int(bbox[1]), height - 1))
    x2 = max(x1 + 1, min(int(bbox[2]), width))
    y2 = max(y1 + 1, min(int(bbox[3]), height))
    return frame[y1:y2, x1:x2]


def main() -> None:
    args = parse_args()
    video_root = Path(args.video_root)
    output_dir = Path(args.output_dir)
    crops_dir = output_dir / "crops"
    output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    input_size = parse_image_size(args.input_size)
    cropper = RobustClipCropper(
        {
            "index_root": args.index_root,
            "padding": args.padding,
            "min_crop_area_ratio": args.min_area_ratio,
            "max_crop_area_ratio": args.max_area_ratio,
            "min_roi_confidence": args.min_roi_confidence,
        }
    )
    video_ids = (
        [item.strip() for item in args.video_ids.split(",") if item.strip()]
        if args.video_ids.strip()
        else read_video_ids(args.video_id_file)
    )

    rows: list[dict[str, Any]] = []
    jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]
    for video_index, video_id in enumerate(video_ids, start=1):
        if not cropper.has_video(video_id):
            rows.append({"video_id": video_id, "status": "missing_index"})
            print(f"[{video_index}/{len(video_ids)}] skip missing index video_id={video_id}", flush=True)
            continue
        video_path = find_video(video_root, video_id)
        duration = get_video_duration(str(video_path))
        width, height, fps, frame_count = video_geometry(video_path)
        windows = build_windows(duration, args.clip_sec, args.stride_sec)
        if args.max_windows_per_video > 0:
            windows = windows[: args.max_windows_per_video]
        written = 0
        for window_index, (start_sec, end_sec) in enumerate(windows):
            proposal = cropper.get_window_roi(video_id, start_sec, end_sec, width, height, input_size)
            row: dict[str, Any] = {
                "video_id": video_id,
                "window_index": window_index,
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
                "middle_sec": float((start_sec + end_sec) * 0.5),
                "valid": bool(proposal.valid),
                "mode": proposal.mode,
                "bbox": json.dumps(list(proposal.bbox), ensure_ascii=False) if proposal.bbox is not None else "",
                "roi_confidence": float(proposal.roi_confidence),
                "area_ratio": float(proposal.area_ratio),
                "goal_score": float(proposal.goal_score),
                "ball_score": float(proposal.ball_score),
                "center_circle_score": float(proposal.center_circle_score),
                "person_support": float(proposal.person_support),
                "fallback_reason": proposal.fallback_reason,
                "image_path": "",
                "status": "invalid_roi" if not proposal.valid else "ok",
            }
            if proposal.valid or args.save_invalid_full_frame:
                frame = read_middle_frame(video_path, start_sec, end_sec)
                if frame is None:
                    row["status"] = "frame_read_failed"
                else:
                    image = safe_crop(frame, proposal.bbox if proposal.valid else None)
                    prefix = "valid" if proposal.valid else "invalid_full"
                    rel_path = Path(video_id) / f"{window_index:05d}_{prefix}_{start_sec:07.1f}_{end_sec:07.1f}_{proposal.mode}.jpg"
                    out_path = crops_dir / rel_path
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out_path), image, jpeg_params)
                    row["image_path"] = str(out_path)
                    written += 1
            rows.append(row)
        print(f"[{video_index}/{len(video_ids)}] video_id={video_id} windows={len(windows)} images={written}", flush=True)

    manifest_path = output_dir / "roi_crops_manifest.csv"
    fieldnames = [
        "video_id",
        "window_index",
        "start_sec",
        "end_sec",
        "middle_sec",
        "valid",
        "mode",
        "bbox",
        "roi_confidence",
        "area_ratio",
        "goal_score",
        "ball_score",
        "center_circle_score",
        "person_support",
        "fallback_reason",
        "image_path",
        "status",
    ]
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    summary = {
        "index_root": args.index_root,
        "video_root": str(video_root),
        "video_id_file": args.video_id_file,
        "output_dir": str(output_dir),
        "clip_sec": args.clip_sec,
        "stride_sec": args.stride_sec,
        "input_size": list(input_size),
        "num_rows": len(rows),
        "num_images": sum(1 for row in rows if row.get("image_path")),
        "num_valid": sum(1 for row in rows if row.get("valid") is True),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"done images={summary['num_images']} manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
