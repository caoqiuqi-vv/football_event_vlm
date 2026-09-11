#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from football_detection_aware import RobustClipCropper

VIDEO_EXTENSIONS = (".mp4", ".MP4", ".mov", ".MOV", ".mkv", ".MKV", ".avi", ".AVI")


def parse_size(value: str | tuple[int, int] | list[int]) -> tuple[int, int]:
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"Expected height,width, got {value}")
        return int(value[0]), int(value[1])
    text = str(value).strip().replace("x", ",").replace("X", ",")
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"Expected size as height,width, got {value!r}")
    height, width = int(parts[0]), int(parts[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Size must be positive, got {value!r}")
    return height, width


def read_id_file(path: str) -> list[str]:
    if not path:
        return []
    result: list[str] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        result.append(line)
    return result


def split_ids(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def infer_video_id(path: Path) -> str:
    return path.stem.split(".")[0]


def collect_video_ids(args: argparse.Namespace) -> list[str]:
    ids: list[str] = []
    ids.extend(split_ids(args.video_id))
    ids.extend(split_ids(args.video_ids))
    ids.extend(read_id_file(args.video_id_file))
    if args.video_path and not ids:
        ids.append(infer_video_id(Path(args.video_path)))
    if args.roi_index and not ids:
        ids.append(infer_video_id(Path(args.roi_index)))
    deduped: list[str] = []
    seen: set[str] = set()
    for video_id in ids:
        if video_id in seen:
            continue
        seen.add(video_id)
        deduped.append(video_id)
    if not deduped:
        raise ValueError("Provide --video-id, --video-ids, --video-id-file, --video-path, or --roi-index")
    return deduped


def find_video(video_root: Path, video_id: str) -> Path:
    for extension in VIDEO_EXTENSIONS:
        candidate = video_root / f"{video_id}{extension}"
        if candidate.exists():
            return candidate
    matches = sorted(path for path in video_root.glob(f"{video_id}*") if path.suffix in VIDEO_EXTENSIONS)
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Missing video_id={video_id} under {video_root}")


def resolve_video_path(args: argparse.Namespace, video_id: str, total_videos: int) -> Path:
    if args.video_path:
        path = Path(args.video_path).expanduser()
        if total_videos > 1:
            expected_id = infer_video_id(path)
            if expected_id != video_id:
                raise ValueError("--video-path can only be used with one video id")
        if not path.exists():
            raise FileNotFoundError(f"Missing video: {path}")
        return path
    return find_video(Path(args.video_root).expanduser(), video_id)


def video_info(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise FileNotFoundError(f"Could not open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if width <= 0 or height <= 0 or fps <= 0.0 or frame_count <= 0:
        raise RuntimeError(f"Invalid video metadata for {path}: {width}x{height} fps={fps} frames={frame_count}")
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration": frame_count / fps,
    }


def build_windows(duration: float, args: argparse.Namespace) -> list[tuple[float, float]]:
    if args.start_sec is not None or args.end_sec is not None:
        if args.start_sec is None or args.end_sec is None:
            raise ValueError("--start-sec and --end-sec must be provided together")
        start = max(float(args.start_sec), 0.0)
        end = min(float(args.end_sec), float(duration))
        if end <= start:
            raise ValueError(f"Invalid explicit window: start={start} end={end} duration={duration}")
        return [(start, end)]

    clip_sec = float(args.clip_sec)
    stride_sec = float(args.stride_sec)
    if clip_sec <= 0 or stride_sec <= 0:
        raise ValueError("--clip-sec and --stride-sec must be positive")
    max_start = max(float(duration) - clip_sec, 0.0)
    starts: list[float] = []
    start = 0.0
    while start <= max_start + 1e-6:
        starts.append(start)
        start += stride_sec
    if not starts:
        starts.append(0.0)
    if starts[-1] < max_start - 1e-6:
        starts.append(max_start)
    windows = [(start, min(start + clip_sec, duration)) for start in starts]
    if int(args.max_windows_per_video) > 0:
        windows = windows[: int(args.max_windows_per_video)]
    return windows


def sample_frame_indices(
    frame_count: int,
    fps: float,
    start_sec: float,
    end_sec: float,
    num_frames: int,
) -> list[int]:
    num_frames = max(int(num_frames), 1)
    start_frame = int(np.clip(round(float(start_sec) * fps), 0, frame_count - 1))
    end_frame = int(np.clip(round(float(end_sec) * fps) - 1, start_frame, frame_count - 1))
    if num_frames == 1:
        return [int(round((start_frame + end_frame) * 0.5))]
    if end_frame <= start_frame:
        return [start_frame] * num_frames
    return np.linspace(start_frame, end_frame, num_frames).round().astype(int).tolist()


def read_frame(cap: cv2.VideoCapture, frame_index: int) -> Any | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    return frame if ok else None


def clamp_bbox(bbox: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(int(bbox[0]), width - 1))
    y1 = max(0, min(int(bbox[1]), height - 1))
    x2 = max(x1 + 1, min(int(bbox[2]), width))
    y2 = max(y1 + 1, min(int(bbox[3]), height))
    return x1, y1, x2, y2


def crop_frame(frame: Any, bbox: tuple[int, int, int, int] | None) -> Any:
    if bbox is None:
        return frame
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = clamp_bbox(bbox, width, height)
    return frame[y1:y2, x1:x2]


def draw_bbox(frame: Any, bbox: tuple[int, int, int, int] | None) -> Any:
    image = frame.copy()
    if bbox is None:
        return image
    height, width = image.shape[:2]
    x1, y1, x2, y2 = clamp_bbox(bbox, width, height)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 3)
    return image


def write_image(path: Path, image: Any, jpeg_quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    else:
        cv2.imwrite(str(path), image)


def proposal_row(video_id: str, video_path: Path, window_index: int, start_sec: float, end_sec: float, proposal: Any) -> dict[str, Any]:
    return {
        "video_id": video_id,
        "video_path": str(video_path),
        "window_index": window_index,
        "start_sec": float(start_sec),
        "end_sec": float(end_sec),
        "roi_valid": int(bool(proposal.valid)),
        "roi_mode": proposal.mode,
        "bbox": json.dumps(list(proposal.bbox), ensure_ascii=False) if proposal.bbox is not None else "",
        "roi_confidence": float(proposal.roi_confidence),
        "area_ratio": float(proposal.area_ratio),
        "goal_score": float(proposal.goal_score),
        "ball_score": float(proposal.ball_score),
        "center_circle_score": float(proposal.center_circle_score),
        "person_support": float(proposal.person_support),
        "fallback_reason": proposal.fallback_reason,
        "num_saved_frames": 0,
        "status": "ok" if proposal.valid else proposal.fallback_reason,
    }


def resolve_index_root(args: argparse.Namespace) -> Path:
    if args.roi_index:
        roi_index = Path(args.roi_index).expanduser()
        if roi_index.suffix != ".pt":
            raise ValueError(f"--roi-index must point to a .pt file, got {roi_index}")
        return roi_index.parent
    return Path(args.index_root).expanduser()


def build_cropper(args: argparse.Namespace) -> RobustClipCropper:
    return RobustClipCropper(
        {
            "index_root": str(resolve_index_root(args)),
            "padding": args.padding,
            "min_crop_area_ratio": args.min_area_ratio,
            "max_crop_area_ratio": args.max_area_ratio,
            "min_roi_confidence": args.min_roi_confidence,
            "goal_conf": args.goal_conf,
            "person_conf": args.person_conf,
            "center_circle_conf": args.center_circle_conf,
            "raw_ball_conf": args.raw_ball_conf,
            "min_goal_frames": args.min_goal_frames,
            "min_ball_points": args.min_ball_points,
            "max_people": args.max_people,
            "max_cached_videos": args.max_cached_videos,
        }
    )


def compute_window_roi(
    cropper: RobustClipCropper,
    *,
    video_id: str,
    start_sec: float,
    end_sec: float,
    width: int,
    height: int,
    target_size: tuple[int, int],
) -> Any:
    """Compute one clip-level ROI proposal in original video coordinates."""
    return cropper.get_window_roi(
        video_id,
        start_sec,
        end_sec,
        width,
        height,
        target_size,
    )


def compute_roi_from_index(
    roi_index: str | Path,
    *,
    start_sec: float,
    end_sec: float,
    width: int,
    height: int,
    target_size: tuple[int, int] = (384, 640),
    video_id: str | None = None,
    padding: float = 0.15,
    min_area_ratio: float = 0.0,
    max_area_ratio: float = 0.35,
    min_roi_confidence: float = 0.40,
    goal_conf: float = 0.40,
    person_conf: float = 0.35,
    center_circle_conf: float = 0.35,
    raw_ball_conf: float = 0.10,
    min_goal_frames: int = 3,
    min_ball_points: int = 3,
    max_people: int = 10,
) -> Any:
    """Compute ROI directly from a compact ROI index file or index directory.

    `roi_index` may be either `/path/to/{video_id}.pt` or the directory that
    contains `{video_id}.pt`. When a single `.pt` file is passed, `video_id` is
    inferred from its filename unless explicitly provided.
    """
    roi_index_path = Path(roi_index).expanduser()
    if roi_index_path.suffix == ".pt":
        index_root = roi_index_path.parent
        resolved_video_id = video_id or infer_video_id(roi_index_path)
    else:
        index_root = roi_index_path
        resolved_video_id = video_id or ""
    if not resolved_video_id:
        raise ValueError("video_id is required when roi_index is an index directory")
    cropper = RobustClipCropper(
        {
            "index_root": str(index_root),
            "padding": padding,
            "min_crop_area_ratio": min_area_ratio,
            "max_crop_area_ratio": max_area_ratio,
            "min_roi_confidence": min_roi_confidence,
            "goal_conf": goal_conf,
            "person_conf": person_conf,
            "center_circle_conf": center_circle_conf,
            "raw_ball_conf": raw_ball_conf,
            "min_goal_frames": min_goal_frames,
            "min_ball_points": min_ball_points,
            "max_people": max_people,
            "max_cached_videos": 1,
        }
    )
    return compute_window_roi(
        cropper,
        video_id=resolved_video_id,
        start_sec=start_sec,
        end_sec=end_sec,
        width=width,
        height=height,
        target_size=target_size,
    )


def export_video(
    args: argparse.Namespace,
    cropper: RobustClipCropper,
    video_id: str,
    video_path: Path,
    target_size: tuple[int, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    info = video_info(video_path)
    windows = build_windows(float(info["duration"]), args)
    window_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    target_h, target_w = target_size
    image_ext = args.image_ext.lower().lstrip(".")
    if image_ext not in {"jpg", "jpeg", "png"}:
        raise ValueError("--image-ext must be jpg, jpeg, or png")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise FileNotFoundError(f"Could not open video: {video_path}")
    try:
        for window_index, (start_sec, end_sec) in enumerate(windows):
            proposal = compute_window_roi(
                cropper,
                video_id=video_id,
                start_sec=start_sec,
                end_sec=end_sec,
                width=int(info["width"]),
                height=int(info["height"]),
                target_size=target_size,
            )
            row = proposal_row(video_id, video_path, window_index, start_sec, end_sec, proposal)
            should_save = bool(proposal.valid) or bool(args.save_invalid_full_frame)
            if not should_save:
                window_rows.append(row)
                continue
            frame_indices = sample_frame_indices(
                int(info["frame_count"]),
                float(info["fps"]),
                start_sec,
                end_sec,
                int(args.num_frames),
            )
            saved = 0
            for slot, frame_index in enumerate(frame_indices):
                frame = read_frame(cap, frame_index)
                frame_time = float(frame_index) / max(float(info["fps"]), 1e-6)
                frame_row = {
                    "video_id": video_id,
                    "window_index": window_index,
                    "frame_slot": slot,
                    "frame_index": frame_index,
                    "frame_time_sec": frame_time,
                    "roi_valid": int(bool(proposal.valid)),
                    "roi_mode": proposal.mode,
                    "bbox": row["bbox"],
                    "resized_crop_path": "",
                    "source_crop_path": "",
                    "overlay_path": "",
                    "status": "ok",
                }
                if frame is None:
                    frame_row["status"] = "frame_read_failed"
                    frame_rows.append(frame_row)
                    continue
                bbox = proposal.bbox if proposal.valid else None
                source_crop = crop_frame(frame, bbox)
                resized_crop = cv2.resize(source_crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
                rel_base = Path(video_id) / f"window_{window_index:05d}_{start_sec:010.3f}_{end_sec:010.3f}" / f"frame_{slot:02d}_{frame_time:010.3f}"
                resized_path = Path(args.output_dir) / "resized" / rel_base.with_suffix(f".{image_ext}")
                write_image(resized_path, resized_crop, int(args.jpeg_quality))
                frame_row["resized_crop_path"] = str(resized_path)
                if args.save_source_crop:
                    source_path = Path(args.output_dir) / "source_crop" / rel_base.with_suffix(f".{image_ext}")
                    write_image(source_path, source_crop, int(args.jpeg_quality))
                    frame_row["source_crop_path"] = str(source_path)
                if args.save_overlay:
                    overlay_path = Path(args.output_dir) / "overlay" / rel_base.with_suffix(f".{image_ext}")
                    write_image(overlay_path, draw_bbox(frame, bbox), int(args.jpeg_quality))
                    frame_row["overlay_path"] = str(overlay_path)
                frame_rows.append(frame_row)
                saved += 1
            row["num_saved_frames"] = saved
            row["status"] = "ok" if saved > 0 else "no_frames_saved"
            window_rows.append(row)
    finally:
        cap.release()
    return window_rows, frame_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone robust ROI crop exporter. It does not load event models or training/eval datasets."
    )
    parser.add_argument("--index-root", default="outputs/football_roi_indices/robust_v2", help="Directory containing {video_id}.pt ROI indices.")
    parser.add_argument("--roi-index", default="", help="Direct path to one {video_id}.pt ROI index; overrides --index-root and can infer --video-id.")
    parser.add_argument("--video-root", default="/mnt/data_16t/football/raw_video_720P", help="Directory containing videos named by video_id.")
    parser.add_argument("--video-path", default="", help="Direct path for a single video. video_id is inferred from the filename if not provided.")
    parser.add_argument("--video-id", default="", help="Single video_id or comma-separated ids.")
    parser.add_argument("--video-ids", default="", help="Comma-separated video ids.")
    parser.add_argument("--video-id-file", default="", help="Text file with one video_id per line.")
    parser.add_argument("--output-dir", default="outputs/football_roi_crop_standalone", help="Output directory.")
    parser.add_argument("--start-sec", type=float, default=None, help="Explicit single-window start time. Requires --end-sec.")
    parser.add_argument("--end-sec", type=float, default=None, help="Explicit single-window end time. Requires --start-sec.")
    parser.add_argument("--clip-sec", type=float, default=10.0, help="Sliding-window length when start/end are not provided.")
    parser.add_argument("--stride-sec", type=float, default=10.0, help="Sliding-window stride when start/end are not provided.")
    parser.add_argument("--max-windows-per-video", type=int, default=0, help="Optional cap for quick inspection; 0 means all windows.")
    parser.add_argument("--num-frames", type=int, default=16, help="Uniform frames saved per window, matching model-style sampling.")
    parser.add_argument("--target-size", default="384,640", help="Resized ROI output size as height,width.")
    parser.add_argument("--image-ext", default="jpg", choices=["jpg", "jpeg", "png"])
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--save-invalid-full-frame", action="store_true", help="When ROI is invalid, save the full frame resized to target-size.")
    parser.add_argument("--save-source-crop", action="store_true", help="Also save original-resolution source ROI crops before resize.")
    parser.add_argument("--save-overlay", action="store_true", help="Also save full-frame images with ROI bbox drawn.")
    parser.add_argument("--padding", type=float, default=0.0)
    parser.add_argument("--min-area-ratio", type=float, default=0.0)
    parser.add_argument("--max-area-ratio", type=float, default=0.35)
    parser.add_argument("--min-roi-confidence", type=float, default=0.40)
    parser.add_argument("--goal-conf", type=float, default=0.40)
    parser.add_argument("--person-conf", type=float, default=0.35)
    parser.add_argument("--center-circle-conf", type=float, default=0.35)
    parser.add_argument("--raw-ball-conf", type=float, default=0.10)
    parser.add_argument("--min-goal-frames", type=int, default=3)
    parser.add_argument("--min-ball-points", type=int, default=3)
    parser.add_argument("--max-people", type=int, default=10)
    parser.add_argument("--max-cached-videos", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_size = parse_size(args.target_size)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_ids = collect_video_ids(args)
    cropper = build_cropper(args)

    all_window_rows: list[dict[str, Any]] = []
    all_frame_rows: list[dict[str, Any]] = []
    for index, video_id in enumerate(video_ids, start=1):
        try:
            if not cropper.has_video(video_id):
                all_window_rows.append({
                    "video_id": video_id,
                    "status": "missing_index",
                    "num_saved_frames": 0,
                })
                print(f"[{index}/{len(video_ids)}] skip video_id={video_id} missing_index", flush=True)
                continue
            video_path = resolve_video_path(args, video_id, len(video_ids))
            window_rows, frame_rows = export_video(args, cropper, video_id, video_path, target_size)
            all_window_rows.extend(window_rows)
            all_frame_rows.extend(frame_rows)
            print(
                f"[{index}/{len(video_ids)}] video_id={video_id} windows={len(window_rows)} "
                f"frames={sum(int(row.get('num_saved_frames', 0) or 0) for row in window_rows)}",
                flush=True,
            )
        except Exception as exc:
            all_window_rows.append({
                "video_id": video_id,
                "status": f"error:{type(exc).__name__}",
                "error": str(exc),
                "num_saved_frames": 0,
            })
            print(f"[{index}/{len(video_ids)}] error video_id={video_id}: {exc}", flush=True)
            if len(video_ids) == 1:
                raise

    window_fields = [
        "video_id",
        "video_path",
        "window_index",
        "start_sec",
        "end_sec",
        "roi_valid",
        "roi_mode",
        "bbox",
        "roi_confidence",
        "area_ratio",
        "goal_score",
        "ball_score",
        "center_circle_score",
        "person_support",
        "fallback_reason",
        "num_saved_frames",
        "status",
        "error",
    ]
    frame_fields = [
        "video_id",
        "window_index",
        "frame_slot",
        "frame_index",
        "frame_time_sec",
        "roi_valid",
        "roi_mode",
        "bbox",
        "resized_crop_path",
        "source_crop_path",
        "overlay_path",
        "status",
    ]
    write_csv(output_dir / "windows.csv", all_window_rows, window_fields)
    write_csv(output_dir / "frames.csv", all_frame_rows, frame_fields)
    summary = {
        "index_root": str(resolve_index_root(args)),
        "roi_index": args.roi_index,
        "video_root": args.video_root,
        "video_path": args.video_path,
        "output_dir": str(output_dir),
        "target_size": list(target_size),
        "num_videos": len(video_ids),
        "num_windows": len(all_window_rows),
        "num_frame_rows": len(all_frame_rows),
        "num_saved_resized_crops": sum(1 for row in all_frame_rows if row.get("resized_crop_path")),
        "num_valid_windows": sum(1 for row in all_window_rows if int(row.get("roi_valid", 0) or 0) == 1),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        f"done windows={summary['num_windows']} crops={summary['num_saved_resized_crops']} "
        f"output_dir={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
