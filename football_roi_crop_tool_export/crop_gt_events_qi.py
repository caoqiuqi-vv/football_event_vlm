#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_football_roi_indices import bbox_values, build_index
from football_detection_aware import RobustClipCropper

LABEL_MAP = {
    "射门": "SHOT",
    "其他射门类型": "SHOT",
    "扑救": "SAVE",
    "角球": "CORNER",
    "任意球": "FREE_KICK",
    "点球": "PENALTY",
    "中圈开球": "KICKOFF",
    "进球": "GOAL",
    "shot": "SHOT",
    "save": "SAVE",
    "corner": "CORNER",
    "freekick": "FREE_KICK",
    "free_kick": "FREE_KICK",
    "free kick": "FREE_KICK",
    "penalty": "PENALTY",
    "kickoff": "KICKOFF",
    "kick_off": "KICKOFF",
    "goal": "GOAL",
}
EVENT_TYPE_MAP = {
    "S0199": "SHOT",
    "B0199": "SHOT",
    "S0401": "SAVE",
    "B0401": "SAVE",
    "S0201": "CORNER",
    "B0201": "CORNER",
    "S0202": "FREE_KICK",
    "B0202": "FREE_KICK",
    "S0101": "PENALTY",
    "B0101": "PENALTY",
    "S1004": "KICKOFF",
    "S06": "KICKOFF",
    "B1004": "KICKOFF",
    "B06": "KICKOFF",
}
CLASS_NAME_TO_ID = {
    "person": 0,
    "player": 0,
    "people": 0,
    "ball": 1,
    "football": 1,
    "goal": 2,
    "goalpost": 2,
    "goal_post": 2,
    "center_circle": 3,
    "centercircle": 3,
    "circle": 3,
}


@dataclass(frozen=True)
class GTEvent:
    event_id: str
    label: str
    time_sec: float
    raw_label: str = ""
    raw_event_type: str = ""
    source_start_sec: float | None = None
    source_end_sec: float | None = None


def parse_time_value(value: Any, default: float = -1.0) -> float:
    if value is None or value == "":
        return default
    if isinstance(value, (int, float)):
        number = float(value)
        return number / 1000.0 if number > 100000 else number
    text = str(value).strip()
    if not text:
        return default
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        number = float(text)
        return number / 1000.0 if number > 100000 else number
    parts = text.split(":")
    try:
        values = [float(part) for part in parts]
    except ValueError:
        return default
    if len(values) == 2:
        return values[0] * 60.0 + values[1]
    if len(values) == 3:
        return values[0] * 3600.0 + values[1] * 60.0 + values[2]
    return default


def parse_hw(value: str) -> tuple[int, int]:
    text = str(value).strip().replace("x", ",").replace("X", ",")
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"Expected height,width or heightxwidth, got {value!r}")
    height, width = int(parts[0]), int(parts[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Size must be positive, got {value!r}")
    return height, width


def parse_letterbox(value: str) -> tuple[int, int] | None:
    if not value:
        return None
    text = value.strip().lower().replace(",", "x")
    parts = [part for part in text.split("x") if part]
    if len(parts) != 2:
        raise ValueError(f"Expected --letterbox WIDTHxHEIGHT, got {value!r}")
    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"Letterbox size must be positive, got {value!r}")
    return width, height


def parse_pad_sec(value: str) -> float:
    pad = float(value)
    if pad <= 0:
        raise ValueError(f"--pad-sec must be positive, got {value}")
    return pad


def normalize_label(raw_label: str, raw_event_type: str) -> str:
    label = str(raw_label or "").strip()
    event_type = str(raw_event_type or "").strip()
    mapped = LABEL_MAP.get(label) or LABEL_MAP.get(label.lower()) or EVENT_TYPE_MAP.get(event_type)
    if mapped:
        return mapped
    fallback = label or event_type or "UNKNOWN"
    return re.sub(r"[^A-Za-z0-9]+", "_", fallback.upper()).strip("_") or "UNKNOWN"


def event_from_item(item: dict[str, Any], idx: int, video_id: str) -> GTEvent | None:
    if item.get("label_correct") is False:
        return None
    raw_label = str(item.get("label", item.get("raw_label", item.get("type", item.get("event", "")))) or "")
    raw_event_type = str(item.get("eventType", item.get("event_type", item.get("code", ""))) or "")
    start_sec = parse_time_value(item.get("startTime", item.get("start_sec", item.get("start", None))), default=-1.0)
    end_sec = parse_time_value(item.get("endTime", item.get("end_sec", item.get("end", None))), default=start_sec)
    time_sec = parse_time_value(item.get("time_sec", item.get("timestamp", item.get("time", None))), default=-1.0)
    if time_sec < 0 and start_sec >= 0:
        time_sec = start_sec
    if time_sec < 0:
        return None
    label = normalize_label(raw_label, raw_event_type)
    event_id = str(item.get("event_id", item.get("id", f"{video_id}_{idx:04d}_{time_sec:010.3f}")))
    return GTEvent(
        event_id=event_id,
        label=label,
        time_sec=float(time_sec),
        raw_label=raw_label,
        raw_event_type=raw_event_type,
        source_start_sec=start_sec if start_sec >= 0 else None,
        source_end_sec=end_sec if end_sec >= 0 else None,
    )


def load_gt_json(path: Path, video_id: str) -> list[GTEvent]:
    if path.suffix.lower() == ".csv":
        events: list[GTEvent] = []
        with path.open(newline="", encoding="utf-8") as handle:
            for idx, row in enumerate(csv.DictReader(handle)):
                event = event_from_item(row, idx, video_id)
                if event is not None:
                    events.append(event)
        return sorted(events, key=lambda event: event.time_sec)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        items = raw.get("data", raw.get("events", raw.get("gt_events", [])))
    else:
        items = raw
    if not isinstance(items, list):
        raise ValueError(f"GT file must contain a list or data[] list: {path}")
    events = []
    for idx, item in enumerate(items):
        if isinstance(item, dict):
            event = event_from_item(item, idx, video_id)
            if event is not None:
                events.append(event)
    return sorted(events, key=lambda event: event.time_sec)


def video_info(video_path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise FileNotFoundError(f"Could not open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0:
        raise RuntimeError(f"Invalid video metadata: {video_path} width={width} height={height} fps={fps} frames={frame_count}")
    return {"width": width, "height": height, "fps": fps, "frame_count": frame_count, "duration": frame_count / fps}


def class_id(value: Any) -> int:
    if value is None:
        return -1
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lstrip("-").isdigit():
            return int(stripped)
        return CLASS_NAME_TO_ID.get(stripped.lower(), -1)
    try:
        return int(value)
    except Exception:
        return -1


def item_objects(item: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("objects", "detections", "Detect4in1", "results"):
        value = item.get(key)
        if isinstance(value, list):
            return value
    return []


def load_json_items(path: Path) -> list[Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        for key in ("frames", "data", "items", "results"):
            if isinstance(raw.get(key), list):
                return raw[key]
    return raw if isinstance(raw, list) else []


def normalize_object(obj: dict[str, Any]) -> dict[str, Any] | None:
    bbox = bbox_values(obj.get("bbox", obj.get("box", obj.get("xyxy"))))
    if bbox is None or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    cls = class_id(obj.get("cls", obj.get("class", obj.get("class_id", obj.get("label", obj.get("name"))))))
    if cls < 0:
        return None
    return {
        "cls": cls,
        "conf": float(obj.get("conf", obj.get("score", obj.get("confidence", 1.0))) or 0.0),
        "bbox": [float(value) for value in bbox],
        "trackID": int(obj.get("trackID", obj.get("track_id", obj.get("id", -1))) or -1),
    }


def append_frame_objects(frames: dict[int, list[dict[str, Any]]], path: Path, offset: int) -> int:
    local_max = -1
    for item in load_json_items(path):
        if not isinstance(item, dict):
            continue
        frame_id = int(item.get("frame_id", item.get("frame", item.get("frameIndex", 0))) or 0)
        local_max = max(local_max, frame_id)
        objects = []
        for obj in item_objects(item):
            if isinstance(obj, dict):
                normalized = normalize_object(obj)
                if normalized is not None:
                    objects.append(normalized)
        if objects:
            frames.setdefault(offset + frame_id, []).extend(objects)
    return local_max


def normalize_ball_frame(frame: dict[str, Any], offset: int) -> dict[str, Any] | None:
    bbox = bbox_values(frame.get("bbox", frame.get("box", frame.get("xyxy"))))
    if bbox is None or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    frame_id = int(frame.get("frame_id", frame.get("frame", frame.get("frameIndex", 0))) or 0)
    return {
        "frame_id": offset + frame_id,
        "bbox": [float(value) for value in bbox],
        "TouchedPeople": bool(frame.get("TouchedPeople", frame.get("touched_people", False))),
    }


def append_ball_tracks(tracks: list[dict[str, Any]], path: Path, offset: int) -> None:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        for key in ("tracks", "trajectories", "key_ball_tracks", "data", "results"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
    if not isinstance(raw, list):
        return
    for track in raw:
        if not isinstance(track, dict):
            continue
        frames_raw = track.get("frames", track.get("points", track.get("trajectory", [])))
        if not isinstance(frames_raw, list):
            continue
        frames = []
        for frame in frames_raw:
            if isinstance(frame, dict):
                normalized = normalize_ball_frame(frame, offset)
                if normalized is not None:
                    frames.append(normalized)
        if frames:
            tracks.append({"frames": frames, "touched": bool(track.get("touched", False))})


def standard_detector_dir(detector_root: Path, video_id: str) -> Path | None:
    for candidate in (detector_root / video_id, detector_root):
        if (candidate / "tracked_objects.json").exists() and (candidate / "metadata.json").exists():
            return candidate
    return None


def part_dirs(detector_root: Path, video_id: str) -> list[Path]:
    roots = [detector_root / video_id / "detection_tracking", detector_root / "detection_tracking", detector_root / video_id, detector_root]
    result: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists() or root in seen:
            continue
        seen.add(root)
        candidates = [path for path in root.iterdir() if path.is_dir()]
        if any((root / name).exists() for name in ("tracked_objects.json", "detections.json", "trajectory_results.json", "key_ball_tracks.json")):
            candidates.append(root)
        for candidate in candidates:
            if any((candidate / name).exists() for name in ("tracked_objects.json", "detections.json", "trajectory_results.json", "key_ball_tracks.json")):
                result.append(candidate)
    deduped: list[Path] = []
    seen_parts: set[Path] = set()
    for path in sorted(result):
        if path not in seen_parts:
            seen_parts.add(path)
            deduped.append(path)
    return deduped


def metadata_for_index(parts: Sequence[Path], info: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for part in parts:
        for name in ("metadata.json", "trajectory_results.json"):
            path = part / name
            if path.exists():
                try:
                    candidate = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    candidate = {}
                if isinstance(candidate, dict):
                    metadata = candidate
                    break
        if metadata:
            break
    sampling = metadata.get("sampling", {}) if isinstance(metadata.get("sampling"), dict) else {}
    image_size = metadata.get("image_size", {}) if isinstance(metadata.get("image_size"), dict) else {}
    fps = float(metadata.get("fps", sampling.get("source_fps", info["fps"])) or info["fps"])
    width = int(image_size.get("width", info["width"]) or info["width"])
    height = int(image_size.get("height", info["height"]) or info["height"])
    return {
        "fps": fps,
        "image_size": {"width": width, "height": height},
        "sampling": {"frame_id_semantics": "original_source_frame_id", "source_fps": fps},
    }


def materialize_part_detector(detector_root: Path, video_id: str, work_root: Path, info: dict[str, Any], part_frame_mode: str) -> Path | None:
    parts = part_dirs(detector_root, video_id)
    if not parts:
        return None
    out_dir = work_root / "_converted_detector" / video_id
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: dict[int, list[dict[str, Any]]] = {}
    tracks: list[dict[str, Any]] = []
    offset = 0
    for part in parts:
        local_max = -1
        for name in ("tracked_objects.json", "detections.json"):
            path = part / name
            if path.exists():
                local_max = max(local_max, append_frame_objects(frames, path, 0 if part_frame_mode == "original" else offset))
        for name in ("key_ball_tracks.json", "trajectory_results.json"):
            path = part / name
            if path.exists():
                append_ball_tracks(tracks, path, 0 if part_frame_mode == "original" else offset)
        if part_frame_mode == "offset" and local_max >= 0:
            offset += local_max + 1
    if not frames:
        return None
    (out_dir / "metadata.json").write_text(json.dumps(metadata_for_index(parts, info), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "tracked_objects.json").write_text(
        json.dumps([{"frame_id": frame_id, "objects": objects} for frame_id, objects in sorted(frames.items())], ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "key_ball_tracks.json").write_text(json.dumps(tracks, ensure_ascii=False), encoding="utf-8")
    return out_dir


def roi_index_is_current(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        return isinstance(payload, dict) and int(payload.get("version", 0)) == 2
    except Exception:
        return False


def ensure_roi_index(args: argparse.Namespace, video_id: str, info: dict[str, Any]) -> Path:
    index_root = Path(args.index_root).expanduser()
    index_root.mkdir(parents=True, exist_ok=True)
    output_path = index_root / f"{video_id}.pt"
    if roi_index_is_current(output_path) and not args.rebuild_index:
        return output_path
    if not args.build_missing_index:
        raise FileNotFoundError(f"Missing ROI index: {output_path}")
    detector_root = Path(args.detector_root).expanduser()
    if not detector_root.exists():
        raise FileNotFoundError(f"Missing --detector-root: {detector_root}")
    source_dir = standard_detector_dir(detector_root, video_id)
    if source_dir is None:
        source_dir = materialize_part_detector(detector_root, video_id, Path(args.out_dir).expanduser(), info, args.part_frame_mode)
    if source_dir is None:
        raise FileNotFoundError(f"No supported detector result found for video_id={video_id} under {detector_root}")
    row = build_index(
        source_dir,
        output_path,
        sample_fps=float(args.roi_index_sample_fps),
        ball_track_fps=float(args.roi_index_ball_track_fps),
        person_conf_floor=float(args.roi_index_person_conf_floor),
        ball_conf_floor=float(args.roi_index_ball_conf_floor),
        goal_conf_floor=float(args.roi_index_goal_conf_floor),
        center_circle_conf_floor=float(args.roi_index_center_circle_conf_floor),
    )
    print(f"built ROI index version=2 video_id={video_id} frames={row['sampled_frames']} objects={row['objects']} output={output_path}", flush=True)
    return output_path


def build_cropper(args: argparse.Namespace) -> RobustClipCropper:
    return RobustClipCropper(
        {
            "index_root": str(Path(args.index_root).expanduser()),
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
            "max_cached_videos": 2,
        }
    )


def clamp_bbox(bbox: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(int(bbox[0]), width - 1))
    y1 = max(0, min(int(bbox[1]), height - 1))
    x2 = max(x1 + 1, min(int(bbox[2]), width))
    y2 = max(y1 + 1, min(int(bbox[3]), height))
    return x1, y1, x2, y2


def even(value: int) -> int:
    return max(value - value % 2, 2)


def letterbox_frame(frame: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(out_w / max(width, 1), out_h / max(height, 1))
    resized_w = max(1, int(round(width * scale)))
    resized_h = max(1, int(round(height * scale)))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.zeros((out_h, out_w, 3), dtype=frame.dtype)
    left = (out_w - resized_w) // 2
    top = (out_h - resized_h) // 2
    canvas[top : top + resized_h, left : left + resized_w] = resized
    return canvas


def h264_writer(output_path: Path, width: int, height: int, fps: float, ffmpeg_bin: str, crf: int) -> subprocess.Popen[bytes]:
    ffmpeg = shutil.which(ffmpeg_bin)
    if ffmpeg is None:
        raise FileNotFoundError(f"Could not find ffmpeg binary: {ffmpeg_bin}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(int(crf)),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def write_clip_h264(
    video_path: Path,
    output_path: Path,
    start_sec: float,
    end_sec: float,
    bbox: tuple[int, int, int, int] | None,
    info: dict[str, Any],
    letterbox: tuple[int, int] | None,
    ffmpeg_bin: str,
    crf: int,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = float(info["fps"])
    frame_count = int(info["frame_count"])
    source_w, source_h = int(info["width"]), int(info["height"])
    start_frame = int(max(0, min(round(start_sec * fps), frame_count - 1)))
    end_frame = int(max(start_frame, min(math.ceil(end_sec * fps) - 1, frame_count - 1)))
    applied_bbox = clamp_bbox(bbox, source_w, source_h) if bbox is not None else None
    if letterbox is not None:
        out_w, out_h = letterbox
        out_w, out_h = even(out_w), even(out_h)
    elif applied_bbox is None:
        out_w, out_h = even(source_w), even(source_h)
    else:
        x1, y1, x2, y2 = applied_bbox
        out_w, out_h = even(x2 - x1), even(y2 - y1)
    proc = h264_writer(output_path, out_w, out_h, fps, ffmpeg_bin, crf)
    if proc.stdin is None:
        cap.release()
        raise RuntimeError("ffmpeg stdin is unavailable")
    written = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    try:
        for _ in range(start_frame, end_frame + 1):
            ok, frame = cap.read()
            if not ok:
                break
            if applied_bbox is not None:
                x1, y1, x2, y2 = applied_bbox
                frame = frame[y1:y2, x1:x2]
            if letterbox is not None:
                frame = letterbox_frame(frame, out_w, out_h)
            elif frame.shape[1] != out_w or frame.shape[0] != out_h:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
            proc.stdin.write(frame.tobytes())
            written += 1
    finally:
        cap.release()
        proc.stdin.close()
        return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}: {output_path}")
    return {
        "start_frame": start_frame,
        "end_frame": end_frame,
        "frames_written": written,
        "output_width": out_w,
        "output_height": out_h,
        "applied_bbox": list(applied_bbox) if applied_bbox is not None else None,
        "codec": "libx264",
    }


def safe_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "UNKNOWN"


def time_token(seconds: float) -> str:
    minutes = int(seconds // 60)
    rest = seconds - minutes * 60
    return f"{minutes:02d}m{rest:04.1f}s"


def pad_token(pad_sec: float) -> str:
    return f"{pad_sec:g}".replace(".", "p")


def output_dir_for_pad(out_dir: Path, pad_sec: float) -> Path:
    return out_dir / f"slices_gt_key_event_{pad_token(pad_sec)}s_QIcrop"


def proposal_dict(proposal: Any) -> dict[str, Any]:
    return {
        "bbox": list(proposal.bbox) if proposal.bbox is not None else None,
        "roi_mode": str(proposal.mode),
        "roi_valid": bool(proposal.valid),
        "roi_confidence": float(proposal.roi_confidence),
        "area_ratio": float(proposal.area_ratio),
        "fallback_reason": str(proposal.fallback_reason),
    }


def export_events(args: argparse.Namespace) -> None:
    video_path = Path(args.video_path).expanduser()
    gt_path = Path(args.gt_json).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    video_id = args.video_id.strip() or video_path.stem.split(".")[0]
    pad_sec = parse_pad_sec(args.pad_sec)
    target_size = parse_hw(args.target_size)
    letterbox = parse_letterbox(args.letterbox)
    info = video_info(video_path)
    events = load_gt_json(gt_path, video_id)
    if not events:
        raise ValueError(f"No GT events found in {gt_path}")
    if not args.index_root:
        args.index_root = str(out_dir / "roi_indices")
    ensure_roi_index(args, video_id, info)
    cropper = build_cropper(args)
    slice_dir = output_dir_for_pad(out_dir, pad_sec)
    slice_dir.mkdir(parents=True, exist_ok=True)
    manifest_events = []
    for idx, event in enumerate(events):
        start_sec = max(0.0, event.time_sec - pad_sec)
        end_sec = min(float(info["duration"]), event.time_sec + pad_sec)
        proposal = cropper.get_window_roi(video_id, start_sec, end_sec, int(info["width"]), int(info["height"]), target_size)
        bbox = proposal.bbox if proposal.valid else None
        status = "ok"
        fallback = "none"
        if not proposal.valid:
            if args.full_frame_fallback:
                fallback = "full_frame"
            else:
                status = "invalid_roi_no_output"
        output_name = f"event_{idx:02d}_{safe_token(event.label)}_{time_token(event.time_sec)}_pm{pad_token(pad_sec)}s_QIcrop.mp4"
        clip_info: dict[str, Any] = {"frames_written": 0, "applied_bbox": None, "codec": "libx264"}
        if status == "ok":
            clip_info = write_clip_h264(
                video_path,
                slice_dir / output_name,
                start_sec,
                end_sec,
                bbox,
                info,
                letterbox,
                args.ffmpeg_bin,
                args.crf,
            )
            if int(clip_info["frames_written"]) <= 0:
                status = "no_frames_written"
        manifest_events.append(
            {
                "event_index": idx,
                "event_id": event.event_id,
                "label": event.label,
                "time_sec": event.time_sec,
                "raw_label": event.raw_label,
                "raw_event_type": event.raw_event_type,
                "source_start_sec": event.source_start_sec,
                "source_end_sec": event.source_end_sec,
                "window": {"pad_sec": pad_sec, "start_sec": start_sec, "end_sec": end_sec},
                **proposal_dict(proposal),
                "fallback": fallback,
                "output_name": output_name if status == "ok" else "",
                "output_path": str(slice_dir / output_name) if status == "ok" else "",
                "frames_written": int(clip_info.get("frames_written", 0)),
                "output_width": clip_info.get("output_width"),
                "output_height": clip_info.get("output_height"),
                "codec": "libx264",
                "status": status,
            }
        )
        print(f"[{idx + 1}/{len(events)}] {event.label} {event.time_sec:.3f}s {status}", flush=True)
    manifest = {
        "manifest_version": 1,
        "roi_index_version": 2,
        "video_id": video_id,
        "video_path": str(video_path),
        "gt_json": str(gt_path),
        "detector_root": str(Path(args.detector_root).expanduser()) if args.detector_root else "",
        "index_root": str(Path(args.index_root).expanduser()),
        "pad_sec": pad_sec,
        "target_size_hw": list(target_size),
        "letterbox_wh": list(letterbox) if letterbox is not None else None,
        "full_frame_fallback": bool(args.full_frame_fallback),
        "num_events": len(manifest_events),
        "num_outputs": sum(1 for item in manifest_events if item["status"] == "ok"),
        "events": manifest_events,
    }
    (slice_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {slice_dir / 'manifest.json'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone QI/RobustClipCropper ROI mp4 exporter for GT football events.")
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--gt-json", required=True)
    parser.add_argument("--detector-root", default="", help="Required only when {index_root}/{video_id}.pt is missing.")
    parser.add_argument("--pad-sec", required=True, help="Symmetric window in seconds, e.g. 2 or 5.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--index-root", default="", help="Directory containing/writing {video_id}.pt; defaults to {out_dir}/roi_indices.")
    parser.add_argument("--letterbox", default="", help="Optional fixed output size WIDTHxHEIGHT, e.g. 1280x720.")
    parser.add_argument("--full-frame-fallback", action="store_true", help="Write full-frame clip when ROI is invalid.")
    parser.add_argument("--no-build-missing-index", dest="build_missing_index", action="store_false", default=True)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--part-frame-mode", choices=["offset", "original"], default="offset")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--target-size", default="384,640", help="ROI fitting size height,width; does not force output mp4 size.")
    parser.add_argument("--padding", type=float, default=0.15)
    parser.add_argument("--min-area-ratio", type=float, default=0.15)
    parser.add_argument("--max-area-ratio", type=float, default=0.60)
    parser.add_argument("--min-roi-confidence", type=float, default=0.40)
    parser.add_argument("--goal-conf", type=float, default=0.40)
    parser.add_argument("--person-conf", type=float, default=0.35)
    parser.add_argument("--center-circle-conf", type=float, default=0.35)
    parser.add_argument("--raw-ball-conf", type=float, default=0.10)
    parser.add_argument("--min-goal-frames", type=int, default=3)
    parser.add_argument("--min-ball-points", type=int, default=3)
    parser.add_argument("--max-people", type=int, default=10)
    parser.add_argument("--roi-index-sample-fps", type=float, default=2.0)
    parser.add_argument("--roi-index-ball-track-fps", type=float, default=10.0)
    parser.add_argument("--roi-index-person-conf-floor", type=float, default=0.20)
    parser.add_argument("--roi-index-ball-conf-floor", type=float, default=0.10)
    parser.add_argument("--roi-index-goal-conf-floor", type=float, default=0.20)
    parser.add_argument("--roi-index-center-circle-conf-floor", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    export_events(parse_args())


if __name__ == "__main__":
    main()
