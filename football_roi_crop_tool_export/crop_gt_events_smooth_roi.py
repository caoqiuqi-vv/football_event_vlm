#!/usr/bin/env python3
"""Smooth per-5s-ROI GT event clip exporter.

Randomly picks N videos that have human-verified SHOT/SAVE (射门/扑救)
annotations, detector-based ROI indices, and source mp4 files. For each video
it picks 3 events and writes one ROI-cropped clip per event.

Window logic
------------
- Preferred clip duration is 10s centered on the event timestamp.
- If the event sits near the video start/end so that a symmetric 10s window
  would be clipped, the window is extended on the free side so the event
  context stays complete; total duration never exceeds 20s.

ROI logic
---------
- The clip window is split into 5s segments. One ROI bbox is computed per
  segment with RobustClipCropper (detection/tracking based: goal/ball/player
  anchors).
- Between consecutive segment ROIs a cosine-eased transition of 1.0s on each
  side of the boundary blends the two boxes, so the ROI moves smoothly and
  the change between multiple ROIs is balanced instead of a hard jump.
- Output contains only the ROI region, resized to the event-segment ROI
  resolution, encoded H.264 with ffmpeg.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from football_detection_aware import RobustClipCropper  # noqa: E402

SHOT_SAVE_LABELS = {"射门": "SHOT", "扑救": "SAVE"}


# ---------------------------------------------------------------------------
# annotation helpers
# ---------------------------------------------------------------------------

def parse_time_value(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number / 1000.0 if number > 100000 else number
    text = str(value).strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        number = float(text)
        return number / 1000.0 if number > 100000 else number
    parts = text.split(":")
    try:
        values = [float(part) for part in parts]
    except ValueError:
        return None
    if len(values) == 2:
        return values[0] * 60.0 + values[1]
    if len(values) == 3:
        return values[0] * 3600.0 + values[1] * 60.0 + values[2]
    return None


def load_shot_save_events(ann_path: Path, video_id: str) -> list[dict[str, Any]]:
    raw = json.loads(ann_path.read_text(encoding="utf-8"))
    items = raw if isinstance(raw, list) else raw.get("data", [])
    events: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        if item.get("label_correct") is not True:
            continue
        label = SHOT_SAVE_LABELS.get(str(item.get("label", "")).strip())
        if label is None:
            continue
        time_sec = parse_time_value(item.get("timestamp"))
        if time_sec is None:
            continue
        events.append(
            {
                "index": idx,
                "id": str(item.get("id", f"{video_id}_{idx:04d}")),
                "label": label,
                "time_sec": float(time_sec),
                "raw_label": str(item.get("label", "")),
                "timestamp": str(item.get("timestamp", "")),
            }
        )
    return sorted(events, key=lambda e: e["time_sec"])


# ---------------------------------------------------------------------------
# window / segment helpers
# ---------------------------------------------------------------------------

def event_window(
    time_sec: float, duration: float, prefer_span: float = 10.0, cap_span: float = 20.0
) -> tuple[float, float]:
    """Preferred [t-5, t+5]; extend on the free side when a boundary clips,
    never exceeding cap_span or the video itself. The event stays inside."""
    half = prefer_span / 2.0
    pre = min(half, time_sec)
    post = min(half, duration - time_sec)
    missing = prefer_span - (pre + post)
    if missing > 1e-6:
        add = min(missing, (duration - time_sec) - post, cap_span - (pre + post))
        post += max(add, 0.0)
    missing = prefer_span - (pre + post)
    if missing > 1e-6:
        add = min(missing, time_sec - pre, cap_span - (pre + post))
        pre += max(add, 0.0)
    w0 = max(0.0, time_sec - pre)
    w1 = min(duration, time_sec + post)
    if w1 - w0 > cap_span:
        w1 = w0 + cap_span
    return w0, w1


def segment_windows(w0: float, w1: float, seg_sec: float = 5.0) -> list[tuple[float, float]]:
    segments: list[tuple[float, float]] = []
    cursor = w0
    while cursor < w1 - 1e-6:
        end = min(cursor + seg_sec, w1)
        segments.append((cursor, end))
        cursor = end
    return segments


# ---------------------------------------------------------------------------
# bbox helpers
# ---------------------------------------------------------------------------

BBox = tuple[float, float, float, float]


def clamp_bbox(bbox: BBox, width: int, height: int) -> BBox:
    x1 = max(0.0, min(float(bbox[0]), width - 2))
    y1 = max(0.0, min(float(bbox[1]), height - 2))
    x2 = max(x1 + 2.0, min(float(bbox[2]), float(width)))
    y2 = max(y1 + 2.0, min(float(bbox[3]), float(height)))
    return x1, y1, x2, y2


def even(value: int) -> int:
    return max(value - value % 2, 2)


def smooth_bbox_path(
    segments: Sequence[dict[str, Any]], w0: float, seg_sec: float, trans_sec: float
) -> Callable[[float], BBox]:
    """Piecewise segment bboxes with cosine-eased blending around each
    segment boundary (trans_sec on each side)."""
    boxes = [tuple(seg["bbox"]) for seg in segments]
    count = len(boxes)

    def bbox_at(time_sec: float) -> BBox:
        # Symmetric cosine-eased blend [boundary - trans, boundary + trans]
        # around each segment boundary; frames exactly on the boundary get
        # the midpoint, so the path is continuous everywhere.
        for k in range(count - 1):
            boundary = w0 + (k + 1) * seg_sec
            if boundary - trans_sec <= time_sec <= boundary + trans_sec:
                d = (time_sec - boundary) / max(trans_sec, 1e-6)
                alpha = 0.5 - 0.5 * math.cos(math.pi * (d + 1.0) / 2.0)
                return tuple(
                    (1.0 - alpha) * a + alpha * b for a, b in zip(boxes[k], boxes[k + 1])
                )
        segment_idx = int(math.floor((time_sec - w0) / max(seg_sec, 1e-6)))
        return boxes[min(max(segment_idx, 0), count - 1)]

    return bbox_at


# ---------------------------------------------------------------------------
# proposal helpers
# ---------------------------------------------------------------------------

def build_cropper(index_root: str) -> RobustClipCropper:
    return RobustClipCropper(
        {
            "index_root": index_root,
            "padding": 0.15,
            "min_crop_area_ratio": 0.15,
            "max_crop_area_ratio": 0.60,
            "min_roi_confidence": 0.40,
            "goal_conf": 0.40,
            "person_conf": 0.35,
            "center_circle_conf": 0.35,
            "raw_ball_conf": 0.10,
            "min_goal_frames": 3,
            "min_ball_points": 3,
            "max_people": 10,
            "max_cached_videos": 2,
        }
    )


def segment_proposals(
    cropper: RobustClipCropper,
    video_id: str,
    segments: Sequence[tuple[float, float]],
    width: int,
    height: int,
    target_size: tuple[int, int],
) -> list[dict[str, Any]]:
    proposals: list[dict[str, Any]] = []
    for seg_idx, (seg_start, seg_end) in enumerate(segments):
        proposal = cropper.get_window_roi(video_id, seg_start, seg_end, width, height, target_size)
        proposals.append(
            {
                "seg_idx": seg_idx,
                "start_sec": seg_start,
                "end_sec": seg_end,
                "bbox": tuple(proposal.bbox) if proposal.valid else None,
                "valid": bool(proposal.valid),
                "mode": str(proposal.mode),
                "roi_confidence": float(proposal.roi_confidence),
                "fallback_reason": str(proposal.fallback_reason),
                "filled_from": None,
            }
        )
    return proposals


def fill_invalid_segments(
    proposals: list[dict[str, Any]],
    cropper: RobustClipCropper,
    video_id: str,
    w0: float,
    w1: float,
    width: int,
    height: int,
    target_size: tuple[int, int],
) -> list[dict[str, Any]] | None:
    """Replace invalid segment bboxes: first a whole-window ROI, then the
    nearest valid segment ROI. Returns None when nothing is available."""
    if all(proposal["valid"] for proposal in proposals):
        return proposals
    valid_boxes = [tuple(p["bbox"]) for p in proposals if p["valid"]]
    if not valid_boxes:
        whole = cropper.get_window_roi(video_id, w0, w1, width, height, target_size)
        if whole.valid:
            valid_boxes = [tuple(whole.bbox)]
    if not valid_boxes:
        return None
    for idx, proposal in enumerate(proposals):
        if proposal["valid"]:
            continue
        valid_idx = [i for i, p in enumerate(proposals) if p["valid"]]
        if valid_idx:
            src = min(valid_idx, key=lambda i: abs(i - idx))
            proposal["bbox"] = tuple(proposals[src]["bbox"])
            proposal["filled_from"] = src
            proposal["fallback_reason"] = f"invalid_roi_filled_from_seg_{src}"
        else:
            proposal["bbox"] = valid_boxes[0]
            proposal["filled_from"] = -1
            proposal["fallback_reason"] = "invalid_roi_filled_from_whole_window"
    return proposals


# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------

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


def write_clip(
    video_path: Path,
    output_path: Path,
    w0: float,
    w1: float,
    bbox_at: Callable[[float], BBox],
    width: int,
    height: int,
    fps: float,
    frame_count: int,
    out_w: int,
    out_h: int,
    ffmpeg_bin: str,
    crf: int,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise FileNotFoundError(f"Could not open video: {video_path}")
    start_frame = max(0, min(int(round(w0 * fps)), frame_count - 1))
    end_frame = max(start_frame, min(int(math.ceil(w1 * fps)) - 1, frame_count - 1))
    proc = h264_writer(output_path, out_w, out_h, fps, ffmpeg_bin, crf)
    if proc.stdin is None:
        cap.release()
        raise RuntimeError("ffmpeg stdin is unavailable")
    written = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    try:
        for frame_idx in range(start_frame, end_frame + 1):
            ok, frame = cap.read()
            if not ok:
                break
            time_sec = frame_idx / max(fps, 1e-6)
            bbox = clamp_bbox(bbox_at(time_sec), width, height)
            x1, y1, x2, y2 = (int(round(v)) for v in bbox)
            crop = frame[y1:y2, x1:x2]
            if crop.shape[1] != out_w or crop.shape[0] != out_h:
                crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
            proc.stdin.write(crop.tobytes())
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
        "codec": "libx264",
    }


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------

def select_events(
    events: list[dict[str, Any]],
    count: int,
    rng: random.Random,
    min_gap_sec: float,
    validity: dict[str, bool] | None = None,
) -> list[dict[str, Any]]:
    """Pick `count` events, avoiding near-duplicates. Prefers SHOT+SAVE mix
    (2 shots + 1 save when available)."""
    usable = list(events)
    if validity is not None:
        valid_events = [e for e in usable if validity.get(e["id"], False)]
        if len(valid_events) >= count:
            usable = valid_events
    shots = [e for e in usable if e["label"] == "SHOT"]
    saves = [e for e in usable if e["label"] == "SAVE"]
    if len(shots) >= 2 and saves:
        picked = rng.sample(shots, 2) + rng.sample(saves, 1)
    elif len(shots) >= count and not saves:
        picked = rng.sample(shots, count)
    elif len(saves) >= count and not shots:
        picked = rng.sample(saves, count)
    elif len(usable) >= count:
        picked = rng.sample(usable, count)
    else:
        picked = list(usable)
    picked = sorted(picked, key=lambda e: e["time_sec"])
    # enforce minimum gap, retry once with the full pool when possible
    too_close = any(
        picked[i + 1]["time_sec"] - picked[i]["time_sec"] < min_gap_sec for i in range(len(picked) - 1)
    )
    if too_close and len(usable) > count:
        for _ in range(10):
            picked = rng.sample(usable, count)
            picked = sorted(picked, key=lambda e: e["time_sec"])
            if all(picked[i + 1]["time_sec"] - picked[i]["time_sec"] >= min_gap_sec for i in range(len(picked) - 1)):
                break
    return picked


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smooth per-5s-ROI GT shot/save event clip exporter.")
    parser.add_argument("--video-root", default="/mnt/data_16t/football/raw_video_720P")
    parser.add_argument("--annotation-root", default="/home/new_users/qiuqi/code/football_events_human_repair")
    parser.add_argument("--index-root", default="/mnt/data_16t/football/football_roi_indices/robust_v2")
    parser.add_argument("--out-dir", default="/mnt/data_16t/football/eval_outputs/gt_event_smooth_roi_clips")
    parser.add_argument("--num-videos", type=int, default=6)
    parser.add_argument("--events-per-video", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--prefer-span", type=float, default=10.0)
    parser.add_argument("--max-span", type=float, default=20.0)
    parser.add_argument("--seg-sec", type=float, default=5.0)
    parser.add_argument("--transition-sec", type=float, default=1.0)
    parser.add_argument("--min-gap-sec", type=float, default=15.0)
    parser.add_argument("--target-size", default="384,640")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--dry-run", action="store_true", help="Select and compute ROIs without encoding mp4.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_root = Path(args.video_root)
    annotation_root = Path(args.annotation_root)
    index_root = Path(args.index_root)
    out_dir = Path(args.out_dir)
    target_h, target_w = (int(part) for part in args.target_size.replace("x", ",").split(","))
    target_size = (target_h, target_w)
    rng = random.Random(args.seed)

    # 1. candidate pool: video + annotation (>=3 verified shot/save) + ROI index
    pool: list[str] = []
    for ann_path in sorted(annotation_root.glob("*.json")):
        video_id = ann_path.stem
        if not (index_root / f"{video_id}.pt").exists():
            continue
        if not (video_root / f"{video_id}.mp4").exists():
            continue
        events = load_shot_save_events(ann_path, video_id)
        if len(events) >= args.events_per_video:
            pool.append(video_id)
    print(f"candidate pool: {len(pool)} videos (video + >=3 verified shot/save + ROI index)", flush=True)
    if len(pool) < args.num_videos:
        raise SystemExit(f"Only {len(pool)} videos available, need {args.num_videos}")

    chosen = rng.sample(pool, args.num_videos)
    cropper = build_cropper(str(index_root))
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "manifest_version": 1,
        "seed": args.seed,
        "num_videos": args.num_videos,
        "events_per_video": args.events_per_video,
        "prefer_span": args.prefer_span,
        "max_span": args.max_span,
        "seg_sec": args.seg_sec,
        "transition_sec": args.transition_sec,
        "target_size_hw": list(target_size),
        "index_root": str(index_root),
        "videos": [],
    }

    for video_id in chosen:
        video_path = video_root / f"{video_id}.mp4"
        ann_path = annotation_root / f"{video_id}.json"
        events = load_shot_save_events(ann_path, video_id)
        cap = cv2.VideoCapture(str(video_path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        duration = frame_count / max(fps, 1e-6)
        if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0:
            raise RuntimeError(f"Invalid video metadata: {video_path}")
        print(f"[video] {video_id} {width}x{height}@{fps:.0f}fps {duration:.0f}s events={len(events)}", flush=True)

        # 2. compute per-event windows + segment proposals
        candidates: list[dict[str, Any]] = []
        for event in events:
            w0, w1 = event_window(event["time_sec"], duration, args.prefer_span, args.max_span)
            segments = segment_windows(w0, w1, args.seg_sec)
            proposals = segment_proposals(cropper, video_id, segments, width, height, target_size)
            filled = fill_invalid_segments(proposals, cropper, video_id, w0, w1, width, height, target_size)
            candidates.append(
                {
                    "event": event,
                    "window": (w0, w1),
                    "segments": segments,
                    "proposals": proposals,
                    "all_valid": all(p["valid"] for p in proposals),
                    "any_valid": any(p["valid"] for p in proposals),
                    "filled": filled,
                }
            )
        validity = {c["event"]["id"]: c["all_valid"] for c in candidates}
        picked = select_events(events, args.events_per_video, rng, args.min_gap_sec, validity)
        picked_ids = {e["id"] for e in picked}
        video_out = out_dir / video_id
        video_out.mkdir(parents=True, exist_ok=True)
        video_entry: dict[str, Any] = {
            "video_id": video_id,
            "video_path": str(video_path),
            "annotation_path": str(ann_path),
            "width": width,
            "height": height,
            "fps": fps,
            "duration": duration,
            "events": [],
        }

        # 3. write one clip per picked event
        for event_idx, event in enumerate(picked):
            candidate = next(c for c in candidates if c["event"]["id"] == event["id"])
            w0, w1 = candidate["window"]
            proposals = candidate["filled"] if candidate["filled"] is not None else candidate["proposals"]
            if proposals is None:
                status = "no_roi_anywhere"
                video_entry["events"].append(
                    {"event": event, "window": [w0, w1], "status": status, "output_path": ""}
                )
                print(f"  [{event_idx + 1}/{len(picked)}] {event['label']} {event['time_sec']:.2f}s -> {status}", flush=True)
                continue
            if not all(p["bbox"] is not None for p in proposals):
                status = "no_roi_anywhere"
                video_entry["events"].append(
                    {"event": event, "window": [w0, w1], "status": status, "output_path": ""}
                )
                print(f"  [{event_idx + 1}/{len(picked)}] {event['label']} {event['time_sec']:.2f}s -> {status}", flush=True)
                continue

            # output resolution = the ROI scale of the segment containing the event
            event_seg_idx = 0
            for seg_idx, (seg_start, seg_end) in enumerate(candidate["segments"]):
                if seg_start - 1e-6 <= event["time_sec"] <= seg_end + 1e-6:
                    event_seg_idx = seg_idx
                    break
            eb = proposals[event_seg_idx]["bbox"]
            out_w, out_h = even(int(round(eb[2] - eb[0]))), even(int(round(eb[3] - eb[1])))
            bbox_at = smooth_bbox_path(proposals, w0, args.seg_sec, args.transition_sec)
            time_token = f"{int(event['time_sec'] // 60):02d}m{event['time_sec'] % 60:04.1f}s"
            output_name = f"event_{event_idx:02d}_{event['label']}_{time_token}_{w1 - w0:g}s_roi5s.mp4"
            output_path = video_out / output_name
            span = w1 - w0
            seg_note = "+".join(
                f"{p['seg_idx']}({'filled' if p['filled_from'] is not None else p['mode']})" for p in proposals
            )
            print(
                f"  [{event_idx + 1}/{len(picked)}] {event['label']} t={event['time_sec']:.2f}s "
                f"window=[{w0:.2f},{w1:.2f}] span={span:.2f}s segs={len(proposals)} roi={seg_note}",
                flush=True,
            )
            clip_info: dict[str, Any] = {"frames_written": 0}
            status = "ok"
            if not args.dry_run:
                clip_info = write_clip(
                    video_path,
                    output_path,
                    w0,
                    w1,
                    bbox_at,
                    width,
                    height,
                    fps,
                    frame_count,
                    out_w,
                    out_h,
                    args.ffmpeg_bin,
                    args.crf,
                )
                if int(clip_info["frames_written"]) <= 0:
                    status = "no_frames_written"
            per_second_bboxes = []
            cursor = w0
            while cursor <= w1 + 1e-6:
                bbox = clamp_bbox(bbox_at(cursor), width, height)
                per_second_bboxes.append({"t": round(cursor, 2), "bbox": [round(v, 1) for v in bbox]})
                cursor += 1.0
            video_entry["events"].append(
                {
                    "event_index": event_idx,
                    "event_id": event["id"],
                    "label": event["label"],
                    "time_sec": event["time_sec"],
                    "timestamp": event["timestamp"],
                    "window": {"start_sec": w0, "end_sec": w1, "span": span},
                    "segments": [
                        {
                            "seg_idx": p["seg_idx"],
                            "start_sec": p["start_sec"],
                            "end_sec": p["end_sec"],
                            "bbox": [round(v, 1) for v in p["bbox"]] if p["bbox"] else None,
                            "valid": p["valid"],
                            "mode": p["mode"],
                            "roi_confidence": p["roi_confidence"],
                            "filled_from": p["filled_from"],
                            "fallback_reason": p["fallback_reason"],
                        }
                        for p in proposals
                    ],
                    "transition_sec": args.transition_sec,
                    "bbox_per_second": per_second_bboxes,
                    "output_path": str(output_path) if status == "ok" else "",
                    "output_size_wh": [out_w, out_h] if status == "ok" else None,
                    "frames_written": int(clip_info.get("frames_written", 0)),
                    "status": status,
                }
            )
        summary["videos"].append(video_entry)
        (video_out / "manifest.json").write_text(
            json.dumps(video_entry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    total_clips = sum(
        1 for v in summary["videos"] for e in v["events"] if e["status"] == "ok"
    )
    print(f"done: {total_clips} clips -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
