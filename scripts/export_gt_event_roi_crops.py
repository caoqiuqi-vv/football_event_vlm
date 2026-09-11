#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import cv2

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_football_events as train_mod  # noqa: E402
from evaluate_football_model import DEFAULT_GT_DIR, DEFAULT_VIDEO_ROOTS, find_video, parse_video_roots  # noqa: E402
from eval_long_video_checkpoint import build_windows  # noqa: E402
from football_roi_crop import (  # noqa: E402
    build_cropper,
    clamp_bbox,
    crop_frame,
    draw_bbox,
    parse_size,
    read_frame,
    sample_frame_indices,
    video_info,
    write_csv,
    write_image,
)


TARGET_LABELS = ("save", "set_piece")


def read_video_ids(raw: str, path: str) -> list[str]:
    ids: list[str] = []
    if raw.strip():
        ids.extend(item.strip() for item in raw.split(",") if item.strip())
    if path:
        for line in Path(path).expanduser().read_text().splitlines():
            item = line.strip()
            if item and not item.startswith("#"):
                ids.append(item)
    seen: set[str] = set()
    result: list[str] = []
    for video_id in ids:
        if video_id in seen:
            continue
        seen.add(video_id)
        result.append(video_id)
    return result


def event_output_labels(event: Any) -> list[str]:
    labels: list[str] = []
    for index, value in enumerate(event.labels):
        if float(value) > 0:
            labels.append(train_mod.LABELS[index])
    return labels


def load_target_events(annotation_path: Path, source: str, video_id: str, labels: set[str]) -> list[dict[str, Any]]:
    events = train_mod.load_annotation_events(annotation_path, source, video_id)
    rows: list[dict[str, Any]] = []
    for event_index, event in enumerate(events):
        event_labels = [label for label in event_output_labels(event) if label in labels]
        if not event_labels:
            continue
        for label in event_labels:
            rows.append(
                {
                    "event_index": event_index,
                    "event_id": event.event_id,
                    "label": label,
                    "time_sec": float(event.anchor_time),
                    "raw_label": event.raw_label,
                }
            )
    return sorted(rows, key=lambda row: (row["time_sec"], row["label"], row["event_id"]))


def select_event_windows(
    duration: float,
    event_time: float,
    clip_sec: float,
    stride_sec: float,
    mode: str,
) -> list[Any]:
    windows = build_windows(duration, clip_sec, stride_sec, include_tail=True)
    covering = [window for window in windows if window.start_sec - 1e-6 <= event_time <= window.end_sec + 1e-6]
    if mode == "covering":
        return covering
    if mode == "nearest":
        candidates = covering or windows
        return [min(candidates, key=lambda window: abs(((window.start_sec + window.end_sec) * 0.5) - event_time))]
    raise ValueError("--window-mode must be covering or nearest")


def safe_time(value: float) -> str:
    return f"{value:010.3f}".replace(".", "p")


def save_event_window(
    *,
    args: argparse.Namespace,
    cropper: Any,
    video_id: str,
    video_path: Path,
    source: str,
    event: dict[str, Any],
    window: Any,
    info: dict[str, Any],
    target_size: tuple[int, int],
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    proposal = cropper.get_window_roi(
        video_id,
        window.start_sec,
        window.end_sec,
        int(info["width"]),
        int(info["height"]),
        target_size,
    )
    bbox = proposal.bbox if proposal.valid else None
    target_h, target_w = target_size
    image_ext = args.image_ext.lower().lstrip(".")
    event_dir = (
        output_dir
        / video_id
        / event["label"]
        / f"event_{event['event_index']:04d}_{safe_time(event['time_sec'])}"
        / f"window_{int(window.index):05d}_{safe_time(window.start_sec)}_{safe_time(window.end_sec)}"
    )
    frame_indices = sample_frame_indices(
        int(info["frame_count"]),
        float(info["fps"]),
        float(window.start_sec),
        float(window.end_sec),
        int(args.num_frames),
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise FileNotFoundError(f"Could not open video: {video_path}")

    frame_rows: list[dict[str, Any]] = []
    saved = 0
    try:
        for slot, frame_index in enumerate(frame_indices):
            frame = read_frame(cap, frame_index)
            frame_time = float(frame_index) / max(float(info["fps"]), 1e-6)
            frame_row = {
                "source": source,
                "video_id": video_id,
                "event_index": event["event_index"],
                "event_id": event["event_id"],
                "label": event["label"],
                "event_time_sec": event["time_sec"],
                "window_index": int(window.index),
                "window_start_sec": float(window.start_sec),
                "window_end_sec": float(window.end_sec),
                "frame_slot": slot,
                "frame_index": int(frame_index),
                "frame_time_sec": frame_time,
                "frame_offset_sec": frame_time - float(event["time_sec"]),
                "roi_valid": int(bool(proposal.valid)),
                "roi_mode": proposal.mode,
                "bbox": json.dumps(list(proposal.bbox), ensure_ascii=False) if proposal.bbox is not None else "",
                "bbox_clamped": "",
                "roi_confidence": float(proposal.roi_confidence),
                "area_ratio": float(proposal.area_ratio),
                "goal_score": float(proposal.goal_score),
                "ball_score": float(proposal.ball_score),
                "center_circle_score": float(proposal.center_circle_score),
                "person_support": float(proposal.person_support),
                "fallback_reason": proposal.fallback_reason,
                "resized_crop_path": "",
                "source_crop_path": "",
                "overlay_path": "",
                "status": "ok",
            }
            if frame is None:
                frame_row["status"] = "frame_read_failed"
                frame_rows.append(frame_row)
                continue
            if bbox is not None:
                frame_row["bbox_clamped"] = json.dumps(list(clamp_bbox(bbox, int(info["width"]), int(info["height"]))), ensure_ascii=False)
            source_crop = crop_frame(frame, bbox)
            resized_crop = cv2.resize(source_crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            frame_stem = f"frame_{slot:02d}_{safe_time(frame_time)}_offset_{frame_time - float(event['time_sec']):+.3f}".replace(".", "p")
            resized_path = event_dir / "resized_roi" / f"{frame_stem}.{image_ext}"
            write_image(resized_path, resized_crop, int(args.jpeg_quality))
            frame_row["resized_crop_path"] = str(resized_path)
            if args.save_source_crop:
                source_path = event_dir / "source_roi" / f"{frame_stem}.{image_ext}"
                write_image(source_path, source_crop, int(args.jpeg_quality))
                frame_row["source_crop_path"] = str(source_path)
            if args.save_overlay:
                overlay_path = event_dir / "overlay" / f"{frame_stem}.{image_ext}"
                write_image(overlay_path, draw_bbox(frame, bbox), int(args.jpeg_quality))
                frame_row["overlay_path"] = str(overlay_path)
            frame_rows.append(frame_row)
            saved += 1
    finally:
        cap.release()

    window_row = {
        "source": source,
        "video_id": video_id,
        "video_path": str(video_path),
        "event_index": event["event_index"],
        "event_id": event["event_id"],
        "label": event["label"],
        "raw_label": event["raw_label"],
        "event_time_sec": event["time_sec"],
        "window_index": int(window.index),
        "window_start_sec": float(window.start_sec),
        "window_end_sec": float(window.end_sec),
        "event_offset_from_window_start_sec": float(event["time_sec"]) - float(window.start_sec),
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
        "num_saved_frames": saved,
        "output_dir": str(event_dir),
        "status": "ok" if saved > 0 else "no_frames_saved",
    }
    return window_row, frame_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export model-input ROI crops around GT save/set_piece events.")
    parser.add_argument("--index-root", default="outputs/football_roi_indices/robust_v2")
    parser.add_argument("--roi-index", default="")
    parser.add_argument("--video-ids", default="")
    parser.add_argument("--video-id-file", default="")
    parser.add_argument("--gt-dir", default=DEFAULT_GT_DIR)
    parser.add_argument("--video-root", action="append", default=[], help="source=/path/to/videos; defaults match evaluate_football_model.py")
    parser.add_argument("--output-dir", default="outputs/gt_event_roi_crops/save_set_piece")
    parser.add_argument("--labels", default="save,set_piece")
    parser.add_argument("--clip-sec", type=float, default=10.0)
    parser.add_argument("--stride-sec", type=float, default=5.0)
    parser.add_argument("--window-mode", default="covering", choices=["covering", "nearest"])
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--target-size", default="384,640")
    parser.add_argument("--image-ext", default="jpg", choices=["jpg", "jpeg", "png"])
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--max-events-per-label", type=int, default=0)
    parser.add_argument("--save-source-crop", action="store_true")
    parser.add_argument("--save-overlay", action="store_true", default=True)
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
    train_mod.configure_label_schema(train_mod.ConfigDict({"task": {"label_schema": "set_piece"}}))
    video_ids = read_video_ids(args.video_ids, args.video_id_file)
    if not video_ids:
        raise ValueError("Provide --video-ids or --video-id-file")
    labels = {item.strip() for item in args.labels.split(",") if item.strip()}
    unsupported = labels - set(TARGET_LABELS)
    if unsupported:
        raise ValueError(f"This debug exporter only supports {TARGET_LABELS}, got unsupported={sorted(unsupported)}")
    target_size = parse_size(args.target_size)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cropper = build_cropper(args)
    video_roots = parse_video_roots(args.video_root or DEFAULT_VIDEO_ROOTS)
    gt_dir = Path(args.gt_dir).expanduser()

    window_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for video_order, video_id in enumerate(video_ids, start=1):
        if not cropper.has_video(video_id):
            window_rows.append({"video_id": video_id, "status": "missing_roi_index"})
            print(f"[{video_order}/{len(video_ids)}] skip video_id={video_id} missing_roi_index", flush=True)
            continue
        source, video_path = find_video(video_id, video_roots)
        annotation_path = gt_dir / f"{video_id}.json"
        if not annotation_path.exists():
            window_rows.append({"video_id": video_id, "status": "missing_annotation", "annotation_path": str(annotation_path)})
            print(f"[{video_order}/{len(video_ids)}] skip video_id={video_id} missing_annotation", flush=True)
            continue
        info = video_info(video_path)
        events = load_target_events(annotation_path, source, video_id, labels)
        if args.max_events_per_label > 0:
            kept: list[dict[str, Any]] = []
            counts = {label: 0 for label in labels}
            for event in events:
                if counts[event["label"]] >= int(args.max_events_per_label):
                    continue
                counts[event["label"]] += 1
                kept.append(event)
            events = kept
        written_windows = 0
        for event in events:
            windows = select_event_windows(
                float(info["duration"]),
                float(event["time_sec"]),
                float(args.clip_sec),
                float(args.stride_sec),
                args.window_mode,
            )
            for window in windows:
                window_row, rows = save_event_window(
                    args=args,
                    cropper=cropper,
                    video_id=video_id,
                    video_path=video_path,
                    source=source,
                    event=event,
                    window=window,
                    info=info,
                    target_size=target_size,
                    output_dir=output_dir,
                )
                window_rows.append(window_row)
                frame_rows.extend(rows)
                written_windows += 1
        print(
            f"[{video_order}/{len(video_ids)}] video_id={video_id} events={len(events)} windows={written_windows} "
            f"frames={sum(int(row.get('num_saved_frames', 0) or 0) for row in window_rows if row.get('video_id') == video_id)}",
            flush=True,
        )

    if window_rows:
        write_csv(output_dir / "event_windows_manifest.csv", window_rows, list(window_rows[0]))
    if frame_rows:
        write_csv(output_dir / "event_frames_manifest.csv", frame_rows, list(frame_rows[0]))
    summary = {
        "video_ids": video_ids,
        "labels": sorted(labels),
        "clip_sec": args.clip_sec,
        "stride_sec": args.stride_sec,
        "window_mode": args.window_mode,
        "num_windows": len([row for row in window_rows if row.get("status") == "ok"]),
        "num_saved_frames": len([row for row in frame_rows if row.get("status") == "ok"]),
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {output_dir / 'event_windows_manifest.csv'}")
    print(f"wrote {output_dir / 'event_frames_manifest.csv'}")
    print(f"wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
