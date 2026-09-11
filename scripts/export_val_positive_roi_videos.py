#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_football_events as train_mod
from football_detection_aware import ROIProposal, RobustClipCropper, invalid_roi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export val positive records as review videos. Each video shows the "
            "full-frame ROI overlay beside the exact resized ROI crop."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/football/dinov3_vitl16_robust_dual_16f_exp4.yaml",
    )
    parser.add_argument("--set", action="append", default=[], help="Config override, for example video.image_size=[384,640].")
    parser.add_argument(
        "--output-dir",
        default="outputs/football_roi_debug/val_positive",
    )
    parser.add_argument("--video-ids", default="", help="Optional comma-separated video ids.")
    parser.add_argument("--labels", default="", help="Optional comma-separated labels, for example save,set_piece.")
    parser.add_argument("--max-samples", type=int, default=0, help="Global cap; 0 exports all selected records.")
    parser.add_argument("--max-per-video", type=int, default=0, help="Per-video cap; 0 exports all selected records.")
    parser.add_argument("--output-fps", type=float, default=2.0)
    parser.add_argument(
        "--frame-mode",
        choices=("model", "continuous"),
        default="model",
        help="model exports exact configured sampled frames; continuous samples at --output-fps.",
    )
    parser.add_argument(
        "--codec",
        default="h264",
        help="Use h264 for FFmpeg/libx264 or a four-character OpenCV FourCC.",
    )
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument(
        "--layout",
        choices=("review", "crop", "both"),
        default="review",
        help="review: overlay and crop side by side; crop: ROI only; both: write both videos.",
    )
    parser.add_argument(
        "--save-invalid-full-frame",
        action="store_true",
        help="Also write crop-only videos for invalid ROI records by resizing the full frame.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build the val positive manifest without decoding videos.")
    return parser.parse_args()


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def safe_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return result or "sample"


def clamp_bbox(
    bbox: Sequence[int | float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1 = max(0, min(int(round(float(bbox[0]))), width - 1))
    y1 = max(0, min(int(round(float(bbox[1]))), height - 1))
    x2 = max(x1 + 1, min(int(round(float(bbox[2]))), width))
    y2 = max(y1 + 1, min(int(round(float(bbox[3]))), height))
    return x1, y1, x2, y2


def resize_with_letterbox(frame: Any, target_size: tuple[int, int]) -> Any:
    target_h, target_w = target_size
    height, width = frame.shape[:2]
    scale = min(target_w / max(width, 1), target_h / max(height, 1))
    resized_w = max(1, int(round(width * scale)))
    resized_h = max(1, int(round(height * scale)))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    left = (target_w - resized_w) // 2
    top = (target_h - resized_h) // 2
    canvas[top : top + resized_h, left : left + resized_w] = resized
    return canvas


def crop_and_resize(
    frame: Any,
    proposal: ROIProposal,
    target_size: tuple[int, int],
) -> Any:
    target_h, target_w = target_size
    if proposal.valid and proposal.bbox is not None:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = clamp_bbox(proposal.bbox, width, height)
        frame = frame[y1:y2, x1:x2]
    return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_CUBIC)


def draw_review_overlay(
    frame: Any,
    proposal: ROIProposal,
    target_size: tuple[int, int],
    lines: Sequence[str],
) -> Any:
    image = frame.copy()
    height, width = image.shape[:2]
    if proposal.valid and proposal.bbox is not None:
        x1, y1, x2, y2 = clamp_bbox(proposal.bbox, width, height)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), max(2, width // 640))
    else:
        cv2.putText(
            image,
            "INVALID ROI",
            (24, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )
    image = resize_with_letterbox(image, target_size)
    font_scale = max(0.42, target_size[0] / 900.0)
    y = 20
    for line in lines:
        cv2.putText(
            image,
            str(line),
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            str(line),
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += max(17, int(round(22 * font_scale / 0.42)))
    return image


def video_geometry(path: str) -> tuple[int, int, float, int]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        raise FileNotFoundError(f"Could not open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0:
        raise RuntimeError(f"Invalid video metadata: {path} {width}x{height} fps={fps} frames={frame_count}")
    return width, height, fps, frame_count


class FfmpegVideoWriter:
    def __init__(
        self,
        path: Path,
        size: tuple[int, int],
        fps: float,
        ffmpeg_bin: str,
    ) -> None:
        height, width = size
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.height = int(height)
        self.width = int(width)
        self.process = subprocess.Popen(
            [
                ffmpeg_bin,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s:v",
                f"{self.width}x{self.height}",
                "-r",
                str(float(fps)),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"Frame size {frame.shape[1]}x{frame.shape[0]} does not match "
                f"writer size {self.width}x{self.height}"
            )
        if self.process.stdin is None:
            raise RuntimeError(f"FFmpeg stdin is closed for {self.path}")
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def release(self) -> None:
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        stderr = (
            self.process.stderr.read().decode("utf-8", errors="replace")
            if self.process.stderr is not None
            else ""
        )
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"FFmpeg failed for {self.path} with code={return_code}: {stderr.strip()}"
            )


def open_writer(
    path: Path,
    size: tuple[int, int],
    fps: float,
    codec: str,
    ffmpeg_bin: str,
) -> Any:
    if codec.lower() in {"h264", "libx264"}:
        return FfmpegVideoWriter(path, size, fps, ffmpeg_bin)
    if len(codec) != 4:
        raise ValueError(
            f"--codec must be h264/libx264 or a four-character FourCC, got {codec!r}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = size
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*codec),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Could not open VideoWriter for {path} with codec={codec}")
    return writer


def positive_labels(record: train_mod.LongVideoRecord) -> list[str]:
    return [
        label
        for label, value in zip(train_mod.LABELS, record.labels)
        if float(value) > 0.0
    ]


def select_records(
    records: Sequence[train_mod.LongVideoRecord],
    *,
    video_ids: set[str],
    labels: set[str],
    max_samples: int,
    max_per_video: int,
) -> list[train_mod.LongVideoRecord]:
    selected: list[train_mod.LongVideoRecord] = []
    counts: dict[str, int] = {}
    positives = sorted(
        (record for record in records if not record.is_negative),
        key=lambda record: (record.video_id, record.anchor_time, record.sample_id),
    )
    for record in positives:
        if video_ids and record.video_id not in video_ids:
            continue
        record_labels = set(positive_labels(record))
        if labels and not record_labels.intersection(labels):
            continue
        if max_per_video > 0 and counts.get(record.video_id, 0) >= max_per_video:
            continue
        selected.append(record)
        counts[record.video_id] = counts.get(record.video_id, 0) + 1
        if max_samples > 0 and len(selected) >= max_samples:
            break
    return selected


def covered_events(
    events: Sequence[train_mod.FootballEvent],
    start_sec: float,
    end_sec: float,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for event in events:
        if start_sec <= event.anchor_time <= end_sec:
            labels = [
                label
                for label, value in zip(train_mod.LABELS, event.labels)
                if float(value) > 0.0
            ]
            result.append(
                {
                    "event_id": event.event_id,
                    "labels": labels,
                    "anchor_time": float(event.anchor_time),
                    "offset_sec": float(event.anchor_time - start_sec),
                }
            )
    return result


def export_record(
    *,
    record: train_mod.LongVideoRecord,
    events: Sequence[train_mod.FootballEvent],
    cropper: RobustClipCropper,
    target_size: tuple[int, int],
    num_model_frames: int,
    output_dir: Path,
    output_fps: float,
    frame_mode: str,
    codec: str,
    ffmpeg_bin: str,
    layout: str,
    save_invalid_full_frame: bool,
    dry_run: bool,
) -> dict[str, Any]:
    start_sec, end_sec = train_mod.centered_window(
        record.anchor_time,
        float(record.base_clip_end - record.base_clip_start),
        record.video_duration,
    )
    width, height, source_fps, frame_count = video_geometry(record.video_path)
    if frame_mode == "model":
        frame_indices = train_mod.segment_frame_indices(
            frame_count,
            source_fps,
            num_model_frames,
            False,
            start_sec=start_sec,
            end_sec=end_sec,
        )
        frame_times = train_mod.frame_times_from_indices(frame_indices, source_fps)
    else:
        frame_times = np.arange(start_sec, end_sec, 1.0 / output_fps, dtype=np.float64).tolist()
        frame_indices = [
            min(max(int(round(time_sec * source_fps)), 0), frame_count - 1)
            for time_sec in frame_times
        ]

    if cropper.has_video(record.video_id):
        proposals, aggregate = cropper.get_clip_rois(
            record.video_id,
            start_sec,
            end_sec,
            frame_times,
            width,
            height,
            target_size,
        )
    else:
        proposals = [invalid_roi("missing_index") for _ in frame_times]
        aggregate = invalid_roi("missing_index")

    labels = positive_labels(record)
    window_events = covered_events(events, start_sec, end_sec)
    relative_dir = Path(record.video_id)
    stem = (
        f"{record.anchor_time:010.3f}_{safe_name('-'.join(labels))}_"
        f"{safe_name(record.sample_id)}"
    )
    review_path = output_dir / "videos" / "review" / relative_dir / f"{stem}.mp4"
    crop_path = output_dir / "videos" / "crop" / relative_dir / f"{stem}.mp4"
    write_review = layout in {"review", "both"}
    any_valid = any(proposal.valid for proposal in proposals)
    write_crop = layout in {"crop", "both"} and (any_valid or save_invalid_full_frame)
    valid_fraction = sum(float(proposal.valid) for proposal in proposals) / max(len(proposals), 1)
    centers = [
        ((proposal.bbox[0] + proposal.bbox[2]) * 0.5, (proposal.bbox[1] + proposal.bbox[3]) * 0.5)
        for proposal in proposals
        if proposal.valid and proposal.bbox is not None
    ]
    center_path_ratio = 0.0
    if len(centers) > 1:
        center_path_ratio = float(
            sum(
                np.linalg.norm(np.asarray(current) - np.asarray(previous))
                for previous, current in zip(centers[:-1], centers[1:])
            )
            / max(np.hypot(width, height), 1.0)
        )
    row: dict[str, Any] = {
        "source": record.source,
        "video_id": record.video_id,
        "sample_id": record.sample_id,
        "labels": ",".join(labels),
        "anchor_time_sec": float(record.anchor_time),
        "start_sec": float(start_sec),
        "end_sec": float(end_sec),
        "covered_events": json.dumps(window_events, ensure_ascii=False),
        "roi_valid": valid_fraction,
        "roi_mode": aggregate.mode,
        "roi_confidence": float(aggregate.roi_confidence),
        "bbox": json.dumps(list(aggregate.bbox), ensure_ascii=False) if aggregate.bbox else "",
        "frame_rois": json.dumps([proposal.to_dict() for proposal in proposals], ensure_ascii=False),
        "center_path_ratio": center_path_ratio,
        "area_ratio": float(aggregate.area_ratio),
        "goal_score": float(aggregate.goal_score),
        "ball_score": float(aggregate.ball_score),
        "center_circle_score": float(aggregate.center_circle_score),
        "person_support": float(aggregate.person_support),
        "fallback_reason": aggregate.fallback_reason,
        "review_video_path": str(review_path) if write_review else "",
        "crop_video_path": str(crop_path) if write_crop else "",
        "num_output_frames": 0,
        "status": "dry_run" if dry_run else "pending",
        "manual_roi_ok": "",
        "manual_ball_covered": "",
        "manual_goal_covered": "",
        "manual_people_covered": "",
        "manual_issue": "",
        "manual_notes": "",
    }
    if dry_run:
        return row

    review_writer = (
        open_writer(
            review_path,
            (target_size[0], target_size[1] * 2),
            output_fps,
            codec,
            ffmpeg_bin,
        )
        if write_review
        else None
    )
    crop_writer = (
        open_writer(crop_path, target_size, output_fps, codec, ffmpeg_bin)
        if write_crop
        else None
    )
    cap = cv2.VideoCapture(record.video_path)
    if not cap.isOpened():
        cap.release()
        if review_writer is not None:
            review_writer.release()
        if crop_writer is not None:
            crop_writer.release()
        raise FileNotFoundError(f"Could not open video: {record.video_path}")

    written = 0
    try:
        for slot, (frame_index, time_sec, proposal) in enumerate(
            zip(frame_indices, frame_times, proposals)
        ):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            crop = crop_and_resize(frame, proposal, target_size)
            if crop_writer is not None:
                crop_writer.write(crop)
            if review_writer is not None:
                overlay = draw_review_overlay(
                    frame,
                    proposal,
                    target_size,
                    (
                        f"id={record.video_id} labels={','.join(labels)} slot={slot + 1}/{len(proposals)}",
                        f"clip={start_sec:.2f}-{end_sec:.2f}s t={time_sec:.2f}s anchor={record.anchor_time:.2f}s",
                        (
                            f"policy={cropper.temporal_mode} roi={proposal.mode} "
                            f"valid={int(proposal.valid)} "
                            f"conf={proposal.roi_confidence:.2f} area={proposal.area_ratio:.3f}"
                        ),
                    ),
                )
                review_writer.write(np.concatenate([overlay, crop], axis=1))
            written += 1
    finally:
        cap.release()
        if review_writer is not None:
            review_writer.release()
        if crop_writer is not None:
            crop_writer.release()
    row["num_output_frames"] = written
    row["status"] = "ok" if written > 0 else "no_frames_written"
    return row


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = [
        "source",
        "video_id",
        "sample_id",
        "labels",
        "anchor_time_sec",
        "start_sec",
        "end_sec",
        "covered_events",
        "roi_valid",
        "roi_mode",
        "roi_confidence",
        "bbox",
        "frame_rois",
        "center_path_ratio",
        "area_ratio",
        "goal_score",
        "ball_score",
        "center_circle_score",
        "person_support",
        "fallback_reason",
        "review_video_path",
        "crop_video_path",
        "num_output_frames",
        "status",
        "error",
        "manual_roi_ok",
        "manual_ball_covered",
        "manual_goal_covered",
        "manual_people_covered",
        "manual_issue",
        "manual_notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    args = parse_args()
    if args.output_fps <= 0:
        raise ValueError("--output-fps must be positive")
    cfg = train_mod.load_config(args.config, args.set)
    train_mod.configure_label_schema(cfg)
    if train_mod.data_mode(cfg) != "long_videos":
        raise ValueError("This exporter requires data.mode=long_videos")
    cropper = RobustClipCropper.from_config(cfg)
    if cropper is None:
        raise ValueError("This exporter requires spatial_crop.mode=robust_detector_aware")

    records, events_by_video = train_mod.load_long_video_records(cfg, "val")
    requested_labels = set(split_csv(args.labels))
    unknown_labels = requested_labels.difference(train_mod.LABELS)
    if unknown_labels:
        raise ValueError(f"Unknown --labels={sorted(unknown_labels)}; expected {train_mod.LABELS}")
    selected = select_records(
        records,
        video_ids=set(split_csv(args.video_ids)),
        labels=requested_labels,
        max_samples=max(args.max_samples, 0),
        max_per_video=max(args.max_per_video, 0),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_size = train_mod.parse_image_size(cfg.video.image_size)

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(selected, start=1):
        try:
            row = export_record(
                record=record,
                events=events_by_video[(record.source, record.video_id)],
                cropper=cropper,
                target_size=target_size,
                num_model_frames=int(cfg.video.num_frames),
                output_dir=output_dir,
                output_fps=float(args.output_fps),
                frame_mode=args.frame_mode,
                codec=args.codec,
                ffmpeg_bin=args.ffmpeg_bin,
                layout=args.layout,
                save_invalid_full_frame=bool(args.save_invalid_full_frame),
                dry_run=bool(args.dry_run),
            )
        except Exception as exc:
            row = {
                "source": record.source,
                "video_id": record.video_id,
                "sample_id": record.sample_id,
                "labels": ",".join(positive_labels(record)),
                "anchor_time_sec": float(record.anchor_time),
                "status": f"error:{type(exc).__name__}",
                "error": str(exc),
            }
        rows.append(row)
        print(
            f"[{index}/{len(selected)}] video_id={record.video_id} "
            f"anchor={record.anchor_time:.3f} labels={','.join(positive_labels(record))} "
            f"status={row['status']} roi={row.get('roi_mode', '')}",
            flush=True,
        )

    write_csv(output_dir / "manifest.csv", rows)
    summary = {
        "config": args.config,
        "overrides": args.set,
        "split": "val",
        "target_size": list(target_size),
        "output_fps": float(args.output_fps),
        "frame_mode": args.frame_mode,
        "temporal_mode": cropper.temporal_mode,
        "layout": args.layout,
        "dry_run": bool(args.dry_run),
        "num_selected": len(selected),
        "num_ok": sum(row.get("status") == "ok" for row in rows),
        "num_invalid_roi": sum(float(row.get("roi_valid", 0) or 0) <= 0.0 for row in rows),
        "num_errors": sum(str(row.get("status", "")).startswith("error:") for row in rows),
        "label_counts": {
            label: sum(label in split_csv(str(row.get("labels", ""))) for row in rows)
            for label in train_mod.LABELS
        },
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"done selected={summary['num_selected']} ok={summary['num_ok']} "
        f"invalid_roi={summary['num_invalid_roi']} errors={summary['num_errors']} "
        f"output_dir={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
