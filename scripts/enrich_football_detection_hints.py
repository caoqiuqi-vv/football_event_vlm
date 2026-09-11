#!/usr/bin/env python
"""Conditionally recover missed footballs in supervised event time ranges.

The script never overwrites the original compact indices.  It runs high-
resolution crop inference only when an indexed frame has no reliable football,
using a goal-centred crop and/or the densest player crop.  Recovered detections
are merged into a new v2-compatible index with per-object provenance.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_football_events import (  # noqa: E402
    ConfigDict,
    configure_label_schema,
    load_config,
    load_long_video_records,
)


SOURCE_ORIGINAL = 0
SOURCE_GOAL_CROP = 1
SOURCE_CROWD_CROP = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-index-root", required=True)
    parser.add_argument("--output-index-root", required=True)
    parser.add_argument(
        "--detector-repo",
        default="/home/new_users/qiuqi/code/football_event_detection",
    )
    parser.add_argument("--weights", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--crop-size", type=int, default=640)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--det-conf", type=float, default=0.10)
    parser.add_argument("--det-iou", type=float, default=0.45)
    parser.add_argument("--ball-conf", type=float, default=0.10)
    parser.add_argument("--recovered-ball-conf", type=float, default=0.15)
    parser.add_argument("--goal-conf", type=float, default=0.40)
    parser.add_argument("--person-conf", type=float, default=0.35)
    parser.add_argument("--crowd-min-people", type=int, default=4)
    parser.add_argument("--max-added-balls-per-frame", type=int, default=3)
    parser.add_argument("--max-sequential-gap-sec", type=float, default=3.0)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def merge_intervals(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted((max(float(a), 0.0), max(float(b), 0.0)) for a, b in intervals if b > a)
    merged: list[list[float]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1] + 0.5:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def supervised_intervals(cfg: ConfigDict) -> tuple[dict[str, list[tuple[float, float]]], dict[str, str]]:
    clip_duration = float(cfg.video.get("clip_duration", 10.0))
    event_margin = float(cfg.video.get("event_margin", 1.0))
    train_jitter = float(cfg.video.get("temporal_jitter_sec", 0.0))
    hard_jitter = float(
        cfg.data.long_video.get("hard_negative", ConfigDict()).get(
            "temporal_jitter_sec", train_jitter
        )
    )
    intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    video_paths: dict[str, str] = {}
    for split in ("train", "val"):
        records, _ = load_long_video_records(cfg, split)
        for record in records:
            video_paths[record.video_id] = record.video_path
            if split == "train" and not record.is_negative:
                start = record.anchor_time - clip_duration + event_margin
                end = record.anchor_time + clip_duration - event_margin
            elif split == "train":
                jitter = hard_jitter if record.sample_id.startswith("hard_neg_") else train_jitter
                start = record.base_clip_start - jitter
                end = record.base_clip_end + jitter
            else:
                start, end = record.base_clip_start, record.base_clip_end
            intervals[record.video_id].append(
                (max(start, 0.0), min(end, record.video_duration))
            )
    return {key: merge_intervals(value) for key, value in intervals.items()}, video_paths


def frames_in_intervals(
    frame_ids: np.ndarray, fps: float, intervals: Sequence[tuple[float, float]]
) -> np.ndarray:
    selected = np.zeros(len(frame_ids), dtype=bool)
    for start, end in intervals:
        lo = int(np.searchsorted(frame_ids, math.floor(start * fps), side="left"))
        hi = int(np.searchsorted(frame_ids, math.ceil(end * fps), side="right"))
        selected[lo:hi] = True
    return selected


def square_crop(box: Sequence[float], crop_size: int, width: int, height: int) -> tuple[int, int, int, int]:
    crop_w, crop_h = min(crop_size, width), min(crop_size, height)
    cx = (float(box[0]) + float(box[2])) * 0.5
    cy = (float(box[1]) + float(box[3])) * 0.5
    left = int(round(cx - crop_w * 0.5))
    top = int(round(cy - crop_h * 0.5))
    left = max(0, min(left, width - crop_w))
    top = max(0, min(top, height - crop_h))
    return left, top, left + crop_w, top + crop_h


def densest_people_crop(
    boxes: np.ndarray,
    confidences: np.ndarray,
    crop_size: int,
    width: int,
    height: int,
    min_people: int,
) -> tuple[int, int, int, int] | None:
    if len(boxes) < min_people:
        return None
    centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
    best: tuple[float, tuple[int, int, int, int]] | None = None
    for center_x, center_y in centers:
        candidate = square_crop(
            (center_x, center_y, center_x, center_y), crop_size, width, height
        )
        inside = (
            (centers[:, 0] >= candidate[0])
            & (centers[:, 0] <= candidate[2])
            & (centers[:, 1] >= candidate[1])
            & (centers[:, 1] <= candidate[3])
        )
        count = int(inside.sum())
        if count < min_people:
            continue
        score = float(count) + 0.1 * float(confidences[inside].sum())
        if best is None or score > best[0]:
            best = (score, candidate)
    return best[1] if best is not None else None


def crop_iou(first: Sequence[int], second: Sequence[int]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(right - left, 0) * max(bottom - top, 0)
    first_area = max(first[2] - first[0], 0) * max(first[3] - first[1], 0)
    second_area = max(second[2] - second[0], 0) * max(second[3] - second[1], 0)
    return intersection / max(first_area + second_area - intersection, 1)


def candidate_jobs(
    payload: dict[str, Any],
    intervals: Sequence[tuple[float, float]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    frame_ids = payload["frame_ids"].numpy()
    offsets = payload["frame_offsets"].numpy()
    classes = payload["classes"].numpy()
    confidences = payload["confidences"].float().numpy()
    boxes = payload["boxes"].float().numpy()
    fps = float(payload["fps"])
    width = int(payload["image_size"]["width"])
    height = int(payload["image_size"]["height"])
    selected = frames_in_intervals(frame_ids, fps, intervals)
    jobs: list[dict[str, Any]] = []
    stats = defaultdict(int)
    for position in np.flatnonzero(selected):
        stats["relevant_frames"] += 1
        lo, hi = int(offsets[position]), int(offsets[position + 1])
        local_classes = classes[lo:hi]
        local_conf = confidences[lo:hi]
        local_boxes = boxes[lo:hi]
        if bool(((local_classes == 1) & (local_conf >= args.ball_conf)).any()):
            stats["original_ball_frames"] += 1
            continue
        stats["missing_ball_frames"] += 1
        frame_jobs: list[tuple[int, tuple[int, int, int, int]]] = []
        goal_mask = (local_classes == 2) & (local_conf >= args.goal_conf)
        if goal_mask.any():
            goal_indices = np.flatnonzero(goal_mask)
            goal_index = goal_indices[int(np.argmax(local_conf[goal_indices]))]
            frame_jobs.append(
                (
                    SOURCE_GOAL_CROP,
                    square_crop(local_boxes[goal_index], args.crop_size, width, height),
                )
            )
            stats["goal_trigger_frames"] += 1
        people_mask = (local_classes == 0) & (local_conf >= args.person_conf)
        crowd_crop = densest_people_crop(
            local_boxes[people_mask],
            local_conf[people_mask],
            args.crop_size,
            width,
            height,
            args.crowd_min_people,
        )
        if crowd_crop is not None:
            stats["crowd_trigger_frames"] += 1
            if not frame_jobs or crop_iou(frame_jobs[0][1], crowd_crop) < 0.70:
                frame_jobs.append((SOURCE_CROWD_CROP, crowd_crop))
            else:
                stats["deduplicated_crowd_crops"] += 1
        if frame_jobs:
            stats["triggered_frames"] += 1
        else:
            stats["no_anchor_frames"] += 1
        for source, crop in frame_jobs:
            jobs.append(
                {
                    "frame_id": int(frame_ids[position]),
                    "source": source,
                    "crop": crop,
                }
            )
    stats["crop_jobs"] = len(jobs)
    return jobs, dict(stats)


def iter_job_batches(
    video_path: str,
    jobs: Sequence[dict[str, Any]],
    fps: float,
    index_width: int,
    index_height: int,
    max_sequential_gap_sec: float,
    batch_size: int,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise FileNotFoundError(f"Could not open video: {video_path}")
    last_frame_id = -1
    last_frame: np.ndarray | None = None
    batch_jobs: list[dict[str, Any]] = []
    batch_crops: list[np.ndarray] = []
    try:
        for job in jobs:
            frame_id = int(job["frame_id"])
            if frame_id != last_frame_id:
                if (
                    last_frame is None
                    or frame_id < last_frame_id
                    or frame_id - last_frame_id > max(int(fps * max_sequential_gap_sec), 1)
                ):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                    ok, frame = cap.read()
                else:
                    ok, frame = True, last_frame
                    for _ in range(last_frame_id + 1, frame_id + 1):
                        ok, frame = cap.read()
                        if not ok:
                            break
                if not ok or frame is None:
                    # Long camera videos occasionally contain one corrupt GOP.
                    # Reopen and seek once; if the exact frame is still unreadable,
                    # skip only its crop jobs instead of losing the whole video/rank.
                    cap.release()
                    cap = cv2.VideoCapture(video_path)
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                        ok, frame = cap.read()
                    if not ok or frame is None:
                        print(
                            f"WARN decode-skip video={video_path} frame={frame_id}",
                            flush=True,
                        )
                        last_frame_id, last_frame = -1, None
                        continue
                last_frame_id, last_frame = frame_id, frame
            index_left, index_top, index_right, index_bottom = job["crop"]
            frame_height, frame_width = last_frame.shape[:2]
            scale_x = frame_width / max(index_width, 1)
            scale_y = frame_height / max(index_height, 1)
            left = max(0, min(int(round(index_left * scale_x)), frame_width - 1))
            top = max(0, min(int(round(index_top * scale_y)), frame_height - 1))
            right = max(left + 1, min(int(round(index_right * scale_x)), frame_width))
            bottom = max(top + 1, min(int(round(index_bottom * scale_y)), frame_height))
            crop = last_frame[top:bottom, left:right]
            if crop.size == 0:
                raise RuntimeError(
                    f"Empty crop video={video_path} frame={frame_id} "
                    f"index_crop={job['crop']} decoded_crop={(left, top, right, bottom)} "
                    f"index_size={(index_width, index_height)} frame_size={(frame_width, frame_height)}"
                )
            runtime_job = dict(job)
            runtime_job["decoded_crop"] = (left, top, right, bottom)
            batch_jobs.append(runtime_job)
            batch_crops.append(crop.copy())
            if len(batch_jobs) >= batch_size:
                yield batch_jobs, batch_crops
                batch_jobs, batch_crops = [], []
        if batch_jobs:
            yield batch_jobs, batch_crops
    finally:
        cap.release()


def recover_balls(
    model: Any,
    detector_module: Any,
    payload: dict[str, Any],
    video_path: str,
    jobs: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[int, list[dict[str, Any]]]:
    recovered: dict[int, list[dict[str, Any]]] = defaultdict(list)
    frame_area = float(payload["image_size"]["width"] * payload["image_size"]["height"])
    for batch_jobs, batch_crops in iter_job_batches(
        video_path,
        jobs,
        float(payload["fps"]),
        int(payload["image_size"]["width"]),
        int(payload["image_size"]["height"]),
        args.max_sequential_gap_sec,
        args.batch_size,
    ):
        detections = detector_module.detect_batch(
            model,
            batch_crops,
            args.imgsz,
            args.det_conf,
            args.det_iou,
        )
        for job, objects in zip(batch_jobs, detections):
            index_left, index_top, index_right, index_bottom = job["crop"]
            decoded_left, decoded_top, decoded_right, decoded_bottom = job["decoded_crop"]
            decoded_width = max(decoded_right - decoded_left, 1)
            decoded_height = max(decoded_bottom - decoded_top, 1)
            scale_x = (index_right - index_left) / decoded_width
            scale_y = (index_bottom - index_top) / decoded_height
            for obj in objects:
                if int(obj.get("cls", -1)) != 1:
                    continue
                confidence = float(obj.get("conf", 0.0))
                if confidence < args.recovered_ball_conf:
                    continue
                x1, y1, x2, y2 = map(float, obj["bbox"])
                mapped = [
                    index_left + x1 * scale_x,
                    index_top + y1 * scale_y,
                    index_left + x2 * scale_x,
                    index_top + y2 * scale_y,
                ]
                box_w, box_h = max(mapped[2] - mapped[0], 0.0), max(mapped[3] - mapped[1], 0.0)
                area_ratio = box_w * box_h / max(frame_area, 1.0)
                aspect = box_w / max(box_h, 1e-6)
                if not (1e-6 <= area_ratio <= 2e-3 and 0.25 <= aspect <= 4.0):
                    continue
                recovered[int(job["frame_id"])].append(
                    {
                        "conf": confidence,
                        "bbox": mapped,
                        "source": int(job["source"]),
                    }
                )
    for frame_id, objects in recovered.items():
        objects.sort(key=lambda item: float(item["conf"]), reverse=True)
        kept: list[dict[str, Any]] = []
        for obj in objects:
            if all(
                crop_iou(
                    tuple(map(int, obj["bbox"])),
                    tuple(map(int, other["bbox"])),
                )
                < 0.50
                for other in kept
            ):
                kept.append(obj)
            if len(kept) >= args.max_added_balls_per_frame:
                break
        recovered[frame_id] = kept
    return recovered


def merge_payload(
    payload: dict[str, Any], recovered: dict[int, list[dict[str, Any]]]
) -> dict[str, Any]:
    frame_ids = payload["frame_ids"].numpy()
    offsets = payload["frame_offsets"].numpy()
    classes, confidences, boxes, track_ids, sources = [], [], [], [], []
    new_offsets = [0]
    old_sources = payload.get("object_sources")
    if old_sources is None:
        old_sources = torch.zeros(len(payload["classes"]), dtype=torch.int8)
    for position, frame_id in enumerate(frame_ids):
        lo, hi = int(offsets[position]), int(offsets[position + 1])
        classes.extend(payload["classes"][lo:hi].tolist())
        confidences.extend(payload["confidences"][lo:hi].float().tolist())
        boxes.extend(payload["boxes"][lo:hi].float().tolist())
        track_ids.extend(payload["track_ids"][lo:hi].tolist())
        sources.extend(old_sources[lo:hi].tolist())
        for obj in recovered.get(int(frame_id), []):
            classes.append(1)
            confidences.append(float(obj["conf"]))
            boxes.append(list(map(float, obj["bbox"])))
            track_ids.append(-1)
            sources.append(int(obj["source"]))
        new_offsets.append(len(classes))
    result = dict(payload)
    result.update(
        {
            "frame_offsets": torch.tensor(new_offsets, dtype=torch.int64),
            "classes": torch.tensor(classes, dtype=torch.int8),
            "confidences": torch.tensor(confidences, dtype=torch.float16),
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "track_ids": torch.tensor(track_ids, dtype=torch.int32),
            "object_sources": torch.tensor(sources, dtype=torch.int8),
            "object_source_names": {
                SOURCE_ORIGINAL: "original",
                SOURCE_GOAL_CROP: "goal_crop_redetect",
                SOURCE_CROWD_CROP: "crowd_crop_redetect",
            },
            "conditional_redetect": True,
        }
    )
    return add_normalized_coordinates(result)


def _normalize_boxes(boxes: torch.Tensor, width: float, height: float) -> torch.Tensor:
    values = boxes.float().reshape(-1, 4).clone()
    if len(values):
        scale = torch.tensor([width, height, width, height], dtype=torch.float32)
        values = (values / scale.clamp(min=1.0)).clamp(0.0, 1.0)
    return values


def add_normalized_coordinates(payload: dict[str, Any]) -> dict[str, Any]:
    """Add resolution-independent xyxy coordinates while preserving v2 compatibility."""
    result = dict(payload)
    width = float(payload["image_size"]["width"])
    height = float(payload["image_size"]["height"])
    result["boxes_normalized"] = _normalize_boxes(payload["boxes"], width, height)
    if "ball_boxes" in payload:
        result["ball_boxes_normalized"] = _normalize_boxes(payload["ball_boxes"], width, height)
    result["box_coordinate_system"] = "absolute_xyxy_in_image_size"
    result["normalized_box_coordinate_system"] = "normalized_xyxy_relative_to_original_source_frame"
    result["normalized_coordinate_reference_size"] = {
        "width": int(round(width)),
        "height": int(round(height)),
    }
    return result


def main() -> None:
    args = parse_args()
    if args.world_size <= 0 or not 0 <= args.rank < args.world_size:
        raise ValueError("rank must be in [0, world_size)")
    cfg = load_config(args.config, [])
    configure_label_schema(cfg)
    intervals, video_paths = supervised_intervals(cfg)
    video_ids = sorted(intervals)[args.rank :: args.world_size]
    if args.max_videos:
        video_ids = video_ids[: args.max_videos]
    input_root = Path(args.input_index_root)
    output_root = Path(args.output_index_root)
    output_root.mkdir(parents=True, exist_ok=True)

    detector_module = None
    model = None
    if not args.plan_only:
        detector_repo = Path(args.detector_repo).resolve()
        sys.path.insert(0, str(detector_repo))
        import run_detection_tracking as detector_module  # type: ignore

        weights = Path(args.weights) if args.weights else detector_module.DEFAULT_WEIGHTS
        load_device = args.device
        # YOLOv5 select_device("cuda:0") rewrites CUDA_VISIBLE_DEVICES to "0".
        # When ranks are already isolated by the launcher this would collapse
        # every process onto physical GPU 0.  An empty device lets YOLO select
        # local cuda:0 without overwriting the external physical mapping.
        if os.environ.get("CUDA_VISIBLE_DEVICES") and args.device in {"0", "cuda", "cuda:0"}:
            load_device = ""
        model = detector_module.load_yolo_model(
            weights, load_device, detector_module.DEFAULT_YOLOV5_REPO
        )
        if args.fp16 and args.device != "cpu" and torch.cuda.is_available():
            model.half()

    aggregate = defaultdict(int)
    rows = []
    started = time.time()
    for sequence, video_id in enumerate(video_ids, start=1):
        source_path = input_root / f"{video_id}.pt"
        output_path = output_root / f"{video_id}.pt"
        if output_path.exists() and not args.rebuild:
            existing = torch.load(output_path, map_location="cpu", weights_only=False)
            if "boxes_normalized" not in existing:
                torch.save(add_normalized_coordinates(existing), output_path)
                print(f"[{sequence}/{len(video_ids)}] upgrade-coordinates {video_id}", flush=True)
            else:
                print(f"[{sequence}/{len(video_ids)}] skip {video_id}", flush=True)
            continue
        payload = torch.load(source_path, map_location="cpu", weights_only=False)
        jobs, stats = candidate_jobs(payload, intervals[video_id], args)
        for key, value in stats.items():
            aggregate[key] += int(value)
        recovered: dict[int, list[dict[str, Any]]] = {}
        if jobs and not args.plan_only:
            recovered = recover_balls(
                model, detector_module, payload, video_paths[video_id], jobs, args
            )
        if not args.plan_only:
            enriched = merge_payload(payload, recovered)
            torch.save(enriched, output_path)
        recovered_frames = len(recovered)
        recovered_objects = sum(len(items) for items in recovered.values())
        aggregate["recovered_frames"] += recovered_frames
        aggregate["recovered_objects"] += recovered_objects
        row = {
            "video_id": video_id,
            **stats,
            "recovered_frames": recovered_frames,
            "recovered_objects": recovered_objects,
            "output": "" if args.plan_only else str(output_path),
        }
        rows.append(row)
        print(
            f"[{sequence}/{len(video_ids)}] {video_id} relevant={stats.get('relevant_frames', 0)} "
            f"missing={stats.get('missing_ball_frames', 0)} crops={stats.get('crop_jobs', 0)} "
            f"recovered_frames={recovered_frames}",
            flush=True,
        )
    summary = {
        "config": args.config,
        "rank": args.rank,
        "world_size": args.world_size,
        "plan_only": args.plan_only,
        "videos": len(video_ids),
        "aggregate": dict(aggregate),
        "elapsed_sec": time.time() - started,
        "rows": rows,
    }
    summary_path = output_root / f"redetect_summary_rank{args.rank:02d}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({**summary["aggregate"], "videos": len(video_ids)}, ensure_ascii=False), flush=True)
    print(f"summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
