#!/usr/bin/env python
"""Offline multi-strategy football detection and temporal pseudo-label tracking.

Detection and tracking are deliberately separate phases. Detection workers can
use every available GPU without waiting for CPU tracking, while compact chunked
JSONL files make the multi-day job resumable at five-minute boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


SCHEMA = "football-ball-pseudolabel-v1"
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".m4v", ".webm"}
DET_AND_TRACK = Path("/home/new_users/qiuqi/code/det_and_track")


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def stable_fingerprint(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def list_videos(source: Path) -> list[Path]:
    if source.is_file():
        return [source.resolve()]
    videos = [
        path.resolve()
        for path in source.iterdir()
        if path.is_file() and path.suffix.casefold() in VIDEO_EXTENSIONS
    ]
    if not videos:
        raise FileNotFoundError(f"no videos found under {source}")
    return sorted(videos)


def read_video_ids(paths: Sequence[str]) -> set[str]:
    """Read newline-delimited video ids, accepting either bare ids or filenames."""
    video_ids: set[str] = set()
    for value in paths:
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"video id list does not exist: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            item = line.strip()
            if not item or item.startswith("#"):
                continue
            video_ids.add(Path(item).stem)
    return video_ids


def balanced_shards(videos: Sequence[Path], count: int) -> list[list[Path]]:
    """Greedy largest-first assignment keeps multi-GPU shards approximately even."""
    if count <= 0:
        raise ValueError("shard count must be positive")
    shards: list[list[Path]] = [[] for _ in range(count)]
    loads = [0] * count
    for video in sorted(videos, key=lambda path: (-path.stat().st_size, path.name)):
        index = min(range(count), key=lambda value: (loads[value], value))
        shards[index].append(video)
        loads[index] += video.stat().st_size
    return shards


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0:
        return 0.0
    area_first = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    area_second = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    return intersection / max(area_first + area_second - intersection, 1e-8)


def merge_candidates(
    candidates: Sequence[dict[str, Any]], iou_threshold: float, limit: int
) -> list[dict[str, Any]]:
    pending = sorted(candidates, key=lambda row: float(row["confidence"]), reverse=True)
    kept: list[dict[str, Any]] = []
    while pending and (limit <= 0 or len(kept) < limit):
        best = dict(pending.pop(0))
        sources = {str(best.pop("source"))}
        remaining = []
        for row in pending:
            if box_iou(best["bbox_xyxy"], row["bbox_xyxy"]) >= iou_threshold:
                sources.add(str(row["source"]))
            else:
                remaining.append(row)
        best["sources"] = sorted(sources)
        kept.append(best)
        pending = remaining
    return kept


def window_starts(length: int, size: int, overlap: float) -> list[int]:
    if length <= size:
        return [0]
    stride = max(1, int(round(size * (1.0 - overlap))))
    starts = list(range(0, length - size + 1, stride))
    final = length - size
    if starts[-1] != final:
        starts.append(final)
    return starts


def _boxes_from_results(
    results: Sequence[Any], *, source: str, width: int, height: int, flip: bool = False
) -> list[list[dict[str, Any]]]:
    output: list[list[dict[str, Any]]] = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        rows: list[dict[str, Any]] = []
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.detach().float().cpu().numpy()
            scores = boxes.conf.detach().float().cpu().numpy()
            for box, score in zip(xyxy, scores):
                x1, y1, x2, y2 = (float(value) for value in box)
                if flip:
                    x1, x2 = float(width) - x2, float(width) - x1
                clipped = [
                    min(max(x1, 0.0), float(width)),
                    min(max(y1, 0.0), float(height)),
                    min(max(x2, 0.0), float(width)),
                    min(max(y2, 0.0), float(height)),
                ]
                if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                    continue
                rows.append(
                    {
                        "bbox_xyxy": clipped,
                        "confidence": float(score),
                        "source": source,
                    }
                )
        output.append(rows)
    return output


class BallDetector:
    def __init__(self, args: argparse.Namespace) -> None:
        from ultralytics import YOLO

        self.args = args
        self.model = YOLO(args.checkpoint)
        names = self.model.names
        accepted = {"football", "soccer_ball", "soccer ball", "ball"}
        matches = [
            int(index)
            for index, name in names.items()
            if str(name).casefold().strip() in accepted
        ]
        if len(matches) != 1:
            raise ValueError(f"ball checkpoint must expose exactly one ball class: {names}")
        self.ball_class = matches[0]

    def _predict(self, images: Sequence[np.ndarray], imgsz: int) -> Sequence[Any]:
        if not images:
            return []
        return self.model.predict(
            list(images),
            imgsz=int(imgsz),
            conf=float(self.args.candidate_confidence),
            iou=float(self.args.detector_iou),
            classes=[self.ball_class],
            device=str(self.args.device),
            half=bool(self.args.half),
            verbose=False,
            max_det=int(self.args.max_detections_per_view),
        )

    def detect(self, frames: Sequence[np.ndarray]) -> list[list[dict[str, Any]]]:
        width, height = self.args.canonical_width, self.args.canonical_height
        combined: list[list[dict[str, Any]]] = [[] for _ in frames]
        full = _boxes_from_results(
            self._predict(frames, self.args.full_image_size),
            source="full",
            width=width,
            height=height,
        )
        for index, rows in enumerate(full):
            combined[index].extend(rows)

        if self.args.sliding_windows:
            crops: list[np.ndarray] = []
            owners: list[tuple[int, int, int]] = []
            size = int(self.args.tile_size)
            for frame_index, frame in enumerate(frames):
                for y0 in window_starts(height, size, self.args.tile_overlap):
                    for x0 in window_starts(width, size, self.args.tile_overlap):
                        crops.append(frame[y0 : y0 + size, x0 : x0 + size])
                        owners.append((frame_index, x0, y0))
            for offset in range(0, len(crops), int(self.args.tile_batch_size)):
                current = crops[offset : offset + int(self.args.tile_batch_size)]
                results = _boxes_from_results(
                    self._predict(current, self.args.tile_image_size),
                    source="tile",
                    width=size,
                    height=size,
                )
                for local_index, rows in enumerate(results):
                    frame_index, x0, y0 = owners[offset + local_index]
                    for row in rows:
                        x1, y1, x2, y2 = row["bbox_xyxy"]
                        row["bbox_xyxy"] = [x1 + x0, y1 + y0, x2 + x0, y2 + y0]
                        combined[frame_index].append(row)

        if self.args.horizontal_flip_tta:
            if self.args.horizontal_flip_missing_only:
                flip_indices = [index for index, rows in enumerate(combined) if not rows]
            else:
                flip_indices = list(range(len(frames)))
            flipped = [np.ascontiguousarray(frames[index][:, ::-1]) for index in flip_indices]
            flip_rows = _boxes_from_results(
                self._predict(flipped, self.args.full_image_size),
                source="hflip",
                width=width,
                height=height,
                flip=True,
            )
            for frame_index, rows in zip(flip_indices, flip_rows):
                combined[frame_index].extend(rows)

        normalized: list[list[dict[str, Any]]] = []
        for rows in combined:
            merged = merge_candidates(rows, self.args.merge_iou, self.args.max_candidates)
            for row in merged:
                x1, y1, x2, y2 = row["bbox_xyxy"]
                row["bbox_xyxy_norm"] = [
                    x1 / width,
                    y1 / height,
                    x2 / width,
                    y2 / height,
                ]
                row["center_norm"] = [
                    (x1 + x2) / (2.0 * width),
                    (y1 + y2) / (2.0 * height),
                ]
            normalized.append(merged)
        return normalized


def video_metadata(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"failed to open video {path}")
    metadata = {
        "source": str(path),
        "video_id": path.stem,
        "fps": float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
        "source_frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "source_width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "source_height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    }
    capture.release()
    if metadata["fps"] <= 0 or metadata["source_frames"] <= 0:
        raise ValueError(f"invalid video metadata {metadata}")
    return metadata


def iter_chunk_frames(
    path: Path,
    start_frame: int,
    end_frame: int,
    stride: int,
    width: int,
    height: int,
) -> Iterable[tuple[int, np.ndarray]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"failed to open video {path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
    frame_id = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES)))
    try:
        while frame_id < end_frame:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_id % stride == 0:
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                yield frame_id, frame
            frame_id += 1
    finally:
        capture.release()


def detection_contract(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint)
    stat = checkpoint.stat()
    return {
        "schema": SCHEMA,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_size": stat.st_size,
        "checkpoint_mtime_ns": stat.st_mtime_ns,
        "candidate_confidence": args.candidate_confidence,
        "detector_iou": args.detector_iou,
        "full_image_size": args.full_image_size,
        "horizontal_flip_tta": args.horizontal_flip_tta,
        "horizontal_flip_missing_only": args.horizontal_flip_missing_only,
        "sliding_windows": args.sliding_windows,
        "tile_size": args.tile_size,
        "tile_overlap": args.tile_overlap,
        "tile_image_size": args.tile_image_size,
        "merge_iou": args.merge_iou,
        "sample_fps": args.sample_fps,
        "canonical_size": [args.canonical_height, args.canonical_width],
    }


def detect_video(path: Path, output_root: Path, detector: BallDetector, args: argparse.Namespace) -> dict[str, Any]:
    metadata = video_metadata(path)
    video_dir = output_root / path.stem
    chunks_dir = video_dir / "detection_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    contract = detection_contract(args)
    fingerprint = stable_fingerprint(contract)
    metadata_path = video_dir / "metadata.json"
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing.get("detection_fingerprint") != fingerprint:
            raise ValueError(
                f"refusing to mix detection contracts in {video_dir}: "
                f"{existing.get('detection_fingerprint')} != {fingerprint}"
            )
    stride = max(1, int(round(metadata["fps"] / float(args.sample_fps))))
    effective_fps = metadata["fps"] / stride
    chunk_frames = max(stride, int(round(metadata["fps"] * args.chunk_seconds)))
    metadata.update(
        {
            "schema": SCHEMA,
            "status": "detecting",
            "sample_stride": stride,
            "effective_sample_fps": effective_fps,
            "canonical_width": args.canonical_width,
            "canonical_height": args.canonical_height,
            "detection_contract": contract,
            "detection_fingerprint": fingerprint,
        }
    )
    atomic_write_json(metadata_path, metadata)
    total_candidates = 0
    sampled_frames = 0
    completed_chunks = 0
    started = time.time()
    source_limit = metadata["source_frames"]
    if args.max_video_seconds > 0:
        source_limit = min(source_limit, int(metadata["fps"] * args.max_video_seconds))
    for chunk_index, start in enumerate(range(0, source_limit, chunk_frames)):
        chunk_path = chunks_dir / f"chunk_{chunk_index:06d}.jsonl"
        if chunk_path.exists() and chunk_path.stat().st_size > 0:
            completed_chunks += 1
            continue
        end = min(start + chunk_frames, source_limit)
        rows: list[str] = []
        buffer_ids: list[int] = []
        buffer_frames: list[np.ndarray] = []

        def flush() -> None:
            nonlocal total_candidates, sampled_frames
            if not buffer_frames:
                return
            detections = detector.detect(buffer_frames)
            for frame_id, candidates in zip(buffer_ids, detections):
                rows.append(
                    json.dumps(
                        {
                            "sample_index": int(frame_id // stride),
                            "source_frame_id": int(frame_id),
                            "timestamp_sec": float(frame_id / metadata["fps"]),
                            "candidates": candidates,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
                total_candidates += len(candidates)
                sampled_frames += 1
            buffer_ids.clear()
            buffer_frames.clear()

        for frame_id, frame in iter_chunk_frames(
            path,
            start,
            end,
            stride,
            args.canonical_width,
            args.canonical_height,
        ):
            buffer_ids.append(frame_id)
            buffer_frames.append(frame)
            if len(buffer_frames) >= args.frame_batch_size:
                flush()
        flush()
        atomic_write_text(chunk_path, "\n".join(rows) + ("\n" if rows else ""))
        completed_chunks += 1
        print(
            f"video={path.stem} chunk={chunk_index} source={start}:{end} "
            f"sampled={len(rows)} elapsed={time.time() - started:.1f}s",
            flush=True,
        )
    metadata.update(
        {
            "status": "detection_complete",
            "completed_chunks": completed_chunks,
            "sampled_frames_current_run": sampled_frames,
            "candidates_current_run": total_candidates,
            "completed_at_unix": time.time(),
        }
    )
    atomic_write_json(metadata_path, metadata)
    atomic_write_json(video_dir / "detection_complete.json", metadata)
    return metadata


def claim_next_video(
    videos: Sequence[Path], output_root: Path, args: argparse.Namespace
) -> tuple[Path, Path] | None:
    """Atomically claim one unfinished video from the shared dynamic queue."""
    claims_root = output_root / "dynamic_claims"
    claims_root.mkdir(parents=True, exist_ok=True)
    for video in sorted(videos, key=lambda path: (-path.stat().st_size, path.name)):
        if (output_root / video.stem / "detection_complete.json").exists():
            continue
        claim = claims_root / video.stem
        try:
            claim.mkdir()
        except FileExistsError:
            continue
        atomic_write_json(
            claim / "owner.json",
            {
                "schema": SCHEMA,
                "video_id": video.stem,
                "source": str(video),
                "pid": os.getpid(),
                "host": os.uname().nodename,
                "device": str(args.device),
                "worker_index": int(args.shard_index),
                "claimed_at_unix": time.time(),
            },
        )
        return video, claim
    return None


def run_detection(args: argparse.Namespace) -> None:
    videos = list_videos(Path(args.source))
    excluded_ids = read_video_ids(args.exclude_video_id_file)
    present_ids = {path.stem for path in videos}
    videos = [path for path in videos if path.stem not in excluded_ids]
    excluded_present = sorted(excluded_ids & present_ids)
    if excluded_ids:
        print(
            f"exclusion files={len(args.exclude_video_id_file)} "
            f"requested={len(excluded_ids)} excluded_present={len(excluded_present)} "
            f"remaining={len(videos)}",
            flush=True,
        )
    if not videos:
        raise ValueError("no videos remain after applying exclusions")
    output_root = Path(args.output_root)
    if args.dynamic_queue:
        assigned = videos
        assignment = "dynamic"
    else:
        assigned = balanced_shards(videos, args.num_shards)[args.shard_index]
        assignment = "static"
    print(
        f"detection_worker shard={args.shard_index}/{args.num_shards} "
        f"queue_videos={len(assigned)} assignment={assignment} device={args.device}",
        flush=True,
    )
    detector = BallDetector(args)
    failures = []
    processed = 0
    while True:
        if args.dynamic_queue:
            claimed = claim_next_video(assigned, output_root, args)
            if claimed is None:
                break
            video, claim = claimed
            queue_label = f"dynamic_claim={claim.name}"
        else:
            if processed >= len(assigned):
                break
            video = assigned[processed]
            queue_label = f"static={processed + 1}/{len(assigned)}"
        processed += 1
        print(f"[{queue_label}] detecting {video}", flush=True)
        try:
            detect_video(video, output_root, detector, args)
        except Exception as error:  # keep the expensive queue moving
            failures.append({"video": str(video), "error": repr(error)})
            print(f"ERROR video={video} error={error!r}", file=sys.stderr, flush=True)
    atomic_write_json(
        output_root / "worker_status" / f"shard_{args.shard_index:02d}.json",
        {
            "schema": SCHEMA,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "assignment": assignment,
            "queue_videos": len(assigned),
            "processed": processed,
            "exclude_video_id_files": list(args.exclude_video_id_file),
            "excluded_video_ids_present": excluded_present,
            "failures": failures,
            "completed_at_unix": time.time(),
        },
    )
    if failures:
        raise RuntimeError(f"{len(failures)} videos failed on shard {args.shard_index}")


def load_detection_rows(video_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for chunk in sorted((video_dir / "detection_chunks").glob("chunk_*.jsonl")):
        with chunk.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    rows.sort(key=lambda row: int(row["sample_index"]))
    return rows


def _bbox_list(value: Any) -> list[float]:
    if isinstance(value, dict):
        return [float(value[key]) for key in ("x1", "y1", "x2", "y2")]
    return [float(value[index]) for index in range(4)]


def _candidate_for_box(box: Sequence[float], candidates: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    best = None
    bx = (float(box[0]) + float(box[2])) * 0.5
    by = (float(box[1]) + float(box[3])) * 0.5
    for candidate in candidates:
        current = candidate["bbox_xyxy"]
        cx = (float(current[0]) + float(current[2])) * 0.5
        cy = (float(current[1]) + float(current[3])) * 0.5
        distance = math.hypot(cx - bx, cy - by)
        overlap = box_iou(box, current)
        score = overlap - 0.002 * distance
        if overlap >= 0.05 or distance <= 12.0:
            if best is None or score > best[0]:
                best = (score, candidate)
    return None if best is None else best[1]


def track_video(video_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    if str(DET_AND_TRACK) not in sys.path:
        sys.path.insert(0, str(DET_AND_TRACK))
    from GenTrackRes import TrackResGenerator
    from repair_ball_tracks_v2 import mark_detection_quality, repair_points_v2

    metadata = json.loads((video_dir / "metadata.json").read_text(encoding="utf-8"))
    rows = load_detection_rows(video_dir)
    if not rows:
        raise ValueError(f"no detection rows for {video_dir}")
    frame_candidates = {int(row["sample_index"]): row["candidates"] for row in rows}
    tracker_frames = [
        {
            "frame_id": int(row["sample_index"]),
            "objects": [
                {
                    "cls": 1,  # GenTrackRes football class contract
                    "conf": float(candidate["confidence"]),
                    "bbox": candidate["bbox_xyxy"],
                }
                for candidate in row["candidates"]
            ],
        }
        for row in rows
    ]
    effective_fps = float(metadata["effective_sample_fps"])
    generator = TrackResGenerator(
        fps=int(round(effective_fps)),
        min_search=args.track_min_search,
        max_gap=args.track_max_gap,
        ball_conf=args.track_confidence,
        stability_px=args.track_stability_px,
        min_pts=args.track_min_points,
        keep_all=True,
        img_wh=(int(metadata["canonical_width"]), int(metadata["canonical_height"])),
        enable_key_ball=True,
        max_match_radius=args.track_max_match_radius,
        max_merge_dist=args.track_max_merge_distance,
        max_interp_gap=args.track_max_interpolation_gap,
        max_ball_candidates_per_frame=args.track_max_candidates,
        enable_edge_merge=False,
        area_ratio_min=args.track_area_ratio_min,
        area_ratio_max=args.track_area_ratio_max,
    )
    raw_tracks = generator._gen_track_res_from_frames(tracker_frames)
    atomic_write_json(video_dir / "ball_tracks.json", raw_tracks)

    points: dict[int, dict[str, Any]] = {}
    for track in raw_tracks:
        track_id = int(track["track_id"])
        duration = len(track.get("frames", []))
        for frame in track.get("frames", []):
            sample_index = int(frame["frame_id"])
            box = _bbox_list(frame["bbox"])
            candidate = _candidate_for_box(box, frame_candidates.get(sample_index, []))
            confidence = float(candidate["confidence"]) if candidate else 0.0
            proposed = {
                "frame_id": sample_index,
                "bbox": box,
                "track_id": track_id,
                "source": "track_observed" if candidate else "track_interpolation",
                "confidence": confidence,
                "repair_confidence": confidence if candidate else 0.50,
                "usable_for_speed": True,
                "track_duration": duration,
                "sources": candidate.get("sources", []) if candidate else [],
            }
            old = points.get(sample_index)
            score = (candidate is not None, duration, confidence)
            old_score = (
                old is not None and old["source"] == "track_observed",
                old.get("track_duration", 0) if old else 0,
                old.get("confidence", 0.0) if old else 0.0,
            )
            if old is None or score > old_score:
                points[sample_index] = proposed

    base_points = [points[index] for index in sorted(points)]
    raw_by_frame = {
        int(row["sample_index"]): [
            {
                "frame_id": int(row["sample_index"]),
                "bbox": candidate["bbox_xyxy"],
                "conf": float(candidate["confidence"]),
            }
            for candidate in row["candidates"]
        ]
        for row in rows
    }
    quality = mark_detection_quality(base_points, raw_by_frame, effective_fps)
    repaired = repair_points_v2(base_points, raw_by_frame, quality, effective_fps)
    source_by_sample = {int(row["sample_index"]): row for row in rows}
    pseudo_rows = []
    width, height = float(metadata["canonical_width"]), float(metadata["canonical_height"])
    for point in repaired:
        sample_index = int(point["frame_id"])
        source_row = source_by_sample.get(sample_index)
        if source_row is None:
            continue
        box = _bbox_list(point["bbox"])
        candidate = _candidate_for_box(box, frame_candidates.get(sample_index, []))
        source = str(point.get("source", "track"))
        if source == "track" or source == "track_observed":
            source = "track_observed" if candidate else "track_interpolation"
        confidence = float(candidate["confidence"]) if candidate else float(
            point.get("repair_confidence", 0.0)
        )
        if source == "track_observed":
            quality_weight = min(1.0, max(0.25, confidence / 0.30))
        elif source == "raw_detection_tracklet_repair":
            quality_weight = min(0.75, 0.50 * float(point.get("repair_confidence", 0.6)))
        else:
            quality_weight = 0.20
        pseudo_rows.append(
            {
                "sample_index": sample_index,
                "source_frame_id": int(source_row["source_frame_id"]),
                "timestamp_sec": float(source_row["timestamp_sec"]),
                "bbox_xyxy": box,
                "bbox_xyxy_norm": [box[0] / width, box[1] / height, box[2] / width, box[3] / height],
                "center_norm": [(box[0] + box[2]) / (2 * width), (box[1] + box[3]) / (2 * height)],
                "confidence": confidence,
                "quality_weight": quality_weight,
                "source": source,
                "track_id": int(point.get("track_id", 0)),
                "usable_for_heatmap": source != "track_interpolation",
                "usable_for_motion": bool(point.get("usable_for_speed", True)),
                "detector_sources": candidate.get("sources", []) if candidate else [],
            }
        )
    atomic_write_text(
        video_dir / "ball_pseudolabels.jsonl",
        "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in pseudo_rows) + "\n",
    )
    summary = {
        "schema": SCHEMA,
        "video_id": metadata["video_id"],
        "sampled_frames": len(rows),
        "raw_candidate_frames": sum(bool(row["candidates"]) for row in rows),
        "raw_candidates": sum(len(row["candidates"]) for row in rows),
        "tracks": len(raw_tracks),
        "tracked_frames": len(base_points),
        "pseudo_frames": len(pseudo_rows),
        "observed_frames": sum(row["source"] == "track_observed" for row in pseudo_rows),
        "raw_repair_frames": sum(row["source"] == "raw_detection_tracklet_repair" for row in pseudo_rows),
        "interpolated_frames": sum(row["source"] in {"track_interpolation", "tracklet_interpolation"} for row in pseudo_rows),
        "heatmap_usable_frames": sum(row["usable_for_heatmap"] for row in pseudo_rows),
        "motion_usable_frames": sum(row["usable_for_motion"] for row in pseudo_rows),
        "tracker_filter_stats": {
            "total": generator.total_ball_dets,
            "confidence_filtered": generator.conf_filtered_ball_dets,
            "candidate_limited": generator.candidate_limited_ball_dets,
        },
        "completed_at_unix": time.time(),
    }
    atomic_write_json(video_dir / "track_summary.json", summary)
    atomic_write_json(video_dir / "tracking_complete.json", summary)
    return summary


def balanced_tracking_shards(video_dirs: Sequence[Path], count: int) -> list[list[Path]]:
    """Balance tracking workers by sampled-frame count rather than directory inode size."""
    weighted: list[tuple[int, Path]] = []
    for video_dir in video_dirs:
        metadata = json.loads((video_dir / "metadata.json").read_text(encoding="utf-8"))
        source_frames = int(metadata["source_frames"])
        sample_stride = max(1, int(metadata["sample_stride"]))
        weighted.append(((source_frames + sample_stride - 1) // sample_stride, video_dir))
    shards: list[list[Path]] = [[] for _ in range(count)]
    loads = [0] * count
    for weight, video_dir in sorted(weighted, key=lambda item: (-item[0], item[1].name)):
        index = min(range(count), key=lambda value: (loads[value], value))
        shards[index].append(video_dir)
        loads[index] += weight
    return shards


def run_tracking(args: argparse.Namespace) -> None:
    root = Path(args.output_root)
    video_dirs = sorted(path.parent for path in root.glob("*/detection_complete.json"))
    if args.num_shards > 1:
        groups = balanced_tracking_shards(video_dirs, args.num_shards)
        video_dirs = groups[args.shard_index]
    summaries = []
    for index, video_dir in enumerate(video_dirs, start=1):
        if args.skip_existing and (video_dir / "tracking_complete.json").exists():
            continue
        print(f"[{index}/{len(video_dirs)}] tracking {video_dir.name}", flush=True)
        summaries.append(track_video(video_dir, args))
    atomic_write_json(
        root / "tracking_status" / f"shard_{args.shard_index:02d}.json",
        {"schema": SCHEMA, "completed": len(summaries), "summaries": summaries},
    )


def run_audit(args: argparse.Namespace) -> None:
    root = Path(args.output_root)
    summaries = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.glob("*/track_summary.json"))
    ]
    sampled = sum(row["sampled_frames"] for row in summaries)
    report = {
        "schema": SCHEMA,
        "videos": len(summaries),
        "sampled_frames": sampled,
        "raw_candidate_coverage": sum(row["raw_candidate_frames"] for row in summaries) / max(sampled, 1),
        "tracked_coverage": sum(row["tracked_frames"] for row in summaries) / max(sampled, 1),
        "pseudo_coverage": sum(row["pseudo_frames"] for row in summaries) / max(sampled, 1),
        "heatmap_usable_coverage": sum(row["heatmap_usable_frames"] for row in summaries) / max(sampled, 1),
        "motion_usable_coverage": sum(row["motion_usable_frames"] for row in summaries) / max(sampled, 1),
        "observed_frames": sum(row["observed_frames"] for row in summaries),
        "raw_repair_frames": sum(row["raw_repair_frames"] for row in summaries),
        "interpolated_frames": sum(row["interpolated_frames"] for row in summaries),
    }
    atomic_write_json(root / "audit_summary.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-root", required=True)
    common.add_argument("--num-shards", type=int, default=1)
    common.add_argument("--shard-index", type=int, default=0)

    detect = subparsers.add_parser("detect", parents=[common])
    detect.add_argument("--source", required=True)
    detect.add_argument("--checkpoint", required=True)
    detect.add_argument(
        "--exclude-video-id-file",
        action="append",
        default=[],
        help="newline-delimited ids to exclude; repeat for multiple held-out splits",
    )
    detect.add_argument("--device", default="0")
    detect.add_argument(
        "--dynamic-queue",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="atomically claim unfinished videos from one shared queue",
    )
    detect.add_argument("--candidate-confidence", type=float, default=0.03)
    detect.add_argument("--detector-iou", type=float, default=0.70)
    detect.add_argument("--full-image-size", type=int, default=1920)
    detect.add_argument("--horizontal-flip-tta", action=argparse.BooleanOptionalAction, default=True)
    detect.add_argument(
        "--horizontal-flip-missing-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    detect.add_argument("--sliding-windows", action=argparse.BooleanOptionalAction, default=True)
    detect.add_argument("--tile-size", type=int, default=640)
    detect.add_argument("--tile-overlap", type=float, default=0.20)
    detect.add_argument("--tile-image-size", type=int, default=1280)
    detect.add_argument("--merge-iou", type=float, default=0.45)
    detect.add_argument("--max-detections-per-view", type=int, default=20)
    detect.add_argument("--max-candidates", type=int, default=20)
    detect.add_argument("--sample-fps", type=float, default=12.0)
    detect.add_argument("--canonical-width", type=int, default=1280)
    detect.add_argument("--canonical-height", type=int, default=720)
    detect.add_argument("--chunk-seconds", type=float, default=300.0)
    detect.add_argument("--frame-batch-size", type=int, default=4)
    detect.add_argument("--tile-batch-size", type=int, default=16)
    detect.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    detect.add_argument("--max-video-seconds", type=float, default=0.0)

    track = subparsers.add_parser("track", parents=[common])
    track.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    track.add_argument("--track-confidence", type=float, default=0.05)
    track.add_argument("--track-min-search", type=int, default=80)
    track.add_argument("--track-max-gap", type=int, default=12)
    track.add_argument("--track-min-points", type=int, default=4)
    track.add_argument("--track-stability-px", type=float, default=3.0)
    track.add_argument("--track-max-match-radius", type=float, default=160.0)
    track.add_argument("--track-max-merge-distance", type=float, default=160.0)
    track.add_argument("--track-max-interpolation-gap", type=int, default=3)
    track.add_argument("--track-max-candidates", type=int, default=5)
    track.add_argument("--track-area-ratio-min", type=float, default=0.25)
    track.add_argument("--track-area-ratio-max", type=float, default=4.0)

    subparsers.add_parser("audit", parents=[common])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    if args.command == "detect":
        run_detection(args)
    elif args.command == "track":
        run_tracking(args)
    else:
        run_audit(args)


if __name__ == "__main__":
    main()
