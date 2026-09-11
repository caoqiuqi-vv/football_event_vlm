#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch


def iter_json_array(path: Path, chunk_size: int = 4 * 1024 * 1024) -> Iterator[Any]:
    """Stream a top-level JSON array without materializing the whole file."""

    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        position = 0
        eof = False
        started = False
        while True:
            if position > chunk_size:
                buffer = buffer[position:]
                position = 0
            if not eof and len(buffer) - position < chunk_size:
                chunk = handle.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True

            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if not started:
                if position >= len(buffer):
                    if eof:
                        raise ValueError(f"Empty JSON file: {path}")
                    continue
                if buffer[position] != "[":
                    raise ValueError(f"Expected a top-level JSON array: {path}")
                position += 1
                started = True

            while position < len(buffer) and (buffer[position].isspace() or buffer[position] == ","):
                position += 1
            if position < len(buffer) and buffer[position] == "]":
                return
            if position >= len(buffer):
                if eof:
                    raise ValueError(f"Unterminated JSON array: {path}")
                continue
            try:
                item, next_position = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if eof:
                    raise
                chunk = handle.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
                continue
            position = next_position
            yield item


def bbox_values(value: Any) -> list[float] | None:
    if isinstance(value, dict):
        keys = ("x1", "y1", "x2", "y2")
        if all(key in value for key in keys):
            return [float(value[key]) for key in keys]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 4:
        return [float(value[index]) for index in range(4)]
    return None


def read_ids(paths: Sequence[str], explicit_ids: Sequence[str]) -> list[str]:
    ids = {str(video_id).strip() for video_id in explicit_ids if str(video_id).strip()}
    for raw_path in paths:
        path = Path(raw_path)
        ids.update(line.strip().split(",")[0] for line in path.read_text().splitlines() if line.strip())
    return sorted(ids)


def build_index(
    video_dir: Path,
    output_path: Path,
    *,
    sample_fps: float,
    ball_track_fps: float,
    person_conf_floor: float,
    ball_conf_floor: float,
    goal_conf_floor: float,
    center_circle_conf_floor: float,
) -> dict[str, Any]:
    metadata = json.loads((video_dir / "metadata.json").read_text())
    sampling = metadata.get("sampling", {})
    semantics = str(sampling.get("frame_id_semantics", ""))
    if semantics and semantics != "original_source_frame_id":
        raise ValueError(f"{video_dir.name}: expected original_source_frame_id, got {semantics}")
    fps = float(metadata.get("fps", sampling.get("source_fps", 0.0)) or 0.0)
    if fps <= 0:
        raise ValueError(f"{video_dir.name}: invalid fps={fps}")
    image_size = metadata.get("image_size", {})
    width, height = int(image_size.get("width", 0)), int(image_size.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError(f"{video_dir.name}: invalid image_size={image_size}")

    sample_stride = max(int(round(fps / max(sample_fps, 1e-6))), 1)
    ball_stride = max(int(round(fps / max(ball_track_fps, 1e-6))), 1)
    frame_ids: list[int] = []
    frame_offsets: list[int] = [0]
    classes: list[int] = []
    confidences: list[float] = []
    boxes: list[list[float]] = []
    track_ids: list[int] = []
    class_floors = {0: person_conf_floor, 1: ball_conf_floor, 2: goal_conf_floor, 3: center_circle_conf_floor}
    tracked_path = video_dir / "tracked_objects.json"
    for item in iter_json_array(tracked_path):
        if not isinstance(item, dict):
            continue
        frame_id = int(item.get("frame_id", item.get("frame", 0)))
        object_sample = frame_id % sample_stride == 0
        ball_sample = frame_id % ball_stride == 0
        if not object_sample and not ball_sample:
            continue
        kept = 0
        objects = item.get("objects", item.get("detections", [])) or []
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            cls_id = int(obj.get("cls", obj.get("class", -1)))
            if cls_id not in class_floors:
                continue
            if (cls_id == 1 and not ball_sample) or (cls_id != 1 and not object_sample):
                continue
            confidence = float(obj.get("conf", obj.get("score", 0.0)))
            if confidence < class_floors[cls_id]:
                continue
            bbox = bbox_values(obj.get("bbox"))
            if bbox is None or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            classes.append(cls_id)
            confidences.append(confidence)
            boxes.append(bbox)
            track_ids.append(int(obj.get("trackID", obj.get("track_id", -1)) or -1))
            kept += 1
        if kept:
            frame_ids.append(frame_id)
            frame_offsets.append(frame_offsets[-1] + kept)

    ball_stride = max(int(round(fps / max(ball_track_fps, 1e-6))), 1)
    key_ball_path = video_dir / "key_ball_tracks.json"
    ball_tracks = json.loads(key_ball_path.read_text()) if key_ball_path.exists() else []
    ball_track_offsets: list[int] = [0]
    ball_frame_ids: list[int] = []
    ball_boxes: list[list[float]] = []
    ball_point_touched: list[bool] = []
    ball_track_touched: list[bool] = []
    for track in ball_tracks:
        if not isinstance(track, dict):
            continue
        frames = [frame for frame in track.get("frames", []) if isinstance(frame, dict)]
        if not frames:
            continue
        selected: list[dict[str, Any]] = []
        for index, frame in enumerate(frames):
            frame_id = int(frame.get("frame_id", 0))
            touched = bool(frame.get("TouchedPeople", frame.get("touched_people", False)))
            if index in (0, len(frames) - 1) or touched or frame_id % ball_stride == 0:
                selected.append(frame)
        for frame in selected:
            bbox = bbox_values(frame.get("bbox"))
            if bbox is None or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            ball_frame_ids.append(int(frame.get("frame_id", 0)))
            ball_boxes.append(bbox)
            ball_point_touched.append(bool(frame.get("TouchedPeople", frame.get("touched_people", False))))
        if ball_track_offsets[-1] == len(ball_frame_ids):
            continue
        ball_track_offsets.append(len(ball_frame_ids))
        ball_track_touched.append(bool(track.get("touched", False)))

    payload = {
        "version": 2,
        "video_id": video_dir.name,
        "fps": fps,
        "image_size": {"width": width, "height": height},
        "frame_id_semantics": "original_source_frame_id",
        "sample_fps": sample_fps,
        "ball_track_fps": ball_track_fps,
        "frame_ids": torch.tensor(frame_ids, dtype=torch.int32),
        "frame_offsets": torch.tensor(frame_offsets, dtype=torch.int64),
        "classes": torch.tensor(classes, dtype=torch.int8),
        "confidences": torch.tensor(confidences, dtype=torch.float16),
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "track_ids": torch.tensor(track_ids, dtype=torch.int32),
        "ball_track_offsets": torch.tensor(ball_track_offsets, dtype=torch.int64),
        "ball_frame_ids": torch.tensor(ball_frame_ids, dtype=torch.int32),
        "ball_boxes": torch.tensor(ball_boxes, dtype=torch.float32).reshape(-1, 4),
        "ball_point_touched": torch.tensor(ball_point_touched, dtype=torch.bool),
        "ball_track_touched": torch.tensor(ball_track_touched, dtype=torch.bool),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return {
        "video_id": video_dir.name,
        "sampled_frames": len(frame_ids),
        "objects": len(classes),
        "ball_tracks": len(ball_track_offsets) - 1,
        "ball_points": len(ball_frame_ids),
        "output": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compact robust-ROI indices from full detector outputs.")
    parser.add_argument("--detector-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split-file", action="append", default=[])
    parser.add_argument("--video-id", action="append", default=[])
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--ball-track-fps", type=float, default=10.0)
    parser.add_argument("--person-conf-floor", type=float, default=0.20)
    parser.add_argument("--ball-conf-floor", type=float, default=0.10)
    parser.add_argument("--goal-conf-floor", type=float, default=0.20)
    parser.add_argument("--center-circle-conf-floor", type=float, default=0.20)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector_root = Path(args.detector_root)
    output_root = Path(args.output_root)
    requested = read_ids(args.split_file, args.video_id)
    if not requested:
        requested = sorted(path.name for path in detector_root.iterdir() if path.is_dir())
    available = [video_id for video_id in requested if (detector_root / video_id / "tracked_objects.json").exists()]
    missing = sorted(set(requested) - set(available))
    print(f"requested={len(requested)} available={len(available)} missing={len(missing)}", flush=True)
    if missing:
        print("missing_detection_video_ids:", flush=True)
        for video_id in missing:
            print(video_id, flush=True)
    rows = []
    started = time.time()
    for index, video_id in enumerate(available, start=1):
        output_path = output_root / f"{video_id}.pt"
        if output_path.exists() and not args.rebuild:
            try:
                existing = torch.load(output_path, map_location="cpu", weights_only=True, mmap=True)
                current_version = int(existing.get("version", 0)) if isinstance(existing, dict) else 0
            except Exception:
                current_version = 0
            if current_version == 2:
                print(f"[{index}/{len(available)}] skip current v2 video_id={video_id}", flush=True)
                continue
            print(f"[{index}/{len(available)}] rebuild stale index version={current_version} video_id={video_id}", flush=True)
        item_started = time.time()
        row = build_index(
            detector_root / video_id,
            output_path,
            sample_fps=args.sample_fps,
            ball_track_fps=args.ball_track_fps,
            person_conf_floor=args.person_conf_floor,
            ball_conf_floor=args.ball_conf_floor,
            goal_conf_floor=args.goal_conf_floor,
            center_circle_conf_floor=args.center_circle_conf_floor,
        )
        rows.append(row)
        print(
            f"[{index}/{len(available)}] video_id={video_id} frames={row['sampled_frames']} "
            f"objects={row['objects']} ball_tracks={row['ball_tracks']} elapsed={time.time() - item_started:.1f}s",
            flush=True,
        )
    summary = {
        "detector_root": str(detector_root),
        "output_root": str(output_root),
        "requested": requested,
        "available": available,
        "missing": missing,
        "built": rows,
        "elapsed_sec": time.time() - started,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"done built={len(rows)} elapsed={summary['elapsed_sec']:.1f}s summary={output_root / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
