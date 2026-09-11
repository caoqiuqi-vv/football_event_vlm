#!/usr/bin/env python
"""Merge ball and goal Teacher outputs into provenance-rich v3 indices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import torch


def jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                row = json.loads(line)
                if isinstance(row, dict):
                    yield row


def first(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if row.get(key) is not None:
            return row[key]
    return None


def xyxy(
    row: dict[str, Any],
    *,
    xywh_default: bool,
    center_xywh: bool = False,
) -> list[float] | None:
    value = first(row, "bbox_xyxy", "xyxy", "bbox", "xywh")
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    x1, y1, a, b = (float(value[index]) for index in range(4))
    key_is_xywh = "xywh" in row or (
        "bbox_xyxy" not in row and "xyxy" not in row and xywh_default
    )
    if key_is_xywh and center_xywh:
        x1, y1, x2, y2 = x1 - a / 2.0, y1 - b / 2.0, x1 + a / 2.0, y1 + b / 2.0
    else:
        x2, y2 = (x1 + a, y1 + b) if key_is_xywh else (a, b)
    return [x1, y1, x2, y2] if x2 > x1 and y2 > y1 else None


def read_detections(
    path: Path,
    fps: float,
    *,
    xywh_default: bool,
    center_xywh: bool = False,
) -> list[tuple[int, float, list[float]]]:
    rows: list[tuple[int, float, list[float]]] = []
    for row in jsonl(path):
        frame = first(row, "source_frame_index", "frame_index", "frame_id", "frame")
        if frame is None:
            timestamp = first(row, "source_timestamp_seconds", "timestamp_seconds", "timestamp")
            if timestamp is None:
                continue
            frame = round(float(timestamp) * fps)
        box = xyxy(row, xywh_default=xywh_default, center_xywh=center_xywh)
        if box is None:
            continue
        score = float(first(row, "score", "confidence", "conf") or 0.0)
        rows.append((int(frame), score, box))
    return rows


def resolve_json(root: Path, video_id: str, filename: str) -> Path:
    direct = root / f"{video_id}.jsonl"
    if direct.is_file():
        return direct
    candidates = sorted((root / video_id).rglob(filename)) if (root / video_id).exists() else []
    if not candidates:
        raise FileNotFoundError(f"missing {filename} for video_id={video_id} under {root}")
    if len(candidates) == 1:
        return candidates[0]
    # Segmented inference is concatenated in stable path order.
    merged = root / ".merged" / f"{video_id}_{filename}"
    merged.parent.mkdir(parents=True, exist_ok=True)
    with merged.open("w", encoding="utf-8") as output:
        for candidate in candidates:
            output.write(candidate.read_text(encoding="utf-8"))
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling-manifest", required=True)
    parser.add_argument("--ball-root", required=True)
    parser.add_argument("--goal-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--ball-filename", default="football_predictions.jsonl")
    parser.add_argument("--goal-filename", default="goal_regions.jsonl")
    parser.add_argument("--ball-checkpoint", required=True)
    parser.add_argument("--goal-checkpoint", required=True)
    parser.add_argument("--ball-teacher-name", default="YOLO football")
    parser.add_argument("--goal-teacher-name", default="YOLO11 person/goal/fieldLine")
    parser.add_argument("--ball-confidence", type=float, default=0.10)
    parser.add_argument("--goal-confidence", type=float, default=0.25)
    args = parser.parse_args()

    manifest_path = Path(args.sampling_manifest)
    manifest = json.loads(manifest_path.read_text())
    teacher_fps = float(manifest["teacher_fps"])
    ball_root, goal_root = Path(args.ball_root), Path(args.goal_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for video in manifest["videos"]:
        video_id = str(video["video_id"])
        cap = cv2.VideoCapture(str(video["video_path"]))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        if fps <= 0 or width <= 0 or height <= 0 or frame_count <= 0:
            raise RuntimeError(f"invalid video geometry: {video['video_path']}")
        stride = max(int(round(fps / teacher_fps)), 1)
        sampled: set[int] = set()
        for interval in video["intervals"]:
            start = max(0, int(round(float(interval["start_sec"]) * fps)))
            end = min(frame_count - 1, int(round(float(interval["end_sec"]) * fps)))
            sampled.update(range(start, end + 1, stride))
        frame_ids = sorted(sampled)
        ball_path = resolve_json(ball_root, video_id, args.ball_filename)
        goal_path = resolve_json(goal_root, video_id, args.goal_filename)
        detections: dict[int, list[tuple[int, float, list[float]]]] = {
            frame: [] for frame in frame_ids
        }
        sampled_tensor = torch.tensor(frame_ids, dtype=torch.int64)
        for class_id, floor, rows in (
            (
                1,
                args.ball_confidence,
                read_detections(ball_path, fps, xywh_default=True, center_xywh=True),
            ),
            (2, args.goal_confidence, read_detections(goal_path, fps, xywh_default=False)),
        ):
            for frame, score, box in rows:
                if score < floor or not len(sampled_tensor):
                    continue
                insertion = int(torch.searchsorted(sampled_tensor, torch.tensor(frame)).item())
                candidates = [i for i in (insertion - 1, insertion) if 0 <= i < len(frame_ids)]
                if not candidates:
                    continue
                nearest = min(candidates, key=lambda index: abs(frame_ids[index] - frame))
                if abs(frame_ids[nearest] - frame) <= max(stride // 2, 1):
                    detections[frame_ids[nearest]].append((class_id, score, box))
        classes: list[int] = []
        confidences: list[float] = []
        boxes: list[list[float]] = []
        offsets = [0]
        for frame in frame_ids:
            for class_id, score, box in detections[frame]:
                classes.append(class_id)
                confidences.append(score)
                boxes.append(box)
            offsets.append(len(classes))
        payload = {
            "version": 3,
            "video_id": video_id,
            "fps": fps,
            "image_size": {"width": width, "height": height},
            "frame_id_semantics": "original_source_frame_id",
            "sample_fps": teacher_fps,
            "frame_ids": torch.tensor(frame_ids, dtype=torch.int32),
            "frame_offsets": torch.tensor(offsets, dtype=torch.int64),
            "classes": torch.tensor(classes, dtype=torch.int8),
            "confidences": torch.tensor(confidences, dtype=torch.float16),
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "provenance": {
                "sampling_manifest": str(manifest_path.resolve()),
                "ball_teacher": args.ball_teacher_name,
                "ball_checkpoint": str(Path(args.ball_checkpoint).resolve()),
                "ball_output": str(ball_path.resolve()),
                "goal_teacher": args.goal_teacher_name,
                "goal_checkpoint": str(Path(args.goal_checkpoint).resolve()),
                "goal_output": str(goal_path.resolve()),
                "ball_confidence": args.ball_confidence,
                "goal_confidence": args.goal_confidence,
            },
        }
        output = output_root / f"{video_id}.pt"
        torch.save(payload, output)
        summaries.append({"video_id": video_id, "frames": len(frame_ids), "objects": len(classes)})
        print(f"[{len(summaries)}/{len(manifest['videos'])}] {video_id} frames={len(frame_ids)} objects={len(classes)}", flush=True)
    summary = {
        "version": 3,
        "video_count": len(summaries),
        "frame_count": sum(row["frames"] for row in summaries),
        "object_count": sum(row["objects"] for row in summaries),
        "ball_checkpoint": str(Path(args.ball_checkpoint).resolve()),
        "goal_checkpoint": str(Path(args.goal_checkpoint).resolve()),
        "videos": summaries,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
