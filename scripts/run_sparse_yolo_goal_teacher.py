#!/usr/bin/env python
"""Run the goal YOLO once over sparse event-window intervals with resume state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO


def chunks(rows: list[tuple[int, float, object]], size: int):
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling-manifest", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=1920)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    manifest = json.loads(Path(args.sampling_manifest).read_text())
    teacher_fps = float(manifest["teacher_fps"])
    model = YOLO(str(Path(args.weights).resolve()))
    goal_ids = [int(key) for key, value in model.names.items() if str(value).casefold() == "goal"]
    if len(goal_ids) != 1:
        raise ValueError(f"checkpoint must expose exactly one goal class: {model.names}")
    goal_id = goal_ids[0]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / f"shard_{args.shard_index:02d}_state.json"
    completed = set()
    if state_path.is_file():
        completed = set(json.loads(state_path.read_text()).get("completed", []))
    for video_index, video in enumerate(manifest["videos"]):
        if video_index % args.num_shards != args.shard_index:
            continue
        video_id = str(video["video_id"])
        video_output = output_root / video_id
        video_output.mkdir(parents=True, exist_ok=True)
        prediction_path = video_output / "goal_regions.jsonl"
        cap = cv2.VideoCapture(str(video["video_path"]))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or frame_count <= 0:
            cap.release()
            raise RuntimeError(f"invalid video: {video['video_path']}")
        stride = max(int(round(fps / teacher_fps)), 1)
        for interval_index, interval in enumerate(video["intervals"]):
            key = f"{video_id}:{interval_index}"
            if key in completed:
                continue
            start = max(0, int(round(float(interval["start_sec"]) * fps)))
            end = min(frame_count - 1, int(round(float(interval["end_sec"]) * fps)))
            selected: list[tuple[int, float, object]] = []
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            frame_index = start
            next_selected = start
            while frame_index <= end:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_index >= next_selected:
                    selected.append((frame_index, frame_index / fps, frame))
                    next_selected += stride
                frame_index += 1
            rows = []
            for batch in chunks(selected, max(args.batch_size, 1)):
                results = model.predict(
                    [item[2] for item in batch],
                    imgsz=args.imgsz,
                    conf=args.confidence,
                    iou=args.iou,
                    device=args.device,
                    verbose=False,
                )
                for (source_frame, timestamp, _), result in zip(batch, results):
                    boxes = result.boxes
                    if boxes is None:
                        continue
                    for cls_id, score, box in zip(
                        boxes.cls.detach().cpu().tolist(),
                        boxes.conf.detach().cpu().tolist(),
                        boxes.xyxy.detach().cpu().tolist(),
                    ):
                        if int(cls_id) == goal_id:
                            rows.append(
                                {
                                    "source_frame_index": source_frame,
                                    "source_timestamp_seconds": timestamp,
                                    "detection_ran": True,
                                    "category_id": goal_id,
                                    "category_name": "goal",
                                    "score": float(score),
                                    "bbox_xyxy": [float(value) for value in box],
                                    "teacher": "yolo11m_person_goal_fieldLine_1920_7.22.pt",
                                }
                            )
            with prediction_path.open("a", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            completed.add(key)
            state_path.write_text(
                json.dumps(
                    {
                        "shard_index": args.shard_index,
                        "num_shards": args.num_shards,
                        "completed": sorted(completed),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n"
            )
            print(
                f"goal shard={args.shard_index}/{args.num_shards} video={video_id} "
                f"interval={interval_index + 1}/{len(video['intervals'])} "
                f"frames={len(selected)} detections={len(rows)}",
                flush=True,
            )
        cap.release()


if __name__ == "__main__":
    main()
