#!/usr/bin/env python
"""Run a single-class YOLO football Teacher over sparse manifest intervals."""

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
    parser.add_argument("--device", default="0")
    parser.add_argument("--img-height", type=int, default=1088)
    parser.add_argument("--img-width", type=int, default=1920)
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-new-intervals", type=int, default=0)
    parser.add_argument(
        "--video-path-override",
        action="append",
        default=[],
        metavar="VIDEO_ID=PATH",
        help="Optionally read selected videos from a local cache without changing manifest keys.",
    )
    args = parser.parse_args()

    video_path_overrides = {}
    for item in args.video_path_override:
        video_id, separator, path = item.partition("=")
        if not separator or not video_id or not path:
            raise ValueError(f"invalid --video-path-override: {item!r}")
        video_path_overrides[video_id] = path

    manifest = json.loads(Path(args.sampling_manifest).read_text())
    teacher_fps = float(manifest["teacher_fps"])
    model = YOLO(str(Path(args.weights).resolve()))
    football_ids = [
        int(key)
        for key, value in model.names.items()
        if str(value).casefold() in {"football", "soccer_ball", "soccer ball", "ball"}
    ]
    if len(football_ids) != 1:
        raise ValueError(f"checkpoint must expose exactly one football class: {model.names}")
    football_id = football_ids[0]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / f"shard_{args.shard_index:02d}_state.json"
    completed: set[str] = set()
    if state_path.is_file():
        completed = set(json.loads(state_path.read_text()).get("completed", []))
    new_intervals = 0
    for video_index, video in enumerate(manifest["videos"]):
        if video_index % args.num_shards != args.shard_index:
            continue
        video_id = str(video["video_id"])
        video_output = output_root / video_id
        video_output.mkdir(parents=True, exist_ok=True)
        prediction_path = video_output / "football_predictions.jsonl"
        video_path = video_path_overrides.get(video_id, str(video["video_path"]))
        cap = cv2.VideoCapture(video_path)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0 or frame_count <= 0:
            cap.release()
            raise RuntimeError(f"invalid video: {video_path}")
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
            if not selected:
                cap.release()
                raise RuntimeError(
                    f"decoded zero frames: video_id={video_id} interval_index={interval_index} start={start} end={end}"
                )
            rows = []
            for batch in chunks(selected, max(args.batch_size, 1)):
                results = model.predict(
                    [item[2] for item in batch],
                    imgsz=(args.img_height, args.img_width),
                    conf=args.confidence,
                    iou=args.iou,
                    device=args.device,
                    half=bool(args.half),
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
                        if int(cls_id) == football_id:
                            rows.append(
                                {
                                    "source_frame_index": source_frame,
                                    "source_timestamp_seconds": timestamp,
                                    "detection_ran": True,
                                    "category_id": football_id,
                                    "category_name": "football",
                                    "score": float(score),
                                    "bbox_xyxy": [float(value) for value in box],
                                    "teacher": Path(args.weights).name,
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
            new_intervals += 1
            print(
                f"ball-yolo shard={args.shard_index}/{args.num_shards} video={video_id} "
                f"interval={interval_index + 1}/{len(video['intervals'])} "
                f"frames={len(selected)} detections={len(rows)}",
                flush=True,
            )
            if args.max_new_intervals and new_intervals >= args.max_new_intervals:
                cap.release()
                return
        cap.release()


if __name__ == "__main__":
    main()
