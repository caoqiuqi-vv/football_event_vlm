#!/usr/bin/env python
"""Score every Stage-1 shot candidate with a complete event-centred clip model.

Unlike ``extract_online_verifier_visual_features.py``, this script runs the
checkpoint's trained temporal head as well as its DINO backbone.  Each
candidate is evaluated independently at a small grid of window-centre shifts;
rows are never filtered, merged, or suppressed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_football_events as train_mod  # noqa: E402
from scripts.eval_long_video_checkpoint import (  # noqa: E402
    SlidingWindowVideoDataset,
    WindowRecord,
    collate_windows,
    load_checkpoint_model,
)
from train_football_events import parse_image_size  # noqa: E402


def parse_float_list(value: str) -> tuple[float, ...]:
    result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one offset is required")
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("offsets must be unique")
    return result


def video_duration(path: Path) -> float:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(path)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    capture.release()
    if fps <= 0.0 or frames <= 0.0:
        raise RuntimeError(f"invalid video metadata: {path}")
    return frames / fps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--video-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--offsets-sec", type=parse_float_list, default=(-2.0, 0.0, 2.0))
    parser.add_argument("--clip-sec", type=float, default=10.0)
    parser.add_argument("--image-size", default="")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--decode-strategy", default="single_seek", choices=("single_seek", "multi_seek"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.rank < args.world_size:
        raise ValueError("rank must be in [0, world_size)")
    if args.clip_sec <= 0.0:
        raise ValueError("clip-sec must be positive")

    cache = np.load(args.cache, allow_pickle=True)
    stage_labels = [str(value) for value in cache["labels"]]
    shot_index = stage_labels.index("shot")
    video_ids = cache["video_ids"].astype(str)
    candidate_times = cache["candidate_times"][:, shot_index].astype(np.float64)
    all_indices = np.arange(len(video_ids), dtype=np.int64)
    assigned_videos = sorted(set(video_ids.tolist()))[args.rank::args.world_size]

    device = torch.device(args.device)
    model, cfg, teacher_labels, _ = load_checkpoint_model(
        str(args.checkpoint), device, [device.index or 0]
    )
    image_size = (
        parse_image_size(args.image_size)
        if args.image_size
        else parse_image_size(cfg.video.image_size)
    )
    num_frames = train_mod.effective_num_frames(cfg)
    offsets = tuple(float(value) for value in args.offsets_sec)

    output_indices: list[int] = []
    teacher_logits: list[np.ndarray] = []
    actual_centres: list[np.ndarray] = []
    for video_number, video_id in enumerate(assigned_videos):
        path = args.video_root / f"{video_id}.mp4"
        duration = video_duration(path)
        rows = all_indices[video_ids == video_id]
        windows: list[WindowRecord] = []
        packed_to_row_offset: list[tuple[int, int, float]] = []
        for row in rows:
            for offset_index, offset in enumerate(offsets):
                requested_center = float(candidate_times[row]) + offset
                start = min(max(requested_center - 0.5 * args.clip_sec, 0.0), max(duration - args.clip_sec, 0.0))
                end = min(start + args.clip_sec, duration)
                packed_index = len(packed_to_row_offset)
                windows.append(WindowRecord(packed_index, start, end))
                packed_to_row_offset.append((int(row), offset_index, 0.5 * (start + end)))

        dataset = SlidingWindowVideoDataset(
            video_path=str(path),
            video_id=video_id,
            windows=windows,
            num_frames=num_frames,
            image_size=image_size,
            normalize_on_cpu=False,
            decode_strategy=args.decode_strategy,
            video_reader_cache_size=2,
            view_mode="single",
            global_image_size=image_size,
            dual_sampling="aligned",
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
            prefetch_factor=1 if args.num_workers > 0 else None,
            collate_fn=collate_windows,
        )
        per_row = np.empty((len(rows), len(offsets), len(teacher_labels)), dtype=np.float32)
        per_row_centres = np.empty((len(rows), len(offsets)), dtype=np.float32)
        row_to_local = {int(row): index for index, row in enumerate(rows)}
        with torch.inference_mode():
            for batch in loader:
                with train_mod.autocast_context(device, bool(cfg.train.amp), str(cfg.train.amp_dtype)):
                    outputs = train_mod.forward_model_batch(model, batch, device, return_aux=True)
                logits = (
                    outputs.get("temporal_logits", outputs["logits"])
                    if isinstance(outputs, dict)
                    else outputs
                )
                values = logits.float().cpu().numpy()
                for batch_index, meta in enumerate(batch["meta"]):
                    row, offset_index, actual_center = packed_to_row_offset[int(meta["index"])]
                    per_row[row_to_local[row], offset_index] = values[batch_index]
                    per_row_centres[row_to_local[row], offset_index] = actual_center
        output_indices.extend(int(value) for value in rows)
        teacher_logits.extend(per_row)
        actual_centres.extend(per_row_centres)
        print(json.dumps({
            "rank": args.rank,
            "video": video_id,
            "video_progress": [video_number + 1, len(assigned_videos)],
            "candidates": len(rows),
            "clips": len(windows),
        }), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"teacher_rank{args.rank:02d}_of_{args.world_size:02d}.npz"
    empty_logits = np.empty((0, len(offsets), len(teacher_labels)), dtype=np.float16)
    empty_centres = np.empty((0, len(offsets)), dtype=np.float32)
    np.savez_compressed(
        output_path,
        indices=np.asarray(output_indices, dtype=np.int64),
        teacher_logits=(np.stack(teacher_logits).astype(np.float16) if teacher_logits else empty_logits),
        actual_window_centres=(np.stack(actual_centres).astype(np.float32) if actual_centres else empty_centres),
        offsets=np.asarray(offsets, dtype=np.float32),
        labels=np.asarray(teacher_labels),
        checkpoint=str(args.checkpoint),
        clip_sec=float(args.clip_sec),
        image_size=np.asarray(image_size, dtype=np.int64),
    )
    (args.output_dir / f"teacher_rank{args.rank:02d}.done.json").write_text(
        json.dumps({
            "rank": args.rank,
            "world_size": args.world_size,
            "rows": len(output_indices),
            "videos": assigned_videos,
            "output": str(output_path),
            "offsets_sec": offsets,
            "checkpoint": str(args.checkpoint),
            "complete_model_with_temporal_head": True,
            "no_temporal_nms": True,
        }, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
