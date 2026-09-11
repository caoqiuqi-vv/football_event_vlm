#!/usr/bin/env python
"""Cache high-resolution DINO features for every online shot candidate.

Rows retain the exact order of the Stage-1 dense NPZ.  Multiple workers can
write disjoint rank shards; no candidate is filtered or temporally merged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_long_video_checkpoint import load_checkpoint_model  # noqa: E402


MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
OFFSETS = tuple(float(value) for value in np.linspace(-1.0, 1.0, 17)) + (
    -8.0, -6.0, -4.0, -2.0, 2.0, 4.0, 6.0, 8.0,
)


def letterbox(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized = cv2.resize(
        frame,
        (max(1, round(frame.shape[1] * scale)), max(1, round(frame.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    top = (height - resized.shape[0]) // 2
    left = (width - resized.shape[1]) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return canvas


def read_frames(
    capture: cv2.VideoCapture,
    center_sec: float,
    duration_sec: float,
    height: int,
    width: int,
) -> torch.Tensor:
    rows = []
    for offset in OFFSETS:
        time_sec = min(max(center_sec + offset, 0.0), max(duration_sec - 1e-3, 0.0))
        capture.set(cv2.CAP_PROP_POS_MSEC, 1000.0 * time_sec)
        ok, frame = capture.read()
        if not ok:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(letterbox(frame, height, width), cv2.COLOR_BGR2RGB)
        rows.append(torch.from_numpy(frame).permute(2, 0, 1))
    values = torch.stack(rows).float().div_(255.0)
    return (values - MEAN) / STD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--video-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--candidate-batch-size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.rank < args.world_size:
        raise ValueError("rank must be in [0, world_size)")
    cache = np.load(args.cache, allow_pickle=True)
    labels = [str(value) for value in cache["labels"]]
    shot_index = labels.index("shot")
    video_ids = cache["video_ids"].astype(str)
    candidate_times = cache["candidate_times"][:, shot_index].astype(np.float64)
    indices = np.arange(len(video_ids), dtype=np.int64)
    assigned_videos = sorted(set(video_ids.tolist()))[args.rank::args.world_size]
    selected = indices[np.isin(video_ids, assigned_videos)]
    device = torch.device(args.device)
    model, _, _, _ = load_checkpoint_model(str(args.checkpoint), device, [device.index or 0])
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    backbone = model.backbone
    backbone.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"visual_rank{args.rank:02d}_of_{args.world_size:02d}.npz"
    features: list[np.ndarray] = []
    output_indices: list[int] = []
    with torch.inference_mode():
        for video_number, video_id in enumerate(assigned_videos):
            video_path = args.video_root / f"{video_id}.mp4"
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise FileNotFoundError(video_path)
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
            duration = frames / max(fps, 1e-6)
            rows = selected[video_ids[selected] == video_id]
            for begin in range(0, len(rows), args.candidate_batch_size):
                batch_rows = rows[begin:begin + args.candidate_batch_size]
                values = torch.cat([
                    read_frames(
                        capture, float(candidate_times[index]), duration,
                        args.height, args.width,
                    )
                    for index in batch_rows
                ], dim=0).to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    result = backbone.forward_features(values)
                    if isinstance(result, list):
                        result = result[0]
                    cls = result["x_norm_clstoken"]
                    patch_mean = result["x_norm_patchtokens"].mean(dim=1)
                    encoded = torch.cat((cls, patch_mean), dim=-1)
                encoded = encoded.reshape(len(batch_rows), len(OFFSETS), -1)
                features.extend(encoded.float().cpu().numpy().astype(np.float16))
                output_indices.extend(int(value) for value in batch_rows)
            capture.release()
            print(json.dumps({
                "rank": args.rank,
                "video": video_id,
                "video_progress": [video_number + 1, len(assigned_videos)],
                "candidates": len(rows),
            }), flush=True)
    np.savez_compressed(
        output_path,
        indices=np.asarray(output_indices, dtype=np.int64),
        visual_features=np.stack(features),
        offsets=np.asarray(OFFSETS, dtype=np.float32),
        checkpoint=str(args.checkpoint),
    )
    (args.output_dir / f"visual_rank{args.rank:02d}.done.json").write_text(
        json.dumps({
            "rank": args.rank,
            "world_size": args.world_size,
            "rows": len(output_indices),
            "videos": assigned_videos,
            "output": str(output_path),
            "no_temporal_nms": True,
        }, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
