#!/usr/bin/env python
"""Build a 1 Hz DINO appearance + camera-residual motion + audio bank.

Run one process per GPU with disjoint ``--shard-index`` values.  This is a
global retrieval stream; the later action examiner still reads dense 720p
frames around retained regions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import lmdb
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PKG = ROOT / "football_e2e_spotter" / "src"
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

from scripts.eval_long_video_checkpoint import load_checkpoint_model  # noqa: E402


SCHEMA = "football.goal_feature_bank.v1"


def read_rgb(transaction: lmdb.Transaction, index: int) -> np.ndarray:
    value = transaction.get(f"f/{index:08d}".encode())
    if value is None:
        raise KeyError(f"missing frame {index}")
    bgr = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"corrupt JPEG at frame {index}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def residual_motion(previous: np.ndarray | None, current: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (160, 96), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    if previous is None:
        return np.zeros(64, dtype=np.float32)
    shift, response = cv2.phaseCorrelate(previous, gray)
    # Reject unstable estimates; in that case raw difference is safer than a
    # destructive warp.  Translation is intentionally detector-independent.
    if not np.isfinite(shift).all() or response < 0.02 or abs(shift[0]) > 48 or abs(shift[1]) > 29:
        aligned = previous
    else:
        transform = np.asarray([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]], dtype=np.float32)
        aligned = cv2.warpAffine(previous, transform, (gray.shape[1], gray.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    difference = cv2.absdiff(gray, aligned)
    # An 8x8 residual grid retains where independently moving players/ball are,
    # unlike a scalar optical-flow magnitude.
    return cv2.resize(difference, (8, 8), interpolation=cv2.INTER_AREA).reshape(-1).astype(np.float32)


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def build_one(
    item: dict,
    *,
    pixel_root: Path,
    output_root: Path,
    split: str,
    model: torch.nn.Module,
    device: torch.device,
    chunk_frames: int,
) -> dict:
    video_id = str(item["media_id"])
    source = pixel_root / split / video_id
    target_dir = output_root / split / video_id
    target = target_dir / "timeline.npz"
    metadata_path = source / "metadata.json"
    if not metadata_path.is_file():
        return {"video_id": video_id, "status": "pixel_cache_missing"}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if target.is_file():
        try:
            with np.load(target, allow_pickle=False) as cached:
                if str(cached["schema"].item()) == SCHEMA and str(cached["checkpoint"].item()) == str(args.checkpoint):
                    return {"video_id": video_id, "status": "skip_ready", "steps": int(cached["timestamps"].shape[0])}
        except Exception:
            pass
    source_fps = float(metadata["sample_fps"])
    if abs(source_fps - 4.0) > 1e-6:
        raise ValueError(f"expected 4 FPS cache, got {source_fps}")
    frame_count = int(metadata["frame_count"])
    indices = np.arange(0, frame_count, 4, dtype=np.int64)
    audio_4hz = np.load(source / "audio_logmel.npy", mmap_mode="r", allow_pickle=False)
    environment = lmdb.open(str(source / "frames.lmdb"), readonly=True, lock=False, readahead=False, max_readers=8)
    network = unwrap(model)
    appearance_parts: list[np.ndarray] = []
    motion_parts: list[np.ndarray] = []
    previous_gray: np.ndarray | None = None
    started = time.time()
    with environment.begin(buffers=True) as transaction:
        for begin in range(0, len(indices), chunk_frames):
            selected = indices[begin:begin + chunk_frames]
            frames = [read_rgb(transaction, int(index)) for index in selected]
            motion = []
            for frame in frames:
                motion.append(residual_motion(previous_gray, frame))
                previous_gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (160, 96), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            tensor = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).unsqueeze(0).to(device, non_blocking=True)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                features = network.encode_frames(tensor).squeeze(0)
            appearance_parts.append(features.float().cpu().numpy().astype(np.float16))
            motion_parts.append(np.stack(motion).astype(np.float16))
    environment.close()
    appearance = np.concatenate(appearance_parts, axis=0)
    motion = np.concatenate(motion_parts, axis=0)
    audio = np.zeros((len(indices), audio_4hz.shape[1]), dtype=np.float32)
    for output_index, source_index in enumerate(indices):
        audio[output_index] = np.asarray(audio_4hz[source_index:min(source_index + 4, len(audio_4hz))], dtype=np.float32).mean(axis=0)
    timestamps = (indices.astype(np.float32) + 0.5) / source_fps
    target_dir.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema=np.asarray(SCHEMA), checkpoint=np.asarray(str(args.checkpoint)),
            video_id=np.asarray(video_id), timestamps=timestamps,
            appearance=appearance, motion=motion, audio=audio.astype(np.float16),
            source_fps=np.asarray(source_fps, dtype=np.float32), feature_fps=np.asarray(1.0, dtype=np.float32),
            annotation_path=np.asarray(str(item["annotation_path"])),
        )
    os.replace(temporary, target)
    return {
        "video_id": video_id, "status": "built", "steps": len(indices),
        "appearance_dim": int(appearance.shape[1]), "seconds": round(time.time() - started, 2),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pixel-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "calibration"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--chunk-frames", type=int, default=12)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard")
    device = torch.device(f"cuda:{args.gpu}")
    model, _cfg, _labels, _thresholds = load_checkpoint_model(str(args.checkpoint.resolve()), device, [args.gpu])
    items = json.loads(args.manifest.read_text(encoding="utf-8"))[args.split]
    rows = []
    for item in items[args.shard_index::args.num_shards]:
        row = build_one(
            item, pixel_root=args.pixel_root.resolve(), output_root=args.output_root.resolve(),
            split=args.split, model=model, device=device, chunk_frames=args.chunk_frames,
        )
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    report = args.output_root / f"build_{args.split}_shard{args.shard_index}.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"schema": SCHEMA, "rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"report={report}", flush=True)

