#!/usr/bin/env python
"""Sharded DINO feature-bank builder with mmap-ready array finalization."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

import build_goal_feature_bank as base


def atomic_npy(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(temporary, path)


def finalize(directory: Path, checkpoint: str, annotation_path: str) -> dict:
    timeline_path = directory / "timeline.npz"
    with np.load(timeline_path, allow_pickle=False) as payload:
        values = {name: np.asarray(payload[name]) for name in ("timestamps", "appearance", "motion", "audio")}
    for name, value in values.items():
        atomic_npy(directory / f"{name}.npy", value)
    metadata = {
        "schema": base.SCHEMA, "checkpoint": checkpoint,
        "video_id": directory.name, "steps": int(values["timestamps"].shape[0]),
        "appearance_dim": int(values["appearance"].shape[1]),
        "motion_dim": int(values["motion"].shape[1]),
        "audio_dim": int(values["audio"].shape[1]),
        "source_fps": 4.0, "feature_fps": 1.0,
        "annotation_path": annotation_path,
    }
    target = directory / "metadata.json"
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    return metadata


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
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard")
    # Legacy-compatible worker reads this global when checking cache identity.
    base.args = args
    device = torch.device(f"cuda:{args.gpu}")
    model, _cfg, _labels, _thresholds = base.load_checkpoint_model(str(args.checkpoint.resolve()), device, [args.gpu])
    items = json.loads(args.manifest.read_text(encoding="utf-8"))[args.split]
    items = items[args.shard_index::args.num_shards]
    if args.limit > 0:
        items = items[:args.limit]
    rows = []
    for item in items:
        video_id = str(item["media_id"])
        row = base.build_one(
            item, pixel_root=args.pixel_root.resolve(), output_root=args.output_root.resolve(),
            split=args.split, model=model, device=device, chunk_frames=args.chunk_frames,
        )
        directory = args.output_root.resolve() / args.split / video_id
        if (directory / "timeline.npz").is_file():
            metadata = finalize(directory, str(args.checkpoint.resolve()), str(item["annotation_path"]))
            row["mmap_finalized"] = True
            row["appearance_dim"] = metadata["appearance_dim"]
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    report = args.output_root / f"build_{args.split}_shard{args.shard_index}.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"schema": base.SCHEMA, "rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"report={report}", flush=True)


if __name__ == "__main__":
    main()

