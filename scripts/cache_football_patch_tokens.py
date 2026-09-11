#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_football_events as train_mod
from scripts.eval_long_video_checkpoint import load_checkpoint_model


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def find_video(video_root: Path, video_id: str) -> Path:
    for suffix in (".mp4", ".mov", ".mkv", ".avi"):
        path = video_root / f"{video_id}{suffix}"
        if path.exists():
            return path
    matches = sorted(video_root.glob(f"{video_id}.*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Missing video_id={video_id} under {video_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache A-best DINO CLS and raw patch tokens for TP/FP ceiling experiments.")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-frames", type=int, default=20)
    parser.add_argument("--frame-chunk", type=int, default=2)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-windows", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases_path = Path(args.cases).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    video_root = Path(args.video_root).expanduser().resolve()
    device = torch.device(args.device)

    case_rows = [row for row in read_csv(cases_path) if row["group"] in {"TP", "FP"}]
    unique: dict[tuple[str, int], dict[str, Any]] = {}
    for row in case_rows:
        key = (row["video_id"], int(row["window_index"]))
        unique[key] = {
            "video_id": row["video_id"],
            "window_index": int(row["window_index"]),
            "start_sec": float(row["start_sec"]),
            "end_sec": float(row["end_sec"]),
        }
    windows = sorted(unique.values(), key=lambda item: (item["video_id"], item["window_index"]))
    if args.max_windows > 0:
        windows = windows[: args.max_windows]

    model, cfg, labels, _ = load_checkpoint_model(args.checkpoint, device, [])
    model.eval()
    if model.backbone is None:
        raise RuntimeError("Checkpoint model has no DINO backbone")
    image_size = train_mod.parse_image_size(cfg.video.image_size)
    manifest_rows: list[dict[str, Any]] = []
    started = time.time()

    with torch.inference_mode():
        for position, window in enumerate(windows, start=1):
            relative = Path(window["video_id"]) / f"w{window['window_index']:05d}.pt"
            cache_path = output_dir / relative
            if cache_path.exists() and not args.force:
                cached = torch.load(cache_path, map_location="cpu", weights_only=False)
                manifest_rows.append(
                    {
                        **window,
                        "cache_path": relative.as_posix(),
                        "num_frames": int(cached["patches"].shape[0]),
                        "num_patches": int(cached["patches"].shape[1]),
                        "feature_dim": int(cached["patches"].shape[2]),
                    }
                )
                continue
            frames, frame_times = train_mod.read_video_segment(
                str(find_video(video_root, window["video_id"])),
                args.num_frames,
                image_size,
                False,
                0.0,
                start_sec=window["start_sec"],
                end_sec=window["end_sec"],
                normalize=True,
                decode_strategy="single_seek",
                return_frame_times=True,
            )
            cls_parts: list[torch.Tensor] = []
            patch_parts: list[torch.Tensor] = []
            for start in range(0, args.num_frames, max(args.frame_chunk, 1)):
                chunk = frames[start : start + args.frame_chunk].to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    features = model.backbone.forward_features(chunk)
                cls_parts.append(features["x_norm_clstoken"].detach().to(device="cpu", dtype=torch.float16))
                patch_parts.append(features["x_norm_patchtokens"].detach().to(device="cpu", dtype=torch.float16))
            payload = {
                **window,
                "image_size": list(image_size),
                "frame_times": frame_times.float(),
                "cls": torch.cat(cls_parts, dim=0),
                "patches": torch.cat(patch_parts, dim=0),
                "labels": list(labels),
            }
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, cache_path)
            manifest_rows.append(
                {
                    **window,
                    "cache_path": relative.as_posix(),
                    "num_frames": int(payload["patches"].shape[0]),
                    "num_patches": int(payload["patches"].shape[1]),
                    "feature_dim": int(payload["patches"].shape[2]),
                }
            )
            elapsed = time.time() - started
            print(
                f"cache {position}/{len(windows)} video={window['video_id']} w={window['window_index']} "
                f"shape={tuple(payload['patches'].shape)} elapsed={elapsed:.1f}s",
                flush=True,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    manifest = {
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "cases": str(cases_path),
        "video_root": str(video_root),
        "image_size": list(image_size),
        "num_frames": args.num_frames,
        "num_windows": len(manifest_rows),
        "windows": manifest_rows,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({key: manifest[key] for key in ("image_size", "num_frames", "num_windows")}, indent=2))
    print(f"wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
