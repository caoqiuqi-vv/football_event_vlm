from __future__ import annotations

"""Build model-free, fixed-duration candidate chunks for verifier baselines.

Unlike sliding windows, the regular part of the video is partitioned into
non-overlapping chunks and each chunk contributes exactly one centre anchor.
The final short remainder contributes one midpoint anchor.  This gives an
honest lower-complexity control for learned proposal streams.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from football_longform_v2.canonical import load_canonical_split  # noqa: E402
from football_longform_v2.config import load_config  # noqa: E402
from football_longform_v2.schema import read_video_ids  # noqa: E402


SCHEMA = "football_longform_v2.shared_candidate_manifest.v1"


def resolve(root: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=("train", "calibration"), required=True)
    parser.add_argument("--chunk-seconds", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.chunk_seconds <= 0:
        raise ValueError("--chunk-seconds must be positive")

    config_path = Path(args.config).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    config = load_config(str(config_path))
    root = Path(config["_project_root"])
    canonical = load_canonical_split(
        resolve(root, str(config["paths"]["canonical_manifest"])), args.split
    )
    id_key = "train_ids" if args.split == "train" else "calibration_ids"
    video_ids = read_video_ids(resolve(root, str(config["paths"][id_key])))
    if tuple(video_ids) != tuple(canonical.media_ids):
        raise RuntimeError("canonical manifest and split ID list disagree")
    feature_root = resolve(root, str(config["paths"]["feature_store"])) / args.split
    labels = [str(label) for label in config["task"]["output_labels"]]

    videos = []
    for video_id in video_ids:
        # Reading the full aligned cache would decompress multi-GB context and
        # motion arrays just to obtain duration.  NumPy loads only the selected
        # member of the NPZ archive here.
        with np.load(feature_root / video_id / "timeline.npz", allow_pickle=False) as payload:
            timestamps = np.asarray(payload["timestamps"], dtype=np.float64)
        duration = float(timestamps[-1]) if timestamps.size else 0.0
        chunk_count = max(int(math.ceil(duration / args.chunk_seconds)), 1)
        candidates = []
        for chunk_index in range(chunk_count):
            start = chunk_index * args.chunk_seconds
            end = min(start + args.chunk_seconds, duration)
            if end < start:
                end = start
            timestamp = 0.5 * (start + end)
            timeline_index = (
                int(np.abs(timestamps - timestamp).argmin())
                if timestamps.size
                else 0
            )
            candidates.append(
                {
                    "candidate_id": f"{video_id}:uniform_{args.chunk_seconds:g}s:{chunk_index:06d}",
                    "timestamp": timestamp,
                    "timeline_index": timeline_index,
                    "source_label": "uniform_chunk",
                    "source_score": 1.0,
                    "source_logit": 0.0,
                    "class_scores": {label: 1.0 for label in labels},
                    "chunk_start": start,
                    "chunk_end": end,
                }
            )
        videos.append(
            {
                "video_id": video_id,
                "source_video": canonical.source_video_by_media_id[video_id],
                "duration_seconds": duration,
                "candidate_count": len(candidates),
                "candidates_per_minute": len(candidates)
                / max(duration / 60.0, 1e-9),
                "candidates": candidates,
            }
        )

    total_duration = sum(float(video["duration_seconds"]) for video in videos)
    candidate_count = sum(len(video["candidates"]) for video in videos)
    result = {
        "schema_version": SCHEMA,
        "split": args.split,
        "strategy": "uniform_nonoverlap_chunk_centres",
        "chunk_seconds": float(args.chunk_seconds),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "checkpoint": None,
        "checkpoint_sha256": None,
        "nms_radius_seconds": None,
        "labels": labels,
        "shard_index": 0,
        "num_shards": 1,
        "video_count": len(videos),
        "candidate_count": candidate_count,
        "duration_seconds": total_duration,
        "candidates_per_minute": candidate_count
        / max(total_duration / 60.0, 1e-9),
        "video_ids": sorted(video_ids),
        "videos": videos,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "split": args.split,
                "chunk_seconds": args.chunk_seconds,
                "video_count": len(videos),
                "candidate_count": candidate_count,
                "candidates_per_minute": result["candidates_per_minute"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
