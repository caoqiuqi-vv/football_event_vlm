from __future__ import annotations

"""Export the leakage-safe A0 shared event proposal stream.

The exporter is intentionally limited to canonical ``train`` and
``calibration`` splits.  It never reads the sealed third-party test split and
it never writes ground-truth labels into the proposal manifest.  Ground truth
is joined only by the downstream evaluator, which keeps proposal generation
independent from threshold selection.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from football_longform_v2.canonical import load_canonical_split  # noqa: E402
from football_longform_v2.config import load_config  # noqa: E402
from football_longform_v2.decoding import decode_proposals  # noqa: E402
from football_longform_v2.evaluation import infer_continuous_timeline  # noqa: E402
from football_longform_v2.feature_store import load_aligned_npz  # noqa: E402
from football_longform_v2.models import TemporalLocator  # noqa: E402
from football_longform_v2.schema import read_video_ids  # noqa: E402


SCHEMA = "football_longform_v2.shared_candidate_manifest.v1"
ALLOWED_SPLITS = ("train", "calibration")


def resolve(root: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_id_path(config: dict[str, Any], root: Path, split: str) -> Path:
    key = "train_ids" if split == "train" else "calibration_ids"
    return resolve(root, str(config["paths"][key]))


def validate_payloads(payloads: list[dict[str, Any]]) -> None:
    if not payloads:
        raise ValueError("at least one candidate shard is required")
    reference = payloads[0]
    invariant_keys = (
        "schema_version",
        "split",
        "config_sha256",
        "checkpoint_sha256",
        "nms_radius_seconds",
        "labels",
    )
    for payload in payloads:
        if payload.get("schema_version") != SCHEMA:
            raise ValueError("unsupported candidate shard schema")
        for key in invariant_keys:
            if payload.get(key) != reference.get(key):
                raise ValueError(f"candidate shards disagree on {key}")


def aggregate(shards: list[Path], output: Path) -> None:
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in shards]
    validate_payloads(payloads)
    videos = [video for payload in payloads for video in payload["videos"]]
    video_ids = [str(video["video_id"]) for video in videos]
    if len(video_ids) != len(set(video_ids)):
        raise RuntimeError("candidate shards contain duplicate video IDs")
    total_duration = sum(float(video["duration_seconds"]) for video in videos)
    candidate_count = sum(len(video["candidates"]) for video in videos)
    reference = payloads[0]
    merged = {
        key: reference[key]
        for key in (
            "schema_version",
            "split",
            "config",
            "config_sha256",
            "checkpoint",
            "checkpoint_sha256",
            "nms_radius_seconds",
            "labels",
        )
    }
    merged.update(
        {
            "shard_index": 0,
            "num_shards": 1,
            "video_count": len(videos),
            "candidate_count": candidate_count,
            "duration_seconds": total_duration,
            "candidates_per_minute": (
                candidate_count / max(total_duration / 60.0, 1e-9)
            ),
            "video_ids": sorted(video_ids),
            "videos": sorted(videos, key=lambda item: str(item["video_id"])),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "split": merged["split"],
                "video_count": merged["video_count"],
                "candidate_count": merged["candidate_count"],
                "candidates_per_minute": merged["candidates_per_minute"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", choices=ALLOWED_SPLITS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--nms-radius-seconds", type=float, default=0.5)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--aggregate", nargs="*")
    args = parser.parse_args()
    output_path = Path(args.output).expanduser().resolve()
    if args.aggregate is not None:
        if not args.aggregate:
            raise ValueError("--aggregate requires one or more shard files")
        aggregate(
            [Path(path).expanduser().resolve() for path in args.aggregate],
            output_path,
        )
        return
    if not args.config or not args.checkpoint or not args.split:
        raise ValueError(
            "--config, --checkpoint and --split are required outside aggregate mode"
        )
    if args.nms_radius_seconds <= 0:
        raise ValueError("--nms-radius-seconds must be positive")
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index/count")

    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    config = load_config(str(config_path))
    root = Path(config["_project_root"])
    canonical = load_canonical_split(
        resolve(root, str(config["paths"]["canonical_manifest"])), args.split
    )
    all_ids = read_video_ids(split_id_path(config, root, args.split))
    if tuple(all_ids) != tuple(canonical.media_ids):
        raise RuntimeError("canonical manifest and split ID list disagree")
    selected_ids = all_ids[args.shard_index :: args.num_shards]

    payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    model = TemporalLocator.from_config(config)
    model.load_state_dict(payload["model"], strict=True)
    device = torch.device(args.device)
    model.to(device).eval()
    feature_root = resolve(root, str(config["paths"]["feature_store"])) / args.split
    labels = tuple(str(label) for label in config["task"]["output_labels"])
    timeline_hz = float(config["features"]["timeline_hz"])
    videos: list[dict[str, Any]] = []

    for video_id in selected_ids:
        timeline = load_aligned_npz(feature_root / video_id / "timeline.npz")
        outputs = infer_continuous_timeline(
            model,
            timeline,
            device=device,
            core_steps=int(round(120.0 * timeline_hz)),
            context_steps=int(round(30.0 * timeline_hz)),
        )
        if outputs.class_logits is None:
            raise RuntimeError("A0 checkpoint did not emit class logits")
        max_logits, max_indices = outputs.class_logits.max(dim=-1)
        proposals = decode_proposals(
            max_logits[:, None].unsqueeze(0),
            timeline.timestamps.unsqueeze(0),
            ("any_event",),
            threshold=0.0,
            nms_radius_seconds={"any_event": float(args.nms_radius_seconds)},
            max_per_minute={"any_event": 1000.0},
        )[0]
        class_probabilities = outputs.class_logits.sigmoid()
        candidates = []
        for ordinal, proposal in enumerate(proposals):
            index = int(proposal.timeline_index)
            class_index = int(max_indices[index])
            candidates.append(
                {
                    "candidate_id": f"{video_id}:shared:{ordinal:06d}",
                    "timestamp": float(proposal.timestamp),
                    "timeline_index": index,
                    "source_label": labels[class_index],
                    "source_score": float(proposal.score),
                    "source_logit": float(max_logits[index]),
                    "class_scores": {
                        label: float(class_probabilities[index, label_index])
                        for label_index, label in enumerate(labels)
                    },
                }
            )
        duration = float(timeline.timestamps[-1]) if timeline.timestamps.numel() else 0.0
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
        print(
            f"split={args.split} shard={args.shard_index}/{args.num_shards} "
            f"video={video_id} candidates={len(candidates)}",
            flush=True,
        )

    total_duration = sum(float(video["duration_seconds"]) for video in videos)
    candidate_count = sum(len(video["candidates"]) for video in videos)
    result = {
        "schema_version": SCHEMA,
        "split": args.split,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "nms_radius_seconds": float(args.nms_radius_seconds),
        "labels": list(labels),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "video_count": len(videos),
        "candidate_count": candidate_count,
        "duration_seconds": total_duration,
        "candidates_per_minute": candidate_count
        / max(total_duration / 60.0, 1e-9),
        "video_ids": sorted(selected_ids),
        "videos": videos,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"output={output_path}", flush=True)


if __name__ == "__main__":
    main()
