#!/usr/bin/env python3
"""Prepare an auditable, resumable dense-inference run for full-dataset repair.

The script preserves the original train/validation/test identity, validates that
the splits are disjoint, inventories every video and annotation, and only reuses
per-video dense outputs when their checkpoint, geometry, protocol and annotation
hash exactly match the requested run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from football_eval_semantics import SCORE_SEMANTICS_VERSION, has_current_score_semantics

VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".avi", ".webm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-ids", type=Path, required=True)
    parser.add_argument("--val-ids", type=Path, required=True)
    parser.add_argument("--test-ids", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, action="append", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reuse-run-dir", type=Path, action="append", default=[])
    parser.add_argument("--clip-sec", type=float, default=10.0)
    parser.add_argument("--stride-sec", type=float, default=5.0)
    parser.add_argument("--image-size", default="720,1280")
    parser.add_argument("--match-tolerance-sec", type=float, default=3.0)
    parser.add_argument("--prediction-postprocess", default="window_overlap")
    parser.add_argument("--score-source", default="clip")
    parser.add_argument("--thresholds", default="checkpoint")
    return parser.parse_args()


def read_ids(path: Path) -> list[str]:
    values = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(values) != len(set(values)):
        raise RuntimeError(f"duplicate video IDs in {path}")
    return values


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_video(video_id: str, roots: list[Path]) -> Path | None:
    for root in roots:
        for suffix in VIDEO_SUFFIXES:
            candidate = root / f"{video_id}{suffix}"
            if candidate.is_file():
                return candidate.resolve()
    return None


def same_path(left: str | Path, right: Path) -> bool:
    try:
        return Path(left).resolve() == right.resolve()
    except (OSError, TypeError, ValueError):
        return False


def compatible_cache(
    source: Path,
    *,
    checkpoint: Path,
    annotation: Path,
    annotation_hash: str,
    args: argparse.Namespace,
) -> tuple[bool, str]:
    summary_path = source / "summary.json"
    metrics_path = source / "metrics.json"
    windows_path = source / "window_predictions.csv"
    if not (summary_path.is_file() and metrics_path.is_file() and windows_path.is_file()):
        return False, "missing_required_files"
    try:
        summary: dict[str, Any] = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return False, f"invalid_summary:{error}"
    checks = {
        "checkpoint": same_path(str(summary.get("checkpoint", "")), checkpoint),
        "annotation_path": same_path(str(summary.get("annotation_path", "")), annotation),
        "annotation_sha256": summary.get("annotation_sha256") == annotation_hash,
        "clip_sec": float(summary.get("clip_sec", -1.0)) == args.clip_sec,
        "stride_sec": float(summary.get("stride_sec", -1.0)) == args.stride_sec,
        "image_size": list(summary.get("image_size", []))
        == [int(item) for item in args.image_size.split(",")],
        "match_tolerance_sec": float(summary.get("match_tolerance_sec", -1.0))
        == args.match_tolerance_sec,
        "prediction_postprocess": summary.get("prediction_postprocess")
        == args.prediction_postprocess,
        "score_semantics_version": has_current_score_semantics(summary),
        "score_source": summary.get("score_source") == args.score_source,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return not failed, ",".join(failed) if failed else "compatible"


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    split_paths = {
        "train": args.train_ids.resolve(),
        "internal_val": args.val_ids.resolve(),
        "sealed_test": args.test_ids.resolve(),
    }
    splits = {name: read_ids(path) for name, path in split_paths.items()}
    split_sets = {name: set(values) for name, values in splits.items()}
    names = list(splits)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = sorted(split_sets[left] & split_sets[right])
            if overlap:
                raise RuntimeError(f"split overlap {left}/{right}: {overlap}")

    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    gt_dir = args.gt_dir.resolve()
    roots = [root.resolve() for root in args.video_root]
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    inventory: list[dict[str, Any]] = []
    missing_assets: list[dict[str, str]] = []
    reused: list[str] = []
    pending: list[str] = []
    rejected_reuse: dict[str, list[str]] = {}
    ordered_ids: list[str] = []
    for split, values in splits.items():
        for video_id in values:
            ordered_ids.append(video_id)
            annotation = (gt_dir / f"{video_id}.json").resolve()
            video = find_video(video_id, roots)
            if video is None or not annotation.is_file():
                missing_assets.append({
                    "video_id": video_id,
                    "split": split,
                    "video": "missing" if video is None else str(video),
                    "annotation": str(annotation) if annotation.is_file() else "missing",
                })
                continue
            annotation_hash = sha256(annotation)
            destination = run_dir / video_id
            cache_source: Path | None = None
            rejection_reasons: list[str] = []
            if destination.exists():
                compatible, reason = compatible_cache(
                    destination,
                    checkpoint=checkpoint,
                    annotation=annotation,
                    annotation_hash=annotation_hash,
                    args=args,
                )
                if compatible:
                    cache_source = destination
                else:
                    raise RuntimeError(
                        f"incompatible existing destination {destination}: {reason}"
                    )
            else:
                for reuse_root in args.reuse_run_dir:
                    candidate = reuse_root.resolve() / video_id
                    compatible, reason = compatible_cache(
                        candidate,
                        checkpoint=checkpoint,
                        annotation=annotation,
                        annotation_hash=annotation_hash,
                        args=args,
                    )
                    if compatible:
                        destination.symlink_to(candidate, target_is_directory=True)
                        cache_source = candidate
                        break
                    if candidate.exists():
                        rejection_reasons.append(f"{candidate}:{reason}")
            if cache_source is not None:
                reused.append(video_id)
                status = "cached"
            else:
                pending.append(video_id)
                status = "pending"
                if rejection_reasons:
                    rejected_reuse[video_id] = rejection_reasons
            inventory.append({
                "video_id": video_id,
                "split": split,
                "video_path": str(video),
                "video_size_bytes": video.stat().st_size,
                "annotation_path": str(annotation),
                "annotation_sha256": annotation_hash,
                "dense_status": status,
                "cache_source": str(cache_source) if cache_source else "",
            })

    if missing_assets:
        raise RuntimeError(f"missing video/annotation assets: {missing_assets}")

    header = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "splits": {name: {"path": str(split_paths[name]), "count": len(values)} for name, values in splits.items()},
        "protocol": {
            "clip_sec": args.clip_sec,
            "stride_sec": args.stride_sec,
            "image_size": args.image_size,
            "match_tolerance_sec": args.match_tolerance_sec,
            "prediction_postprocess": args.prediction_postprocess,
            "score_source": args.score_source,
            "score_semantics_version": SCORE_SEMANTICS_VERSION,
            "thresholds": args.thresholds,
            "nms": False,
        },
        "summary": {
            "videos": len(ordered_ids),
            "cached": len(reused),
            "pending": len(pending),
            "missing_assets": len(missing_assets),
        },
        "rejected_reuse": rejected_reuse,
        "videos": inventory,
    }
    atomic_write_text(run_dir / "full_review_inventory.json", json.dumps(header, ensure_ascii=False, indent=2) + "\n")
    atomic_write_text(run_dir / "all_video_ids.txt", "\n".join(ordered_ids) + "\n")
    atomic_write_text(run_dir / "pending_video_ids.txt", "\n".join(pending) + ("\n" if pending else ""))
    atomic_write_text(run_dir / "cached_video_ids.txt", "\n".join(reused) + ("\n" if reused else ""))
    print(json.dumps(header["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
