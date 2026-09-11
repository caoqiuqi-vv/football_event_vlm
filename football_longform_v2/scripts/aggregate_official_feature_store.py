"""Aggregate feature-cache truth after sharded extraction."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
for path in (PROJECT_ROOT / "src", WORKSPACE_ROOT, PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_official_feature_store import (  # noqa: E402
    CACHE_SCHEMA, atomic_json, cache_duration_validation, cache_validation,
    canonical_items, config_digest, resolve, sha256_file, source_duration_seconds,
)
from build_rgb_timeline import find_video  # noqa: E402
from football_longform_v2.canonical import load_canonical_split  # noqa: E402
from football_longform_v2.config import load_config  # noqa: E402
from football_longform_v2.schema import assert_disjoint_splits, read_video_ids  # noqa: E402


def canonical_run_sha256(items: list[tuple[str, str]]) -> str:
    return hashlib.sha256(
        json.dumps(items, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Atomically aggregate official-DINO cache truth after sharded extraction."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--video-root", default="/mnt/data_16t/football/raw_video_720P")
    parser.add_argument("--splits", nargs="+", default=["train", "calibration"])
    parser.add_argument("--manifest-name", default="manifest.json")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    selected = tuple(args.splits)
    if set(selected) - {"train", "calibration"} or len(selected) != len(set(selected)):
        raise ValueError("only unique train/calibration splits are allowed")
    config = load_config(args.config)
    root = Path(config["_project_root"])
    paths = config["paths"]
    split_ids = {
        "train": read_video_ids(resolve(root, paths["train_ids"])),
        "calibration": read_video_ids(resolve(root, paths["calibration_ids"])),
        "thirdparty_test": read_video_ids(resolve(root, paths["thirdparty_test_ids"])),
    }
    assert_disjoint_splits(*split_ids.values())
    for split in ("train", "calibration"):
        if load_canonical_split(resolve(root, paths["canonical_manifest"]), split).media_ids != split_ids[split]:
            raise RuntimeError(f"canonical manifest and configured {split} IDs disagree")
    feature_root = resolve(root, paths["feature_store"])
    video_root = Path(args.video_root).expanduser().resolve()
    context, motion = config["features"]["context"], config["features"]["motion"]
    config_sha256 = config_digest(Path(config["_config_path"]))
    weights_sha256 = sha256_file(resolve(root, str(context["weights"])))
    items = canonical_items(split_ids, selected)
    records, counts = [], {"cached_valid": 0, "missing": 0, "invalid_cache": 0}
    for global_index, (split, video_id) in enumerate(items):
        output = feature_root / split / video_id / "timeline.npz"
        if not output.is_file():
            status, reason = "missing", "missing"
        else:
            valid, reason = cache_validation(
                output, context_dim=int(context["dim"]), motion_dim=int(motion["dim"]),
                expected_config_sha256=config_sha256, expected_weights_sha256=weights_sha256,
            )
            if valid:
                try:
                    valid, reason = cache_duration_validation(
                        output, source_duration_s=source_duration_seconds(find_video(video_root, video_id))
                    )
                except FileNotFoundError as error:
                    valid, reason = False, str(error)
            status = "skipped_valid" if valid else "invalid_cache"
        if status == "skipped_valid":
            counts["cached_valid"] += 1
        else:
            counts[status] += 1
        records.append({"global_index": global_index, "split": split, "video_id": video_id,
                        "status": status, "reason": reason, "output": str(output)})
    payload = {
        "cache_schema": CACHE_SCHEMA,
        "experiment_id": config["experiment_id"],
        "aggregation": "cache_validation_truth",
        "run_id": args.run_id,
        "updated_unix": time.time(),
        "total": len(records),
        "processed": len(records),
        "completed": counts["cached_valid"],
        "skipped_valid": counts["cached_valid"],
        "failed": counts["invalid_cache"],
        "requested_splits": list(selected),
        "canonical_run_sha256": canonical_run_sha256(items),
        "config": str(config["_config_path"]),
        "config_sha256": config_sha256,
        "weights": str(resolve(root, str(context["weights"]))),
        "weights_sha256": weights_sha256,
        **counts,
        "records": records,
    }
    output = feature_root / args.manifest_name
    atomic_json(output, payload)
    print("manifest={} cached_valid={}/{} missing={} invalid={}".format(
        output, counts["cached_valid"], len(records), counts["missing"], counts["invalid_cache"]
    ))


if __name__ == "__main__":
    main()
