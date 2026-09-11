#!/usr/bin/env python
"""Tail-safe, explicitly sharded pixel/audio cache builder.

Each process is serial; parallelism is obtained by launching disjoint
``--shard-index`` values.  This avoids Python 3.13 ProcessPool pickling issues
with compatibility-loaded legacy builders.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from build_pixel_audio_store_fixed_v2 import base, extract_log_mel_fixed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=("train", "calibration"), required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard")
    config_path = Path(args.config).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = config["data"]
    manifest_path = Path(data["canonical_manifest"])
    if not manifest_path.is_absolute():
        manifest_path = (config_path.parents[2] / manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = list(manifest[args.split])[args.shard_index::args.num_shards]
    contract = {
        "schema": base.SCHEMA,
        "sample_fps": float(data["sample_fps"]),
        "image_size": [int(value) for value in data["image_size"]],
        "jpeg_quality": int(data["jpeg_quality"]),
        "audio_sample_rate": int(data["audio_sample_rate"]),
        "audio_mel_bins": int(data["audio_mel_bins"]),
        "timestamp_rule": "(frame_index+0.5)/sample_fps",
        "canonical_manifest_sha256": base.sha256_file(manifest_path),
    }
    contract_sha = hashlib.sha256(json.dumps(
        contract, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    common = {
        "split": args.split,
        "output_root": str(Path(data["frame_store"]).expanduser().resolve()),
        "sample_fps": float(data["sample_fps"]),
        "image_size": tuple(int(value) for value in data["image_size"]),
        "jpeg_quality": int(data["jpeg_quality"]),
        "audio_sample_rate": int(data["audio_sample_rate"]),
        "audio_mel_bins": int(data["audio_mel_bins"]),
        "config_sha256": contract_sha,
    }
    base.extract_log_mel = extract_log_mel_fixed
    rows = []
    for item in items:
        row = base.build_one(item, **common)
        rows.append(row)
        print(json.dumps({
            key: row[key] for key in (
                "video_id", "status", "frame_count", "duration_seconds", "decode_quality"
            ) if key in row
        }, ensure_ascii=False), flush=True)
    output_root = Path(common["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": base.SCHEMA,
        "split": args.split,
        "config": str(config_path),
        "preprocessing_contract": contract,
        "preprocessing_sha256": contract_sha,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "requested": len(items),
        "completed": len(rows),
        "rows": rows,
        "sealed_test_media_opened": False,
    }
    path = output_root / f"build_{args.split}_serial_shard{args.shard_index}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"report={path}", flush=True)


if __name__ == "__main__":
    main()
