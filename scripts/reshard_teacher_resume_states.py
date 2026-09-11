#!/usr/bin/env python
"""Expand resumable Teacher shards without repeating completed intervals."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--old-shards", type=int, required=True)
    parser.add_argument("--new-shards", type=int, required=True)
    args = parser.parse_args()
    if args.old_shards < 1 or args.new_shards <= args.old_shards:
        raise ValueError("new-shards must be greater than old-shards >= 1")

    root = Path(args.root)
    completed: set[str] = set()
    for shard_index in range(args.old_shards):
        state_path = root / f"shard_{shard_index:02d}_state.json"
        if not state_path.is_file():
            raise FileNotFoundError(state_path)
        payload = json.loads(state_path.read_text())
        completed.update(str(key) for key in payload.get("completed", []))

    for shard_index in range(args.new_shards):
        payload = {
            "shard_index": shard_index,
            "num_shards": args.new_shards,
            # Every new worker receives the union.  It only visits videos in
            # its own modulo shard, so previously finished work is skipped.
            "completed": sorted(completed),
        }
        state_path = root / f"shard_{shard_index:02d}_state.json"
        temporary = state_path.with_suffix(f".json.tmp.{os.getpid()}")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        os.replace(temporary, state_path)
    print(
        json.dumps(
            {
                "root": str(root.resolve()),
                "old_shards": args.old_shards,
                "new_shards": args.new_shards,
                "completed_union": len(completed),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
