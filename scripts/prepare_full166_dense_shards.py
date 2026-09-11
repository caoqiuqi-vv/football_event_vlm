#!/usr/bin/env python3
"""Create balanced, mutually exclusive dense-inference shards.

Completed outputs and strictly validated cached outputs are excluded. Cached
outputs are exposed as symlinks so the final all-video aggregation can use the
normal evaluator compatibility checks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED = ("summary.json", "metrics.json", "window_predictions.csv", "gt_events.json")


def complete(path: Path) -> bool:
    return all((path / name).is_file() for name in REQUIRED)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,4,6")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    inventory = json.loads((run_dir / "full_review_inventory.json").read_text())
    gpu_ids = [int(item) for item in args.gpus.split(",") if item.strip()]
    shard_dir = run_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    remaining: list[dict] = []
    cached = completed = 0
    for row in inventory["videos"]:
        video_id = str(row["video_id"])
        destination = run_dir / video_id
        cache_source = str(row.get("cache_source") or "")
        if cache_source:
            source = Path(cache_source).resolve()
            if not complete(source):
                raise RuntimeError(f"invalid cached output for {video_id}: {source}")
            if destination.exists() or destination.is_symlink():
                if destination.resolve() != source and not complete(destination):
                    raise RuntimeError(f"occupied incomplete cache destination: {destination}")
            else:
                destination.symlink_to(source, target_is_directory=True)
            cached += 1
            continue
        if complete(destination):
            completed += 1
            continue
        remaining.append(row)

    # Longest-processing-time scheduling; video bytes are a robust proxy for
    # decoded duration for this homogeneous 720p corpus.
    bins = [{"gpu": gpu, "bytes": 0, "videos": []} for gpu in gpu_ids]
    for row in sorted(remaining, key=lambda x: int(x.get("video_size_bytes", 0)), reverse=True):
        target = min(bins, key=lambda x: x["bytes"])
        target["videos"].append(str(row["video_id"]))
        target["bytes"] += int(row.get("video_size_bytes", 0))

    for item in bins:
        path = shard_dir / f"gpu{item['gpu']}_video_ids.txt"
        path.write_text("\n".join(item["videos"]) + ("\n" if item["videos"] else ""))
    plan = {
        "gpus": gpu_ids,
        "total_inventory": len(inventory["videos"]),
        "cached": cached,
        "completed_local": completed,
        "remaining": len(remaining),
        "shards": bins,
    }
    (shard_dir / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({**plan, "shards": [
        {"gpu": x["gpu"], "videos": len(x["videos"]), "gib": round(x["bytes"] / 2**30, 2)}
        for x in bins
    ]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
