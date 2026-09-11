from __future__ import annotations

"""Run one restart-safe sequential VideoMAE extraction worker per physical GPU."""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILDER = Path(__file__).with_name("build_videomae_chunk_store.py")


def run_worker(
    *,
    gpu: int,
    shard_index: int,
    num_shards: int,
    canonical_config: Path,
    a1_config: Path,
    checkpoint: Path,
    output_root: Path,
    batch_size: int,
    chunk_seconds: float,
    log_dir: Path,
) -> dict:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    rows = []
    log_path = log_dir / f"extract_gpu{gpu}_shard{shard_index}.log"
    with log_path.open("a", encoding="utf-8") as log:
        for split in ("train", "calibration"):
            command = [
                sys.executable,
                str(BUILDER),
                "--canonical-config", str(canonical_config),
                "--a1-config", str(a1_config),
                "--checkpoint", str(checkpoint),
                "--split", split,
                "--output-root", str(output_root),
                "--device", "cuda:0",
                "--batch-size", str(batch_size),
                "--chunk-seconds", str(chunk_seconds),
                "--shard-index", str(shard_index),
                "--num-shards", str(num_shards),
            ]
            log.write(json.dumps({
                "event": "start", "gpu": gpu, "split": split, "command": command,
            }) + "\n")
            log.flush()
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT.parent,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            rows.append({"split": split, "returncode": completed.returncode})
            if completed.returncode:
                raise RuntimeError(
                    f"extractor failed gpu={gpu} shard={shard_index} split={split}; "
                    f"see {log_path}"
                )
    return {"gpu": gpu, "shard_index": shard_index, "runs": rows, "log": str(log_path)}


def verify_complete(output_root: Path, expected: dict[str, int]) -> dict:
    report = {}
    for split, count in expected.items():
        split_root = output_root / split
        ready = []
        if split_root.is_dir():
            for item in split_root.iterdir():
                if item.is_dir() and all(
                    (item / name).is_file()
                    for name in ("features.npy", "timestamps.npy", "metadata.json")
                ):
                    ready.append(item.name)
        report[split] = {"expected": count, "ready": len(ready), "video_ids": sorted(ready)}
        if len(ready) != count:
            raise RuntimeError(f"incomplete {split} chunk store: {len(ready)}/{count}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-config", required=True)
    parser.add_argument("--a1-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--gpus", default="0,6,7")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--chunk-seconds", type=float, default=4.0)
    parser.add_argument("--expected-train", type=int, default=135)
    parser.add_argument("--expected-calibration", type=int, default=18)
    args = parser.parse_args()
    gpus = tuple(int(item) for item in args.gpus.split(",") if item.strip())
    if not gpus or len(gpus) != len(set(gpus)) or min(gpus) < 0:
        raise ValueError("--gpus must contain unique non-negative physical GPU IDs")
    output_root = Path(args.output_root).expanduser().resolve()
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "num_shards": len(gpus),
        "canonical_config": Path(args.canonical_config).expanduser().resolve(),
        "a1_config": Path(args.a1_config).expanduser().resolve(),
        "checkpoint": Path(args.checkpoint).expanduser().resolve(),
        "output_root": output_root,
        "batch_size": args.batch_size,
        "chunk_seconds": args.chunk_seconds,
        "log_dir": log_dir,
    }
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [
            pool.submit(run_worker, gpu=gpu, shard_index=index, **common)
            for index, gpu in enumerate(gpus)
        ]
        workers = [future.result() for future in futures]
    report = {
        "schema": "football_longform_v2.videomae_chunk_coordinator.v1",
        "checkpoint": str(common["checkpoint"]),
        "a1_config": str(common["a1_config"]),
        "chunk_seconds": args.chunk_seconds,
        "batch_size_per_gpu": args.batch_size,
        "physical_gpus": list(gpus),
        "workers": workers,
        "completion": verify_complete(output_root, {
            "train": args.expected_train,
            "calibration": args.expected_calibration,
        }),
        "forbidden_test_media_opened": False,
    }
    manifest = output_root / "manifest.json"
    manifest.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"manifest={manifest}")


if __name__ == "__main__":
    main()
