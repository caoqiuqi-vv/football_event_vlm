#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_football_events as football  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit one deterministic epoch of the video/event-balanced sampler."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--per-gpu-batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--positive-fraction", type=float, default=0.25)
    parser.add_argument("--unique-videos-per-effective-batch", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def distribution(values: list[int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"min": 0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "max": 0, "mean": 0.0}
    return {
        "min": int(array.min()),
        "p10": float(np.quantile(array, 0.10)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "max": int(array.max()),
        "mean": float(array.mean()),
    }


def main() -> None:
    args = parse_args()
    if args.world_size < 1:
        raise ValueError("--world-size must be positive")
    cfg = football.load_config(args.config, [])
    football.configure_label_schema(cfg)
    records, _ = football.load_long_video_records(cfg, args.split)
    samplers = [
        football.VideoEventBalancedSampler(
            records,
            records,
            num_replicas=args.world_size,
            rank=rank,
            local_batch_size=args.per_gpu_batch_size,
            grad_accum_steps=args.grad_accum_steps,
            drop_last=True,
            seed=int(cfg.get("seed", 42)),
            positive_fraction=args.positive_fraction,
            unique_videos_per_effective_batch=(
                args.unique_videos_per_effective_batch
            ),
        )
        for rank in range(args.world_size)
    ]
    for sampler in samplers:
        sampler.set_epoch(args.epoch)
    rank_orders = [list(sampler) for sampler in samplers]
    total_size = samplers[0].total_size
    global_order = [
        rank_orders[position % args.world_size][position // args.world_size]
        for position in range(total_size)
    ]

    global_batch = args.world_size * args.per_gpu_batch_size
    effective_batch = global_batch * args.grad_accum_steps
    video_keys = [
        (records[index].source, records[index].video_id) for index in global_order
    ]
    microbatch_unique_videos = [
        len(set(video_keys[start : start + global_batch]))
        for start in range(0, total_size, global_batch)
    ]
    effective_unique_videos = [
        len(set(video_keys[start : start + effective_batch]))
        for start in range(0, total_size, effective_batch)
    ]
    record_counts = Counter(global_order)
    video_counts = Counter(video_keys)
    label_counts = {
        label: int(
            sum(records[index].labels[label_index] > 0 for index in global_order)
        )
        for label_index, label in enumerate(football.LABELS)
    }
    focus_anchor_counts = {
        label: int(
            sum(
                bool(records[index].focus_labels)
                and records[index].focus_labels[label_index] > 0
                for index in global_order
            )
        )
        for label_index, label in enumerate(football.LABELS)
    }
    payload = {
        "config": args.config,
        "split": args.split,
        "epoch": args.epoch,
        "source_records": football.summarize_records(records),
        "sampler": samplers[0].summary(),
        "sampled_epoch": {
            "num_samples": total_size,
            "positive_count": int(
                sum(not records[index].is_negative for index in global_order)
            ),
            "positive_fraction": float(
                sum(not records[index].is_negative for index in global_order)
                / max(total_size, 1)
            ),
            "window_label_counts": label_counts,
            "focus_anchor_counts": focus_anchor_counts,
            "unique_records": len(record_counts),
            "repeated_draws": total_size - len(record_counts),
            "record_draw_count": distribution(list(record_counts.values())),
            "video_draw_count": distribution(list(video_counts.values())),
            "unique_videos_per_global_microbatch": distribution(
                microbatch_unique_videos
            ),
            "unique_videos_per_effective_batch": distribution(
                effective_unique_videos
            ),
        },
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
