#!/usr/bin/env python
"""Cache exact fixed-stride online predictions for the Val15 calibration split."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_football_events as football
import train_football_events_online_simulation  # noqa: F401


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    cfg = football.load_config(args.config, [])
    cfg["model"]["init_checkpoint"] = str(Path(args.checkpoint).resolve())
    cfg["model"]["init_checkpoint_strict"] = True
    cfg["output_dir"] = str(args.output_dir)
    audit = cfg["eval"]["external_audit"]
    audit["enabled"] = True
    audit["split"] = "val"
    online = audit["online_mode"]
    online["enabled"] = True
    online["window_stride_sec"] = 5.0
    online["nms_radius_sec"] = 5.0
    online["tolerance_sec"] = 5.0
    online["capped_clip_sec"] = 10.0
    cfg["eval"]["tuned_min_recall"] = {
        "shot": 0.85, "save": 0.80, "set_piece": 0.80,
    }

    world_size = int(os.environ.get("WORLD_SIZE", "1") or 1)
    local_rank = int(os.environ.get("LOCAL_RANK", "0") or 0)
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        cfg["device"] = f"cuda:{local_rank}"
        cfg["gpu_ids"] = list(range(world_size))
    rank = dist.get_rank() if football.distributed_training_active() else 0
    device = torch.device(cfg.device)
    football.configure_runtime_threads(cfg)
    football.configure_label_schema(cfg)
    football.seed_everything(int(cfg.get("seed", 42)) + rank, True)
    topology = football.resolve_runtime_topology(cfg, device)
    if football.distributed_training_active():
        cfg["data"]["num_workers"] = int(topology["num_workers_per_gpu"])

    started = time.time()
    dataset, records = football.prepare_external_audit_dataset(cfg)
    if dataset is None:
        raise RuntimeError("online Val15 dataset was not built")
    loader = football.make_loader(
        dataset,
        cfg,
        is_train=False,
        batch_size=int(topology["eval_per_gpu_batch_size"]),
        distributed=football.distributed_training_active(),
    )
    model = football.make_model(cfg, use_cached_features=False, device=device)
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"online_val15 checkpoint={cfg.model.init_checkpoint} "
            f"world_size={world_size} records={len(records)}",
            flush=True,
        )
    cache = args.output_dir / "online_val15_epoch_000.npz"
    metrics = football.evaluate(
        model, loader, cfg, device, online_cache_path=cache,
    )
    if rank == 0:
        output = args.output_dir / "online_val15_window_tuned_report.json"
        output.write_text(
            json.dumps(
                {
                    "checkpoint": str(Path(args.checkpoint).resolve()),
                    "records": len(records),
                    "runtime_sec": time.time() - started,
                    "metrics": metrics,
                },
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"report={output} cache={cache}", flush=True)
    if football.distributed_training_active():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
