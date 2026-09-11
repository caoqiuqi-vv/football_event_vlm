#!/usr/bin/env python
"""Evaluate a warm-start checkpoint as an immutable epoch-0 anchor.

This deliberately reuses the training program's exact Val15 and External18
dataset/evaluate implementations while skipping every training/optimizer path.
Thresholds are tuned once on Val15 and then fixed for External18 online event
evaluation.  Both passes are DDP-sharded and the External18 raw predictions are
cached for later threshold-policy rescoring.
"""
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
import train_football_events_online_simulation  # noqa: F401  # installs exact online eval patches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = football.load_config(args.config, args.overrides)
    cfg["model"]["init_checkpoint"] = str(Path(args.checkpoint).resolve())
    cfg["model"]["init_checkpoint_strict"] = True
    cfg["eval"]["tuned_min_recall"] = {
        "shot": 0.85,
        "save": 0.80,
        "set_piece": 0.80,
    }
    cfg["eval"]["external_audit"]["enabled"] = True
    cfg["eval"]["external_audit"]["online_mode"]["enabled"] = True
    cfg["eval"]["external_audit"]["online_mode"]["window_stride_sec"] = 5.0
    cfg["eval"]["external_audit"]["online_mode"]["nms_radius_sec"] = 5.0
    cfg["eval"]["external_audit"]["online_mode"]["tolerance_sec"] = 5.0
    cfg["eval"]["external_audit"]["online_mode"]["capped_clip_sec"] = 10.0
    cfg["output_dir"] = str(args.output_dir)

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
    # Build Val15 through the same evaluation-dataset constructor, without
    # constructing or probing the unused train split. Then restore External18
    # with exact online fixed-stride expansion.
    audit_cfg = cfg["eval"]["external_audit"]
    external_split = str(audit_cfg.get("split", "external"))
    audit_cfg["split"] = "val"
    audit_cfg["online_mode"]["enabled"] = False
    val_dataset, val_records = football.prepare_external_audit_dataset(cfg)
    audit_cfg["split"] = external_split
    audit_cfg["online_mode"]["enabled"] = True
    external_dataset, external_records = football.prepare_external_audit_dataset(cfg)
    if external_dataset is None:
        raise RuntimeError("External18 online dataset is disabled or missing")
    eval_batch_size = int(topology["eval_per_gpu_batch_size"])
    distributed = football.distributed_training_active()
    val_loader = football.make_loader(
        val_dataset, cfg, is_train=False, batch_size=eval_batch_size,
        distributed=distributed,
    )
    external_loader = football.make_loader(
        external_dataset, cfg, is_train=False, batch_size=eval_batch_size,
        distributed=distributed,
    )
    model = football.make_model(cfg, use_cached_features=False, device=device)

    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"epoch0_anchor checkpoint={cfg.model.init_checkpoint} "
            f"world_size={world_size} batch_per_gpu={eval_batch_size} "
            f"val_records={len(val_records)} external_records={len(external_records)}",
            flush=True,
        )

    val_started = time.time()
    val_metrics = football.evaluate(model, val_loader, cfg, device)
    val_sec = time.time() - val_started
    fixed_thresholds = val_metrics.get("thresholds") if rank == 0 else None
    if distributed:
        payload = [fixed_thresholds]
        dist.broadcast_object_list(payload, src=0)
        fixed_thresholds = payload[0]

    external_started = time.time()
    cache_path = args.output_dir / "external18_online_epoch_000.npz"
    external_metrics = football.evaluate(
        model,
        external_loader,
        cfg,
        device,
        fixed_thresholds=fixed_thresholds,
        online_cache_path=cache_path,
    )
    external_sec = time.time() - external_started

    if rank == 0:
        report = {
            "protocol": {
                "name": "epoch0_val15_to_external18_online",
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "threshold_calibration_split": "internal_val15",
                "test_split": "untouched_external18",
                "tuned_min_recall": dict(cfg.eval.tuned_min_recall),
                "online_window_sec": float(cfg.video.clip_duration),
                "online_stride_sec": 5.0,
                "pointnms_radius_sec": 5.0,
                "one_to_one_tolerance_sec": 5.0,
                "review_clip_cap_sec": 10.0,
            },
            "thresholds": fixed_thresholds,
            "val15": val_metrics,
            "external18": external_metrics,
            "runtime": {
                "world_size": world_size,
                "per_gpu_batch_size": eval_batch_size,
                "val_sec": val_sec,
                "external_sec": external_sec,
                "total_sec": time.time() - started,
            },
        }
        output_path = args.output_dir / "epoch0_anchor_report.json"
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        online = external_metrics.get("online_event", {})
        print(f"thresholds={fixed_thresholds}", flush=True)
        print(
            "external18_online "
            f"precision={online.get('micro_precision', 0.0):.6f} "
            f"recall={online.get('micro_recall', 0.0):.6f} "
            f"fp={online.get('fp', 0)} "
            f"review_min={online.get('nms_capped_union_minutes', 0.0):.3f} "
            f"participation={online.get('nms_capped_participation_ratio', 0.0):.6f}",
            flush=True,
        )
        print(f"report={output_path} cache={cache_path}", flush=True)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
