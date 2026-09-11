#!/usr/bin/env python
"""FeatureMap-R1: learn class heatmaps from event losses with one DINO pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

import train_football_events as football


def safe_checkpoint(path: str) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=True)


def load_featuremap_r1(cfg: Any, device: torch.device) -> nn.Module:
    init_cfg = cfg.model.featuremap_r1
    global_path = str(init_cfg.global_init_checkpoint)
    spatial_path = str(init_cfg.spatial_init_checkpoint)
    cfg.model.init_checkpoint = ""
    base = football.make_model(cfg, use_cached_features=False, device=device)

    global_checkpoint = safe_checkpoint(global_path)
    football.load_model_init_checkpoint(
        base,
        global_path,
        checkpoint=global_checkpoint,
        expected_backbone=str(cfg.model.backbone),
        strict=False,
    )
    spatial_checkpoint = safe_checkpoint(spatial_path)
    spatial_state = football.strip_module_prefix(spatial_checkpoint["model"])
    target_state = base.state_dict()
    matched: dict[str, Tensor] = {}
    for key, value in spatial_state.items():
        if not key.startswith("spatial_attention."):
            continue
        if key in target_state and tuple(value.shape) == tuple(target_state[key].shape):
            matched[key] = value
    if not matched:
        raise RuntimeError("No spatial-attention weights matched FeatureMap-R1")
    base.load_state_dict(matched, strict=False)

    # The validated global classifier is the immutable reference. Spatial query,
    # local temporal modeling, local auxiliary heads, and conditioned fusion stay
    # trainable. Since inputs never require gradients, the frozen ViT pass does
    # not retain a backward graph.
    base.freeze_global_parameters()
    print(
        f"Loaded FeatureMap-R1 global={global_path} spatial={spatial_path} "
        f"spatial_matched={len(matched)}",
        flush=True,
    )
    trainable = sum(p.numel() for p in base.parameters() if p.requires_grad)
    total = sum(p.numel() for p in base.parameters())
    print(
        f"FeatureMap-R1 params trainable={trainable} total={total} "
        f"trainable_fraction={trainable/max(total,1):.6f}",
        flush=True,
    )
    return base.to(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = football.load_config(args.config, [])
    football.configure_runtime_threads(cfg)
    football.configure_label_schema(cfg)
    device = torch.device(cfg.device)
    football.resolve_runtime_topology(cfg, device)
    football.seed_everything(int(cfg.seed), bool(cfg.deterministic))
    if args.dry_run:
        football.run_dry_run(cfg)
        return

    train_dataset, val_dataset, train_records, val_records = football.prepare_datasets(
        cfg, use_cache=False
    )
    print("train", football.summarize_records(train_records), flush=True)
    print("val", football.summarize_records(val_records), flush=True)
    train_loader = football.make_loader(train_dataset, cfg, is_train=True)
    val_loader = football.make_loader(
        val_dataset, cfg, is_train=False, batch_size=int(cfg.eval.batch_size)
    )
    model = load_featuremap_r1(cfg, device)
    gpu_ids = [int(value) for value in cfg.get("gpu_ids", [])]
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])
        print(f"Using FeatureMap-R1 DataParallel gpu_ids={gpu_ids}", flush=True)
    if args.eval_only:
        metrics = football.evaluate(model, val_loader, cfg, device)
        path = Path(cfg.output_dir) / "eval_only_metrics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
        return
    football.train(model, train_loader, val_loader, cfg, device, train_records)


if __name__ == "__main__":
    main()
