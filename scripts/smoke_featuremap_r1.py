#!/usr/bin/env python
"""Verify classification loss reaches FeatureMap-R1 heatmap parameters."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

import train_football_events as football
from train_football_events_featuremap_r1 import load_featuremap_r1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = football.load_config(args.config, [])
    cfg["gpu_ids"] = [0]
    cfg["device"] = "cuda:0"
    cfg.train["per_gpu_batch_size"] = 1
    cfg.eval["per_gpu_batch_size"] = 1
    cfg.data["num_workers_per_gpu"] = 0
    football.configure_runtime_threads(cfg)
    football.configure_label_schema(cfg)
    football.resolve_runtime_topology(cfg, torch.device(cfg.device))
    football.seed_everything(int(cfg.seed), bool(cfg.deterministic))
    train_dataset, _, _, _ = football.prepare_datasets(cfg, use_cache=False)
    batch = next(iter(football.make_loader(train_dataset, cfg, is_train=True)))
    model = load_featuremap_r1(cfg, torch.device(cfg.device))
    model.train()
    torch.cuda.reset_peak_memory_stats()
    with football.autocast_context(torch.device(cfg.device), True, "bf16"):
        outputs = football.forward_model_batch(
            model, batch, torch.device(cfg.device), return_aux=True
        )
        targets = batch["targets"].to(cfg.device)
        masks = batch["label_masks"].to(cfg.device)
        local = F.binary_cross_entropy_with_logits(
            outputs["spatial_clip_logits"], targets, reduction="none"
        )
        fused = F.binary_cross_entropy_with_logits(
            outputs["logits"], targets, reduction="none"
        )
        loss = football.masked_mean(local + fused, masks)
    loss.backward()
    spatial = model.spatial_attention
    print(
        "FEATUREMAP_R1_SMOKE_OK",
        {
            "input": tuple(batch["inputs"].shape),
            "attention": tuple(outputs["spatial_attention_maps"].shape),
            "spatial_frame_logits": tuple(
                outputs["spatial_frame_event_logits"].shape
            ),
            "base_query_grad": float(spatial.base_queries.grad.abs().mean().cpu()),
            "context_query_grad": float(
                spatial.context_query.weight.grad.abs().mean().cpu()
            ),
            "patch_key_grad": float(spatial.patch_key.weight.grad.abs().mean().cpu()),
            "max_cuda_gib": round(
                torch.cuda.max_memory_allocated() / 1024**3, 3
            ),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
