#!/usr/bin/env python
"""Verify content dominates static queries and receives localization gradient."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

import train_football_events as football
from train_football_events_featuremap_r1_v10 import (
    load_featuremap_r1_v10,
    load_v10_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_v10_config(args.config, [])
    cfg["gpu_ids"] = [0]
    cfg["device"] = "cuda:0"
    cfg["train"]["per_gpu_batch_size"] = 2
    cfg["eval"]["per_gpu_batch_size"] = 2
    cfg["data"]["num_workers_per_gpu"] = 0
    football.configure_runtime_threads(cfg)
    football.configure_label_schema(cfg)
    football.resolve_runtime_topology(cfg, torch.device(cfg.device))
    football.seed_everything(int(cfg.seed), bool(cfg.deterministic))
    train_dataset, _, _, _ = football.prepare_datasets(cfg, use_cache=False)
    batch = next(iter(football.make_loader(train_dataset, cfg, is_train=True)))
    device = torch.device(cfg.device)
    model = load_featuremap_r1_v10(cfg, device)
    model.eval()

    with football.autocast_context(device, True, "bf16"):
        outputs = football.forward_model_batch(
            model, batch, device, return_aux=True
        )
        targets = batch["targets"].to(device)
        masks = batch["label_masks"].to(device)
        loss = football.masked_mean(
            F.binary_cross_entropy_with_logits(
                outputs["spatial_clip_logits"], targets, reduction="none"
            ),
            masks,
        )
    loss.backward()
    inputs = batch["inputs"].to(device)
    with torch.no_grad(), football.autocast_context(device, True, "bf16"):
        features, _ = model.encode_frames_with_patch_tokens(inputs)
        global_outputs = model._global_branch_outputs(features)
        static_norm, context_norm, motion_norm = (
            model.spatial_attention.static_dynamic_norms(
                global_outputs["frame_tokens"]
            )
        )
    context_grad = model.spatial_attention.context_query.weight.grad
    patch_grad = model.spatial_attention.patch_key.weight.grad
    print(
        "FEATUREMAP_R1_V10_SMOKE_OK",
        {
            "loss": float(loss.detach().cpu()),
            "static_norm": float(static_norm.cpu()),
            "context_norm": float(context_norm.cpu()),
            "motion_norm": float(motion_norm.cpu()),
            "dynamic_to_static": float(
                ((context_norm + motion_norm) / static_norm.clamp_min(1e-8)).cpu()
            ),
            "context_grad": float(context_grad.abs().mean().cpu()),
            "patch_key_grad": float(patch_grad.abs().mean().cpu()),
            "base_query_trainable": model.spatial_attention.base_queries.requires_grad,
            "scale_trainable": model.spatial_attention.direct_log_scale.requires_grad,
            "entropy": float(
                outputs["spatial_attention_entropy"].mean().detach().cpu()
            ),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
