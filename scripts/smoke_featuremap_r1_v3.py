#!/usr/bin/env python
"""Verify event loss directly updates FeatureMap-R1 v3 heatmaps."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

import train_football_events as football
from train_football_events_featuremap_r1_v3 import (
    load_featuremap_r1_v3,
    load_v3_config,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_v3_config(args.config, [])
    cfg.gpu_ids = [0]
    cfg.device = "cuda:0"
    cfg.train.per_gpu_batch_size = 1
    cfg.eval.per_gpu_batch_size = 1
    cfg.data.num_workers_per_gpu = 0
    football.configure_runtime_threads(cfg)
    football.configure_label_schema(cfg)
    football.resolve_runtime_topology(cfg, torch.device(cfg.device))
    football.seed_everything(int(cfg.seed), bool(cfg.deterministic))
    train_dataset, _, _, _ = football.prepare_datasets(cfg, use_cache=False)
    batch = next(iter(football.make_loader(train_dataset, cfg, is_train=True)))
    model = load_featuremap_r1_v3(cfg, torch.device(cfg.device))
    model.eval()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )

    def forward_loss():
        with football.autocast_context(torch.device(cfg.device), True, "bf16"):
            outputs = football.forward_model_batch(
                model, batch, torch.device(cfg.device), return_aux=True
            )
            targets = batch["targets"].to(cfg.device)
            masks = batch["label_masks"].to(cfg.device)
            loss = football.masked_mean(
                F.binary_cross_entropy_with_logits(
                    outputs["spatial_clip_logits"], targets, reduction="none"
                ),
                masks,
            )
        return outputs, loss

    before, loss = forward_loss()
    before_attention = before["spatial_attention_maps"].detach().clone()
    loss.backward()
    direct_grad = model.spatial_attention.patch_key.weight.grad.detach().clone()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    after, after_loss = forward_loss()
    attention_change = (
        after["spatial_attention_maps"].detach() - before_attention
    ).abs().mean()
    print(
        "FEATUREMAP_R1_V3_SMOKE_OK",
        {
            "loss_before": float(loss.detach().cpu()),
            "loss_after": float(after_loss.detach().cpu()),
            "attention_change": float(attention_change.cpu()),
            "patch_key_grad": float(direct_grad.abs().mean().cpu()),
            "attention_entropy": float(
                before["spatial_attention_entropy"].mean().detach().cpu()
            ),
            "spatial_logit_abs_mean": float(
                before["spatial_clip_logits"].abs().mean().detach().cpu()
            ),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
