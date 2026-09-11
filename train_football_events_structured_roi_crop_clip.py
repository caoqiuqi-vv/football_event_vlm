#!/usr/bin/env python
"""Train frozen-E2 dual-query ROI crop features for clip-only correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

import train_football_events as football
from football_structured_roi_crop_clip import (
    load_structured_roi_crop_clip_model,
)
from train_football_events_featuremap_structured_roi import (
    load_structured_roi_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def build_clip_only_optimizer(
    model: nn.Module, cfg: Any
) -> torch.optim.Optimizer:
    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not parameters:
        raise RuntimeError("clip-only ROI model has no trainable parameters")
    return torch.optim.AdamW(
        [
            {
                "name": "roi_clip_fusion",
                "params": parameters,
                "lr": float(cfg.train.lr),
            }
        ],
        weight_decay=float(cfg.train.weight_decay),
    )


def validate_decoupled_config(cfg: Any) -> None:
    forbidden = {
        "frame_det_loss_weight": cfg.train.get(
            "frame_det_loss_weight", 0.0
        ),
        "spatial_temporal_localization_loss_weight": cfg.train.get(
            "spatial_temporal_localization_loss_weight", 0.0
        ),
        "structured_frame_temporal_localization_loss_weight": cfg.train.get(
            "structured_frame_temporal_localization_loss_weight", 0.0
        ),
        "spatial_attention_mil_loss_weight": cfg.train.get(
            "spatial_attention_mil_loss_weight", 0.0
        ),
        "spatial_attention_query_diversity_loss_weight": cfg.train.get(
            "spatial_attention_query_diversity_loss_weight", 0.0
        ),
        "spatial_attention_concentration_loss_weight": cfg.train.get(
            "spatial_attention_concentration_loss_weight", 0.0
        ),
        "spatial_attention_overlap_loss_weight": cfg.train.get(
            "spatial_attention_overlap_loss_weight", 0.0
        ),
        "spatial_counterfactual_loss_weight": cfg.train.get(
            "spatial_counterfactual_loss_weight", 0.0
        ),
        "global_conditioned_correction_loss_weight": cfg.train.get(
            "global_conditioned_correction_loss_weight", 0.0
        ),
    }
    active = {
        key: float(value or 0.0)
        for key, value in forbidden.items()
        if float(value or 0.0) != 0.0
    }
    if active:
        raise ValueError(
            "clip-only ROI experiment forbids frame/spatial proposer losses: "
            f"{active}"
        )
    if float(cfg.train.get("spatial_clip_loss_weight", 0.0) or 0.0) <= 0:
        raise ValueError(
            "ROI crop classifier requires spatial_clip_loss_weight > 0"
        )


def smoke_test(
    model: nn.Module,
    loader,
    cfg: Any,
    device: torch.device,
) -> None:
    batch = next(iter(loader))
    model.eval()
    with torch.inference_mode(), football.autocast_context(
        device, bool(cfg.train.amp), str(cfg.train.amp_dtype)
    ):
        outputs = football.forward_model_batch(
            model, batch, device, return_aux=True
        )
    if not isinstance(outputs, dict):
        raise RuntimeError("smoke test expected auxiliary dictionary")
    global_frame = outputs["global_frame_event_logits"]
    public_frame = outputs["frame_event_logits"]
    max_frame_difference = float(
        (public_frame - global_frame).abs().max().cpu()
    )
    if max_frame_difference != 0.0:
        raise RuntimeError(
            "ROI crop branch changed global frame detection: "
            f"max_difference={max_frame_difference}"
        )
    print(
        json.dumps(
            {
                "logits": list(outputs["logits"].shape),
                "global_logits": list(outputs["global_logits"].shape),
                "frame_logits": list(public_frame.shape),
                "roi_indices": list(outputs["roi_indices"].shape),
                "roi_crop_params": list(
                    outputs["roi_crop_params"].shape
                ),
                "roi_selected_attention": list(
                    outputs["roi_selected_attention"].shape
                ),
                "max_global_frame_difference": max_frame_difference,
                "trainable_parameters": sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    cfg = load_structured_roi_config(args.config, args.overrides)
    validate_decoupled_config(cfg)
    football.configure_runtime_threads(cfg)
    football.configure_label_schema(cfg)
    device = torch.device(cfg.device)
    football.resolve_runtime_topology(cfg, device)
    football.seed_everything(int(cfg.seed), bool(cfg.deterministic))

    train_dataset, val_dataset, train_records, val_records = (
        football.prepare_datasets(cfg, use_cache=False)
    )
    print("train", football.summarize_records(train_records), flush=True)
    print("val", football.summarize_records(val_records), flush=True)
    train_loader = football.make_loader(
        train_dataset, cfg, is_train=True
    )
    val_loader = football.make_loader(
        val_dataset,
        cfg,
        is_train=False,
        batch_size=int(cfg.eval.batch_size),
    )
    model: nn.Module = load_structured_roi_crop_clip_model(cfg, device)
    gpu_ids = [int(value) for value in cfg.get("gpu_ids", [])]
    if len(gpu_ids) > 1 and not args.smoke:
        model = nn.DataParallel(
            model,
            device_ids=gpu_ids,
            output_device=gpu_ids[0],
        )
        print(
            f"Using ROI crop clip DataParallel gpu_ids={gpu_ids}",
            flush=True,
        )

    if args.dry_run:
        print("dry_run=ok", flush=True)
        return
    if args.smoke:
        smoke_test(model, train_loader, cfg, device)
        return

    football.build_optimizer = build_clip_only_optimizer
    if args.eval_only:
        metrics = football.evaluate(
            model, val_loader, cfg, device
        )
        output_path = Path(cfg.output_dir) / "eval_only_metrics.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2)
        )
        print(f"metrics={output_path}", flush=True)
        return
    football.train(
        model,
        train_loader,
        val_loader,
        cfg,
        device,
        train_records,
    )


if __name__ == "__main__":
    main()
