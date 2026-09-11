#!/usr/bin/env python
"""Structured ROI spatial-token and top-k temporal fusion experiment."""

from __future__ import annotations

from pathlib import Path
import argparse
import json

import torch
from torch import nn

import train_football_events as football
from football_sparsemax_roi_residual import upgrade_roi_only_residual
from football_structured_roi_temporal import (
    StructuredROITemporalFusion,
    upgrade_structured_roi_attention,
)


_load_config = football.load_config


class StructuredROIVideoEventClassifier(football.VideoEventClassifier):
    def _adaspot_spatial_feature_fusion(
        self,
        global_outputs: dict[str, torch.Tensor],
        spatial_outputs: dict[str, torch.Tensor],
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        return self.spatial_feature_structured(global_outputs, spatial_outputs)


def load_structured_roi_model(cfg, device):
    init_path = str(cfg.model.featuremap_structured.init_checkpoint)
    if not Path(init_path).is_file():
        raise FileNotFoundError(init_path)

    original_class = football.VideoEventClassifier
    football.VideoEventClassifier = StructuredROIVideoEventClassifier
    try:
        cfg.model.init_checkpoint = ""
        model = football.make_model(
            cfg, use_cached_features=False, device=device
        )
    finally:
        football.VideoEventClassifier = original_class

    model.spatial_attention = upgrade_roi_only_residual(
        model.spatial_attention, sparsemax_temperature=4.0
    ).to(device)
    checkpoint = torch.load(init_path, map_location="cpu", weights_only=True)
    checkpoint_state = football.strip_module_prefix(
        checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    )
    structured_init = any(
        key.startswith("spatial_feature_structured.")
        for key in checkpoint_state
    )
    if not structured_init:
        football.load_model_init_checkpoint(
            model,
            init_path,
            checkpoint=checkpoint,
            expected_backbone=str(cfg.model.backbone),
            strict=True,
        )

    image_h, image_w = football.parse_image_size(cfg.video.image_size)
    spatial_cfg = cfg.model.featuremap_structured
    spatial_grid = tuple(
        int(value) for value in spatial_cfg.get("spatial_grid", [3, 5])
    )
    model.spatial_attention = upgrade_structured_roi_attention(
        model.spatial_attention,
        patch_grid=(image_h // 16, image_w // 16),
        spatial_grid=spatial_grid,
        sparsemax_temperature=float(
            spatial_cfg.get("sparsemax_temperature", 4.0)
        ),
        bin_bandwidth=float(spatial_cfg.get("bin_bandwidth", 0.55)),
    ).to(device)

    for name in (
        "spatial_feature_global_align",
        "spatial_feature_local_align",
        "spatial_feature_output",
    ):
        module = getattr(model, name, None)
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad = False

    model.spatial_feature_structured = StructuredROITemporalFusion(
        hidden_dim=int(cfg.model.hidden_dim),
        num_labels=len(football.LABELS),
        num_frames=int(cfg.video.num_frames),
        queries_per_class=int(
            cfg.model.spatial_attention.queries_per_class
        ),
        spatial_grid=spatial_grid,
        token_dim=int(spatial_cfg.get("token_dim", 256)),
        topk_frames=int(spatial_cfg.get("topk_frames", 4)),
        exploration_frames=int(
            spatial_cfg.get("exploration_frames", 2)
        ),
        temporal_layers=int(spatial_cfg.get("temporal_layers", 2)),
        num_heads=int(spatial_cfg.get("num_heads", 8)),
        dropout=float(cfg.model.dropout),
        gate_init=float(spatial_cfg.get("gate_init", 0.10)),
        max_clip_delta=float(spatial_cfg.get("max_clip_delta", 1.0)),
        max_frame_delta=float(spatial_cfg.get("max_frame_delta", 0.75)),
    ).to(device)
    if structured_init:
        football.load_model_init_checkpoint(
            model,
            init_path,
            checkpoint=checkpoint,
            expected_backbone=str(cfg.model.backbone),
            strict=True,
        )
    model.spatial_attention_mode = "adaspot_feature_fusion"
    model.freeze_global_parameters()

    trainable = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    expected_prefixes = (
        "spatial_attention.",
        "spatial_feature_structured.",
    )
    unexpected = [
        name for name in trainable
        if not name.startswith(expected_prefixes)
    ]
    if unexpected:
        raise RuntimeError(
            "Structured ROI experiment has unexpected trainable parameters: "
            f"{unexpected[:20]}"
        )
    print(
        f"Loaded structured ROI init={init_path} "
        f"epoch={checkpoint.get('epoch')} matched_strict=true "
        f"structured_init={structured_init} "
        f"patch_grid={(image_h // 16, image_w // 16)} "
        f"spatial_grid={spatial_grid} trainable_count={len(trainable)}",
        flush=True,
    )
    return model.to(device)


def load_structured_roi_config(path, overrides):
    cfg = _load_config(path, overrides)
    cfg.model.spatial_attention.mode = "adaspot_feature_fusion"
    cfg.model.spatial_attention.return_attention_maps = True
    cfg.model.freeze_spatial_probe = False
    cfg.model.architecture_version = "featuremap_structured_roi_temporal"
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_structured_roi_config(args.config, args.overrides)
    football.configure_runtime_threads(cfg)
    football.configure_label_schema(cfg)
    device = torch.device(cfg.device)
    football.resolve_runtime_topology(cfg, device)
    football.seed_everything(int(cfg.seed), bool(cfg.deterministic))
    if args.dry_run:
        football.run_dry_run(cfg)
        return
    train_dataset, val_dataset, train_records, val_records = (
        football.prepare_datasets(cfg, use_cache=False)
    )
    print("train", football.summarize_records(train_records), flush=True)
    print("val", football.summarize_records(val_records), flush=True)
    train_loader = football.make_loader(train_dataset, cfg, is_train=True)
    val_loader = football.make_loader(
        val_dataset, cfg, is_train=False, batch_size=int(cfg.eval.batch_size)
    )
    model = load_structured_roi_model(cfg, device)
    gpu_ids = [int(value) for value in cfg.get("gpu_ids", [])]
    if len(gpu_ids) > 1:
        model = nn.DataParallel(
            model, device_ids=gpu_ids, output_device=gpu_ids[0]
        )
        print(f"Using structured ROI DataParallel gpu_ids={gpu_ids}", flush=True)
    if args.eval_only:
        metrics = football.evaluate(model, val_loader, cfg, device)
        path = Path(cfg.output_dir) / "eval_only_metrics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
        return
    football.train(
        model, train_loader, val_loader, cfg, device, train_records
    )


if __name__ == "__main__":
    main()

