#!/usr/bin/env python
"""Check full-model anchor equivalence after loading fromlast_e8 plus sparse ROI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_football_events import configure_label_schema, load_config, make_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(
        args.config,
        [
            f"model.init_checkpoint={args.checkpoint}",
            "model.init_checkpoint_strict=false",
            "model.freeze_loaded_backbone=true",
            "model.freeze_global_branch=true",
            "model.controlled_online_train_scope=spatial_residual",
            "model.spatial_attention.enabled=true",
            "model.spatial_attention.mode=residual",
            "model.spatial_attention.attention_mode=topk_softmax",
            "model.spatial_attention.topk_ratio=0.03",
            "model.spatial_attention.queries_per_class=2",
            "model.spatial_attention.residual_max_delta=1.0",
            "model.spatial_attention.return_attention_maps=true",
        ],
    )
    configure_label_schema(cfg)
    device = torch.device(args.device)
    model = make_model(cfg, use_cached_features=False, device=device).eval()
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    unexpected_trainable = [
        name for name in trainable if not name.startswith("spatial_attention.")
    ]
    if unexpected_trainable:
        raise AssertionError(f"Non-ROI parameters remain trainable: {unexpected_trainable[:20]}")
    inputs = torch.randn(1, 2, 3, 224, 224, device=device)
    with torch.inference_mode():
        outputs = model(inputs, return_aux=True)
    anchor = outputs["retention_reference_logits"]
    final = outputs["logits"]
    delta = outputs["spatial_residual_logits"]
    max_anchor_error = float((final - anchor).abs().max())
    max_init_delta = float(delta.abs().max())
    active_fraction = float(outputs["spatial_active_patch_fraction"])
    if max_anchor_error != 0.0 or max_init_delta != 0.0:
        raise AssertionError(
            f"Sparse ROI changed anchor at initialization: error={max_anchor_error} "
            f"delta={max_init_delta}"
        )
    if abs(active_fraction - 0.03) > 0.002:
        raise AssertionError(f"Unexpected active patch fraction: {active_fraction}")
    print(
        "sparse_roi_full_model_init=PASS "
        f"trainable_params={sum(p.numel() for p in model.parameters() if p.requires_grad)} "
        f"trainable_tensors={len(trainable)} max_anchor_error={max_anchor_error} "
        f"active_patch_fraction={active_fraction:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
