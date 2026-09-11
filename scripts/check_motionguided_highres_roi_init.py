#!/usr/bin/env python
"""Check exact-anchor initialization of motion-guided high-resolution ROI."""

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
            "model.controlled_online_train_scope=highres_residual",
            "model.highres_glimpse.enabled=true",
            "model.highres_glimpse.candidates=2",
            "model.highres_glimpse.frames_per_candidate=2",
            "model.highres_glimpse.crop_size=224",
            "model.highres_glimpse.attention_dim=128",
            "model.highres_glimpse.nms_radius=0",
            "model.highres_glimpse.fusion_mode=bounded_residual",
            "model.highres_glimpse.residual_max_delta=1.0",
            "model.highres_glimpse.residual_gate_init=0.1",
            "model.highres_glimpse.motion_prior_weight=1.5",
            "model.highres_glimpse.attention_temperature=0.7",
            "model.spatial_attention.enabled=false",
        ],
    )
    configure_label_schema(cfg)
    device = torch.device(args.device)
    model = make_model(cfg, use_cached_features=False, device=device).eval()
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    unexpected_trainable = [
        name
        for name in trainable
        if not name.startswith(
            (
                "highres_crop_", "highres_event_query_embedding",
                "highres_local_", "highres_residual_",
            )
        )
    ]
    if unexpected_trainable:
        raise AssertionError(f"Non-ROI parameters remain trainable: {unexpected_trainable[:20]}")
    inputs = torch.rand(1, 2, 3, 256, 448, device=device)
    highres_pool = torch.randint(
        0, 256, (1, 4, 3, 256, 448), dtype=torch.uint8, device=device
    )
    global_times = torch.tensor([[0.0, 1.0]], device=device)
    pool_times = torch.linspace(0.0, 1.0, 4, device=device).unsqueeze(0)
    with torch.inference_mode():
        outputs = model(
            inputs,
            highres_pool_inputs=highres_pool,
            global_frame_times=global_times,
            highres_pool_times=pool_times,
            return_aux=True,
        )
    anchor = outputs["global_logits"]
    final = outputs["logits"]
    delta = outputs["highres_residual_delta"]
    max_anchor_error = float((final - anchor).abs().max())
    max_init_delta = float(delta.abs().max())
    gate_mean = float(outputs["highres_residual_gate"].mean())
    if max_anchor_error != 0.0 or max_init_delta != 0.0:
        raise AssertionError(
            f"High-resolution ROI changed anchor at initialization: error={max_anchor_error} "
            f"delta={max_init_delta}"
        )
    print(
        "motionguided_highres_roi_init=PASS "
        f"trainable_params={sum(p.numel() for p in model.parameters() if p.requires_grad)} "
        f"trainable_tensors={len(trainable)} max_anchor_error={max_anchor_error} "
        f"residual_gate_mean={gate_mean:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
