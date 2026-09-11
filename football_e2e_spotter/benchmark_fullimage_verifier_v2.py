#!/usr/bin/env python3
"""One-step VRAM preflight for the strict full-image verifier contract."""

from __future__ import annotations

import argparse
import json

import torch

from dinov3.hub.backbones import dinov3_vitl16
from football_e2e_spotter.verifier_fullimage import FullImageDinoVerifier
from football_e2e_spotter.verifier_lora_runtime import configure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-backbone", action="store_true")
    parser.add_argument("--weights", default="checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
    args = parser.parse_args()
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    backbone = dinov3_vitl16(pretrained=True, weights=args.weights, check_hash=False)
    model = FullImageDinoVerifier(backbone, audio_dim=64).to(device)
    if args.train_backbone:
        configure(backbone, warmup=False, lora_rank=16)
    else:
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
    model.train()
    frames = torch.randn(1, 25, 3, 512, 896, device=device)
    candidate = torch.randn(1, 512, device=device)
    shared = torch.randn(1, 64, 512, device=device)
    audio = torch.randn(1, 1, 64, device=device)
    result = model(frames, candidate, shared, audio)
    loss = result["class_logits"].float().square().mean() + result["time_delta_sec"].float().square().mean()
    loss.backward()
    torch.cuda.synchronize(device)
    print(json.dumps({
        "full_image_only": True,
        "train_backbone": args.train_backbone,
        "peak_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 3),
        "class_shape": list(result["class_logits"].shape),
        "crop_boxes": result["crop_boxes"],
    }), flush=True)


if __name__ == "__main__":
    main()
