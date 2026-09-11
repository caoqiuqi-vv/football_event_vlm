#!/usr/bin/env python3
"""VRAM preflight for checkpointed full-image DINOv3 verifier fine-tuning."""

from __future__ import annotations

import argparse
import json

import torch

from dinov3.hub.backbones import dinov3_vitl16
from football_e2e_spotter.verifier_fullimage_checkpointed import CheckpointedFullImageDinoVerifier
from football_e2e_spotter.verifier_lora_runtime import configure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame-chunk-size", type=int, default=4)
    parser.add_argument("--weights", default="checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
    args = parser.parse_args()
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    backbone = dinov3_vitl16(pretrained=True, weights=args.weights, check_hash=False)
    model = CheckpointedFullImageDinoVerifier(backbone, audio_dim=64, frame_chunk_size=args.frame_chunk_size).to(device)
    configure(backbone, warmup=False, lora_rank=16)
    model.train()
    result = model(
        torch.randn(1, 25, 3, 512, 896, device=device),
        torch.randn(1, 512, device=device),
        torch.randn(1, 64, 512, device=device),
        torch.randn(1, 1, 64, device=device),
    )
    (result["class_logits"].float().square().mean() + result["time_delta_sec"].float().square().mean()).backward()
    torch.cuda.synchronize(device)
    print(json.dumps({
        "full_image_only": True, "roi_branch": False, "frames": 25,
        "resolution": [512, 896], "frame_chunk_size": args.frame_chunk_size,
        "peak_gib": round(torch.cuda.max_memory_allocated(device) / 2**30, 3),
    }), flush=True)


if __name__ == "__main__":
    main()
