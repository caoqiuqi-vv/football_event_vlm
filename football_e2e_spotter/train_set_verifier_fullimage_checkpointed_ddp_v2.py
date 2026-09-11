#!/usr/bin/env python3
"""DDP full-image/no-ROI verifier with uncertainty-aware temporal training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from dinov3.hub.backbones import dinov3_vitl16
from football_e2e_spotter.set_spotting import NO_EVENT_INDEX
from football_e2e_spotter.verifier_data import VerifierCandidateDataset, assign_oof_targets, read_rows
from football_e2e_spotter.verifier_fullimage_checkpointed import CheckpointedFullImageDinoVerifier
from football_e2e_spotter.verifier_lora_runtime import configure


def uncertainty_time_nll(delta: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    """Robust Laplace-like temporal NLL; log_sigma is bounded by the model."""
    robust_error = F.smooth_l1_loss(delta, torch.zeros_like(delta), reduction="none")
    return (torch.exp(-log_sigma) * robust_error + log_sigma).mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--weights", default="checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--frame-chunk-size", type=int, default=4)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank, local_rank = dist.get_rank(), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    rows = assign_oof_targets(read_rows(args.candidates), args.annotations)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (output / "oof_target_summary.json").write_text(json.dumps({
            "rows": len(rows), "matched": sum(row["matched"] for row in rows),
            "no_temporal_nms": True, "verifier_variant": "full_image_only",
            "roi_branch": False, "frame_chunk_size": args.frame_chunk_size,
            "temporal_loss": "uncertainty_aware_smooth_l1_nll",
        }, indent=2) + "\n")

    dataset = VerifierCandidateDataset(rows, 512, 896)
    sampler = DistributedSampler(dataset, shuffle=True, drop_last=False)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, pin_memory=True)
    backbone = dinov3_vitl16(pretrained=True, weights=args.weights, check_hash=False)
    model = CheckpointedFullImageDinoVerifier(backbone, audio_dim=64, frame_chunk_size=args.frame_chunk_size).to(device)
    configure(backbone, warmup=False, lora_rank=16)
    trainable_backbone = {name for name, value in backbone.named_parameters() if value.requires_grad}
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False, find_unused_parameters=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.02)

    for epoch in range(args.epochs):
        if epoch == args.warmup_epochs:
            for name, parameter in model.module.backbone.named_parameters():
                parameter.requires_grad_(name in trainable_backbone)
        sampler.set_epoch(epoch)
        model.train()
        total = torch.zeros((), device=device)
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = model(batch["frames"], batch["candidate"], batch["shared"], batch["audio"])
                positive = batch["label"] != NO_EVENT_INDEX
                loss = F.cross_entropy(result["class_logits"], batch["label"])
                loss = loss + .5 * F.binary_cross_entropy_with_logits(result["quality_logits"], positive.float())
                if positive.any():
                    residual = result["time_delta_sec"][positive] - batch["delta"][positive].clamp(-2, 2)
                    loss = loss + 2 * uncertainty_time_nll(residual, result["log_sigma"][positive])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += loss.detach()
        dist.all_reduce(total)
        total /= dist.get_world_size()
        if rank == 0:
            torch.save({
                "epoch": epoch, "model": model.module.state_dict(), "no_temporal_nms": True,
                "world_size": dist.get_world_size(), "verifier_variant": "full_image_only",
                "roi_branch": False, "frame_chunk_size": args.frame_chunk_size,
                "temporal_loss": "uncertainty_aware_smooth_l1_nll",
            }, output / "last.pt")
            print(json.dumps({"epoch": epoch, "mean_rank_loss": float(total / max(len(loader), 1)), "world_size": dist.get_world_size(), "roi_branch": False}), flush=True)
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
