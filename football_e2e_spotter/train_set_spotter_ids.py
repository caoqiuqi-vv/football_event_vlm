#!/usr/bin/env python3
"""Stage-1 trainer with an explicit ID file, used for leakage-free OOF folds."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from football_e2e_spotter.set_data import TemporalSetCoreDataset, set_collate
from football_e2e_spotter.set_spotting import TemporalSetSpotter, set_spotting_loss


def ids(path: str) -> tuple[str, ...]: return tuple(x.strip() for x in Path(path).read_text().splitlines() if x.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--ids", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=12); parser.add_argument("--batch-size", type=int, default=1); parser.add_argument("--workers", type=int, default=4); parser.add_argument("--device", default="cuda")
    args = parser.parse_args(); cfg = yaml.safe_load(Path(args.config).read_text()); data, stage = cfg["data"], cfg["stage1"]
    train = TemporalSetCoreDataset(ids(args.ids), store_root=data["frame_store"], split="train", annotations=data["annotations"], sequence_frames=int((stage["core_seconds"] + 2 * stage["context_seconds"]) * data["sample_fps"]), sequences_per_video=1, positive_probability=.5, temporal_jitter_seconds=0, seed=cfg["seed"], class_sigma_seconds={}, family_sigma_seconds={}, ignore_radius_seconds={}, max_offset_seconds=0, core_seconds=stage["core_seconds"], context_seconds=stage["context_seconds"], background_ratio=1)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=set_collate, pin_memory=True)
    model = TemporalSetSpotter(sample_fps=data["sample_fps"], core_seconds=stage["core_seconds"], context_seconds=stage["context_seconds"], num_slots=stage["num_slots"], mel_bins=data["audio_mel_bins"], **stage["model"]).to(args.device)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.03); output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        train.set_epoch(epoch); model.train(); loss = torch.zeros((), device=args.device)
        for batch in loader:
            out = model(batch["frames"].to(args.device), batch["audio"].to(args.device), batch["valid"].to(args.device)); loss = set_spotting_loss(out, batch["targets"])["loss"]
            optim.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optim.step()
        torch.save({"epoch": epoch, "model": model.state_dict(), "config": cfg, "ids": list(ids(args.ids))}, output / "last.pt")
        print({"epoch": epoch, "loss": float(loss.detach()), "examples": len(train)}, flush=True)


if __name__ == "__main__": main()
