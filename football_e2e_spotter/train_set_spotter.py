"""Single-node training entrypoint for the NMS-free Stage-1 Spotter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

from football_e2e_spotter.set_data import TemporalSetCoreDataset, set_collate
from football_e2e_spotter.set_spotting import TemporalSetSpotter, set_spotting_loss


def _ids(path: str) -> tuple[str, ...]:
    return tuple(line.strip() for line in Path(path).read_text().splitlines() if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    data_cfg, stage_cfg = cfg["data"], cfg["stage1"]
    dataset = TemporalSetCoreDataset(
        _ids(data_cfg["train_ids"]), store_root=data_cfg["frame_store"], split="train", annotations=data_cfg["annotations"],
        sequence_frames=int((stage_cfg["core_seconds"] + 2 * stage_cfg["context_seconds"]) * data_cfg["sample_fps"]), sequences_per_video=1,
        positive_probability=.5, temporal_jitter_seconds=0, seed=int(cfg["seed"]), class_sigma_seconds={}, family_sigma_seconds={}, ignore_radius_seconds={}, max_offset_seconds=0,
        core_seconds=stage_cfg["core_seconds"], context_seconds=stage_cfg["context_seconds"], background_ratio=1.0,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=set_collate, pin_memory=True)
    model = TemporalSetSpotter(sample_fps=data_cfg["sample_fps"], core_seconds=stage_cfg["core_seconds"], context_seconds=stage_cfg["context_seconds"], num_slots=stage_cfg["num_slots"], mel_bins=data_cfg["audio_mel_bins"], **stage_cfg["model"]).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.03)
    output_dir = Path(cfg["output_dir"]); output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    for epoch in range(args.epochs):
        dataset.set_epoch(epoch); model.train()
        for batch in loader:
            output = model(batch["frames"].to(args.device), batch["audio"].to(args.device), batch["valid"].to(args.device))
            loss = set_spotting_loss(output, batch["targets"])["loss"]
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        torch.save({"epoch": epoch, "model": model.state_dict(), "config": cfg}, output_dir / "last.pt")
        print(json.dumps({"epoch": epoch, "loss": float(loss.detach())}), flush=True)


if __name__ == "__main__":
    main()
