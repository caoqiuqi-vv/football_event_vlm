#!/usr/bin/env python3
"""Long-video evaluator with exact structured-ROI checkpoint restoration."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import eval_long_video_checkpoint as evaluator  # noqa: E402
import train_football_events as football  # noqa: E402
from train_football_events_featuremap_structured_roi import load_structured_roi_model  # noqa: E402


def load_structured_checkpoint(checkpoint_path: str, device: torch.device, gpu_ids: list[int]):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint or "config" not in checkpoint:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    cfg = evaluator.to_config(checkpoint["config"])
    evaluator.configure_label_schema(cfg)
    architecture = str(cfg.model.get("architecture_version", ""))
    if architecture != "featuremap_structured_roi_temporal":
        return evaluator._original_load_checkpoint_model(checkpoint_path, device, gpu_ids)

    labels = list(checkpoint.get("labels", football.LABELS))
    thresholds = checkpoint.get("thresholds") or {label: 0.5 for label in labels}
    thresholds = {str(label): float(value) for label, value in thresholds.items()}
    old_model_init = cfg.model.get("init_checkpoint", "")
    structured_cfg = cfg.model.featuremap_structured
    old_structured_init = structured_cfg.get("init_checkpoint", "")
    cfg.model.init_checkpoint = ""
    structured_cfg.init_checkpoint = checkpoint_path
    try:
        model = load_structured_roi_model(cfg, device)
    finally:
        cfg.model.init_checkpoint = old_model_init
        structured_cfg.init_checkpoint = old_structured_init

    source = football.strip_module_prefix(checkpoint["model"])
    target = model.state_dict()
    missing = sorted(set(target) - set(source))
    unexpected = sorted(set(source) - set(target))
    shape_mismatch = sorted(
        key for key in set(source) & set(target)
        if tuple(source[key].shape) != tuple(target[key].shape)
    )
    if missing or unexpected or shape_mismatch:
        raise RuntimeError(
            "Structured checkpoint did not restore exactly: "
            f"missing={missing[:12]} unexpected={unexpected[:12]} "
            f"shape_mismatch={shape_mismatch[:12]}"
        )
    print(
        f"Verified exact structured checkpoint restore: keys={len(target)} "
        "missing=0 unexpected=0 shape_mismatch=0",
        flush=True,
    )
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])
    model.eval()
    return model, cfg, labels, thresholds


if __name__ == "__main__":
    evaluator._original_load_checkpoint_model = evaluator.load_checkpoint_model
    evaluator.load_checkpoint_model = load_structured_checkpoint
    evaluator.main()
