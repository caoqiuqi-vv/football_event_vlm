#!/usr/bin/env python
"""Fast unit checks for sparse class-aware ROI residual invariants."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_football_events import TemporalConditionedSpatialAttention


def main() -> None:
    torch.manual_seed(7)
    module = TemporalConditionedSpatialAttention(
        patch_dim=64,
        hidden_dim=64,
        attention_dim=32,
        num_labels=3,
        queries_per_class=2,
        context_layers=1,
        temporal_layers=1,
        num_heads=4,
        dropout=0.0,
        max_frames=16,
        gate_init=0.05,
        dynamic_query_scale_init=0.0,
        actionness_query_scale_init=0.0,
        attention_mode="topk_softmax",
        topk_ratio=0.03,
        residual_max_delta=1.0,
        return_attention_maps=True,
    ).eval()
    frames = torch.randn(2, 4, 64)
    patches = torch.randn(2, 4, 200, 64)
    actionness = torch.randn(2, 4, 3)
    with torch.no_grad():
        outputs = module(frames, patches, actionness)
    residual = outputs["residual_logits"]
    attention = outputs["attention_maps"]
    active = (attention > 0).sum(dim=-1)
    expected_topk = 6
    assert torch.equal(active, torch.full_like(active, expected_topk))
    assert torch.equal(residual, torch.zeros_like(residual))
    assert float(residual.abs().max()) <= 1.0
    assert abs(float(outputs["active_patch_fraction"]) - 0.03) < 1e-7

    if torch.cuda.is_available():
        cuda_module = module.to("cuda")
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            amp_outputs = cuda_module(
                frames.to("cuda"), patches.to("cuda"), actionness.to("cuda")
            )
        assert torch.equal(
            amp_outputs["residual_logits"],
            torch.zeros_like(amp_outputs["residual_logits"]),
        )
        assert torch.equal(
            (amp_outputs["attention_maps"] > 0).sum(dim=-1),
            torch.full_like(
                (amp_outputs["attention_maps"] > 0).sum(dim=-1), expected_topk
            ),
        )
        module = cuda_module.to("cpu")

    with torch.no_grad():
        module.clip_residual_head.weight.fill_(100.0)
        module.clip_residual_head.bias.fill_(100.0)
        bounded = module(frames, patches, actionness)["residual_logits"]
    assert float(bounded.abs().max()) <= 1.0 + 1e-6
    print("sparse_roi_invariants=PASS topk=6/200 init_delta=0 max_abs_delta<=1")


if __name__ == "__main__":
    main()
