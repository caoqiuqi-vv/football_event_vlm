from __future__ import annotations

import argparse
import json

import torch

from football_longform_v2.config import load_config
from football_longform_v2.decoding import decode_proposals
from football_longform_v2.losses import locator_loss
from football_longform_v2.models import TemporalLocator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    config = load_config(args.config)
    torch.manual_seed(int(config["seed"]))
    device = torch.device(args.device)
    model = TemporalLocator.from_config(config).to(device)
    batch, steps = 2, 121
    context_dim = int(config["features"]["context"]["dim"])
    motion_dim = int(config["features"]["motion"]["dim"])
    timestamps = torch.arange(steps, device=device).float().unsqueeze(0).repeat(batch, 1) / 5.0
    context = torch.randn(batch, steps, context_dim, device=device)
    motion = torch.randn(batch, steps, motion_dim, device=device)
    targets = torch.zeros(
        batch, steps, len(config["task"]["proposal_families"]), device=device
    )
    targets[:, 30, 0] = 1.0
    targets[:, 70, 1] = 1.0
    targets[:, 100, 2] = 1.0
    class_targets = torch.zeros(
        batch, steps, len(config["task"]["output_labels"]), device=device
    )
    for label_index in range(class_targets.shape[-1]):
        class_targets[:, 10 + 15 * label_index, label_index] = 1.0
    outputs = model(context, motion, timestamps)
    losses = locator_loss(
        outputs, targets, class_targets=class_targets,
        class_valid_mask=torch.ones_like(class_targets, dtype=torch.bool),
        state_targets=class_targets,
        state_valid_mask=torch.ones_like(class_targets, dtype=torch.bool),
        state_weight=0.1,
        alpha=0.75, density_balanced=True, class_weight=0.5,
    )
    losses["loss"].backward()
    gradient_parameters = sum(
        1 for parameter in model.parameters()
        if parameter.grad is not None and parameter.grad.abs().sum() > 0
    )
    if gradient_parameters == 0:
        raise RuntimeError("no non-zero gradients in locator smoke train")
    if not any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.save_conditioner.parameters()
    ):
        raise RuntimeError("save conditioner received no gradient")
    proposals = decode_proposals(
        outputs["rgb_logits"].detach(), timestamps,
        tuple(config["task"]["proposal_families"]),
        threshold=float(config["decode"]["threshold"]),
        nms_radius_seconds=config["decode"]["nms_radius_seconds"],
        max_per_minute=config["decode"]["max_proposals_per_minute"],
    )
    print(json.dumps({
        "loss": float(losses["loss"].detach()),
        "family_output_shape": list(outputs["rgb_logits"].shape),
        "class_output_shape": list(outputs["class_logits"].shape),
        "class_heatmap_loss": float(losses["class_heatmap_loss"].detach()),
        "state_heatmap_loss": float(losses["state_heatmap_loss"].detach()),
        "nonzero_gradient_parameters": gradient_parameters,
        "proposal_counts": [len(items) for items in proposals],
        "device": str(device),
    }, indent=2))


if __name__ == "__main__":
    main()
