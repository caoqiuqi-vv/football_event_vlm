"""Asymmetric multi-label loss aligned to review-block selection."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def asymmetric_focal(logits: Tensor, targets: Tensor, positive_weight: Tensor | None = None) -> Tensor:
    probability = logits.sigmoid()
    positive = -targets * (1.0 - probability).pow(1.0) * F.logsigmoid(logits)
    negative = -(1.0 - targets) * probability.pow(4.0) * F.logsigmoid(-logits)
    if positive_weight is not None:
        positive = positive * positive_weight
    return (positive + 0.25 * negative).mean()


def block_retriever_loss(output: dict[str, Tensor], block_targets: Tensor, dense_targets: Tensor) -> tuple[Tensor, dict[str, float]]:
    counts = block_targets.sum(dim=(0, 1)).clamp_min(1.0)
    class_weight = (counts.sum() / counts).sqrt().clamp(1.0, 6.0).reshape(1, 1, -1)
    block = asymmetric_focal(output["block_logits"], block_targets, class_weight)
    any_target = block_targets.amax(dim=-1)
    any_event = asymmetric_focal(output["any_logits"], any_target)
    dense = asymmetric_focal(output["dense_logits"], dense_targets)
    restart_target = block_targets[..., 2:].amax(dim=-1)
    whistle_blocks = output["whistle_logits"].reshape(output["whistle_logits"].shape[0], 3, 20).amax(dim=-1)
    # Whistle is soft evidence: false negatives are weakly penalized because
    # amateur footage can have missing/inaudible audio.
    whistle = (restart_target * F.softplus(-whistle_blocks) + 0.1 * (1 - restart_target) * F.softplus(whistle_blocks)).mean()
    total = block + 0.35 * any_event + 0.25 * dense + 0.08 * whistle
    return total, {"loss": float(total.detach()), "block": float(block.detach()), "any": float(any_event.detach()), "dense": float(dense.detach()), "whistle": float(whistle.detach())}

