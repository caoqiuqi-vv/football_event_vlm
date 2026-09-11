from __future__ import annotations

import torch

from football_frame_heatmap import positive_core_context_heatmap_loss


def test_positive_core_context_is_equal_weighted_and_shoulder_ignored() -> None:
    matrix = torch.zeros(1, 12, 1, requires_grad=True)
    matrix.data[:, 0] = 2.0
    matrix.data[:, 1] = 99.0
    matrix.data[:, 2:] = 1.0
    targets = torch.zeros_like(matrix)
    targets[:, 0] = 1.0
    targets[:, 1] = 0.3
    loss, diagnostics = positive_core_context_heatmap_loss(
        matrix,
        targets,
        torch.ones_like(targets),
        torch.ones(1, 1),
        torch.ones(1, 1),
        core_min_target=0.5,
        context_max_target=0.1,
    )
    assert torch.allclose(loss, torch.tensor(1.5))
    loss.backward()
    assert matrix.grad is not None
    assert torch.allclose(matrix.grad[:, 0].abs().sum(), torch.tensor(0.5))
    assert torch.allclose(matrix.grad[:, 2:].abs().sum(), torch.tensor(0.5))
    assert matrix.grad[:, 1].abs().sum() == 0
    assert diagnostics["frame_heatmap_core_slots"] == 1
    assert diagnostics["frame_heatmap_context_slots"] == 10


def test_negative_clip_keeps_ordinary_frame_mean() -> None:
    matrix = torch.arange(1.0, 5.0).reshape(1, 4, 1)
    targets = torch.zeros_like(matrix)
    loss, diagnostics = positive_core_context_heatmap_loss(
        matrix,
        targets,
        torch.ones_like(targets),
        torch.zeros(1, 1),
        torch.ones(1, 1),
    )
    assert torch.allclose(loss, matrix.mean())
    assert diagnostics["frame_heatmap_balanced_pos_slots"] == 0
    assert diagnostics["frame_heatmap_balanced_neg_slots"] == 1
