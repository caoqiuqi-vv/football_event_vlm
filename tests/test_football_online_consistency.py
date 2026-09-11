from __future__ import annotations

import torch

from football_online_consistency import (
    consistency_warmup_factor,
    online_pair_consistency_loss,
)


def _paired_batch() -> dict:
    return {
        "targets": torch.tensor(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        ),
        "label_masks": torch.ones(3, 3),
        "frame_times": torch.tensor(
            [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [10.0, 11.0, 12.0]]
        ),
        "meta": [
            {
                "online_pair_id": "video:event",
                "online_pair_role": "central",
                "online_pair_class_mask": (1.0, 0.0, 0.0),
                "online_pair_anchor_time": 1.5,
            },
            {
                "online_pair_id": "video:event",
                "online_pair_role": "edge",
                "online_pair_class_mask": (1.0, 0.0, 0.0),
                "online_pair_anchor_time": 1.5,
            },
            {},  # Unrelated clean background may share the batch.
        ],
    }


def test_clip_consistency_is_asymmetric_and_has_positive_floor() -> None:
    logits = torch.tensor(
        [[-2.0, 0.0, 0.0], [-3.0, 0.0, 0.0], [4.0, 4.0, 4.0]],
        requires_grad=True,
    )
    loss, metrics = online_pair_consistency_loss(
        {"logits": logits},
        _paired_batch(),
        {
            "enabled": True,
            "clip_weight": 1.0,
            "response_weight": 0.0,
            "teacher_logit_floor": 0.0,
        },
        epoch=1,
        step=1,
        steps_per_epoch=10,
    )
    loss.backward()

    assert metrics["online_pair_consistency_pairs"] == 1.0
    assert metrics["online_pair_consistency_slots"] == 1.0
    assert metrics["online_pair_teacher_floor_fraction"] == 1.0
    assert logits.grad is not None
    assert logits.grad[0].abs().sum() == 0  # stop-gradient central teacher
    assert logits.grad[1, 0] < 0  # edge logit is pulled upward toward floor 0
    assert logits.grad[2].abs().sum() == 0  # unrelated background is untouched


def test_response_curve_is_aligned_in_absolute_time() -> None:
    logits = torch.zeros(3, 3, requires_grad=True)
    curve = torch.zeros(3, 3, 3, requires_grad=True)
    with torch.no_grad():
        curve[0, :, 0] = torch.tensor([0.0, 1.0, 2.0])
        curve[1, :, 0] = torch.tensor([1.0, 2.0, 8.0])
    loss, metrics = online_pair_consistency_loss(
        {"logits": logits, "response_curve_logits": curve},
        _paired_batch(),
        {
            "enabled": True,
            "clip_weight": 0.0,
            "response_weight": 1.0,
            "response_support_radius_sec": 2.0,
        },
        epoch=1,
        step=1,
        steps_per_epoch=10,
    )

    # Only absolute times 1 and 2 overlap; both curves agree there.  The edge
    # value at absolute t=3 is outside the central window and must not leak in.
    assert torch.allclose(loss, torch.zeros_like(loss))
    assert metrics["online_pair_response_points"] == 2.0


def test_warmup_starts_at_requested_epoch() -> None:
    cfg = {"start_epoch": 2, "warmup_steps": 4}
    assert consistency_warmup_factor(
        cfg, epoch=1, step=10, steps_per_epoch=10
    ) == 0.0
    assert consistency_warmup_factor(
        cfg, epoch=2, step=1, steps_per_epoch=10
    ) == 0.25
    assert consistency_warmup_factor(
        cfg, epoch=2, step=4, steps_per_epoch=10
    ) == 1.0
