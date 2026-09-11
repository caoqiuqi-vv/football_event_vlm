from __future__ import annotations

import torch
from torch import nn

from football_object_motion.ball_backbone import BallLoRALinear
from football_object_motion.model import ObjectMotionEvidenceAdapter
from football_object_motion.teacher import OnlineObjectMotionTeacher


def _adapter() -> ObjectMotionEvidenceAdapter:
    return ObjectMotionEvidenceAdapter(
        patch_dim=16,
        hidden_dim=16,
        num_labels=3,
        num_heads=4,
        temporal_layers=1,
        dropout=0.0,
        heatmap_upsample_factor=2,
        ball_fpn_dim=16,
        object_cross_attention_enabled=True,
        object_cross_attention_max_delta=0.35,
        object_cross_attention_gate_init=0.25,
    )


def test_stride8_fpn_shapes_and_detector_gradient() -> None:
    adapter = _adapter()
    patches = torch.randn(2, 3, 6, 16, requires_grad=True)
    layers = torch.randn(2, 3, 4, 6, 16, requires_grad=True)
    times = torch.arange(3, dtype=torch.float32).repeat(2, 1)

    outputs = adapter(
        patches,
        times,
        grid_h=2,
        grid_w=3,
        ball_patch_layers=layers,
    )

    assert outputs["heatmap_logits"].shape == (2, 3, 24, 3)
    assert outputs["ball_lora_logits"].shape == (2, 3, 24)
    assert outputs["ball_student_features"].shape == (2, 3, 24, 16)
    assert outputs["object_tokens"].shape == (2, 3, 2, 16)
    assert outputs["object_visibility"].shape == (2, 3, 2)
    assert outputs["centers"].shape == (2, 3, 3, 2)

    outputs["heatmap_logits"][..., 0].sum().backward()
    assert layers.grad is not None
    assert float(layers.grad.abs().sum()) > 0.0


def test_event_cross_attention_detaches_detector_tokens() -> None:
    adapter = _adapter()
    # A zero-initialized correction is safe at startup. Make it nonzero here
    # so the test can inspect the downstream gradient boundary.
    with torch.no_grad():
        adapter.object_cross_attention.delta_head[-1].weight.fill_(0.01)
    patches = torch.randn(1, 3, 6, 16, requires_grad=True)
    layers = torch.randn(1, 3, 4, 6, 16, requires_grad=True)
    global_event = torch.randn(1, 16, requires_grad=True)
    times = torch.arange(3, dtype=torch.float32).reshape(1, 3)

    motion = adapter(
        patches,
        times,
        grid_h=2,
        grid_w=3,
        ball_patch_layers=layers,
    )
    fused = adapter.fuse_global_event(global_event, motion)
    fused["correction"].sum().backward()

    assert global_event.grad is not None
    assert float(global_event.grad.abs().sum()) > 0.0
    assert patches.grad is None
    assert layers.grad is None
    assert fused["attention"].shape == (1, 3, 9)


def test_shared_anchor_ball_lora_receives_event_gradient() -> None:
    overlay = BallLoRALinear(nn.Linear(8, 8), rank=2, alpha=2.0)
    overlay.enabled = True
    inputs = torch.randn(4, 8)
    overlay(inputs).square().mean().backward()

    assert overlay.ball_lora_b.grad is not None
    assert float(overlay.ball_lora_b.grad.abs().sum()) > 0.0


def test_teacher_misses_can_be_unknown() -> None:
    teacher = OnlineObjectMotionTeacher(
        device=torch.device("cpu"),
        ball_checkpoint="ball.pt",
        scene_checkpoint="scene.pt",
        patch_size=8,
        absence_presence_weights=(0.0, 0.0, 0.0),
        half=False,
    )
    assert teacher.absence_presence_weights == (0.0, 0.0, 0.0)
