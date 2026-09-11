from __future__ import annotations

import torch
from torch import nn

import train_football_events as football
from train_football_events import ConfigDict, VideoEventClassifier


class _DummyVideoBackbone(nn.Module):
    is_video_backbone = True
    num_features = 8
    output_feature_dim = 16
    global_feature_dim = 8

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))


def _model() -> VideoEventClassifier:
    return VideoEventClassifier(
        backbone=_DummyVideoBackbone(),
        frame_feature_dim=16,
        hidden_dim=16,
        num_labels=3,
        fusion="videomae_residual_transformer",
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        max_frames=4,
        view_fusion="single",
        videomae_temporal_gate_init=0.25,
    )


def test_videomae_residual_starts_as_exact_global_classifier() -> None:
    model = _model().eval()
    features = torch.randn(2, 4, 16)
    outputs = model._global_branch_outputs(features)

    torch.testing.assert_close(outputs["logits"], outputs["global_action_logits"])
    torch.testing.assert_close(
        outputs["temporal_residual_gate"],
        torch.full((2, 3), 0.25),
    )
    assert outputs["frame_event_logits"].shape == (2, 4, 3)


def test_frame_localization_does_not_consume_repeated_global_feature() -> None:
    model = _model().eval()
    features = torch.randn(2, 4, 16)
    changed_global = features.clone()
    changed_global[..., :8] += torch.arange(8, dtype=features.dtype)

    original = model._global_branch_outputs(features)
    changed = model._global_branch_outputs(changed_global)

    torch.testing.assert_close(
        original["frame_event_logits"], changed["frame_event_logits"]
    )
    assert not torch.allclose(
        original["global_action_logits"], changed["global_action_logits"]
    )


def test_zero_initialized_delta_unblocks_temporal_gradient_after_one_step() -> None:
    model = _model().train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    features = torch.randn(2, 4, 16)

    first = model._global_branch_outputs(features)
    first["logits"].square().mean().backward()
    assert model.video_temporal_delta_head is not None
    assert model.video_temporal_delta_head[-1].weight.grad is not None
    assert model.video_temporal_delta_head[-1].weight.grad.abs().sum() > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    second = model._global_branch_outputs(features)
    second["logits"].square().mean().backward()
    temporal_gradient = sum(
        parameter.grad.abs().sum()
        for parameter in model.temporal.parameters()
        if parameter.grad is not None
    )
    assert temporal_gradient > 0


def test_temporal_only_supervision_reaches_transformer_after_warmup_step() -> None:
    model = _model().train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    features = torch.randn(4, 4, 16)
    targets = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
    )

    first = model._global_branch_outputs(features)
    nn.functional.binary_cross_entropy_with_logits(
        first["temporal_delta_logits"], targets
    ).backward()
    assert model.video_temporal_delta_head is not None
    assert model.video_temporal_delta_head[-1].weight.grad is not None
    assert model.video_temporal_delta_head[-1].weight.grad.abs().sum() > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    second = model._global_branch_outputs(features)
    nn.functional.binary_cross_entropy_with_logits(
        second["temporal_delta_logits"], targets
    ).backward()
    temporal_gradient = sum(
        parameter.grad.abs().sum()
        for parameter in model.temporal.parameters()
        if parameter.grad is not None
    )
    assert temporal_gradient > 0


class _SyntheticTemporalEvaluationModel(nn.Module):
    def forward(self, inputs: torch.Tensor, *, return_aux: bool = False, **_: object):
        assert return_aux
        batch = inputs.shape[0]
        global_logits = torch.zeros(batch, 3, device=inputs.device)
        fused_logits = torch.full((batch, 3), -4.0, device=inputs.device)
        fused_logits[0, 0] = 4.0
        fused_logits[1, 1] = 4.0
        fused_logits[2, 2] = 4.0
        return {
            "logits": fused_logits,
            "global_logits": fused_logits,
            "global_action_logits": global_logits,
            "temporal_delta_logits": fused_logits - global_logits,
            "temporal_residual_gate": torch.full_like(fused_logits, 0.25),
            "frame_event_logits": torch.zeros(
                batch, 4, 3, device=inputs.device
            ),
        }


def test_evaluate_reports_distinct_global_and_temporal_branches() -> None:
    football.configure_label_schema(
        ConfigDict({"task": {"label_schema": "set_piece"}})
    )
    targets = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    batch = {
        "inputs": torch.zeros(6, 1),
        "targets": targets,
        "label_masks": torch.ones_like(targets),
        "meta": [
            {"source": "synthetic", "video_id": f"video_{index}"}
            for index in range(6)
        ],
    }
    cfg = ConfigDict(
        {
            "model": ConfigDict({
                "temporal_fusion": "videomae_residual_transformer",
                "spatial_attention": ConfigDict({"enabled": False}),
            }),
            "train": ConfigDict({
                "amp": False,
                "amp_dtype": "bf16",
                "frame_eval_topk": 4,
            }),
            "eval": ConfigDict({"threshold": 0.5}),
        }
    )

    metrics = football.evaluate(
        _SyntheticTemporalEvaluationModel(), [batch], cfg, torch.device("cpu")
    )

    branches = metrics["temporal_branches"]
    assert set(branches) == {
        "global_action",
        "temporal_only",
        "global_plus_temporal",
    }
    assert branches["temporal_only"]["tuned"]["mAP"] == metrics["tuned"]["mAP"]
    assert branches["global_plus_temporal"]["tuned"]["mAP"] == metrics["tuned"]["mAP"]
    assert (
        branches["global_plus_temporal"]["tuned"]["mAP"]
        > branches["global_action"]["tuned"]["mAP"]
    )
    assert set(metrics["temporal_gate_stats"]) == {"shot", "save", "set_piece"}
