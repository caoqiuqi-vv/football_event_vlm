from __future__ import annotations

import torch

import train_football_events as football
from football_no_evidence import balanced_no_evidence_loss


def _head(*, anti_erasure: bool) -> football.IndependentPreprojectionEvidenceHead:
    torch.manual_seed(7)
    return football.IndependentPreprojectionEvidenceHead(
        patch_dim=8,
        raw_feature_dim=32,
        num_feature_layers=2,
        class_dim=64,
        num_labels=3,
        queries_per_class=1,
        temporal_layers=1,
        temporal_heads=4,
        dropout=0.0,
        max_frames=3,
        bottleneck_frames=2,
        anti_erasure_no_evidence=anti_erasure,
    ).eval()


def test_anti_erasure_appearance_remains_patch_dependent_at_high_null_mass() -> None:
    head = _head(anti_erasure=True)
    raw = torch.randn(1, 3, 32)
    patch_a = torch.randn(1, 3, 5, 8)
    patch_b = patch_a.clone()
    patch_b[:, :, 0, 0] += 2.0
    assert head.no_evidence_content_scorer is not None
    initial_content_a = head(raw, patch_a)["no_evidence_content_logits"]
    initial_content_b = head(raw, patch_b)["no_evidence_content_logits"]
    assert not torch.allclose(initial_content_a, initial_content_b)

    # Force a high content-dependent no-evidence score for input A. There is no
    # free bias: the score is still produced from patch summary features.
    parameter_count = sum(
        parameter.numel()
        for parameter in head.no_evidence_content_scorer.parameters()
    )
    assert parameter_count == 32
    with torch.no_grad():
        normalized = head.patch_norm(patch_a)
        scores = head.patch_scores(normalized).reshape(1, 3, 5, 3, 1).permute(
            0, 1, 3, 4, 2
        )
        score_mean = scores.mean(dim=-1)
        statistics = torch.stack(
            [
                scores.amax(dim=-1) - score_mean,
                scores.topk(k=4, dim=-1).values.mean(dim=-1) - score_mean,
                scores.std(dim=-1, unbiased=False),
            ],
            dim=-1,
        )
        hidden = head.no_evidence_content_scorer[:-1](statistics).mean(
            dim=(0, 1, 2, 3)
        )
        direction = hidden / hidden.square().sum().clamp_min(1e-6)
        head.no_evidence_content_scorer[-1].weight.copy_(
            direction.reshape(1, -1) * 12.0
        )
    output_a = head(raw, patch_a)
    output_b = head(raw, patch_b)

    assert output_a["no_evidence_mass"].min() > 0.99
    # The key invariant: even near mass=1, appearance is normalized over and
    # computed from real patches, rather than replaced by a null constant.
    assert not torch.allclose(
        output_a["patch_appearance"], output_b["patch_appearance"]
    )
    assert head.no_evidence_logits.requires_grad is False
    assert head.no_evidence_values.requires_grad is False


def test_balanced_loss_equalizes_event_and_background_gradient_mass() -> None:
    raw_logits = torch.zeros(1, 12, 1, requires_grad=True)
    mass = raw_logits.sigmoid()
    targets = torch.zeros(1, 12, 1)
    targets[:, 0] = 1.0
    targets[:, 1] = 0.3  # Gaussian shoulder: deliberately ignored.
    batch = {
        "frame_targets": targets,
        "frame_target_masks": torch.ones_like(targets),
        "label_masks": torch.ones(1, 1),
    }
    loss, diagnostics = balanced_no_evidence_loss(
        {"class_evidence_no_evidence_mass": mass},
        batch,
        torch.device("cpu"),
        {"event_min_target": 0.5, "background_max_target": 0.1},
        labels=["shot"],
    )
    loss.backward()
    assert raw_logits.grad is not None
    event_gradient = raw_logits.grad[:, 0].abs().sum()
    background_gradient = raw_logits.grad[:, 2:].abs().sum()
    assert torch.allclose(event_gradient, background_gradient, atol=1e-6)
    assert raw_logits.grad[:, 1].abs().sum() == 0
    assert diagnostics[
        "class_evidence_no_evidence_event_mass_valid_slots"
    ] == 1.0
    assert diagnostics[
        "class_evidence_no_evidence_background_mass_valid_slots"
    ] == 10.0


def test_slot_weighted_mass_is_not_diluted_by_empty_batches() -> None:
    values = {
        "class_evidence_no_evidence_event_mass_numerator": 1.6,
        "class_evidence_no_evidence_event_mass_valid_slots": 2.0,
        # An arbitrary ordinary per-batch mean illustrates why it must not be
        # used as the aggregate truth when other batches contain no event slots.
        "class_evidence_no_evidence_event_mass": 0.08,
    }
    football.add_slot_weighted_no_evidence_metrics(values)
    assert values[
        "class_evidence_no_evidence_event_mass_slot_weighted"
    ] == 0.8


def test_require_both_sides_excludes_background_only_classes() -> None:
    raw_logits = torch.zeros(1, 6, 3, requires_grad=True)
    targets = torch.zeros_like(raw_logits)
    targets[:, 0, 0] = 1.0
    batch = {
        "frame_targets": targets,
        "frame_target_masks": torch.ones_like(targets),
        "label_masks": torch.ones(1, 3),
    }
    loss, diagnostics = balanced_no_evidence_loss(
        {"class_evidence_no_evidence_mass": raw_logits.sigmoid()},
        batch,
        torch.device("cpu"),
        {
            "event_min_target": 0.5,
            "background_max_target": 0.1,
            "require_both_sides_per_class": True,
        },
        labels=["shot", "save", "set_piece"],
    )
    loss.backward()
    assert raw_logits.grad is not None
    assert raw_logits.grad[:, :, 1:].abs().sum() == 0
    event_gradient = raw_logits.grad[:, 0, 0].abs()
    background_gradient = raw_logits.grad[:, 1:, 0].abs().sum()
    assert torch.allclose(event_gradient, background_gradient, atol=1e-6)
    assert diagnostics["class_evidence_no_evidence_active_classes"] == 1.0
