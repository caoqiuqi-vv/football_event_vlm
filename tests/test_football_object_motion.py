from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

from football_object_motion.losses import (
    _pairwise_residual_ranking,
    _distribution_quality_mask,
    _motion_quality_pair_mask,
    balanced_relation_bce,
    bidirectional_guard_loss,
    validate_same_video_pairs,
    object_motion_auxiliary_loss,
)
from football_object_motion.model import ObjectMotionEvidenceAdapter
from football_object_motion.monitor import REQUIRED, build_status
from football_object_motion.teacher import ball_teacher_quality
from football_object_motion.train import SameVideoPairBatchSampler, dense_protocol_prediction, full_window_coverage, full_window_segment_bounds


def _cfg():
    model = {
        "object_motion": {
            "positive_threshold": 0.05,
            "negative_weights": [0.5, 0.25, 0.05],
            "presence_negative_weights": [1.0, 1.0, 0.25],
            "object_loss_weights": [2.0, 1.0, 0.25],
            "learned_gate_budget": 0.25,
            "residual_saturation_threshold": 0.85,
        }
    }
    train = {
        "object_motion_heatmap_loss_weight": 0.5,
        "object_motion_distribution_loss_weight": 0.5,
        "object_motion_presence_loss_weight": 0.25,
        "object_motion_coordinate_loss_weight": 0.25,
        "object_motion_consistency_loss_weight": 0.2,
        "object_motion_frame_loss_weight": 0.5,
        "object_motion_dense_rank_loss_weight": 0.15,
        "object_motion_no_evidence_loss_weight": 0.05,
        "object_motion_residual_energy_loss_weight": 0.5,
        "object_motion_gate_budget_loss_weight": 0.2,
        "object_motion_saturation_loss_weight": 0.2,
    }
    return SimpleNamespace(model=model, train=train)


def test_zero_initialized_adapter_preserves_anchor_and_backpropagates():
    torch.manual_seed(3)
    batch_size, frames, grid_h, grid_w, patch_dim, labels = 2, 9, 4, 7, 32, 3
    module = ObjectMotionEvidenceAdapter(
        patch_dim=patch_dim,
        hidden_dim=64,
        num_labels=labels,
        num_heads=4,
        temporal_layers=1,
        dropout=0.0,
    )
    patches = torch.randn(batch_size, frames, grid_h * grid_w, patch_dim)
    times = torch.linspace(0.0, 4.0, frames).repeat(batch_size, 1)
    raw = module(patches, times, grid_h=grid_h, grid_w=grid_w)
    assert torch.equal(raw["clip_residual"], torch.zeros_like(raw["clip_residual"]))
    assert torch.equal(raw["frame_residual"], torch.zeros_like(raw["frame_residual"]))

    outputs = {
        "object_motion_heatmap_logits": raw["heatmap_logits"],
        "object_motion_presence_logits": raw["presence_logits"],
        "object_motion_centers": raw["centers"],
        "object_motion_velocity": raw["velocity"],
        "object_motion_frame_logits": raw["frame_residual"],
        "object_motion_frame_residual": raw["frame_residual"],
        "object_motion_clip_residual": raw["clip_residual"],
        "object_motion_clip_gate": raw["clip_gate"],
        "object_motion_frame_gate": raw["frame_gate"],
    }
    heatmaps = torch.zeros(batch_size, frames, grid_h * grid_w, 3)
    heatmaps[:, :, 3, 0] = 1.0
    heatmaps[:, :, 10:14, 1] = 1.0
    heatmaps[:, :, 15:20, 2] = 1.0
    presence = torch.ones(batch_size, frames, 3)
    coordinates = torch.zeros(batch_size, frames, 3, 4)
    coordinate_masks = torch.ones(batch_size, frames, 3)
    coordinate_masks[..., 2] = 0.0
    batch = {
        "object_motion_heatmap_targets": heatmaps,
        "object_motion_heatmap_masks": torch.ones_like(heatmaps),
        "object_motion_presence_targets": presence,
        "object_motion_presence_masks": torch.ones_like(presence),
        "object_motion_coordinate_targets": coordinates,
        "object_motion_coordinate_masks": coordinate_masks,
        "object_motion_times": times,
        "object_motion_frame_targets": torch.zeros(batch_size, frames, labels),
        "object_motion_frame_target_masks": torch.ones(batch_size, frames, labels),
    }
    clip_targets = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    loss, components = object_motion_auxiliary_loss(
        outputs,
        batch,
        _cfg(),
        torch.device("cpu"),
        clip_targets=clip_targets,
        clip_label_masks=torch.ones_like(clip_targets),
    )
    assert torch.isfinite(loss)
    assert components["object_motion_teacher_valid_fraction"] == 1.0
    assert 0.0 <= components["object_motion_distribution_loss"] <= 2.0
    assert 0.0 <= components["object_motion_ball_presence_precision"] <= 1.0
    assert 0.0 <= components["object_motion_goal_presence_recall"] <= 1.0

    outputs["object_motion_learned_clip_gate"] = torch.ones_like(
        outputs["object_motion_clip_gate"]
    )
    outputs["object_motion_learned_frame_gate"] = torch.ones_like(
        outputs["object_motion_frame_gate"]
    )
    outputs["object_motion_raw_clip_residual_logits"] = torch.full_like(
        outputs["object_motion_clip_residual"], 5.0
    )
    outputs["object_motion_raw_frame_residual"] = torch.full_like(
        outputs["object_motion_frame_residual"], 5.0
    )
    outputs["object_motion_adapter_clip_residual"] = torch.full_like(
        outputs["object_motion_clip_residual"], 0.5
    )
    outputs["object_motion_adapter_frame_residual"] = torch.full_like(
        outputs["object_motion_frame_residual"], 0.5
    )
    guarded_loss, guarded_components = object_motion_auxiliary_loss(
        outputs,
        batch,
        _cfg(),
        torch.device("cpu"),
        clip_targets=clip_targets,
        clip_label_masks=torch.ones_like(clip_targets),
    )
    assert guarded_loss > loss
    assert guarded_components["object_motion_residual_energy_loss"] > 0
    assert guarded_components["object_motion_gate_budget_loss"] > 0
    assert guarded_components["object_motion_saturation_loss"] > 0
    loss.backward()
    assert module.heatmap_head[-1].weight.grad is not None


def test_no_object_presence_has_explicit_fallback_token():
    module = ObjectMotionEvidenceAdapter(
        patch_dim=16,
        hidden_dim=32,
        num_labels=3,
        num_heads=4,
        temporal_layers=1,
        dropout=0.0,
    )
    with torch.no_grad():
        for presence_head in module.presence_heads:
            presence_head[-1].weight.zero_()
            presence_head[-1].bias.fill_(-12.0)
    patches = torch.randn(1, 5, 12, 16)
    times = torch.linspace(0.0, 2.0, 5).unsqueeze(0)
    outputs = module(patches, times, grid_h=3, grid_w=4)
    assert outputs["presence_logits"].sigmoid().max() < 1e-4
    assert torch.isfinite(outputs["clip_residual"]).all()


def test_event_gradient_is_decoupled_from_detector_heads():
    torch.manual_seed(5)
    module = ObjectMotionEvidenceAdapter(
        patch_dim=16,
        hidden_dim=32,
        num_labels=3,
        num_heads=4,
        temporal_layers=1,
        dropout=0.0,
        detach_detector_for_event=True,
    )
    with torch.no_grad():
        module.frame_residual_head[-1].weight.fill_(0.01)
        module.clip_residual_head[-1].weight.fill_(0.01)
    patches = torch.randn(2, 5, 12, 16)
    times = torch.linspace(0.0, 2.0, 5).repeat(2, 1)
    outputs = module(patches, times, grid_h=3, grid_w=4)
    event_loss = outputs["frame_residual"].sum() + outputs["clip_residual"].sum()
    event_loss.backward()
    assert module.relation_projection[1].weight.grad is not None
    assert module.heatmap_head[-1].weight.grad is None
    assert all(
        parameter.grad is None
        for head in module.presence_heads
        for parameter in head.parameters()
    )


def test_ball_evidence_bounds_effective_gate_and_residual():
    module = ObjectMotionEvidenceAdapter(
        patch_dim=16,
        hidden_dim=32,
        num_labels=3,
        num_heads=4,
        temporal_layers=1,
        dropout=0.0,
        frame_residual_max_delta=0.15,
        relation_delta=0.15,
        clip_residual_max_delta=0.25,
        gate_max=0.4,
        evidence_gate_floor=0.02,
    )
    with torch.no_grad():
        module.clip_gate_head[-1].weight.zero_()
        module.clip_gate_head[-1].bias.fill_(12.0)
        module.frame_residual_head[-1].weight.fill_(1.0)
        module.clip_residual_head[-1].weight.fill_(1.0)
        module.presence_heads[0][-1].weight.zero_()
        module.presence_heads[0][-1].bias.fill_(-12.0)
    patches = torch.randn(1, 5, 12, 16)
    times = torch.linspace(0.0, 2.0, 5).unsqueeze(0)
    low_ball = module(patches, times, grid_h=3, grid_w=4)
    with torch.no_grad():
        module.presence_heads[0][-1].bias.fill_(12.0)
    high_ball = module(patches, times, grid_h=3, grid_w=4)
    assert torch.equal(
        low_ball["learned_frame_gate"], high_ball["learned_frame_gate"]
    )
    assert low_ball["frame_gate"].max() < high_ball["frame_gate"].max()
    assert high_ball["frame_gate"].max() <= 0.4 + 1e-6
    assert high_ball["clip_gate"].max() <= 0.4 + 1e-6
    assert high_ball["frame_residual"].abs().max() <= 0.15 * 0.4 + 1e-6
    assert high_ball["clip_residual"].abs().max() <= 0.25 * 0.4 + 1e-6


def test_monitor_rejects_gate_and_residual_saturation():
    values = {key: 0.1 for key in REQUIRED}
    values.update(
        {
            "epoch": 1.0,
            "step": 20.0,
            "grad_norm_head": 0.2,
            "object_motion_teacher_valid_fraction": 1.0,
            "object_motion_learned_frame_gate_mean": 0.96,
            "object_motion_residual_saturation_fraction": 0.30,
        }
    )
    row = " ".join(f"{key}={value}" for key, value in values.items())
    with TemporaryDirectory() as directory:
        output_dir = Path(directory)
        (output_dir / "train_console.log").write_text(row + "\n")
        status = build_status(output_dir)
    assert status["healthy"] is False
    assert any("gate_mean" in failure for failure in status["failures"])
    assert any("saturation fraction" in failure for failure in status["failures"])


def test_pairwise_ranking_uses_positive_negative_pairs():
    residual = torch.tensor(
        [[0.4, -0.2], [-0.1, 0.3], [0.2, 0.1]], requires_grad=True
    )
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    masks = torch.ones_like(targets)
    good_loss, pair_count = _pairwise_residual_ranking(
        residual, targets, masks, margin=0.1, temperature=0.1
    )
    bad_residual = -residual.detach().requires_grad_(True)
    bad_loss, bad_pair_count = _pairwise_residual_ranking(
        bad_residual, targets, masks, margin=0.1, temperature=0.1
    )
    assert pair_count == bad_pair_count == 4
    assert good_loss < bad_loss
    good_loss.backward()
    assert residual.grad is not None

def test_v3_full_window_is_51_absolute_sorted_frames():
    bounds = full_window_segment_bounds(100.0, 110.0)
    assert bounds == ((100.0, 104.0), (103.0, 107.0), (106.0, 110.0))
    times = torch.cat([torch.linspace(start, end, 17) for start, end in bounds]).sort().values
    assert times.numel() == 51
    assert full_window_coverage(times, 100.0, 110.0) >= 0.95


def test_v3_alpha_zero_identity_relation_gradient_and_detector_detach():
    torch.manual_seed(9)
    module = ObjectMotionEvidenceAdapter(patch_dim=16, hidden_dim=32, num_labels=3, num_heads=4, temporal_layers=1, dropout=0.0, gate_floor=0.25)
    patches = torch.randn(2, 5, 12, 16)
    times = torch.linspace(0.0, 2.0, 5).repeat(2, 1)
    motion = module(patches, times, grid_h=3, grid_w=4)
    anchor = torch.randn(2, 3)
    final = anchor + 0.0 * motion["clip_residual"]
    assert torch.equal(final, anchor)
    targets = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    loss = balanced_relation_bce(motion["raw_clip_residual"], targets, torch.ones_like(targets))
    loss.backward()
    head_grad = module.clip_residual_head[-1].weight.grad.norm().item()
    assert head_grad > 1e-5
    assert module.heatmap_head[-1].weight.grad is None
    assert all(parameter.grad is None for head in module.presence_heads for parameter in head.parameters())


def test_v3_pair_validation_failfast_and_legal_pair():
    targets = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    legal = [
        {"source": "s", "video_id": "v", "pair_id": "p0", "pair_role": "positive", "pair_label": "shot", "relation_class_index": 0, "sampled_clip_center": 10.0, "nearest_same_class_gt_gap": 0.0},
        {"source": "s", "video_id": "v", "pair_id": "p0", "pair_role": "negative", "pair_label": "shot", "relation_class_index": 0, "sampled_clip_center": 20.0, "reviewed_negative": True, "full_clean_window": True, "nearest_same_class_gt_gap": 8.0, "review_manifest": "reviewed.json"},
    ]
    assert validate_same_video_pairs(legal, targets) == 1
    illegal = [legal[0], dict(legal[1], video_id="other")]
    try:
        validate_same_video_pairs(illegal, targets)
    except ValueError as error:
        assert "illegal relation pair" in str(error)
    else:
        raise AssertionError("illegal pair did not fail fast")

    unusable_masks = torch.tensor([[1.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
    assert validate_same_video_pairs(
        legal,
        targets,
        unusable_masks,
        skip_unusable_supervision=True,
    ) == 0
    try:
        validate_same_video_pairs(legal, targets, unusable_masks)
    except ValueError as error:
        assert "illegal relation supervision" in str(error)
    else:
        raise AssertionError("strict validation accepted unusable supervision")


def test_v3_teacher_tri_state_and_trajectory_masks():
    confidence = torch.tensor([[0.0, 0.15, 0.15, 0.15, 0.35, 0.15]])
    centers = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [1.0, 1.0]]])
    quality = ball_teacher_quality(confidence, centers, jump_limit=0.35)
    assert torch.isclose(quality["presence"][0, 0], torch.tensor(0.05))
    assert torch.equal(quality["coordinate"][0, 1:5], torch.ones(4))
    assert quality["coordinate"][0, 5].item() == 0.0
    assert quality["motion"][0, 5].item() == 0.0


def test_v3_bidirectional_guard_only_uses_reviewed_full_clean_negatives():
    anchor = torch.zeros(3, 1)
    final = torch.tensor([[-0.2], [0.3], [0.8]], requires_grad=True)
    targets = torch.tensor([[1.0], [0.0], [0.0]])
    metas = [{}, {"reviewed_negative": True, "full_clean_window": True}, {"reviewed_negative": False}]
    loss, upward, downward = bidirectional_guard_loss(final, anchor, targets, torch.ones_like(targets), metas)
    assert upward == 1 and downward == 1
    assert torch.isclose(loss, torch.tensor(0.35))

def test_v3_dense_protocol_semantics_are_not_mixed():
    clip = torch.tensor([0.8])
    frames = torch.tensor([[0.1, 0.9, 0.2]])
    times = torch.tensor([[10.0, 11.0, 12.0]])
    center = torch.tensor([15.0])
    score, when = dense_protocol_prediction("clip_center", clip, frames, times, center)
    assert torch.equal(score, clip) and torch.equal(when, center)
    score, when = dense_protocol_prediction("clip_x_frame_peak", clip, frames, times, center)
    assert torch.isclose(score, torch.tensor([0.72])).all()
    assert torch.equal(when, torch.tensor([11.0]))


def test_v3_pair_sampler_is_deterministic_and_rank_sharded():
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    left = SameVideoPairBatchSampler(pairs, rank=0, world_size=2, seed=7)
    right = SameVideoPairBatchSampler(pairs, rank=1, world_size=2, seed=7)
    assert list(left) == list(SameVideoPairBatchSampler(pairs, rank=0, world_size=2, seed=7))
    assert sorted(list(left) + list(right)) == sorted([list(pair) for pair in pairs])

def test_v3_pair_sampler_drops_nondivisible_complete_pair_equally():
    pairs = [(index * 2, index * 2 + 1) for index in range(5)]
    ranks = [SameVideoPairBatchSampler(pairs, rank=rank, world_size=2, seed=11) for rank in range(2)]
    assert [len(sampler) for sampler in ranks] == [2, 2]
    batches = [batch for sampler in ranks for batch in sampler]
    assert len(batches) == 4 and all(len(batch) == 2 for batch in batches)
    first_epoch = list(ranks[0])
    ranks[0].set_epoch(1)
    assert list(ranks[0]) != first_epoch


def test_v3_relation_frame_clip_and_fusion_caps_are_independent():
    module = ObjectMotionEvidenceAdapter(patch_dim=16, hidden_dim=32, num_labels=3, num_heads=4, temporal_layers=1, dropout=0.0, relation_delta=1.0, frame_residual_max_delta=0.15, clip_residual_max_delta=0.25, fusion_delta=0.5, gate_floor=1.0, gate_max=1.0)
    with torch.no_grad():
        module.frame_residual_head[-1].weight.fill_(5.0)
        module.frame_residual_head[-1].bias.fill_(5.0)
        module.clip_residual_head[-1].weight.fill_(5.0)
        module.clip_residual_head[-1].bias.fill_(5.0)
    outputs = module(torch.randn(2, 5, 12, 16), torch.linspace(0.0, 2.0, 5).repeat(2, 1), grid_h=3, grid_w=4)
    assert outputs["frame_residual"].abs().max() <= 0.15 + 1e-6
    assert outputs["clip_residual"].abs().max() <= 0.25 + 1e-6
    assert (outputs["frame_residual"] * module.fusion_delta).abs().max() <= 0.075 + 1e-6
    assert (outputs["clip_residual"] * module.fusion_delta).abs().max() <= 0.125 + 1e-6

def test_v3_distribution_and_motion_keep_continuous_teacher_quality():
    target_mass = torch.tensor([[[1.0], [1.0], [0.0]]])
    masks = torch.tensor([[[[0.25]], [[1.0]], [[0.05]]]])
    distribution = _distribution_quality_mask(target_mass, masks)
    assert torch.equal(distribution, torch.tensor([[[0.25], [1.0], [0.0]]]))
    coordinates = torch.ones(1, 4, 1)
    motion_quality = torch.tensor([[[0.0], [0.25], [1.0], [0.0]]])
    motion = _motion_quality_pair_mask(coordinates, motion_quality)
    assert torch.equal(motion, torch.tensor([[[0.0], [0.25], [1.0], [0.0]]]))
