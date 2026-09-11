"""Real tensor/gradient regressions, without loading DINO weights or YOLO."""
import torch
import pytest
from torch import nn

from football_object_motion.ball_backbone import BallLoRALinear
from football_object_motion.model import ObjectMotionEvidenceAdapter, ObjectContextReadout
from football_object_motion.losses import trusted_no_evidence_mask


def make_adapter(**options):
    return ObjectMotionEvidenceAdapter(
        patch_dim=8, hidden_dim=16, num_labels=3, num_heads=4,
        temporal_layers=1, dropout=0.0, object_cross_attention_enabled=True,
        heatmap_upsample_factor=2, ball_fpn_dim=8,
        **options,
    )


def inputs():
    layers = torch.randn(1, 5, 4, 9, 8, requires_grad=True)
    return layers, torch.tensor([[0.0, 0.4, 0.8, 1.2, 1.6]])


def open_heads(adapter):
    # Zero-init intentionally blocks upstream gradients on step zero; inspect
    # the nonzero-head state that is reached after an optimizer update.
    with torch.no_grad():
        nn.init.normal_(adapter.object_cross_attention.delta_head[-1].weight, std=0.05)
        if adapter.object_cross_attention.frame_enabled:
            nn.init.normal_(adapter.object_cross_attention.frame_delta_head[-1].weight, std=0.05)


def nonzero_grad(module):
    return sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None) > 0


@pytest.mark.parametrize('enabled', [False, True])
def test_relation_gradient_switch_keeps_detector_boundary(enabled):
    torch.manual_seed(12)
    adapter = make_adapter(event_relation_grad_enabled=enabled)
    open_heads(adapter)
    layers, times = inputs()
    motion = adapter(layers[:, :, -1], times, grid_h=3, grid_w=3, ball_patch_layers=layers)
    fused = adapter.fuse_global_event(torch.randn(1, 16), motion)
    fused['correction'].square().sum().backward()
    assert nonzero_grad(adapter.temporal) == enabled
    assert nonzero_grad(adapter.relation_projection) == enabled
    assert layers.grad is None
    assert not nonzero_grad(adapter.ball_layer_heads)
    assert not nonzero_grad(adapter.presence_heads)


def test_legacy_checkpoint_loads_without_new_keys_in_grad_mode():
    old = make_adapter()
    new = make_adapter(event_relation_grad_enabled=True)
    new.load_state_dict(old.state_dict(), strict=True)
    old.eval()
    new.eval()
    layers, times = inputs()
    global_token = torch.randn(1, 16)
    for adapter in (old, new):
        open_heads(adapter)
    # Same values; only the autograd boundary differs.
    new.load_state_dict(old.state_dict(), strict=True)
    def run(adapter):
        motion = adapter(layers[:, :, -1], times, grid_h=3, grid_w=3, ball_patch_layers=layers)
        return adapter.fuse_global_event(global_token, motion)['correction']
    torch.testing.assert_close(run(old), run(new), rtol=0, atol=0)


def test_context_zero_initialization_and_checkpoint_compatibility():
    old = make_adapter()
    open_heads(old)
    adapter = make_adapter(event_relation_grad_enabled=True, event_context_enabled=True, event_frame_fusion_enabled=True)
    missing, unexpected = adapter.load_experiment_state_dict(old.state_dict())
    assert not unexpected
    assert all(key.startswith(('event_context.', 'object_cross_attention.')) for key in missing)
    layers, times = inputs()
    motion = adapter(layers[:, :, -1], times, grid_h=3, grid_w=3, ball_patch_layers=layers)
    fused = adapter.fuse_global_event(torch.randn(1, 16), motion)
    assert motion['event_context_tokens'].shape == (1, 5, 4, 8)
    assert torch.count_nonzero(fused['correction']) == 0
    assert fused['frame_correction'].shape == (1, 5, 3)
    assert torch.count_nonzero(fused['frame_correction']) == 0
    resumed = make_adapter(event_relation_grad_enabled=True, event_context_enabled=True, event_frame_fusion_enabled=True)
    missing, unexpected = resumed.load_experiment_state_dict(adapter.state_dict())
    assert not missing and not unexpected


@pytest.mark.parametrize('feature_grad', [False, True])
def test_frame_event_loss_trains_context_and_optional_dense_lora(feature_grad):
    torch.manual_seed(22)
    overlay = BallLoRALinear(nn.Linear(8, 8), rank=2, alpha=2)
    overlay.enabled = True
    adapter = make_adapter(
        event_relation_grad_enabled=True, event_context_enabled=True,
        event_context_feature_grad=feature_grad, event_frame_fusion_enabled=True,
    )
    open_heads(adapter)
    raw, times = inputs()
    layers = overlay(raw)
    motion = adapter(layers[:, :, -1], times, grid_h=3, grid_w=3, ball_patch_layers=layers)
    fused = adapter.fuse_global_event(torch.randn(1, 16), motion)
    target = torch.zeros_like(fused['frame_correction'])
    target[:, 2, 0] = 1
    nn.functional.binary_cross_entropy_with_logits(fused['frame_correction'], target).backward()
    assert nonzero_grad(adapter.event_context)
    assert nonzero_grad(adapter.temporal)
    assert nonzero_grad(overlay) == feature_grad
    assert not nonzero_grad(adapter.ball_layer_heads)
    assert not nonzero_grad(adapter.presence_heads)
    assert float(fused['frame_correction'].abs().max()) <= adapter.frame_residual_max_delta


def test_context_routes_stop_gradient_but_values_can_train():
    readout = ObjectContextReadout(8, feature_grad=True)
    features = torch.randn(1, 2, 9, 8, requires_grad=True)
    weights = torch.rand(1, 2, 9, 3, requires_grad=True)
    readout(features, weights, 3, 3).square().sum().backward()
    assert weights.grad is None
    assert features.grad is not None and features.grad.abs().sum() > 0


def test_uniform_control_is_independent_of_spatial_routing():
    readout = ObjectContextReadout(8, uniform=True)
    features = torch.randn(1, 2, 9, 8)
    a = readout(features, torch.rand(1, 2, 9, 3), 3, 3)
    b = readout(features, torch.rand(1, 2, 9, 3), 3, 3)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_time_and_type_encoding_changes_direct_object_attention():
    torch.manual_seed(3)
    adapter = make_adapter(event_context_enabled=True)
    fusion = adapter.object_cross_attention.eval()
    tokens = torch.randn(1, 5, 4, 8)
    relation = torch.zeros(1, 5, 16)
    visibility = torch.ones(1, 5, 4)
    times = torch.arange(5).reshape(1, 5).float()
    global_token = torch.randn(1, 16)
    a = fusion(global_token, tokens, relation, visibility, times)['attended_tokens']
    b = fusion(global_token, tokens.flip(1), relation, visibility, times)['attended_tokens']
    assert not torch.allclose(a, b, atol=1e-7, rtol=1e-7)


def test_invalid_feature_configuration_fails_closed():
    with pytest.raises(ValueError, match='require object_cross_attention'):
        ObjectMotionEvidenceAdapter(patch_dim=8, hidden_dim=16, num_labels=3, num_heads=4, event_frame_fusion_enabled=True)


def test_unknown_teacher_frames_are_not_no_evidence_negatives():
    targets = torch.zeros(1, 4, 3)
    masks = torch.zeros_like(targets)
    masks[:, 1, :2] = 1
    masks[:, 2, :2] = 1
    targets[:, 2, 0] = 1
    masks[:, 3, :2] = 0.05  # Weak teacher absence is not confirmed absence.
    assert trusted_no_evidence_mask(targets, masks).tolist() == [[False, True, False, False]]
