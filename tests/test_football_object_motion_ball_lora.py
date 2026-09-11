from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from football_object_motion.ball_backbone import (
    BallLoRALinear,
    ball_lora_enabled,
    inject_ball_lora,
)
from football_object_motion.losses import (
    AdjacentPairResidualQueue,
    _ball_lora_losses,
    ball_feature_preserve_mask,
)
from football_object_motion.model import ObjectMotionEvidenceAdapter
from football_object_motion.train import SameVideoPairQueueBatchSampler, full_window_segment_bounds


def _adapter_and_overlay():
    base = nn.Linear(8, 8)
    overlay = BallLoRALinear(base, rank=4, alpha=4.0)
    adapter = ObjectMotionEvidenceAdapter(
        patch_dim=8,
        hidden_dim=16,
        num_labels=3,
        num_heads=4,
        temporal_layers=1,
        dropout=0.0,
        ball_topk=4,
    )
    return adapter, overlay


def _ball_outputs(adapter, overlay):
    raw = torch.randn(1, 3, 4, 16, 8)
    with ball_lora_enabled(overlay, True):
        layers = overlay(raw)
    return adapter(
        layers[:, :, -1],
        torch.tensor([[0.0, 0.5, 1.0]]),
        grid_h=4,
        grid_w=4,
        ball_patch_layers=layers,
    )


def test_ball_lora_disabled_is_bitwise_and_context_restores_after_error():
    _, overlay = _adapter_and_overlay()
    inputs = torch.randn(2, 5, 8)
    expected = overlay.base(inputs)
    assert torch.equal(overlay(inputs), expected)
    try:
        with ball_lora_enabled(overlay, True):
            assert overlay.enabled
            raise RuntimeError("sentinel")
    except RuntimeError:
        pass
    assert not overlay.enabled
    assert torch.equal(overlay(inputs), expected)


def test_ball_topk4_center_entropy_and_convex_multilayer_weights():
    adapter, overlay = _adapter_and_overlay()
    outputs = _ball_outputs(adapter, overlay)
    sparse = outputs["sparse_attention"][..., 0]
    assert torch.equal((sparse > 0).sum(dim=2), torch.full((1, 3), 4))
    assert torch.allclose(sparse.sum(dim=2), torch.ones(1, 3))
    weights = outputs["ball_layer_weights"]
    assert torch.isclose(weights.sum(), torch.tensor(1.0))
    assert weights.max() <= 0.7 + 1e-6
    entropy = outputs["ball_student_entropy"]
    assert torch.isfinite(entropy).all() and entropy.min() >= 0 and entropy.max() <= 1
    assert outputs["ball_student_center"].abs().max() <= 1.0 + 1e-6


def test_only_strong_ball_loss_updates_ball_lora_and_event_path_does_not():
    adapter, overlay = _adapter_and_overlay()
    outputs = _ball_outputs(adapter, overlay)
    targets = torch.zeros(1, 3, 16, 3)
    targets[:, :, 2, 0] = 1.0
    coordinates = torch.zeros(1, 3, 3, 4)
    cfg = {"image_size": [4, 4], "patch_size": 1}

    strong_masks = torch.zeros_like(targets)
    strong_masks[..., 0] = 1.0
    losses, _ = _ball_lora_losses(
        {
            "object_motion_ball_lora_logits": outputs["ball_lora_logits"],
            "object_motion_ball_student_features": outputs["ball_student_features"],
            "object_motion_ball_student_center": outputs["ball_student_center"],
        },
        targets,
        strong_masks,
        coordinates,
        torch.ones(1, 3, 3),
        cfg,
    )
    sum(losses.values()).backward()
    assert overlay.ball_lora_b.grad is not None
    assert overlay.ball_lora_b.grad.norm() > 1e-6

    for parameter in overlay.parameters():
        parameter.grad = None
    outputs = _ball_outputs(adapter, overlay)
    weak_masks = torch.zeros_like(targets)
    weak_masks[..., 0] = 0.25
    weak_loss = F.binary_cross_entropy_with_logits(
        outputs["heatmap_logits"][..., 0], targets[..., 0]
    )
    weak_loss.backward()
    assert overlay.ball_lora_a.grad is None
    assert overlay.ball_lora_b.grad is None

    for parameter in overlay.parameters():
        parameter.grad = None
    with torch.no_grad():
        adapter.clip_residual_head[-1].weight.fill_(0.1)
    outputs = _ball_outputs(adapter, overlay)
    event_and_other = outputs["raw_clip_residual"].sum() + outputs["heatmap_logits"][..., 1:].sum()
    event_and_other.backward()
    assert overlay.ball_lora_a.grad is None
    assert overlay.ball_lora_b.grad is None


def test_nonball_preserve_mask_dilates_two_patches():
    targets = torch.zeros(1, 1, 49)
    targets[..., 24] = 1.0
    mask = ball_feature_preserve_mask(targets, grid_h=7, grid_w=7, dilation_patches=2)
    assert mask.reshape(7, 7)[1:6, 1:6].sum() == 0
    assert mask.sum() == 24


def test_pair_queue_sampler_flips_without_crossing_and_queue_rejects_cross_pair():
    pairs = [(0, 1), (2, 3)]
    sampler = SameVideoPairQueueBatchSampler(pairs, seed=5)
    epoch0 = list(sampler)
    assert all(epoch0[index][0] // 2 == epoch0[index + 1][0] // 2 for index in range(0, len(epoch0), 2))
    sampler.set_epoch(1)
    epoch1 = list(sampler)
    assert all(epoch1[index][0] % 2 == 1 and epoch1[index + 1][0] % 2 == 0 for index in range(0, len(epoch1), 2))

    queue = AdjacentPairResidualQueue()
    tensor = torch.zeros(1, 3, requires_grad=True)
    target = torch.zeros(1, 3)
    mask = torch.ones(1, 3)
    queue.consume(tensor, target, mask, {"pair_id": "a", "pair_role": "negative"}, margin=0.1, temperature=0.1, min_gap_sec=5.0)
    try:
        queue.consume(tensor, target, mask, {"pair_id": "b", "pair_role": "positive"}, margin=0.1, temperature=0.1, min_gap_sec=5.0)
    except ValueError as error:
        assert "cross pair_id" in str(error)
    else:
        raise AssertionError("queue crossed pair boundary")


def test_ball_lora_checkpoint_contains_only_overlay_delta_for_backbone():
    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(8, 24)
            self.proj = nn.Linear(8, 8)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = Attention()

    backbone = nn.Module()
    backbone.blocks = nn.ModuleList([Block() for _ in range(10)])
    assert inject_ball_lora(backbone, last_blocks=8) == 16
    trainable = [name for name, parameter in backbone.named_parameters() if parameter.requires_grad]
    assert trainable and all(name.endswith(("ball_lora_a", "ball_lora_b")) for name in trainable)
    delta = {name: value for name, value in backbone.state_dict().items() if "ball_lora_" in name}
    assert len(delta) == 32



def test_balllora_full_window_has_33_sorted_frames_and_endpoints():
    bounds = full_window_segment_bounds(100.0, 110.0)
    times = torch.cat([torch.linspace(start, end, 11) for start, end in bounds]).sort().values
    assert times.numel() == 33
    assert times[0] == 100.0 and times[-1] == 110.0
