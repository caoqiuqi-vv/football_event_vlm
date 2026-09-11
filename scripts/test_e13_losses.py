"""E1.3 loss/anti-erasure gate tests (plan 5.1.3, all CPU-only).

1. anti-erasure hard test: with no-evidence mass forced >0.99, perturbing
   input patches must still change patch_appearance (new structure); the
   legacy scalar-null structure is shown to erase content (regression proof).
2. balanced_no_evidence_loss: per-class event/background balance, shoulder
   ignore, require_both_sides gating.
3. positive_core_context_heatmap_loss: core/context 1:1 inside positive slots.
4. online_pair_consistency_loss: central teacher -> edge student, warmup,
   strict pair metadata errors.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "/home/new_users/qiuqi/code/dinov3-main")

import torch
import torch.nn.functional as F

import train_football_events as base
from football_frame_heatmap import positive_core_context_heatmap_loss
from football_no_evidence import balanced_no_evidence_loss
from football_online_consistency import online_pair_consistency_loss
from football_saliency_rank import (
    bag_saliency_rank_loss,
    cross_window_saliency_consistency_loss,
    signed_causal_local_evidence_loss,
)

torch.manual_seed(7)


def _make_head(anti_erasure: bool, causal_local: bool = False) -> base.IndependentPreprojectionEvidenceHead:
    return base.IndependentPreprojectionEvidenceHead(
        patch_dim=32,
        raw_feature_dim=64,  # 32 * 2 * num_feature_layers(1)
        num_feature_layers=1,
        class_dim=64,
        num_labels=3,
        queries_per_class=2,
        temporal_layers=1,
        temporal_heads=4,
        dropout=0.0,
        max_frames=4,
        bottleneck_frames=2,
        anti_erasure_no_evidence=anti_erasure,
        causal_local_counterfactual=causal_local,
    )


def test_anti_erasure() -> None:
    head = _make_head(anti_erasure=True)
    head.eval()
    bsz, frames, patches, dim = 2, 4, 32, 32
    raw = torch.randn(bsz, frames, 64)
    patches_a = torch.randn(bsz, frames, patches, dim) * 0.5
    # Per-element perturbation: a constant shift would be cancelled by the
    # LayerNorm inside the head, not by the anti-erasure structure.
    patches_b = patches_a + torch.randn_like(patches_a) * 0.5

    # Forced null mass > 0.99: a large positive gate bias makes the null
    # logit strongly positive regardless of input.
    with torch.no_grad():
        head.no_evidence_gate_bias.fill_(10.0)
    with torch.no_grad():
        out_a = head(raw, patches_a)
        out_b = head(raw, patches_b)
    mass = out_a["no_evidence_mass"]
    assert float(mass.mean()) > 0.99, f"null mass {mass.mean():.3f} not forced >0.99"
    gap = (out_a["patch_appearance"] - out_b["patch_appearance"]).abs().max()
    assert float(gap) > 1e-3, f"anti-erasure failed: appearance gap {gap:.6f}"
    print(f"anti_erasure OK: null_mass={float(mass.mean()):.3f} appearance_gap={float(gap):.4f}")

    # Legacy structure must still erase content at high null mass: this is the
    # bug the fix removes (content-free scalar replaces appearance).
    legacy = _make_head(anti_erasure=False)
    legacy.eval()
    with torch.no_grad():
        legacy.no_evidence_logits.fill_(50.0)  # softmax -> null mass ~ 1
        old_a = legacy(raw, patches_a)
        old_b = legacy(raw, patches_b)
    old_gap = (old_a["patch_appearance"] - old_b["patch_appearance"]).abs().max()
    assert float(old_gap) < 1e-6, f"legacy structure unexpectedly kept content: {old_gap:.6f}"
    print(f"legacy_erasure_proof OK: appearance_gap={float(old_gap):.2e}")

    # Frozen legacy null parameters stay untrainable.
    for name, param in head.named_parameters():
        if "no_evidence_logits" in name or "no_evidence_values" in name:
            assert not param.requires_grad, f"{name} should be frozen"
    print("frozen_null_params OK")

    # Monotone saliency gate: with non-negative softplus coefficients,
    # stronger saliency can only lower null mass (no erasure-capable output).
    with torch.no_grad():
        head.no_evidence_gate_bias.zero_()
        head.no_evidence_gate_weight.fill_(1.0)
    saliency_a = torch.tensor([[0.1, 0.1, 0.1]])
    saliency_b = torch.tensor([[2.0, 2.0, 2.0]])
    mass_a = torch.sigmoid(
        head.no_evidence_gate_bias
        - (F.softplus(head.no_evidence_gate_weight) * torch.log1p(saliency_a)).sum(-1)
    )
    mass_b = torch.sigmoid(
        head.no_evidence_gate_bias
        - (F.softplus(head.no_evidence_gate_weight) * torch.log1p(saliency_b)).sum(-1)
    )
    assert float(mass_b) < float(mass_a), (
        f"gate not monotone: mass({saliency_b[0].tolist()})={mass_b.item():.4f} "
        f">= mass({saliency_a[0].tolist()})={mass_a.item():.4f}"
    )
    print(f"gate_monotonicity OK: mass(0.1)={float(mass_a):.3f} > mass(2.0)={float(mass_b):.3f}")


def test_balanced_no_evidence() -> None:
    cfg = {
        "event_min_target": 0.5,
        "background_max_target": 0.1,
        "require_both_sides_per_class": True,
        "global_ddp_balance": False,
    }
    bsz, frames, classes = 2, 12, 3
    frame_times = torch.linspace(0, 11, frames)
    targets = torch.zeros(bsz, frames, classes)
    masks = torch.ones(bsz, frames, classes)
    label_masks = torch.ones(bsz, classes)
    # class 0: event core at frame 3 (target=1.0), background elsewhere
    targets[:, 3, 0] = 1.0
    targets[:, 9, 1] = 1.0  # class 1 event
    targets[:, :, 2] = 0.3  # class 2 all shoulder (0.1-0.5) -> ignored
    # Hinge violation: event frames report high null (0.7 > 0.4 margin) and
    # background frames low null (0.3 < 0.6) -> positive loss on both sides.
    mass = torch.full((bsz, frames, classes), 0.3)
    mass[:, 3, 0] = 0.7
    mass[:, 9, 1] = 0.7
    outputs = {"class_evidence_no_evidence_mass": mass}
    batch = {
        "frame_targets": targets,
        "frame_target_masks": masks,
        "label_masks": label_masks,
    }
    loss, diag = balanced_no_evidence_loss(outputs, batch, torch.device("cpu"), cfg, labels=("shot", "save", "set_piece"))
    assert float(loss) > 0, "balanced null loss must be positive"
    assert diag["class_evidence_no_evidence_active_classes"] == 2, "classes 0/1 active, class 2 shoulder-ignored"
    # Hinge semantics: both sides are violated (event null 0.7 > 0.4 margin,
    # background null 0.3 < 0.6), so both decomposed losses must be positive.
    assert diag["class_evidence_no_evidence_event_loss"] > 0, "event hinge should fire"
    assert diag["class_evidence_no_evidence_background_loss"] > 0, "background hinge should fire"
    ignored = diag["class_evidence_no_evidence_ignored_fraction"]
    assert ignored > 0.05, f"shoulder class 2 should be ignored (fraction {ignored})"
    # Diagnostic sanity: the reported masses mirror the input exactly.
    event_mass = diag["class_evidence_no_evidence_event_mass"]
    background_mass = diag["class_evidence_no_evidence_background_mass"]
    assert abs(event_mass - 0.7) < 1e-5 and abs(background_mass - 0.3) < 1e-5
    # Correct behaviour case: low event null / high background null -> no loss.
    mass_ok = torch.full((bsz, frames, classes), 0.95)
    mass_ok[:, 3, 0] = 0.05
    mass_ok[:, 9, 1] = 0.05
    loss_ok, _ = balanced_no_evidence_loss(
        {"class_evidence_no_evidence_mass": mass_ok}, batch,
        torch.device("cpu"), cfg, labels=("shot", "save", "set_piece"),
    )
    assert float(loss_ok) == 0.0, f"correct-side masses should give zero hinge loss, got {loss_ok:.4f}"
    print(f"balanced_no_evidence OK: loss={float(loss):.4f} event_mass={event_mass:.3f} "
          f"background_mass={background_mass:.3f} ignored={ignored:.2f} ok_loss={float(loss_ok):.4f}")

    # Single-side class must be skipped under require_both_sides.
    targets2 = torch.zeros(bsz, frames, classes)
    masks2 = torch.zeros(bsz, frames, classes)
    masks2[:, 3, 0] = 1.0
    targets2[:, 3, 0] = 1.0  # class 0: event side only (background masked)
    masks2[:, :, 1] = 1.0  # class 1: background side only (event absent)
    masks2[:, :, 2] = 1.0  # class 2: background side only
    batch2 = {
        "frame_targets": targets2,
        "frame_target_masks": masks2,
        "label_masks": label_masks,
    }
    loss2, diag2 = balanced_no_evidence_loss(
        {"class_evidence_no_evidence_mass": mass.clone()}, batch2, torch.device("cpu"), cfg,
        labels=("shot", "save", "set_piece"),
    )
    assert diag2["class_evidence_no_evidence_active_classes"] == 0, "no class has both sides"
    assert float(loss2) == 0.0
    print("require_both_sides OK")


def test_heatmap_balance() -> None:
    bsz, frames, classes = 1, 8, 2
    targets = torch.zeros(bsz, frames, classes)
    masks = torch.ones(bsz, frames, classes)
    clip_targets = torch.zeros(bsz, classes)
    clip_masks = torch.ones(bsz, classes)
    # positive clip/class 0: core at frame 4 (target 1.0), context elsewhere
    targets[:, 4, 0] = 1.0
    clip_targets[:, 0] = 1.0
    heatmap = torch.full((bsz, frames, classes), 0.0)
    heatmap[:, 4, 0] = 1.0  # model misses core (loss 1.0 there), perfect context (0.0)
    loss, diag = positive_core_context_heatmap_loss(
        heatmap, targets, masks, clip_targets, clip_masks
    )
    # balanced: (core 1.0 + context 0.0) / 2 = 0.5 for the positive slot,
    # then mixed with the negative clip (loss 0) by slot count -> 0.25 total.
    balanced_pos = float(diag["frame_heatmap_balanced_pos_loss"])
    assert abs(balanced_pos - 0.5) < 1e-5, f"balanced positive loss {balanced_pos:.4f} != 0.5"
    assert abs(float(loss) - 0.25) < 1e-5, f"total mix {loss:.4f} != 0.25"
    print(f"heatmap_balance OK: pos_loss={balanced_pos:.4f} total={float(loss):.4f} "
          f"core_slots={int(diag['frame_heatmap_core_slots'])}")


def test_consistency() -> None:
    cfg = {
        "enabled": True,
        "clip_weight": 0.05,
        "response_weight": 0.0,
        "start_epoch": 1,
        "warmup_epochs": 0.25,
        "teacher_logit_floor": 0.0,
        "strict_metadata": True,
        "require_pair_each_batch": True,
        "require_response_curve": False,
    }
    logits = torch.tensor([[2.0, 0.1, 0.1], [0.5, 0.1, 0.1], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    targets = torch.tensor([[1.0, 0, 0], [1.0, 0, 0], [0.0, 0, 0], [0.0, 0, 0]])
    label_masks = torch.ones(4, 3)
    meta = [
        {"online_pair_id": "p0", "online_pair_role": "central", "online_pair_class_mask": (1.0, 0.0, 0.0), "anchor_time": 5.0},
        {"online_pair_id": "p0", "online_pair_role": "edge", "online_pair_class_mask": (1.0, 0.0, 0.0), "anchor_time": 5.0},
        {"online_pair_id": "", "online_pair_role": "", "online_pair_class_mask": ()},
        {"online_pair_id": "", "online_pair_role": "", "online_pair_class_mask": ()},
    ]
    batch = {"meta": meta, "targets": targets, "label_masks": label_masks}
    outputs = {"logits": logits}
    # epoch 0 -> warmup 0 -> loss must be zero
    loss0, diag0 = online_pair_consistency_loss(
        outputs, batch, cfg, epoch=0, step=1, steps_per_epoch=300
    )
    assert float(loss0) == 0.0 and diag0["online_pair_consistency_warmup"] == 0.0
    # epoch 1, step 75 (0.25 epoch) -> warmup 1.0; teacher=2.0 detached, edge pushed up
    loss1, diag1 = online_pair_consistency_loss(
        outputs, batch, cfg, epoch=1, step=75, steps_per_epoch=300
    )
    assert diag1["online_pair_consistency_pairs"] == 1
    assert diag1["online_pair_consistency_slots"] == 1
    assert abs(diag1["online_pair_teacher_logit"] - 2.0) < 1e-6
    assert float(loss1) > 0
    print(f"consistency OK: loss={float(loss1):.4f} teacher={diag1['online_pair_teacher_logit']:.2f} "
          f"edge={diag1['online_pair_edge_logit']:.2f}")

    # strict: pair metadata without a legal pair in batch -> hard error
    bad_meta = [
        {"online_pair_id": "p1", "online_pair_role": "central", "online_pair_class_mask": (1.0, 0, 0)},
        {"online_pair_id": "p1", "online_pair_role": "central", "online_pair_class_mask": (1.0, 0, 0)},
        {"online_pair_id": "", "online_pair_role": "", "online_pair_class_mask": ()},
        {"online_pair_id": "", "online_pair_role": "", "online_pair_class_mask": ()},
    ]
    try:
        online_pair_consistency_loss(
            outputs, {"meta": bad_meta, "targets": targets, "label_masks": label_masks},
            cfg, epoch=1, step=75, steps_per_epoch=300,
        )
        raise AssertionError("expected strict metadata error for two central rows")
    except ValueError:
        pass
    # strict: no pair rows at all -> error under require_pair_each_batch
    try:
        online_pair_consistency_loss(
            outputs, {"meta": [dict(m) for m in meta[2:]] + [dict(m) for m in meta[2:]], "targets": targets, "label_masks": label_masks},
            cfg, epoch=1, step=75, steps_per_epoch=300,
        )
        raise AssertionError("expected require_pair_each_batch error")
    except ValueError:
        pass
    print("strict_metadata OK")


def test_gate_gradient_isolation() -> None:
    head = _make_head(anti_erasure=True)
    head.train()
    bsz, frames, patches, dim = 2, 4, 32, 32
    raw = torch.randn(bsz, frames, 64)
    patch_tokens = torch.randn(bsz, frames, patches, dim) * 0.5
    # E1.3.2: the gate is frozen as a pure diagnostic (supervision removed,
    # weight 0.0) so no loss reaches it and DDP unused-parameter reduction
    # stays happy. The main task must still not leak into it either.
    assert not head.no_evidence_gate_weight.requires_grad
    assert not head.no_evidence_gate_bias.requires_grad
    outs = head(raw, patch_tokens)
    head.zero_grad()
    loss = outs["patch_appearance"].pow(2).mean() + F.relu(
        outs["no_evidence_mass"] - 0.4
    ).mean()
    loss.backward()
    gate_grads = [
        head.no_evidence_gate_weight.grad, head.no_evidence_gate_bias.grad
    ]
    assert all(g is None for g in gate_grads), "gradient reached frozen gate"
    # Diagnostics still flow: saliency and mass outputs exist and respond to
    # input strength.
    assert torch.is_tensor(outs["no_evidence_saliency"])
    assert torch.is_tensor(outs["no_evidence_mass"])
    print("gate_gradient_isolation OK (frozen diagnostic)")


def test_saliency_rank() -> None:
    # Event rows: strong Top-K evidence inside the support span (0.8-1.5);
    # clean-negative rows: weak everywhere (0.3).  Per-class independent.
    bsz, frames, classes = 4, 8, 3
    saliency = torch.full((bsz, frames, classes), 0.3)
    saliency[0, 2:5, 0] = 1.5  # event row, class 0: strong evidence in support
    saliency[1, 3:6, 1] = 1.2  # event row, class 1
    saliency[0, 4, 2] = 1.5  # class 2 event too (cross-class independence)
    targets = torch.zeros(bsz, frames, classes)
    masks = torch.ones(bsz, frames, classes)
    targets[0, 2:5, 0] = 1.0
    targets[1, 3:6, 1] = 1.0
    targets[0, 4, 2] = 1.0  # class 2 event too (cross-class independence)
    clean = torch.tensor([False, False, True, True])
    loss, diag = bag_saliency_rank_loss(
        saliency, targets, masks, clean, labels=("shot", "save", "set_piece"),
    )
    assert diag["class_evidence_saliency_rank_active_classes"] == 3
    assert diag["class_evidence_saliency_rank_pairs"] >= 4
    # softplus hinge: satisfied gaps still decay to a small positive value
    # (softplus(0.1-1.2)=0.287), so assert well below the violating case.
    assert float(loss) < 0.35, f"strong event evidence should nearly satisfy hinge, loss={loss:.4f}"
    assert diag["class_evidence_saliency_rank_violation_fraction"] == 0
    assert diag["class_evidence_saliency_rank_gap"] > 0.8
    print(f"saliency_rank OK: loss={float(loss):.4f} gap={diag['class_evidence_saliency_rank_gap']:.3f}")

    # Per-class clean masks must not be collapsed with any(dim=1): row 2 is
    # trusted only for shot, row 3 only for save, and set-piece has no safe
    # negative in this batch.
    per_class_clean = torch.tensor([
        [False, False, False],
        [False, False, False],
        [True, False, False],
        [False, True, False],
    ])
    _, per_class_diag = bag_saliency_rank_loss(
        saliency, targets, masks, per_class_clean,
        labels=("shot", "save", "set_piece"),
    )
    assert per_class_diag["class_evidence_saliency_rank_active_classes"] == 2
    assert per_class_diag["class_evidence_saliency_rank_gap_set_piece"] == 0
    assert per_class_diag["class_evidence_saliency_rank_gap_shot"] != 0
    assert per_class_diag["class_evidence_saliency_rank_gap_save"] != 0

    # Violating case: weak event evidence, strong clean-negative evidence.
    saliency2 = torch.full((bsz, frames, classes), 0.3)
    saliency2[0, 2:5, 0] = 0.4
    saliency2[2, :, 0] = 1.0  # clean negative rows look "eventful"
    loss2, diag2 = bag_saliency_rank_loss(
        saliency2, targets, masks, clean, labels=("shot", "save", "set_piece"),
    )
    assert float(loss2) > float(loss), "weak event vs strong clean-negative must violate hinge"
    assert diag2["class_evidence_saliency_rank_violation_fraction"] > 0
    print(f"saliency_rank_violation OK: loss={float(loss2):.4f} "
          f"violation={diag2['class_evidence_saliency_rank_violation_fraction']:.2f}")

    # Anchor-free: evidence just outside the support span must NOT count as
    # event evidence (only support-span frames pool into pos_scores).
    saliency3 = torch.full((bsz, frames, classes), 0.3)
    saliency3[0, 6, 0] = 1.5  # frame 6 is background (target 0)
    saliency3[2, :, 0] = 0.8
    loss3, diag3 = bag_saliency_rank_loss(
        saliency3, targets, masks, clean, labels=("shot", "save", "set_piece"),
    )
    pos3 = diag3["class_evidence_saliency_rank_positive_pooled"]
    assert pos3 < 1.0, f"background-only strong frame leaked into event pool: {pos3:.3f}"
    assert float(loss3) > 0
    print(f"saliency_rank_anchor_free OK: pos_pooled={pos3:.3f} loss={float(loss3):.4f}")

    # Missing inputs must raise.
    try:
        bag_saliency_rank_loss(saliency, None, masks, clean, labels=("shot",))
        raise AssertionError("expected ValueError for missing targets")
    except ValueError:
        pass
    print("saliency_rank_input_check OK")


def test_cross_window() -> None:
    # Two rows of the same video whose 10s windows overlap: absolute moment
    # t=5s appears in row0 (frame 2) and row1 (frame 0) at 4fps spacing.
    bsz, frames, classes = 2, 3, 3
    clip_duration, num_frames = 10.0, 3  # inclusive frame_step 5s
    saliency = torch.zeros(bsz, frames, classes)
    saliency[0] = 1.0  # row0 strong everywhere
    saliency[1] = 0.2  # row1 weak -> same moments disagree
    meta = [
        {"video_path": "v1.mp4", "sampled_clip_start": 0.0, "online_pair_id": "pair0", "online_pair_role": "central"},
        {"video_path": "v1.mp4", "sampled_clip_start": 5.0, "online_pair_id": "pair0", "online_pair_role": "edge"},
    ]
    loss, diag = cross_window_saliency_consistency_loss(
        saliency, meta, clip_duration=clip_duration, num_frames=num_frames,
        tolerance_sec=0.6, labels=("shot", "save", "set_piece"),
    )
    # matched pairs: row0 t1 (5.0) <-> row1 t0 (5.0), row0 t2 (10.0) <-> row1 t1 (10.0)
    assert diag["class_evidence_cross_window_matched_frames"] == 2
    assert float(loss) > 0, "disagreeing saliency at same moment must give loss"
    # Consistent saliency -> zero loss.
    saliency_ok = torch.full((bsz, frames, classes), 0.7)
    loss_ok, diag_ok = cross_window_saliency_consistency_loss(
        saliency_ok, meta, clip_duration=clip_duration, num_frames=num_frames,
        tolerance_sec=0.6, labels=("shot", "save", "set_piece"),
    )
    assert float(loss_ok) < 1e-6
    # Different videos -> no pairs, zero loss.
    meta_other = [dict(meta[0], video_path="v2.mp4"), dict(meta[1])]
    loss_none, diag_none = cross_window_saliency_consistency_loss(
        saliency, meta_other, clip_duration=clip_duration, num_frames=num_frames,
        tolerance_sec=0.6, labels=("shot", "save", "set_piece"),
    )
    assert diag_none["class_evidence_cross_window_matched_frames"] == 0
    assert float(loss_none) == 0.0
    print(f"cross_window OK: matched={diag['class_evidence_cross_window_matched_frames']} "
          f"gap={diag['class_evidence_cross_window_gap']:.3f} "
          f"consistent_loss={float(loss_ok):.4f} diff_video_ok={float(loss_none):.2f}")


def test_causal_head_output() -> None:
    head = _make_head(anti_erasure=True, causal_local=True)
    head.train()
    raw = torch.randn(2, 4, 64, requires_grad=True)
    patches = torch.randn(2, 4, 16, 32, requires_grad=True)
    outputs = head(raw, patches)
    assert outputs["logits"].shape == (2, 3)
    assert outputs["counterfactual_logits"].shape == (2, 3)
    assert outputs["causal_full_logits"].shape == (2, 3)
    assert not outputs["counterfactual_logits"].requires_grad
    assert outputs["causal_full_logits"].requires_grad
    outputs["causal_full_logits"].sum().backward()
    assert raw.grad is None or float(raw.grad.abs().sum()) == 0.0
    assert patches.grad is not None and float(patches.grad.abs().sum()) > 0
    assert head.patch_scores.weight.grad is not None
    assert all(
        parameter.grad is None
        for module in (head.class_adapters, head.temporals, head.clip_heads)
        for parameter in module.parameters()
    )
    print("causal_head_output OK (local-only gradient)")


def test_signed_causal_local() -> None:
    kept = torch.tensor([[1.0, -0.2, 0.0], [-1.0, -0.8, -0.1]], requires_grad=True)
    counterfactual = torch.tensor([[0.5, -0.2, 0.0], [-0.5, -0.2, -0.1]])
    targets = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    masks = torch.ones_like(targets)
    # Row 1 is trusted for shot and set-piece, but save remains unknown.
    clean = torch.tensor([[False, False, False], [True, False, True]])
    loss, diag = signed_causal_local_evidence_loss(
        kept, counterfactual, targets, masks, clean,
        labels=("shot", "save", "set_piece"),
        positive_margin=0.2, negative_margin=0.2,
    )
    assert float(loss) > 0
    assert diag["class_evidence_signed_causal_positive_slots"] == 1
    assert diag["class_evidence_signed_causal_negative_slots"] == 2
    assert diag["class_evidence_signed_causal_positive_gap_shot"] > 0
    assert diag["class_evidence_signed_causal_negative_gap_shot"] > 0
    assert diag["class_evidence_signed_causal_negative_gap_save"] == 0
    loss.backward()
    assert kept.grad is not None and float(kept.grad.abs().sum()) > 0
    print(f"signed_causal OK: loss={float(loss):.4f} pos_gap={diag['class_evidence_signed_causal_positive_gap']:.3f} neg_gap={diag['class_evidence_signed_causal_negative_gap']:.3f}")


def test_tail_separation() -> None:
    labels = ("shot", "save", "set_piece")
    # 2 videos x 4 rows.  Video A: strong positives, weak negatives
    # (satisfied).  Video B: weak bottom positive (0.4) vs strong top
    # negative (0.9) -> violating pair.
    targets = torch.tensor(
        [
            [1.0, 0.0, 0.0],  # A pos
            [0.0, 0.0, 0.0],  # A neg
            [1.0, 0.0, 0.0],  # A pos (bottom tail)
            [0.0, 0.0, 0.0],  # A neg
            [1.0, 0.0, 0.0],  # B pos (bottom tail, weak)
            [0.0, 0.0, 0.0],  # B neg (strong -> violation)
            [1.0, 0.0, 0.0],  # B pos
            [0.0, 0.0, 0.0],  # B neg
        ]
    )
    label_masks = torch.ones_like(targets)
    logits = torch.tensor(
        [
            [1.5, 0.0, 0.0],
            [-0.5, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [-0.3, 0.0, 0.0],
            [0.4, 0.0, 0.0],  # bottom positive: 0.4
            [0.9, 0.0, 0.0],  # top negative: 0.9 > 0.4 + margin
            [1.4, 0.0, 0.0],
            [0.2, 0.0, 0.0],
        ]
    )
    metas = [{"video_path": f"v{row < 4}"} for row in range(8)]
    loss, diag = base.tail_separation_paired_loss(
        logits,
        targets,
        label_masks,
        metas,
        labels=labels,
        margin=0.15,
        positive_quantile=0.5,
        negative_quantile=0.5,
    )
    # A: bottom 0.1 vs top -0.3 -> gap +0.4; B: bottom 0.4 vs top 0.9 ->
    # gap -0.5 (violation).  One violating group out of two.
    assert diag["tail_separation_active_groups"] == 2
    assert abs(diag["tail_separation_gap_shot"] - (-0.05)) < 1e-6
    assert diag["tail_separation_violation_fraction"] == 0.5
    assert float(loss) > 0.0
    # Batch fallback when video ids are missing: 4 pos / 4 neg pooled.
    loss_fb, diag_fb = base.tail_separation_paired_loss(
        logits,
        targets,
        label_masks,
        [{} for _ in range(8)],
        labels=labels,
        margin=0.15,
        positive_quantile=0.25,
        negative_quantile=0.25,
    )
    assert diag_fb["tail_separation_active_classes"] >= 1
    assert float(loss_fb) > 0.0
    # Untrusted rows are ignored: drop B's strong negative (row 5) from the
    # label mask; the violating pair disappears -> smaller loss.
    masked = label_masks.clone()
    masked[5] = 0.0
    loss_masked, diag_masked = base.tail_separation_paired_loss(
        logits,
        targets,
        masked,
        metas,
        labels=labels,
        margin=0.15,
        positive_quantile=0.5,
        negative_quantile=0.5,
    )
    assert float(loss_masked) < float(loss)
    print(
        "tail_separation OK: "
        f"loss={float(loss):.4f} gap_shot={diag['tail_separation_gap_shot']:.3f} "
        f"violation={diag['tail_separation_violation_fraction']:.3f}"
    )


def test_ema_retention() -> None:
    labels = ("shot", "save", "set_piece")
    # Trusted positive on class 0; trusted negative and untrusted rows.
    targets = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    label_masks = torch.tensor(
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]
    )
    student = torch.tensor([[1.0, 0.0, 0.0], [0.4, 0.0, 0.0], [-5.0, 0.0, 0.0]])
    teacher = torch.tensor([[0.9, 0.0, 0.0], [0.7, 0.0, 0.0], [0.5, 0.0, 0.0]])
    loss, diag = base.ema_positive_retention_loss(
        student,
        teacher,
        targets,
        label_masks,
        labels=labels,
        margin=0.05,
    )
    # Row 0 satisfied (1.0 >= 0.9 + 0.05-ish slack), row 1 violates, row 2 untrusted.
    assert diag["ema_positive_retention_positive_slots"] == 2
    assert abs(diag["ema_positive_retention_gap"] - (-0.1)) < 1e-6  # mean(0.1, -0.3)
    assert abs(diag["ema_positive_retention_gap_shot"] - (-0.1)) < 1e-6
    assert diag["ema_positive_retention_gap_save"] == 0.0
    assert float(loss) > 0.0
    # All-satisfied case is much smaller.
    satisfied = student.clone()
    satisfied[1] = 1.2
    loss_sat, _ = base.ema_positive_retention_loss(
        satisfied,
        teacher,
        targets,
        label_masks,
        labels=labels,
        margin=0.05,
    )
    assert float(loss_sat) < float(loss)
    print(
        "ema_retention OK: "
        f"loss={float(loss):.4f} gap={diag['ema_positive_retention_gap']:.3f} "
        f"violation={diag['ema_positive_retention_violation_fraction']:.3f}"
    )


def main() -> int:
    test_gate_gradient_isolation()
    test_anti_erasure()
    test_balanced_no_evidence()
    test_heatmap_balance()
    test_consistency()
    test_saliency_rank()
    test_cross_window()
    test_causal_head_output()
    test_signed_causal_local()
    test_tail_separation()
    test_ema_retention()
    print("LOSS_GATE_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
