import torch

import train_football_events as training

training.configure_label_schema({"task": {"label_schema": "set_piece"}})


def test_save_implies_shot_and_set_piece_masks_unlabeled_shot():
    shot, save, set_piece = (training.LABELS.index(name) for name in ("shot", "save", "set_piece"))
    targets = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    masks = torch.ones_like(targets)
    frame_targets = torch.zeros(2, 4, 3)
    frame_targets[0, 2, save] = 1.0
    frame_masks = torch.ones_like(frame_targets)
    batch = {"frame_targets": frame_targets, "frame_target_masks": frame_masks}
    cfg = {
        "label_semantics": {
            "enabled": True,
            "positive_implications": {"save": ["shot"]},
            "ambiguous_negative_masks": {"set_piece": ["shot"]},
        }
    }

    resolved_targets, resolved_masks, diagnostics = training.apply_training_label_semantics(
        targets, masks, batch, cfg
    )

    assert resolved_targets[0, shot].item() == 1.0
    assert batch["frame_targets"][0, 2, shot].item() == 1.0
    assert resolved_masks[1, shot].item() == 0.0
    assert batch["frame_target_masks"][1, :, shot].sum().item() == 0.0
    assert diagnostics["label_semantics_implied_save_to_shot"] == 1.0
    assert diagnostics["label_semantics_masked_set_piece_to_shot"] == 1.0


def test_explicit_shot_is_not_masked_by_set_piece():
    targets = torch.tensor([[1.0, 0.0, 1.0]])
    masks = torch.ones_like(targets)
    resolved_targets, resolved_masks, _ = training.apply_training_label_semantics(
        targets, masks, {},
        {"label_semantics": {"enabled": True, "ambiguous_negative_masks": {"set_piece": ["shot"]}}},
    )
    assert resolved_targets[0, training.LABELS.index("shot")].item() == 1.0
    assert resolved_masks[0, training.LABELS.index("shot")].item() == 1.0
