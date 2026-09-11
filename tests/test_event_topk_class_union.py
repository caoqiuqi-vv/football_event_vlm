from __future__ import annotations

import unittest

import torch

from train_football_events import TemporalTransformer, VideoEventClassifier


class EventTopkClassUnionTest(unittest.TestCase):
    def test_class_union_uses_shot_save_and_ignores_set_piece_peak(self) -> None:
        model = VideoEventClassifier(
            backbone=None,
            frame_feature_dim=4,
            hidden_dim=8,
            num_labels=3,
            fusion="event_topk_transformer",
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            max_frames=8,
            event_topk=4,
            context_frames=0,
            event_topk_strategy="class_union",
            event_topk_per_class=2,
            event_topk_class_indices=(0, 1),
        )
        frame_tokens = torch.zeros(1, 8, 8)
        logits = torch.zeros(1, 8, 3)
        logits[0, 1, 0] = 7.0
        logits[0, 6, 0] = 6.0
        logits[0, 2, 1] = 7.0
        logits[0, 7, 1] = 6.0
        logits[0, 4, 2] = 20.0
        _, indices = model._select_event_topk(frame_tokens, logits)
        self.assertEqual(indices[0].tolist(), [1, 2, 6, 7])

    def test_temporal_transformer_accepts_original_position_indices(self) -> None:
        temporal = TemporalTransformer(
            hidden_dim=8,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            max_frames=8,
        )
        output = temporal(
            torch.randn(2, 4, 8),
            position_indices=torch.tensor([[0, 2, 5, 7], [1, 3, 4, 6]]),
        )
        self.assertEqual(tuple(output.shape), (2, 8))


    def test_straight_through_keeps_hard_forward_and_backpropagates(self) -> None:
        model = VideoEventClassifier(
            backbone=None,
            frame_feature_dim=4,
            hidden_dim=8,
            num_labels=3,
            fusion="event_topk_transformer",
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            max_frames=8,
            event_topk=4,
            context_frames=0,
            event_topk_strategy="class_union",
            event_topk_per_class=2,
            event_topk_class_indices=(0, 1),
            event_topk_gradient="straight_through",
            event_topk_temperature=1.0,
        )
        frame_tokens = torch.randn(1, 8, 8, requires_grad=True)
        frame_logits = torch.randn(1, 8, 3, requires_grad=True)
        selected, indices = model._select_event_topk(frame_tokens, frame_logits)
        expected = frame_tokens.gather(
            1, indices.unsqueeze(-1).expand(-1, -1, frame_tokens.shape[-1])
        )
        self.assertTrue(torch.equal(selected.detach(), expected.detach()))
        selected.square().sum().backward()
        self.assertIsNotNone(frame_logits.grad)
        self.assertTrue(torch.isfinite(frame_logits.grad).all())
        self.assertGreater(float(frame_logits.grad.abs().sum()), 0.0)

    def test_event_anchor_selects_dense_neighborhoods_and_context(self) -> None:
        model = VideoEventClassifier(
            backbone=None,
            frame_feature_dim=4,
            hidden_dim=8,
            num_labels=3,
            fusion="event_anchor_transformer",
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            max_frames=12,
            context_frames=2,
            event_anchor_topk_per_class=(1, 0, 1),
            event_anchor_class_indices=(0, 1, 2),
            event_anchor_offsets=(-1, 0, 1),
            event_anchor_nms_radius=2,
            event_anchor_max_frames=8,
        )
        frame_tokens = torch.zeros(1, 12, 8)
        logits = torch.zeros(1, 12, 3)
        logits[0, 2, 0] = 7.0
        logits[0, 8, 2] = 9.0
        _, indices = model._select_event_anchor_frames(frame_tokens, logits)
        self.assertEqual(indices[0].tolist(), [0, 1, 2, 3, 7, 8, 9, 11])

    def test_event_anchor_nms_keeps_separated_anchors(self) -> None:
        model = VideoEventClassifier(
            backbone=None,
            frame_feature_dim=4,
            hidden_dim=8,
            num_labels=3,
            fusion="event_anchor_transformer",
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            max_frames=12,
            context_frames=0,
            event_anchor_topk_per_class=(2, 0, 0),
            event_anchor_class_indices=(0,),
            event_anchor_offsets=(0,),
            event_anchor_nms_radius=2,
            event_anchor_max_frames=2,
        )
        frame_tokens = torch.zeros(1, 12, 8)
        logits = torch.zeros(1, 12, 3)
        logits[0, 4, 0] = 9.0
        logits[0, 5, 0] = 8.0
        logits[0, 9, 0] = 7.0
        _, indices = model._select_event_anchor_frames(frame_tokens, logits)
        self.assertEqual(indices[0].tolist(), [4, 9])


if __name__ == "__main__":
    unittest.main()
