from __future__ import annotations

import unittest

import torch

from football_dual_token_fusion import (
    DualViewTokenFusionTransformer,
    FullRoiCrossAttentionFusion,
    bounded_asymmetric_residual,
    staggered_local_indices,
    staggered_multi_roi_indices,
)
from train_football_events import VideoEventClassifier, configure_label_schema, to_config


class DualTokenFusionTest(unittest.TestCase):
    def setUp(self) -> None:
        configure_label_schema(to_config({"task": {"label_schema": "set_piece"}}))

    def test_staggered_indices_keep_overlap_and_add_midpoints(self) -> None:
        global_indices = list(range(0, 160, 10))
        local_indices = staggered_local_indices(global_indices, overlap_frames=8)
        self.assertEqual(len(local_indices), 16)
        self.assertEqual(local_indices, sorted(local_indices))
        self.assertEqual(len(set(global_indices) & set(local_indices)), 8)
        self.assertEqual(len(set(global_indices + local_indices)), 24)

    def test_multi_roi_indices_add_two_distinct_timelines(self) -> None:
        global_indices = list(range(0, 160, 10))
        roi_a, roi_b = staggered_multi_roi_indices(
            global_indices, segment_start=0, segment_end=159
        )
        self.assertEqual(len(roi_a), 16)
        self.assertEqual(len(roi_b), 16)
        self.assertEqual(roi_a, sorted(roi_a))
        self.assertEqual(roi_b, sorted(roi_b))
        self.assertFalse(set(roi_a) & set(roi_b))
        self.assertFalse(set(global_indices) & set(roi_a))
        self.assertFalse(set(global_indices) & set(roi_b))
        self.assertEqual(len(set(global_indices + roi_a + roi_b)), 48)

    def test_transformer_sorts_timestamps_and_tracks_views(self) -> None:
        module = DualViewTokenFusionTransformer(
            hidden_dim=8,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            num_labels=3,
        ).eval()
        outputs = module(
            torch.randn(2, 4, 8),
            torch.randn(2, 4, 8),
            torch.tensor([[0.0, 1.0, 2.0, 3.0]]).expand(2, -1),
            torch.tensor([[0.0, 0.5, 2.0, 2.5]]).expand(2, -1),
            torch.ones(2, 4),
        )
        self.assertEqual(tuple(outputs["attention"].shape), (2, 8, 3))
        self.assertEqual(tuple(outputs["query_features"].shape), (2, 3, 8))
        self.assertTrue(
            bool((outputs["times"][:, 1:] >= outputs["times"][:, :-1]).all())
        )
        self.assertEqual(
            outputs["view_ids"][0].tolist(),
            [0, 1, 1, 0, 0, 1, 1, 0],
        )
    def test_invalid_roi_tokens_are_masked_from_class_attention(self) -> None:
        module = DualViewTokenFusionTransformer(
            hidden_dim=8,
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            num_labels=3,
        ).eval()
        outputs = module(
            torch.randn(1, 2, 8),
            torch.randn(1, 2, 8),
            torch.tensor([[0.0, 1.0]]),
            torch.tensor([[0.0, 1.0]]),
            torch.zeros(1, 2),
        )
        roi_tokens = outputs["view_ids"] == 1
        self.assertFalse(bool(outputs["valid"][roi_tokens].any()))
        self.assertTrue(
            torch.equal(outputs["attention"][roi_tokens], torch.zeros(2, 3))
        )


    def test_bounded_residual_has_asymmetric_limits(self) -> None:
        raw = torch.tensor([[-100.0, 0.0, 100.0]], requires_grad=True)
        residual = bounded_asymmetric_residual(
            raw,
            positive_max=0.5,
            negative_max=2.0,
        )
        self.assertGreaterEqual(float(residual.min()), -2.0)
        self.assertLessEqual(float(residual.max()), 0.5)
        residual[:, 1].sum().backward()
        self.assertGreater(float(raw.grad[:, 1]), 0.0)

    def test_bounded_residual_accepts_per_class_limits(self) -> None:
        raw = torch.tensor([[10.0, 10.0, -10.0]])
        residual = bounded_asymmetric_residual(
            raw,
            positive_max=torch.tensor([0.1, 0.5, 1.0]),
            negative_max=torch.tensor([0.2, 1.0, 2.0]),
        )

        self.assertLessEqual(float(residual[0, 0]), 0.10001)
        self.assertLessEqual(float(residual[0, 1]), 0.50001)
        self.assertGreaterEqual(float(residual[0, 2]), -2.00001)

    def test_model_registers_per_class_view_fusion_delta(self) -> None:
        model = VideoEventClassifier(
            backbone=None,
            frame_feature_dim=8,
            hidden_dim=8,
            num_labels=3,
            fusion="cls_transformer",
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            max_frames=4,
            view_fusion="dual_class_query_fusion",
            roi_meta_dim=16,
            view_fusion_layers=1,
            view_fusion_heads=2,
            view_fusion_positive_delta_per_class={
                "shot": 0.1,
                "save": 0.5,
                "set_piece": 0.4,
            },
            view_fusion_negative_delta_per_class={
                "shot": 0.5,
                "save": 2.0,
                "set_piece": 1.5,
            },
        )

        self.assertTrue(
            torch.allclose(
                model.view_fusion_positive_delta.cpu(),
                torch.tensor([0.1, 0.5, 0.4]),
            )
        )
        self.assertTrue(
            torch.allclose(
                model.view_fusion_negative_delta.cpu(),
                torch.tensor([0.5, 2.0, 1.5]),
            )
        )

    def test_model_starts_as_exact_global_residual(self) -> None:
        model = VideoEventClassifier(
            backbone=None,
            frame_feature_dim=8,
            hidden_dim=8,
            num_labels=3,
            fusion="cls_transformer",
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            max_frames=4,
            view_fusion="dual_token_fusion",
            roi_meta_dim=16,
            view_fusion_layers=1,
            view_fusion_heads=2,
        ).eval()
        outputs = model(
            torch.randn(2, 4, 8),
            roi_inputs=torch.randn(2, 4, 8),
            roi_meta=torch.zeros(2, 16),
            roi_valid=torch.ones(2),
            roi_frame_valid=torch.ones(2, 4),
            roi_frame_meta=torch.zeros(2, 4, 16),
            global_frame_times=torch.tensor(
                [[0.0, 1.0, 2.0, 3.0]]
            ).expand(2, -1),
            local_frame_times=torch.tensor(
                [[0.0, 1.0, 2.0, 3.0]]
            ).expand(2, -1),
            return_aux=True,
        )
        self.assertTrue(torch.equal(outputs["logits"], outputs["global_logits"]))
        self.assertEqual(tuple(outputs["fused_frame_event_logits"].shape), (2, 4, 3))
        self.assertEqual(tuple(outputs["fused_frame_attention"].shape), (2, 8, 3))


        self.assertEqual(
            outputs["fused_frame_view_ids"][0].tolist(),
            [0, 0, 0, 0],
        )

    def test_cross_attention_starts_as_global_tokens(self) -> None:
        module = FullRoiCrossAttentionFusion(
            hidden_dim=8,
            num_heads=2,
            dropout=0.0,
            num_layers=2,
        ).eval()
        global_tokens = torch.randn(2, 4, 8)
        outputs = module(
            global_tokens,
            torch.randn(2, 4, 8),
            torch.tensor([[0.0, 1.0, 2.0, 3.0]]).expand(2, -1),
            torch.tensor([[0.0, 1.0, 2.0, 3.0]]).expand(2, -1),
            torch.ones(2, 4),
        )
        self.assertTrue(torch.equal(outputs["tokens"], global_tokens))
        self.assertEqual(tuple(outputs["attention"].shape), (2, 4, 4))

    def test_cross_attention_model_starts_as_exact_global(self) -> None:
        model = VideoEventClassifier(
            backbone=None,
            frame_feature_dim=8,
            hidden_dim=8,
            num_labels=3,
            fusion="cls_transformer",
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            max_frames=4,
            view_fusion="dual_cross_attention",
            roi_meta_dim=16,
            view_fusion_layers=1,
            view_fusion_heads=2,
        ).eval()
        outputs = model(
            torch.randn(2, 4, 8),
            roi_inputs=torch.randn(2, 4, 8),
            roi_meta=torch.zeros(2, 16),
            roi_valid=torch.ones(2),
            roi_frame_valid=torch.ones(2, 4),
            roi_frame_meta=torch.zeros(2, 4, 16),
            global_frame_times=torch.tensor([[0.0, 1.0, 2.0, 3.0]]).expand(2, -1),
            local_frame_times=torch.tensor([[0.0, 1.0, 2.0, 3.0]]).expand(2, -1),
            return_aux=True,
        )
        self.assertTrue(torch.equal(outputs["logits"], outputs["global_logits"]))
        self.assertEqual(tuple(outputs["frame_event_logits"].shape), (2, 4, 3))
        self.assertNotIn("local_frame_event_logits", outputs)

    def test_class_query_fusion_uses_fused_frame_logits(self) -> None:
        model = VideoEventClassifier(
            backbone=None,
            frame_feature_dim=8,
            hidden_dim=8,
            num_labels=3,
            fusion="cls_transformer",
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            max_frames=4,
            view_fusion="dual_class_query_fusion",
            roi_meta_dim=16,
            view_fusion_layers=1,
            view_fusion_heads=2,
        ).eval()
        outputs = model(
            torch.randn(2, 4, 8),
            roi_inputs=torch.randn(2, 4, 8),
            roi_meta=torch.zeros(2, 16),
            roi_valid=torch.ones(2),
            roi_frame_valid=torch.ones(2, 4),
            roi_frame_meta=torch.zeros(2, 4, 16),
            global_frame_times=torch.tensor([[0.0, 1.0, 2.0, 3.0]]).expand(2, -1),
            local_frame_times=torch.tensor([[0.0, 1.0, 2.0, 3.0]]).expand(2, -1),
            return_aux=True,
        )
        self.assertTrue(torch.equal(outputs["logits"], outputs["global_logits"]))
        self.assertEqual(tuple(outputs["frame_event_logits"].shape), (2, 4, 3))
        self.assertEqual(tuple(outputs["fused_frame_event_logits"].shape), (2, 4, 3))
        self.assertNotIn("local_frame_event_logits", outputs)

    def test_class_query_fusion_restores_global_tokens_for_frame_logits(self) -> None:
        class DummyMixedTokenTemporal(torch.nn.Module):
            def forward(
                self,
                global_tokens,
                local_tokens,
                global_times,
                local_times,
                local_quality,
            ):
                batch_size, _, hidden_dim = global_tokens.shape
                encoded = global_tokens.new_zeros((batch_size, 8, hidden_dim))
                encoded[:, :, 0] = torch.tensor(
                    [10.0, 100.0, 11.0, 101.0, 12.0, 102.0, 13.0, 103.0],
                    device=encoded.device,
                    dtype=encoded.dtype,
                )
                return {
                    "query_features": global_tokens.new_zeros((batch_size, 3, hidden_dim)),
                    "temporal": global_tokens.new_zeros((batch_size, hidden_dim)),
                    "attention": global_tokens.new_zeros((batch_size, 8, 3)),
                    "encoded_tokens": encoded,
                    "times": global_tokens.new_tensor(
                        [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
                    ).expand(batch_size, -1),
                    "view_ids": torch.tensor(
                        [[0, 1, 0, 1, 0, 1, 0, 1]],
                        device=global_tokens.device,
                    ).expand(batch_size, -1),
                    "order": torch.tensor(
                        [[0, 4, 1, 5, 2, 6, 3, 7]],
                        device=global_tokens.device,
                    ).expand(batch_size, -1),
                    "valid": torch.ones(
                        batch_size, 8, dtype=torch.bool, device=global_tokens.device
                    ),
                }

        model = VideoEventClassifier(
            backbone=None,
            frame_feature_dim=4,
            hidden_dim=4,
            num_labels=3,
            fusion="cls_transformer",
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            max_frames=4,
            view_fusion="dual_class_query_fusion",
            roi_meta_dim=16,
            view_fusion_layers=1,
            view_fusion_heads=2,
        ).eval()
        model.dual_token_temporal = DummyMixedTokenTemporal()
        model.frame_event_head = torch.nn.Linear(4, 3, bias=False)
        with torch.no_grad():
            model.frame_event_head.weight.zero_()
            model.frame_event_head.weight[:, 0] = 1.0

        outputs = model(
            torch.randn(1, 4, 4),
            roi_inputs=torch.randn(1, 4, 4),
            roi_meta=torch.zeros(1, 16),
            roi_valid=torch.ones(1),
            roi_frame_valid=torch.ones(1, 4),
            roi_frame_meta=torch.zeros(1, 4, 16),
            global_frame_times=torch.tensor([[0.0, 1.0, 2.0, 3.0]]),
            local_frame_times=torch.tensor([[0.0, 1.0, 2.0, 3.0]]),
            return_aux=True,
        )

        expected = torch.tensor([10.0, 11.0, 12.0, 13.0])
        self.assertTrue(
            torch.equal(outputs["fused_frame_event_logits"][0, :, 0], expected)
        )
        self.assertTrue(
            torch.equal(outputs["frame_event_logits"][0, :, 0], expected)
        )

if __name__ == "__main__":
    unittest.main()
