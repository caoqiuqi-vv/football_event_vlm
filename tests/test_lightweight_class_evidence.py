import unittest

import torch

from train_football_events import LightweightClassEvidenceHead, VideoEventClassifier


class LightweightClassEvidenceHeadTest(unittest.TestCase):
    def make_head(self):
        return LightweightClassEvidenceHead(
            patch_dim=32,
            hidden_dim=24,
            attention_dim=16,
            evidence_dim=20,
            num_labels=3,
            queries_per_class=2,
            topk_per_class=(4, 3, 2),
            context_frames_per_class=(2, 2, 2),
            positive_max_per_class=(0.5, 0.6, 1.2),
            negative_max_per_class=(1.5, 1.8, 0.4),
            dropout=0.0,
        )

    def test_zero_start_and_class_specific_selection(self):
        torch.manual_seed(7)
        head = self.make_head().eval()
        global_tokens = torch.randn(2, 8, 24)
        patches = torch.randn(2, 8, 12, 32)
        frame_logits = torch.randn(2, 8, 3)
        frame_logits[0, 5, 1] = 20.0
        outputs = head(global_tokens, patches, frame_logits)
        self.assertEqual(outputs["correction"].shape, (2, 3))
        self.assertTrue(torch.equal(
            outputs["correction"], torch.zeros_like(outputs["correction"])
        ))
        self.assertIn(5, outputs["selected_indices"][0, 1].tolist())
        self.assertEqual(outputs["attention_entropy"].shape[:4], (2, 3, 6, 2))
        self.assertEqual(
            outputs["selected_valid"][0].sum(dim=1).tolist(), [6, 5, 4]
        )
        self.assertTrue((outputs["temporal_weights"][~outputs["selected_valid"]] == 0).all())

    def test_gradient_reaches_all_class_correction_weights(self):
        torch.manual_seed(11)
        head = self.make_head().train()
        outputs = head(
            torch.randn(2, 8, 24),
            torch.randn(2, 8, 12, 32),
            torch.randn(2, 8, 3),
        )
        targets = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            outputs["correction"], targets
        )
        loss.backward()
        self.assertIsNotNone(head.correction_weight.grad)
        self.assertTrue((head.correction_weight.grad.abs().sum(dim=1) > 0).all())


    def test_classifier_aux_output_integration(self):
        class DummyDino(torch.nn.Module):
            num_features = 32

            def forward_features(self, frames):
                pooled = frames.mean(dim=(1, 2, 3)).unsqueeze(-1)
                cls = pooled.expand(-1, self.num_features)
                patches = cls.unsqueeze(1).expand(-1, 4, -1)
                return {
                    "x_norm_clstoken": cls,
                    "x_norm_patchtokens": patches,
                }

        model = VideoEventClassifier(
            backbone=DummyDino(),
            frame_feature_dim=64,
            hidden_dim=24,
            num_labels=3,
            fusion="cls_transformer",
            num_layers=1,
            num_heads=4,
            dropout=0.0,
            max_frames=8,
            frame_feature_mode="last_cls_patch_mean",
            class_evidence_enabled=True,
            class_evidence_attention_dim=16,
            class_evidence_hidden_dim=20,
            class_evidence_queries_per_class=2,
            class_evidence_topk_per_class=(4, 3, 2),
            class_evidence_context_frames_per_class=(2, 2, 2),
            class_evidence_positive_max_per_class=(0.5, 0.6, 1.2),
            class_evidence_negative_max_per_class=(1.5, 1.8, 0.4),
        ).eval()
        with torch.inference_mode():
            outputs = model(torch.randn(2, 8, 3, 16, 16), return_aux=True)
        self.assertIn("class_evidence_selected_valid", outputs)
        self.assertIn("class_evidence_temporal_weights", outputs)
        self.assertEqual(
            outputs["class_evidence_selected_valid"].shape, (2, 3, 6)
        )
        self.assertTrue(torch.equal(outputs["logits"], outputs["global_logits"]))

    def test_global_local_temporal_starts_from_transferred_baseline(self):
        class DummyDino(torch.nn.Module):
            num_features = 32

            def forward_features(self, frames):
                pooled = frames.mean(dim=(1, 2, 3)).unsqueeze(-1)
                cls = pooled.expand(-1, self.num_features)
                patches = cls.unsqueeze(1).expand(-1, 4, -1)
                return {
                    "x_norm_clstoken": cls,
                    "x_norm_patchtokens": patches,
                }

        model = VideoEventClassifier(
            backbone=DummyDino(),
            frame_feature_dim=64,
            hidden_dim=24,
            num_labels=3,
            fusion="cls_transformer",
            num_layers=2,
            num_heads=4,
            dropout=0.0,
            max_frames=8,
            frame_feature_mode="last_cls_patch_mean",
            class_evidence_enabled=True,
            class_evidence_mode="global_local_temporal",
            class_evidence_attention_dim=16,
            class_evidence_queries_per_class=2,
        ).eval()
        inputs = torch.randn(2, 8, 3, 16, 16)
        with torch.inference_mode():
            outputs = model(inputs, return_aux=True)
        torch.testing.assert_close(outputs["logits"], outputs["global_logits"])
        self.assertEqual(
            outputs["class_evidence_local_tokens"].shape, (2, 8, 3, 24)
        )
        self.assertEqual(
            outputs["class_evidence_fused_frame_tokens"].shape, (2, 8, 24)
        )
        self.assertEqual(
            outputs["class_evidence_frame_event_logits"].shape, (2, 8, 3)
        )
        self.assertEqual(
            outputs["class_evidence_presence"].shape, (2, 8, 3)
        )
        self.assertEqual(
            outputs["class_evidence_no_evidence_logits"].shape, (2, 3)
        )
        self.assertNotIn("spatial_fusion_delta", outputs)

        model.train()
        train_outputs = model(inputs, return_aux=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            train_outputs["logits"], torch.rand(2, 3)
        )
        loss.backward()
        head = model.class_evidence_head
        self.assertIsNotNone(head)
        self.assertIsNotNone(head.fusion_proj.weight.grad)
        local_grad = head.fusion_proj.weight.grad[:, 24:]
        self.assertGreater(float(local_grad.abs().sum()), 0.0)

    def test_local_frame_supervision_reaches_queries_on_first_step(self):
        class DummyDino(torch.nn.Module):
            num_features = 32

            def forward_features(self, frames):
                pooled = frames.mean(dim=(1, 2, 3)).unsqueeze(-1)
                cls = pooled.expand(-1, self.num_features)
                # Spatially distinct patches are required to test attention
                # query gradients rather than a degenerate uniform map.
                spatial_pattern = torch.arange(
                    4 * self.num_features,
                    device=frames.device,
                    dtype=frames.dtype,
                ).reshape(1, 4, self.num_features).sin()
                temporal_pattern = torch.linspace(
                    -1.0, 1.0, self.num_features,
                    device=frames.device,
                    dtype=frames.dtype,
                ).reshape(1, 1, self.num_features)
                patches = spatial_pattern + pooled.reshape(-1, 1, 1) * temporal_pattern
                return {
                    "x_norm_clstoken": cls,
                    "x_norm_patchtokens": patches,
                }

        model = VideoEventClassifier(
            backbone=DummyDino(), frame_feature_dim=64, hidden_dim=24,
            num_labels=3, fusion="cls_transformer", num_layers=1,
            num_heads=4, dropout=0.0, max_frames=8,
            frame_feature_mode="last_cls_patch_mean",
            class_evidence_enabled=True,
            class_evidence_mode="global_local_temporal",
            class_evidence_attention_dim=16,
            class_evidence_queries_per_class=3,
        ).train()
        outputs = model(torch.randn(2, 8, 3, 16, 16), return_aux=True)
        frame_targets = torch.rand_like(outputs["class_evidence_frame_event_logits"])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            outputs["class_evidence_frame_event_logits"], frame_targets
        )
        loss.backward()
        head = model.class_evidence_head
        self.assertIsNotNone(head)
        self.assertIsNotNone(head.base_queries.grad)
        self.assertGreater(float(head.base_queries.grad.abs().sum()), 0.0)
        self.assertIsNotNone(head.motion_value.weight.grad)
        self.assertGreater(float(head.motion_value.weight.grad.abs().sum()), 0.0)

if __name__ == "__main__":
    unittest.main()
