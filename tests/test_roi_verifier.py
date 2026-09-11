import unittest

import torch

from train_football_events import VideoEventClassifier, positive_retention_loss


def make_model(
    roi_verifier_class_indices: tuple[int, ...] = (),
) -> VideoEventClassifier:
    return VideoEventClassifier(
        backbone=None,
        frame_feature_dim=8,
        hidden_dim=8,
        num_labels=3,
        fusion="cls_transformer",
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        max_frames=4,
        view_fusion="dual_verifier",
        roi_meta_dim=16,
        roi_verifier_min_confidence=0.6,
        roi_verifier_class_indices=roi_verifier_class_indices,
    )


class RoiVerifierTest(unittest.TestCase):
    def test_low_confidence_is_exact_global_fallback(self):
        model = make_model().eval()
        with torch.no_grad():
            model.roi_gate[-1].bias.fill_(5.0)
            model.roi_residual_head[-1].bias.fill_(-1.0)
        inputs = torch.randn(2, 4, 8)
        roi_inputs = torch.randn(2, 4, 8)
        roi_meta = torch.zeros(2, 16)
        roi_meta[:, 1] = torch.tensor([0.59, 1.0])
        outputs = model(
            inputs,
            roi_inputs=roi_inputs,
            roi_meta=roi_meta,
            roi_valid=torch.ones(2),
            return_aux=True,
        )
        self.assertTrue(torch.equal(outputs["logits"][0], outputs["global_logits"][0]))
        self.assertFalse(torch.equal(outputs["logits"][1], outputs["global_logits"][1]))
        self.assertTrue(torch.all(outputs["logits"][1] <= outputs["global_logits"][1]))

    def test_class_mask_preserves_unverified_label_exactly(self):
        model = make_model((0, 1)).eval()
        with torch.no_grad():
            model.roi_gate[-1].bias.fill_(5.0)
            model.roi_residual_head[-1].bias.fill_(1.0)
        roi_meta = torch.zeros(2, 16)
        roi_meta[:, 1] = 1.0
        outputs = model(
            torch.randn(2, 4, 8),
            roi_inputs=torch.randn(2, 4, 8),
            roi_meta=roi_meta,
            roi_valid=torch.ones(2),
            return_aux=True,
        )

        torch.testing.assert_close(
            outputs["logits"][:, 2], outputs["global_logits"][:, 2]
        )
        torch.testing.assert_close(outputs["roi_gate"][:, 2], torch.zeros(2))
        self.assertTrue(torch.all(outputs["logits"][:, :2] <= outputs["global_logits"][:, :2]))
    def test_freeze_global_includes_frame_event_head(self):
        model = make_model()
        model.freeze_global_parameters()
        for module in (model.frame_proj, model.frame_event_head, model.temporal, model.head):
            self.assertTrue(all(not parameter.requires_grad for parameter in module.parameters()))
        self.assertTrue(any(parameter.requires_grad for parameter in model.local_temporal.parameters()))
        self.assertTrue(any(parameter.requires_grad for parameter in model.roi_residual_head.parameters()))

    def test_positive_retention_only_uses_labeled_positives(self):
        fused = torch.tensor([[0.0, -2.0, -3.0]], requires_grad=True)
        outputs = {
            "global_logits": torch.tensor([[1.0, 2.0, 3.0]]),
            "logits": fused,
        }
        targets = torch.tensor([[1.0, 0.0, 1.0]])
        masks = torch.tensor([[1.0, 1.0, 0.0]])
        loss = positive_retention_loss(outputs, targets, masks)
        self.assertAlmostEqual(float(loss), 1.0, places=6)
        loss.backward()
        self.assertLess(float(fused.grad[0, 0]), 0.0)
        self.assertEqual(float(fused.grad[0, 1]), 0.0)
        self.assertEqual(float(fused.grad[0, 2]), 0.0)


if __name__ == "__main__":
    unittest.main()
