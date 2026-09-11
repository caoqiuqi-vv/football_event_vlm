import unittest

import torch
from torch import nn
from torch.nn import functional as F

from train_football_events import VideoEventClassifier


class TinyPatchBackbone(nn.Module):
    num_features = 8

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, self.num_features, bias=False)

    def forward_features(self, x):
        patches = F.avg_pool2d(x, kernel_size=16, stride=16)
        patches = patches.flatten(2).transpose(1, 2)
        patches = self.proj(patches)
        return {
            "x_norm_clstoken": patches.mean(dim=1),
            "x_norm_patchtokens": patches,
        }


class HighResolutionGlimpseTest(unittest.TestCase):
    def test_forward_and_crop_selector_gradient(self):
        torch.manual_seed(7)
        backbone = TinyPatchBackbone()
        for parameter in backbone.parameters():
            parameter.requires_grad = False
        model = VideoEventClassifier(
            backbone=backbone,
            frame_feature_dim=16,
            hidden_dim=16,
            num_labels=3,
            fusion="cls_transformer",
            num_layers=1,
            num_heads=4,
            dropout=0.0,
            max_frames=4,
            highres_glimpse_enabled=True,
            highres_glimpse_candidates=2,
            highres_glimpse_frames_per_candidate=2,
            highres_glimpse_crop_size=32,
            highres_glimpse_attention_dim=8,
            highres_glimpse_nms_radius=1,
        )
        global_inputs = torch.randint(0, 256, (2, 4, 3, 32, 48), dtype=torch.uint8)
        dense_inputs = torch.randint(0, 256, (2, 8, 3, 48, 64), dtype=torch.uint8)
        clip_targets = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        clip_masks = torch.ones_like(clip_targets)
        outputs = model(
            global_inputs,
            highres_pool_inputs=dense_inputs,
            clip_targets=clip_targets,
            clip_label_masks=clip_masks,
            return_aux=True,
        )
        self.assertEqual(outputs["logits"].shape, (2, 3))
        self.assertEqual(outputs["highres_local_frame_event_logits"].shape, (2, 4, 3))
        self.assertEqual(outputs["highres_crop_centers"].shape, (2, 2, 2))
        self.assertEqual(outputs["highres_crop_scales"].shape, (2, 2, 2))
        selected = outputs["highres_candidate_indices"]
        self.assertTrue(torch.all((selected[:, 0] - selected[:, 1]).abs() > 1))

        outputs["logits"].sum().backward()
        query_grad = model.highres_crop_query[1].weight.grad
        scale_grad = model.highres_crop_scale[1].weight.grad
        fusion_grad = model.highres_token_fusion[1].weight.grad
        self.assertIsNotNone(query_grad)
        self.assertIsNotNone(scale_grad)
        self.assertIsNotNone(fusion_grad)
        self.assertGreater(float(query_grad.abs().sum()), 0.0)
        self.assertGreater(float(scale_grad.abs().sum()), 0.0)
        self.assertGreater(float(fusion_grad.abs().sum()), 0.0)
        self.assertIn("highres_local_logits", outputs)
        self.assertIn("highres_shuffled_local_logits", outputs)
        self.assertEqual(outputs["highres_shuffled_local_logits"].shape, (2, 1, 3))
        self.assertEqual(outputs["highres_causal_valid_mask"].shape, (2, 1, 3))
        expected_donor_mask = torch.tensor(
            [[[1.0, 0.0, 1.0]], [[0.0, 1.0, 1.0]]]
        )
        torch.testing.assert_close(
            outputs["highres_causal_valid_mask"], expected_donor_mask
        )
        self.assertEqual(outputs["highres_evidence_gates"].shape, (2, 2, 1))

        # Local auxiliary supervision must shape the crop/evidence modules but
        # must not update the historical primary temporal model or classifier.
        model.zero_grad(set_to_none=True)
        local_outputs = model(
            global_inputs,
            highres_pool_inputs=dense_inputs,
            clip_targets=clip_targets,
            clip_label_masks=clip_masks,
            return_aux=True,
        )
        local_outputs["highres_local_logits"].sum().backward()
        self.assertTrue(all(p.grad is None for p in model.temporal.parameters()))
        self.assertTrue(all(p.grad is None for p in model.head.parameters()))
        self.assertIsNotNone(model.highres_local_classifier[1].weight.grad)
        self.assertIsNotNone(model.highres_crop_query[1].weight.grad)


if __name__ == "__main__":
    unittest.main()
