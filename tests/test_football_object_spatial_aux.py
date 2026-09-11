from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from football_object_spatial_aux import ObjectSpatialAuxHead, ObjectTeacherTargetProvider


class ObjectTeacherTargetProviderTest(unittest.TestCase):
    def test_ball_and_goal_are_patch_aligned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            torch.save(
                {
                    "version": 3,
                    "video_id": "v1",
                    "fps": 10.0,
                    "image_size": {"width": 160, "height": 80},
                    "frame_ids": torch.tensor([10], dtype=torch.int32),
                    "frame_offsets": torch.tensor([0, 2], dtype=torch.int64),
                    "classes": torch.tensor([1, 2], dtype=torch.int8),
                    "confidences": torch.tensor([0.9, 0.8]),
                    "boxes": torch.tensor(
                        [[72.0, 32.0, 88.0, 48.0], [120.0, 16.0, 159.0, 64.0]]
                    ),
                    "provenance": {"ball_teacher": "unit", "goal_teacher": "unit"},
                },
                root / "v1.pt",
            )
            provider = ObjectTeacherTargetProvider(
                root, image_size=(80, 160), patch_size=16, max_frame_gap_sec=0.2
            )
            targets, masks = provider.targets("v1", [1.0])
            self.assertEqual(tuple(targets.shape), (1, 50, 2))
            self.assertTrue(bool((masks == 1).all()))
            ball_peak = int(targets[0, :, 0].argmax())
            self.assertEqual((ball_peak // 10, ball_peak % 10), (2, 4))
            self.assertGreater(float(targets[0, :, 1].sum()), 1.0)

    def test_missing_video_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = ObjectTeacherTargetProvider(
                tmp, image_size=(80, 160), patch_size=16
            )
            targets, masks = provider.targets("missing", [0.0, 1.0])
            self.assertEqual(float(targets.sum()), 0.0)
            self.assertEqual(float(masks.sum()), 0.0)


class ObjectSpatialAuxHeadTest(unittest.TestCase):
    def test_residual_is_exactly_zero_at_initialization(self) -> None:
        torch.manual_seed(7)
        head = ObjectSpatialAuxHead(
            patch_dim=32,
            hidden_dim=16,
            num_labels=3,
            temporal_layers=1,
            dropout=0.0,
            topk_ratio=0.1,
        ).eval()
        outputs = head(torch.randn(2, 4, 20, 32))
        self.assertEqual(tuple(outputs["heatmap_logits"].shape), (2, 4, 20, 2))
        self.assertTrue(torch.equal(outputs["residual"], torch.zeros_like(outputs["residual"])))
        self.assertTrue(
            torch.allclose(outputs["attention_maps"].sum(dim=2), torch.ones(2, 4, 2))
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for bf16 autocast")
    def test_bf16_autocast_forward_and_backward(self) -> None:
        torch.manual_seed(11)
        head = ObjectSpatialAuxHead(
            patch_dim=32,
            hidden_dim=16,
            num_labels=3,
            temporal_layers=1,
            dropout=0.0,
            topk_ratio=0.1,
        ).cuda()
        tokens = torch.randn(2, 4, 20, 32, device="cuda", requires_grad=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = head(tokens)
            loss = outputs["heatmap_logits"].float().square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(outputs["attention_maps"].dtype, torch.bfloat16)
        self.assertIsNotNone(head.heatmap_head[-1].weight.grad)

if __name__ == "__main__":
    unittest.main()
