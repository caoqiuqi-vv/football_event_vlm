from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import torch

from football_object_spatial_aux import object_teacher_heatmap_loss


class ObjectTeacherHeatmapLossTest(unittest.TestCase):
    def test_negative_weights_are_not_canceled_by_normalization(self) -> None:
        outputs = {
            "object_heatmap_logits": torch.zeros(1, 1, 1, 2),
            "object_spatial_residual": torch.zeros(1, 3),
        }
        batch = {
            "object_heatmap_targets": torch.zeros(1, 1, 1, 2),
            "object_heatmap_masks": torch.ones(1, 1, 1, 2),
        }
        cfg = SimpleNamespace(
            model={
                "object_spatial_aux": {
                    "negative_weights": [0.02, 0.08],
                    "positive_threshold": 0.05,
                }
            }
        )
        loss, components = object_teacher_heatmap_loss(
            outputs, batch, cfg, torch.device("cpu")
        )
        expected = math.log(2.0) * (0.02 + 0.08) / 2.0
        self.assertAlmostEqual(float(loss), expected, places=6)
        self.assertAlmostEqual(components["object_heatmap_negative_loss"], expected, places=6)


if __name__ == "__main__":
    unittest.main()
