from __future__ import annotations

import unittest

import torch

from football_longform_v2.decoding import decode_family_conditioned_classes
from football_longform_v2.evaluation import (
    average_precision, infer_continuous_timeline, operating_point_at_recall,
)
from football_longform_v2.feature_store import AlignedTimeline
from football_longform_v2.models import TemporalLocator


class EvaluationTest(unittest.TestCase):
    def test_average_precision_counts_unproposed_positive(self) -> None:
        self.assertAlmostEqual(
            average_precision(torch.tensor([0.9]), torch.tensor([True]), positive_count=2) or 0.0,
            0.5,
        )

    def test_operating_point_selects_highest_threshold_reaching_target(self) -> None:
        half = operating_point_at_recall(
            [0.9, 0.8, 0.7, 0.6], [True, False, True, False],
            positive_count=2, target_recall=0.5, total_minutes=2.0,
        )
        self.assertIsNotNone(half)
        self.assertAlmostEqual(half["threshold"], 0.9, places=6)
        self.assertAlmostEqual(half["precision"], 1.0)
        full = operating_point_at_recall(
            [0.9, 0.8, 0.7, 0.6], [True, False, True, False],
            positive_count=2, target_recall=1.0, total_minutes=2.0,
        )
        self.assertAlmostEqual(full["threshold"], 0.7, places=6)
        self.assertAlmostEqual(full["precision"], 2.0 / 3.0)
        self.assertIsNone(operating_point_at_recall(
            [0.5], [False], positive_count=0, target_recall=0.85
        ))

    def test_family_conditioned_decoder_separates_shot_and_later_save(self) -> None:
        timestamps = torch.arange(50, dtype=torch.float32) / 5.0
        family_logits = torch.linspace(-20.0, -10.0, 50).unsqueeze(-1).repeat(1, 3)
        family_logits[10, 0] = 10.0
        family_logits[30, 1] = 10.0
        class_logits = torch.full((50, 6), -10.0)
        class_logits[9, 0] = 8.0
        class_logits[13, 1] = 8.0
        class_logits[31, 2] = 8.0
        proposals = decode_family_conditioned_classes(
            family_logits, class_logits, timestamps,
            ("shot_chain", "restart", "generic_event"),
            ("shot", "save", "corner", "penalty", "freekick", "kickoff"),
            label_to_family={
                "shot": "shot_chain", "save": "shot_chain", "corner": "restart",
                "penalty": "restart", "freekick": "restart", "kickoff": "restart",
            },
            family_nms_radius_seconds={
                "shot_chain": 3.0, "restart": 6.0, "generic_event": 3.0,
            },
            class_nms_radius_seconds={label: 1.0 for label in (
                "shot", "save", "corner", "penalty", "freekick", "kickoff"
            )},
            local_search_radius_seconds={label: 4.0 for label in (
                "shot", "save", "corner", "penalty", "freekick", "kickoff"
            )},
            max_class_per_minute={label: 100.0 for label in (
                "shot", "save", "corner", "penalty", "freekick", "kickoff"
            )},
        )
        best_shot = max(proposals["shot"], key=lambda item: item.score)
        best_save = max(proposals["save"], key=lambda item: item.score)
        best_corner = max(proposals["corner"], key=lambda item: item.score)
        self.assertAlmostEqual(best_shot.timestamp, float(timestamps[9]), places=5)
        self.assertAlmostEqual(best_save.timestamp, float(timestamps[13]), places=5)
        self.assertAlmostEqual(best_corner.timestamp, float(timestamps[31]), places=5)

    def test_continuous_inference_emits_each_step_once(self) -> None:
        torch.manual_seed(4)
        model = TemporalLocator(4, 3, hidden_dim=8, levels=2, blocks_per_level=1)
        timeline = AlignedTimeline(
            timestamps=torch.arange(23, dtype=torch.float32) / 5.0,
            context=torch.randn(23, 4),
            motion=torch.randn(23, 3),
            context_valid=torch.ones(23, dtype=torch.bool),
            motion_valid=torch.ones(23, dtype=torch.bool),
        )
        result = infer_continuous_timeline(
            model, timeline, device=torch.device("cpu"), core_steps=7, context_steps=3
        )
        self.assertEqual(result.logits.shape, (23, 3))
        self.assertIsNotNone(result.class_logits)
        self.assertEqual(result.class_logits.shape, (23, 6))
        self.assertTrue(torch.equal(result.coverage, torch.ones(23, dtype=torch.int32)))


if __name__ == "__main__":
    unittest.main()
