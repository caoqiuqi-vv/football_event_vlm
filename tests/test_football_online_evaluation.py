from __future__ import annotations

import unittest

import numpy as np

from football_online_evaluation import tune_online_event_thresholds


class OnlineEventThresholdTest(unittest.TestCase):
    def test_per_class_floor_and_f1_policies(self) -> None:
        labels = ["shot", "save", "set_piece"]
        times = np.asarray([[0, 0, 0], [5, 5, 5], [10, 10, 10], [15, 15, 15]])
        probs = np.asarray(
            [[.1, .2, .1], [.9, .2, .1], [.2, .8, .1], [.15, .1, .1]],
            dtype=np.float32,
        )
        metas = [
            {
                "source": "s",
                "video_id": "v",
                "online_gt_anchors": ((5.0,), (10.0,), ()),
            }
            for _ in range(4)
        ]
        thresholds, diagnostics = tune_online_event_thresholds(
            probs,
            times,
            metas,
            labels,
            ["precision", "f1", "f1"],
            [.85, None, None],
            nms_radius_sec=1.0,
            tolerance_sec=1.0,
        )
        self.assertAlmostEqual(float(thresholds[0]), .9, places=5)
        self.assertAlmostEqual(float(thresholds[1]), .8, places=5)
        self.assertEqual(float(thresholds[2]), 1.0)
        self.assertEqual(diagnostics["shot"]["recall"], 1.0)
        self.assertEqual(diagnostics["save"]["precision"], 1.0)
        self.assertEqual(diagnostics["set_piece"]["reason"], "no_ground_truth")


    def test_partial_video_class_is_excluded_from_threshold_tuning(self) -> None:
        labels = ["shot", "save", "set_piece"]
        probs = np.asarray([[.9, .1, .1], [.8, .1, .1], [.95, .1, .1]], dtype=np.float32)
        times = np.asarray([[5, 5, 5], [15, 15, 15], [100, 100, 100]], dtype=np.float32)
        metas = [
            {"source": "s", "video_id": "complete", "online_gt_anchors": ((5.0,), (), ())},
            {"source": "s", "video_id": "complete", "online_gt_anchors": ((5.0,), (), ())},
            {"source": "s", "video_id": "partial", "online_gt_anchors": ((), (), ())},
        ]
        masks = np.ones_like(probs)
        masks[2, 0] = 0.0
        thresholds, diagnostics = tune_online_event_thresholds(
            probs, times, metas, labels, ["f1"] * 3, [None] * 3, masks=masks,
            nms_radius_sec=1.0, tolerance_sec=1.0,
        )
        self.assertAlmostEqual(float(thresholds[0]), .9, places=5)
        self.assertEqual(diagnostics["shot"]["partial_video_count"], 1)
        self.assertEqual(diagnostics["shot"]["partial_video_ids"], ["partial"])


if __name__ == "__main__":
    unittest.main()
