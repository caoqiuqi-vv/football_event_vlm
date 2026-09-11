from __future__ import annotations

import unittest

from scripts.analyze_dense_pr_ceiling import (
    VideoLabelScores,
    evaluate_data,
    exact_curve,
    parse_exclusions,
    search_per_class_thresholds,
)


class DensePrCeilingTest(unittest.TestCase):
    def test_window_predictions_share_unique_gt_recall(self) -> None:
        data = VideoLabelScores(
            video_id="v1",
            label="shot",
            scores=(0.9, 0.8, 0.7),
            matched_gt_indices=((0,), (0,), ()),
            num_gt=1,
        )
        metrics = evaluate_data(data, 0.8)
        self.assertEqual((metrics["tp"], metrics["fp"]), (2, 0))
        self.assertEqual(metrics["num_matched_gt"], 1)
        self.assertEqual(metrics["recall"], 1.0)
        curve = exact_curve([data])
        self.assertEqual(curve[0]["threshold"], 0.9)
        self.assertEqual(curve[-1]["fp"], 1)

    def test_search_finds_precision_constrained_thresholds(self) -> None:
        thresholds = [0.5, 0.8]
        grid = {
            "shot": {
                0.5: {"tp": 2, "fp": 2, "num_gt": 1, "num_matched_gt": 1},
                0.8: {"tp": 1, "fp": 0, "num_gt": 1, "num_matched_gt": 1},
            },
            "save": {
                0.5: {"tp": 1, "fp": 2, "num_gt": 1, "num_matched_gt": 1},
                0.8: {"tp": 1, "fp": 0, "num_gt": 1, "num_matched_gt": 1},
            },
        }
        result = search_per_class_thresholds(
            grid,
            ["shot", "save"],
            thresholds,
            precision_floor=0.6,
            recall_target=1.0,
        )
        selected = result["max_recall_at_precision_floor"]
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected["thresholds"], {"shot": 0.8, "save": 0.8})
        self.assertEqual(selected["precision"], 1.0)
        self.assertEqual(selected["recall"], 1.0)

    def test_exclusion_parser_uses_video_label_pair(self) -> None:
        self.assertEqual(
            parse_exclusions(["2027572406738604033:set_piece"]),
            {("2027572406738604033", "set_piece")},
        )


if __name__ == "__main__":
    unittest.main()
