from __future__ import annotations

import unittest

from scripts.compare_football_checkpoint_metrics import compare_candidate


def checkpoint_metrics(shot_p: float, shot_r: float, save_p: float, save_r: float):
    return {
        "path": "checkpoint.pt",
        "epoch": 1,
        "metrics": {
            "mAP": 0.7,
            "per_class": {
                "shot": {"precision": shot_p, "recall": shot_r, "ap": 0.8},
                "save": {"precision": save_p, "recall": save_r, "ap": 0.6},
            },
        },
    }


class CompareFootballCheckpointMetricsTest(unittest.TestCase):
    def test_recall_guard_accepts_precision_gain_with_small_recall_drop(self) -> None:
        baseline = checkpoint_metrics(0.60, 0.85, 0.45, 0.81)
        candidate = checkpoint_metrics(0.64, 0.845, 0.48, 0.805)

        result = compare_candidate(baseline, candidate, ["shot", "save"], 0.01)

        self.assertTrue(result["recall_guard_pass"])
        self.assertTrue(result["precision_improved"])
        self.assertAlmostEqual(result["precision_delta_mean"], 0.035)

    def test_recall_guard_rejects_large_drop_even_when_precision_improves(self) -> None:
        baseline = checkpoint_metrics(0.60, 0.85, 0.45, 0.81)
        candidate = checkpoint_metrics(0.70, 0.82, 0.55, 0.80)

        result = compare_candidate(baseline, candidate, ["shot", "save"], 0.01)

        self.assertFalse(result["recall_guard_pass"])
        self.assertFalse(result["per_class"]["shot"]["recall_guard_pass"])


if __name__ == "__main__":
    unittest.main()
