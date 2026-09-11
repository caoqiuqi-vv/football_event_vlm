from __future__ import annotations

import unittest

from scripts.compare_football_eval_protocols import compare_runs


def run(shot_p: float, shot_r: float, save_p: float, save_r: float):
    def metrics(p: float, r: float):
        return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r)}

    per_class = {"shot": metrics(shot_p, shot_r), "save": metrics(save_p, save_r)}
    return {
        "labels": ["shot", "save"],
        "thresholds": {"shot": 0.2, "save": 0.3},
        "protocols": {
            "point_nms": {"per_class": per_class},
            "window_overlap": {"per_class": per_class},
        },
    }


class CompareFootballEvalProtocolsTest(unittest.TestCase):
    def test_accepts_precision_gain_with_guarded_recall(self) -> None:
        baseline = run(0.30, 0.80, 0.20, 0.75)
        candidate = run(0.34, 0.795, 0.23, 0.745)

        result = compare_runs(
            baseline,
            [("candidate.json", candidate)],
            ["point_nms"],
            ["shot", "save"],
            0.01,
        )

        point = result["candidates"][0]["protocols"]["point_nms"]
        self.assertTrue(point["recall_guard_pass"])
        self.assertTrue(point["precision_improved"])
        self.assertAlmostEqual(point["precision_delta_mean"], 0.035)

    def test_allows_each_model_to_use_its_tuned_thresholds(self) -> None:
        baseline = run(0.30, 0.80, 0.20, 0.75)
        candidate = run(0.34, 0.795, 0.23, 0.745)
        candidate["thresholds"]["shot"] = 0.25

        result = compare_runs(
            baseline,
            [("candidate.json", candidate)],
            ["point_nms"],
            ["shot", "save"],
            0.01,
        )

        self.assertFalse(result["candidates"][0]["same_thresholds_as_baseline"])
        self.assertEqual(result["candidates"][0]["thresholds"]["shot"], 0.25)

    def test_rejects_any_class_recall_drop_over_tolerance(self) -> None:
        baseline = run(0.30, 0.80, 0.20, 0.75)
        candidate = run(0.40, 0.78, 0.25, 0.75)

        result = compare_runs(
            baseline,
            [("candidate.json", candidate)],
            ["point_nms"],
            ["shot", "save"],
            0.01,
        )

        self.assertFalse(result["candidates"][0]["protocols"]["point_nms"]["recall_guard_pass"])


if __name__ == "__main__":
    unittest.main()
