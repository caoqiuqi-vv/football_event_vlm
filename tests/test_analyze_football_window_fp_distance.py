from __future__ import annotations

import unittest

from scripts.analyze_football_window_fp_distance import (
    gap_bin_name,
    summarize_counts,
    window_gap_sec,
)


class WindowFalsePositiveDistanceTest(unittest.TestCase):
    def test_window_gap_includes_matching_tolerance(self) -> None:
        self.assertEqual(window_gap_sec(10.0, 20.0, 5.0, 5.0), 0.0)
        self.assertEqual(window_gap_sec(10.0, 20.0, 25.0, 5.0), 0.0)
        self.assertAlmostEqual(window_gap_sec(10.0, 20.0, 3.5, 5.0), 1.5)
        self.assertAlmostEqual(window_gap_sec(10.0, 20.0, 27.0, 5.0), 2.0)

    def test_gap_bins_have_stable_boundaries(self) -> None:
        bounds = (5.0, 15.0)
        self.assertEqual(gap_bin_name(5.0, bounds), "gap_0_5s")
        self.assertEqual(gap_bin_name(5.1, bounds), "gap_5_15s")
        self.assertEqual(gap_bin_name(15.1, bounds), "gap_gt_15s")

    def test_summary_reports_fp_fractions(self) -> None:
        summary = summarize_counts(
            {"tp": 3, "fp": 2, "gap_0_5s": 1, "gap_gt_15s": 1}
        )
        self.assertEqual(summary["num_pred"], 5)
        self.assertAlmostEqual(summary["precision"], 0.6)
        self.assertAlmostEqual(summary["gap_0_5s_fraction_of_fp"], 0.5)
        self.assertAlmostEqual(summary["gap_gt_15s_fraction_of_fp"], 0.5)


if __name__ == "__main__":
    unittest.main()
