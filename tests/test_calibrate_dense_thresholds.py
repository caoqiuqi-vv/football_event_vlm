from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_dense_pr_ceiling import VideoLabelScores  # noqa: E402
from calibrate_dense_thresholds import best_row, exact_curve  # noqa: E402
from cross_validate_dense_threshold import requested_video_ids  # noqa: E402


class DenseThresholdCalibrationTest(unittest.TestCase):
    def test_requested_video_ids_reads_file_and_comments(self) -> None:
        path = self.id() + ".txt"
        temp_path = Path("/tmp") / path
        temp_path.write_text("v1\n# ignored\n\nv2\n", encoding="utf-8")
        try:
            self.assertEqual(requested_video_ids("", str(temp_path)), ["v1", "v2"])
            with self.assertRaises(ValueError):
                requested_video_ids("v3", str(temp_path))
        finally:
            temp_path.unlink(missing_ok=True)

    def test_exact_curve_selects_best_precision_at_recall_floor(self) -> None:
        data = [
            VideoLabelScores(
                video_id="v1",
                label="shot",
                scores=(0.9, 0.8, 0.7, 0.2),
                matched_gt_indices=((0,), (), (1,), ()),
                num_gt=2,
            )
        ]
        selected = best_row(
            exact_curve(data), recall_floor=1.0, objective="precision"
        )
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertAlmostEqual(float(selected["threshold"]), 0.7)
        self.assertAlmostEqual(float(selected["precision"]), 2.0 / 3.0)
        self.assertAlmostEqual(float(selected["recall"]), 1.0)


if __name__ == "__main__":
    unittest.main()
