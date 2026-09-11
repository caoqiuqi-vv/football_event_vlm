from __future__ import annotations

import unittest

from scripts.audit_football_robust_roi import bbox_iou, select_audit_rows


class FootballROIAuditTest(unittest.TestCase):
    def test_bbox_iou(self) -> None:
        self.assertEqual(bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(bbox_iou([0, 0, 5, 5], [6, 6, 10, 10]), 0.0)
        self.assertAlmostEqual(bbox_iou([0, 0, 10, 10], [5, 0, 15, 10]), 1.0 / 3.0)

    def test_selection_covers_mode_confidence_and_sample_type(self) -> None:
        rows = [
            {
                "row_id": "a",
                "video_id": "v",
                "start_sec": 0.0,
                "mode": "goal_ball",
                "valid": True,
                "roi_confidence": 0.9,
                "is_background": False,
            },
            {
                "row_id": "b",
                "video_id": "v",
                "start_sec": 5.0,
                "mode": "ball_players",
                "valid": True,
                "roi_confidence": 0.5,
                "is_background": True,
            },
            {
                "row_id": "c",
                "video_id": "v",
                "start_sec": 10.0,
                "mode": "invalid",
                "valid": False,
                "roi_confidence": 0.0,
                "is_background": True,
            },
        ]
        selected = select_audit_rows(rows, per_bucket=1)
        self.assertEqual({row["row_id"] for row in selected}, {"a", "b", "c"})


if __name__ == "__main__":
    unittest.main()
