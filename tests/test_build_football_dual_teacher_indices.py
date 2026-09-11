from __future__ import annotations

import unittest

from scripts.build_football_dual_teacher_indices import xyxy


class DualTeacherIndexBoxParsingTest(unittest.TestCase):
    def test_soccermind_center_xywh_is_converted_to_xyxy(self) -> None:
        row = {"xywh": [100.0, 50.0, 8.0, 6.0]}
        self.assertEqual(
            xyxy(row, xywh_default=True, center_xywh=True),
            [96.0, 47.0, 104.0, 53.0],
        )

    def test_goal_xyxy_is_not_reinterpreted(self) -> None:
        row = {"bbox_xyxy": [10.0, 20.0, 30.0, 40.0]}
        self.assertEqual(
            xyxy(row, xywh_default=False, center_xywh=False),
            [10.0, 20.0, 30.0, 40.0],
        )


if __name__ == "__main__":
    unittest.main()
