from __future__ import annotations

import unittest

from scripts.build_precision60_hard_negatives import (
    dedupe_candidates,
    merge_label_candidates,
    overlaps_any_gt,
    parse_label_values,
)


class Precision60HardNegativeTest(unittest.TestCase):
    def test_rejects_window_near_any_gt(self) -> None:
        self.assertTrue(
            overlaps_any_gt(
                [20.0],
                start_sec=5.0,
                end_sec=15.0,
                safety_margin_sec=5.0,
            )
        )
        self.assertFalse(
            overlaps_any_gt(
                [20.1],
                start_sec=5.0,
                end_sec=15.0,
                safety_margin_sec=5.0,
            )
        )

    def test_deduplication_prefers_higher_score(self) -> None:
        rows = [
            {"center_sec": 10.0, "score": 0.7},
            {"center_sec": 12.0, "score": 0.9},
            {"center_sec": 30.0, "score": 0.8},
        ]
        selected = dedupe_candidates(rows, gap_sec=5.0, limit=2)
        self.assertEqual([row["score"] for row in selected], [0.9, 0.8])

    def test_label_values_require_every_label(self) -> None:
        self.assertEqual(
            parse_label_values("shot=0.5,save=0.6", ["shot", "save"]),
            {"shot": 0.5, "save": 0.6},
        )
        with self.assertRaises(ValueError):
            parse_label_values("shot=0.5", ["shot", "save"])


    def test_merge_label_candidates_avoids_duplicate_decode(self) -> None:
        rows = [
            {
                "video_id": "v1",
                "label": "shot",
                "score": 0.8,
                "start_sec": 10.0,
                "end_sec": 20.0,
                "center_sec": 15.0,
            },
            {
                "video_id": "v1",
                "label": "save",
                "score": 0.7,
                "start_sec": 10.0,
                "end_sec": 20.0,
                "center_sec": 15.0,
            },
        ]
        merged = merge_label_candidates(rows)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["labels"], ["save", "shot"])
        self.assertEqual(merged[0]["score"], 0.8)
        self.assertEqual(merged[0]["label_scores"], {"shot": 0.8, "save": 0.7})


if __name__ == "__main__":
    unittest.main()
