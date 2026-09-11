from __future__ import annotations

import unittest

from scripts.fit_football_candidate_reranker import Candidate, probability_logit
from scripts.run_football_verifier_protocol_v1 import make_ui_data, select_point_thresholds
from scripts.select_validation_thresholds_cached import evaluate_selected


def candidate(video: str, label: str, index: int, start: float, peak: float, prob: float) -> Candidate:
    value = probability_logit(prob)
    return Candidate(
        video_id=video,
        label=label,
        window_index=index,
        window_start_sec=start,
        window_end_sec=start + 10.0,
        center_time_sec=start + 5.0,
        peak_time_sec=peak,
        clip_prob=prob,
        features=(value, value, value, 0.0, 0.0, value, 1.0),
        target=1,
        nearest_gt_distance_sec=0.0,
    )


class VerifierProtocolV1Test(unittest.TestCase):
    def test_ui_workload_deduplicates_overlapping_labels(self) -> None:
        rows = [
            candidate("v", "shot", 0, 0.0, 5.0, 0.9),
            candidate("v", "save", 0, 0.0, 5.0, 0.9),
        ]
        scores = {item.key: item.clip_prob for item in rows}
        gt = {"v": {"shot": [5.0], "save": [5.0]}}
        data = make_ui_data(rows, scores, gt, ("shot", "save"))
        result = evaluate_selected(data, ("shot", "save"), {"shot": 0.5, "save": 0.5}, 0.0, 30.0, 10.0)
        self.assertEqual(result["workload"]["num_review_segments"], 1)
        self.assertAlmostEqual(result["workload"]["full_segment_minutes"], 10.0 / 60.0)
        self.assertEqual(result["full_segment_human_visible"]["visible_gt"], 2)

    def test_unreachable_recall_floor_is_explicit(self) -> None:
        rows = [candidate("v", "shot", 0, 0.0, 5.0, 0.9)]
        scores = {rows[0].key: 0.9}
        thresholds, audit = select_point_thresholds(
            rows,
            scores,
            {"v": {"shot": [5.0, 30.0]}},
            ("shot",),
            0.9,
            nms_radius_sec=5.0,
            match_tolerance_sec=5.0,
            grid_size=5,
        )
        self.assertIn("shot", thresholds)
        self.assertFalse(audit["shot"]["requested_floor_reachable"])
        self.assertAlmostEqual(audit["shot"]["reachable_recall_ceiling"], 0.5)


if __name__ == "__main__":
    unittest.main()
