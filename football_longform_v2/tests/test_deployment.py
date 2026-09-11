from __future__ import annotations

import unittest

from football_longform_v2.decoding import Proposal
from football_longform_v2.deployment import (
    OPERATING_POINTS_SCHEMA,
    apply_frozen_operating_points,
    build_frozen_operating_points,
)


class DeploymentTest(unittest.TestCase):
    def report(self) -> dict:
        return {
            "evaluation_split": "calibration", "strict_complete_split": True,
            "evaluated_video_count": 18,
            "class_proposal_route": "family_conditioned_local_search_v1",
            "class_local_search_radius_seconds": {"shot": 2.0, "penalty": 4.0},
            "labels": {
                "shot": {
                    "point_support": 100, "max_proposals_per_minute": 3.0,
                    "candidate_ceiling_recall_at_2s": 0.96,
                    "operating_point_for_target_recall_at_2s": {
                        "target_achieved": True, "threshold": 0.7,
                        "target_recall": 0.9, "recall": 0.91, "precision": 0.5,
                        "fp_per_minute": 1.2,
                    },
                },
                "penalty": {
                    "point_support": 0, "max_proposals_per_minute": 0.5,
                    "operating_point_for_target_recall_at_2s": None,
                },
            },
        }

    def test_freeze_and_apply_preserve_high_recall_fallback_contract(self) -> None:
        operating = build_frozen_operating_points(
            self.report(), checkpoint="checkpoint.pt",
            checkpoint_sha256="checkpoint-hash", report_sha256="report-hash",
        )
        self.assertEqual(operating["schema"], OPERATING_POINTS_SCHEMA)
        self.assertEqual(operating["labels"]["shot"]["threshold"], 0.7)
        self.assertEqual(
            operating["labels"]["penalty"]["status"], "no_calibration_support"
        )
        proposals = {
            "shot": [
                Proposal(0, "shot", 10.0, 0.8, 50),
                Proposal(0, "shot", 20.0, 0.6, 100),
            ],
            "penalty": [
                Proposal(0, "penalty", 30.0, 0.4, 150),
                Proposal(0, "penalty", 40.0, 0.9, 200),
            ],
        }
        events = apply_frozen_operating_points(
            proposals, operating, duration_minutes=1.0, video_id="video-a"
        )
        self.assertEqual(
            [(item["label"], item["timestamp_seconds"]) for item in events],
            [("shot", 10.0), ("penalty", 40.0)],
        )
        penalty = next(item for item in events if item["label"] == "penalty")
        self.assertEqual(penalty["review_priority"], "high")
        self.assertEqual(penalty["selection"], "uncalibrated_budget_fallback")

    def test_freeze_rejects_partial_calibration(self) -> None:
        report = self.report()
        report["evaluated_video_count"] = 17
        with self.assertRaises(ValueError):
            build_frozen_operating_points(
                report, checkpoint="checkpoint.pt",
                checkpoint_sha256="x", report_sha256="y",
            )


if __name__ == "__main__":
    unittest.main()
