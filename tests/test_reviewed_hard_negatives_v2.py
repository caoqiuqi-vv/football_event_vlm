from __future__ import annotations

import csv
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from scripts.build_reviewed_hard_negatives_v2 import (
    build_manifest,
    dedupe_candidates,
    parse_label_scores,
)


class ReviewedHardNegativesV2Test(unittest.TestCase):
    def test_parse_label_scores_uses_default_and_overrides(self) -> None:
        self.assertEqual(
            parse_label_scores("shot=0.4", ["shot", "save"], default=0.3),
            {"shot": 0.4, "save": 0.3},
        )
        with self.assertRaises(ValueError):
            parse_label_scores("bad=0.4", ["shot"], default=0.3)

    def test_dedupe_prefers_medium_target_score_not_highest_extreme(self) -> None:
        candidates = [
            {"score": 0.74, "target_score": 0.55, "center_sec": 10.0},
            {"score": 0.56, "target_score": 0.55, "center_sec": 30.0},
            {"score": 0.45, "target_score": 0.55, "center_sec": 50.0},
        ]
        selected = dedupe_candidates(candidates, gap_sec=0.0, limit=2)
        self.assertEqual([row["score"] for row in selected], [0.56, 0.45])

    def test_manifest_filters_reviewed_ids_score_band_and_gt_safety(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "eval"
            reviewed_video = run_dir / "video_a"
            unreviewed_video = run_dir / "video_b"
            reviewed_video.mkdir(parents=True)
            unreviewed_video.mkdir()

            for video_dir in (reviewed_video, unreviewed_video):
                with (video_dir / "window_predictions.csv").open("w", newline="") as file:
                    writer = csv.DictWriter(
                        file,
                        fieldnames=[
                            "index",
                            "start_sec",
                            "end_sec",
                            "global_prob_shot",
                            "global_prob_save",
                        ],
                    )
                    writer.writeheader()
                    writer.writerows(
                        [
                            {
                                "index": 0,
                                "start_sec": 0,
                                "end_sec": 10,
                                "global_prob_shot": 0.56,
                                "global_prob_save": 0.1,
                            },
                            {
                                "index": 1,
                                "start_sec": 20,
                                "end_sec": 30,
                                "global_prob_shot": 0.95,
                                "global_prob_save": 0.1,
                            },
                            {
                                "index": 2,
                                "start_sec": 40,
                                "end_sec": 50,
                                "global_prob_shot": 0.58,
                                "global_prob_save": 0.1,
                            },
                        ]
                    )
                with (video_dir / "gt_events.csv").open("w", newline="") as file:
                    writer = csv.DictWriter(file, fieldnames=["label", "time_sec"])
                    writer.writeheader()
                    writer.writerow({"label": "shot", "time_sec": 45})

            reviewed_ids = root / "reviewed.txt"
            reviewed_ids.write_text("video_a\n")
            args = Namespace(
                eval_run_dir=str(run_dir),
                output=str(root / "manifest.json"),
                reviewed_video_ids=[str(reviewed_ids)],
                branch="global",
                labels="shot",
                min_score=0.35,
                max_score=0.75,
                target_score=0.55,
                min_scores="",
                max_scores="",
                target_scores="",
                safety_margin_sec=2.0,
                dedupe_gap_sec=0.0,
                max_per_video_per_label=10,
                source="test",
            )

            manifest = build_manifest(args)
            self.assertEqual(manifest["num_hard_negatives"], 1)
            item = manifest["hard_negatives"][0]
            self.assertEqual(item["video_id"], "video_a")
            self.assertEqual(item["window_index"], 0)
            self.assertEqual(manifest["num_videos_skipped_not_reviewed"], 1)


if __name__ == "__main__":
    unittest.main()
