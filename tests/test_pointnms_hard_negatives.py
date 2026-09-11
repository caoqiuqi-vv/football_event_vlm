from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.build_pointnms_hard_negatives import build_manifest


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class PointNmsHardNegativeManifestTest(unittest.TestCase):
    def test_keeps_only_unmatched_safe_high_score_false_positives(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "eval_run"
            video_dir = run_dir / "v1"
            ids_path = root / "train_ids.txt"
            ids_path.write_text("v1\n")

            write_csv(
                video_dir / "predicted_events.csv",
                [
                    {"label": "shot", "time_sec": 100.0, "score": 0.70},
                    {"label": "shot", "time_sec": 110.0, "score": 0.80},
                    {"label": "save", "time_sec": 200.0, "score": 0.20},
                    {"label": "save", "time_sec": 300.0, "score": 0.50},
                ],
                ["label", "time_sec", "score"],
            )
            write_csv(
                video_dir / "matches.csv",
                [{"label": "shot", "pred_time_sec": 110.0, "pred_score": 0.80}],
                ["label", "pred_time_sec", "pred_score"],
            )
            write_csv(
                video_dir / "gt_events.csv",
                [
                    {"label": "shot", "time_sec": 110.0},
                    {"label": "save", "time_sec": 304.0},
                ],
                ["label", "time_sec"],
            )

            manifest = build_manifest(
                SimpleNamespace(
                    eval_run_dir=str(run_dir),
                    reviewed_video_ids=[str(ids_path)],
                    labels="shot,save",
                    min_score=0.0,
                    max_score=1.0,
                    min_scores="shot=0.25,save=0.30",
                    max_scores="",
                    safety_margin_sec=8.0,
                    safety_label_mode="same_label",
                    dedupe_gap_sec=10.0,
                    max_per_video_per_label=8,
                    source="xbotgo_0608",
                )
            )

            self.assertEqual(manifest["num_hard_negatives"], 1)
            self.assertEqual(manifest["per_label_counts"], {"shot": 1, "save": 0})
            item = manifest["hard_negatives"][0]
            self.assertEqual(item["video_id"], "v1")
            self.assertEqual(item["labels"], ["shot"])
            self.assertAlmostEqual(item["center_sec"], 100.0)
            self.assertAlmostEqual(item["label_scores"]["shot"], 0.70)
            self.assertEqual(
                manifest["skipped"],
                {"matched_predictions": 1, "near_gt": 1, "score_filter": 1},
            )


if __name__ == "__main__":
    unittest.main()
