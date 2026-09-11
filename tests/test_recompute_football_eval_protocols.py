from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.recompute_football_eval_protocols import load_run_metadata, recompute_protocols


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class RecomputeFootballEvalProtocolsTest(unittest.TestCase):
    def test_reuses_windows_for_both_matching_protocols(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            video_dir = run_dir / "video1"
            video_dir.mkdir()
            summary = {
                "labels": ["shot", "save", "set_piece"],
                "thresholds": {"shot": 0.5, "save": 0.5, "set_piece": 0.5},
            }
            (video_dir / "summary.json").write_text(json.dumps(summary))
            write_csv(
                video_dir / "window_predictions.csv",
                [
                    {"index": 0, "start_sec": 0, "end_sec": 10, "prob_shot": 0.9, "prob_save": 0.1, "prob_set_piece": 0.1},
                    {"index": 1, "start_sec": 5, "end_sec": 15, "prob_shot": 0.8, "prob_save": 0.1, "prob_set_piece": 0.1},
                ],
            )
            write_csv(
                video_dir / "gt_events.csv",
                [{"label": "shot", "time_sec": 5, "start_sec": 5, "end_sec": 5, "event_id": "shot-1"}],
            )

            labels, thresholds = load_run_metadata(run_dir, ["video1"])
            output, rows = recompute_protocols(
                run_dir,
                ["video1"],
                labels,
                thresholds,
                score_prefix="prob",
                nms_radius_sec=5.0,
                match_tolerance_sec=5.0,
            )

            point = output["protocols"]["point_nms"]["per_class"]["shot"]
            overlap = output["protocols"]["window_overlap"]["per_class"]["shot"]
            self.assertEqual((point["tp"], point["fp"], point["fn"]), (1, 0, 0))
            self.assertEqual((overlap["tp"], overlap["fp"], overlap["fn"]), (2, 0, 0))
            self.assertEqual(overlap["recall"], 1.0)
            self.assertEqual(len(rows), 6)

    def test_rejects_mixed_thresholds_across_videos(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            for video_id, shot_threshold in (("a", 0.4), ("b", 0.5)):
                video_dir = run_dir / video_id
                video_dir.mkdir()
                (video_dir / "window_predictions.csv").write_text("index,start_sec,end_sec,prob_shot\n")
                (video_dir / "summary.json").write_text(
                    json.dumps({"labels": ["shot"], "thresholds": {"shot": shot_threshold}})
                )

            with self.assertRaisesRegex(ValueError, "Threshold mismatch"):
                load_run_metadata(run_dir, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
