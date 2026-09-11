from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from tools.football_event_review.build_review_manifest import build_video_entry
from tools.football_event_review.server import ReviewStore


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class FootballEventReviewTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.eval_dir = self.root / "v1"
        self.eval_dir.mkdir()
        self.video_path = self.root / "v1.mp4"
        self.video_path.write_bytes(b"video")
        write_csv(
            self.eval_dir / "window_predictions.csv",
            ["index", "start_sec", "end_sec", "prob_shot", "prob_save", "prob_set_piece", "roi_valid", "roi_confidence", "roi_proposal_mode"],
            [
                {"index": 0, "start_sec": 0, "end_sec": 10, "prob_shot": .7, "prob_save": .2, "prob_set_piece": .1, "roi_valid": 1, "roi_confidence": .8, "roi_proposal_mode": "goal_ball"},
                {"index": 1, "start_sec": 5, "end_sec": 15, "prob_shot": .8, "prob_save": .3, "prob_set_piece": .1, "roi_valid": 1, "roi_confidence": .7, "roi_proposal_mode": "goal_players"},
            ],
        )
        write_csv(
            self.eval_dir / "predicted_events.csv",
            ["label", "time_sec", "start_sec", "end_sec", "support_start_sec", "support_end_sec", "score", "window_indices"],
            [
                {"label": "shot", "time_sec": 5, "start_sec": 0, "end_sec": 10, "support_start_sec": 0, "support_end_sec": 10, "score": .7, "window_indices": "[0]"},
                {"label": "shot", "time_sec": 10, "start_sec": 5, "end_sec": 15, "support_start_sec": 5, "support_end_sec": 15, "score": .8, "window_indices": "[1]"},
            ],
        )
        write_csv(
            self.eval_dir / "frame_event_window_scores.csv",
            ["window_index", "start_sec", "end_sec", "branch", "label", "target", "max_frame_prob"],
            [
                {"window_index": 0, "start_sec": 0, "end_sec": 10, "branch": "global", "label": "shot", "target": 0, "max_frame_prob": .4},
                {"window_index": 1, "start_sec": 5, "end_sec": 15, "branch": "global", "label": "shot", "target": 0, "max_frame_prob": .6},
            ],
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_adapter_merges_and_joins_frame_evidence(self) -> None:
        entry = build_video_entry("v1", self.eval_dir, self.video_path, ("shot", "save", "set_piece"), 5.0)
        self.assertEqual(len(entry["events"]), 1)
        event = entry["events"][0]
        self.assertEqual(event["time_sec"], 10)
        self.assertEqual(event["merged_predictions"], 2)
        self.assertEqual(event["window_indices"], [0, 1])
        self.assertAlmostEqual(event["frame_detection_scores"]["shot"], .6)

    def test_review_persists_undoes_and_exports(self) -> None:
        entry = build_video_entry("v1", self.eval_dir, self.video_path, ("shot", "save", "set_piece"), 5.0)
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(json.dumps({"labels": ["shot", "save", "set_piece"], "videos": [entry]}))
        store = ReviewStore(manifest_path, self.root / "reviews.sqlite3")
        event_id = entry["events"][0]["id"]
        updated = store.update_review(event_id, {"status": "modified", "corrected_label": "save", "corrected_time_sec": 10.25, "reviewer": "qa"})
        self.assertEqual(updated["review"]["status"], "modified")
        exported = store.export_rows()
        self.assertEqual(exported[0]["label"], "save")
        self.assertEqual(exported[0]["time_sec"], 10.25)
        undone = store.undo(event_id)
        self.assertEqual(undone["review"]["status"], "unreviewed")
        self.assertEqual(store.export_rows(), [])


if __name__ == "__main__":
    unittest.main()
