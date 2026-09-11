from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


UI_ROOT = Path(__file__).resolve().parents[1] / "tools" / "football_event_review"
sys.path.insert(0, str(UI_ROOT))

from server_hierarchical import HierarchicalReviewStore, patch_hierarchical_html, patch_hierarchical_js  # noqa: E402


class HierarchicalReviewTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.video = self.root / "v1.mp4"
        self.video.write_bytes(b"video")
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps({
            "labels": ["shot", "save", "set_piece"],
            "videos": [{
                "video_id": "v1", "video_path": str(self.video), "duration_sec": 20,
                "timeline": [],
                "events": [
                    {"id": "shot1", "video_id": "v1", "label": "shot", "time_sec": 5, "score": .8},
                    {"id": "sp1", "video_id": "v1", "label": "set_piece", "time_sec": 10, "score": .7},
                ],
            }],
        }))
        self.store = HierarchicalReviewStore(self.manifest, self.root / "reviews.sqlite3")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_shot_details_persist_export_and_undo(self) -> None:
        event = self.store.update_review("shot1", {
            "status": "accepted", "corrected_label": "shot",
            "secondary_labels": ["shot_on_target", "goal"], "corrected_time_sec": 5,
        })
        self.assertEqual(event["review"]["secondary_labels"], ["shot_on_target", "goal"])
        row = self.store.export_rows()[0]
        self.assertEqual(json.loads(row["secondary_labels"]), ["shot_on_target", "goal"])
        self.assertEqual(self.store.undo("shot1")["review"]["status"], "unreviewed")

    def test_set_piece_type_is_required_and_single(self) -> None:
        with self.assertRaisesRegex(ValueError, "必须选择"):
            self.store.update_review("sp1", {"status": "accepted", "corrected_label": "set_piece"})
        event = self.store.update_review("sp1", {
            "status": "accepted", "corrected_label": "set_piece",
            "secondary_labels": ["free_kick"],
        })
        self.assertEqual(event["review"]["secondary_labels"], ["free_kick"])

    def test_static_runtime_patches_apply(self) -> None:
        app = (UI_ROOT / "static" / "app.js").read_text()
        html = (UI_ROOT / "static" / "index.html").read_text()
        self.assertIn("shotDetailButtons", patch_hierarchical_js(app))
        self.assertIn("setPieceTypePanel", patch_hierarchical_html(html))


if __name__ == "__main__":
    unittest.main()
