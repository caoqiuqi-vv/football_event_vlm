from __future__ import annotations

import sys
import unittest
from pathlib import Path


UI_ROOT = Path(__file__).resolve().parents[1] / "tools" / "football_event_review"
sys.path.insert(0, str(UI_ROOT))

import server_hierarchical as hierarchical  # noqa: E402
import server_hierarchical_v26  # noqa: E402,F401


class ReviewNavigationV26Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        source = (UI_ROOT / "static" / "app.js").read_text(encoding="utf-8")
        cls.javascript = hierarchical.patch_hierarchical_js(source)

    def test_last_segment_does_not_dereference_a_missing_group(self) -> None:
        self.assertIn(
            "const targetId = targetGroup ? segmentRepresentative(targetGroup)?.id : null;",
            self.javascript,
        )
        self.assertNotIn("selectEvent(segmentRepresentative(targetGroup).id", self.javascript)

    def test_next_unreviewed_crosses_video_and_wraps_once(self) -> None:
        self.assertIn("async function navigateAcrossVideos", self.javascript)
        self.assertIn("candidateIndexes.push((startIndex + direction * offset + videos.length) % videos.length)", self.javascript)
        self.assertIn("await loadVideo(summary.video_id)", self.javascript)
        self.assertIn('toast("全部片段均已审核完成")', self.javascript)

    def test_adjacent_navigation_crosses_video_boundary(self) -> None:
        self.assertIn("await navigateAcrossVideos(direction, false, true)", self.javascript)

    def test_existing_async_save_handler_remains_async(self) -> None:
        self.assertIn("async function saveDecision", self.javascript)


if __name__ == "__main__":
    unittest.main()
