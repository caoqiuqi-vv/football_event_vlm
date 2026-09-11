from __future__ import annotations

import json
import secrets
import sys
import tempfile
import unittest
from pathlib import Path


TOOL_DIR = Path(__file__).resolve().parents[1] / "tools" / "football_event_review"
sys.path.insert(0, str(TOOL_DIR))

import server_multiuser_v30 as multiuser  # noqa: E402
import server_multiuser_v36 as v36  # noqa: E402


class TeamCalibrationStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.video_path = self.root / "video.mp4"
        self.video_path.write_bytes(b"not-a-real-video")
        self.manifest_path = self.root / "manifest.json"
        self.manifest_path.write_text(
            json.dumps(
                {
                    "labels": ["shot", "save", "freekick", "corner", "kickoff", "throw_in"],
                    "videos": [
                        {
                            "video_id": "video-1",
                            "video_path": str(self.video_path),
                            "duration_sec": 120.0,
                            "team_palette": {
                                "teamA": {"hex": "#cc2233", "display_name": "红队"},
                                "teamB": {"hex": "#2255cc", "display_name": "蓝队"},
                            },
                            "team_calibration": {"sample_start_sec": 25.0, "sample_end_sec": 55.0},
                            "events": [
                                {
                                    "id": "event-1",
                                    "label": "shot",
                                    "time_sec": 42.0,
                                    "start_sec": 37.0,
                                    "end_sec": 47.0,
                                    "score": 0.9,
                                    "source": "gt+ai",
                                }
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        salt = secrets.token_hex(16)
        password = "test-password"
        self.access_path = self.root / "access.json"
        self.access_path.write_text(
            json.dumps(
                {
                    "users": [
                        {
                            "user_id": "reviewer-1",
                            "slug": "reviewer-1",
                            "display_name": "审核员一",
                            "password_salt": salt,
                            "password_hash": multiuser.MultiUserReviewStore.password_digest(password, salt),
                            "video_ids": ["video-1"],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        self.store = v36.TeamCalibrationStore(
            self.manifest_path,
            self.root / "reviews.sqlite3",
            self.access_path,
            self.root / "proxy",
        )
        self.user = self.store.users_by_id["reviewer-1"]

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_profile_is_versioned_and_human_confirmed_palette_is_authoritative(self) -> None:
        initial = self.store.team_profile("video-1")
        self.assertEqual(initial["teams"]["teamA"]["hex"], "#cc2233")
        self.assertEqual(initial["status"], "unconfirmed")
        updated = self.store.update_team_profile_for_user(
            self.user,
            "video-1",
            {
                "expected_revision": 0,
                "status": "confirmed",
                "teams": {
                    "teamA": {"hex": "#ee3322", "display_name": "红衣队"},
                    "teamB": {"hex": "#eeeeee", "display_name": "白衣队"},
                },
                "period_split_sec": 61.5,
                "first_period_left_team": "teamA",
                "second_period_left_team": "teamB",
            },
        )
        self.assertEqual(updated["revision"], 1)
        self.assertEqual(updated["teams"]["teamB"]["hex"], "#eeeeee")
        payload = self.store.video_payload_for_user("reviewer-1", "video-1")
        self.assertEqual(payload["team_palette"], updated["teams"])
        with self.store.connect() as connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM video_team_profile_history").fetchone()[0],
                1,
            )
        with self.assertRaises(multiuser.RevisionConflict):
            self.store.update_team_profile_for_user(
                self.user,
                "video-1",
                {
                    "expected_revision": 0,
                    "teams": updated["teams"],
                    "first_period_left_team": "teamA",
                    "second_period_left_team": "teamB",
                },
            )

    def test_event_team_and_field_side_are_exported(self) -> None:
        updated = self.store.update_review(
            "event-1",
            {
                "status": "accepted",
                "event_team": "teamA",
                "goal_side": "right",
                "field_side": "right",
                "reviewer": "审核员一",
            },
        )
        self.assertEqual(updated["review"]["field_side"], "right")
        exported = self.store.export_rows()
        self.assertEqual(exported[0]["event_team"], "teamA")
        self.assertEqual(exported[0]["goal_side"], "right")
        self.assertEqual(exported[0]["field_side"], "right")

    def test_generated_ui_contains_team_calibration_and_event_side_controls(self) -> None:
        html = v36.patch_html_v36((v36.base.STATIC_ROOT / "index.html").read_text(encoding="utf-8"))
        javascript = v36.patch_js_v36((v36.base.STATIC_ROOT / "app.js").read_text(encoding="utf-8"))
        self.assertIn('id="teamSetupPanel"', html)
        self.assertIn("/team-profile", javascript)
        self.assertIn("data-field-side", javascript)
        self.assertIn("上衣主色", html)


if __name__ == "__main__":
    unittest.main()
