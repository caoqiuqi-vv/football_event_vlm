from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from football_legacy_indexed_cropper import LegacyIndexedWindowCropper


class FootballLegacyIndexedCropperTest(unittest.TestCase):
    def test_max_goal_all_ball_baseline_uses_v2_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = {
                "version": 2,
                "fps": 30.0,
                "image_size": {"width": 1000, "height": 600},
                "frame_ids": torch.tensor([0, 15, 30], dtype=torch.int32),
                "frame_offsets": torch.tensor([0, 3, 6, 9], dtype=torch.int64),
                "classes": torch.tensor([2, 1, 0] * 3, dtype=torch.int8),
                "confidences": torch.tensor([0.9, 0.8, 0.9] * 3, dtype=torch.float16),
                "boxes": torch.tensor(
                    [
                        [700, 150, 900, 350], [600, 250, 620, 270], [550, 200, 650, 500],
                        [705, 150, 905, 350], [610, 250, 630, 270], [560, 200, 660, 500],
                        [710, 150, 910, 350], [620, 250, 640, 270], [570, 200, 670, 500],
                    ],
                    dtype=torch.float32,
                ),
            }
            torch.save(payload, Path(directory) / "video.pt")
            cropper = LegacyIndexedWindowCropper(directory, target_aspect=5 / 3, roi_samples=3)
            cropper._geometry_cache["dummy.mp4"] = (1000, 600)

            roi, stats = cropper.get_window_roi("dummy.mp4", "video", 0.0, 2.0)

            self.assertIsNotNone(roi)
            self.assertEqual(stats["roi_proposal_mode"], "legacy_goal")
            self.assertEqual(stats["crop_goal_count"], 3)
            self.assertLessEqual(stats["crop_area_ratio"], 0.85)


if __name__ == "__main__":
    unittest.main()
