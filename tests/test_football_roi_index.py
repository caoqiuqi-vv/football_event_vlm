from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from football_detection_aware import RobustClipCropper
from scripts.build_football_roi_indices import build_index


class FootballROIIndexTest(unittest.TestCase):
    def test_rejects_non_global_frame_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            video_dir = Path(directory) / "video"
            video_dir.mkdir()
            (video_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "fps": 30.0,
                        "image_size": {"width": 1920, "height": 1080},
                        "sampling": {"frame_id_semantics": "local_chunk_frame_id"},
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "original_source_frame_id"):
                build_index(
                    video_dir,
                    Path(directory) / "video.pt",
                    sample_fps=2.0,
                    ball_track_fps=10.0,
                    person_conf_floor=0.2,
                    ball_conf_floor=0.1,
                    goal_conf_floor=0.2,
                    center_circle_conf_floor=0.2,
                )

    def test_empty_detection_index_returns_invalid_roi(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = {
                "version": 2,
                "video_id": "video",
                "fps": 30.0,
                "image_size": {"width": 1920, "height": 1080},
                "frame_ids": torch.empty(0, dtype=torch.int32),
                "frame_offsets": torch.tensor([0], dtype=torch.int64),
                "classes": torch.empty(0, dtype=torch.int8),
                "confidences": torch.empty(0, dtype=torch.float16),
                "boxes": torch.empty((0, 4), dtype=torch.float32),
                "track_ids": torch.empty(0, dtype=torch.int32),
                "ball_track_offsets": torch.tensor([0], dtype=torch.int64),
                "ball_frame_ids": torch.empty(0, dtype=torch.int32),
                "ball_boxes": torch.empty((0, 4), dtype=torch.float32),
                "ball_point_touched": torch.empty(0, dtype=torch.bool),
                "ball_track_touched": torch.empty(0, dtype=torch.bool),
            }
            torch.save(payload, Path(directory) / "video.pt")
            proposal = RobustClipCropper({"index_root": directory}).get_window_roi(
                "video", 0.0, 10.0, 1920, 1080, (384, 640)
            )
            self.assertFalse(proposal.valid)
            self.assertEqual(proposal.fallback_reason, "no_sampled_detection_frames")


if __name__ == "__main__":
    unittest.main()
