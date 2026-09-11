from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from football_detection_aware import DetectionHintRenderer
from scripts.enrich_football_detection_hints import (
    add_normalized_coordinates,
    candidate_jobs,
    densest_people_crop,
    merge_payload,
)


def make_index(path: Path) -> dict:
    payload = {
        "version": 2,
        "video_id": "video",
        "fps": 30.0,
        "frame_id_semantics": "original_source_frame_id",
        "sample_fps": 2.0,
        "image_size": {"width": 200, "height": 100},
        "frame_ids": torch.tensor([30], dtype=torch.int32),
        "frame_offsets": torch.tensor([0, 9], dtype=torch.int64),
        "classes": torch.tensor([1, 1, 1, 1, 2, 0, 0, 0, 0], dtype=torch.int8),
        "confidences": torch.tensor([0.9, 0.8, 0.7, 0.6, 0.9, 0.8, 0.8, 0.8, 0.8]),
        "boxes": torch.tensor(
            [
                [10, 10, 14, 14],
                [30, 10, 34, 14],
                [50, 10, 54, 14],
                [70, 10, 74, 14],
                [150, 20, 195, 80],
                [20, 30, 40, 80],
                [45, 30, 65, 80],
                [70, 30, 90, 80],
                [95, 30, 115, 80],
            ],
            dtype=torch.float32,
        ),
        "track_ids": torch.full((9,), -1, dtype=torch.int32),
        "ball_track_offsets": torch.tensor([0], dtype=torch.int64),
        "ball_frame_ids": torch.empty(0, dtype=torch.int32),
        "ball_boxes": torch.empty((0, 4), dtype=torch.float32),
        "ball_point_touched": torch.empty(0, dtype=torch.bool),
        "ball_track_touched": torch.empty(0, dtype=torch.bool),
    }
    torch.save(payload, path)
    return payload


class DetectionHintRendererTest(unittest.TestCase):
    def test_renders_top_three_balls_and_goal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            make_index(Path(directory) / "video.pt")
            renderer = DetectionHintRenderer(
                {
                    "index_root": directory,
                    "topk_ball": 3,
                    "topk_goal": 1,
                    "ball_alpha": 1.0,
                    "goal_alpha": 1.0,
                    "raw_rgb_prob": 0.0,
                    "ball_dropout_prob": 0.0,
                    "goal_dropout_prob": 0.0,
                }
            )
            frame = np.zeros((100, 200, 3), dtype=np.uint8)
            rendered = renderer.render(frame, "video", 31, is_train=False)
            self.assertEqual(rendered.shape, frame.shape)
            self.assertGreater(int(rendered.sum()), 0)
            self.assertEqual(int(rendered[12, 72].sum()), 0)

    def test_raw_rgb_dropout_is_exact_noop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            make_index(Path(directory) / "video.pt")
            renderer = DetectionHintRenderer(
                {"index_root": directory, "raw_rgb_prob": 1.0}
            )
            frame = np.full((100, 200, 3), 17, dtype=np.uint8)
            with patch("football_detection_aware.random.random", return_value=0.0):
                rendered = renderer.render(frame, "video", 30, is_train=True)
            np.testing.assert_array_equal(rendered, frame)

    def test_renderer_prefers_normalized_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.pt"
            payload = add_normalized_coordinates(make_index(path))
            # Deliberately corrupt the legacy absolute boxes. Rendering must
            # still use the normalized source-frame coordinates.
            payload["boxes"] = torch.zeros_like(payload["boxes"])
            torch.save(payload, path)
            renderer = DetectionHintRenderer(
                {
                    "index_root": directory,
                    "topk_ball": 1,
                    "topk_goal": 0,
                    "ball_alpha": 1.0,
                    "raw_rgb_prob": 0.0,
                    "ball_dropout_prob": 0.0,
                }
            )
            rendered = renderer.render(np.zeros((200, 400, 3), dtype=np.uint8), "video", 30, is_train=False)
            self.assertGreater(int(rendered[24, 24].sum()), 0)


class ConditionalRedetectTest(unittest.TestCase):
    def test_densest_people_crop_and_merge_provenance(self) -> None:
        boxes = np.asarray(
            [[10, 10, 20, 40], [30, 10, 40, 40], [50, 10, 60, 40], [70, 10, 80, 40]],
            dtype=np.float32,
        )
        crop = densest_people_crop(
            boxes, np.ones(4, dtype=np.float32), 100, 200, 100, 4
        )
        self.assertIsNotNone(crop)
        with tempfile.TemporaryDirectory() as directory:
            payload = make_index(Path(directory) / "video.pt")
        merged = merge_payload(
            payload,
            {30: [{"conf": 0.5, "bbox": [100, 10, 105, 15], "source": 2}]},
        )
        self.assertEqual(len(merged["classes"]), len(payload["classes"]) + 1)
        self.assertEqual(int(merged["object_sources"][-1]), 2)
        self.assertIn("boxes_normalized", merged)
        torch.testing.assert_close(
            merged["boxes_normalized"][-1],
            torch.tensor([0.5, 0.1, 0.525, 0.15]),
        )
        self.assertEqual(
            merged["normalized_box_coordinate_system"],
            "normalized_xyxy_relative_to_original_source_frame",
        )

    def test_no_redetect_when_original_ball_exists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = make_index(Path(directory) / "video.pt")
        args = argparse.Namespace(
            ball_conf=0.1,
            goal_conf=0.4,
            person_conf=0.35,
            crop_size=64,
            crowd_min_people=4,
        )
        jobs, stats = candidate_jobs(payload, [(0.0, 2.0)], args)
        self.assertEqual(jobs, [])
        self.assertEqual(stats["original_ball_frames"], 1)


if __name__ == "__main__":
    unittest.main()
