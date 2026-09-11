from __future__ import annotations

import unittest

import numpy as np
import torch

from football_roi_scoring import rank_ball_candidates, rank_object_candidates, stable_nearby_people


class FootballROIScoringTest(unittest.TestCase):
    def test_people_selection_values_dense_stable_group(self) -> None:
        boxes = []
        confidences = []
        track_ids = []
        frame_ids = []
        # One isolated player is closest to the goal, while four persistent
        # players form the more informative action cluster slightly farther away.
        centers = [(150, 300), (360, 280), (390, 290), (420, 300), (450, 310)]
        for track_id, (cx, cy) in enumerate(centers):
            for frame_id in (0, 15, 30, 45):
                boxes.append([cx - 15, cy - 40, cx + 15, cy + 40])
                confidences.append(0.9)
                track_ids.append(track_id)
                frame_ids.append(frame_id)
        selected, support = stable_nearby_people(
            np.asarray(boxes, dtype=np.float32),
            np.asarray(confidences, dtype=np.float32),
            np.asarray(track_ids, dtype=np.int32),
            np.asarray(frame_ids, dtype=np.int32),
            [[80, 240, 130, 340]],
            width=960,
            height=540,
            max_people=3,
        )
        selected_centers = [(box[0] + box[2]) * 0.5 for box in selected]
        self.assertGreaterEqual(sum(center >= 350 for center in selected_centers), 2)
        self.assertGreater(support, 0.0)

    def test_single_frame_large_goal_cannot_beat_stable_goal_cluster(self) -> None:
        stable = [
            [100, 100, 200, 180],
            [102, 101, 202, 181],
            [101, 99, 201, 179],
            [103, 100, 203, 180],
        ]
        boxes = np.asarray(stable + [[0, 0, 900, 500]], dtype=np.float32)
        confidences = np.asarray([0.75, 0.78, 0.80, 0.77, 0.99], dtype=np.float32)
        track_ids = np.asarray([7, 7, 7, 7, 99], dtype=np.int32)
        frame_ids = np.asarray([0, 15, 30, 45, 60], dtype=np.int32)

        candidates = rank_object_candidates(
            boxes,
            confidences,
            track_ids,
            frame_ids,
            width=960,
            height=540,
            min_frames=3,
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].frame_count, 4)
        self.assertLess(candidates[0].box[2], 250)

    def test_long_unsupported_ball_gap_does_not_expand_roi(self) -> None:
        frame_ids = torch.tensor([0, 3, 6, 9, 90], dtype=torch.int32)
        boxes = torch.tensor(
            [
                [40, 40, 48, 48],
                [42, 41, 50, 49],
                [44, 42, 52, 50],
                [46, 43, 54, 51],
                [700, 400, 708, 408],
            ],
            dtype=torch.float32,
        )
        payload = {
            "fps": 30.0,
            "image_size": {"width": 960, "height": 540},
            "ball_track_offsets": torch.tensor([0, 5], dtype=torch.int64),
            "ball_frame_ids": frame_ids,
            "ball_boxes": boxes,
            "ball_point_touched": torch.zeros(5, dtype=torch.bool),
            "ball_track_touched": torch.zeros(1, dtype=torch.bool),
        }
        people_boxes = np.asarray([[30, 20, 80, 180]], dtype=np.float32)
        raw_frames = np.asarray([0, 3, 6, 9], dtype=np.int32)
        raw_boxes = boxes[:4].numpy()

        candidates = rank_ball_candidates(
            payload,
            0,
            100,
            width=960,
            height=540,
            people_boxes=people_boxes,
            raw_ball_frames=raw_frames,
            raw_ball_boxes=raw_boxes,
            min_points=3,
            min_area_ratio=1e-6,
            max_area_ratio=2e-3,
        )

        self.assertEqual(len(candidates), 1)
        self.assertLess(candidates[0].box[2], 100)
        self.assertLessEqual(candidates[0].max_gap_sec, 0.11)
        self.assertEqual(candidates[0].detection_support, 1.0)


if __name__ == "__main__":
    unittest.main()
