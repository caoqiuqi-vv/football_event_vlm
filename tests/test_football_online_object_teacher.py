import unittest

import torch

from football_online_object_teacher import OnlineObjectTeacherTargeter


class _Boxes:
    def __init__(self, score, box):
        self.conf = torch.tensor([score], dtype=torch.float32)
        self.xyxy = torch.tensor([box], dtype=torch.float32)


class _Result:
    def __init__(self, score, box):
        self.boxes = _Boxes(score, box)


class OnlineObjectTeacherTargeterTest(unittest.TestCase):
    def test_fills_missing_positive_and_negative_object_channels(self):
        targeter = OnlineObjectTeacherTargeter(
            device=torch.device("cpu"),
            ball_checkpoint="unused_ball.pt",
            goal_checkpoint="unused_goal.pt",
            patch_size=16,
            ball_confidence=0.10,
            goal_confidence=0.25,
            ball_sigma_patches=1.25,
            goal_dilation_patches=0.5,
            input_size=(32, 32),
            batch_size=2,
            half=False,
        )
        targeter.ball_model = object()
        targeter.goal_model = object()
        targeter.ball_class_id = 0
        targeter.goal_class_id = 1

        def fake_predict(model, frames, *, class_id, confidence):
            self.assertEqual(tuple(frames.shape), (2, 3, 32, 32))
            if model is targeter.ball_model:
                return [_Result(0.9, [12, 12, 20, 20]) for _ in range(len(frames))]
            return [_Result(0.8, [16, 4, 31, 28]) for _ in range(len(frames))]

        targeter._predict = fake_predict
        batch = {
            "inputs": torch.zeros(1, 2, 3, 32, 32, dtype=torch.uint8),
            "object_heatmap_targets": torch.zeros(1, 2, 4, 2),
            "object_heatmap_masks": torch.zeros(1, 2, 4, 2),
        }
        metrics = targeter.fill_missing(batch)
        self.assertEqual(metrics["online_object_teacher_frames"], 2.0)
        self.assertTrue(torch.all(batch["object_heatmap_masks"] == 1))
        self.assertGreater(float(batch["object_heatmap_targets"][..., 0].max()), 0.0)
        self.assertGreater(float(batch["object_heatmap_targets"][..., 1].max()), 0.0)

        cached_metrics = targeter.fill_missing(batch)
        self.assertEqual(cached_metrics["online_object_teacher_frames"], 0.0)


if __name__ == "__main__":
    unittest.main()

