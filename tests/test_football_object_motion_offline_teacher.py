from __future__ import annotations

import numpy as np
import pytest
import torch

from football_object_motion.offline_teacher import OfflineTrackedBallTeacher
from football_object_motion.teacher import OnlineObjectMotionTeacher


def _empty_batch() -> dict:
    batch = {
        "object_motion_inputs": torch.zeros(1, 3, 3, 16, 32),
        "object_motion_times": torch.tensor([[0.0, 0.1, 0.2]]),
        "meta": [{"video_id": "video_a"}],
    }
    batch.update(OnlineObjectMotionTeacher.empty(1, 3, 8))
    return batch


def test_offline_track_teacher_preserves_unknown_and_weak_interpolation(tmp_path) -> None:
    np.savez(
        tmp_path / "video_a.npz",
        timestamp_sec=np.asarray([0.0, 0.1], dtype=np.float32),
        bbox_xyxy_norm=np.asarray(
            [[0.20, 0.25, 0.30, 0.35], [0.30, 0.35, 0.40, 0.45]],
            dtype=np.float32,
        ),
        confidence=np.asarray([0.8, 0.4], dtype=np.float16),
        quality_weight=np.asarray([1.0, 0.5], dtype=np.float16),
        flags=np.asarray([3, 2], dtype=np.uint8),
        source_code=np.asarray([1, 3], dtype=np.uint8),
    )
    teacher = OfflineTrackedBallTeacher(
        index_root=str(tmp_path),
        patch_size=8,
        max_time_delta_sec=0.04,
        ball_sigma_patches=1.0,
    )
    batch = _empty_batch()
    metrics = teacher.fill(batch)

    assert metrics["offline_ball_teacher_match_fraction"] == pytest.approx(2 / 3)
    assert batch["object_motion_heatmap_masks"][0, 0, :, 0].sum() > 0
    assert batch["object_motion_coordinate_masks"][0, 0, 0] == 1.0
    # Interpolated track points inform visibility/motion but are not strong
    # spatial pseudo-labels.
    assert batch["object_motion_heatmap_masks"][0, 1, :, 0].sum() == 0
    assert batch["object_motion_coordinate_masks"][0, 1, 0] == 0
    assert batch["object_motion_motion_quality"][0, 1, 0] == pytest.approx(0.5)
    assert batch["object_motion_presence_masks"][0, 1, 0] == pytest.approx(0.125)
    # No nearby pseudo-label is unknown, never a negative target.
    assert batch["object_motion_teacher_filled"][0, 2] == 0
    assert batch["object_motion_presence_masks"][0, 2, 0] == 0


def test_missing_video_is_unknown(tmp_path) -> None:
    teacher = OfflineTrackedBallTeacher(
        index_root=str(tmp_path),
        patch_size=8,
    )
    batch = _empty_batch()
    batch["meta"][0]["video_id"] = "not_indexed"
    metrics = teacher.fill(batch)

    assert metrics["offline_ball_teacher_missing_videos"] == 1
    assert batch["object_motion_teacher_filled"].sum() == 0
    assert batch["object_motion_heatmap_masks"][..., 0].sum() == 0


def test_online_teacher_can_be_goal_only() -> None:
    teacher = OnlineObjectMotionTeacher(
        device=torch.device("cpu"),
        ball_checkpoint="unused_ball.pt",
        scene_checkpoint="goal.pt",
        patch_size=8,
        enabled_objects=("goal",),
        half=False,
    )
    assert teacher.enabled_objects == frozenset({"goal"})


def test_online_teacher_rejects_unknown_object() -> None:
    with pytest.raises(ValueError, match="enabled_objects"):
        OnlineObjectMotionTeacher(
            device=torch.device("cpu"),
            ball_checkpoint="ball.pt",
            scene_checkpoint="scene.pt",
            patch_size=8,
            enabled_objects=("referee",),
            half=False,
        )
