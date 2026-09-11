from __future__ import annotations

import numpy as np

from scripts.extract_whistle_activity_candidates import connected_activity
from scripts.goal_oriented_review_policy import make_review_segments, match_selected


def test_adjacent_same_class_candidates_remain_two_instances() -> None:
    rows = [
        {"candidate_id": "a", "video_id": "v", "time_sec": 10.0, "score": 0.9},
        {"candidate_id": "b", "video_id": "v", "time_sec": 11.0, "score": 0.8},
    ]
    metrics = match_selected(rows, {"v": [10.0, 11.0]}, tolerance_sec=0.6)
    assert metrics["tp"] == 2
    assert metrics["fp"] == 0


def test_same_time_cross_class_instances_are_evaluated_independently() -> None:
    shot = [{"candidate_id": "s", "video_id": "v", "time_sec": 10.0, "score": 0.9}]
    save = [{"candidate_id": "g", "video_id": "v", "time_sec": 10.0, "score": 0.9}]
    assert match_selected(shot, {"v": [10.0]}, 0.1)["tp"] == 1
    assert match_selected(save, {"v": [10.0]}, 0.1)["tp"] == 1


def test_display_interval_grouping_does_not_drop_candidate_ids() -> None:
    rows = [
        {
            "candidate_id": "shot-1", "video_id": "v", "time_sec": 10.0,
            "score": 0.9, "label": "shot", "source": "visual",
        },
        {
            "candidate_id": "save-1", "video_id": "v", "time_sec": 10.2,
            "score": 0.8, "label": "save", "source": "visual",
        },
    ]
    segments = make_review_segments(
        rows, {"v": 100.0}, review_sec=10.0, max_segment_sec=20.0,
    )
    assert len(segments) == 1
    assert set(segments[0]["candidate_ids"].split(";")) == {"shot-1", "save-1"}


def test_whistle_connected_activity_is_signal_grouping_not_point_nms() -> None:
    times = np.asarray([0.0, 0.2, 0.4, 1.4, 1.6])
    scores = np.asarray([0.0, 1.0, 0.8, 1.1, 0.9])
    rows = connected_activity(times, scores, threshold=0.5, max_gap_sec=0.3)
    assert len(rows) == 2
    assert rows[0]["activity_points"] == 2
    assert rows[1]["activity_points"] == 2
