from __future__ import annotations

import unittest

from train_football_events import LongVideoRecord, nested_sampling_window


def make_record(*, is_negative: bool, anchor_time: float) -> LongVideoRecord:
    return LongVideoRecord(
        source="test",
        split="train",
        video_id="video",
        sample_id="negative" if is_negative else "positive",
        video_path="video.mp4",
        annotation_path="events.json",
        anchor_time=anchor_time,
        base_clip_start=10.0,
        base_clip_end=20.0,
        video_duration=30.0,
        is_negative=is_negative,
        labels=(0.0, 0.0, 0.0) if is_negative else (1.0, 0.0, 0.0),
        label_mask=(1.0, 1.0, 1.0),
    )


class DenseSamplingWindowTest(unittest.TestCase):
    def test_positive_window_contains_anchor_and_context_bounds(self) -> None:
        record = make_record(is_negative=False, anchor_time=11.0)
        start, end = nested_sampling_window(record, 10.0, 20.0, 4.0)
        self.assertEqual((start, end), (10.0, 14.0))
        self.assertLessEqual(start, record.anchor_time)
        self.assertGreaterEqual(end, record.anchor_time)

    def test_negative_window_uses_context_center(self) -> None:
        record = make_record(is_negative=True, anchor_time=0.0)
        self.assertEqual(
            nested_sampling_window(record, 10.0, 20.0, 4.0),
            (13.0, 17.0),
        )

    def test_full_duration_is_backward_compatible(self) -> None:
        record = make_record(is_negative=False, anchor_time=11.0)
        self.assertEqual(
            nested_sampling_window(record, 10.0, 20.0, 10.0),
            (10.0, 20.0),
        )


