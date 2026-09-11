from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from train_football_events import (
    FootballLongVideoDataset,
    LongVideoRecord,
    configure_label_schema,
    to_config,
)


class _CapturingEvidenceProvider:
    def __init__(self) -> None:
        self.times: list[float] | None = None

    def features(self, video_id: str, absolute_times: list[float]) -> torch.Tensor:
        self.times = list(absolute_times)
        return torch.zeros(len(absolute_times), 1)


class _CapturingObjectTeacher:
    def __init__(self) -> None:
        self.times: list[float] | None = None

    def targets(
        self, video_id: str, absolute_times: list[float]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.times = list(absolute_times)
        shape = (len(absolute_times), 1, 2)
        return torch.zeros(shape), torch.zeros(shape)


class FootballProviderTimeAlignmentTest(unittest.TestCase):
    def setUp(self) -> None:
        configure_label_schema(to_config({"task": {"label_schema": "set_piece"}}))

    def test_dataset_passes_decoded_absolute_times_to_offline_providers(self) -> None:
        record = LongVideoRecord(
            source="test",
            split="train",
            video_id="v1",
            sample_id="negative_v1",
            video_path="/unused/v1.mp4",
            annotation_path="/unused/v1.json",
            anchor_time=102.5,
            base_clip_start=100.0,
            base_clip_end=105.0,
            video_duration=200.0,
            is_negative=True,
            labels=(0.0, 0.0, 0.0),
            label_mask=(1.0, 1.0, 1.0),
        )
        evidence = _CapturingEvidenceProvider()
        object_teacher = _CapturingObjectTeacher()
        dataset = FootballLongVideoDataset(
            [record],
            {("test", "v1"): []},
            num_frames=3,
            image_size=(8, 8),
            clip_duration=5.0,
            sampling_duration=5.0,
            event_margin=0.5,
            temporal_jitter_sec=0.0,
            is_train=False,
            hflip_prob=0.0,
            frame_supervision=False,
            evidence_provider=evidence,
            object_teacher_provider=object_teacher,
        )
        decoded_times = torch.tensor([100.0, 102.5, 105.0])
        with patch(
            "train_football_events.read_video_segment",
            return_value=(torch.zeros(3, 3, 8, 8), decoded_times),
        ) as reader:
            item = dataset[0]

        self.assertEqual(evidence.times, [100.0, 102.5, 105.0])
        self.assertEqual(object_teacher.times, [100.0, 102.5, 105.0])
        self.assertIn("evidence_features", item)
        self.assertIn("object_heatmap_targets", item)
        self.assertTrue(reader.call_args.kwargs["return_frame_times"])


if __name__ == "__main__":
    unittest.main()
