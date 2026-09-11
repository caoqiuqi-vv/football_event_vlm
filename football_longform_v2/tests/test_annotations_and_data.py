from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from football_longform_v2.annotations import (
    Event,
    build_family_supervision_mask,
    build_family_targets,
    build_label_supervision_mask,
    build_label_targets,
    family_event_times,
    load_event_annotations,
    load_events,
    parse_timestamp,
)
from football_longform_v2.data import VideoBalancedTimelineDataset, VideoCohortBatchSampler
from football_longform_v2.losses import locator_loss, sigmoid_focal_loss


class AnnotationAndDataTest(unittest.TestCase):
    def test_timestamp_notation_from_reviewed_exports(self) -> None:
        self.assertAlmostEqual(parse_timestamp("00:01:21.976") or -1.0, 81.976, places=4)
        self.assertAlmostEqual(parse_timestamp("12:03.5") or -1.0, 723.5, places=4)
        self.assertIsNone(parse_timestamp("not-a-time"))

    def test_wrapped_annotations_and_family_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.json"
            path.write_text(
                json.dumps({"data": [
                    {"label": "其他射门类型", "startTime": "4.0"},
                    {"label": "角球", "startTime": 8.0},
                    {"label": "扑救", "startTime": 5.0, "label_correct": False},
                ]}, ensure_ascii=False),
                encoding="utf-8",
            )
            events = load_events(path)
        self.assertEqual(
            [(event.label, event.family) for event in events],
            [("shot", "shot_chain"), ("corner", "restart")],
        )
        times = torch.arange(0.0, 10.0, 0.2)
        targets, offsets = build_family_targets(
            times, events, ("shot_chain", "restart", "generic_event")
        )
        self.assertEqual(targets.shape, (50, 3))
        self.assertAlmostEqual(float(times[targets[:, 0].argmax()]), 4.0, places=4)
        self.assertAlmostEqual(float(times[targets[:, 1].argmax()]), 8.0, places=4)
        self.assertEqual(offsets.shape, targets.shape)

    def test_shot_chain_uses_shots_and_only_orphan_saves_as_anchors(self) -> None:
        events = (
            Event("shot", "shot_chain", 10.0),
            Event("save", "shot_chain", 11.0),
            Event("shot", "shot_chain", 20.0),
            Event("save", "shot_chain", 30.0),
            Event("corner", "restart", 40.0),
        )
        self.assertEqual(family_event_times(events, "shot_chain"), [10.0, 20.0, 30.0])
        self.assertEqual(family_event_times(events, "restart"), [40.0])
        self.assertEqual(family_event_times(events, "generic_event"), [10.0, 20.0, 30.0, 40.0])

    def test_rejected_events_create_family_specific_ignore_zones(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.json"
            path.write_text(
                json.dumps([
                    {"label": "射门", "timestamp": 5.0, "label_correct": False},
                    {"label": "角球", "timestamp": 8.0, "label_correct": False},
                    {"label": "射门", "timestamp": 9.0, "label_correct": True},
                ], ensure_ascii=False),
                encoding="utf-8",
            )
            annotations = load_event_annotations(path, require_reviewed=True)
        self.assertEqual(len(annotations.accepted), 1)
        self.assertEqual(len(annotations.rejected), 2)
        times = torch.arange(0.0, 11.0)
        mask = build_family_supervision_mask(
            times, annotations.rejected, ("shot_chain", "restart", "generic_event"),
            ignore_radius_seconds={"shot_chain": 1.0, "restart": 1.0, "generic_event": 1.0},
        )
        self.assertFalse(bool(mask[5, 0]))
        self.assertTrue(bool(mask[5, 1]))
        self.assertFalse(bool(mask[5, 2]))
        self.assertFalse(bool(mask[8, 1]))
        class_targets = build_label_targets(
            times, annotations.accepted, ("shot", "save", "corner")
        )
        self.assertEqual(class_targets.shape, (11, 3))
        self.assertEqual(int(class_targets[:, 0].argmax()), 9)
        class_mask = build_label_supervision_mask(
            times, annotations.rejected, ("shot", "save", "corner"),
            ignore_radius_seconds={"shot": 1.0, "save": 1.0, "corner": 1.0},
        )
        self.assertFalse(bool(class_mask[5, 0]))
        self.assertTrue(bool(class_mask[5, 2]))
        self.assertFalse(bool(class_mask[8, 2]))

    def test_density_balanced_focal_is_invariant_to_repeated_easy_negative_density(self) -> None:
        short_logits = torch.zeros(1, 10, 1)
        short_targets = torch.zeros_like(short_logits)
        short_targets[:, 0] = 1.0
        long_logits = torch.zeros(1, 1000, 1)
        long_targets = torch.zeros_like(long_logits)
        long_targets[:, 0] = 1.0
        short_loss = sigmoid_focal_loss(
            short_logits, short_targets, alpha=0.75, density_balanced=True
        )
        long_loss = sigmoid_focal_loss(
            long_logits, long_targets, alpha=0.75, density_balanced=True
        )
        self.assertAlmostEqual(float(short_loss), float(long_loss), places=6)
        outputs = {
            "rgb_logits": short_logits, "offsets": torch.zeros_like(short_logits),
            "class_logits": short_logits.clone(),
            "state_logits": short_logits.clone(),
        }
        family_mask = torch.ones_like(short_targets, dtype=torch.bool)
        family_mask[:, 0] = False
        losses = locator_loss(
            outputs, short_targets, offset_targets=torch.zeros_like(short_targets),
            valid_mask=family_mask, density_balanced=True,
            class_targets=short_targets, class_valid_mask=family_mask, class_weight=0.5,
            state_targets=short_targets, state_valid_mask=family_mask, state_weight=0.1,
        )
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertGreater(float(losses["class_heatmap_loss"]), 0.0)
        self.assertGreater(float(losses["state_heatmap_loss"]), 0.0)

    def test_video_cohort_sampler_is_diverse_complete_and_worker_affine(self) -> None:
        sampler = VideoCohortBatchSampler(
            35, 3, batch_size=4, worker_count=2, seed=7
        )
        batches = list(sampler)
        self.assertEqual(len(batches), len(sampler))
        self.assertEqual(
            sorted(index for batch in batches for index in batch), list(range(35 * 3))
        )
        self.assertTrue(all(
            len({index % 35 for index in batch}) == len(batch) for batch in batches
        ))
        self.assertEqual(
            [index % 35 for index in batches[0]],
            [index % 35 for index in batches[2]],
        )
        self.assertEqual(
            [index % 35 for index in batches[1]],
            [index % 35 for index in batches[3]],
        )

    def test_video_balanced_dataset_reads_feature_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature_root = root / "features"
            annotation_root = root / "annotations"
            annotation_root.mkdir()
            for video_id in ("v1", "v2"):
                (feature_root / video_id).mkdir(parents=True)
                np.savez_compressed(
                    feature_root / video_id / "timeline.npz",
                    timestamps=np.arange(20, dtype=np.float32) / 2.0,
                    context=np.ones((20, 4), dtype=np.float32),
                    motion=np.ones((20, 3), dtype=np.float32),
                    context_valid=np.ones(20, dtype=np.bool_),
                    motion_valid=np.ones(20, dtype=np.bool_),
                )
                (annotation_root / f"{video_id}.json").write_text(
                    json.dumps({"data": [{"label": "射门", "startTime": 5.0}]}, ensure_ascii=False),
                    encoding="utf-8",
                )
            dataset = VideoBalancedTimelineDataset(
                ("v1", "v2"),
                feature_store=feature_root,
                annotations=annotation_root,
                families=("shot_chain", "restart", "generic_event"),
                timeline_hz=2.0,
                sequence_seconds=4.0,
                labels=("shot", "save", "corner"),
                blocks_per_video=2,
                seed=3,
            )
            first = dataset[0]
            second = dataset[1]
            dataset_length = len(dataset)
        self.assertEqual(first["context"].shape, (8, 4))
        self.assertEqual(second["motion"].shape, (8, 3))
        self.assertEqual(first["targets"].shape, (8, 3))
        self.assertEqual(first["valid"].shape, (8, 3))
        self.assertEqual(first["class_targets"].shape, (8, 3))
        self.assertEqual(first["class_state_targets"].shape, (8, 3))
        self.assertTrue(torch.all(first["class_state_targets"] >= first["class_targets"]))
        self.assertEqual(first["class_valid"].shape, (8, 3))
        self.assertEqual(dataset_length, 4)


if __name__ == "__main__":
    unittest.main()
