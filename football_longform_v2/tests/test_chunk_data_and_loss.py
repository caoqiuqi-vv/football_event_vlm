from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from football_longform_v2.annotations import Event
from football_longform_v2.chunk_data import SequentialChunkDataset, build_family_state_targets
from football_longform_v2.chunk_evaluation import (
    decode_class_peaks, evaluate_sequential_chunk_locator,
)
from football_longform_v2.losses import sequential_chunk_locator_loss
from football_longform_v2.models import SequentialChunkLocator

import importlib.util


class SequentialChunkDataTest(unittest.TestCase):
    def test_long_video_report_prioritizes_shot_gate_and_includes_uniform_controls(self) -> None:
        class FixedPeakModel(torch.nn.Module):
            def forward(self, features, valid, chunk_phase):
                del valid, chunk_phase
                batch, steps, _ = features.shape
                logits = torch.full((batch, steps, 4), -10.0, device=features.device)
                logits[:, 8, 0] = 10.0
                return {"class_logits": logits, "offsets": torch.zeros_like(logits)}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_dir = root / "store" / "calibration" / "video1"
            annotation_dir = root / "annotations"
            video_dir.mkdir(parents=True)
            annotation_dir.mkdir()
            np.save(video_dir / "features.npy", np.random.randn(20, 12).astype(np.float16))
            np.save(video_dir / "timestamps.npy", (0.25 + 0.5 * np.arange(20)).astype(np.float32))
            (video_dir / "metadata.json").write_text(json.dumps({
                "schema": "football_longform_v2.videomae_chunk_store.v1",
                "video_id": "video1", "feature_dim": 12, "tubelet_seconds": 0.5,
                "chunk_seconds": 4.0, "checkpoint_sha256": "abc",
            }))
            (annotation_dir / "video1.json").write_text(json.dumps([
                {"label": "射门", "startTime": 4.0, "label_correct": True},
            ]))
            dataset = SequentialChunkDataset(
                ("video1",), store_root=root / "store", split="calibration",
                annotations=annotation_dir, block_seconds=12.0, blocks_per_video=1,
                labels=SequentialChunkLocator.LABELS,
            )
            labels = dataset.labels
            report = evaluate_sequential_chunk_locator(
                FixedPeakModel(), dataset, device=torch.device("cpu"),
                nms_radius_seconds={label: 1.0 for label in labels},
                target_recall={label: 0.9 if label == "shot" else 0.85 for label in labels},
                minimum_precision_at_target_recall={label: 0.1 for label in labels},
                maximum_fp_per_minute={label: 1.0 for label in labels},
                uniform_baseline_intervals_seconds=(4.0,), amp_dtype=None,
            )
            self.assertEqual(report["selection_tuple"][:2], [1, 0])
            self.assertEqual(report["selection_tuple"][4], 1)
            baseline = report["uniform_time_sampling_baselines"]["4s"]["3s"]["shot"]
            self.assertEqual(baseline["support"], 1)
            self.assertEqual(baseline["recall"], 1.0)
            self.assertGreater(baseline["proposals_per_minute"], 0.0)

    def test_false_budget_threshold_respects_zero_and_positive_budgets(self) -> None:
        script = Path(__file__).resolve().parents[1] / "scripts/freeze_a2_operating_points.py"
        spec = importlib.util.spec_from_file_location("freeze_a2", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        zero = module.false_budget_threshold([0.9, 0.5], 0.0, 10.0)
        self.assertEqual(zero["selected_false_peaks"], 0)
        one = module.false_budget_threshold([0.9, 0.5, 0.1], 0.1, 10.0)
        self.assertEqual(one["false_budget"], 1)
        self.assertEqual(one["selected_false_peaks"], 1)

    def test_family_state_retains_later_save_phase(self) -> None:
        timestamps = 0.5 * torch.arange(20)
        events = (
            Event("shot", "shot_chain", 2.0),
            Event("save", "shot_chain", 6.0),
        )
        targets = build_family_state_targets(
            timestamps, events, ("shot_chain", "restart"),
            sigma_seconds={"shot_chain": 1.0, "restart": 1.0},
        )
        self.assertGreater(float(targets[4, 0]), 0.99)
        self.assertGreater(float(targets[12, 0]), 0.99)

    def test_class_peak_decode_uses_offsets_without_duplicate_plateau_peaks(self) -> None:
        logits = torch.full((9, 5), -8.0)
        logits[4, 0] = 8.0
        offsets = torch.zeros_like(logits)
        offsets[4, 0] = 0.25
        timestamps = 0.25 + 0.5 * torch.arange(9)
        proposals = decode_class_peaks(
            logits, offsets, timestamps,
            ("shot", "save", "corner", "freekick", "penalty"),
            nms_radius_seconds={
                "shot": 1.0, "save": 1.0, "corner": 1.0,
                "freekick": 1.0, "penalty": 1.0,
            },
        )
        self.assertEqual(len(proposals["shot"]), 1)
        self.assertAlmostEqual(proposals["shot"][0].timestamp, 2.5)

    def test_mmap_dataset_targets_padding_and_backward(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_dir = root / "store" / "train" / "video1"
            annotation_dir = root / "annotations"
            video_dir.mkdir(parents=True)
            annotation_dir.mkdir()
            np.save(video_dir / "features.npy", np.random.randn(20, 12).astype(np.float16))
            np.save(video_dir / "timestamps.npy", (0.25 + 0.5 * np.arange(20)).astype(np.float32))
            (video_dir / "metadata.json").write_text(json.dumps({
                "schema": "football_longform_v2.videomae_chunk_store.v1",
                "video_id": "video1",
                "feature_dim": 12,
                "tubelet_seconds": 0.5,
                "chunk_seconds": 4.0,
                "checkpoint_sha256": "abc",
            }))
            (annotation_dir / "video1.json").write_text(json.dumps([
                {"label": "射门", "startTime": 4.0, "label_correct": True},
                {"label": "扑救", "startTime": 4.5, "label_correct": True},
                {"label": "角球", "startTime": 8.0, "label_correct": True},
            ]))
            dataset = SequentialChunkDataset(
                ("video1",), store_root=root / "store", split="train",
                annotations=annotation_dir, block_seconds=12.0, blocks_per_video=1,
                positive_probability=1.0,
                labels=SequentialChunkLocator.LABELS,
            )
            item = dataset[0]
            self.assertEqual(tuple(item["features"].shape), (24, 12))
            self.assertEqual(int(item["valid"].sum()), 20)
            self.assertEqual(item["chunk_phase"].tolist()[:10], [0, 1, 2, 3, 4, 5, 6, 7, 0, 1])
            self.assertGreater(float(item["class_targets"][:, 0].max()), 0.9)
            self.assertGreater(float(item["family_targets"][:, 0].max()), 0.9)
            self.assertTrue(bool((~item["class_valid"][20:]).all()))

            model = SequentialChunkLocator(input_dim=12, hidden_dim=24, dilations=(1, 2))
            outputs = model(
                item["features"].unsqueeze(0), item["valid"].unsqueeze(0),
                item["chunk_phase"].unsqueeze(0),
            )
            losses = sequential_chunk_locator_loss(
                outputs,
                class_targets=item["class_targets"].unsqueeze(0),
                family_targets=item["family_targets"].unsqueeze(0),
                offset_targets=item["offset_targets"].unsqueeze(0),
                class_valid_mask=item["class_valid"].unsqueeze(0),
                family_valid_mask=item["family_valid"].unsqueeze(0),
            )
            losses["loss"].backward()
            self.assertTrue(torch.isfinite(losses["loss"]))
            self.assertGreater(float(model.input_projection[1].weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
