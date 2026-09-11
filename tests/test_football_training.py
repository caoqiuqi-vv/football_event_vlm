from __future__ import annotations

import random
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from train_football_events import (
    ConfigDict,
    configure_label_schema,
    label_mask_for_sample,
    label_mask_from_counts,
    load_config,
    per_video_metrics,
    resolve_pos_weight,
    seed_everything,
    to_config,
)


ROOT = Path(__file__).resolve().parents[1]


class FootballTrainingOptimizationTest(unittest.TestCase):
    def setUp(self) -> None:
        configure_label_schema(to_config({"task": {"label_schema": "set_piece"}}))

    def test_fixed_pos_weight_does_not_change_with_sample_ratio(self) -> None:
        positive = SimpleNamespace(labels=(1.0, 0.0, 0.0), label_mask=(1.0, 1.0, 1.0))
        negative = SimpleNamespace(labels=(0.0, 0.0, 0.0), label_mask=(1.0, 1.0, 1.0))
        cfg = ConfigDict({"pos_weight": [1.0, 1.0, 1.0], "pos_weight_max": 50.0})

        weights_r1, mode_r1 = resolve_pos_weight([positive, negative], cfg)
        weights_r5, mode_r5 = resolve_pos_weight([positive] + [negative] * 5, cfg)

        self.assertEqual(mode_r1, "fixed")
        self.assertEqual(mode_r5, "fixed")
        torch.testing.assert_close(weights_r1, torch.ones(3))
        torch.testing.assert_close(weights_r5, torch.ones(3))

    def test_seed_everything_replays_python_numpy_and_torch(self) -> None:
        seed_everything(1234, deterministic=False)
        first = (random.random(), float(np.random.rand()), float(torch.rand(1).item()))
        seed_everything(1234, deterministic=False)
        second = (random.random(), float(np.random.rand()), float(torch.rand(1).item()))
        self.assertEqual(first, second)

    def test_sparse_video_masks_only_untrusted_label(self) -> None:
        mask = label_mask_from_counts({"shot": 9, "save": 11, "set_piece": 2}, min_events_per_label=4)
        self.assertEqual(mask, (1.0, 1.0, 0.0))
        self.assertEqual(label_mask_for_sample(mask, (0.0, 0.0, 0.0)), (1.0, 1.0, 0.0))

    def test_per_video_metrics_use_default_threshold_and_label_masks(self) -> None:
        targets = np.asarray(
            [
                [1, 0, 0],
                [0, 1, 0],
                [1, 0, 1],
                [0, 0, 0],
            ],
            dtype=np.int32,
        )
        probs = np.asarray(
            [
                [0.9, 0.1, 0.1],
                [0.8, 0.7, 0.9],
                [0.6, 0.2, 0.8],
                [0.1, 0.2, 0.3],
            ],
            dtype=np.float32,
        )
        masks = np.asarray(
            [
                [1, 1, 1],
                [1, 1, 0],
                [1, 1, 1],
                [1, 1, 1],
            ],
            dtype=np.float32,
        )
        metas = [
            {"source": "xbotgo", "video_id": "video_a"},
            {"source": "xbotgo", "video_id": "video_a"},
            {"source": "xbotgo", "video_id": "video_b"},
            {"source": "xbotgo", "video_id": "video_b"},
        ]

        rows = per_video_metrics(targets, probs, masks, metas, np.full(3, 0.5, dtype=np.float32))

        self.assertEqual([row["video_id"] for row in rows], ["video_a", "video_b"])
        video_a = rows[0]["per_class"]
        self.assertEqual((video_a["shot"]["tp"], video_a["shot"]["fp"], video_a["shot"]["fn"]), (1, 1, 0))
        self.assertEqual(video_a["shot"]["precision"], 0.5)
        self.assertEqual(video_a["shot"]["recall"], 1.0)
        self.assertEqual(video_a["set_piece"]["valid_count"], 1)
        self.assertEqual(video_a["set_piece"]["unknown_count"], 1)
        self.assertEqual(video_a["shot"]["threshold"], 0.5)

    def test_primary_configs_enable_all_four_optimizations(self) -> None:
        configs = [
            ROOT / "configs/football/dinov3_vitl16_lvd1689m_long_set_piece_freeze_temporal.yaml",
            ROOT / "configs/football/dinov3_vitl16_lvd1689m_long_set_piece_lora_temporal.yaml",
        ]
        for path in configs:
            with self.subTest(path=path.name):
                cfg = load_config(str(path), [])
                self.assertEqual(cfg.seed, 42)
                self.assertTrue(cfg.deterministic)
                self.assertFalse(cfg.data.long_video.negative_require_all_labels)
                self.assertEqual(list(cfg.train.pos_weight), [1.0, 1.0, 1.0])
                self.assertEqual(float(cfg.eval.threshold), 0.5)


if __name__ == "__main__":
    unittest.main()
