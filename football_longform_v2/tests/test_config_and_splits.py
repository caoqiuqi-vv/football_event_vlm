from __future__ import annotations

import unittest
from pathlib import Path

from football_longform_v2.config import load_config
from football_longform_v2.schema import assert_disjoint_splits, read_video_ids


ROOT = Path(__file__).resolve().parents[1]


class ConfigAndSplitTest(unittest.TestCase):
    def test_rgb_only_contract_and_frozen_test(self) -> None:
        config = load_config(ROOT / "configs/lf_a0_rgb_only.yaml")
        self.assertTrue(config["features"]["context"]["required"])
        self.assertTrue(config["features"]["motion"]["required"])
        self.assertFalse(config["features"]["entity"]["enabled"])
        paths = config["paths"]
        train = read_video_ids((ROOT / paths["train_ids"]).resolve())
        calibration = read_video_ids((ROOT / paths["calibration_ids"]).resolve())
        test = read_video_ids((ROOT / paths["thirdparty_test_ids"]).resolve())
        assert_disjoint_splits(train, calibration, test)
        self.assertEqual((len(train), len(calibration), len(test)), (147, 20, 18))


if __name__ == "__main__":
    unittest.main()

