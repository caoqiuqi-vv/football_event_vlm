from __future__ import annotations

import unittest

import torch

from football_e2e_spotter.set_pipeline import choose_threshold, core_starts


class SetPipelineTest(unittest.TestCase):
    def test_cores_are_contiguous_not_overlapping(self) -> None:
        self.assertEqual(core_starts(65, 30), [0.0, 30.0, 60.0])

    def test_threshold_is_recall_constrained(self) -> None:
        threshold, metric = choose_threshold(torch.tensor([.9, .8, .7, .1]), torch.tensor([1, 0, 1, 0], dtype=torch.bool), target_recall=1.0)
        self.assertAlmostEqual(threshold, .7, places=5)
        self.assertEqual(metric["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
