from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

import train_football_events as football


class RelativeContextLossTest(unittest.TestCase):
    def setUp(self) -> None:
        self.previous = dict(football.LABEL_TO_INDEX)
        football.LABEL_TO_INDEX = {"set_piece": 0}
        self.cfg = SimpleNamespace(
            train={
                "raw_context_span_objective": "relative_logit_rank",
                "raw_context_span_relative_temperature": 0.5,
                "raw_context_span_relative_margin": 0.15,
                "raw_context_span_relative_softness": 0.05,
            }
        )
        self.batch = {
            "set_piece_context_span_mask": torch.tensor(
                [[1.0, 1.0, 0.0, 0.0]]
            ),
            "set_piece_context_span_row_mask": torch.tensor([1.0]),
        }

    def tearDown(self) -> None:
        football.LABEL_TO_INDEX = self.previous

    def loss(self, values: list[float]) -> torch.Tensor:
        outputs = {
            "frame_event_logits": torch.tensor(values).reshape(1, 4, 1)
        }
        loss, _ = football.raw_set_piece_context_span_coverage_loss(
            outputs, self.batch, torch.device("cpu"), self.cfg
        )
        return loss

    def test_invariant_to_absolute_logit_shift(self) -> None:
        base = self.loss([0.4, 0.2, 0.1, -0.1])
        shifted = self.loss([5.4, 5.2, 5.1, 4.9])
        self.assertAlmostEqual(float(base), float(shifted), places=6)

    def test_rewards_relative_inside_evidence(self) -> None:
        good = self.loss([1.0, 0.8, 0.0, -0.2])
        bad = self.loss([0.0, -0.2, 1.0, 0.8])
        self.assertLess(float(good), float(bad))
        self.assertEqual(float(good), 0.0)


if __name__ == "__main__":
    unittest.main()

