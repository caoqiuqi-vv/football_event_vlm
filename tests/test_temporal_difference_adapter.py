from __future__ import annotations

import unittest

import torch

from train_football_events import TemporalDifferenceAdapter


class TemporalDifferenceAdapterTest(unittest.TestCase):
    def test_first_and_second_order_boundaries(self) -> None:
        tokens = torch.tensor([[[0.0], [1.0], [3.0], [6.0]]])
        first, second = TemporalDifferenceAdapter.temporal_differences(tokens)
        self.assertTrue(
            torch.equal(first, torch.tensor([[[0.0], [1.0], [2.0], [3.0]]]))
        )
        self.assertTrue(
            torch.equal(second, torch.tensor([[[0.0], [0.0], [1.0], [1.0]]]))
        )

    def test_zero_initialized_residual_preserves_baseline(self) -> None:
        torch.manual_seed(3)
        adapter = TemporalDifferenceAdapter(
            hidden_dim=8,
            mode="delta2",
            dropout=0.0,
            gate_init=0.25,
        )
        tokens = torch.randn(2, 5, 8)
        enhanced, residual, gate = adapter(tokens)
        self.assertTrue(torch.equal(enhanced, tokens))
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))
        self.assertAlmostEqual(float(gate.detach()), 0.25, places=5)

    def test_adapter_receives_gradient(self) -> None:
        adapter = TemporalDifferenceAdapter(
            hidden_dim=4,
            mode="delta2",
            dropout=0.0,
        )
        tokens = torch.randn(2, 4, 4)
        enhanced, _, _ = adapter(tokens)
        enhanced.square().mean().backward()
        final_weight = adapter.adapter[-1].weight
        self.assertIsNotNone(final_weight.grad)
        self.assertGreater(float(final_weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
