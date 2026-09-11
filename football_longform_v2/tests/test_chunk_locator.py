from __future__ import annotations

import unittest

import torch

from football_longform_v2.models import CausalShotContext, SequentialChunkLocator


class CausalShotContextTest(unittest.TestCase):
    def test_future_values_do_not_change_past_context(self) -> None:
        module = CausalShotContext(history_steps=4)
        baseline = torch.zeros(1, 8, 1)
        changed = baseline.clone()
        changed[:, 5:, 0] = 1.0
        first = module(baseline)
        second = module(changed)
        torch.testing.assert_close(first[:, :5], second[:, :5])
        self.assertGreater(float(second[:, 5:].sum()), 0.0)


class SequentialChunkLocatorTest(unittest.TestCase):
    def test_shapes_gradients_and_invalid_mask(self) -> None:
        model = SequentialChunkLocator(
            input_dim=32,
            hidden_dim=48,
            dilations=(1, 2, 4),
            shot_history_steps=5,
        )
        features = torch.randn(2, 31, 32, requires_grad=True)
        valid = torch.ones(2, 31, dtype=torch.bool)
        valid[1, -3:] = False
        outputs = model(features, valid)
        self.assertEqual(tuple(outputs["class_logits"].shape), (2, 31, 4))
        self.assertEqual(tuple(outputs["family_logits"].shape), (2, 31, 2))
        self.assertEqual(tuple(outputs["offsets"].shape), (2, 31, 4))
        self.assertEqual(model.receptive_field_steps, 29)
        self.assertTrue(bool((outputs["class_logits"][1, -3:] == -20.0).all()))
        outputs["class_logits"][:, :-3].mean().backward()
        self.assertIsNotNone(features.grad)
        self.assertGreater(float(features.grad.abs().sum()), 0.0)
        self.assertIsNotNone(model.chunk_phase_embedding.weight.grad)


if __name__ == "__main__":
    unittest.main()
