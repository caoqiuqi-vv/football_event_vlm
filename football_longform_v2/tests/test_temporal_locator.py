from __future__ import annotations

import unittest

import torch

from football_longform_v2.models import TemporalLocator


class TemporalLocatorTest(unittest.TestCase):
    def make_inputs(self) -> tuple[torch.Tensor, ...]:
        torch.manual_seed(7)
        return (
            torch.randn(2, 65, 32),
            torch.randn(2, 65, 16),
            torch.arange(65).float().unsqueeze(0).repeat(2, 1) / 5.0,
        )

    def test_rgb_only_forward_backward(self) -> None:
        model = TemporalLocator(32, 16, hidden_dim=32, levels=3, blocks_per_level=1)
        context, motion, timestamps = self.make_inputs()
        output = model(context, motion, timestamps)
        self.assertEqual(output["rgb_logits"].shape, (2, 65, 3))
        self.assertTrue(torch.equal(output["rgb_logits"], output["logits"]))
        output["logits"].square().mean().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_save_head_uses_past_shot_peak_features(self) -> None:
        model = TemporalLocator(
            32, 16, hidden_dim=32, levels=3, blocks_per_level=1,
            timeline_hz=5.0, save_lookback_seconds=3.0,
        )
        context, motion, timestamps = self.make_inputs()
        output = model(context, motion, timestamps)
        self.assertEqual(output["class_logits"].shape, (2, 65, 6))
        self.assertEqual(output["state_logits"].shape, (2, 65, 6))
        shot_probability = output["base_class_logits"][..., 0].sigmoid()
        expected = torch.stack([
            shot_probability[:, max(0, index - 15):index + 1].max(dim=1).values
            for index in range(65)
        ], dim=1)
        self.assertTrue(torch.allclose(output["shot_condition_max"], expected, atol=1e-6))
        self.assertGreaterEqual(float(output["shot_condition_lag_seconds"].min()), 0.0)
        self.assertLessEqual(float(output["shot_condition_lag_seconds"].max()), 3.0)
        output["class_logits"].square().mean().backward()
        self.assertTrue(any(
            parameter.grad is not None for parameter in model.save_conditioner.parameters()
        ))

    def test_entity_residual_is_bounded_and_optional(self) -> None:
        model = TemporalLocator(
            32, 16, hidden_dim=32, levels=3, blocks_per_level=1,
            entity_dim=5, max_entity_residual_logit=0.15,
        )
        model.eval()
        context, motion, timestamps = self.make_inputs()
        rgb = model(context, motion, timestamps)
        corrupted = model(context, motion, timestamps, entity=1000.0 * torch.randn(2, 65, 5))
        self.assertTrue(torch.allclose(rgb["rgb_logits"], corrupted["rgb_logits"]))
        self.assertLessEqual(float(corrupted["entity_delta"].detach().abs().max()), 0.150001)


if __name__ == "__main__":
    unittest.main()

