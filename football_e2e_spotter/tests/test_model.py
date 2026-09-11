from __future__ import annotations

import unittest

import torch

from football_e2e_spotter.model import CausalShotContext, FootballE2ESpotter


class ModelTest(unittest.TestCase):
    def test_shot_context_is_causal(self) -> None:
        module = CausalShotContext(history_steps=4)
        baseline = torch.zeros(1, 10, 1)
        changed = baseline.clone()
        changed[:, 6:] = 1.0
        first = module(baseline)
        second = module(changed)
        torch.testing.assert_close(first[:, :6], second[:, :6])

    def test_end_to_end_forward_backward(self) -> None:
        model = FootballE2ESpotter(
            sample_fps=2.0,
            mel_bins=16,
            audio_dim=24,
            hidden_dim=32,
            imagenet_initialization=False,
            refinement_stages=("block3", "block4"),
        )
        frames = torch.randn(1, 6, 3, 64, 96)
        audio = torch.randn(1, 6, 16)
        valid = torch.ones(1, 6, dtype=torch.bool)
        valid[:, -1] = False
        output = model(frames, audio, valid)
        self.assertEqual(tuple(output["class_logits"].shape), (1, 6, 4))
        self.assertEqual(tuple(output["family_logits"].shape), (1, 6, 2))
        self.assertEqual(tuple(output["offsets"].shape), (1, 6, 4))
        self.assertTrue(bool((output["class_logits"][:, -1] == -20.0).all()))
        output["class_logits"][:, :-1].mean().backward()
        self.assertIsNotNone(model.visual.stem[0].weight.grad)
        self.assertGreater(float(model.visual.stem[0].weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()

