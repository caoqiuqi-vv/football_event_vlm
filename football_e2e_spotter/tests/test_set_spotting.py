from __future__ import annotations

import unittest

import torch
from torch import nn

from football_e2e_spotter.set_spotting import EventInstance, SET_LABELS, TemporalSetSpotter, decode_slots, one_to_one_metrics, set_spotting_loss
from football_e2e_spotter.verifier import DinoVerifier


class TinyPatchBackbone(nn.Module):
    def forward_features(self, inputs: torch.Tensor):
        pooled = inputs.mean(1).unfold(1, 16, 16).unfold(2, 16, 16).mean((-1, -2))
        return {"x_norm_patchtokens": pooled.flatten(1, 2).unsqueeze(-1).repeat(1, 1, 8)}


class SetSpottingTest(unittest.TestCase):
    def test_nearby_same_class_slots_are_preserved(self) -> None:
        outputs = {"class_logits": torch.tensor([[[9., 0, 0, 0, 0, -3], [8., 0, 0, 0, 0, -3]]]), "time_normalized": torch.tensor([[.10, .12]]), "quality_logits": torch.tensor([[8., 8.]]), "log_sigma": torch.zeros(1, 2)}
        predictions = decode_slots(outputs, core_start_seconds=0, core_seconds=30, thresholds={label: .1 for label in SET_LABELS})
        self.assertEqual(len(predictions), 2)
        metric = one_to_one_metrics(predictions, [EventInstance(0, .10), EventInstance(0, .12)], core_seconds=30, tolerance_seconds=.7)
        self.assertEqual(metric["shot"]["tp"], 2)

    def test_cross_class_same_time_is_independent(self) -> None:
        predictions = [
            type("P", (), {"label": "shot", "time_sec": 5., "score": .9})(),
            type("P", (), {"label": "save", "time_sec": 5., "score": .8})(),
        ]
        metric = one_to_one_metrics(predictions, [EventInstance(0, 5.), EventInstance(1, 5.)], tolerance_seconds=.1)
        self.assertEqual(metric["shot"]["tp"], 1)
        self.assertEqual(metric["save"]["tp"], 1)

    def test_stage1_forward_and_matching_loss(self) -> None:
        model = TemporalSetSpotter(sample_fps=2, core_seconds=2, context_seconds=1, num_slots=4, mel_bins=4, audio_dim=8, hidden_dim=32, decoder_heads=4, decoder_layers=1, imagenet_initialization=False)
        output = model(torch.randn(1, 8, 3, 64, 96), torch.randn(1, 8, 4))
        loss = set_spotting_loss(output, [[EventInstance(0, .2), EventInstance(1, .25)]])
        loss["loss"].backward()
        self.assertEqual(tuple(output["class_logits"].shape), (1, 4, 6))
        self.assertEqual(int(loss["matched_instances"]), 2)

    def test_verifier_contract_and_output(self) -> None:
        verifier = DinoVerifier(TinyPatchBackbone(), patch_dim=8, candidate_dim=8, audio_dim=4, hidden_dim=32, minimum_height=32, minimum_width=48, fusion_layers=1)
        output = verifier(torch.randn(1, 25, 3, 32, 48), torch.randn(1, 8), torch.randn(1, 6, 32), torch.randn(1, 4, 4))
        self.assertEqual(tuple(output["class_logits"].shape), (1, 6))
        self.assertEqual(tuple(output["crop_boxes"].shape), (1, 2, 3))


if __name__ == "__main__":
    unittest.main()
