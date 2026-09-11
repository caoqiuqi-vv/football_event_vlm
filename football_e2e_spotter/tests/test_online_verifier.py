from __future__ import annotations

import unittest

import torch

from football_e2e_spotter.online_verifier import (
    CandidateContextSetVerifier,
    ShotTarget,
    decode_shot_slots,
    shot_set_verifier_loss,
)


class OnlineVerifierTest(unittest.TestCase):
    def test_forward_and_loss_preserve_two_nearby_shots(self) -> None:
        model = CandidateContextSetVerifier(
            6, hidden_dim=32, num_slots=4,
            encoder_layers=1, decoder_layers=1, heads=4,
        )
        output = model(
            torch.randn(2, 7, 6),
            torch.rand(2, 7),
            torch.ones(2, 7, dtype=torch.bool),
        )
        loss = shot_set_verifier_loss(
            output,
            [[ShotTarget(0.40), ShotTarget(0.43)], []],
        )
        loss["loss"].backward()
        self.assertEqual(tuple(output["class_logits"].shape), (2, 7, 2))
        self.assertEqual(int(loss["matched_instances"]), 2)

    def test_candidate_anchors_inherit_stage1_times(self) -> None:
        model = CandidateContextSetVerifier(
            4, hidden_dim=32, num_slots=2,
            encoder_layers=1, decoder_layers=1, heads=4, dropout=0.0,
        )
        torch.nn.init.zeros_(model.time_head.weight)
        torch.nn.init.zeros_(model.time_head.bias)
        candidate_times = torch.tensor([[0.20, 0.70, -0.20, 1.20]])
        mask = torch.tensor([[True, True, True, True]])
        output = model(torch.randn(1, 4, 4), candidate_times, mask)
        self.assertTrue(torch.allclose(output["time_normalized"], candidate_times))
        expected_output_mask = torch.tensor([[True, True, False, False]])
        self.assertTrue(torch.equal(output["slot_mask"], expected_output_mask))

    def test_residual_score_initializes_as_exact_stage1_identity(self) -> None:
        model = CandidateContextSetVerifier(
            4, hidden_dim=32, encoder_layers=1, decoder_layers=1,
            heads=4, dropout=0.0,
        )
        baseline = torch.tensor([[-2.0, 0.5, 1.5]])
        output = model(
            torch.randn(1, 3, 4), torch.tensor([[0.1, 0.4, 0.8]]),
            torch.ones(1, 3, dtype=torch.bool), baseline_logits=baseline,
        )
        self.assertTrue(torch.allclose(output["ranking_logits"], baseline))

    def test_decoder_does_not_apply_temporal_nms(self) -> None:
        output = {
            "class_logits": torch.tensor([[[8.0, -2.0], [7.0, -2.0]]]),
            "quality_logits": torch.tensor([[8.0, 8.0]]),
            "time_normalized": torch.tensor([[0.40, 0.42]]),
            "log_sigma": torch.zeros(1, 2),
        }
        predictions = decode_shot_slots(
            output, core_start_sec=0.0, core_duration_sec=30.0, threshold=0.1
        )
        self.assertEqual(len(predictions), 2)
        self.assertAlmostEqual(float(predictions[0]["time_sec"]), 12.0, places=4)
        self.assertAlmostEqual(float(predictions[1]["time_sec"]), 12.6, places=4)

    def test_masked_candidate_is_supported(self) -> None:
        model = CandidateContextSetVerifier(
            4, hidden_dim=32, num_slots=2,
            encoder_layers=1, decoder_layers=1, heads=4,
        )
        output = model(
            torch.randn(1, 3, 4),
            torch.tensor([[0.1, 0.5, 0.0]]),
            torch.tensor([[True, True, False]]),
        )
        self.assertTrue(torch.isfinite(output["class_logits"]).all())


if __name__ == "__main__":
    unittest.main()
