from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from football_object_spatial_aux import ObjectSpatialAuxHead
from football_events.ball_goal_relation import (
    RELATION_NAMES,
    ball_goal_relation_aux_loss,
)
from football_events.mechanism_a import (
    gradient_alignment,
    shared_lora_named_parameters,
    validate_mechanism_a_config,
)
from football_events.tracked_ball_teacher import TrackedBallTargetProvider
from train_football_events import (
    VideoEventClassifier,
    resolve_evaluation_candidate_time_mode,
)


def mechanism_config(*, variant: str, object_weight: float, relation_weight: float = 0.0) -> dict:
    return {
        "model": {
            "mechanism_a": {
                "enabled": True,
                "variant": variant,
                "gradient_diagnostics": {"enabled": variant == "object_aux"},
            },
            "controlled_online_train_scope": "mechanism_a",
            "view_fusion": "single",
            "freeze_loaded_backbone": False,
            "lora": {"enabled": True, "train_norm": False},
            "object_spatial_aux": {
                "enabled": True,
                "representation_only": True,
                "teacher_format": "tracked_ball_npz_v1",
            },
        },
        "eval": {"candidate_time_mode": "auto"},
        "video": {"hflip_prob": 0.0},
        "cache": {"enabled": False},
        "train": {
            "object_heatmap_loss_weight": object_weight,
            "ball_goal_relation_loss_weight": relation_weight,
        },
    }


class MechanismAConfigTest(unittest.TestCase):
    def test_paired_variants_are_accepted(self) -> None:
        validate_mechanism_a_config(
            mechanism_config(variant="control", object_weight=0.0)
        )
        validate_mechanism_a_config(
            mechanism_config(variant="object_aux", object_weight=0.25)
        )

    def test_relation_variant_is_accepted_only_with_composite_teacher(self) -> None:
        cfg = mechanism_config(
            variant="relation_aux", object_weight=0.25, relation_weight=0.2
        )
        cfg["model"]["mechanism_a"]["gradient_diagnostics"]["enabled"] = True
        cfg["model"]["object_spatial_aux"][
            "teacher_format"
        ] = "tracked_ball_goal_relation_v1"
        cfg["model"]["object_spatial_aux"]["goal_teacher_index_root"] = "/goal"
        cfg["model"]["ball_goal_relation_aux"] = {"enabled": True}
        validate_mechanism_a_config(cfg)

    def test_direct_residual_route_is_rejected(self) -> None:
        cfg = mechanism_config(variant="object_aux", object_weight=0.25)
        cfg["model"]["object_spatial_aux"]["representation_only"] = False
        with self.assertRaisesRegex(ValueError, "representation_only"):
            validate_mechanism_a_config(cfg)

    def test_confounding_loss_is_rejected(self) -> None:
        cfg = mechanism_config(variant="object_aux", object_weight=0.25)
        cfg["train"]["frame_det_loss_weight"] = 0.2
        with self.assertRaisesRegex(ValueError, "frame_det_loss_weight"):
            validate_mechanism_a_config(cfg)

    def test_candidate_time_auto_ignores_detection_auxiliary_targets(self) -> None:
        cfg = mechanism_config(variant="object_aux", object_weight=0.25)
        self.assertEqual(
            resolve_evaluation_candidate_time_mode(cfg), "window_center"
        )
        cfg["train"]["ball_goal_relation_loss_weight"] = 0.2
        self.assertEqual(
            resolve_evaluation_candidate_time_mode(cfg), "window_center"
        )
        cfg["train"]["frame_det_loss_weight"] = 0.1
        self.assertEqual(
            resolve_evaluation_candidate_time_mode(cfg), "frame_peak"
        )
        cfg["eval"]["candidate_time_mode"] = "window_center"
        self.assertEqual(
            resolve_evaluation_candidate_time_mode(cfg), "window_center"
        )


class TrackedBallTargetProviderTest(unittest.TestCase):
    def test_quality_weighted_ball_target_and_unknown_goal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez(
                root / "v1.npz",
                timestamp_sec=np.asarray([1.0, 2.0], dtype=np.float64),
                bbox_xyxy_norm=np.asarray(
                    [[0.4, 0.4, 0.6, 0.6], [0.1, 0.1, 0.2, 0.2]],
                    dtype=np.float32,
                ),
                confidence=np.asarray([0.9, 0.9], dtype=np.float32),
                quality_weight=np.asarray([0.75, 0.5], dtype=np.float32),
                flags=np.asarray([1, 0], dtype=np.uint8),
                source_code=np.asarray([1, 1], dtype=np.uint8),
            )
            provider = TrackedBallTargetProvider(
                root,
                image_size=(80, 160),
                patch_size=16,
                max_frame_gap_sec=0.1,
            )
            targets, masks = provider.targets("v1", [1.0, 2.0, 3.0])

            self.assertEqual(tuple(targets.shape), (3, 50, 2))
            ball_peak = int(targets[0, :, 0].argmax())
            self.assertEqual((ball_peak // 10, ball_peak % 10), (2, 4))
            self.assertTrue(torch.allclose(masks[0, :, 0], torch.full((50,), 0.75)))
            self.assertEqual(float(masks[1:].sum()), 0.0)
            self.assertEqual(float(targets[..., 1].sum()), 0.0)
            self.assertEqual(float(masks[..., 1].sum()), 0.0)

            _, _, metadata = provider.targets_with_metadata(
                "v1", [1.0, 2.0, 3.0]
            )
            self.assertAlmostEqual(float(metadata["confidence"][0]), 0.9)
            self.assertAlmostEqual(float(metadata["quality"][0]), 0.75)
            self.assertEqual(float(metadata["valid"].sum()), 1.0)
            self.assertTrue(
                torch.allclose(metadata["bbox_xyxy_norm"][0], torch.tensor([0.4, 0.4, 0.6, 0.6]))
            )

    def test_time_shift_control_uses_a_different_frame_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez(
                root / "v1.npz",
                timestamp_sec=np.asarray([1.0, 31.0], dtype=np.float64),
                bbox_xyxy_norm=np.asarray(
                    [[0.1, 0.4, 0.2, 0.6], [0.8, 0.4, 0.9, 0.6]],
                    dtype=np.float32,
                ),
                confidence=np.ones(2, dtype=np.float32),
                quality_weight=np.ones(2, dtype=np.float32),
                flags=np.ones(2, dtype=np.uint8),
            )
            provider = TrackedBallTargetProvider(
                root,
                image_size=(80, 160),
                patch_size=16,
                max_frame_gap_sec=0.1,
                teacher_time_offset_sec=30.0,
            )
            targets, _ = provider.targets("v1", [1.0])
            peak = int(targets[0, :, 0].argmax())
            self.assertEqual((peak // 10, peak % 10), (2, 8))

    def test_missing_video_is_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = TrackedBallTargetProvider(
                tmp, image_size=(80, 160), patch_size=16
            )
            targets, masks = provider.targets("missing", [0.0, 1.0])
            self.assertEqual(float(targets.sum()), 0.0)
            self.assertEqual(float(masks.sum()), 0.0)


class MechanismAGradientTest(unittest.TestCase):
    def test_alignment_uses_shared_parameter_without_populating_grad(self) -> None:
        parameter = nn.Parameter(torch.tensor([1.0, 2.0]))
        event_loss = parameter.square().sum()
        object_loss = parameter.sum()
        metrics = gradient_alignment(
            event_loss,
            object_loss,
            [("backbone.block.lora_a", parameter)],
            object_weight=0.25,
        )
        self.assertIsNone(parameter.grad)
        self.assertGreater(metrics["mechanism_a_event_grad_norm"], 0.0)
        self.assertGreater(metrics["mechanism_a_object_grad_norm"], 0.0)
        self.assertGreater(metrics["mechanism_a_grad_cosine"], 0.0)
        self.assertAlmostEqual(
            metrics["mechanism_a_weighted_object_to_event_grad_ratio"],
            0.25 * metrics["mechanism_a_object_to_event_grad_ratio"],
        )

    def test_shared_parameter_selector_excludes_heads(self) -> None:
        class TinyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = nn.Module()
                self.backbone.adapter = nn.Module()
                self.backbone.adapter.register_parameter(
                    "lora_a", nn.Parameter(torch.ones(1))
                )
                self.head = nn.Linear(1, 1)

        selected = shared_lora_named_parameters(TinyModel())
        self.assertEqual([name for name, _ in selected], ["backbone.adapter.lora_a"])


class MechanismAForwardTest(unittest.TestCase):
    def test_object_residual_cannot_change_representation_only_logits(self) -> None:
        class DummyBackbone(nn.Module):
            num_features = 4

            def __init__(self) -> None:
                super().__init__()
                self.placeholder = nn.Parameter(torch.zeros(()))

        torch.manual_seed(9)
        model = VideoEventClassifier(
            backbone=DummyBackbone(),
            frame_feature_dim=8,
            hidden_dim=8,
            num_labels=3,
            fusion="cls_transformer",
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            max_frames=2,
            object_spatial_aux_enabled=True,
            object_spatial_aux_representation_only=True,
        ).eval()

        def fake_encode(
            self: VideoEventClassifier, inputs: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            batch, frames = inputs.shape[:2]
            return (
                torch.ones(batch, frames, 8),
                torch.ones(batch, frames, 6, 4),
            )

        model.encode_frames_with_patch_tokens = types.MethodType(fake_encode, model)
        assert model.object_spatial_aux is not None
        with torch.no_grad():
            model.object_spatial_aux.residual_head[-1].bias.fill_(1.0)
            inputs = torch.zeros(1, 2, 3, 4, 4)
            protected = model(inputs, return_aux=True)
            self.assertTrue(
                torch.equal(protected["logits"], protected["global_logits"])
            )
            self.assertNotIn("object_spatial_residual", protected)

            model.object_spatial_aux_representation_only = False
            routed = model(inputs, return_aux=True)
            self.assertGreater(
                float(routed["object_spatial_residual"].abs().max()), 0.0
            )
            self.assertFalse(torch.equal(routed["logits"], routed["global_logits"]))

    def test_relation_auxiliary_reads_frame_tokens_without_routing_logits(self) -> None:
        class DummyBackbone(nn.Module):
            num_features = 4

            def __init__(self) -> None:
                super().__init__()
                self.placeholder = nn.Parameter(torch.zeros(()))

        model = VideoEventClassifier(
            backbone=DummyBackbone(),
            frame_feature_dim=8,
            hidden_dim=8,
            num_labels=3,
            fusion="cls_transformer",
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            max_frames=2,
            object_spatial_aux_enabled=True,
            object_spatial_aux_representation_only=True,
            ball_goal_relation_aux_enabled=True,
            ball_goal_relation_aux_targets=len(RELATION_NAMES),
            ball_goal_relation_aux_grid_size=(2, 3),
        ).eval()

        def fake_encode(
            self: VideoEventClassifier, inputs: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            batch, frames = inputs.shape[:2]
            return torch.ones(batch, frames, 8), torch.randn(batch, frames, 6, 4)

        model.encode_frames_with_patch_tokens = types.MethodType(fake_encode, model)
        outputs = model(torch.zeros(1, 2, 3, 4, 4), return_aux=True)
        self.assertTrue(torch.equal(outputs["logits"], outputs["global_logits"]))
        self.assertEqual(
            tuple(outputs["ball_goal_relation_logits"].shape), (1, 2, len(RELATION_NAMES))
        )
        batch = {
            "ball_goal_relation_targets": torch.zeros(1, 2, len(RELATION_NAMES)),
            "ball_goal_relation_masks": torch.ones(1, 2, len(RELATION_NAMES)),
            "ball_context_masks": torch.ones(1, 2),
        }
        cfg = {"model": {"ball_goal_relation_aux": {"context_loss_weight": 0.25}}}
        loss, components = ball_goal_relation_aux_loss(
            outputs, batch, cfg, torch.device("cpu")
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss), 0.0)
        self.assertIn("ball_candidate_entropy", components)


if __name__ == "__main__":
    unittest.main()
