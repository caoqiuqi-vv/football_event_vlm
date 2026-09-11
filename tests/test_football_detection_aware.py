from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from football_detection_aware import ROI_META_DIM, ROIProposal, RobustClipCropper, invalid_roi
from train_football_events import (
    VideoEventClassifier,
    checkpoint_best_selection_score,
    load_config,
    resize_temporal_pos_embed,
    resolve_runtime_topology,
)


ROOT = Path(__file__).resolve().parents[1]


class FootballDetectionAwareTest(unittest.TestCase):
    def test_resume_preserves_negative_checkpoint_selection_score(self) -> None:
        checkpoint = {
            "best_macro_f1": -0.3886,
            "metrics": {"tuned": {"macro_f1": 0.6705}},
        }
        self.assertAlmostEqual(checkpoint_best_selection_score(checkpoint), -0.3886)

    def test_legacy_resume_falls_back_to_tuned_macro_f1(self) -> None:
        checkpoint = {"metrics": {"tuned": {"macro_f1": 0.6705}}}
        self.assertAlmostEqual(checkpoint_best_selection_score(checkpoint), 0.6705)

    def test_roi_size_is_integer_multiple_of_classifier_input(self) -> None:
        cropper = RobustClipCropper({"index_root": "/tmp/unused"})
        roi, area, reason = cropper._fit_roi(
            [[100, 120, 520, 360]],
            width=1280,
            height=720,
            target_size=(384, 640),
        )
        self.assertEqual(reason, "ok")
        self.assertIsNotNone(roi)
        assert roi is not None
        self.assertEqual(roi[2] - roi[0], 640)
        self.assertEqual(roi[3] - roi[1], 384)
        self.assertAlmostEqual(area, 640 * 384 / (1280 * 720))

    def test_roi_metadata_has_stable_gate_dimension(self) -> None:
        proposal = ROIProposal(
            bbox=(10, 20, 110, 80),
            valid=True,
            mode="goal_ball",
            roi_confidence=0.8,
            goal_score=0.9,
            ball_score=0.7,
            person_support=0.5,
            area_ratio=0.3,
        )
        metadata = proposal.meta_vector()
        self.assertEqual(tuple(metadata.shape), (ROI_META_DIM,))
        self.assertEqual(float(metadata[0]), 1.0)
        self.assertAlmostEqual(float(metadata[1]), 0.8, places=6)
        self.assertEqual(float(metadata[7]), 1.0)
        self.assertEqual(float(metadata[11]), 1.0)
        self.assertEqual(float(metadata[11:].sum()), 1.0)

    def test_invalid_roi_cannot_change_dual_view_prediction(self) -> None:
        model = VideoEventClassifier(
            backbone=None,
            frame_feature_dim=4,
            hidden_dim=8,
            num_labels=3,
            fusion="mean",
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            max_frames=2,
            view_fusion="dual_gate",
            roi_meta_dim=ROI_META_DIM,
        ).eval()
        global_features = torch.randn(2, 2, 4)
        local_features = torch.randn(2, 2, 4)
        invalid_metadata = torch.stack([invalid_roi("noise").meta_vector()] * 2)
        with torch.no_grad():
            outputs = model(
                global_features,
                roi_inputs=local_features,
                roi_meta=invalid_metadata,
                roi_valid=torch.zeros(2),
                return_aux=True,
            )
        torch.testing.assert_close(outputs["logits"], outputs["global_logits"])
        torch.testing.assert_close(outputs["roi_gate"], torch.zeros_like(outputs["roi_gate"]))

        valid = ROIProposal(
            bbox=(0, 0, 100, 60),
            valid=True,
            mode="ball_players",
            roi_confidence=1.0,
            ball_score=0.9,
            area_ratio=0.2,
            fallback_reason="ok",
        )
        valid_metadata = torch.stack([valid.meta_vector()] * 2)
        with torch.no_grad():
            valid_outputs = model(
                global_features,
                roi_inputs=local_features,
                roi_meta=valid_metadata,
                roi_valid=torch.ones(2),
                return_aux=True,
            )
        self.assertTrue(torch.all(valid_outputs["roi_gate"] > 0))
        self.assertTrue(torch.all(valid_outputs["roi_gate"] <= 1))
        lower = torch.minimum(valid_outputs["global_logits"], valid_outputs["local_logits"])
        upper = torch.maximum(valid_outputs["global_logits"], valid_outputs["local_logits"])
        self.assertTrue(torch.all(valid_outputs["logits"] >= lower))
        self.assertTrue(torch.all(valid_outputs["logits"] <= upper))

    def test_dynamic_clip_falls_back_to_fixed_window_roi(self) -> None:
        cropper = RobustClipCropper(
            {
                "index_root": "/tmp/unused",
                "temporal_mode": "dynamic",
                "temporal_smoothing_window_sec": 0.0,
                "temporal_max_hold_sec": 0.0,
                "dynamic_fallback_to_clip": True,
            }
        )
        fixed = ROIProposal(
            bbox=(20, 20, 120, 80),
            valid=True,
            mode="ball_players",
            roi_confidence=0.7,
            ball_score=0.7,
            area_ratio=0.075,
            fallback_reason="ok",
        )
        with patch.object(
            cropper,
            "get_window_roi",
            side_effect=[fixed, invalid_roi("miss"), invalid_roi("miss")],
        ):
            proposals, aggregate = cropper.get_clip_rois(
                "video",
                0.0,
                10.0,
                [1.0, 9.0],
                width=400,
                height=200,
                target_size=(60, 100),
            )
        self.assertTrue(aggregate.valid)
        self.assertEqual(aggregate.fallback_reason, "dynamic_fixed_fallback")
        for proposal in proposals:
            self.assertTrue(proposal.valid)
            self.assertEqual(proposal.bbox, fixed.bbox)
            self.assertEqual(proposal.fallback_reason, "dynamic_fixed_fallback")

    def test_dynamic_augmentation_invalid_uses_fixed_fallback(self) -> None:
        cropper = RobustClipCropper({"index_root": "/tmp/unused", "temporal_mode": "dynamic"})
        fixed = ROIProposal(
            bbox=(20, 20, 120, 80),
            valid=True,
            mode="ball_players",
            roi_confidence=0.7,
            ball_score=0.7,
            area_ratio=0.075,
            fallback_reason="ok",
        )
        proposals, aggregate = cropper.augment_sequence(
            [invalid_roi("miss"), invalid_roi("miss")],
            400,
            200,
            {"invalid_prob": 1.0},
            (60, 100),
            fixed_fallback=fixed,
        )
        self.assertTrue(aggregate.valid)
        self.assertEqual(aggregate.fallback_reason, "train_dynamic_fixed_fallback")
        self.assertTrue(all(proposal.bbox == fixed.bbox for proposal in proposals))

    def test_dynamic_roi_smoothing_moves_holds_then_expires(self) -> None:
        cropper = RobustClipCropper(
            {
                "index_root": "/tmp/unused",
                "temporal_mode": "dynamic",
                "temporal_smoothing_window_sec": 0.0,
                "temporal_max_hold_sec": 1.1,
                "temporal_confidence_decay_sec": 1.0,
                "temporal_max_center_jump_ratio": 0.0,
            }
        )
        first = ROIProposal(
            bbox=(20, 20, 120, 80),
            valid=True,
            mode="ball_players",
            roi_confidence=0.8,
            ball_score=0.8,
            area_ratio=0.15,
            fallback_reason="ok",
        )
        second = ROIProposal(
            bbox=(120, 20, 220, 80),
            valid=True,
            mode="ball_players",
            roi_confidence=0.8,
            ball_score=0.8,
            area_ratio=0.15,
            fallback_reason="ok",
        )
        proposals = [first, second, invalid_roi("miss"), invalid_roi("miss")]
        smoothed = cropper.smooth_temporal_proposals(
            proposals,
            [0.0, 1.0, 2.0, 4.0],
            width=400,
            height=200,
            target_size=(60, 100),
        )
        self.assertTrue(smoothed[0].valid)
        self.assertTrue(smoothed[1].valid)
        self.assertTrue(smoothed[2].valid)
        self.assertEqual(smoothed[2].fallback_reason, "temporal_hold")
        self.assertLess(smoothed[2].roi_confidence, second.roi_confidence)
        self.assertFalse(smoothed[3].valid)
        self.assertEqual(smoothed[3].fallback_reason, "temporal_gap_exceeded")
        for proposal in smoothed[:3]:
            assert proposal.bbox is not None
            self.assertEqual(proposal.bbox[2] - proposal.bbox[0], 100)
            self.assertEqual(proposal.bbox[3] - proposal.bbox[1], 60)

    def test_dynamic_sequence_augmentation_applies_shared_quantized_scale(self) -> None:
        cropper = RobustClipCropper({"index_root": "/tmp/unused", "temporal_mode": "dynamic"})
        proposal = ROIProposal(
            bbox=(20, 20, 120, 80),
            valid=True,
            mode="ball_players",
            roi_confidence=0.8,
            ball_score=0.8,
            area_ratio=0.0625,
            fallback_reason="ok",
        )
        cfg = {
            "invalid_prob": 0.0,
            "ball_drop_prob": 0.0,
            "goal_drop_prob": 0.0,
            "isolated_false_positive_prob": 0.0,
            "center_jitter": 0.0,
            "scale_min": 2.0,
            "scale_max": 2.0,
        }
        augmented, aggregate = cropper.augment_sequence(
            [proposal, proposal], 400, 240, cfg, (60, 100)
        )
        self.assertTrue(aggregate.valid)
        for item in augmented:
            assert item.bbox is not None
            self.assertEqual(item.bbox[2] - item.bbox[0], 200)
            self.assertEqual(item.bbox[3] - item.bbox[1], 120)
            self.assertAlmostEqual(item.area_ratio, 0.25)

    def test_feature_quality_fusion_starts_from_global_and_masks_invalid_roi(self) -> None:
        model = VideoEventClassifier(
            backbone=None,
            frame_feature_dim=4,
            hidden_dim=8,
            num_labels=3,
            fusion="mean",
            num_layers=1,
            num_heads=2,
            dropout=0.0,
            max_frames=4,
            view_fusion="dual_feature_quality",
            roi_meta_dim=ROI_META_DIM,
        ).eval()
        global_features = torch.randn(2, 4, 4)
        local_features = torch.randn(2, 4, 4)
        roi_meta = torch.zeros(2, ROI_META_DIM)
        roi_meta[:, :2] = 1.0
        roi_frame_meta = roi_meta[:, None, :].expand(-1, 4, -1).clone()
        with torch.no_grad():
            outputs = model(
                global_features,
                roi_inputs=local_features,
                roi_meta=roi_meta,
                roi_valid=torch.ones(2),
                roi_frame_meta=roi_frame_meta,
                roi_frame_valid=torch.ones(2, 4),
                return_aux=True,
            )
        torch.testing.assert_close(outputs["logits"], outputs["global_logits"])
        self.assertEqual(tuple(outputs["roi_frame_quality"].shape), (2, 4))
        self.assertEqual(tuple(outputs["roi_quality_logits"].shape), (2, 3))

        with torch.no_grad():
            invalid_outputs = model(
                global_features,
                roi_inputs=local_features,
                roi_meta=torch.zeros_like(roi_meta),
                roi_valid=torch.zeros(2),
                roi_frame_meta=torch.zeros_like(roi_frame_meta),
                roi_frame_valid=torch.zeros(2, 4),
                return_aux=True,
            )
        torch.testing.assert_close(invalid_outputs["logits"], invalid_outputs["global_logits"])
        torch.testing.assert_close(
            invalid_outputs["roi_gate"], torch.zeros_like(invalid_outputs["roi_gate"])
        )

    def test_temporal_pos_embed_resizes_when_frame_count_changes(self) -> None:
        source = torch.arange(1 * 33 * 4, dtype=torch.float32).reshape(1, 33, 4)
        resized = resize_temporal_pos_embed(source, (1, 17, 4))
        self.assertIsNotNone(resized)
        assert resized is not None
        self.assertEqual(tuple(resized.shape), (1, 17, 4))
        torch.testing.assert_close(resized[:, 0, :], source[:, 0, :])

    def test_clip_level_anchor_dropout_changes_metadata_once(self) -> None:
        cropper = RobustClipCropper({"index_root": "/tmp/unused"})
        proposal = ROIProposal(
            bbox=(10, 20, 110, 80),
            valid=True,
            mode="goal_ball",
            roi_confidence=0.9,
            goal_score=0.9,
            ball_score=0.8,
            person_support=0.4,
            area_ratio=0.1,
            fallback_reason="ok",
        )
        cfg = {
            "invalid_prob": 0.0,
            "ball_drop_prob": 1.0,
            "goal_drop_prob": 0.0,
            "isolated_false_positive_prob": 0.0,
            "center_jitter": 0.0,
            "scale_min": 1.0,
            "scale_max": 1.0,
        }
        with patch("football_detection_aware.random.random", side_effect=[0.9, 0.0, 0.9, 0.9]):
            augmented = cropper.augment(proposal, 200, 120, cfg, (60, 100))
        self.assertTrue(augmented.valid)
        self.assertEqual(augmented.mode, "goal_players")
        self.assertEqual(augmented.ball_score, 0.0)
        self.assertEqual(augmented.fallback_reason, "train_ball_dropout")
        self.assertEqual(augmented.bbox[2] - augmented.bbox[0], 100)
        self.assertEqual(augmented.bbox[3] - augmented.bbox[1], 60)

    def test_experiments_keep_validation_sampling_and_threshold_comparable(self) -> None:
        paths = [
            ROOT / "configs/football/dinov3_vitl16_detector47_global_control.yaml",
            ROOT / "configs/football/dinov3_vitl16_robust_dual_exp2.yaml",
            ROOT / "configs/football/dinov3_vitl16_robust_dual_16f_exp4.yaml",
        ]
        configs = [load_config(str(path), []) for path in paths]
        for cfg in configs:
            self.assertEqual(cfg.seed, 42)
            self.assertTrue(cfg.deterministic)
            self.assertFalse(cfg.data.long_video.negative_require_all_labels)
            self.assertEqual(float(cfg.data.long_video.negative_ratio_by_split.val), 2.0)
            self.assertEqual(list(cfg.train.pos_weight), [1.0, 2.5, 4.0])
            self.assertEqual(float(cfg.eval.threshold), 0.5)
        self.assertEqual(float(configs[0].data.long_video.negative_ratio_by_split.train), 2.0)
        self.assertEqual(float(configs[1].data.long_video.negative_ratio_by_split.train), 2.0)
        self.assertEqual(float(configs[2].data.long_video.negative_ratio), 5.0)
        self.assertEqual(float(configs[2].data.long_video.negative_ratio_by_split.train), 5.0)
        self.assertEqual(int(configs[0].train.batch_size), 1)
        self.assertEqual(int(configs[0].train.grad_accum_steps), 24)
        self.assertEqual(int(configs[1].train.batch_size), 1)
        self.assertEqual(int(configs[1].train.grad_accum_steps), 24)
        self.assertEqual(int(configs[0].train.epochs), 6)
        self.assertEqual(int(configs[1].train.epochs), 6)
        self.assertEqual(float(configs[0].train.lr), 0.0002)
        self.assertEqual(float(configs[1].train.lr), 0.0002)
        self.assertEqual(int(configs[2].eval.per_gpu_batch_size), 1)
        self.assertEqual(list(configs[2].gpu_ids), [0, 1])
        topology = resolve_runtime_topology(configs[2], torch.device("cuda"))
        self.assertEqual(int(topology["train_global_batch_size"]), 8)
        self.assertEqual(int(topology["eval_global_batch_size"]), 2)
        self.assertAlmostEqual(float(topology["lr_global"]), 0.0001)
        self.assertEqual(configs[0].model.view_fusion, "single")
        self.assertEqual(configs[1].model.view_fusion, "dual_gate")
        self.assertEqual(configs[2].model.view_fusion, "dual_gate")
        self.assertEqual(list(configs[0].video.image_size), list(configs[1].spatial_crop.global_image_size))
        self.assertEqual(list(configs[2].spatial_crop.global_image_size), [512, 896])
        self.assertEqual(list(configs[1].video.image_size), list(configs[2].video.image_size))
        self.assertEqual(int(configs[1].video.num_frames), 32)
        self.assertEqual(int(configs[2].video.num_frames), 16)
        self.assertEqual(str(configs[2].spatial_crop.output_view), "dual")
        self.assertEqual(str(configs[2].spatial_crop.mode), "robust_detector_aware")
        self.assertEqual(float(configs[1].train.local_loss_weight), 0.30)
        self.assertEqual(float(configs[2].train.local_loss_weight), 0.30)
        for cfg in configs[1:]:
            self.assertEqual(float(cfg.spatial_crop.noise_augmentation.invalid_prob), 0.15)
            self.assertEqual(float(cfg.spatial_crop.noise_augmentation.ball_drop_prob), 0.10)
            self.assertEqual(float(cfg.spatial_crop.noise_augmentation.goal_drop_prob), 0.10)
            self.assertEqual(float(cfg.spatial_crop.noise_augmentation.isolated_false_positive_prob), 0.05)


if __name__ == "__main__":
    unittest.main()
