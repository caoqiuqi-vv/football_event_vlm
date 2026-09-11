from __future__ import annotations

import unittest

import torch
from torch import nn

from train_football_events import (
    ConfigDict,
    TemporalConditionedSpatialAttention,
    VideoEventClassifier,
    global_conditioned_correction_loss,
    spatial_attention_auxiliary_loss,
    spatial_counterfactual_causal_loss,
    spatial_temporal_localization_loss,
)


class TinyPatchBackbone(nn.Module):
    num_features = 12

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, self.num_features)
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(4)])

    def _patches(self, frames: torch.Tensor, layer: int) -> torch.Tensor:
        pixels = frames.permute(0, 2, 3, 1).reshape(frames.shape[0], -1, 3)
        return self.proj(pixels) + float(layer) * 0.1

    def forward_features(self, frames: torch.Tensor):
        patches = self._patches(frames, len(self.blocks) - 1)
        return {
            "x_norm_clstoken": patches.mean(dim=1),
            "x_norm_patchtokens": patches,
        }

    def get_intermediate_layers(
        self,
        frames: torch.Tensor,
        *,
        n,
        return_class_token: bool,
        norm: bool,
    ):
        del norm
        outputs = []
        for layer in n:
            patches = self._patches(frames, int(layer))
            cls = patches.mean(dim=1)
            outputs.append((patches, cls) if return_class_token else patches)
        return outputs


class TemporalConditionedSpatialAttentionTest(unittest.TestCase):
    def make_module(self, *, return_maps: bool = False):
        torch.manual_seed(7)
        return TemporalConditionedSpatialAttention(
            patch_dim=12,
            hidden_dim=16,
            attention_dim=8,
            num_labels=3,
            queries_per_class=2,
            context_layers=1,
            temporal_layers=1,
            num_heads=4,
            dropout=0.0,
            max_frames=6,
            gate_init=0.05,
            return_attention_maps=return_maps,
        )

    def test_forward_preserves_reference_at_initialization(self) -> None:
        module = self.make_module(return_maps=True)
        global_tokens = torch.randn(2, 5, 16)
        patch_tokens = torch.randn(2, 5, 9, 12)
        outputs = module(global_tokens, patch_tokens)

        self.assertEqual(outputs["residual_logits"].shape, (2, 3))
        self.assertEqual(outputs["frame_event_logits"].shape, (2, 5, 3))
        self.assertEqual(outputs["gate"].shape, (2, 5, 3))
        self.assertEqual(outputs["attention_maps"].shape, (2, 5, 3, 2, 9))
        self.assertEqual(outputs["attention_overlap"].shape, (2, 5, 3))
        self.assertTrue(
            torch.allclose(
                outputs["residual_logits"],
                torch.zeros_like(outputs["residual_logits"]),
            )
        )
        self.assertTrue(
            torch.allclose(
                outputs["gate"],
                torch.full_like(outputs["gate"], 0.05),
                atol=1e-5,
            )
        )
        entropy = outputs["attention_entropy"]
        self.assertTrue(bool((entropy >= 0).all()))
        self.assertTrue(bool((entropy <= 1.0 + 1e-5).all()))

    def test_clip_mil_trains_dynamic_queries_before_residual_moves(self) -> None:
        module = self.make_module()
        global_tokens = torch.randn(2, 5, 16, requires_grad=True)
        patch_tokens = torch.randn(2, 5, 9, 12, requires_grad=True)
        branch = module(global_tokens, patch_tokens)
        outputs = {
            "spatial_frame_event_logits": branch["frame_event_logits"],
            "spatial_query_diversity_loss": branch[
                "query_diversity_loss"
            ],
            "spatial_gate": branch["gate"],
            "spatial_attention_entropy": branch["attention_entropy"],
            "spatial_attention_overlap": branch["attention_overlap"],
        }
        targets = torch.tensor(
            [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
        )
        masks = torch.ones_like(targets)
        cfg = ConfigDict(
            {
                "train": {
                    "spatial_attention_mil_loss_weight": 0.3,
                    "spatial_attention_query_diversity_loss_weight": 0.01,
                    "spatial_attention_concentration_loss_weight": 0.1,
                    "spatial_attention_overlap_loss_weight": 0.05,
                    "spatial_attention_entropy_target": 0.85,
                    "spatial_attention_mil_topk": 2,
                }
            }
        )
        loss, components = spatial_attention_auxiliary_loss(
            outputs, targets, masks, cfg
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("spatial_attention_mil_pos_loss", components)
        self.assertIn("spatial_attention_mil_neg_loss", components)
        self.assertIn("spatial_attention_concentration_loss", components)
        self.assertIn("spatial_attention_overlap_loss", components)
        self.assertGreater(
            components["spatial_attention_concentration_loss"], 0.0
        )
        loss.backward()
        self.assertIsNotNone(module.base_queries.grad)
        self.assertGreater(float(module.base_queries.grad.abs().sum()), 0.0)
        self.assertIsNotNone(patch_tokens.grad)
        self.assertGreater(float(patch_tokens.grad.abs().sum()), 0.0)


    def test_gt_window_temporal_loss_rejects_far_frame_shortcut(self) -> None:
        logits = torch.tensor(
            [[[0.0], [2.0], [0.0], [5.0], [0.0]]],
            requires_grad=True,
        )
        batch = {
            "frame_targets": torch.tensor(
                [[[0.0], [1.0], [0.2], [0.0], [0.0]]]
            ),
            "frame_target_masks": torch.tensor(
                [[[0.0], [1.0], [1.0], [0.0], [0.0]]]
            ),
            "targets": torch.ones((1, 1)),
            "label_masks": torch.ones((1, 1)),
        }
        cfg = ConfigDict(
            {
                "train": {
                    "spatial_temporal_topk": 2,
                    "spatial_temporal_near_target_threshold": 0.5,
                    "spatial_temporal_rank_margin": 0.5,
                    "spatial_temporal_heatmap_weight": 0.5,
                    "spatial_temporal_mil_weight": 0.3,
                    "spatial_temporal_rank_weight": 0.2,
                }
            }
        )
        loss, components = spatial_temporal_localization_loss(
            {"spatial_frame_event_logits": logits},
            batch,
            cfg,
            torch.device("cpu"),
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(components["spatial_temporal_rank_loss"], 1.0)
        loss.backward()
        self.assertLess(float(logits.grad[0, 1, 0]), 0.0)
        self.assertGreater(float(logits.grad[0, 3, 0]), 0.0)

    def test_corrected_frame_temporal_loss_routes_gradient(self) -> None:
        logits = torch.tensor(
            [[[0.0], [2.0], [0.0], [5.0], [0.0]]],
            requires_grad=True,
        )
        batch = {
            "frame_targets": torch.tensor(
                [[[0.0], [1.0], [0.2], [0.0], [0.0]]]
            ),
            "frame_target_masks": torch.tensor(
                [[[0.0], [1.0], [1.0], [0.0], [0.0]]]
            ),
            "targets": torch.ones((1, 1)),
            "label_masks": torch.ones((1, 1)),
        }
        cfg = ConfigDict(
            {
                "train": {
                    "spatial_temporal_topk": 2,
                    "spatial_temporal_near_target_threshold": 0.5,
                    "spatial_temporal_rank_margin": 0.5,
                    "spatial_temporal_heatmap_weight": 0.5,
                    "spatial_temporal_mil_weight": 0.3,
                    "spatial_temporal_rank_weight": 0.2,
                }
            }
        )
        loss, components = spatial_temporal_localization_loss(
            {"structured_frame_event_logits": logits},
            batch,
            cfg,
            torch.device("cpu"),
            logits_key="structured_frame_event_logits",
            component_prefix="structured_frame_temporal",
        )
        self.assertIn("structured_frame_temporal_loss", components)
        loss.backward()
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_sparse_roi_counterfactual_trains_attention_queries(self) -> None:
        from football_sparsemax_roi_temporal import upgrade_sparsemax_roi_temporal

        module = upgrade_sparsemax_roi_temporal(
            self.make_module(), sparsemax_temperature=2.0
        )
        global_tokens = torch.randn(2, 5, 16)
        patch_tokens = torch.randn(2, 5, 9, 12)
        branch = module(global_tokens, patch_tokens)
        self.assertEqual(branch["erased_clip_logits"].shape, (2, 3))
        targets = torch.tensor(
            [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
        )
        loss, components = spatial_counterfactual_causal_loss(
            {
                "spatial_clip_logits": branch["clip_logits"],
                "spatial_erased_clip_logits": branch[
                    "erased_clip_logits"
                ],
            },
            targets,
            torch.ones_like(targets),
            ConfigDict(
                {"train": {"spatial_counterfactual_margin": 0.5}}
            ),
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("spatial_counterfactual_gap", components)
        loss.backward()
        self.assertIsNotNone(module.base_queries.grad)
        self.assertGreater(float(module.base_queries.grad.abs().sum()), 0.0)

    def test_video_classifier_integrates_spatial_residual(self) -> None:
        model = VideoEventClassifier(
            backbone=TinyPatchBackbone(),
            frame_feature_dim=24,
            hidden_dim=16,
            num_labels=3,
            fusion="cls_transformer",
            num_layers=1,
            num_heads=4,
            dropout=0.0,
            max_frames=5,
            spatial_attention_enabled=True,
            spatial_attention_dim=8,
            spatial_attention_queries_per_class=2,
            spatial_attention_context_layers=1,
            spatial_attention_temporal_layers=1,
            spatial_attention_heads=4,
            spatial_attention_gate_init=0.05,
        )
        inputs = torch.randn(2, 5, 3, 2, 2)
        outputs = model(inputs, return_aux=True)
        self.assertEqual(outputs["logits"].shape, (2, 3))
        self.assertEqual(outputs["spatial_frame_event_logits"].shape, (2, 5, 3))
        self.assertTrue(
            torch.allclose(outputs["logits"], outputs["global_logits"])
        )
        self.assertTrue(
            torch.allclose(
                outputs["retention_reference_logits"],
                outputs["global_logits"],
            )
        )


    def test_residual_multilayer_starts_from_identical_baseline(self) -> None:
        torch.manual_seed(19)
        common = dict(
            frame_feature_dim=24,
            hidden_dim=16,
            num_labels=3,
            fusion="cls_transformer",
            num_layers=1,
            num_heads=4,
            dropout=0.0,
            max_frames=5,
            spatial_attention_enabled=True,
            spatial_attention_dim=8,
            spatial_attention_queries_per_class=2,
            spatial_attention_context_layers=1,
            spatial_attention_temporal_layers=1,
            spatial_attention_heads=4,
            spatial_attention_gate_init=0.05,
        )
        baseline = VideoEventClassifier(
            backbone=TinyPatchBackbone(),
            frame_feature_mode="last_cls_patch_mean",
            **common,
        )
        residual = VideoEventClassifier(
            backbone=TinyPatchBackbone(),
            frame_feature_mode="residual_multilayer_cls_patch_attn",
            frame_feature_layers=(-3, -2, -1),
            frame_patch_pool="attn",
            **common,
        )
        missing, unexpected = residual.load_state_dict(
            baseline.state_dict(), strict=False
        )
        self.assertFalse(unexpected)
        self.assertNotIn("frame_proj.1.weight", missing)
        baseline.eval()
        residual.eval()
        inputs = torch.randn(2, 5, 3, 2, 2)
        baseline_outputs = baseline(inputs, return_aux=True)
        residual_outputs = residual(inputs, return_aux=True)
        self.assertTrue(
            torch.allclose(
                residual_outputs["global_logits"],
                baseline_outputs["global_logits"],
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                residual_outputs["logits"],
                baseline_outputs["logits"],
                atol=1e-6,
            )
        )
        self.assertEqual(
            float(residual_outputs["multi_layer_frame_residual"].abs().max()),
            0.0,
        )
        loss = residual_outputs["logits"].sum()
        loss = loss + residual_outputs["spatial_frame_event_logits"].sum()
        loss.backward()
        self.assertGreater(
            float(residual.multi_layer_frame_adapter[-1].weight.grad.abs().sum()),
            0.0,
        )
        self.assertGreater(
            float(residual.multi_layer_patch_adapter[-1].weight.grad.abs().sum()),
            0.0,
        )

    def test_probe_uses_independent_spatial_logits_and_preserves_global(self) -> None:
        torch.manual_seed(23)
        model = VideoEventClassifier(
            backbone=TinyPatchBackbone(),
            frame_feature_dim=24,
            hidden_dim=16,
            num_labels=3,
            fusion="cls_transformer",
            num_layers=1,
            num_heads=4,
            dropout=0.0,
            max_frames=5,
            frame_feature_mode="last_cls_patch_mean",
            spatial_attention_enabled=True,
            spatial_attention_dim=8,
            spatial_attention_queries_per_class=2,
            spatial_attention_context_layers=1,
            spatial_attention_temporal_layers=1,
            spatial_attention_heads=4,
            spatial_attention_dynamic_query_scale_init=0.0,
            spatial_attention_mode="probe",
            spatial_attention_patch_mode="residual_multilayer",
            spatial_attention_feature_layers=(-3, -2, -1),
        )
        model.freeze_global_parameters()
        inputs = torch.randn(2, 5, 3, 2, 2)
        outputs = model(inputs, return_aux=True)
        self.assertTrue(
            torch.allclose(outputs["logits"], outputs["spatial_clip_logits"])
        )
        self.assertEqual(
            float(outputs["spatial_context_query_scale"].detach()), 0.0
        )
        self.assertEqual(
            float(outputs["spatial_motion_query_scale"].detach()), 0.0
        )
        outputs["logits"].sum().backward()
        self.assertGreater(
            float(model.spatial_patch_adapter[-1].weight.grad.abs().sum()),
            0.0,
        )
        self.assertIsNotNone(model.spatial_attention.context_query_scale.grad)

    def test_adaptive_fusion_starts_from_exact_global_reference(self) -> None:
        torch.manual_seed(29)
        model = VideoEventClassifier(
            backbone=TinyPatchBackbone(),
            frame_feature_dim=24,
            hidden_dim=16,
            num_labels=3,
            fusion="cls_transformer",
            num_layers=1,
            num_heads=4,
            dropout=0.0,
            max_frames=5,
            spatial_attention_enabled=True,
            spatial_attention_dim=8,
            spatial_attention_queries_per_class=2,
            spatial_attention_context_layers=1,
            spatial_attention_temporal_layers=1,
            spatial_attention_heads=4,
            spatial_attention_mode="adaptive_fusion",
            spatial_attention_fusion_gate_init=0.15,
        )
        outputs = model(torch.randn(2, 5, 3, 2, 2), return_aux=True)
        self.assertTrue(
            torch.allclose(outputs["logits"], outputs["global_logits"])
        )
        self.assertTrue(
            torch.allclose(
                outputs["spatial_fusion_gate"],
                torch.full_like(outputs["spatial_fusion_gate"], 0.15),
                atol=1e-5,
            )
        )
        self.assertEqual(
            float(outputs["spatial_fusion_delta"].abs().max()), 0.0
        )
    def test_adaspot_feature_fusion_starts_exact_and_trains_output(self) -> None:
        torch.manual_seed(30)
        model = VideoEventClassifier(
            backbone=TinyPatchBackbone(),
            frame_feature_dim=24,
            hidden_dim=16,
            num_labels=3,
            fusion="cls_transformer",
            num_layers=1,
            num_heads=4,
            dropout=0.0,
            max_frames=5,
            spatial_attention_enabled=True,
            spatial_attention_dim=8,
            spatial_attention_queries_per_class=2,
            spatial_attention_context_layers=1,
            spatial_attention_temporal_layers=1,
            spatial_attention_heads=4,
            spatial_attention_mode="adaspot_feature_fusion",
        )
        model.freeze_global_parameters()
        inputs = torch.randn(2, 5, 3, 2, 2)
        outputs = model(inputs, return_aux=True)
        self.assertTrue(torch.allclose(outputs["logits"], outputs["global_logits"]))
        self.assertEqual(outputs["frame_event_logits"].shape, (2, 5, 3))
        self.assertEqual(outputs["spatial_feature_delta"].shape, (2, 5, 3, 16))
        self.assertEqual(float(outputs["spatial_feature_delta"].abs().max()), 0.0)
        loss = outputs["logits"].sum()
        loss.backward()
        self.assertGreater(
            float(model.spatial_feature_output[-1].weight.grad.abs().sum()), 0.0
        )
        self.assertIsNone(model.temporal.encoder.layers[0].self_attn.in_proj_weight.grad)


    def test_conditioned_residual_starts_from_frozen_global(self) -> None:
        torch.manual_seed(31)
        model = VideoEventClassifier(
            backbone=TinyPatchBackbone(),
            frame_feature_dim=24,
            hidden_dim=16,
            num_labels=3,
            fusion="cls_transformer",
            num_layers=1,
            num_heads=4,
            dropout=0.0,
            max_frames=5,
            spatial_attention_enabled=True,
            spatial_attention_dim=8,
            spatial_attention_queries_per_class=2,
            spatial_attention_context_layers=1,
            spatial_attention_temporal_layers=1,
            spatial_attention_heads=4,
            spatial_attention_mode="conditioned_residual",
            spatial_attention_correction_dim=16,
            spatial_attention_correction_gate_init=0.25,
            spatial_attention_correction_max_delta=2.0,
        )
        model.freeze_global_parameters()
        model.freeze_spatial_probe_parameters()
        outputs = model(torch.randn(2, 5, 3, 2, 2), return_aux=True)
        self.assertTrue(torch.allclose(outputs["logits"], outputs["global_logits"]))
        self.assertEqual(float(outputs["spatial_fusion_delta"].abs().max()), 0.0)
        self.assertTrue(
            torch.allclose(
                outputs["spatial_fusion_gate"],
                torch.full_like(outputs["spatial_fusion_gate"], 0.25),
                atol=1e-5,
            )
        )
        trainable = [name for name, value in model.named_parameters() if value.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("spatial_conditioned_") for name in trainable))

    def test_conditioned_correction_pushes_fp_down_and_fn_up(self) -> None:
        correction = torch.zeros((2, 3), requires_grad=True)
        global_logits = torch.tensor(
            [[1.0, -1.0, 1.0], [-1.0, -1.0, 1.0]]
        )
        targets = torch.tensor(
            [[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]]
        )
        outputs = {
            "global_logits": global_logits,
            "logits": global_logits + correction,
            "spatial_fusion_delta": correction,
        }
        loss, components = global_conditioned_correction_loss(
            outputs,
            targets,
            torch.ones_like(targets),
            threshold_probs=(0.5, 0.5, 0.5),
            margin_logit=0.25,
        )
        loss.backward()
        self.assertEqual(components["global_correction_fp_slots"], 2.0)
        self.assertEqual(components["global_correction_fn_slots"], 2.0)
        self.assertGreater(float(correction.grad[0, 0]), 0.0)
        self.assertGreater(float(correction.grad[1, 2]), 0.0)
        self.assertLess(float(correction.grad[0, 1]), 0.0)
        self.assertLess(float(correction.grad[1, 0]), 0.0)
        self.assertEqual(float(correction.grad[0, 2]), 0.0)
        self.assertEqual(float(correction.grad[1, 1]), 0.0)

    def test_per_class_balanced_mil_does_not_hide_positive_loss(self) -> None:
        frame_logits = torch.full((4, 2, 3), -2.0)
        outputs = {"spatial_frame_event_logits": frame_logits}
        targets = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        masks = torch.ones_like(targets)
        all_cfg = ConfigDict(
            {"train": {"spatial_attention_mil_loss_weight": 1.0}}
        )
        balanced_cfg = ConfigDict(
            {
                "train": {
                    "spatial_attention_mil_loss_weight": 1.0,
                    "spatial_attention_mil_balance": "per_class_equal_pos_neg",
                }
            }
        )
        all_loss, _ = spatial_attention_auxiliary_loss(
            outputs, targets, masks, all_cfg
        )
        balanced_loss, components = spatial_attention_auxiliary_loss(
            outputs, targets, masks, balanced_cfg
        )
        self.assertGreater(float(balanced_loss), float(all_loss))
        expected = 0.5 * (
            components["spatial_attention_mil_pos_loss"]
            + components["spatial_attention_mil_neg_loss"]
        )
        self.assertAlmostEqual(float(balanced_loss), expected, places=5)

if __name__ == "__main__":
    unittest.main()

