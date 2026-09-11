from __future__ import annotations

import unittest

import torch
from torch import nn

from train_football_events import (
    AdaptiveSpatialTokenPool,
    ConfigDict,
    VideoEventClassifier,
    spatial_token_pooling_auxiliary_loss,
)


class TinyPatchBackbone(nn.Module):
    num_features = 12

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, self.num_features)
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(4)])

    def forward_features(self, frames: torch.Tensor):
        pixels = frames.permute(0, 2, 3, 1).reshape(frames.shape[0], -1, 3)
        patches = self.proj(pixels)
        return {
            "x_norm_clstoken": patches.mean(dim=1),
            "x_norm_patchtokens": patches,
        }


class AdaptiveSpatialTokenPoolTest(unittest.TestCase):
    def make_pool(self, *, return_maps: bool = True):
        torch.manual_seed(13)
        return AdaptiveSpatialTokenPool(
            patch_dim=12,
            hidden_dim=16,
            attention_dim=8,
            num_queries=4,
            num_labels=3,
            slot_dropout=0.0,
            gate_init=0.25,
            return_attention_maps=return_maps,
        )

    def make_model(
        self,
        *,
        spatial_tokens: bool,
        fusion_mode: str = "legacy_sequence",
    ):
        return VideoEventClassifier(
            backbone=TinyPatchBackbone(),
            frame_feature_dim=24,
            hidden_dim=16,
            num_labels=3,
            fusion="cls_transformer",
            num_layers=1,
            num_heads=4,
            dropout=0.0,
            max_frames=5,
            spatial_token_pooling_enabled=spatial_tokens,
            spatial_token_pooling_num_queries=4,
            spatial_token_pooling_attention_dim=8,
            spatial_token_pooling_slot_dropout=0.0,
            spatial_token_pooling_gate_init=0.25,
            spatial_token_pooling_fusion_mode=fusion_mode,
            spatial_token_pooling_relation_layers=2,
        )

    def test_pool_keeps_four_distinct_local_tokens(self) -> None:
        pool = self.make_pool()
        global_tokens = torch.randn(2, 5, 16)
        patches = torch.randn(2, 5, 9, 12)
        outputs = pool(global_tokens, patches)

        self.assertEqual(outputs["sequence"].shape, (2, 5, 5, 16))
        self.assertEqual(outputs["local_tokens"].shape, (2, 5, 4, 16))
        self.assertEqual(outputs["slot_logits"].shape, (2, 5, 4, 3))
        self.assertEqual(outputs["attention_maps"].shape, (2, 5, 4, 9))
        self.assertEqual(outputs["attention_overlap"].shape, (2, 5))
        self.assertTrue(
            torch.allclose(
                outputs["slot_gate"],
                torch.full_like(outputs["slot_gate"], 0.25),
                atol=1e-5,
            )
        )
        entropy = outputs["attention_entropy"]
        self.assertTrue(bool((entropy >= 0).all()))
        self.assertTrue(bool((entropy <= 1.0 + 1e-5).all()))

    def test_classifier_uses_repeated_frame_positions(self) -> None:
        model = self.make_model(spatial_tokens=True)
        outputs = model(torch.randn(2, 5, 3, 2, 2), return_aux=True)

        self.assertEqual(outputs["logits"].shape, (2, 3))
        self.assertEqual(
            outputs["spatial_token_slot_logits"].shape, (2, 5, 4, 3)
        )
        self.assertEqual(outputs["frame_event_logits"].shape, (2, 5, 3))
        # The temporal checkpoint remains compatible with the 16-frame model;
        # global/local slot identity is represented by slot_type_embed.
        self.assertEqual(model.temporal.pos_embed.shape, (1, 6, 16))
        self.assertEqual(
            model.spatial_token_pool.slot_type_embed.shape, (1, 1, 5, 16)
        )

    def test_joint_token_temporal_path_preserves_frame_outputs(self) -> None:
        model = self.make_model(
            spatial_tokens=True,
            fusion_mode="joint_token_temporal_dropout",
        )
        model.eval()
        outputs = model(torch.randn(2, 5, 3, 2, 2), return_aux=True)

        self.assertEqual(outputs["logits"].shape, (2, 3))
        self.assertEqual(outputs["frame_event_logits"].shape, (2, 5, 3))
        self.assertEqual(
            outputs["spatial_token_slot_logits"].shape, (2, 5, 4, 3)
        )
        self.assertIsNotNone(model.spatial_token_joint_temporal)
        self.assertEqual(
            model.spatial_token_joint_temporal.pos_embed.shape,
            (1, 5, 5, 16),
        )

    def test_slot_dropout_is_clip_consistent(self) -> None:
        torch.manual_seed(29)
        pool = AdaptiveSpatialTokenPool(
            patch_dim=12,
            hidden_dim=16,
            attention_dim=8,
            num_queries=8,
            num_labels=3,
            slot_dropout=0.5,
            gate_init=0.25,
        )
        pool.train()
        outputs = pool(torch.randn(2, 5, 16), torch.randn(2, 5, 9, 12))
        present = outputs["temporal_local_tokens"].abs().sum(dim=-1) > 0
        self.assertTrue(bool((present == present[:, :1]).all()))

    def test_auxiliary_mil_trains_queries_and_patches(self) -> None:
        pool = self.make_pool(return_maps=False)
        global_tokens = torch.randn(2, 5, 16, requires_grad=True)
        patches = torch.randn(2, 5, 9, 12, requires_grad=True)
        branch = pool(global_tokens, patches)
        outputs = {
            "spatial_token_slot_logits": branch["slot_logits"],
            "spatial_token_query_diversity_loss": branch[
                "query_diversity_loss"
            ],
            "spatial_token_attention_overlap": branch["attention_overlap"],
            "spatial_token_attention_entropy": branch["attention_entropy"],
            "spatial_token_slot_gate": branch["slot_gate"],
            "spatial_token_context_scale": branch["context_scale"],
        }
        targets = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        masks = torch.ones_like(targets)
        cfg = ConfigDict(
            {
                "train": {
                    "spatial_token_mil_loss_weight": 0.15,
                    "spatial_token_mil_topk": 4,
                    "spatial_token_mil_balance": "per_class_equal_pos_neg",
                    "spatial_token_query_diversity_loss_weight": 0.005,
                    "spatial_token_overlap_loss_weight": 0.01,
                }
            }
        )
        loss, components = spatial_token_pooling_auxiliary_loss(
            outputs, targets, masks, cfg
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("spatial_token_mil_pos_loss", components)
        self.assertIn("spatial_token_mil_neg_loss", components)
        loss.backward()
        self.assertIsNotNone(pool.base_queries.grad)
        self.assertGreater(float(pool.base_queries.grad.abs().sum()), 0.0)
        self.assertIsNotNone(patches.grad)
        self.assertGreater(float(patches.grad.abs().sum()), 0.0)

    def test_old_checkpoint_loads_without_temporal_shape_changes(self) -> None:
        torch.manual_seed(23)
        baseline = self.make_model(spatial_tokens=False)
        candidate = self.make_model(spatial_tokens=True)
        missing, unexpected = candidate.load_state_dict(
            baseline.state_dict(), strict=False
        )

        self.assertFalse(unexpected)
        self.assertTrue(missing)
        self.assertTrue(
            all(key.startswith("spatial_token_pool.") for key in missing)
        )
        self.assertTrue(
            torch.equal(
                candidate.temporal.pos_embed,
                baseline.temporal.pos_embed,
            )
        )
        self.assertTrue(
            torch.equal(candidate.head[1].weight, baseline.head[1].weight)
        )


if __name__ == "__main__":
    unittest.main()
