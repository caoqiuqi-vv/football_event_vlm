from __future__ import annotations

import torch

from football_sparsemax_roi_temporal import SparsemaxROITemporalAttention
from train_football_events import TemporalConditionedSpatialAttention


class ROIOnlyResidualAttention(SparsemaxROITemporalAttention):
    """Expose an ungated ROI-only residual for strict baseline correction."""

    def forward(self, global_frame_tokens, patch_tokens):
        result = super().forward(global_frame_tokens, patch_tokens)
        # raw_residual_logits is computed only from ROI temporal class_features.
        # No learned gate may suppress ROI learning or generate a global bypass.
        result["residual_logits"] = result["raw_residual_logits"]
        result["clip_gate"] = torch.ones_like(result["raw_residual_logits"])
        return result


def upgrade_roi_only_residual(
    module: TemporalConditionedSpatialAttention,
    *,
    sparsemax_temperature: float = 4.0,
) -> ROIOnlyResidualAttention:
    upgraded = ROIOnlyResidualAttention(
        patch_dim=module.patch_norm.normalized_shape[0],
        hidden_dim=module.context_norm.normalized_shape[0],
        attention_dim=module.attention_dim,
        num_labels=module.num_labels,
        queries_per_class=module.queries_per_class,
        context_layers=len(module.context_encoder.layers),
        temporal_layers=len(module.temporal_encoder.layers),
        num_heads=module.context_encoder.layers[0].self_attn.num_heads,
        dropout=float(module.region_adapter[3].p),
        max_frames=module.context_pos_embed.shape[1],
        gate_init=0.25,
        dynamic_query_scale_init=float(module.context_query_scale.detach()),
        return_attention_maps=module.return_attention_maps,
        sparsemax_temperature=sparsemax_temperature,
    )
    upgraded.load_state_dict(module.state_dict(), strict=False)
    with torch.no_grad():
        upgraded.class_tokens.zero_()
        upgraded.temporal_pos_embed[:, 0].zero_()
        upgraded.clip_classifier.bias.zero_()
        upgraded.clip_residual_head.bias.zero_()
    upgraded.class_tokens.requires_grad_(False)
    upgraded.temporal_pos_embed.requires_grad_(False)
    upgraded.clip_classifier.bias.requires_grad_(False)
    upgraded.clip_residual_head.bias.requires_grad_(False)
    return upgraded
