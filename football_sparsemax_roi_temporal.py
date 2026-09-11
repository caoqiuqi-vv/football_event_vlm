from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from train_football_events import TemporalConditionedSpatialAttention


def sparsemax(logits: Tensor, dim: int = -1) -> Tensor:
    """Differentiable Euclidean projection onto the probability simplex."""
    shifted = logits - logits.max(dim=dim, keepdim=True).values
    sorted_logits = torch.sort(shifted, dim=dim, descending=True).values
    size = logits.shape[dim]
    ranks_shape = [1] * logits.ndim
    ranks_shape[dim] = size
    ranks = torch.arange(
        1, size + 1, device=logits.device, dtype=logits.dtype
    ).reshape(ranks_shape)
    cumulative = sorted_logits.cumsum(dim)
    support = 1.0 + ranks * sorted_logits > cumulative
    support_size = support.sum(dim=dim, keepdim=True).clamp_min(1)
    threshold = (
        cumulative.gather(dim, support_size - 1) - 1.0
    ) / support_size.to(logits.dtype)
    return (shifted - threshold).clamp_min(0.0)


class SparsemaxROITemporalAttention(TemporalConditionedSpatialAttention):
    """Differentiable sparse ROI selection plus frame/query temporal reasoning."""

    def __init__(self, *args, sparsemax_temperature: float = 4.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.sparsemax_temperature = float(sparsemax_temperature)
        hidden_dim = self.temporal_norm.normalized_shape[0]
        self.roi_query_identity = nn.Parameter(
            torch.zeros(1, 1, self.queries_per_class, hidden_dim)
        )
        nn.init.trunc_normal_(self.roi_query_identity, std=0.02)

    def forward(
        self, global_frame_tokens: Tensor, patch_tokens: Tensor
    ) -> dict[str, Tensor]:
        # Keep the small head in fp32 while the frozen high-resolution ViT stays
        # under bf16 autocast. Sparse simplex projection is sensitive to ties.
        with torch.autocast(device_type="cuda", enabled=False):
            return self._forward_fp32(
                global_frame_tokens.float(), patch_tokens.float()
            )

    def _forward_fp32(
        self, global_frame_tokens: Tensor, patch_tokens: Tensor
    ) -> dict[str, Tensor]:
        bsz, frames, hidden_dim = global_frame_tokens.shape
        context = global_frame_tokens + self.context_pos_embed[:, :frames]
        context = self.context_norm(self.context_encoder(context))
        motion = torch.zeros_like(context)
        if frames > 1:
            motion[:, 1:] = context[:, 1:] - context[:, :-1]

        dynamic_query = (
            self.context_query_scale * self.context_query(context)
            + self.motion_query_scale * self.motion_query(motion)
        ).reshape(
            bsz,
            frames,
            self.num_labels,
            self.queries_per_class,
            self.attention_dim,
        )
        queries = self.query_norm(self.base_queries + dynamic_query)
        normalized_patches = self.patch_norm(patch_tokens)
        keys = self.key_norm(self.patch_key(normalized_patches))
        values = self.patch_value(normalized_patches)
        scores = torch.einsum(
            "btcqa,btna->btcqn", queries, keys
        ) / math.sqrt(float(self.attention_dim))
        attention = sparsemax(
            scores / max(self.sparsemax_temperature, 1e-4), dim=-1
        )

        region_queries = torch.einsum(
            "btcqn,btnh->btcqh", attention, values
        )
        region_delta = self.region_adapter(region_queries)

        # Counterfactual evidence from all patches outside the selected sparse
        # support.  This stays differentiable with respect to the attention map
        # and reuses the frozen DINO pass.
        patch_count = attention.shape[-1]
        erased_attention = (1.0 - attention) / max(float(patch_count - 1), 1.0)
        erased_region_queries = torch.einsum(
            "btcqn,btnh->btcqh", erased_attention, values
        )
        erased_region_delta = self.region_adapter(erased_region_queries)

        # Preserve both query identity and frame identity through temporal
        # reasoning. Positive and negative clips follow the identical path.
        local_tokens = region_delta.permute(0, 2, 1, 3, 4).reshape(
            bsz * self.num_labels,
            frames * self.queries_per_class,
            hidden_dim,
        )
        local_pos = (
            self.temporal_pos_embed[:, 1 : frames + 1].unsqueeze(2)
            + self.roi_query_identity
        ).reshape(1, frames * self.queries_per_class, hidden_dim)
        local_tokens = local_tokens + local_pos
        class_tokens = self.class_tokens.expand(bsz, -1, -1).reshape(
            bsz * self.num_labels, 1, hidden_dim
        ) + self.temporal_pos_embed[:, :1]
        sequence = self.temporal_encoder(
            torch.cat([class_tokens, local_tokens], dim=1)
        )
        class_features = self.temporal_norm(sequence[:, 0]).reshape(
            bsz, self.num_labels, hidden_dim
        )
        clip_logits = self.clip_classifier(class_features).squeeze(-1)
        raw_residual_logits = self.clip_residual_head(class_features).squeeze(-1)

        erased_local_tokens = erased_region_delta.permute(0, 2, 1, 3, 4).reshape(
            bsz * self.num_labels,
            frames * self.queries_per_class,
            hidden_dim,
        ) + local_pos
        erased_sequence = self.temporal_encoder(
            torch.cat([class_tokens, erased_local_tokens], dim=1)
        )
        erased_class_features = self.temporal_norm(erased_sequence[:, 0]).reshape(
            bsz, self.num_labels, hidden_dim
        )
        erased_clip_logits = self.clip_classifier(erased_class_features).squeeze(-1)

        expanded_context = context.unsqueeze(2).unsqueeze(3).expand(
            -1, -1, self.num_labels, self.queries_per_class, -1
        )
        query_gate = torch.sigmoid(
            self.region_gate(
                torch.cat(
                    [
                        expanded_context,
                        region_delta,
                        (region_delta - expanded_context).abs(),
                    ],
                    dim=-1,
                )
            )
        ).squeeze(-1)
        query_pool = self.query_pool_logits.softmax(dim=-1)
        roi_frame_features = (
            region_delta
            * query_pool.reshape(
                1, 1, self.num_labels, self.queries_per_class, 1
            )
        ).sum(dim=3)
        frame_gate = (
            query_gate
            * query_pool.reshape(1, 1, self.num_labels, self.queries_per_class)
        ).sum(dim=-1)
        clip_gate = torch.topk(
            frame_gate, k=min(3, frames), dim=1
        ).values.mean(dim=1)
        residual_logits = clip_gate * raw_residual_logits

        query_frame_logits = self.frame_event_head(region_delta).squeeze(-1)
        frame_event_logits = torch.logsumexp(query_frame_logits, dim=-1) - math.log(
            float(self.queries_per_class)
        )

        probabilities = attention.clamp_min(1e-8)
        entropy = -(probabilities * probabilities.log()).sum(dim=-1)
        entropy = entropy / max(math.log(float(patch_tokens.shape[2])), 1e-6)
        if self.queries_per_class > 1:
            normalized_attention = F.normalize(attention, dim=-1)
            similarity = torch.einsum(
                "btcqn,btckn->btcqk", normalized_attention, normalized_attention
            )
            eye = torch.eye(
                self.queries_per_class,
                device=similarity.device,
                dtype=similarity.dtype,
            ).reshape(1, 1, 1, self.queries_per_class, self.queries_per_class)
            attention_overlap = (similarity * (1.0 - eye)).sum(
                dim=(-1, -2)
            ) / float(self.queries_per_class * (self.queries_per_class - 1))
        else:
            attention_overlap = entropy.new_zeros(
                (bsz, frames, self.num_labels)
            )

        result = {
            "clip_logits": clip_logits,
            "erased_clip_logits": erased_clip_logits,
            "residual_logits": residual_logits,
            "raw_residual_logits": raw_residual_logits,
            "frame_event_logits": frame_event_logits,
            "query_frame_logits": query_frame_logits,
            "gate": frame_gate,
            "clip_gate": clip_gate,
            "attention_entropy": entropy,
            "attention_overlap": attention_overlap,
            "query_diversity_loss": self.query_diversity_loss(),
            "context_query_scale": self.context_query_scale,
            "motion_query_scale": self.motion_query_scale,
            "class_features": class_features,
            "roi_frame_features": roi_frame_features,
            "roi_support_size": (attention > 0).sum(dim=-1).float(),
        }
        if self.return_attention_maps:
            result["attention_maps"] = attention
        return result


def upgrade_sparsemax_roi_temporal(
    module: TemporalConditionedSpatialAttention,
    *,
    sparsemax_temperature: float = 4.0,
) -> SparsemaxROITemporalAttention:
    upgraded = SparsemaxROITemporalAttention(
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
    return upgraded
