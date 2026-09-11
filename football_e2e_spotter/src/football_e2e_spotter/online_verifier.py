"""Candidate-context set verifier for dense online football spotting.

The verifier consumes every Stage-1 candidate in a fixed non-overlapping core
plus context.  It emits independent event slots and therefore removes window
duplicates through one-to-one set supervision rather than temporal NMS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .set_spotting import hungarian_assignment


@dataclass(frozen=True)
class ShotTarget:
    time_normalized: float


class CandidateContextSetVerifier(nn.Module):
    """Shot-first set verifier over Stage-1 and optional visual evidence."""

    def __init__(
        self,
        feature_dim: int,
        *,
        visual_dim: int = 0,
        hidden_dim: int = 256,
        num_slots: int = 8,
        encoder_layers: int = 3,
        decoder_layers: int = 3,
        heads: int = 8,
        dropout: float = 0.15,
        candidate_anchored: bool = True,
        max_time_delta_sec: float = 2.0,
        core_duration_sec: float = 30.0,
        residual_scoring: bool = True,
        max_score_residual: float = 2.0,
    ) -> None:
        super().__init__()
        self.num_slots = int(num_slots)
        self.candidate_anchored = bool(candidate_anchored)
        self.max_time_delta_normalized = (
            float(max_time_delta_sec) / float(core_duration_sec)
        )
        self.residual_scoring = bool(residual_scoring)
        self.max_score_residual = float(max_score_residual)
        self.feature_projection = nn.Linear(int(feature_dim), hidden_dim)
        self.visual_projection = (
            nn.Linear(int(visual_dim), hidden_dim) if int(visual_dim) > 0 else None
        )
        self.time_projection = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_dim, heads, 4 * hidden_dim, dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, encoder_layers)
        self.slot_queries = nn.Parameter(torch.randn(self.num_slots, hidden_dim) * 0.02)
        self.candidate_query_bias = nn.Parameter(
            torch.randn(1, 1, hidden_dim) * 0.02
        )
        decoder_layer = nn.TransformerDecoderLayer(
            hidden_dim, heads, 4 * hidden_dim, dropout,
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, decoder_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.class_head = nn.Linear(hidden_dim, 2)  # shot, no_event
        self.score_residual_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.score_residual_head.weight)
        nn.init.zeros_(self.score_residual_head.bias)
        self.time_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.time_head.weight)
        nn.init.zeros_(self.time_head.bias)
        self.log_sigma_head = nn.Linear(hidden_dim, 1)
        self.quality_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        features: Tensor,
        candidate_times_normalized: Tensor,
        candidate_mask: Tensor,
        visual_features: Tensor | None = None,
        baseline_logits: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if features.ndim != 3:
            raise ValueError("features must be [batch,candidates,feature_dim]")
        if candidate_times_normalized.shape != features.shape[:2]:
            raise ValueError("candidate time shape must match features")
        if candidate_mask.shape != features.shape[:2]:
            raise ValueError("candidate mask shape must match features")
        times = candidate_times_normalized.clamp(-1.0, 2.0)
        time_basis = torch.stack(
            (times, torch.sin(torch.pi * times), torch.cos(torch.pi * times)), dim=-1
        )
        memory = self.feature_projection(features) + self.time_projection(time_basis)
        if self.visual_projection is not None:
            if visual_features is None or visual_features.shape[:2] != features.shape[:2]:
                raise ValueError("configured visual features must match candidate axes")
            memory = memory + self.visual_projection(visual_features)
        invalid = ~candidate_mask.bool()
        memory = self.encoder(memory, src_key_padding_mask=invalid)
        if self.candidate_anchored:
            queries = memory + self.candidate_query_bias
        else:
            queries = self.slot_queries.unsqueeze(0).expand(features.shape[0], -1, -1)
        slots = self.norm(
            self.decoder(queries, memory, memory_key_padding_mask=invalid)
        )
        if self.candidate_anchored:
            refined_times = candidate_times_normalized + (
                self.max_time_delta_normalized
                * self.time_head(slots).tanh().squeeze(-1)
            )
            slot_mask = (
                candidate_mask.bool()
                & (candidate_times_normalized >= 0.0)
                & (candidate_times_normalized < 1.0)
            )
        else:
            refined_times = self.time_head(slots).sigmoid().squeeze(-1)
            slot_mask = torch.ones_like(refined_times, dtype=torch.bool)
        class_logits = self.class_head(slots)
        ranking_logits = class_logits[..., 0] - class_logits[..., 1]
        if self.residual_scoring and baseline_logits is not None:
            if baseline_logits.shape != features.shape[:2]:
                raise ValueError("baseline logit shape must match candidate axes")
            ranking_logits = baseline_logits + self.max_score_residual * (
                self.score_residual_head(slots).tanh().squeeze(-1)
            )
            class_logits = torch.stack(
                (ranking_logits, torch.zeros_like(ranking_logits)), dim=-1
            )
        return {
            "class_logits": class_logits,
            "ranking_logits": ranking_logits,
            "time_normalized": refined_times,
            "log_sigma": self.log_sigma_head(slots).squeeze(-1).clamp(-4.0, 2.0),
            "quality_logits": ranking_logits,
            "slot_embeddings": slots,
            "slot_mask": slot_mask,
        }


def shot_set_verifier_loss(
    outputs: Mapping[str, Tensor],
    targets: Sequence[Sequence[ShotTarget]],
    *,
    supervision_weights: Tensor | None = None,
    no_event_weight: float = 0.08,
    time_cost: float = 3.0,
    time_loss_weight: float = 3.0,
    quality_loss_weight: float = 0.5,
) -> dict[str, Tensor]:
    """Hungarian set loss; nearby GT shots remain separate instances."""
    logits = outputs["class_logits"]
    times = outputs["time_normalized"]
    total = logits.new_zeros(())
    matched_total = 0
    for batch_index, events in enumerate(targets):
        slot_mask = outputs.get("slot_mask")
        if slot_mask is None:
            valid_indices = torch.arange(logits.shape[1], device=logits.device)
        else:
            valid_indices = torch.where(slot_mask[batch_index].bool())[0]
        slots = len(valid_indices)
        if len(events) > slots:
            raise ValueError(
                f"core has {len(events)} targets but only {slots} candidate slots"
            )
        valid_logits = logits[batch_index, valid_indices]
        valid_times = times[batch_index, valid_indices]
        valid_supervision = (
            torch.ones(slots, dtype=logits.dtype, device=logits.device)
            if supervision_weights is None
            else supervision_weights[batch_index, valid_indices].to(logits.dtype)
        )
        classes = torch.ones(slots, dtype=torch.long, device=logits.device)
        quality = torch.zeros(slots, dtype=logits.dtype, device=logits.device)
        time_loss = logits.new_zeros(())
        if events:
            target_times = torch.tensor(
                [event.time_normalized for event in events],
                dtype=times.dtype,
                device=times.device,
            )
            class_cost = -valid_logits.log_softmax(-1)[:, 0]
            cost = class_cost.unsqueeze(0).expand(len(events), -1)
            cost = cost + float(time_cost) * (
                valid_times.unsqueeze(0) - target_times.unsqueeze(1)
            ).abs()
            rows, columns = hungarian_assignment(cost)
            classes[columns] = 0
            quality[columns] = 1.0
            matched_total += len(rows)
            matched_indices = valid_indices[columns]
            log_sigma = outputs["log_sigma"][batch_index, matched_indices]
            residual_sec = 30.0 * (
                times[batch_index, matched_indices] - target_times[rows]
            )
            time_loss = F.smooth_l1_loss(
                residual_sec, torch.zeros_like(residual_sec), beta=1.0
            ) + 0.01 * log_sigma.square().mean()
        ce = F.cross_entropy(valid_logits, classes, reduction="none")
        confidence = valid_logits.softmax(-1).gather(
            1, classes.unsqueeze(1)
        ).squeeze(1)
        weights = torch.where(
            classes == 1,
            float(no_event_weight) * valid_supervision,
            torch.ones_like(ce),
        )
        class_loss = (ce * (1.0 - confidence).square() * weights).sum()
        class_loss = class_loss / weights.sum().clamp_min(1.0)
        quality_values = F.binary_cross_entropy_with_logits(
            outputs["quality_logits"][batch_index, valid_indices], quality,
            reduction="none",
        )
        quality_weights = torch.where(
            quality > 0,
            torch.ones_like(quality_values),
            valid_supervision,
        )
        quality_loss = (quality_values * quality_weights).sum()
        quality_loss = quality_loss / quality_weights.sum().clamp_min(1.0)
        total = total + class_loss + float(time_loss_weight) * time_loss
        total = total + float(quality_loss_weight) * quality_loss
    return {
        "loss": total / max(len(targets), 1),
        "matched_instances": logits.new_tensor(float(matched_total)),
    }


def decode_shot_slots(
    outputs: Mapping[str, Tensor],
    *,
    core_start_sec: float,
    core_duration_sec: float,
    threshold: float,
    batch_index: int = 0,
) -> list[dict[str, float | int]]:
    if "ranking_logits" in outputs:
        scores = outputs["ranking_logits"][batch_index].sigmoid()
    else:
        probabilities = outputs["class_logits"][batch_index].softmax(-1)[:, 0]
        quality = outputs["quality_logits"][batch_index].sigmoid()
        scores = probabilities * quality
    slot_mask = outputs.get("slot_mask")
    result: list[dict[str, float | int]] = []
    for slot_index, score in enumerate(scores):
        if slot_mask is not None and not bool(slot_mask[batch_index, slot_index]):
            continue
        if float(score) < float(threshold):
            continue
        result.append({
            "slot_index": slot_index,
            "score": float(score),
            "time_sec": float(
                core_start_sec
                + core_duration_sec * outputs["time_normalized"][batch_index, slot_index]
            ),
            "uncertainty_sec": float(
                outputs["log_sigma"][batch_index, slot_index].exp()
                * core_duration_sec
            ),
        })
    return result
