"""NMS-free temporal set spotting for five football event classes.

Predictions are independent query slots.  Nearby events are never merged;
one-to-one assignment is used only for training/evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .model import AudioEncoder, RegNetTemporalEncoder, TemporalResidualBlock


SET_LABELS = ("shot", "save", "freekick", "corner", "kickoff")
NO_EVENT_INDEX = len(SET_LABELS)
FAMILY_LABELS = ("shot_chain", "restart")


@dataclass(frozen=True)
class EventInstance:
    """A core-relative event time in [0, 1], independent of other events."""

    label: int
    time_normalized: float
    uncertainty_sec: float = 0.5


@dataclass(frozen=True)
class SlotPrediction:
    label: str
    time_sec: float
    score: float
    uncertainty_sec: float
    slot_index: int


def hungarian_assignment(cost: Tensor) -> tuple[Tensor, Tensor]:
    """Minimum cost rows-to-columns assignment for cost[T,Q] where T <= Q."""
    if cost.ndim != 2 or cost.shape[0] > cost.shape[1]:
        raise ValueError("cost must be [targets, slots] with targets <= slots")
    targets, slots = cost.shape
    if targets == 0:
        empty = torch.empty(0, dtype=torch.long, device=cost.device)
        return empty, empty
    values = cost.detach().float().cpu().tolist()
    u, v = [0.0] * (targets + 1), [0.0] * (slots + 1)
    p, way = [0] * (slots + 1), [0] * (slots + 1)
    for i in range(1, targets + 1):
        p[0], j0 = i, 0
        minv, used = [float("inf")] * (slots + 1), [False] * (slots + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], float("inf"), 0
            for j in range(1, slots + 1):
                if not used[j]:
                    current = values[i0 - 1][j - 1] - u[i0] - v[j]
                    if current < minv[j]:
                        minv[j], way[j] = current, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(slots + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            previous = way[j0]
            p[j0] = p[previous]
            j0 = previous
            if j0 == 0:
                break
    columns = [0] * (targets + 1)
    for column in range(1, slots + 1):
        if p[column]:
            columns[p[column]] = column
    return (
        torch.arange(targets, device=cost.device),
        torch.tensor(columns[1:], device=cost.device, dtype=torch.long) - 1,
    )


class TemporalSetSpotter(nn.Module):
    """Stage-1 RGB/audio Spotter with 24 unordered event slots per core."""

    def __init__(
        self,
        *,
        sample_fps: float = 4.0,
        core_seconds: float = 30.0,
        context_seconds: float = 15.0,
        num_slots: int = 24,
        mel_bins: int = 64,
        audio_dim: int = 128,
        hidden_dim: int = 512,
        decoder_layers: int = 4,
        decoder_heads: int = 8,
        dropout: float = 0.15,
        imagenet_initialization: bool = True,
        refinement_stages: Iterable[str] = ("block2", "block3", "block4"),
    ) -> None:
        super().__init__()
        self.sample_fps, self.core_seconds, self.context_seconds = float(sample_fps), float(core_seconds), float(context_seconds)
        self.num_slots = int(num_slots)
        self.visual = RegNetTemporalEncoder(imagenet_initialization=imagenet_initialization, refinement_stages=refinement_stages)
        self.audio = AudioEncoder(mel_bins, audio_dim, dropout)
        visual_dim = self.visual.output_dim
        self.fusion = nn.Sequential(nn.LayerNorm(3 * visual_dim + audio_dim), nn.Linear(3 * visual_dim + audio_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden_dim, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.temporal = nn.Sequential(*[TemporalResidualBlock(hidden_dim, dilation, dropout) for dilation in (1, 2, 4, 8)])
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.slot_queries = nn.Parameter(torch.randn(self.num_slots, hidden_dim) * 0.02)
        layer = nn.TransformerDecoderLayer(hidden_dim, decoder_heads, 4 * hidden_dim, dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, decoder_layers)
        self.slot_norm = nn.LayerNorm(hidden_dim)
        self.class_head = nn.Linear(hidden_dim, NO_EVENT_INDEX + 1)
        self.time_head = nn.Linear(hidden_dim, 1)
        self.log_sigma_head = nn.Linear(hidden_dim, 1)
        self.quality_head = nn.Linear(hidden_dim, 1)
        self.family_head = nn.Linear(hidden_dim, len(FAMILY_LABELS))

    @property
    def core_start_index(self) -> int:
        return int(round(self.context_seconds * self.sample_fps))

    @property
    def core_frame_count(self) -> int:
        return int(round(self.core_seconds * self.sample_fps))

    def forward(self, frames: Tensor, audio: Tensor, valid_mask: Tensor | None = None) -> dict[str, Tensor]:
        if frames.ndim != 5 or frames.shape[2] != 3 or audio.shape[:2] != frames.shape[:2]:
            raise ValueError("frames=[B,T,3,H,W] and matching audio=[B,T,M] are required")
        visual = self.visual(frames)
        delta = F.pad(visual[:, 1:] - visual[:, :-1], (0, 0, 1, 0))
        fused = self.fusion(torch.cat((visual, delta, delta.abs(), self.audio(audio)), dim=-1))
        if valid_mask is not None:
            fused = fused * valid_mask.to(fused.dtype).unsqueeze(-1)
        memory, _ = self.gru(fused)
        memory = self.memory_norm(self.temporal(memory))
        queries = self.slot_queries.unsqueeze(0).expand(frames.shape[0], -1, -1)
        slots = self.slot_norm(self.decoder(queries, memory))
        start, end = self.core_start_index, self.core_start_index + self.core_frame_count
        if end > memory.shape[1]:
            raise ValueError("input is shorter than context + core")
        return {
            "class_logits": self.class_head(slots),
            "time_normalized": self.time_head(slots).sigmoid().squeeze(-1),
            "log_sigma": self.log_sigma_head(slots).squeeze(-1).clamp(-4, 2),
            "quality_logits": self.quality_head(slots).squeeze(-1),
            "slot_embeddings": slots,
            "core_memory": memory[:, start:end],
            "family_logits": self.family_head(memory[:, start:end]),
        }


def set_spotting_loss(outputs: Mapping[str, Tensor], targets: Sequence[Sequence[EventInstance]], *, no_event_weight: float = 0.1) -> dict[str, Tensor]:
    """Focal classification, matched time NLL, and quality supervision."""
    logits, times = outputs["class_logits"], outputs["time_normalized"]
    total, matched = logits.new_zeros(()), 0
    for batch_index, events in enumerate(targets):
        slots = logits.shape[1]
        classes = torch.full((slots,), NO_EVENT_INDEX, dtype=torch.long, device=logits.device)
        quality = torch.zeros(slots, device=logits.device)
        time_loss = logits.new_zeros(())
        if events:
            labels = torch.tensor([event.label for event in events], dtype=torch.long, device=logits.device)
            target_times = torch.tensor([event.time_normalized for event in events], device=logits.device)
            class_cost = -logits[batch_index].log_softmax(-1)[:, labels].transpose(0, 1)
            rows, cols = hungarian_assignment(class_cost + 3.0 * (times[batch_index][None] - target_times[:, None]).abs())
            classes[cols], quality[cols], matched = labels[rows], 1.0, matched + len(rows)
            sigma = outputs["log_sigma"][batch_index, cols].exp().clamp_min(1e-3)
            residual = times[batch_index, cols] - target_times[rows]
            time_loss = (F.smooth_l1_loss(residual / sigma, torch.zeros_like(residual), reduction="none") + outputs["log_sigma"][batch_index, cols]).mean()
        ce = F.cross_entropy(logits[batch_index], classes, reduction="none")
        confidence = logits[batch_index].softmax(-1).gather(1, classes[:, None]).squeeze(1)
        weights = torch.where(classes == NO_EVENT_INDEX, torch.full_like(ce, no_event_weight), torch.ones_like(ce))
        class_loss = (ce * (1 - confidence).square() * weights).sum() / weights.sum().clamp_min(1)
        quality_loss = F.binary_cross_entropy_with_logits(outputs["quality_logits"][batch_index], quality)
        total = total + class_loss + 3.0 * time_loss + 0.5 * quality_loss
    return {"loss": total / max(len(targets), 1), "matched_instances": logits.new_tensor(float(matched))}


def decode_slots(outputs: Mapping[str, Tensor], *, core_start_seconds: float, core_seconds: float, thresholds: Mapping[str, float], batch_index: int = 0) -> list[SlotPrediction]:
    """Decode slots independently.  There is deliberately no NMS in this function."""
    probabilities = outputs["class_logits"][batch_index].softmax(-1)
    quality = outputs["quality_logits"][batch_index].sigmoid()
    sigma = outputs["log_sigma"][batch_index].exp()
    predictions = []
    for index in range(probabilities.shape[0]):
        label_index = int(probabilities[index, :NO_EVENT_INDEX].argmax())
        label, score = SET_LABELS[label_index], float(probabilities[index, label_index] * quality[index])
        if score >= float(thresholds[label]):
            predictions.append(SlotPrediction(label, float(core_start_seconds + core_seconds * outputs["time_normalized"][batch_index, index]), score, float(sigma[index]), index))
    return predictions


def one_to_one_metrics(predictions: Sequence[SlotPrediction], targets: Sequence[EventInstance], *, core_start_seconds: float = 0.0, core_seconds: float = 1.0, tolerance_seconds: float = 3.0) -> dict[str, dict[str, float]]:
    """Metrics only: repeated nearby predictions remain FP instead of being removed."""
    report = {}
    for label_index, label in enumerate(SET_LABELS):
        available = [core_start_seconds + core_seconds * event.time_normalized for event in targets if event.label == label_index]
        tp = fp = 0
        for prediction in sorted((item for item in predictions if item.label == label), key=lambda item: item.score, reverse=True):
            if available:
                nearest = min(range(len(available)), key=lambda item: abs(available[item] - prediction.time_sec))
                if abs(available[nearest] - prediction.time_sec) <= tolerance_seconds:
                    tp += 1
                    available.pop(nearest)
                    continue
            fp += 1
        fn = len(available)
        report[label] = {"tp": tp, "fp": fp, "fn": fn, "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)}
    return report
