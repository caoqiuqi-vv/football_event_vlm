"""Inference, OOF-manifest, and calibration utilities for the two-stage pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import Tensor

from .data import PixelAudioSequenceDataset
from .set_spotting import SET_LABELS, SlotPrediction, TemporalSetSpotter, decode_slots


@dataclass(frozen=True)
class CandidateRecord:
    video_id: str
    core_start_seconds: float
    slot_index: int
    label: str
    time_sec: float
    score: float
    uncertainty_sec: float
    stage1_logits: list[float]
    slot_embedding: list[float]


def core_starts(duration_seconds: float, core_seconds: float = 30.0) -> list[float]:
    """Contiguous core intervals; no output interval overlaps another one."""
    count = max(int(torch.ceil(torch.tensor(duration_seconds / core_seconds))), 1)
    return [index * core_seconds for index in range(count)]


@torch.inference_mode()
def infer_video_candidates(
    model: TemporalSetSpotter,
    dataset: PixelAudioSequenceDataset,
    video_id: str,
    *,
    device: torch.device,
    thresholds: Mapping[str, float],
) -> list[CandidateRecord]:
    """Run contiguous cores and export independent slots without temporal NMS."""
    model.eval()
    frames_total = int(dataset.metadata[video_id]["frame_count"])
    duration = float(dataset.metadata[video_id]["duration_seconds"])
    fps, context = model.sample_fps, model.context_frames if hasattr(model, "context_frames") else int(round(model.context_seconds * model.sample_fps))
    records: list[CandidateRecord] = []
    for start_seconds in core_starts(duration, model.core_seconds):
        core_start = int(round(start_seconds * fps))
        raw = list(range(core_start - context, core_start - context + int(round((model.core_seconds + 2 * model.context_seconds) * fps))))
        valid = torch.tensor([[0 <= index < frames_total for index in raw]], dtype=torch.bool, device=device)
        indices = [min(max(index, 0), frames_total - 1) for index in raw]
        frames = dataset.normalize_eval_frames(dataset.read_frame_indices(video_id, indices)).unsqueeze(0).to(device)
        audio = torch.from_numpy(dataset.audio[video_id][indices].astype("float32", copy=True)).unsqueeze(0).to(device)
        output = model(frames, audio, valid)
        decoded = decode_slots(output, core_start_seconds=start_seconds, core_seconds=model.core_seconds, thresholds=thresholds)
        for item in decoded:
            index = item.slot_index
            records.append(CandidateRecord(video_id, start_seconds, index, item.label, item.time_sec, item.score, item.uncertainty_sec, output["class_logits"][0, index].float().cpu().tolist(), output["slot_embeddings"][0, index].float().cpu().tolist()))
    return records


def write_candidates(path: str | Path, candidates: Sequence[CandidateRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(asdict(item), ensure_ascii=False) + "\n" for item in candidates), encoding="utf-8")


def choose_threshold(scores: Tensor, matched: Tensor, *, target_recall: float) -> tuple[float, dict[str, float]]:
    """Maximum precision operating point subject to target recall, no suppression."""
    if scores.ndim != 1 or matched.shape != scores.shape:
        raise ValueError("scores and matched must be equal one-dimensional tensors")
    support = int(matched.sum())
    if support == 0:
        return 1.0, {"precision": 0.0, "recall": 0.0, "tp": 0.0, "fp": 0.0}
    order = scores.argsort(descending=True)
    ranked = matched[order].bool()
    required = int(torch.ceil(torch.tensor(target_recall * support)).item())
    cumulative = ranked.long().cumsum(0)
    eligible = torch.nonzero(cumulative >= required).flatten()
    if not len(eligible):
        cutoff = len(order) - 1
    else:
        cutoff = int(eligible[0])
    selected = ranked[: cutoff + 1]
    tp, fp = int(selected.sum()), int((~selected).sum())
    return float(scores[order[cutoff]]), {"precision": tp / max(tp + fp, 1), "recall": tp / support, "tp": float(tp), "fp": float(fp)}


def candidate_label_tensor(candidates: Sequence[CandidateRecord]) -> Tensor:
    return torch.tensor([SET_LABELS.index(item.label) for item in candidates], dtype=torch.long)
