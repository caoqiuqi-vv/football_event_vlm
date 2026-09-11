from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

from .annotations import LABELS
from .data import PixelAudioSequenceDataset
from .model import FootballE2ESpotter


@dataclass(frozen=True)
class Proposal:
    label: str
    timestamp: float
    score: float
    timeline_index: int


def decode_peaks(
    class_logits: Tensor,
    offsets: Tensor,
    timestamps: Tensor,
    *,
    nms_radius_seconds: Mapping[str, float],
) -> dict[str, list[Proposal]]:
    probability = class_logits.sigmoid()
    step = float(torch.median(timestamps[1:] - timestamps[:-1])) if timestamps.numel() > 1 else 1.0
    output = {}
    for label_index, label in enumerate(LABELS):
        scores = probability[:, label_index]
        radius = max(int(round(float(nms_radius_seconds[label]) / max(step, 1e-6))), 1)
        pooled = F.max_pool1d(
            scores.reshape(1, 1, -1), kernel_size=2 * radius + 1,
            stride=1, padding=radius,
        ).reshape(-1)
        left = torch.cat((scores[:1], scores[:-1]))
        right = torch.cat((scores[1:], scores[-1:]))
        indices = torch.nonzero(
            (scores >= pooled) & ((scores > left) | (scores > right)), as_tuple=False
        ).flatten()
        output[label] = [
            Proposal(
                label=label,
                timestamp=min(max(
                    float(timestamps[index] + offsets[index, label_index]),
                    float(timestamps[0]),
                ), float(timestamps[-1])),
                score=float(scores[index]), timeline_index=int(index),
            )
            for index in indices.tolist()
        ]
    return output


def scored_matches(
    proposals: list[Proposal], targets: list[float], tolerance_seconds: float
) -> tuple[list[float], list[bool], int]:
    unmatched = list(targets)
    scores, matches = [], []
    for proposal in sorted(proposals, key=lambda item: item.score, reverse=True):
        scores.append(proposal.score)
        if not unmatched:
            matches.append(False)
            continue
        distances = [abs(proposal.timestamp - target) for target in unmatched]
        index = min(range(len(unmatched)), key=lambda item: distances[item])
        matched = distances[index] <= tolerance_seconds
        matches.append(matched)
        if matched:
            unmatched.pop(index)
    return scores, matches, len(unmatched)


def average_precision(scores: list[float], matches: list[bool], support: int) -> float | None:
    if support <= 0:
        return None
    if not scores:
        return 0.0
    score = torch.tensor(scores)
    target = torch.tensor(matches, dtype=torch.bool)
    order = torch.argsort(score, descending=True, stable=True)
    ranked = target[order].float()
    precision = ranked.cumsum(0) / torch.arange(1, ranked.numel() + 1)
    return float((precision * ranked).sum() / support)


def operating_point(
    scores: list[float],
    matches: list[bool],
    support: int,
    target_recall: float,
    total_minutes: float,
) -> dict | None:
    if support <= 0:
        return None
    if not scores:
        return {
            "target_recall": target_recall, "target_achieved": False,
            "threshold": None, "recall": 0.0, "precision": None,
            "tp": 0, "fp": 0, "proposals": 0, "fp_per_minute": 0.0,
        }
    score = torch.tensor(scores)
    target = torch.tensor(matches, dtype=torch.bool)
    order = torch.argsort(score, descending=True, stable=True)
    required = int(math.ceil(target_recall * support - 1e-12))
    cumulative = target[order].long().cumsum(0)
    reaching = torch.nonzero(cumulative >= required, as_tuple=False).flatten()
    cutoff = int(reaching[0]) if reaching.numel() else score.numel() - 1
    threshold = float(score[order][cutoff])
    selected = score >= threshold
    tp = int(target[selected].sum())
    count = int(selected.sum())
    fp = count - tp
    recall = tp / support
    return {
        "target_recall": float(target_recall),
        "target_achieved": recall >= float(target_recall),
        "threshold": threshold, "recall": recall,
        "precision": tp / count if count else None,
        "tp": tp, "fp": fp, "proposals": count,
        "fp_per_minute": fp / max(total_minutes, 1e-9),
    }


@torch.inference_mode()
def infer_full_video(
    model: FootballE2ESpotter,
    dataset: PixelAudioSequenceDataset,
    video_id: str,
    *,
    device: torch.device,
    core_frames: int = 128,
    halo_frames: int = 12,
    amp_dtype: torch.dtype | None = torch.bfloat16,
) -> tuple[Tensor, Tensor, Tensor]:
    """Encode each core frame once; only a small feature-map halo is repeated."""
    frame_count = int(dataset.metadata[video_id]["frame_count"])
    visual_parts = []
    autocast = device.type == "cuda" and amp_dtype is not None
    for core_start in range(0, frame_count, core_frames):
        core_end = min(core_start + core_frames, frame_count)
        read_start = max(core_start - halo_frames, 0)
        read_end = min(core_end + halo_frames, frame_count)
        indices = list(range(read_start, read_end))
        frames = dataset.normalize_eval_frames(
            dataset.read_frame_indices(video_id, indices)
        ).unsqueeze(0).to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype if amp_dtype is not None else torch.float32,
            enabled=autocast,
        ):
            visual = model.encode_visual(frames)[0]
        left = core_start - read_start
        right = left + (core_end - core_start)
        visual_parts.append(visual[left:right].float().cpu())
    visual = torch.cat(visual_parts, dim=0)
    if visual.shape[0] != frame_count:
        raise RuntimeError("full-video visual coverage is not exactly once per core frame")
    audio = torch.from_numpy(np.array(
        dataset.audio[video_id], dtype=np.float32, copy=True
    ))
    valid = torch.ones((1, frame_count), dtype=torch.bool, device=device)
    with torch.autocast(
        device_type=device.type,
        dtype=amp_dtype if amp_dtype is not None else torch.float32,
        enabled=autocast,
    ):
        outputs = model.forward_from_embeddings(
            visual.unsqueeze(0).to(device), audio.unsqueeze(0).to(device), valid
        )
    timestamps = (torch.arange(frame_count, dtype=torch.float32) + 0.5) / dataset.sample_fps
    return (
        outputs["class_logits"][0].float().cpu(),
        outputs["offsets"][0].float().cpu(),
        timestamps,
    )


@torch.inference_mode()
def evaluate_full_videos(
    model: FootballE2ESpotter,
    dataset: PixelAudioSequenceDataset,
    *,
    device: torch.device,
    nms_radius_seconds: Mapping[str, float],
    target_recall: Mapping[str, float],
    minimum_precision_at_target_recall: Mapping[str, float],
    maximum_fp_per_minute: Mapping[str, float],
    tolerances_seconds: tuple[float, ...] = (2.0, 3.0, 5.0),
    operating_tolerance_seconds: float = 3.0,
    uniform_baseline_intervals_seconds: tuple[float, ...] = (3.0, 4.0),
    core_frames: int = 128,
    halo_frames: int = 12,
    amp_dtype: torch.dtype | None = torch.bfloat16,
) -> dict:
    model.eval()
    cells = {
        tolerance: {label: {"scores": [], "matches": [], "support": 0} for label in LABELS}
        for tolerance in tolerances_seconds
    }
    uniform = {
        interval: {
            tolerance: {label: {"matches": 0, "proposals": 0, "support": 0} for label in LABELS}
            for tolerance in tolerances_seconds
        } for interval in uniform_baseline_intervals_seconds
    }
    positive_peak_scores = {label: [] for label in LABELS}
    proposal_count = {label: 0 for label in LABELS}
    operating_records = {label: [] for label in LABELS}
    video_rows = []
    total_seconds = 0.0
    for video_id in dataset.video_ids:
        logits, offsets, timestamps = infer_full_video(
            model, dataset, video_id, device=device, core_frames=core_frames,
            halo_frames=halo_frames, amp_dtype=amp_dtype,
        )
        proposals = decode_peaks(
            logits, offsets, timestamps, nms_radius_seconds=nms_radius_seconds
        )
        probability = logits.sigmoid()
        events = dataset.events[video_id]
        duration = float(dataset.metadata[video_id]["duration_seconds"])
        total_seconds += duration
        row = {"video_id": video_id, "duration_seconds": duration, "classes": {}}
        for label_index, label in enumerate(LABELS):
            targets = [event.timestamp for event in events if event.label == label]
            proposal_count[label] += len(proposals[label])
            row["classes"][label] = {"support": len(targets), "local_maxima": len(proposals[label])}
            for target in targets:
                near = (timestamps - target).abs() <= operating_tolerance_seconds
                positive_peak_scores[label].append(
                    float(probability[near, label_index].max()) if near.any() else 0.0
                )
            for tolerance in tolerances_seconds:
                scores, matches, missed = scored_matches(
                    proposals[label], targets, float(tolerance)
                )
                cell = cells[tolerance][label]
                cell["scores"].extend(scores)
                cell["matches"].extend(matches)
                cell["support"] += len(targets)
                row["classes"][label][f"missed_at_{tolerance:g}s"] = missed
                if abs(tolerance - operating_tolerance_seconds) < 1e-9:
                    operating_records[label].append({
                        "video_id": video_id, "support": len(targets),
                        "scores": scores, "matches": matches, "missed": missed,
                    })
            for interval in uniform_baseline_intervals_seconds:
                count = max(int(math.ceil(duration / interval)), 1)
                baseline = [
                    Proposal(label, min((index + 0.5) * interval, duration), 1.0, index)
                    for index in range(count)
                ]
                for tolerance in tolerances_seconds:
                    _, matches, _ = scored_matches(baseline, targets, tolerance)
                    cell = uniform[interval][tolerance][label]
                    cell["matches"] += sum(matches)
                    cell["proposals"] += len(baseline)
                    cell["support"] += len(targets)
        video_rows.append(row)

    total_minutes = total_seconds / 60.0
    tolerance_reports = {}
    for tolerance, label_cells in cells.items():
        report = {}
        for label, cell in label_cells.items():
            support = int(cell["support"])
            op = operating_point(
                cell["scores"], cell["matches"], support,
                float(target_recall[label]), total_minutes,
            )
            false_scores = [
                score for score, matched in zip(cell["scores"], cell["matches"])
                if not matched
            ]
            positives = positive_peak_scores[label]
            report[label] = {
                "support": support,
                "average_precision": average_precision(cell["scores"], cell["matches"], support),
                "operating_point": op,
                "local_maxima_per_minute": proposal_count[label] / max(total_minutes, 1e-9),
                "positive_peak_p10": float(torch.tensor(positives).quantile(0.1)) if positives else None,
                "false_peak_p90": float(torch.tensor(false_scores).quantile(0.9)) if false_scores else None,
            }
        tolerance_reports[f"{tolerance:g}s"] = report

    uniform_reports = {}
    for interval, tolerance_cells in uniform.items():
        by_tolerance = {}
        for tolerance, label_cells in tolerance_cells.items():
            by_label = {}
            for label, cell in label_cells.items():
                support, matches, count = cell["support"], cell["matches"], cell["proposals"]
                by_label[label] = {
                    "support": support, "matches": matches,
                    "recall": matches / support if support else None,
                    "precision": matches / count if count else None,
                    "proposals_per_minute": count / max(total_minutes, 1e-9),
                    "fp_per_minute": (count - matches) / max(total_minutes, 1e-9),
                }
            by_tolerance[f"{tolerance:g}s"] = by_label
        uniform_reports[f"{interval:g}s"] = by_tolerance

    primary = tolerance_reports[f"{operating_tolerance_seconds:g}s"]
    achieved = {
        label: bool(primary[label]["operating_point"]["target_achieved"])
        for label in LABELS
    }
    deficits = {
        label: max(
            float(target_recall[label]) - float(primary[label]["operating_point"]["recall"]), 0.0
        ) for label in LABELS
    }
    gates = {}
    violations = {}
    for label in LABELS:
        op = primary[label]["operating_point"]
        precision = float(op["precision"] or 0.0)
        fp_rate = float(op["fp_per_minute"] or 0.0)
        precision_floor = float(minimum_precision_at_target_recall[label])
        fp_ceiling = float(maximum_fp_per_minute[label])
        violation = (
            max(precision_floor - precision, 0.0) / max(precision_floor, 1e-9)
            + max(fp_rate - fp_ceiling, 0.0) / max(fp_ceiling, 1e-9)
        )
        violations[label] = violation
        gates[label] = achieved[label] and violation == 0.0
    selection = [
        int(achieved["shot"]),
        sum(int(achieved[label]) for label in LABELS if label != "shot"),
        -deficits["shot"],
        -sum(deficits[label] for label in LABELS if label != "shot"),
        sum(int(value) for value in gates.values()),
        -sum(violations.values()),
        sum(float(primary[label]["operating_point"]["precision"] or 0.0) for label in LABELS) / len(LABELS),
        sum(float(primary[label]["average_precision"] or 0.0) for label in LABELS) / len(LABELS),
    ]
    return {
        "schema": "football_e2e_spotter.full_video_evaluation.v1",
        "video_count": len(dataset.video_ids), "video_ids": list(dataset.video_ids),
        "total_minutes": total_minutes,
        "operating_tolerance_seconds": operating_tolerance_seconds,
        "target_recall": dict(target_recall),
        "minimum_precision_at_target_recall": dict(minimum_precision_at_target_recall),
        "maximum_fp_per_minute": dict(maximum_fp_per_minute),
        "selection_tuple": selection,
        "selection_rule": "shot_recall_then_other_recalls_then_deployment_gates_then_precision_then_ap",
        "tolerances": tolerance_reports,
        "uniform_time_sampling_baselines": uniform_reports,
        "operating_match_records": operating_records,
        "videos": video_rows,
        "hard_sample_mining": False,
        "detection_or_tracking_input": False,
    }

