from __future__ import annotations

"""Strict long-video evaluation for the sequential VideoMAE locator."""

import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor

from .chunk_data import SequentialChunkDataset
from .decoding import Proposal
from .evaluation import average_precision, operating_point_at_recall, scored_point_matches


def decode_class_peaks(
    class_logits: Tensor,
    offsets: Tensor,
    timestamps: Tensor,
    labels: tuple[str, ...],
    *,
    nms_radius_seconds: Mapping[str, float],
) -> dict[str, list[Proposal]]:
    """Return every class-local maximum; thresholds are selected after aggregation."""
    if class_logits.shape != offsets.shape or class_logits.shape[0] != timestamps.numel():
        raise ValueError("class logits, offsets and timestamps disagree")
    probability = class_logits.sigmoid()
    step = (
        float(torch.median(timestamps[1:] - timestamps[:-1]).clamp_min(1e-6))
        if timestamps.numel() > 1 else 1.0
    )
    output: dict[str, list[Proposal]] = {}
    for label_index, label in enumerate(labels):
        scores = probability[:, label_index]
        radius_steps = max(int(round(float(nms_radius_seconds[label]) / step)), 1)
        pooled = F.max_pool1d(
            scores.reshape(1, 1, -1),
            kernel_size=2 * radius_steps + 1,
            stride=1,
            padding=radius_steps,
        ).reshape(-1)
        left = torch.cat((scores[:1], scores[:-1]))
        right = torch.cat((scores[1:], scores[-1:]))
        # A completely flat low-confidence region is not an event proposal.
        # Requiring a strict rise on at least one side also prevents every
        # point of a quantized BF16 plateau from becoming a duplicate peak.
        indices = torch.nonzero(
            (scores >= pooled) & ((scores > left) | (scores > right)),
            as_tuple=False,
        ).flatten()
        proposals = []
        for index in indices.tolist():
            timestamp = float(timestamps[index] + offsets[index, label_index])
            timestamp = min(max(timestamp, float(timestamps[0])), float(timestamps[-1]))
            proposals.append(Proposal(
                batch_index=0,
                family=label,
                timestamp=timestamp,
                score=float(scores[index]),
                timeline_index=index,
            ))
        output[label] = proposals
    return output


@torch.inference_mode()
def evaluate_sequential_chunk_locator(
    model: torch.nn.Module,
    dataset: SequentialChunkDataset,
    *,
    device: torch.device,
    nms_radius_seconds: Mapping[str, float],
    target_recall: Mapping[str, float],
    minimum_precision_at_target_recall: Mapping[str, float] | None = None,
    maximum_fp_per_minute: Mapping[str, float] | None = None,
    tolerances_seconds: tuple[float, ...] = (2.0, 3.0, 5.0),
    operating_tolerance_seconds: float = 3.0,
    uniform_baseline_intervals_seconds: tuple[float, ...] = (3.0, 4.0),
    amp_dtype: torch.dtype | None = torch.bfloat16,
) -> dict:
    """Evaluate complete held-out videos with per-video one-to-one matching."""
    labels = dataset.labels
    model.eval()
    per_tolerance = {
        float(tolerance): {
            label: {"scores": [], "matches": [], "support": 0}
            for label in labels
        }
        for tolerance in tolerances_seconds
    }
    positive_peak_scores = {label: [] for label in labels}
    total_seconds = 0.0
    proposal_count = {label: 0 for label in labels}
    operating_match_records = {label: [] for label in labels}
    uniform_cells = {
        float(interval): {
            float(tolerance): {
                label: {"matches": 0, "proposals": 0, "support": 0}
                for label in labels
            }
            for tolerance in tolerances_seconds
        }
        for interval in uniform_baseline_intervals_seconds
    }
    video_rows = []
    autocast_enabled = device.type == "cuda" and amp_dtype is not None
    for video_id in dataset.video_ids:
        features, timestamps = dataset.full_timeline(video_id)
        total_seconds += max(float(timestamps[-1] - timestamps[0]), dataset.step_seconds)
        features = features.unsqueeze(0).to(device, non_blocking=True)
        valid = torch.ones(features.shape[:2], dtype=torch.bool, device=device)
        chunk_phase = torch.arange(
            features.shape[1], device=device, dtype=torch.long
        ).remainder(dataset.tubelets_per_chunk).unsqueeze(0)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype if amp_dtype is not None else torch.float32,
            enabled=autocast_enabled,
        ):
            outputs = model(features, valid, chunk_phase)
        class_logits = outputs["class_logits"][0].float().cpu()
        offsets = outputs["offsets"][0].float().cpu()
        proposals = decode_class_peaks(
            class_logits, offsets, timestamps, labels,
            nms_radius_seconds=nms_radius_seconds,
        )
        events = dataset.events[video_id]
        video_start = max(float(timestamps[0]) - 0.5 * dataset.step_seconds, 0.0)
        video_end = float(timestamps[-1]) + 0.5 * dataset.step_seconds
        row = {"video_id": video_id, "tubelet_count": int(timestamps.numel()), "classes": {}}
        probability = class_logits.sigmoid()
        for label_index, label in enumerate(labels):
            targets = [event.timestamp for event in events if event.label == label]
            proposal_count[label] += len(proposals[label])
            row["classes"][label] = {
                "support": len(targets), "proposal_count": len(proposals[label])
            }
            for target in targets:
                near = (timestamps - float(target)).abs() <= operating_tolerance_seconds
                positive_peak_scores[label].append(
                    float(probability[near, label_index].max()) if near.any() else 0.0
                )
            for tolerance in tolerances_seconds:
                scores, matches, missed = scored_point_matches(
                    proposals[label], targets, tolerance_seconds=float(tolerance)
                )
                cell = per_tolerance[float(tolerance)][label]
                cell["scores"].extend(scores)
                cell["matches"].extend(matches)
                cell["support"] += len(targets)
                row["classes"][label][f"missed_at_{tolerance:g}s"] = missed
                if abs(float(tolerance) - float(operating_tolerance_seconds)) <= 1e-9:
                    operating_match_records[label].append({
                        "video_id": video_id,
                        "support": len(targets),
                        "scores": scores,
                        "matches": matches,
                        "missed": missed,
                    })
            for interval in uniform_baseline_intervals_seconds:
                interval = float(interval)
                count = max(int(math.ceil((video_end - video_start) / interval)), 1)
                uniform_proposals = [
                    Proposal(
                        batch_index=0,
                        family=label,
                        timestamp=min(video_start + (index + 0.5) * interval, video_end),
                        score=1.0,
                        timeline_index=index,
                    )
                    for index in range(count)
                    if video_start + index * interval < video_end
                ]
                for tolerance in tolerances_seconds:
                    _, matches, _ = scored_point_matches(
                        uniform_proposals, targets, tolerance_seconds=float(tolerance)
                    )
                    cell = uniform_cells[interval][float(tolerance)][label]
                    cell["matches"] += sum(matches)
                    cell["proposals"] += len(uniform_proposals)
                    cell["support"] += len(targets)
        video_rows.append(row)

    total_minutes = total_seconds / 60.0
    tolerance_reports = {}
    for tolerance, cells in per_tolerance.items():
        label_reports = {}
        for label, cell in cells.items():
            support = int(cell["support"])
            ap = average_precision(
                torch.tensor(cell["scores"]),
                torch.tensor(cell["matches"]),
                positive_count=support,
            )
            operating_point = operating_point_at_recall(
                cell["scores"], cell["matches"], positive_count=support,
                target_recall=float(target_recall[label]), total_minutes=total_minutes,
            )
            false_scores = [
                score for score, match in zip(cell["scores"], cell["matches"]) if not match
            ]
            positives = positive_peak_scores[label]
            label_reports[label] = {
                "support": support,
                "average_precision": ap,
                "operating_point": operating_point,
                "local_maxima_per_minute": proposal_count[label] / max(total_minutes, 1e-9),
                "positive_peak_p10": (
                    float(torch.tensor(positives).quantile(0.1)) if positives else None
                ),
                "false_peak_p90": (
                    float(torch.tensor(false_scores).quantile(0.9)) if false_scores else None
                ),
            }
        tolerance_reports[f"{tolerance:g}s"] = label_reports

    operating_key = f"{float(operating_tolerance_seconds):g}s"
    primary = tolerance_reports[operating_key]
    supported = [label for label in labels if primary[label]["support"] > 0]
    recall_achieved = {
        label: bool(primary[label]["operating_point"]["target_achieved"])
        for label in supported
    }
    shot_supported = "shot" in supported
    shot_achieved = int(recall_achieved.get("shot", False))
    other_achieved = sum(
        int(recall_achieved[label]) for label in supported if label != "shot"
    )
    precisions = [
        float(primary[label]["operating_point"]["precision"] or 0.0)
        for label in supported
    ]
    aps = [float(primary[label]["average_precision"] or 0.0) for label in supported]
    deficits = [
        max(
            float(target_recall[label])
            - float(primary[label]["operating_point"]["recall"]),
            0.0,
        )
        for label in supported
    ]
    shot_deficit = (
        max(
            float(target_recall["shot"])
            - float(primary["shot"]["operating_point"]["recall"]),
            0.0,
        )
        if shot_supported else 0.0
    )
    other_deficit = sum(
        max(
            float(target_recall[label])
            - float(primary[label]["operating_point"]["recall"]),
            0.0,
        )
        for label in supported if label != "shot"
    )
    gate_passed = 0
    normalized_gate_violation = 0.0
    if minimum_precision_at_target_recall is not None and maximum_fp_per_minute is not None:
        for label in supported:
            operating = primary[label]["operating_point"]
            precision = float(operating["precision"] or 0.0)
            fp_per_minute = float(operating["fp_per_minute"] or 0.0)
            precision_floor = float(minimum_precision_at_target_recall[label])
            fp_ceiling = float(maximum_fp_per_minute[label])
            precision_violation = max(precision_floor - precision, 0.0) / max(precision_floor, 1e-9)
            fp_violation = max(fp_per_minute - fp_ceiling, 0.0) / max(fp_ceiling, 1e-9)
            passed = recall_achieved[label] and precision_violation == 0.0 and fp_violation == 0.0
            gate_passed += int(passed)
            normalized_gate_violation += precision_violation + fp_violation
    selection_tuple = [
        shot_achieved,
        other_achieved,
        -shot_deficit,
        -other_deficit,
        gate_passed,
        -normalized_gate_violation,
        sum(precisions) / max(len(precisions), 1),
        sum(aps) / max(len(aps), 1),
    ]
    uniform_baselines = {}
    for interval, tolerance_cells in uniform_cells.items():
        reports = {}
        for tolerance, cells in tolerance_cells.items():
            class_reports = {}
            for label, cell in cells.items():
                support = int(cell["support"])
                matches = int(cell["matches"])
                proposals = int(cell["proposals"])
                class_reports[label] = {
                    "support": support,
                    "matches": matches,
                    "recall": matches / support if support else None,
                    "precision": matches / proposals if proposals else None,
                    "proposals_per_minute": proposals / max(total_minutes, 1e-9),
                    "fp_per_minute": (proposals - matches) / max(total_minutes, 1e-9),
                }
            reports[f"{tolerance:g}s"] = class_reports
        uniform_baselines[f"{interval:g}s"] = reports
    return {
        "schema": "football_longform_v2.sequential_chunk_evaluation.v1",
        "video_count": len(dataset.video_ids),
        "video_ids": list(dataset.video_ids),
        "total_minutes": total_minutes,
        "operating_tolerance_seconds": float(operating_tolerance_seconds),
        "target_recall": dict(target_recall),
        "minimum_precision_at_target_recall": (
            dict(minimum_precision_at_target_recall)
            if minimum_precision_at_target_recall is not None else None
        ),
        "maximum_fp_per_minute": (
            dict(maximum_fp_per_minute) if maximum_fp_per_minute is not None else None
        ),
        "selection_tuple": selection_tuple,
        "selection_rule": (
            "shot_recall_gate_then_other_recall_count_then_recall_deficits_then_"
            "deployment_gates_then_precision_then_ap_at_operating_tolerance"
        ),
        "tolerances": tolerance_reports,
        "uniform_time_sampling_baselines": uniform_baselines,
        "operating_match_records": operating_match_records,
        "videos": video_rows,
    }
