from __future__ import annotations

import math
from pathlib import Path

from .decoding import Proposal


OPERATING_POINTS_SCHEMA = "football_longform_v2.operating_points.v1"


def build_frozen_operating_points(
    report: dict, *, checkpoint: str | Path, checkpoint_sha256: str,
    report_sha256: str,
) -> dict:
    """Freeze calibration-only thresholds before any external-test inference."""
    if report.get("evaluation_split") != "calibration":
        raise ValueError("operating points must come from calibration")
    if not report.get("strict_complete_split") or report.get("evaluated_video_count") != 18:
        raise ValueError("operating points require the strict complete 18-video calibration split")
    frozen_labels: dict[str, dict] = {}
    for label, metrics in report.get("labels", {}).items():
        support = int(metrics.get("point_support") or 0)
        operating = metrics.get("operating_point_for_target_recall_at_2s")
        if support == 0:
            frozen_labels[label] = {
                "status": "no_calibration_support", "threshold": None,
                "target_recall": 0.90 if label == "shot" else 0.85,
                "calibration_support": 0,
                "fallback_max_proposals_per_minute": metrics.get("max_proposals_per_minute"),
            }
            continue
        if not isinstance(operating, dict):
            raise ValueError(f"missing operating point for supported label {label}")
        frozen_labels[label] = {
            "status": (
                "calibrated_target_achieved"
                if operating.get("target_achieved") else "target_unreachable_use_candidate_ceiling"
            ),
            "threshold": operating.get("threshold"),
            "target_recall": operating.get("target_recall"),
            "calibration_recall": operating.get("recall"),
            "calibration_precision": operating.get("precision"),
            "calibration_fp_per_minute": operating.get("fp_per_minute"),
            "calibration_support": support,
            "candidate_ceiling_recall_at_2s": metrics.get("candidate_ceiling_recall_at_2s"),
            "fallback_max_proposals_per_minute": metrics.get("max_proposals_per_minute"),
        }
    if not frozen_labels:
        raise ValueError("calibration report has no label metrics")
    return {
        "schema": OPERATING_POINTS_SCHEMA,
        "source": "strict_calibration_only",
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "calibration_report_sha256": report_sha256,
        "class_proposal_route": report.get("class_proposal_route"),
        "class_local_search_radius_seconds": report.get("class_local_search_radius_seconds"),
        "labels": frozen_labels,
        "external_threshold_tuning_forbidden": True,
    }


def apply_frozen_operating_points(
    proposals: dict[str, list[Proposal]], operating_points: dict, *,
    duration_minutes: float, video_id: str,
) -> list[dict]:
    """Filter proposals without labels and emit a human-review-friendly event contract."""
    if operating_points.get("schema") != OPERATING_POINTS_SCHEMA:
        raise ValueError("unsupported operating points schema")
    predictions: list[dict] = []
    for label, candidates in proposals.items():
        rule = operating_points.get("labels", {}).get(label)
        if not isinstance(rule, dict):
            raise ValueError(f"no frozen operating point for label {label}")
        threshold = rule.get("threshold")
        if threshold is not None:
            selected = [item for item in candidates if item.score >= float(threshold)]
            selection = "frozen_calibration_threshold"
        else:
            budget = float(rule.get("fallback_max_proposals_per_minute") or 0.0)
            limit = max(1, int(math.ceil(max(duration_minutes, 1.0 / 60.0) * budget)))
            selected = sorted(candidates, key=lambda item: item.score, reverse=True)[:limit]
            selection = "uncalibrated_budget_fallback"
        for item in selected:
            predictions.append({
                "video_id": video_id, "label": label,
                "timestamp_seconds": item.timestamp, "score": item.score,
                "selection": selection, "threshold": threshold,
                "calibration_status": rule.get("status"),
                "review_priority": (
                    "high" if rule.get("status") != "calibrated_target_achieved" else "normal"
                ),
                "uncertainty": 1.0 - item.score,
            })
    return sorted(predictions, key=lambda item: (item["timestamp_seconds"], item["label"]))
