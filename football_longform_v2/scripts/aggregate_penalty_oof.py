from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from football_longform_v2.evaluation import average_precision, operating_point_at_recall  # noqa: E402


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float] | None:
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def aggregate_oof_reports(oof: dict, reports: list[dict]) -> dict:
    if oof.get("schema") != "football_longform_v2.penalty_oof_folds.v1":
        raise ValueError("unsupported OOF manifest schema")
    expected = {int(item["fold"]): item for item in oof.get("folds", [])}
    observed = {int(report.get("fold", -1)): report for report in reports}
    if set(observed) != set(expected) or len(observed) != len(reports):
        raise ValueError("OOF reports must contain each manifest fold exactly once")
    all_ids: list[str] = []
    all_scores: list[float] = []
    all_labels: list[bool] = []
    support = 0
    total_minutes = 0.0
    normalized_reports = []
    for fold_index in sorted(expected):
        fold = expected[fold_index]
        report = observed[fold_index]
        if report.get("schema") != "football_longform_v2.penalty_oof_fold_report.v1":
            raise ValueError(f"fold {fold_index}: unsupported report schema")
        if not report.get("development_only"):
            raise ValueError(f"fold {fold_index}: report is not development-only")
        if report.get("fixed_test_overlap_count") != 0 or report.get("fixed_test_labels_or_predictions_used"):
            raise ValueError(f"fold {fold_index}: fixed test contamination")
        if report.get("evaluated_video_ids") != fold["validation_media_ids"]:
            raise ValueError(f"fold {fold_index}: evaluated IDs disagree with OOF manifest")
        metrics = report.get("penalty", {})
        fold_support = int(metrics.get("point_support") or 0)
        if fold_support != int(fold["validation_penalty_event_count"]):
            raise ValueError(f"fold {fold_index}: penalty support mismatch")
        scores = [float(value) for value in metrics.get("raw_candidate_scores", [])]
        labels = [bool(value) for value in metrics.get("raw_candidate_match_labels_at_2s", [])]
        if len(scores) != len(labels):
            raise ValueError(f"fold {fold_index}: raw score/label length mismatch")
        minutes = float(report.get("evaluated_duration_minutes") or 0.0)
        if minutes <= 0.0:
            raise ValueError(f"fold {fold_index}: missing evaluated duration")
        all_ids.extend(report["evaluated_video_ids"])
        all_scores.extend(scores)
        all_labels.extend(labels)
        support += fold_support
        total_minutes += minutes
        normalized_reports.append((fold_index, fold_support, scores, labels))
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("OOF validation videos overlap across folds")
    if len(all_ids) != int(oof.get("source_train_video_count", -1)):
        raise ValueError("OOF reports do not cover every canonical train video exactly once")
    if support != int(oof.get("source_penalty_event_count", -1)):
        raise ValueError("aggregate penalty support disagrees with OOF manifest")

    operating = operating_point_at_recall(
        all_scores, all_labels, positive_count=support, target_recall=0.85,
        total_minutes=total_minutes,
    )
    threshold = operating.get("threshold") if operating else None
    per_fold_at_global_threshold = []
    if threshold is not None:
        for fold_index, fold_support, scores, labels in normalized_reports:
            selected = [index for index, score in enumerate(scores) if score >= float(threshold)]
            true_positives = sum(labels[index] for index in selected)
            proposals = len(selected)
            per_fold_at_global_threshold.append({
                "fold": fold_index,
                "support": fold_support,
                "tp": true_positives,
                "proposals": proposals,
                "recall": true_positives / fold_support if fold_support else None,
                "precision": true_positives / proposals if proposals else None,
            })
    aggregate_tp = int(operating.get("tp") or 0) if operating else 0
    confidence = wilson_interval(aggregate_tp, support)
    candidate_tp = sum(all_labels)
    candidate_ceiling = candidate_tp / support if support else None
    target_met = bool(operating and operating.get("target_achieved"))
    if candidate_ceiling is None or candidate_ceiling < 0.85:
        bottleneck = "candidate_recall_bottleneck"
        action = "upgrade_restart_state_and_short_term_rgb_representation"
    elif not target_met:
        bottleneck = "ranking_bottleneck"
        action = "penalty_conditioned_verifier_and_hard_positive_mining"
    else:
        bottleneck = "target_met_optimize_precision"
        action = "validate_full-train_score_transfer_then_reduce_false_positives"
    return {
        "schema": "football_longform_v2.penalty_oof_aggregate.v1",
        "development_only": True,
        "fold_count": len(reports),
        "evaluated_video_count": len(all_ids),
        "penalty_support": support,
        "candidate_ceiling_recall_at_2s": candidate_ceiling,
        "point_ap_at_2s": average_precision(
            torch.tensor(all_scores), torch.tensor(all_labels, dtype=torch.bool),
            positive_count=support,
        ),
        "operating_point_for_target_recall_at_2s": operating,
        "recall_wilson_95_interval_at_operating_point": (
            {"lower": confidence[0], "upper": confidence[1]} if confidence else None
        ),
        "per_fold_at_global_oof_threshold": per_fold_at_global_threshold,
        "worst_fold_recall_at_global_oof_threshold": min(
            (item["recall"] for item in per_fold_at_global_threshold if item["recall"] is not None),
            default=None,
        ),
        "bottleneck": bottleneck,
        "recommended_next_action": action,
        "fixed_test_labels_or_predictions_used": False,
        "deployment_threshold_eligible": False,
        "deployment_threshold_block_reason": (
            "OOF scores come from five fold-specific models; validate score transfer on a "
            "separate positive calibration set before freezing a full-train deployment threshold."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate five leakage-safe penalty OOF reports.")
    parser.add_argument("--oof-manifest", required=True)
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    oof_path = Path(args.oof_manifest).expanduser().resolve()
    oof = json.loads(oof_path.read_text(encoding="utf-8"))
    reports = [
        json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
        for path in args.reports
    ]
    result = aggregate_oof_reports(oof, reports)
    result["oof_manifest"] = str(oof_path)
    result["fold_reports"] = [str(Path(path).expanduser().resolve()) for path in args.reports]
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "per_fold_at_global_oof_threshold"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
