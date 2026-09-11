#!/usr/bin/env python
"""Leakage-safe candidate verifier protocol for football long-video evaluation.

The calibration split is used for video-grouped OOF fitting and threshold
selection.  The external split is read only after the final verifier and all
thresholds have been frozen.  Dense DINO inference is deliberately outside of
this script; this stage consumes its cached CSV artifacts and is CPU-only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fit_football_candidate_reranker import (  # noqa: E402
    Candidate,
    apply_models,
    clip_scores,
    collect_gt,
    evaluate_predictions,
    fit_final_models,
    fit_oof_models,
    load_candidates,
)
from scripts.select_validation_thresholds_cached import evaluate_selected  # noqa: E402


DEFAULT_LABELS = ("shot", "save", "set_piece")
REQUIRED_FILES = ("window_predictions.csv", "frame_event_logits.csv", "gt_events.csv")


def read_ids(path: Path) -> list[str]:
    values = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate video ids in {path}")
    return values


def validate_cache(run_dir: Path, video_ids: Sequence[str]) -> dict[str, Any]:
    missing: dict[str, list[str]] = {}
    for video_id in video_ids:
        absent = [name for name in REQUIRED_FILES if not (run_dir / video_id / name).exists()]
        if absent:
            missing[video_id] = absent
    return {
        "run_dir": str(run_dir.resolve()),
        "requested_videos": len(video_ids),
        "complete_videos": len(video_ids) - len(missing),
        "missing": missing,
    }


def split_fingerprint(video_ids: Sequence[str]) -> str:
    payload = "\n".join(sorted(video_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_durations(run_dir: Path, video_ids: Sequence[str]) -> dict[str, float]:
    """Read exact evaluator duration instead of inferring it from windows."""
    result: dict[str, float] = {}
    for video_id in video_ids:
        summary_path = run_dir / video_id / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            duration = float(summary.get("duration_sec", 0.0) or 0.0)
            if duration > 0:
                result[video_id] = duration
    return result


def threshold_candidates(scores: Sequence[float], grid_size: int) -> list[float]:
    values = np.asarray(sorted(set(float(value) for value in scores)), dtype=np.float64)
    if not len(values):
        return []
    if len(values) > grid_size:
        indices = np.linspace(0, len(values) - 1, grid_size).round().astype(int)
        values = values[indices]
    epsilon = max(1e-12, abs(float(values[0])) * 1e-12)
    return [float(values[0] - epsilon), *[float(value) for value in values]]


def select_point_thresholds(
    candidates: Sequence[Candidate],
    scores: dict[tuple[str, str, int], float],
    gt_by_video: dict[str, dict[str, list[float]]],
    labels: Sequence[str],
    requested_floor: float,
    *,
    nms_radius_sec: float,
    match_tolerance_sec: float,
    grid_size: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Select each class threshold on calibration only.

    If the candidate/NMS ceiling is below the requested floor, the threshold is
    selected at the reachable ceiling and the shortfall is explicit in output.
    """
    thresholds: dict[str, float] = {}
    diagnostics: dict[str, Any] = {}
    for label in labels:
        label_candidates = [item for item in candidates if item.label == label]
        values = threshold_candidates([scores[item.key] for item in label_candidates], grid_size)
        if not values:
            raise RuntimeError(f"no verifier scores for label={label}")
        evaluated: list[dict[str, Any]] = []
        for threshold in values:
            metrics = evaluate_predictions(
                candidates,
                scores,
                {item: threshold if item == label else 2.0 for item in labels},
                gt_by_video,
                labels,
                nms_radius_sec=nms_radius_sec,
                match_tolerance_sec=match_tolerance_sec,
                use_peak_time=True,
            )["per_class"][label]
            evaluated.append({"threshold": threshold, **metrics})
        ceiling = max(float(item["recall"]) for item in evaluated)
        effective_floor = min(float(requested_floor), ceiling)
        feasible = [item for item in evaluated if float(item["recall"]) + 1e-12 >= effective_floor]
        best = max(
            feasible,
            key=lambda item: (
                float(item["precision"]),
                float(item["f1"]),
                float(item["recall"]),
                float(item["threshold"]),
            ),
        )
        thresholds[label] = float(best["threshold"])
        diagnostics[label] = {
            "requested_recall_floor": float(requested_floor),
            "reachable_recall_ceiling": ceiling,
            "effective_recall_floor": effective_floor,
            "requested_floor_reachable": bool(ceiling + 1e-12 >= requested_floor),
            "selected": best,
        }
    return thresholds, diagnostics


def make_ui_data(
    candidates: Sequence[Candidate],
    scores: dict[tuple[str, str, int], float],
    gt_by_video: dict[str, dict[str, list[float]]],
    labels: Sequence[str],
    durations: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, Any]] = {}
    for item in candidates:
        key = (item.video_id, item.window_index)
        row = grouped.setdefault(
            key,
            {
                "video_id": item.video_id,
                "start_sec": item.window_start_sec,
                "end_sec": item.window_end_sec,
            },
        )
        row[f"prob_{item.label}"] = float(scores[item.key])
    result: list[dict[str, Any]] = []
    for video_id in sorted(gt_by_video):
        rows = [row for (vid, _), row in grouped.items() if vid == video_id]
        rows.sort(key=lambda row: (float(row["start_sec"]), float(row["end_sec"])))
        duration = float((durations or {}).get(video_id, 0.0))
        if duration <= 0:
            duration = max((float(row["end_sec"]) for row in rows), default=0.0)
        gts = [
            {"label": label, "time_sec": float(time_sec)}
            for label in labels
            for time_sec in gt_by_video[video_id][label]
        ]
        result.append({"video_id": video_id, "rows": rows, "gts": gts, "duration_sec": duration})
    return result


def evaluate_all(
    candidates: Sequence[Candidate],
    scores: dict[tuple[str, str, int], float],
    thresholds: dict[str, float],
    gt_by_video: dict[str, dict[str, list[float]]],
    labels: Sequence[str],
    durations: dict[str, float] | None = None,
    *,
    nms_radius_sec: float,
    match_tolerance_sec: float,
    max_review_segment_sec: float,
    ui_cap_sec: float,
) -> dict[str, Any]:
    point = evaluate_predictions(
        candidates,
        scores,
        thresholds,
        gt_by_video,
        labels,
        nms_radius_sec=nms_radius_sec,
        match_tolerance_sec=match_tolerance_sec,
        use_peak_time=True,
    )
    ui = evaluate_selected(
        make_ui_data(candidates, scores, gt_by_video, labels, durations),
        labels,
        thresholds,
        match_tolerance_sec,
        max_review_segment_sec,
        ui_cap_sec,
    )
    return {"pointnms_strict_1to1": point, "window_and_ui": ui}


def write_predictions(
    path: Path,
    candidates: Sequence[Candidate],
    scores: dict[tuple[str, str, int], float],
    thresholds: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "video_id", "label", "window_index", "start_sec", "end_sec", "center_sec",
        "peak_sec", "clip_prob", "verifier_score", "threshold", "selected",
        "training_target", "nearest_gt_distance_sec",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in candidates:
            score = float(scores[item.key])
            writer.writerow(
                {
                    "video_id": item.video_id,
                    "label": item.label,
                    "window_index": item.window_index,
                    "start_sec": item.window_start_sec,
                    "end_sec": item.window_end_sec,
                    "center_sec": item.center_time_sec,
                    "peak_sec": item.peak_time_sec,
                    "clip_prob": item.clip_prob,
                    "verifier_score": score,
                    "threshold": thresholds[item.label],
                    "selected": int(score >= thresholds[item.label]),
                    "training_target": "" if item.target is None else item.target,
                    "nearest_gt_distance_sec": "" if item.nearest_gt_distance_sec is None else item.nearest_gt_distance_sec,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict cal15 -> external18 football candidate verifier protocol")
    parser.add_argument("--calibration-run-dir", required=True, type=Path)
    parser.add_argument("--calibration-video-id-file", required=True, type=Path)
    parser.add_argument("--external-run-dir", required=True, type=Path)
    parser.add_argument("--external-video-id-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--recall-floor", type=float, default=0.90)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--regularization-c", type=float, default=0.1)
    parser.add_argument("--threshold-grid-size", type=int, default=101)
    parser.add_argument("--match-tolerance-sec", type=float, default=5.0)
    parser.add_argument("--ignore-radius-sec", type=float, default=8.0)
    parser.add_argument("--neighbor-radius-sec", type=float, default=5.1)
    parser.add_argument("--nms-radius-sec", type=float, default=5.0)
    parser.add_argument("--support-threshold", type=float, default=0.02)
    parser.add_argument("--max-review-segment-sec", type=float, default=30.0)
    parser.add_argument("--ui-cap-sec", type=float, default=10.0)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = tuple(item.strip() for item in args.labels.split(",") if item.strip())
    calibration_ids = read_ids(args.calibration_video_id_file)
    external_ids = read_ids(args.external_video_id_file)
    overlap = sorted(set(calibration_ids) & set(external_ids))
    if overlap:
        raise ValueError(f"calibration/external leakage detected: {overlap[:20]}")
    cache_audit = {
        "calibration": validate_cache(args.calibration_run_dir, calibration_ids),
        "external": validate_cache(args.external_run_dir, external_ids),
    }
    protocol = {
        "name": "football_candidate_verifier_cal15_external18_v1",
        "fit_scope": "calibration videos only; video-grouped OOF",
        "threshold_scope": "calibration OOF only",
        "external_scope": "single frozen-model/frozen-threshold evaluation only",
        "labels": list(labels),
        "calibration_video_ids": calibration_ids,
        "external_video_ids": external_ids,
        "calibration_split_sha256": split_fingerprint(calibration_ids),
        "external_split_sha256": split_fingerprint(external_ids),
        "cache_audit": cache_audit,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "protocol_audit.json").write_text(json.dumps(protocol, indent=2) + "\n")
    if args.check_only:
        print(json.dumps(protocol, indent=2))
        if cache_audit["calibration"]["missing"] or cache_audit["external"]["missing"]:
            raise SystemExit(2)
        return
    if cache_audit["calibration"]["missing"] or cache_audit["external"]["missing"]:
        raise FileNotFoundError("dense cache incomplete; see protocol_audit.json")

    candidate_kwargs = {
        "match_tolerance_sec": args.match_tolerance_sec,
        "ignore_radius_sec": args.ignore_radius_sec,
        "neighbor_radius_sec": args.neighbor_radius_sec,
        "support_thresholds": {label: args.support_threshold for label in labels},
    }

    # No external artifact is read before every learned object is frozen below.
    calibration_candidates = load_candidates(
        args.calibration_run_dir, calibration_ids, labels, **candidate_kwargs
    )
    calibration_gt = collect_gt(args.calibration_run_dir, calibration_ids, labels)
    calibration_durations = load_durations(args.calibration_run_dir, calibration_ids)
    oof_scores, folds = fit_oof_models(
        calibration_candidates, labels, folds=args.folds, regularization_c=args.regularization_c
    )
    verifier_thresholds, verifier_selection = select_point_thresholds(
        calibration_candidates,
        oof_scores,
        calibration_gt,
        labels,
        args.recall_floor,
        nms_radius_sec=args.nms_radius_sec,
        match_tolerance_sec=args.match_tolerance_sec,
        grid_size=args.threshold_grid_size,
    )
    final_models = fit_final_models(
        calibration_candidates, labels, regularization_c=args.regularization_c
    )

    baseline_scores = clip_scores(calibration_candidates)
    baseline_thresholds, baseline_selection = select_point_thresholds(
        calibration_candidates,
        baseline_scores,
        calibration_gt,
        labels,
        args.recall_floor,
        nms_radius_sec=args.nms_radius_sec,
        match_tolerance_sec=args.match_tolerance_sec,
        grid_size=args.threshold_grid_size,
    )
    calibration_report = {
        "threshold_source": "video-grouped OOF on calibration split",
        "folds": folds,
        "baseline_selection": baseline_selection,
        "verifier_selection": verifier_selection,
        "baseline": evaluate_all(
            calibration_candidates, baseline_scores, baseline_thresholds, calibration_gt, labels, calibration_durations,
            nms_radius_sec=args.nms_radius_sec, match_tolerance_sec=args.match_tolerance_sec,
            max_review_segment_sec=args.max_review_segment_sec, ui_cap_sec=args.ui_cap_sec,
        ),
        "verifier_oof": evaluate_all(
            calibration_candidates, oof_scores, verifier_thresholds, calibration_gt, labels, calibration_durations,
            nms_radius_sec=args.nms_radius_sec, match_tolerance_sec=args.match_tolerance_sec,
            max_review_segment_sec=args.max_review_segment_sec, ui_cap_sec=args.ui_cap_sec,
        ),
    }
    frozen = {
        "version": 1,
        "feature_names": [
            "clip_logit", "frame_max_logit", "frame_top4_mean_logit", "frame_sharpness",
            "peak_offset_norm", "neighbor_clip_mean_logit", "neighbor_support_count",
        ],
        "models": final_models,
        "thresholds": verifier_thresholds,
        "baseline_thresholds": baseline_thresholds,
        "recall_floor": args.recall_floor,
        "match_tolerance_sec": args.match_tolerance_sec,
        "nms_radius_sec": args.nms_radius_sec,
        "calibration_split_sha256": split_fingerprint(calibration_ids),
    }
    (args.output_dir / "calibration_oof_report.json").write_text(json.dumps(calibration_report, indent=2) + "\n")
    (args.output_dir / "frozen_verifier.json").write_text(json.dumps(frozen, indent=2) + "\n")
    write_predictions(
        args.output_dir / "calibration_oof_predictions.csv",
        calibration_candidates,
        oof_scores,
        verifier_thresholds,
    )

    # External data starts here.  No fit or threshold selection is permitted.
    external_candidates = load_candidates(args.external_run_dir, external_ids, labels, **candidate_kwargs)
    external_gt = collect_gt(args.external_run_dir, external_ids, labels)
    external_durations = load_durations(args.external_run_dir, external_ids)
    external_verifier_scores = apply_models(external_candidates, final_models)
    external_baseline_scores = clip_scores(external_candidates)
    external_report = {
        "data_isolation": "external scores were not used for model fit or threshold selection",
        "baseline_fixed_from_calibration": evaluate_all(
            external_candidates, external_baseline_scores, baseline_thresholds, external_gt, labels, external_durations,
            nms_radius_sec=args.nms_radius_sec, match_tolerance_sec=args.match_tolerance_sec,
            max_review_segment_sec=args.max_review_segment_sec, ui_cap_sec=args.ui_cap_sec,
        ),
        "verifier_fixed_from_calibration": evaluate_all(
            external_candidates, external_verifier_scores, verifier_thresholds, external_gt, labels, external_durations,
            nms_radius_sec=args.nms_radius_sec, match_tolerance_sec=args.match_tolerance_sec,
            max_review_segment_sec=args.max_review_segment_sec, ui_cap_sec=args.ui_cap_sec,
        ),
    }
    (args.output_dir / "external18_report.json").write_text(json.dumps(external_report, indent=2) + "\n")
    write_predictions(
        args.output_dir / "external18_predictions.csv",
        external_candidates,
        external_verifier_scores,
        verifier_thresholds,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "external18": external_report}, indent=2))


if __name__ == "__main__":
    main()
