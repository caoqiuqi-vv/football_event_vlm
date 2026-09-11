#!/usr/bin/env python3
"""Score blinded Astra decisions over frozen DINO dense candidates."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


FINE_LABELS = ("shot", "save", "free_kick", "corner", "kickoff")
SET_PIECE_LABELS = {"free_kick", "corner", "kickoff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def metric(tp: int, fp: int, fn: int, errors: list[float] | None = None) -> dict[str, Any]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    result = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": safe_div(2 * precision * recall, precision + recall),
    }
    if errors is not None:
        result["matched_time_mae_sec"] = safe_div(sum(errors), len(errors)) if errors else None
    return result


def match(
    gt_by_video: dict[str, list[float]],
    predictions: list[dict[str, Any]],
    tolerance: float,
) -> dict[str, Any]:
    pred_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        pred_by_video[prediction["video_id"]].append(prediction)
    tp = fp = fn = 0
    errors: list[float] = []
    for video_id, gt_times in gt_by_video.items():
        unmatched = set(range(len(gt_times)))
        ordered = sorted(
            pred_by_video[video_id], key=lambda item: (-item["score"], item["time_sec"])
        )
        for prediction in ordered:
            eligible = [
                index
                for index in unmatched
                if abs(gt_times[index] - prediction["time_sec"]) <= tolerance
            ]
            if not eligible:
                fp += 1
                continue
            chosen = min(
                eligible, key=lambda index: abs(gt_times[index] - prediction["time_sec"])
            )
            unmatched.remove(chosen)
            tp += 1
            errors.append(abs(gt_times[chosen] - prediction["time_sec"]))
        fn += len(unmatched)
    return metric(tp, fp, fn, errors)


def logit(probability: float) -> float:
    probability = min(max(float(probability), 1e-5), 1 - 1e-5)
    return math.log(probability / (1 - probability))


def best_operating_point(
    gt_by_video: dict[str, list[float]],
    candidates: list[dict[str, Any]],
    tolerance: float,
    required_recall: float,
) -> dict[str, Any] | None:
    thresholds = sorted({float(item["score"]) for item in candidates}, reverse=True)
    best = None
    for threshold in thresholds:
        selected = [item for item in candidates if item["score"] >= threshold]
        current = match(gt_by_video, selected, tolerance)
        if current["recall"] + 1e-12 < required_recall:
            continue
        value = {"threshold": threshold, "num_predictions": len(selected), **current}
        if best is None or (value["precision"], value["threshold"]) > (
            best["precision"],
            best["threshold"],
        ):
            best = value
    return best


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.benchmark_dir / "manifest.json").read_text(encoding="utf-8"))
    public_segments = {row["segment_id"]: row for row in jsonl(args.benchmark_dir / "blind_segments.jsonl")}
    private_candidates = {
        row["candidate_id"]: row for row in jsonl(args.benchmark_dir / "private_dino_key.jsonl")
    }
    response_segments: dict[str, dict[str, Any]] = {}
    duplicate_segments = []
    for prediction_file in args.predictions:
        for row in jsonl(prediction_file):
            segment_id = row.get("segment_id")
            if segment_id in response_segments:
                duplicate_segments.append(segment_id)
            response_segments[segment_id] = row
    response_candidates: dict[str, dict[str, Any]] = {}
    duplicate_candidates = []
    invalid_candidates = []
    for segment_id, response in response_segments.items():
        if segment_id not in public_segments:
            invalid_candidates.append({"segment_id": segment_id, "reason": "unknown_segment"})
            continue
        for candidate in response.get("candidates", []):
            candidate_id = candidate.get("candidate_id")
            if candidate_id in response_candidates:
                duplicate_candidates.append(candidate_id)
            response_candidates[candidate_id] = candidate
    expected_ids = set(private_candidates)
    received_ids = set(response_candidates)

    videos = sorted({row["video_id"] for row in private_candidates.values()})
    gt: dict[str, dict[str, list[float]]] = {
        label: {video_id: [] for video_id in videos} for label in FINE_LABELS
    }
    for video_id in videos:
        annotation = json.loads(
            (args.annotations_dir / f"{video_id}.json").read_text(encoding="utf-8")
        )
        for event in annotation["events"]:
            label = event.get("semantic_label", event.get("label"))
            if label in FINE_LABELS:
                gt[label][video_id].append(float(event["time_sec"]))
        for label in FINE_LABELS:
            gt[label][video_id].sort()

    astra_candidates: dict[str, list[dict[str, Any]]] = {label: [] for label in FINE_LABELS}
    dino_candidates: dict[str, list[dict[str, Any]]] = {label: [] for label in FINE_LABELS}
    explicit_candidates: dict[str, list[dict[str, Any]]] = {label: [] for label in FINE_LABELS}
    fusion_candidates: dict[float, dict[str, list[dict[str, Any]]]] = {
        alpha: {label: [] for label in FINE_LABELS} for alpha in (0.0, 0.25, 0.5, 0.75, 1.0)
    }
    explicit_kept_ids = set()
    for candidate_id, private in private_candidates.items():
        response = response_candidates.get(candidate_id)
        if response is None:
            continue
        segment = public_segments[private["segment_id"]]
        labels = set(response.get("labels", []))
        confidences = response.get("confidence", {})
        refined_times = response.get("event_time_sec_relative_to_clip", {})
        for label in FINE_LABELS:
            astra_score = float(confidences.get(label, 0.0))
            family = "set_piece" if label in SET_PIECE_LABELS else label
            dino_score = float(private["dino_probabilities"][family])
            refined = refined_times.get(label) if isinstance(refined_times, dict) else None
            time_sec = (
                float(segment["input_start_sec"]) + float(refined)
                if isinstance(refined, (int, float))
                else float(private["anchor_time_sec"])
            )
            common = {
                "candidate_id": candidate_id,
                "video_id": private["video_id"],
                "time_sec": time_sec,
            }
            astra_candidates[label].append({**common, "score": astra_score})
            dino_candidates[label].append({**common, "score": dino_score})
            if label in labels:
                explicit_candidates[label].append({**common, "score": astra_score})
                explicit_kept_ids.add(candidate_id)
            for alpha in fusion_candidates:
                score = alpha * logit(dino_score) + (1 - alpha) * logit(astra_score)
                fusion_candidates[alpha][label].append({**common, "score": score})

    report: dict[str, Any] = {
        "schema_version": "football_astra_dino_verifier_score_v1",
        "benchmark": str(args.benchmark_dir.resolve()),
        "note": "All threshold and alpha searches in this three-test-video pilot are diagnostic oracle analyses, not deployable calibration. Production thresholds must be fitted on validation videos.",
        "coverage": {
            "expected_segments": len(public_segments),
            "received_segments": len(response_segments),
            "expected_candidates": len(expected_ids),
            "received_candidates": len(received_ids & expected_ids),
            "missing_candidate_ids": sorted(expected_ids - received_ids),
            "unknown_candidate_ids": sorted(received_ids - expected_ids),
            "duplicate_segments": duplicate_segments,
            "duplicate_candidates": duplicate_candidates,
            "invalid": invalid_candidates,
        },
        "workload": {
            "stage1_unique_candidate_times": len(expected_ids),
            "astra_explicit_kept_candidate_times": len(explicit_kept_ids),
            "candidate_reduction": 1 - safe_div(len(explicit_kept_ids), len(expected_ids)),
        },
        "tolerances": {},
    }
    for tolerance in (3.0, 5.0):
        tolerance_report: dict[str, Any] = {"per_class": {}}
        for label in FINE_LABELS:
            all_candidate_metric = match(gt[label], astra_candidates[label], tolerance)
            max_recall = all_candidate_metric["recall"]
            retain_floor = max_recall * 0.95
            absolute_floor = 0.90 if label == "shot" else 0.85
            fusion = {}
            for alpha, by_label in fusion_candidates.items():
                fusion[str(alpha)] = {
                    "retain_95pct_of_candidate_coverage": best_operating_point(
                        gt[label], by_label[label], tolerance, retain_floor
                    ),
                    "absolute_recall_floor": best_operating_point(
                        gt[label], by_label[label], tolerance, absolute_floor
                    ),
                }
            tolerance_report["per_class"][label] = {
                "gt_count": sum(len(values) for values in gt[label].values()),
                "temporal_candidate_coverage": all_candidate_metric,
                "stage1_dino_ranking": {
                    "retain_95pct_of_candidate_coverage": best_operating_point(
                        gt[label], dino_candidates[label], tolerance, retain_floor
                    ),
                    "absolute_recall_floor": best_operating_point(
                        gt[label], dino_candidates[label], tolerance, absolute_floor
                    ),
                },
                "astra_explicit_decision": match(
                    gt[label], explicit_candidates[label], tolerance
                ),
                "astra_ranking": {
                    "retain_95pct_of_candidate_coverage": best_operating_point(
                        gt[label], astra_candidates[label], tolerance, retain_floor
                    ),
                    "absolute_recall_floor": best_operating_point(
                        gt[label], astra_candidates[label], tolerance, absolute_floor
                    ),
                },
                "fusion_diagnostic": fusion,
            }
        report["tolerances"][f"tolerance_{int(tolerance)}s"] = tolerance_report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
