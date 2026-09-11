#!/usr/bin/env python3
"""Score Astra clip classification and chunk localization predictions."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


LABELS = ("shot", "save", "free_kick", "corner", "kickoff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--clip-predictions", type=Path)
    parser.add_argument("--chunk-predictions", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    return rows


def safe_div(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": safe_div(2 * precision * recall, precision + recall),
    }


def index_unique(rows: list[dict[str, Any]], key: str) -> tuple[dict[str, Any], list[str]]:
    indexed: dict[str, Any] = {}
    duplicates = []
    for row in rows:
        identity = str(row.get(key, ""))
        if not identity:
            duplicates.append("<missing>")
        elif identity in indexed:
            duplicates.append(identity)
        else:
            indexed[identity] = row
    return indexed, duplicates


def score_clips(manifest_rows: list[dict[str, Any]], predictions: Path) -> dict[str, Any]:
    prediction_rows, duplicates = index_unique(read_jsonl(predictions), "sample_id")
    known_ids = {row["sample_id"] for row in manifest_rows}
    invalid_labels = []
    counts = {label: {"tp": 0, "fp": 0, "fn": 0} for label in LABELS}
    exact = 0
    answered = 0
    for sample in manifest_rows:
        prediction = prediction_rows.get(sample["sample_id"])
        if prediction is None:
            continue
        answered += 1
        raw_labels = prediction.get("labels", [])
        if not isinstance(raw_labels, list):
            raw_labels = []
            invalid_labels.append({"sample_id": sample["sample_id"], "label": "<non-list>"})
        predicted = set()
        for label in raw_labels:
            if label not in LABELS:
                invalid_labels.append({"sample_id": sample["sample_id"], "label": label})
            else:
                predicted.add(label)
        actual = set(sample["gt_labels"])
        exact += predicted == actual
        for label in LABELS:
            if label in predicted and label in actual:
                counts[label]["tp"] += 1
            elif label in predicted:
                counts[label]["fp"] += 1
            elif label in actual:
                counts[label]["fn"] += 1
    metrics = {label: prf(**value) for label, value in counts.items()}
    aggregate = {key: sum(value[key] for value in counts.values()) for key in ("tp", "fp", "fn")}
    return {
        "samples": len(manifest_rows),
        "answered": answered,
        "coverage": safe_div(answered, len(manifest_rows)),
        "exact_match_accuracy_on_all_samples": safe_div(exact, len(manifest_rows)),
        "per_class": metrics,
        "micro": prf(**aggregate),
        "duplicate_response_ids": duplicates,
        "unknown_response_ids": sorted(set(prediction_rows) - known_ids),
        "invalid_labels": invalid_labels,
    }


def match_one_class(
    gt_times: list[float], predictions: list[dict[str, float]], tolerance: float
) -> tuple[int, int, int, list[float]]:
    unmatched = set(range(len(gt_times)))
    errors = []
    # This follows deployment scoring: higher-confidence predictions claim GT first.
    ordered = sorted(predictions, key=lambda row: (-row["confidence"], row["time_sec"]))
    tp = 0
    fp = 0
    for prediction in ordered:
        eligible = [
            index for index in unmatched
            if abs(gt_times[index] - prediction["time_sec"]) <= tolerance
        ]
        if not eligible:
            fp += 1
            continue
        matched = min(eligible, key=lambda index: abs(gt_times[index] - prediction["time_sec"]))
        unmatched.remove(matched)
        tp += 1
        errors.append(abs(gt_times[matched] - prediction["time_sec"]))
    return tp, fp, len(unmatched), errors


def score_chunks(
    manifest_rows: list[dict[str, Any]], predictions: Path, duration_hours: float
) -> dict[str, Any]:
    response_rows, duplicates = index_unique(read_jsonl(predictions), "chunk_id")
    known_ids = {row["chunk_id"] for row in manifest_rows}
    gt_by_video_label: dict[tuple[str, str], list[float]] = defaultdict(list)
    pred_by_video_label: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    invalid_events = []
    out_of_core = []
    answered = 0
    for chunk in manifest_rows:
        video_id = chunk["video_id"]
        for event in chunk["gt_events"]:
            gt_by_video_label[(video_id, event["label"])].append(float(event["time_sec"]))
        response = response_rows.get(chunk["chunk_id"])
        if response is None:
            continue
        answered += 1
        events = response.get("events", [])
        if not isinstance(events, list):
            invalid_events.append({"chunk_id": chunk["chunk_id"], "reason": "events_not_list"})
            continue
        for ordinal, event in enumerate(events):
            label = event.get("label") if isinstance(event, dict) else None
            relative_time = event.get("time_sec_relative_to_chunk") if isinstance(event, dict) else None
            if label not in LABELS or not isinstance(relative_time, (int, float)):
                invalid_events.append(
                    {"chunk_id": chunk["chunk_id"], "event_index": ordinal, "event": event}
                )
                continue
            absolute_time = float(chunk["input_start_sec"]) + float(relative_time)
            if not (chunk["core_start_sec"] <= absolute_time < chunk["core_end_sec"]):
                out_of_core.append(
                    {"chunk_id": chunk["chunk_id"], "event_index": ordinal, "event": event}
                )
                continue
            confidence = event.get("confidence", 0.5)
            if not isinstance(confidence, (int, float)) or not math.isfinite(float(confidence)):
                confidence = 0.5
            pred_by_video_label[(video_id, label)].append(
                {"time_sec": absolute_time, "confidence": float(confidence)}
            )

    tolerance_reports = {}
    for tolerance in (1.0, 3.0, 5.0):
        class_counts = {label: {"tp": 0, "fp": 0, "fn": 0, "errors": []} for label in LABELS}
        per_video: dict[str, dict[str, Any]] = defaultdict(dict)
        video_ids = sorted({row["video_id"] for row in manifest_rows})
        for video_id in video_ids:
            for label in LABELS:
                tp, fp, fn, errors = match_one_class(
                    gt_by_video_label[(video_id, label)],
                    pred_by_video_label[(video_id, label)],
                    tolerance,
                )
                class_counts[label]["tp"] += tp
                class_counts[label]["fp"] += fp
                class_counts[label]["fn"] += fn
                class_counts[label]["errors"].extend(errors)
                per_video[video_id][label] = prf(tp, fp, fn)
        per_class = {}
        for label, values in class_counts.items():
            report = prf(values["tp"], values["fp"], values["fn"])
            report["matched_time_mae_sec"] = (
                sum(values["errors"]) / len(values["errors"]) if values["errors"] else None
            )
            report["fp_per_hour"] = safe_div(values["fp"], duration_hours)
            per_class[label] = report
        aggregate = {
            key: sum(values[key] for values in class_counts.values())
            for key in ("tp", "fp", "fn")
        }
        micro = prf(**aggregate)
        micro["fp_per_hour"] = safe_div(aggregate["fp"], duration_hours)
        tolerance_reports[f"tolerance_{int(tolerance)}s"] = {
            "per_class": per_class,
            "micro": micro,
            "per_video": per_video,
        }
    return {
        "chunks": len(manifest_rows),
        "answered": answered,
        "coverage": safe_div(answered, len(manifest_rows)),
        "duration_hours": duration_hours,
        "metrics": tolerance_reports,
        "duplicate_response_ids": duplicates,
        "unknown_response_ids": sorted(set(response_rows) - known_ids),
        "invalid_events": invalid_events,
        "out_of_core_events_ignored": out_of_core,
    }


def main() -> None:
    args = parse_args()
    benchmark_manifest = json.loads(
        (args.benchmark_dir / "benchmark_manifest.json").read_text(encoding="utf-8")
    )
    report: dict[str, Any] = {
        "benchmark": str(args.benchmark_dir.resolve()),
        "schema_version": benchmark_manifest["schema_version"],
    }
    if args.clip_predictions:
        report["protocol_a"] = score_clips(
            read_jsonl(args.benchmark_dir / "clip_manifest.jsonl"),
            args.clip_predictions,
        )
    if args.chunk_predictions:
        duration_hours = sum(
            float(video["duration_sec"]) for video in benchmark_manifest["videos"]
        ) / 3600.0
        report["protocol_b"] = score_chunks(
            read_jsonl(args.benchmark_dir / "chunk_manifest.jsonl"),
            args.chunk_predictions,
            duration_hours,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

