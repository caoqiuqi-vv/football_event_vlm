#!/usr/bin/env python3
"""Build a bounded GT + recall-constrained dense review queue.

All existing GT events are shown so wrong labels/times can be repaired. Dense
model windows are selected with the highest per-video/class score threshold
that reaches a configurable dense-window GT coverage target. Thus every *selected*
model result is reviewable without forcing annotators to inspect every low
checkpoint-threshold response.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LABELS = ("shot", "save", "set_piece")
FAMILY = {"shot": "shot_save", "save": "shot_save", "set_piece": "set_piece"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--video-id-file", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-segment-sec", type=float, default=20.0)
    parser.add_argument("--gt-context-sec", type=float, default=5.0)
    parser.add_argument("--match-tolerance-sec", type=float, default=3.0)
    parser.add_argument(
        "--recall-floors", default="shot=0.90,save=0.85,set_piece=0.85",
        help="Per-class dense candidate recall floors used to minimize review load",
    )
    parser.add_argument(
        "--adaptive-report", type=Path, default=None,
        help="Optional leakage-free median-logit report; preferred over per-video GT tuning",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_ids(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def read_windows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_label_floats(value: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in value.split(","):
        key, raw = item.split("=", 1)
        if key not in LABELS:
            raise ValueError(f"unknown label in mapping: {key}")
        result[key] = float(raw)
    missing = set(LABELS) - set(result)
    if missing or any(not 0.0 <= number <= 1.0 for number in result.values()):
        raise ValueError(f"invalid recall floors, missing={sorted(missing)}: {result}")
    return result


def row_center(row: dict[str, str]) -> float:
    return (float(row["start_sec"]) + float(row["end_sec"])) / 2.0


def logit(probability: float) -> float:
    value = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


def sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def load_adaptive_parameters(path: Path | None) -> dict[str, dict[str, dict[str, float]]]:
    if path is None:
        return {}
    report = load_json(path)
    result: dict[str, dict[str, dict[str, float]]] = {}
    for label in LABELS:
        folds = report["per_class"][label]["adaptive_median_logit_loov"]["folds"]
        result[label] = {
            str(fold["held_out_video_id"]): {
                "alpha": float(fold["alpha"]),
                "adaptive_threshold": float(fold["adaptive_threshold"]),
            }
            for fold in folds
        }
    return result


def covered_gt_count(
    windows: list[dict[str, str]], gt_times: list[float], label: str,
    threshold: float, tolerance_sec: float,
) -> int:
    """Match the evaluator's ``window_overlap`` unique-GT recall semantics."""
    selected = [
        row for row in windows
        if float(row.get(f"prob_{label}") or 0.0) >= threshold
    ]
    return sum(
        any(
            float(row["start_sec"]) - tolerance_sec <= gt_time
            <= float(row["end_sec"]) + tolerance_sec
            for row in selected
        )
        for gt_time in gt_times
    )


def highest_recall_threshold(
    windows: list[dict[str, str]], gt_times: list[float], label: str,
    recall_floor: float, tolerance_sec: float, fallback: float,
) -> tuple[float, int]:
    if not gt_times:
        return fallback, 0
    required = int(math.ceil(recall_floor * len(gt_times) - 1e-12))
    scores = sorted(
        {float(row.get(f"prob_{label}") or 0.0) for row in windows}, reverse=True
    )
    if not scores:
        return 1.0, 0
    left, right = 0, len(scores) - 1
    answer = len(scores) - 1
    while left <= right:
        middle = (left + right) // 2
        matched = covered_gt_count(
            windows, gt_times, label, scores[middle], tolerance_sec
        )
        if matched >= required:
            answer = middle
            right = middle - 1
        else:
            left = middle + 1
    threshold = scores[answer]
    matched = covered_gt_count(
        windows, gt_times, label, threshold, tolerance_sec
    )
    return threshold, matched


def active_labels(row: dict[str, str], thresholds: dict[str, float]) -> set[str]:
    return {
        label
        for label in LABELS
        if float(row.get(f"prob_{label}") or 0.0) >= thresholds[label]
    }


def chunk_family_windows(
    windows: list[dict[str, str]], max_segment_sec: float,
    thresholds: dict[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    active: list[dict[str, Any]] = []
    for row in windows:
        labels = active_labels(row, thresholds)
        if not labels:
            continue
        active.append({
            "index": int(row["index"]),
            "center_sec": (float(row["start_sec"]) + float(row["end_sec"])) / 2.0,
            "start_sec": float(row["start_sec"]),
            "end_sec": float(row["end_sec"]),
            "labels": labels,
        })
        for family in sorted({FAMILY[label] for label in labels}):
            family_labels = {label for label in labels if FAMILY[label] == family}
            by_family[family].append({
                "index": int(row["index"]),
                "center_sec": (float(row["start_sec"]) + float(row["end_sec"])) / 2.0,
                "start_sec": float(row["start_sec"]),
                "end_sec": float(row["end_sec"]),
                "labels": family_labels,
                "score": max(float(row[f"prob_{label}"]) for label in family_labels),
                "family": family,
            })

    atoms: list[dict[str, Any]] = []
    for family, rows in by_family.items():
        rows.sort(key=lambda item: item["index"])
        bursts: list[list[dict[str, Any]]] = []
        for row in rows:
            if not bursts or row["index"] > bursts[-1][-1]["index"] + 1:
                bursts.append([row])
            else:
                bursts[-1].append(row)
        for burst in bursts:
            chunks: list[list[dict[str, Any]]] = []
            for row in burst:
                if (
                    not chunks
                    or max(chunks[-1][-1]["end_sec"], row["end_sec"])
                    - min(chunks[-1][0]["start_sec"], row["start_sec"])
                    > max_segment_sec
                ):
                    chunks.append([row])
                else:
                    chunks[-1].append(row)
            for chunk in chunks:
                peak = max(chunk, key=lambda item: (item["score"], -item["center_sec"]))
                # A run of overlapping positive windows is one model response,
                # not N separate human tasks. Show the peak window while retaining
                # every contributing raw window index for audit/drill-down.
                start_sec = peak["start_sec"]
                end_sec = peak["end_sec"]
                labels = set().union(*(item["labels"] for item in chunk))
                atoms.append({
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "labels": labels,
                    "sources": {"candidate"},
                    "anchors": [{
                        "source": "candidate",
                        "family": family,
                        "labels": sorted(labels),
                        "time_sec": peak["center_sec"],
                        "score": peak["score"],
                        "support_start_sec": start_sec,
                        "support_end_sec": end_sec,
                        "window_indices": [item["index"] for item in chunk],
                    }],
                })
    return atoms, active


def merge_bounded(atoms: list[dict[str, Any]], max_segment_sec: float) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for atom in sorted(atoms, key=lambda item: (item["start_sec"], item["end_sec"])):
        if (
            not merged
            or atom["start_sec"] > merged[-1]["end_sec"]
            or max(merged[-1]["end_sec"], atom["end_sec"])
            - min(merged[-1]["start_sec"], atom["start_sec"])
            > max_segment_sec
        ):
            merged.append({
                "start_sec": float(atom["start_sec"]),
                "end_sec": float(atom["end_sec"]),
                "labels": set(atom["labels"]),
                "sources": set(atom["sources"]),
                "anchors": list(atom["anchors"]),
            })
            continue
        current = merged[-1]
        current["start_sec"] = min(current["start_sec"], float(atom["start_sec"]))
        current["end_sec"] = max(current["end_sec"], float(atom["end_sec"]))
        current["labels"].update(atom["labels"])
        current["sources"].update(atom["sources"])
        current["anchors"].extend(atom["anchors"])
    return merged


def serializable(video_id: str, segment: dict[str, Any]) -> dict[str, Any]:
    return {
        "video_id": video_id,
        "start_sec": segment["start_sec"],
        "end_sec": segment["end_sec"],
        "duration_sec": segment["end_sec"] - segment["start_sec"],
        "labels": sorted(segment["labels"]),
        "sources": sorted(segment["sources"]),
        "has_any_gt": any(anchor["source"] == "gt" for anchor in segment["anchors"]),
        "has_same_label_gt": any(anchor["source"] == "gt" for anchor in segment["anchors"]),
        "anchors": segment["anchors"],
    }


def main() -> None:
    args = parse_args()
    recall_floors = parse_label_floats(args.recall_floors)
    adaptive_parameters = load_adaptive_parameters(args.adaptive_report)
    ids = read_ids(args.video_id_file)
    inventory = load_json(args.inventory)
    inventory_by_video = {
        str(row["video_id"]): row
        for row in inventory["videos"]
    }
    split_by_video = {
        str(row["video_id"]): str(row["split"])
        for row in inventory["videos"]
    }
    if not set(ids).issubset(split_by_video):
        raise RuntimeError("video list contains IDs absent from inventory")

    all_segments: list[dict[str, Any]] = []
    per_video: list[dict[str, Any]] = []
    active_counts: Counter[str] = Counter()
    matched_counts: Counter[str] = Counter()
    gt_counts: Counter[str] = Counter()
    total_gt = 0
    total_duration = 0.0
    for video_id in ids:
        required = ("summary.json", "window_predictions.csv", "gt_events.json")
        local_dir = args.run_dir / video_id
        cache_source = inventory_by_video[video_id].get("cache_source")
        if all((local_dir / name).is_file() for name in required):
            video_dir = local_dir
        elif cache_source and all((Path(cache_source) / name).is_file() for name in required):
            video_dir = Path(cache_source)
        else:
            video_dir = local_dir
        missing = [name for name in required if not (video_dir / name).is_file()]
        if missing:
            raise RuntimeError(f"dense output incomplete for {video_id}: {missing}")
        summary = load_json(video_dir / "summary.json")
        duration = float(summary["duration_sec"])
        total_duration += duration
        windows = read_windows(video_dir / "window_predictions.csv")
        gt = load_json(video_dir / "gt_events.json")
        total_gt += len(gt)
        gt_by_label = {
            label: [float(event["time_sec"]) for event in gt if event["label"] == label]
            for label in LABELS
        }
        fallback_thresholds = {
            label: float(summary.get("thresholds", {}).get(label, 0.5))
            for label in LABELS
        }
        thresholds: dict[str, float] = {}
        per_label_matches: dict[str, int] = {}
        for label in LABELS:
            if adaptive_parameters:
                params = adaptive_parameters[label].get(video_id)
                if params is None:
                    raise RuntimeError(f"adaptive report missing {video_id}/{label}")
                location = statistics.median(
                    [logit(float(row.get(f"prob_{label}") or 0.0)) for row in windows]
                ) if windows else 0.0
                threshold = sigmoid(
                    params["adaptive_threshold"] + params["alpha"] * location
                )
            else:
                threshold, _ = highest_recall_threshold(
                    windows, gt_by_label[label], label, recall_floors[label],
                    args.match_tolerance_sec, fallback_thresholds[label],
                )
                # Safe legacy fallback: never explode a hard video's queue.
                threshold = max(threshold, fallback_thresholds[label])
            matched = covered_gt_count(
                windows, gt_by_label[label], label, threshold,
                args.match_tolerance_sec,
            )
            thresholds[label] = threshold
            per_label_matches[label] = matched
            matched_counts[label] += matched
            gt_counts[label] += len(gt_by_label[label])
        candidate_atoms, active = chunk_family_windows(
            windows, args.max_segment_sec, thresholds
        )
        for row in active:
            active_counts.update(row["labels"])
        gt_atoms = []
        for event in gt:
            time_sec = float(event["time_sec"])
            gt_atoms.append({
                "start_sec": max(0.0, time_sec - args.gt_context_sec),
                "end_sec": min(duration, time_sec + args.gt_context_sec),
                "labels": {str(event["label"])},
                "sources": {"gt"},
                "anchors": [{
                    "source": "gt",
                    "label": str(event["label"]),
                    "time_sec": time_sec,
                    "support_start_sec": max(0.0, time_sec - args.gt_context_sec),
                    "support_end_sec": min(duration, time_sec + args.gt_context_sec),
                }],
            })
        segments = merge_bounded(candidate_atoms + gt_atoms, args.max_segment_sec)

        uncovered_candidates = []
        for atom in candidate_atoms:
            for anchor in atom["anchors"]:
                for label in anchor["labels"]:
                    if not any(
                        label in segment["labels"]
                        and segment["start_sec"] <= anchor["time_sec"] <= segment["end_sec"]
                        for segment in segments
                    ):
                        uncovered_candidates.append((anchor["window_indices"], label))
        uncovered_gt = [
            event
            for event in gt
            if not any(
                event["label"] in segment["labels"]
                and segment["start_sec"] <= float(event["time_sec"]) <= segment["end_sec"]
                for segment in segments
            )
        ]
        if uncovered_candidates or uncovered_gt:
            raise RuntimeError(
                f"coverage failure {video_id}: candidates={uncovered_candidates[:10]} gt={uncovered_gt[:10]}"
            )
        serialized = [serializable(video_id, segment) for segment in segments]
        all_segments.extend(serialized)
        review_seconds = sum(segment["duration_sec"] for segment in serialized)
        per_video.append({
            "video_id": video_id,
            "split": split_by_video[video_id],
            "duration_sec": duration,
            "gt_events": len(gt),
            "active_model_windows": len(active),
            "selected_model_candidates": len(candidate_atoms),
            "active_model_windows_by_label": dict(Counter(
                label for row in active for label in row["labels"]
            )),
            "selection_thresholds": thresholds,
            "target_recall_floors": recall_floors,
            "threshold_strategy": (
                "median_logit_cross_video_adaptive" if adaptive_parameters
                else "guarded_per_video_diagnostic"
            ),
            "selection_recall": {
                label: (
                    per_label_matches[label] / len(gt_by_label[label])
                    if gt_by_label[label] else None
                )
                for label in LABELS
            },
            "recall_floor_shortfall": {
                label: bool(
                    gt_by_label[label]
                    and per_label_matches[label] / len(gt_by_label[label])
                    + 1e-12 < recall_floors[label]
                )
                for label in LABELS
            },
            "review_segments": len(serialized),
            "review_duration_sec": review_seconds,
            "review_ratio": review_seconds / duration if duration else 0.0,
            "gt_coverage": 1.0,
            "selected_model_candidate_coverage": 1.0,
            "dense_source_dir": str(video_dir.resolve()),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in all_segments:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(args.output)
    review_duration = sum(row["review_duration_sec"] for row in per_video)
    report = {
        "schema_version": 2,
        "protocol": "all_gt_union_recall_floor_dense_outputs_bounded_v2",
        "semantics": {
            "nms": False,
            "interval_merge": "viewing-only; source anchors and dense files are retained",
            "assignment_unit": "whole_video",
            "candidate_selection": {
                "scope": "per_video_per_class",
                "objective": (
                    "median-logit normalization with parameters learned on other videos"
                    if adaptive_parameters else
                    "highest guarded score threshold targeting dense window_overlap unique-GT recall"
                ),
                "match_tolerance_sec": args.match_tolerance_sec,
                "recall_floors": recall_floors,
                "no_gt_fallback": "checkpoint threshold",
                "adaptive_report": str(args.adaptive_report.resolve()) if args.adaptive_report else None,
            },
            "guarantees": [
                "every old GT timestamp is inside at least one review segment",
                "every selected dense response peak is inside a same-label review segment",
            ],
        },
        "summary": {
            "videos": len(ids),
            "source_video_hours": total_duration / 3600.0,
            "gt_events": total_gt,
            "active_model_windows_by_label": dict(active_counts),
            "selected_model_candidates": sum(row["selected_model_candidates"] for row in per_video),
            "selection_recall_by_label": {
                label: matched_counts[label] / gt_counts[label] if gt_counts[label] else None
                for label in LABELS
            },
            "selection_matches_by_label": dict(matched_counts),
            "gt_by_label": dict(gt_counts),
            "review_segments": len(all_segments),
            "review_hours": review_duration / 3600.0,
            "review_ratio": review_duration / total_duration if total_duration else 0.0,
            "gt_coverage": 1.0,
            "selected_model_candidate_coverage": 1.0,
        },
        "per_video": per_video,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
