#!/usr/bin/env python
"""Fast cached long-video threshold selection with honest UI workload accounting."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_long_video_checkpoint import pred_matches_gt
from retune_all_dense_window_runs import maximum_matching
from select_window_overlap_thresholds_by_budget import (
    choose_threshold_values,
    load_data,
    make_predictions,
    merge_intervals,
    metric,
    read_video_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select precision-optimal thresholds at a recall floor from cached dense outputs."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--video-id-file", type=Path, required=True)
    parser.add_argument("--labels", default="shot,save,set_piece")
    parser.add_argument(
        "--score-prefixes",
        default="prob,clip_prob,response_prob,frame_max_prob,frame_topk_mean_prob",
    )
    parser.add_argument("--recall-floor", type=float, default=0.80)
    parser.add_argument("--grid-size", type=int, default=31)
    parser.add_argument("--match-tolerance-sec", type=float, default=5.0)
    parser.add_argument("--max-review-segment-sec", type=float, default=30.0)
    parser.add_argument("--ui-cap-sec", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def aggregate(per_class: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return metric(
        sum(int(item["tp"]) for item in per_class.values()),
        sum(int(item["fp"]) for item in per_class.values()),
        sum(int(item["num_gt"]) for item in per_class.values()),
    )


def class_metric(
    data: Sequence[dict[str, Any]],
    label: str,
    threshold: float,
    tolerance_sec: float,
) -> dict[str, Any]:
    tp = fp = gt_count = 0
    for item in data:
        predictions = make_predictions(item["rows"], [label], {label: threshold})
        gts = [gt for gt in item["gts"] if gt["label"] == label]
        edges = [
            [
                gt_index
                for gt_index, gt in enumerate(gts)
                if pred_matches_gt(pred, gt, tolerance_sec, matching_mode="window")
            ]
            for pred in predictions
        ]
        matched = maximum_matching(edges)
        tp += matched
        fp += len(predictions) - matched
        gt_count += len(gts)
    return metric(tp, fp, gt_count)


def positive_rows(
    rows: Sequence[dict[str, Any]], labels: Sequence[str], thresholds: dict[str, float]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        scores = {
            label: float(row.get(f"prob_{label}", 0.0) or 0.0)
            for label in labels
            if float(row.get(f"prob_{label}", 0.0) or 0.0) >= thresholds[label]
        }
        if not scores:
            continue
        result.append(
            {
                "start_sec": float(row["start_sec"]),
                "end_sec": float(row["end_sec"]),
                "peak_sec": 0.5 * (float(row["start_sec"]) + float(row["end_sec"])),
                "scores": scores,
                "peak_score": max(scores.values()),
            }
        )
    return result


def make_ui_segments(
    rows: Sequence[dict[str, Any]],
    labels: Sequence[str],
    thresholds: dict[str, float],
    duration_sec: float,
    max_segment_sec: float,
) -> list[dict[str, Any]]:
    positives = positive_rows(rows, labels, thresholds)
    if not positives:
        return []
    runs: list[tuple[float, float]] = []
    for row in sorted(positives, key=lambda item: (item["start_sec"], item["end_sec"])):
        start, end = float(row["start_sec"]), float(row["end_sec"])
        if runs and start <= runs[-1][1]:
            runs[-1] = (runs[-1][0], max(runs[-1][1], end))
        else:
            runs.append((start, end))
    segments: list[dict[str, Any]] = []
    limit = max(float(max_segment_sec), 1.0)
    for run_start, run_end in runs:
        start = run_start
        while start < run_end - 1e-9:
            end = min(start + limit, run_end)
            sources = [
                row
                for row in positives
                if float(row["end_sec"]) > start and float(row["start_sec"]) < end
            ]
            labels_present = sorted({label for row in sources for label in row["scores"]})
            peak = max(
                sources,
                key=lambda row: (
                    float(row["peak_score"]),
                    -abs(float(row["peak_sec"]) - 0.5 * (start + end)),
                ),
            )
            segments.append(
                {
                    "start_sec": max(0.0, start),
                    "end_sec": min(duration_sec, end),
                    "labels": labels_present,
                    "peak_sec": float(peak["peak_sec"]),
                    "peak_score": float(peak["peak_score"]),
                }
            )
            start = end
    return segments


def cap_segment(segment: dict[str, Any], cap_sec: float) -> tuple[float, float]:
    start, end = float(segment["start_sec"]), float(segment["end_sec"])
    if end - start <= cap_sec:
        return start, end
    center = float(segment["peak_sec"])
    left, right = center - 0.5 * cap_sec, center + 0.5 * cap_sec
    if left < start:
        right += start - left
        left = start
    if right > end:
        left -= right - end
        right = end
    return max(start, left), min(end, right)


def visible_metrics(
    gts: Sequence[dict[str, Any]],
    intervals: Sequence[tuple[float, float]],
    labels: Sequence[str],
    tolerance_sec: float,
) -> dict[str, Any]:
    per_class: dict[str, dict[str, Any]] = {}
    for label in labels:
        label_gts = [gt for gt in gts if gt["label"] == label]
        visible = sum(
            any(
                start - tolerance_sec <= float(gt["time_sec"]) <= end + tolerance_sec
                for start, end in intervals
            )
            for gt in label_gts
        )
        per_class[label] = {
            "visible_gt": int(visible),
            "num_gt": len(label_gts),
            "recall": visible / len(label_gts) if label_gts else 0.0,
        }
    total_visible = sum(item["visible_gt"] for item in per_class.values())
    total_gt = sum(item["num_gt"] for item in per_class.values())
    return {
        "recall": total_visible / total_gt if total_gt else 0.0,
        "visible_gt": total_visible,
        "num_gt": total_gt,
        "per_class": per_class,
    }


def segment_label_metrics(
    segments_by_video: Sequence[tuple[list[dict[str, Any]], list[dict[str, Any]]]],
    labels: Sequence[str],
    tolerance_sec: float,
) -> dict[str, Any]:
    per_class: dict[str, dict[str, Any]] = {}
    for label in labels:
        tp = fp = gt_count = 0
        for segments, gts in segments_by_video:
            label_segments = [segment for segment in segments if label in segment["labels"]]
            label_gts = [gt for gt in gts if gt["label"] == label]
            edges = [
                [
                    gt_index
                    for gt_index, gt in enumerate(label_gts)
                    if float(segment["start_sec"]) - tolerance_sec
                    <= float(gt["time_sec"])
                    <= float(segment["end_sec"]) + tolerance_sec
                ]
                for segment in label_segments
            ]
            matched = maximum_matching(edges)
            tp += matched
            fp += len(label_segments) - matched
            gt_count += len(label_gts)
        per_class[label] = metric(tp, fp, gt_count)
    return {"per_class": per_class, "micro": aggregate(per_class)}


def evaluate_selected(
    data: Sequence[dict[str, Any]],
    labels: Sequence[str],
    thresholds: dict[str, float],
    tolerance_sec: float,
    max_segment_sec: float,
    cap_sec: float,
) -> dict[str, Any]:
    per_class: dict[str, dict[str, Any]] = {}
    for label in labels:
        per_class[label] = class_metric(data, label, thresholds[label], tolerance_sec)
    window_micro = aggregate(per_class)

    total_duration_sec = 0.0
    full_review_sec = 0.0
    capped_review_sec = 0.0
    num_segments = 0
    full_visible_totals = {label: {"visible_gt": 0, "num_gt": 0} for label in labels}
    capped_visible_totals = {label: {"visible_gt": 0, "num_gt": 0} for label in labels}
    segments_by_video: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
    per_video: list[dict[str, Any]] = []

    for item in data:
        duration = float(item["duration_sec"])
        total_duration_sec += duration
        segments = make_ui_segments(
            item["rows"], labels, thresholds, duration, max_segment_sec
        )
        full_intervals = merge_intervals(
            [(float(segment["start_sec"]), float(segment["end_sec"])) for segment in segments]
        )
        capped_intervals = merge_intervals([cap_segment(segment, cap_sec) for segment in segments])
        full_sec = sum(end - start for start, end in full_intervals)
        capped_sec = sum(end - start for start, end in capped_intervals)
        full_review_sec += full_sec
        capped_review_sec += capped_sec
        num_segments += len(segments)
        full_visible = visible_metrics(item["gts"], full_intervals, labels, tolerance_sec)
        capped_visible = visible_metrics(item["gts"], capped_intervals, labels, tolerance_sec)
        for label in labels:
            for target, source in (
                (full_visible_totals, full_visible["per_class"]),
                (capped_visible_totals, capped_visible["per_class"]),
            ):
                target[label]["visible_gt"] += int(source[label]["visible_gt"])
                target[label]["num_gt"] += int(source[label]["num_gt"])
        segments_by_video.append((segments, item["gts"]))
        per_video.append(
            {
                "video_id": item["video_id"],
                "duration_sec": duration,
                "num_review_segments": len(segments),
                "full_review_sec": full_sec,
                "capped_review_sec": capped_sec,
                "full_visible_recall": full_visible["recall"],
                "capped_visible_recall": capped_visible["recall"],
            }
        )

    def finish_visible(values: dict[str, dict[str, int]]) -> dict[str, Any]:
        finished = {
            label: {
                **item,
                "recall": item["visible_gt"] / item["num_gt"] if item["num_gt"] else 0.0,
            }
            for label, item in values.items()
        }
        visible = sum(item["visible_gt"] for item in finished.values())
        num_gt = sum(item["num_gt"] for item in finished.values())
        return {
            "recall": visible / num_gt if num_gt else 0.0,
            "visible_gt": visible,
            "num_gt": num_gt,
            "per_class": finished,
        }

    total_minutes = total_duration_sec / 60.0
    num_candidates = sum(
        sum(
            int(per_class[label]["num_pred"])
            for label in labels
        )
        for _ in [0]
    )
    return {
        "thresholds": thresholds,
        "threshold_string": ",".join(f"{label}={thresholds[label]:.9g}" for label in labels),
        "window_strict_1to1": {"per_class": per_class, "micro": window_micro},
        "review_segment_label": segment_label_metrics(
            segments_by_video, labels, tolerance_sec
        ),
        "workload": {
            "total_video_minutes": total_minutes,
            "num_window_label_candidates": num_candidates,
            "num_review_segments": num_segments,
            "full_segment_minutes": full_review_sec / 60.0,
            "full_segment_participation_pct": 100.0 * full_review_sec / total_duration_sec if total_duration_sec else 0.0,
            "capped_minutes": capped_review_sec / 60.0,
            "capped_participation_pct": 100.0 * capped_review_sec / total_duration_sec if total_duration_sec else 0.0,
            "ui_cap_sec": cap_sec,
            "max_review_segment_sec": max_segment_sec,
        },
        "full_segment_human_visible": finish_visible(full_visible_totals),
        "capped_human_visible": finish_visible(capped_visible_totals),
        "per_video": per_video,
    }


def search_prefix(
    run_dir: Path,
    video_ids: Sequence[str],
    labels: Sequence[str],
    score_prefix: str,
    recall_floor: float,
    grid_size: int,
    tolerance_sec: float,
    max_segment_sec: float,
    cap_sec: float,
) -> dict[str, Any]:
    data = load_data(run_dir, video_ids, labels, score_prefix)
    score_values = {label: [] for label in labels}
    for item in data:
        for row in item["rows"]:
            for label in labels:
                score_values[label].append(float(row.get(f"prob_{label}", 0.0) or 0.0))
    grids = {
        label: choose_threshold_values(score_values[label], grid_size, [0.5])
        for label in labels
    }
    cached = {
        label: {
            threshold: class_metric(data, label, threshold, tolerance_sec)
            for threshold in grids[label]
        }
        for label in labels
    }
    best: tuple[tuple[float, int, float], dict[str, float], dict[str, Any]] | None = None
    combinations = 0
    feasible = 0
    for values in itertools.product(*(grids[label] for label in labels)):
        combinations += 1
        thresholds = dict(zip(labels, values))
        per_class = {label: cached[label][thresholds[label]] for label in labels}
        micro = aggregate(per_class)
        if float(micro["recall"]) + 1e-12 < recall_floor:
            continue
        feasible += 1
        rank = (
            float(micro["precision"]),
            -int(micro["num_pred"]),
            float(micro["recall"]),
        )
        if best is None or rank > best[0]:
            best = (rank, thresholds, micro)
    if best is None:
        raise RuntimeError(
            f"No threshold combination reached recall floor {recall_floor:g} for {score_prefix}"
        )
    selected = evaluate_selected(
        data, labels, best[1], tolerance_sec, max_segment_sec, cap_sec
    )
    return {
        "score_prefix": score_prefix,
        "selection_objective": "max strict-window precision subject to strict-window micro recall floor",
        "recall_floor": recall_floor,
        "grid_size": grid_size,
        "grid_combinations": combinations,
        "feasible_combinations": feasible,
        "selected": selected,
    }


def flatten(result: dict[str, Any]) -> dict[str, Any]:
    selected = result["selected"]
    window = selected["window_strict_1to1"]
    segment = selected["review_segment_label"]
    workload = selected["workload"]
    row: dict[str, Any] = {
        "score_prefix": result["score_prefix"],
        "recall_floor": result["recall_floor"],
        "threshold_string": selected["threshold_string"],
        "window_precision": window["micro"]["precision"],
        "window_recall": window["micro"]["recall"],
        "segment_precision": segment["micro"]["precision"],
        "segment_recall": segment["micro"]["recall"],
        "full_segment_minutes": workload["full_segment_minutes"],
        "full_segment_participation_pct": workload["full_segment_participation_pct"],
        "full_human_visible_recall": selected["full_segment_human_visible"]["recall"],
        "capped_minutes": workload["capped_minutes"],
        "capped_participation_pct": workload["capped_participation_pct"],
        "capped_human_visible_recall": selected["capped_human_visible"]["recall"],
        "num_review_segments": workload["num_review_segments"],
        "num_window_label_candidates": workload["num_window_label_candidates"],
    }
    for label in selected["thresholds"]:
        row[f"threshold_{label}"] = selected["thresholds"][label]
        row[f"{label}_precision"] = window["per_class"][label]["precision"]
        row[f"{label}_recall"] = window["per_class"][label]["recall"]
        row[f"{label}_segment_precision"] = segment["per_class"][label]["precision"]
        row[f"{label}_segment_recall"] = segment["per_class"][label]["recall"]
        row[f"{label}_full_visible_recall"] = selected["full_segment_human_visible"]["per_class"][label]["recall"]
        row[f"{label}_capped_visible_recall"] = selected["capped_human_visible"]["per_class"][label]["recall"]
    return row


def main() -> None:
    args = parse_args()
    labels = [item.strip() for item in args.labels.split(",") if item.strip()]
    prefixes = [item.strip() for item in args.score_prefixes.split(",") if item.strip()]
    video_ids = read_video_ids(args.video_id_file)
    results = [
        search_prefix(
            args.run_dir,
            video_ids,
            labels,
            prefix,
            args.recall_floor,
            args.grid_size,
            args.match_tolerance_sec,
            args.max_review_segment_sec,
            args.ui_cap_sec,
        )
        for prefix in prefixes
    ]
    rows = [flatten(result) for result in results]
    best = max(
        rows,
        key=lambda row: (
            float(row["segment_precision"]),
            float(row["window_precision"]),
            -float(row["full_segment_participation_pct"]),
        ),
    )
    payload = {
        "protocol": "cached_long_video_precision_at_recall_with_ui_workload_v1",
        "run_dir": str(args.run_dir),
        "video_id_file": str(args.video_id_file),
        "video_ids": video_ids,
        "labels": labels,
        "match_tolerance_sec": args.match_tolerance_sec,
        "selection_note": "Thresholds are selected on this validation set; report external-test metrics only after applying them unchanged.",
        "results": results,
        "summary_rows": rows,
        "recommended_score_prefix": best["score_prefix"],
        "recommended_summary": best,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    fields = list(rows[0])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(args.output), "csv": str(csv_path), "recommended": best}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
