#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from eval_long_video_checkpoint import (  # noqa: E402
    TARGET_LABELS,
    compute_event_metrics,
    point_nms_predictions,
    window_overlap_predictions,
    write_csv,
)


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def logit(prob: float) -> float:
    prob = min(max(float(prob), 1e-6), 1.0 - 1e-6)
    return math.log(prob / (1.0 - prob))


def parse_label_floats(raw: str, labels: Sequence[str], defaults: dict[str, float]) -> dict[str, float]:
    values = {label: float(defaults.get(label, 0.5)) for label in labels}
    if not raw.strip():
        return values
    for item in raw.split(","):
        if not item.strip():
            continue
        label, value = item.split("=", 1)
        label = label.strip()
        if label not in values:
            raise ValueError(f"Unsupported label '{label}', expected one of {list(labels)}")
        values[label] = float(value)
    return values


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_gt_events(path: Path, labels: Sequence[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for row in read_csv(path):
        if row["label"] not in labels:
            continue
        time_sec = float(row["time_sec"])
        events.append(
            {
                "label": row["label"],
                "time_sec": time_sec,
                "start_sec": float(row.get("start_sec") or time_sec),
                "end_sec": float(row.get("end_sec") or time_sec),
                "event_id": row.get("event_id", ""),
            }
        )
    return events


def load_video_ids(run_dir: Path, raw: str) -> list[str]:
    if raw.strip():
        return [item.strip() for item in raw.split(",") if item.strip()]
    return sorted(
        path.name
        for path in run_dir.iterdir()
        if path.is_dir() and (path / "window_predictions.csv").exists()
    )


def infer_labels(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return list(TARGET_LABELS)
    fields = set(rows[0])
    return [label for label in TARGET_LABELS if f"prob_{label}" in fields]


def branch_prob(row: dict[str, str], label: str, branch: str) -> float:
    if branch == "fused":
        key = f"prob_{label}"
    else:
        key = f"{branch}_prob_{label}"
    if key not in row or row[key] == "":
        raise KeyError(f"Missing branch score column '{key}' in window_predictions.csv")
    return float(row[key])


def build_branch_rows(
    rows: list[dict[str, str]],
    labels: Sequence[str],
    branch: str,
    *,
    confidence_mode: str,
    confidence_threshold: float,
    confidence_power: float,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        new_row: dict[str, Any] = {
            "index": int(row["index"]),
            "start_sec": float(row["start_sec"]),
            "end_sec": float(row["end_sec"]),
        }
        confidence = float(row.get("roi_confidence") or 0.0)
        roi_valid = float(row.get("roi_valid") or 0.0)
        for label in labels:
            global_prob = branch_prob(row, label, "global")
            local_prob = branch_prob(row, label, "local")
            fused_prob = branch_prob(row, label, "fused")
            if branch in ("fused", "global", "local"):
                prob = {"fused": fused_prob, "global": global_prob, "local": local_prob}[branch]
            elif branch in ("gate_no_confidence", "gate_confidence_sqrt", "gate_confidence_squared", "gate_positive_only"):
                current_gate = float(row.get(f"roi_gate_{label}") or 0.0)
                learned_gate = 0.0
                if roi_valid > 0.0 and confidence > 1e-6:
                    learned_gate = min(max(current_gate / confidence, 0.0), 1.0)
                if branch == "gate_no_confidence":
                    alpha = learned_gate
                elif branch == "gate_confidence_sqrt":
                    alpha = learned_gate * math.sqrt(min(max(confidence, 0.0), 1.0))
                elif branch == "gate_confidence_squared":
                    alpha = learned_gate * min(max(confidence, 0.0), 1.0) ** 2
                else:
                    alpha = current_gate if local_prob > global_prob else 0.0
                prob = sigmoid(logit(global_prob) + alpha * (logit(local_prob) - logit(global_prob)))
            elif branch == "confidence_switch":
                prob = local_prob if roi_valid > 0.0 and confidence >= confidence_threshold else global_prob
            elif branch == "confidence_logit":
                alpha = 0.0
                if roi_valid > 0.0 and confidence >= confidence_threshold:
                    alpha = min(max(confidence, 0.0), 1.0) ** confidence_power
                prob = sigmoid(logit(global_prob) + alpha * (logit(local_prob) - logit(global_prob)))
            elif branch == "confidence_boost":
                alpha = 0.0
                if roi_valid > 0.0 and confidence >= confidence_threshold and local_prob > global_prob:
                    alpha = min(max(confidence, 0.0), 1.0) ** confidence_power
                prob = sigmoid(logit(global_prob) + alpha * (logit(local_prob) - logit(global_prob)))
            else:
                raise ValueError(f"Unsupported branch={branch}")
            new_row[f"prob_{label}"] = prob
        out.append(new_row)
    return out


def predictions_from_rows(
    rows: list[dict[str, Any]],
    labels: Sequence[str],
    thresholds: dict[str, float],
    postprocess: str,
    nms_radius_sec: float,
) -> tuple[list[dict[str, Any]], str, bool]:
    if postprocess == "window_overlap":
        return window_overlap_predictions(rows, labels, thresholds), "window", True
    if postprocess == "point_nms":
        return point_nms_predictions(rows, labels, thresholds, nms_radius_sec), "point", False
    raise ValueError("postprocess must be window_overlap or point_nms")


def init_totals(labels: Sequence[str]) -> dict[str, dict[str, int]]:
    return {
        label: {"tp": 0, "fp": 0, "fn": 0, "num_pred": 0, "num_gt": 0, "num_matched_gt": 0}
        for label in labels
    }


def finalize_totals(totals: dict[str, dict[str, int]], allow_many_predictions_per_gt: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    per_class: dict[str, Any] = {}
    for label, total in totals.items():
        precision = total["tp"] / (total["tp"] + total["fp"]) if total["tp"] + total["fp"] else 0.0
        if allow_many_predictions_per_gt:
            recall = total["num_matched_gt"] / total["num_gt"] if total["num_gt"] else 0.0
        else:
            recall = total["tp"] / (total["tp"] + total["fn"]) if total["tp"] + total["fn"] else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {**total, "precision": precision, "recall": recall, "f1": f1}
    micro_tp = sum(item["tp"] for item in totals.values())
    micro_fp = sum(item["fp"] for item in totals.values())
    micro_fn = sum(item["fn"] for item in totals.values())
    micro_precision = micro_tp / (micro_tp + micro_fp) if micro_tp + micro_fp else 0.0
    if allow_many_predictions_per_gt:
        total_gt = sum(item["num_gt"] for item in totals.values())
        matched_gt = sum(item["num_matched_gt"] for item in totals.values())
        micro_recall = matched_gt / total_gt if total_gt else 0.0
    else:
        micro_recall = micro_tp / (micro_tp + micro_fn) if micro_tp + micro_fn else 0.0
    micro_f1 = 2.0 * micro_precision * micro_recall / (micro_precision + micro_recall) if micro_precision + micro_recall else 0.0
    return per_class, {"tp": micro_tp, "fp": micro_fp, "fn": micro_fn, "precision": micro_precision, "recall": micro_recall, "f1": micro_f1}


def evaluate_branch(
    run_dir: Path,
    video_ids: Sequence[str],
    labels: Sequence[str],
    branch: str,
    thresholds: dict[str, float],
    postprocess: str,
    match_tolerance_sec: float,
    nms_radius_sec: float,
    *,
    confidence_threshold: float,
    confidence_power: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    totals = init_totals(labels)
    per_video_rows: list[dict[str, Any]] = []
    allow_many_predictions_per_gt = postprocess == "window_overlap"
    for video_id in video_ids:
        video_dir = run_dir / video_id
        raw_rows = read_csv(video_dir / "window_predictions.csv")
        branch_rows = build_branch_rows(
            raw_rows,
            labels,
            branch,
            confidence_mode=branch,
            confidence_threshold=confidence_threshold,
            confidence_power=confidence_power,
        )
        predictions, matching_mode, allow_many = predictions_from_rows(
            branch_rows,
            labels,
            thresholds,
            postprocess,
            nms_radius_sec,
        )
        metrics = compute_event_metrics(
            predictions,
            load_gt_events(video_dir / "gt_events.csv", labels),
            match_tolerance_sec,
            matching_mode=matching_mode,
            allow_many_predictions_per_gt=allow_many,
        )
        for label in labels:
            item = metrics["per_class"][label]
            for key in totals[label]:
                totals[label][key] += int(item[key])
            per_video_rows.append(
                {
                    "branch": branch,
                    "video_id": video_id,
                    "label": label,
                    "precision": item["precision"],
                    "recall": item["recall"],
                    "f1": item["f1"],
                    "tp": item["tp"],
                    "fp": item["fp"],
                    "fn": item["fn"],
                    "num_pred": item["num_pred"],
                    "num_gt": item["num_gt"],
                    "num_matched_gt": item["num_matched_gt"],
                }
            )
    per_class, micro = finalize_totals(totals, allow_many_predictions_per_gt)
    return {"branch": branch, "per_class": per_class, "micro": micro}, per_video_rows


def label_near_gt(row: dict[str, str], gt_events: list[dict[str, Any]], label: str, tolerance_sec: float) -> bool:
    start = float(row["start_sec"]) - tolerance_sec
    end = float(row["end_sec"]) + tolerance_sec
    return any(event["label"] == label and start <= float(event["time_sec"]) <= end for event in gt_events)


def confidence_bucket(value: float, edges: Sequence[float]) -> str:
    prev = 0.0
    for edge in edges:
        if value < edge:
            return f"[{prev:.2f},{edge:.2f})"
        prev = edge
    return f"[{prev:.2f},1.00]"


def analyze_buckets(
    run_dir: Path,
    video_ids: Sequence[str],
    labels: Sequence[str],
    tolerance_sec: float,
    bucket_edges: Sequence[float],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for video_id in video_ids:
        video_dir = run_dir / video_id
        gt_events = load_gt_events(video_dir / "gt_events.csv", labels)
        for row in read_csv(video_dir / "window_predictions.csv"):
            confidence = float(row.get("roi_confidence") or 0.0)
            bucket = confidence_bucket(confidence, bucket_edges)
            for label in labels:
                key = (label, bucket)
                item = groups.setdefault(
                    key,
                    {
                        "label": label,
                        "roi_confidence_bucket": bucket,
                        "count": 0,
                        "positive_count": 0,
                        "roi_valid_count": 0,
                        "roi_confidence_sum": 0.0,
                        "roi_gate_sum": 0.0,
                        "global_prob_sum": 0.0,
                        "local_prob_sum": 0.0,
                        "fused_prob_sum": 0.0,
                        "local_minus_global_sum": 0.0,
                        "positive_local_minus_global_sum": 0.0,
                        "negative_local_minus_global_sum": 0.0,
                    },
                )
                is_positive = label_near_gt(row, gt_events, label, tolerance_sec)
                local_minus_global = branch_prob(row, label, "local") - branch_prob(row, label, "global")
                item["count"] += 1
                item["positive_count"] += int(is_positive)
                item["roi_valid_count"] += int(float(row.get("roi_valid") or 0.0) > 0.0)
                item["roi_confidence_sum"] += confidence
                item["roi_gate_sum"] += float(row.get(f"roi_gate_{label}") or 0.0)
                item["global_prob_sum"] += branch_prob(row, label, "global")
                item["local_prob_sum"] += branch_prob(row, label, "local")
                item["fused_prob_sum"] += branch_prob(row, label, "fused")
                item["local_minus_global_sum"] += local_minus_global
                if is_positive:
                    item["positive_local_minus_global_sum"] += local_minus_global
                else:
                    item["negative_local_minus_global_sum"] += local_minus_global
    rows: list[dict[str, Any]] = []
    for item in groups.values():
        count = max(int(item["count"]), 1)
        positives = max(int(item["positive_count"]), 1)
        negatives = max(int(item["count"]) - int(item["positive_count"]), 1)
        rows.append(
            {
                "label": item["label"],
                "roi_confidence_bucket": item["roi_confidence_bucket"],
                "count": item["count"],
                "positive_count": item["positive_count"],
                "positive_rate": item["positive_count"] / count,
                "roi_valid_rate": item["roi_valid_count"] / count,
                "mean_roi_confidence": item["roi_confidence_sum"] / count,
                "mean_roi_gate": item["roi_gate_sum"] / count,
                "mean_global_prob": item["global_prob_sum"] / count,
                "mean_local_prob": item["local_prob_sum"] / count,
                "mean_fused_prob": item["fused_prob_sum"] / count,
                "mean_local_minus_global": item["local_minus_global_sum"] / count,
                "positive_mean_local_minus_global": item["positive_local_minus_global_sum"] / positives,
                "negative_mean_local_minus_global": item["negative_local_minus_global_sum"] / negatives,
            }
        )
    return sorted(rows, key=lambda row: (row["label"], row["roi_confidence_bucket"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze global/local/fused ROI branch scores from saved eval outputs.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--video-ids", default="")
    parser.add_argument("--output-prefix", default="roi_branch_analysis")
    parser.add_argument("--postprocess", default="window_overlap", choices=["window_overlap", "point_nms"])
    parser.add_argument("--thresholds", default="")
    parser.add_argument("--match-tolerance-sec", type=float, default=5.0)
    parser.add_argument("--nms-radius-sec", type=float, default=5.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.6)
    parser.add_argument("--confidence-power", type=float, default=1.0)
    parser.add_argument("--bucket-edges", default="0.2,0.4,0.6,0.8")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    video_ids = load_video_ids(run_dir, args.video_ids)
    if not video_ids:
        raise ValueError(f"No videos found under {run_dir}")
    first_rows = read_csv(run_dir / video_ids[0] / "window_predictions.csv")
    labels = infer_labels(first_rows)
    thresholds = parse_label_floats(args.thresholds, labels, {label: 0.5 for label in labels})
    bucket_edges = [float(item) for item in args.bucket_edges.split(",") if item.strip()]
    branches = ["fused", "global", "local", "gate_no_confidence", "gate_confidence_sqrt", "gate_confidence_squared", "gate_positive_only", "confidence_switch", "confidence_logit", "confidence_boost"]

    summaries: list[dict[str, Any]] = []
    per_video_rows: list[dict[str, Any]] = []
    for branch in branches:
        summary, rows = evaluate_branch(
            run_dir,
            video_ids,
            labels,
            branch,
            thresholds,
            args.postprocess,
            args.match_tolerance_sec,
            args.nms_radius_sec,
            confidence_threshold=args.confidence_threshold,
            confidence_power=args.confidence_power,
        )
        summaries.append(summary)
        per_video_rows.extend(rows)

    aggregate_rows: list[dict[str, Any]] = []
    for summary in summaries:
        for label, item in summary["per_class"].items():
            aggregate_rows.append(
                {
                    "branch": summary["branch"],
                    "label": label,
                    "precision": item["precision"],
                    "recall": item["recall"],
                    "f1": item["f1"],
                    "tp": item["tp"],
                    "fp": item["fp"],
                    "fn": item["fn"],
                    "num_pred": item["num_pred"],
                    "num_gt": item["num_gt"],
                    "num_matched_gt": item["num_matched_gt"],
                }
            )
        micro = summary["micro"]
        aggregate_rows.append(
            {
                "branch": summary["branch"],
                "label": "micro",
                "precision": micro["precision"],
                "recall": micro["recall"],
                "f1": micro["f1"],
                "tp": micro["tp"],
                "fp": micro["fp"],
                "fn": micro["fn"],
                "num_pred": "",
                "num_gt": "",
                "num_matched_gt": "",
            }
        )

    bucket_rows = analyze_buckets(run_dir, video_ids, labels, args.match_tolerance_sec, bucket_edges)
    output = {
        "run_dir": str(run_dir),
        "video_ids": video_ids,
        "labels": labels,
        "postprocess": args.postprocess,
        "thresholds": thresholds,
        "match_tolerance_sec": args.match_tolerance_sec,
        "nms_radius_sec": args.nms_radius_sec,
        "confidence_threshold": args.confidence_threshold,
        "confidence_power": args.confidence_power,
        "summaries": summaries,
        "bucket_stats": bucket_rows,
    }

    json_path = run_dir / f"{args.output_prefix}.json"
    aggregate_csv = run_dir / f"{args.output_prefix}_metrics.csv"
    per_video_csv = run_dir / f"{args.output_prefix}_per_video_metrics.csv"
    bucket_csv = run_dir / f"{args.output_prefix}_bucket_stats.csv"
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    write_csv(aggregate_csv, aggregate_rows, list(aggregate_rows[0]) if aggregate_rows else [])
    write_csv(per_video_csv, per_video_rows, list(per_video_rows[0]) if per_video_rows else [])
    write_csv(bucket_csv, bucket_rows, list(bucket_rows[0]) if bucket_rows else [])

    for row in aggregate_rows:
        if row["label"] == "micro":
            continue
        print(
            f"{row['branch']} {row['label']}: P={row['precision']:.4f} "
            f"R={row['recall']:.4f} F1={row['f1']:.4f} "
            f"TP/FP/FN={row['tp']}/{row['fp']}/{row['fn']}"
        )
    print(f"wrote {json_path}")
    print(f"wrote {aggregate_csv}")
    print(f"wrote {per_video_csv}")
    print(f"wrote {bucket_csv}")


if __name__ == "__main__":
    main()
