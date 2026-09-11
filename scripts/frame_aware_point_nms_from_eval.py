#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from eval_long_video_checkpoint import compute_event_metrics, write_csv


LABELS = ["shot", "save"]


def parse_label_floats(raw: str, defaults: dict[str, float]) -> dict[str, float]:
    values = dict(defaults)
    if not raw.strip():
        return values
    for item in raw.split(","):
        if not item.strip():
            continue
        label, value = item.split("=", 1)
        label = label.strip()
        if label not in values:
            raise ValueError(f"Unsupported label '{label}', expected one of {sorted(values)}")
        values[label] = float(value)
    return values


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def load_gt_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["label"] not in LABELS:
                continue
            time_sec = float(row["time_sec"])
            events.append(
                {
                    "label": row["label"],
                    "time_sec": time_sec,
                    "start_sec": float(row["start_sec"] or time_sec),
                    "end_sec": float(row["end_sec"] or time_sec),
                    "event_id": row.get("event_id", ""),
                }
            )
    return events


def load_frame_peaks(video_dir: Path, branch: str) -> dict[tuple[int, str], dict[str, float]]:
    grouped: dict[tuple[int, str], list[tuple[float, float]]] = {}
    prob_prefix = {
        "global": "frame_prob_",
        "local": "local_frame_prob_",
        "roi_fused": "roi_fused_frame_prob_",
    }[branch]
    with (video_dir / "frame_event_logits.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            window_index = int(row["window_index"])
            frame_time = float(row["frame_time_sec"])
            for label in LABELS:
                grouped.setdefault((window_index, label), []).append(
                    (frame_time, float(row[f"{prob_prefix}{label}"]))
                )

    peaks: dict[tuple[int, str], dict[str, float]] = {}
    for key, values in grouped.items():
        ranked = sorted(values, key=lambda item: item[1], reverse=True)
        peak_time, peak_prob = ranked[0]
        top4 = ranked[: min(4, len(ranked))]
        top4_mean = sum(prob for _, prob in top4) / len(top4)
        peaks[key] = {
            "peak_time_sec": peak_time,
            "peak_prob": peak_prob,
            "sharpness": peak_prob - top4_mean,
        }
    return peaks


def point_nms(candidates: list[dict[str, Any]], radius_sec: float) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: (-float(item["score"]), float(item["time_sec"]))):
        nearby = next(
            (
                old
                for old in kept
                if abs(float(candidate["time_sec"]) - float(old["time_sec"])) <= radius_sec
            ),
            None,
        )
        if nearby is not None:
            nearby["num_windows"] += 1
            nearby["window_indices"].extend(candidate["window_indices"])
            nearby["support_start_sec"] = min(
                float(nearby["support_start_sec"]), float(candidate["support_start_sec"])
            )
            nearby["support_end_sec"] = max(
                float(nearby["support_end_sec"]), float(candidate["support_end_sec"])
            )
            continue
        kept.append(dict(candidate))
    return kept


def build_candidates(
    video_dir: Path,
    *,
    branch: str,
    clip_thresholds: dict[str, float],
    frame_mins: dict[str, float],
    alpha: float,
    beta: float,
    sharpness_scale: float,
) -> list[dict[str, Any]]:
    peaks = load_frame_peaks(video_dir, branch)
    candidates: list[dict[str, Any]] = []
    with (video_dir / "window_predictions.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            window_index = int(row["index"])
            support_start = float(row["start_sec"])
            support_end = float(row["end_sec"])
            for label in LABELS:
                clip_prob = float(row[f"prob_{label}"])
                if clip_prob < clip_thresholds[label]:
                    continue
                peak = peaks.get((window_index, label))
                if peak is None or peak["peak_prob"] < frame_mins[label]:
                    continue
                fused_score = (
                    (clip_prob ** alpha)
                    * (peak["peak_prob"] ** beta)
                    * sigmoid(sharpness_scale * peak["sharpness"])
                )
                candidates.append(
                    {
                        "label": label,
                        "time_sec": peak["peak_time_sec"],
                        "start_sec": peak["peak_time_sec"],
                        "end_sec": peak["peak_time_sec"],
                        "support_start_sec": support_start,
                        "support_end_sec": support_end,
                        "score": fused_score,
                        "clip_prob": clip_prob,
                        "frame_peak_prob": peak["peak_prob"],
                        "frame_sharpness": peak["sharpness"],
                        "window_indices": [window_index],
                        "num_windows": 1,
                    }
                )
    return candidates


def evaluate_run(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    run_dir = Path(args.run_dir)
    video_ids = [item.strip() for item in args.video_ids.split(",") if item.strip()]
    if not video_ids:
        video_ids = sorted(
            path.name
            for path in run_dir.iterdir()
            if path.is_dir() and (path / "window_predictions.csv").exists()
        )

    clip_thresholds = parse_label_floats(args.clip_thresholds, {"shot": 0.20, "save": 0.35})
    frame_mins = parse_label_floats(args.frame_mins, {"shot": 0.05, "save": 0.04})
    final_thresholds = parse_label_floats(args.final_thresholds, {"shot": 0.26, "save": 0.22})

    per_video_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    totals = {
        label: {"tp": 0, "fp": 0, "fn": 0, "num_pred": 0, "num_gt": 0, "num_matched_gt": 0}
        for label in LABELS
    }

    for video_id in video_ids:
        video_dir = run_dir / video_id
        candidates = build_candidates(
            video_dir,
            branch=args.branch,
            clip_thresholds=clip_thresholds,
            frame_mins=frame_mins,
            alpha=args.alpha,
            beta=args.beta,
            sharpness_scale=args.sharpness_scale,
        )
        predictions: list[dict[str, Any]] = []
        for label in LABELS:
            label_candidates = [
                candidate
                for candidate in candidates
                if candidate["label"] == label and float(candidate["score"]) >= final_thresholds[label]
            ]
            predictions.extend(point_nms(label_candidates, args.nms_radius_sec))

        gt_events = load_gt_events(video_dir / "gt_events.csv")
        metrics = compute_event_metrics(
            predictions,
            gt_events,
            args.match_tolerance_sec,
            matching_mode="point",
            allow_many_predictions_per_gt=False,
        )

        for pred in predictions:
            prediction_rows.append(
                {
                    "video_id": video_id,
                    "label": pred["label"],
                    "time_sec": pred["time_sec"],
                    "score": pred["score"],
                    "clip_prob": pred["clip_prob"],
                    "frame_peak_prob": pred["frame_peak_prob"],
                    "frame_sharpness": pred["frame_sharpness"],
                    "num_windows": pred["num_windows"],
                    "window_indices": json.dumps(pred["window_indices"]),
                }
            )

        for label in LABELS:
            item = metrics["per_class"][label]
            for key in totals[label]:
                totals[label][key] += int(item[key])
            per_video_rows.append(
                {
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

    per_class: dict[str, Any] = {}
    for label, total in totals.items():
        precision = total["tp"] / (total["tp"] + total["fp"]) if total["tp"] + total["fp"] else 0.0
        recall = total["tp"] / (total["tp"] + total["fn"]) if total["tp"] + total["fn"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {**total, "precision": precision, "recall": recall, "f1": f1}

    micro_tp = sum(item["tp"] for item in totals.values())
    micro_fp = sum(item["fp"] for item in totals.values())
    micro_fn = sum(item["fn"] for item in totals.values())
    micro_precision = micro_tp / (micro_tp + micro_fp) if micro_tp + micro_fp else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if micro_tp + micro_fn else 0.0
    summary = {
        "postprocess": "frame_aware_point_nms",
        "labels": LABELS,
        "video_ids": video_ids,
        "branch": args.branch,
        "clip_thresholds": clip_thresholds,
        "frame_mins": frame_mins,
        "final_thresholds": final_thresholds,
        "alpha": args.alpha,
        "beta": args.beta,
        "sharpness_scale": args.sharpness_scale,
        "nms_radius_sec": args.nms_radius_sec,
        "match_tolerance_sec": args.match_tolerance_sec,
        "per_class": per_class,
        "micro": {
            "tp": micro_tp,
            "fp": micro_fp,
            "fn": micro_fn,
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": (
                2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                if micro_precision + micro_recall
                else 0.0
            ),
        },
        "per_video": per_video_rows,
    }
    return summary, per_video_rows, prediction_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frame-aware PointNMS postprocess from saved football eval outputs.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--video-ids", default="")
    parser.add_argument("--output-prefix", default="frame_aware_point_nms")
    parser.add_argument("--branch", default="roi_fused", choices=["global", "local", "roi_fused"])
    parser.add_argument("--clip-thresholds", default="shot=0.20,save=0.35")
    parser.add_argument("--frame-mins", default="shot=0.05,save=0.04")
    parser.add_argument("--final-thresholds", default="shot=0.26,save=0.22")
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--sharpness-scale", type=float, default=8.0)
    parser.add_argument("--nms-radius-sec", type=float, default=5.0)
    parser.add_argument("--match-tolerance-sec", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    summary, per_video_rows, prediction_rows = evaluate_run(args)
    json_path = run_dir / f"{args.output_prefix}_metrics.json"
    csv_path = run_dir / f"{args.output_prefix}_metrics.csv"
    pred_path = run_dir / f"{args.output_prefix}_predictions.csv"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    write_csv(csv_path, per_video_rows, list(per_video_rows[0]) if per_video_rows else [])
    if prediction_rows:
        write_csv(pred_path, prediction_rows, list(prediction_rows[0]))

    for label in LABELS:
        item = summary["per_class"][label]
        print(
            f"{label}: P={item['precision']:.4f} R={item['recall']:.4f} "
            f"F1={item['f1']:.4f} TP/FP/FN={item['tp']}/{item['fp']}/{item['fn']}"
        )
    micro = summary["micro"]
    print(
        f"micro: P={micro['precision']:.4f} R={micro['recall']:.4f} "
        f"F1={micro['f1']:.4f} TP/FP/FN={micro['tp']}/{micro['fp']}/{micro['fn']}"
    )
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    if prediction_rows:
        print(f"wrote {pred_path}")


if __name__ == "__main__":
    main()
