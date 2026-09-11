#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from eval_long_video_checkpoint import (  # noqa: E402
    compute_event_metrics,
    point_nms_predictions,
    window_overlap_predictions,
    write_csv,
)

PROTOCOLS = ("point_nms", "window_overlap")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def parse_thresholds(raw: str, labels: Sequence[str], defaults: dict[str, float]) -> dict[str, float]:
    values = {label: float(defaults[label]) for label in labels}
    if not raw.strip():
        return values
    for item in raw.split(","):
        if not item.strip():
            continue
        label, value = item.split("=", 1)
        label = label.strip()
        if label not in values:
            raise ValueError(f"Unsupported threshold label '{label}', expected one of {list(labels)}")
        values[label] = float(value)
    return values


def load_video_ids(run_dir: Path, raw: str) -> list[str]:
    if raw.strip():
        video_ids = [item.strip() for item in raw.split(",") if item.strip()]
    else:
        video_ids = sorted(
            path.name
            for path in run_dir.iterdir()
            if path.is_dir() and (path / "window_predictions.csv").is_file()
        )
    missing = [
        video_id
        for video_id in video_ids
        if not (run_dir / video_id / "window_predictions.csv").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing window predictions for video_ids={missing}")
    return video_ids


def load_run_metadata(run_dir: Path, video_ids: Sequence[str]) -> tuple[list[str], dict[str, float]]:
    if not video_ids:
        raise ValueError(f"No completed video evaluations found under {run_dir}")
    summaries = [json.loads((run_dir / video_id / "summary.json").read_text()) for video_id in video_ids]
    labels = [str(label) for label in summaries[0]["labels"]]
    thresholds = {label: float(summaries[0]["thresholds"][label]) for label in labels}
    for video_id, summary in zip(video_ids[1:], summaries[1:]):
        if [str(label) for label in summary["labels"]] != labels:
            raise ValueError(f"Label mismatch in {video_id}/summary.json")
        current = {label: float(summary["thresholds"][label]) for label in labels}
        if current != thresholds:
            raise ValueError(f"Threshold mismatch in {video_id}/summary.json: {current} != {thresholds}")
    return labels, thresholds


def load_gt_events(path: Path, labels: Sequence[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for row in read_csv(path):
        label = row["label"]
        if label not in labels:
            continue
        time_sec = float(row["time_sec"])
        events.append(
            {
                "label": label,
                "time_sec": time_sec,
                "start_sec": float(row.get("start_sec") or time_sec),
                "end_sec": float(row.get("end_sec") or time_sec),
                "event_id": row.get("event_id", ""),
            }
        )
    return events


def normalize_score_columns(
    rows: list[dict[str, str]], labels: Sequence[str], score_prefix: str
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {
            "index": int(row["index"]),
            "start_sec": float(row["start_sec"]),
            "end_sec": float(row["end_sec"]),
        }
        for label in labels:
            source = f"{score_prefix}_{label}"
            if source not in row or row[source] == "":
                raise KeyError(f"Missing score column '{source}' in window_predictions.csv")
            item[f"prob_{label}"] = float(row[source])
        normalized.append(item)
    return normalized


def init_totals(labels: Sequence[str]) -> dict[str, dict[str, int]]:
    return {
        label: {"tp": 0, "fp": 0, "fn": 0, "num_pred": 0, "num_gt": 0, "num_matched_gt": 0}
        for label in labels
    }


def finalize_totals(
    totals: dict[str, dict[str, int]], *, allow_many_predictions_per_gt: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    per_class: dict[str, Any] = {}
    for label, total in totals.items():
        precision = total["tp"] / (total["tp"] + total["fp"]) if total["tp"] + total["fp"] else 0.0
        if allow_many_predictions_per_gt:
            recall = total["num_matched_gt"] / total["num_gt"] if total["num_gt"] else 0.0
        else:
            recall = total["tp"] / (total["tp"] + total["fn"]) if total["tp"] + total["fn"] else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {**total, "precision": precision, "recall": recall, "f1": f1}

    tp = sum(item["tp"] for item in totals.values())
    fp = sum(item["fp"] for item in totals.values())
    fn = sum(item["fn"] for item in totals.values())
    precision = tp / (tp + fp) if tp + fp else 0.0
    if allow_many_predictions_per_gt:
        num_gt = sum(item["num_gt"] for item in totals.values())
        matched = sum(item["num_matched_gt"] for item in totals.values())
        recall = matched / num_gt if num_gt else 0.0
    else:
        recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return per_class, {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def evaluate_protocol(
    run_dir: Path,
    video_ids: Sequence[str],
    labels: Sequence[str],
    thresholds: dict[str, float],
    protocol: str,
    *,
    score_prefix: str,
    nms_radius_sec: float,
    match_tolerance_sec: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    totals = init_totals(labels)
    per_video: list[dict[str, Any]] = []
    allow_many = protocol == "window_overlap"
    for video_id in video_ids:
        video_dir = run_dir / video_id
        rows = normalize_score_columns(read_csv(video_dir / "window_predictions.csv"), labels, score_prefix)
        if protocol == "point_nms":
            predictions = point_nms_predictions(rows, labels, thresholds, nms_radius_sec)
            matching_mode = "point"
        elif protocol == "window_overlap":
            predictions = window_overlap_predictions(rows, labels, thresholds)
            matching_mode = "window"
        else:
            raise ValueError(f"Unsupported protocol={protocol}")
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
            per_video.append({"protocol": protocol, "video_id": video_id, "label": label, **item})
    per_class, micro = finalize_totals(totals, allow_many_predictions_per_gt=allow_many)
    return {"per_class": per_class, "micro": micro}, per_video


def recompute_protocols(
    run_dir: Path,
    video_ids: Sequence[str],
    labels: Sequence[str],
    thresholds: dict[str, float],
    *,
    score_prefix: str,
    nms_radius_sec: float,
    match_tolerance_sec: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol_metrics: dict[str, Any] = {}
    per_video_rows: list[dict[str, Any]] = []
    for protocol in PROTOCOLS:
        metrics, rows = evaluate_protocol(
            run_dir,
            video_ids,
            labels,
            thresholds,
            protocol,
            score_prefix=score_prefix,
            nms_radius_sec=nms_radius_sec,
            match_tolerance_sec=match_tolerance_sec,
        )
        protocol_metrics[protocol] = metrics
        per_video_rows.extend(rows)
    output = {
        "run_dir": str(run_dir),
        "video_ids": list(video_ids),
        "labels": list(labels),
        "thresholds": thresholds,
        "score_prefix": score_prefix,
        "nms_radius_sec": nms_radius_sec,
        "match_tolerance_sec": match_tolerance_sec,
        "protocols": protocol_metrics,
        "per_video": per_video_rows,
    }
    return output, per_video_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute PointNMS and window-overlap metrics from one saved football inference run."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--video-ids", default="")
    parser.add_argument("--thresholds", default="", help="Optional label=value list; defaults to saved summary thresholds.")
    parser.add_argument("--score-prefix", default="prob", help="Column prefix, e.g. prob or global_prob.")
    parser.add_argument("--nms-radius-sec", type=float, default=5.0)
    parser.add_argument("--match-tolerance-sec", type=float, default=5.0)
    parser.add_argument("--output-prefix", default="protocol_comparison")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    video_ids = load_video_ids(run_dir, args.video_ids)
    labels, saved_thresholds = load_run_metadata(run_dir, video_ids)
    thresholds = parse_thresholds(args.thresholds, labels, saved_thresholds)
    output, per_video_rows = recompute_protocols(
        run_dir,
        video_ids,
        labels,
        thresholds,
        score_prefix=args.score_prefix,
        nms_radius_sec=args.nms_radius_sec,
        match_tolerance_sec=args.match_tolerance_sec,
    )
    json_path = run_dir / f"{args.output_prefix}.json"
    csv_path = run_dir / f"{args.output_prefix}_per_video.csv"
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    write_csv(csv_path, per_video_rows, list(per_video_rows[0]) if per_video_rows else [])
    for protocol in PROTOCOLS:
        for label in labels:
            item = output["protocols"][protocol]["per_class"][label]
            print(
                f"{protocol} class={label} p={item['precision']:.4f} r={item['recall']:.4f} "
                f"f1={item['f1']:.4f} tp/fp/fn={item['tp']}/{item['fp']}/{item['fn']}"
            )
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
