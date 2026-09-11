#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_LABELS = ("shot", "save", "set_piece")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize dense football predictions as independent window classifications."
    )
    parser.add_argument("--eval-run-dir", required=True)
    parser.add_argument(
        "--thresholds",
        default="checkpoint",
        help="checkpoint, a scalar, or label values such as shot=0.35,save=0.45,set_piece=0.8",
    )
    parser.add_argument(
        "--sweep",
        default="0.05:0.95:0.05",
        help="Threshold sweep as start:end:step or comma-separated values.",
    )
    parser.add_argument("--labels", default="", help="Defaults to labels in run_config.json.")
    return parser.parse_args()


def safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def finalize_counts(counts: dict[str, int]) -> dict[str, Any]:
    tp, fp, fn, tn = (int(counts[key]) for key in ("tp", "fp", "fn", "tn"))
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "num_windows": tp + fp + fn + tn,
        "num_positive_windows": tp + fn,
        "num_negative_windows": fp + tn,
        "num_predicted_positive_windows": tp + fp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def parse_sweep(text: str) -> list[float]:
    if ":" not in text:
        values = [float(item.strip()) for item in text.split(",") if item.strip()]
    else:
        parts = [float(item) for item in text.split(":")]
        if len(parts) != 3:
            raise ValueError("--sweep must be start:end:step or comma-separated values")
        start, end, step = parts
        if step <= 0:
            raise ValueError("--sweep step must be positive")
        values = []
        value = start
        while value <= end + step * 1e-6:
            values.append(value)
            value += step
    if not values or any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("Sweep thresholds must be in [0, 1]")
    return sorted(set(round(value, 10) for value in values))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_thresholds(video_dirs: Sequence[Path], labels: Sequence[str]) -> dict[str, float]:
    resolved: dict[str, float] | None = None
    for video_dir in video_dirs:
        summary_path = video_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        current = {label: float(summary["thresholds"][label]) for label in labels}
        if resolved is not None and any(abs(current[label] - resolved[label]) > 1e-8 for label in labels):
            raise ValueError("Checkpoint thresholds differ across video directories")
        resolved = current
    if resolved is None:
        raise FileNotFoundError("No per-video summary.json with checkpoint thresholds found")
    return resolved


def parse_thresholds(text: str, video_dirs: Sequence[Path], labels: Sequence[str]) -> dict[str, float]:
    if text == "checkpoint":
        return checkpoint_thresholds(video_dirs, labels)
    if "=" not in text:
        value = float(text)
        return {label: value for label in labels}
    values: dict[str, float] = {}
    for item in text.split(","):
        label, raw_value = item.split("=", 1)
        values[label.strip()] = float(raw_value)
    missing = set(labels) - set(values)
    if missing:
        raise ValueError(f"Missing thresholds for labels: {sorted(missing)}")
    return {label: values[label] for label in labels}


def event_in_window(event: dict[str, Any], window: dict[str, str]) -> bool:
    time_sec = float(event.get("time_sec", event.get("start_sec", 0.0)))
    return float(window["start_sec"]) <= time_sec < float(window["end_sec"])


def evaluate_video_label(
    windows: Sequence[dict[str, str]],
    gt_events: Sequence[dict[str, Any]],
    label: str,
    threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    label_events = [event for event in gt_events if event.get("label") == label]
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    predictions: list[bool] = []
    probabilities: list[float] = []
    for window in windows:
        target = any(event_in_window(event, window) for event in label_events)
        probability = float(window[f"prob_{label}"])
        prediction = probability >= threshold
        predictions.append(prediction)
        probabilities.append(probability)
        if prediction and target:
            counts["tp"] += 1
        elif prediction:
            counts["fp"] += 1
        elif target:
            counts["fn"] += 1
        else:
            counts["tn"] += 1

    event_rows: list[dict[str, Any]] = []
    for event_index, event in enumerate(label_events):
        covering = [index for index, window in enumerate(windows) if event_in_window(event, window)]
        predicted_covering = [index for index in covering if predictions[index]]
        event_rows.append(
            {
                "label": label,
                "event_index": event_index,
                "event_id": event.get("event_id", ""),
                "time_sec": float(event.get("time_sec", event.get("start_sec", 0.0))),
                "threshold": threshold,
                "num_covering_windows": len(covering),
                "num_predicted_positive_windows": len(predicted_covering),
                "max_probability": max((probabilities[index] for index in covering), default=0.0),
                "hit": int(bool(predicted_covering)),
                "covering_window_indices": ",".join(str(windows[index]["index"]) for index in covering),
                "positive_window_indices": ",".join(str(windows[index]["index"]) for index in predicted_covering),
            }
        )

    metrics = finalize_counts(counts)
    num_hit_events = sum(int(row["hit"]) for row in event_rows)
    metrics.update(
        {
            "threshold": threshold,
            "num_gt_events": len(event_rows),
            "num_hit_gt_events": num_hit_events,
            "event_coverage_recall": safe_div(num_hit_events, len(event_rows)),
        }
    )
    return metrics, event_rows


def aggregate(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    counts = {key: sum(int(row[key]) for row in rows) for key in ("tp", "fp", "fn", "tn")}
    metrics = finalize_counts(counts)
    num_gt_events = sum(int(row["num_gt_events"]) for row in rows)
    num_hit_gt_events = sum(int(row["num_hit_gt_events"]) for row in rows)
    metrics.update(
        {
            "num_gt_events": num_gt_events,
            "num_hit_gt_events": num_hit_gt_events,
            "event_coverage_recall": safe_div(num_hit_gt_events, num_gt_events),
        }
    )
    return metrics


def main() -> None:
    args = parse_args()
    run_dir = Path(args.eval_run_dir)
    run_config_path = run_dir / "run_config.json"
    run_config = json.loads(run_config_path.read_text()) if run_config_path.exists() else {}
    labels = [item.strip() for item in args.labels.split(",") if item.strip()]
    labels = labels or list(run_config.get("labels", DEFAULT_LABELS))
    configured_video_ids = list(run_config.get("video_ids", []))
    video_dirs = [run_dir / video_id for video_id in configured_video_ids]
    if not video_dirs:
        video_dirs = sorted(path for path in run_dir.iterdir() if path.is_dir())
    video_dirs = [
        path
        for path in video_dirs
        if (path / "window_predictions.csv").exists() and (path / "gt_events.json").exists()
    ]
    if not video_dirs:
        raise FileNotFoundError(f"No complete per-video outputs found under {run_dir}")

    selected_thresholds = parse_thresholds(args.thresholds, video_dirs, labels)
    sweep_thresholds = parse_sweep(args.sweep)
    video_data = {
        video_dir.name: (
            read_csv(video_dir / "window_predictions.csv"),
            json.loads((video_dir / "gt_events.json").read_text()),
        )
        for video_dir in video_dirs
    }

    fixed_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for video_id, (windows, gt_events) in video_data.items():
        for label in labels:
            metrics, events = evaluate_video_label(windows, gt_events, label, selected_thresholds[label])
            fixed_rows.append({"video_id": video_id, "label": label, **metrics})
            event_rows.extend({"video_id": video_id, **event} for event in events)

    aggregate_rows = []
    for label in labels:
        item = aggregate(row for row in fixed_rows if row["label"] == label)
        aggregate_rows.append({"label": label, "threshold": selected_thresholds[label], **item})
    micro = aggregate(fixed_rows)

    sweep_video_rows: list[dict[str, Any]] = []
    for threshold in sweep_thresholds:
        for video_id, (windows, gt_events) in video_data.items():
            for label in labels:
                metrics, _ = evaluate_video_label(windows, gt_events, label, threshold)
                sweep_video_rows.append({"video_id": video_id, "label": label, **metrics})

    sweep_aggregate_rows: list[dict[str, Any]] = []
    for threshold in sweep_thresholds:
        threshold_rows = [row for row in sweep_video_rows if row["threshold"] == threshold]
        for label in labels:
            item = aggregate(row for row in threshold_rows if row["label"] == label)
            sweep_aggregate_rows.append({"scope": "class", "label": label, "threshold": threshold, **item})
        sweep_aggregate_rows.append(
            {"scope": "micro", "label": "micro", "threshold": threshold, **aggregate(threshold_rows)}
        )

    metric_fields = [
        "video_id", "label", "threshold", "tp", "fp", "fn", "tn", "num_windows",
        "num_positive_windows", "num_negative_windows", "num_predicted_positive_windows",
        "precision", "recall", "f1", "num_gt_events", "num_hit_gt_events", "event_coverage_recall",
    ]
    write_csv(run_dir / "window_metrics_by_video.csv", fixed_rows, metric_fields)
    write_csv(run_dir / "window_threshold_sweep_by_video.csv", sweep_video_rows, metric_fields)
    write_csv(
        run_dir / "window_threshold_sweep_aggregate.csv",
        sweep_aggregate_rows,
        ["scope", "label", "threshold", *metric_fields[3:]],
    )
    write_csv(
        run_dir / "window_gt_event_hits.csv",
        event_rows,
        [
            "video_id", "label", "event_index", "event_id", "time_sec", "threshold",
            "num_covering_windows", "num_predicted_positive_windows", "max_probability", "hit",
            "covering_window_indices", "positive_window_indices",
        ],
    )
    summary = {
        "protocol": "independent_window_classification",
        "positive_window_definition": "same-label GT time_sec in [start_sec, end_sec)",
        "eval_run_dir": str(run_dir),
        "labels": labels,
        "video_ids": list(video_data),
        "thresholds": selected_thresholds,
        "per_video": fixed_rows,
        "per_class": {row["label"]: row for row in aggregate_rows},
        "micro": micro,
    }
    (run_dir / "window_metrics_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"wrote {run_dir / 'window_metrics_by_video.csv'}")
    print(f"wrote {run_dir / 'window_threshold_sweep_by_video.csv'}")
    print(f"wrote {run_dir / 'window_threshold_sweep_aggregate.csv'}")
    print(f"wrote {run_dir / 'window_gt_event_hits.csv'}")
    print(f"wrote {run_dir / 'window_metrics_summary.json'}")


if __name__ == "__main__":
    main()
