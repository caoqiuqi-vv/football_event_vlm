#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_long_video_checkpoint import (
    compute_event_metrics,
    merge_predictions,
    point_nms_predictions,
    window_overlap_predictions,
)


DEFAULT_LABELS = ["shot", "save", "set_piece"]


def parse_key_values(raw_items: Sequence[str], *, name: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in raw_items:
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(f"{name} item must be key=value, got {item!r}")
            key, value = item.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key or not value:
                raise ValueError(f"{name} item must be key=value, got {item!r}")
            result[key] = value
    return result


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def read_video_ids(args: argparse.Namespace, source_runs: dict[str, Path]) -> list[str]:
    ids: list[str] = []
    if args.video_ids:
        ids.extend(item.strip() for item in args.video_ids.split(",") if item.strip())
    if args.video_id_file:
        for line in Path(args.video_id_file).read_text().splitlines():
            item = line.strip()
            if item and not item.startswith("#"):
                ids.append(item)
    if not ids:
        sets = []
        for run_dir in source_runs.values():
            sets.append({path.name for path in run_dir.iterdir() if path.is_dir()})
        if sets:
            ids = sorted(set.intersection(*sets))
    seen: set[str] = set()
    unique: list[str] = []
    for video_id in ids:
        if video_id not in seen:
            unique.append(video_id)
            seen.add(video_id)
    return unique


def load_thresholds(
    *,
    raw: str,
    labels: Sequence[str],
    label_sources: dict[str, str],
    source_runs: dict[str, Path],
    video_ids: Sequence[str],
) -> dict[str, float]:
    raw = raw.strip()
    if raw.lower() != "checkpoint":
        if "=" not in raw:
            value = float(raw)
            return {label: value for label in labels}
        parsed = parse_key_values([raw], name="--thresholds")
        missing = [label for label in labels if label not in parsed]
        if missing:
            raise ValueError(f"--thresholds missing labels: {missing}")
        return {label: float(parsed[label]) for label in labels}

    thresholds: dict[str, float] = {}
    threshold_sources: dict[str, str] = {}
    for label in labels:
        source_name = label_sources[label]
        run_dir = source_runs[source_name]
        values: set[float] = set()
        for video_id in video_ids:
            summary_path = run_dir / video_id / "summary.json"
            if not summary_path.exists():
                raise FileNotFoundError(f"Missing summary.json for checkpoint thresholds: {summary_path}")
            summary = json.loads(summary_path.read_text())
            per_video_thresholds = summary.get("thresholds", {})
            if label not in per_video_thresholds:
                raise ValueError(f"Missing checkpoint threshold label={label} in {summary_path}")
            values.add(float(per_video_thresholds[label]))
        if len(values) != 1:
            raise ValueError(f"Inconsistent checkpoint thresholds for label={label} source={source_name}: {values}")
        thresholds[label] = next(iter(values))
        threshold_sources[label] = source_name
    return thresholds


def merge_video_rows(
    *,
    video_id: str,
    labels: Sequence[str],
    label_sources: dict[str, str],
    source_runs: dict[str, Path],
    thresholds: dict[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows_by_source: dict[str, list[dict[str, str]]] = {}
    for source_name, run_dir in source_runs.items():
        path = run_dir / video_id / "window_predictions.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        rows_by_source[source_name] = read_csv(path)

    lengths = {source_name: len(rows) for source_name, rows in rows_by_source.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Window row count mismatch for video={video_id}: {lengths}")

    reference_source = next(iter(source_runs))
    reference_rows = rows_by_source[reference_source]
    merged_rows: list[dict[str, Any]] = []
    for row_index, ref in enumerate(reference_rows):
        merged: dict[str, Any] = {
            "index": int(ref["index"]),
            "start_sec": float(ref["start_sec"]),
            "end_sec": float(ref["end_sec"]),
        }
        for source_name, rows in rows_by_source.items():
            row = rows[row_index]
            if int(row["index"]) != merged["index"]:
                raise ValueError(f"Window index mismatch video={video_id} row={row_index} source={source_name}")
            for key in ("start_sec", "end_sec"):
                if abs(float(row[key]) - float(merged[key])) > 1e-6:
                    raise ValueError(f"Window {key} mismatch video={video_id} row={row_index} source={source_name}")
        for label in labels:
            source_name = label_sources[label]
            source_row = rows_by_source[source_name][row_index]
            prob_key = f"prob_{label}"
            if prob_key not in source_row:
                raise ValueError(f"Missing {prob_key} for source={source_name} video={video_id}")
            prob = float(source_row[prob_key])
            merged[prob_key] = prob
            merged[f"pred_{label}"] = int(prob >= thresholds[label])
            merged[f"source_{label}"] = source_name
        merged_rows.append(merged)

    gt_path = source_runs[reference_source] / video_id / "gt_events.json"
    if not gt_path.exists():
        raise FileNotFoundError(gt_path)
    gt_events = json.loads(gt_path.read_text())
    return merged_rows, gt_events


def build_predictions(
    rows: list[dict[str, Any]],
    *,
    labels: Sequence[str],
    thresholds: dict[str, float],
    postprocess: str,
    nms_radius_sec: float,
    merge_gap_sec: float,
) -> tuple[list[dict[str, Any]], str, bool]:
    if postprocess == "window_overlap":
        return window_overlap_predictions(rows, labels, thresholds), "window", True
    if postprocess == "point_nms":
        return point_nms_predictions(rows, labels, thresholds, nms_radius_sec), "point", False
    if postprocess == "interval_merge":
        return merge_predictions(rows, labels, thresholds, merge_gap_sec), "interval", False
    raise ValueError(f"Unsupported postprocess={postprocess}")


def metric_zero() -> dict[str, Any]:
    return {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "num_pred": 0,
        "num_gt": 0,
        "num_matched_gt": 0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }


def finalize_counts(item: dict[str, Any], *, allow_many_predictions_per_gt: bool) -> dict[str, Any]:
    tp = int(item.get("tp", 0))
    fp = int(item.get("fp", 0))
    fn = int(item.get("fn", 0))
    num_gt = int(item.get("num_gt", tp + fn))
    num_matched_gt = int(item.get("num_matched_gt", tp))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = (num_matched_gt / num_gt) if allow_many_predictions_per_gt and num_gt else (
        tp / (tp + fn) if tp + fn else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    item.update(
        {
            "num_pred": int(item.get("num_pred", tp + fp)),
            "num_gt": num_gt,
            "num_matched_gt": num_matched_gt,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )
    return item


def write_outputs(
    *,
    output_dir: Path,
    video_id: str,
    rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    gt_events: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    video_dir = output_dir / video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    (video_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    (video_dir / "predicted_events.json").write_text(json.dumps(predictions, ensure_ascii=False, indent=2))
    (video_dir / "gt_events.json").write_text(json.dumps(gt_events, ensure_ascii=False, indent=2))
    window_fields = ["index", "start_sec", "end_sec"]
    for label in DEFAULT_LABELS:
        if any(f"prob_{label}" in row for row in rows):
            window_fields.extend([f"prob_{label}", f"pred_{label}", f"source_{label}"])
    write_csv(video_dir / "window_predictions.csv", rows, window_fields)
    write_csv(
        video_dir / "predicted_events.csv",
        predictions,
        [
            "label",
            "time_sec",
            "start_sec",
            "end_sec",
            "support_start_sec",
            "support_end_sec",
            "score",
            "num_windows",
            "window_indices",
        ],
    )
    write_csv(
        video_dir / "gt_events.csv",
        gt_events,
        ["label", "time_sec", "start_sec", "end_sec", "raw_label", "event_type", "event_id", "num_events", "event_ids"],
    )
    write_csv(
        video_dir / "matches.csv",
        metrics.get("matches", []),
        [
            "label",
            "pred_time_sec",
            "pred_start_sec",
            "pred_end_sec",
            "pred_score",
            "gt_time_sec",
            "gt_event_id",
            "distance_sec",
        ],
    )


def summarize(output_dir: Path, video_ids: Sequence[str], labels: Sequence[str], *, allow_many_predictions_per_gt: bool) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    totals = {label: metric_zero() for label in labels}
    micro = metric_zero()
    for video_id in video_ids:
        metrics_path = output_dir / video_id / "metrics.json"
        if not metrics_path.exists():
            rows.append({"video_id": video_id, "status": "missing"})
            continue
        metrics = json.loads(metrics_path.read_text())
        for label in labels:
            item = metrics["per_class"].get(label, metric_zero())
            row = {
                "video_id": video_id,
                "label": label,
                "tp": int(item["tp"]),
                "fp": int(item["fp"]),
                "fn": int(item["fn"]),
                "num_pred": int(item["num_pred"]),
                "num_gt": int(item["num_gt"]),
                "num_matched_gt": int(item.get("num_matched_gt", item["tp"])),
                "precision": float(item["precision"]),
                "recall": float(item["recall"]),
                "f1": float(item["f1"]),
                "status": "done",
            }
            rows.append(row)
            for key in ("tp", "fp", "fn", "num_pred", "num_gt", "num_matched_gt"):
                totals[label][key] += row[key]
                micro[key] += row[key]

    totals = {
        label: finalize_counts(item, allow_many_predictions_per_gt=allow_many_predictions_per_gt)
        for label, item in totals.items()
    }
    micro = finalize_counts(micro, allow_many_predictions_per_gt=allow_many_predictions_per_gt)
    summary = {
        "allow_many_predictions_per_gt": allow_many_predictions_per_gt,
        "per_video": rows,
        "per_class": totals,
        "micro": micro,
    }
    (output_dir / "summary_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    write_csv(
        output_dir / "summary_metrics.csv",
        rows,
        [
            "video_id",
            "label",
            "tp",
            "fp",
            "fn",
            "num_pred",
            "num_gt",
            "num_matched_gt",
            "precision",
            "recall",
            "f1",
            "status",
        ],
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Route per-label probabilities from multiple football dense-eval runs, "
            "then recompute event metrics with the standard postprocess."
        )
    )
    parser.add_argument(
        "--source-run",
        action="append",
        required=True,
        help="Named eval run directory, e.g. e1=outputs/football_eval_runs/e1 or e2=outputs/football_eval_runs/e2.",
    )
    parser.add_argument(
        "--label-source",
        required=True,
        help="Label routing map, e.g. shot=e2,save=e2,set_piece=e1.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--video-ids", default="")
    parser.add_argument("--video-id-file", default="")
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument(
        "--thresholds",
        default="checkpoint",
        help="checkpoint, scalar, or label list such as shot=0.35,save=0.45,set_piece=0.65.",
    )
    parser.add_argument(
        "--prediction-postprocess",
        default="point_nms",
        choices=["window_overlap", "point_nms", "interval_merge"],
    )
    parser.add_argument("--nms-radius-sec", type=float, default=5.0)
    parser.add_argument("--merge-gap-sec", type=float, default=2.0)
    parser.add_argument("--match-tolerance-sec", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_run_map = parse_key_values(args.source_run, name="--source-run")
    source_runs = {name: Path(path) for name, path in source_run_map.items()}
    for name, run_dir in source_runs.items():
        if not run_dir.exists():
            raise FileNotFoundError(f"Missing source run {name}={run_dir}")
    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    label_sources = parse_key_values([args.label_source], name="--label-source")
    missing_labels = [label for label in labels if label not in label_sources]
    if missing_labels:
        raise ValueError(f"--label-source missing labels: {missing_labels}")
    unknown_sources = {source for source in label_sources.values() if source not in source_runs}
    if unknown_sources:
        raise ValueError(f"--label-source references unknown source runs: {sorted(unknown_sources)}")

    video_ids = read_video_ids(args, source_runs)
    if not video_ids:
        raise RuntimeError("No video ids to route")
    thresholds = load_thresholds(
        raw=args.thresholds,
        labels=labels,
        label_sources=label_sources,
        source_runs=source_runs,
        video_ids=video_ids,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    allow_many_predictions_per_gt = args.prediction_postprocess == "window_overlap"
    matching_mode = "window" if allow_many_predictions_per_gt else (
        "point" if args.prediction_postprocess == "point_nms" else "interval"
    )

    for video_id in video_ids:
        rows, gt_events = merge_video_rows(
            video_id=video_id,
            labels=labels,
            label_sources=label_sources,
            source_runs=source_runs,
            thresholds=thresholds,
        )
        predictions, _, _ = build_predictions(
            rows,
            labels=labels,
            thresholds=thresholds,
            postprocess=args.prediction_postprocess,
            nms_radius_sec=args.nms_radius_sec,
            merge_gap_sec=args.merge_gap_sec,
        )
        metrics = compute_event_metrics(
            predictions,
            gt_events,
            args.match_tolerance_sec,
            matching_mode=matching_mode,
            allow_many_predictions_per_gt=allow_many_predictions_per_gt,
        )
        write_outputs(
            output_dir=output_dir,
            video_id=video_id,
            rows=rows,
            predictions=predictions,
            gt_events=gt_events,
            metrics=metrics,
        )

    summary = summarize(
        output_dir,
        video_ids,
        labels,
        allow_many_predictions_per_gt=allow_many_predictions_per_gt,
    )
    run_config = {
        "source_runs": {name: str(path) for name, path in source_runs.items()},
        "label_sources": label_sources,
        "labels": labels,
        "thresholds": thresholds,
        "prediction_postprocess": args.prediction_postprocess,
        "nms_radius_sec": args.nms_radius_sec,
        "merge_gap_sec": args.merge_gap_sec,
        "match_tolerance_sec": args.match_tolerance_sec,
        "video_ids": list(video_ids),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2))
    print(json.dumps({"output_dir": str(output_dir), "micro": summary["micro"], "per_class": summary["per_class"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
