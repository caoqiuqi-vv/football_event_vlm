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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_long_video_checkpoint import (  # noqa: E402
    compute_event_metrics,
    merge_predictions,
    point_nms_predictions,
    window_overlap_predictions,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Average aligned football long-video window predictions from multiple eval runs."
    )
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--labels", default="", help="Comma-separated labels. Defaults to prob_* columns shared by all runs.")
    parser.add_argument("--weights", default="", help="Comma-separated non-negative weights; defaults to uniform.")
    parser.add_argument("--blend", choices=("mean", "logit_mean", "geom_mean"), default="mean")
    parser.add_argument("--thresholds", default="first", help="first, 0.5, or shot=0.4,save=0.5,set_piece=0.6")
    parser.add_argument(
        "--prediction-postprocess",
        choices=("window_overlap", "point_nms", "interval_merge"),
        default="window_overlap",
    )
    parser.add_argument("--match-tolerance-sec", type=float, default=2.0)
    parser.add_argument("--nms-radius-sec", type=float, default=5.0)
    parser.add_argument("--merge-gap-sec", type=float, default=2.0)
    parser.add_argument("--gt-merge-gap-sec", type=float, default=0.0)
    parser.add_argument("--allow-many-predictions-per-gt", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def labels_from_rows(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return []
    return sorted(key.removeprefix("prob_") for key in rows[0] if key.startswith("prob_"))


def parse_weights(raw: str, count: int) -> list[float]:
    if not raw.strip():
        return [1.0 / count] * count
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if len(values) != count:
        raise ValueError(f"--weights must have {count} values, got {len(values)}")
    if any(value < 0 for value in values) or sum(values) <= 0:
        raise ValueError("--weights must be non-negative and sum to a positive value")
    total = sum(values)
    return [value / total for value in values]


def parse_thresholds(raw: str, labels: Sequence[str], first_summary: dict[str, Any]) -> dict[str, float]:
    text = raw.strip()
    if text == "first":
        thresholds = first_summary.get("thresholds") or {}
        return {label: float(thresholds.get(label, 0.5)) for label in labels}
    if not text:
        return {label: 0.5 for label in labels}
    if "=" not in text:
        value = float(text)
        return {label: value for label in labels}
    thresholds = {label: 0.5 for label in labels}
    for item in text.split(","):
        if not item.strip():
            continue
        label, value = item.split("=", 1)
        label = label.strip()
        if label not in thresholds:
            raise ValueError(f"Unknown threshold label={label}; labels={labels}")
        thresholds[label] = float(value)
    return thresholds


def row_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("source", "")),
        str(row.get("video_id", "")),
        f"{float(row['start_sec']):.6f}",
        f"{float(row['end_sec']):.6f}",
        f"{float(row.get('center_sec', row['start_sec'])):.6f}",
    )


def probability_blend(values: list[float], weights: list[float], mode: str) -> float:
    eps = 1e-6
    if mode == "mean":
        return sum(weight * value for weight, value in zip(weights, values))
    if mode == "logit_mean":
        total = 0.0
        for weight, value in zip(weights, values):
            clipped = min(max(value, eps), 1.0 - eps)
            total += weight * math.log(clipped / (1.0 - clipped))
        return 1.0 / (1.0 + math.exp(-total))
    if mode == "geom_mean":
        total = 0.0
        for weight, value in zip(weights, values):
            total += weight * math.log(max(value, eps))
        return math.exp(total)
    raise ValueError(f"Unsupported blend={mode}")


def common_video_ids(run_dirs: list[Path]) -> list[str]:
    sets: list[set[str]] = []
    for run_dir in run_dirs:
        ids = {
            path.name
            for path in run_dir.iterdir()
            if path.is_dir() and (path / "window_predictions.csv").is_file()
        }
        sets.append(ids)
    if not sets:
        return []
    return sorted(set.intersection(*sets))


def load_gt_events(video_dir: Path) -> list[dict[str, Any]]:
    rows = read_csv(video_dir / "gt_events.csv")
    result: list[dict[str, Any]] = []
    for row in rows:
        result.append(
            {
                "source": row.get("source", ""),
                "video_id": row.get("video_id", ""),
                "label": row["label"],
                "time_sec": float(row.get("time_sec", row.get("anchor_time", 0.0))),
                "start_sec": float(row.get("start_sec", row.get("time_sec", 0.0))),
                "end_sec": float(row.get("end_sec", row.get("time_sec", 0.0))),
            }
        )
    return result


def blend_video(
    run_dirs: list[Path],
    video_id: str,
    labels: list[str],
    weights: list[float],
    blend: str,
    thresholds: dict[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_run_rows = [read_csv(run_dir / video_id / "window_predictions.csv") for run_dir in run_dirs]
    if len({len(rows) for rows in per_run_rows}) != 1:
        raise ValueError(f"Run rows have different lengths for video_id={video_id}")
    reference_keys = [row_key(row) for row in per_run_rows[0]]
    for run_index, rows in enumerate(per_run_rows[1:], start=1):
        keys = [row_key(row) for row in rows]
        if keys != reference_keys:
            raise ValueError(f"Window rows are not aligned for video_id={video_id} run_index={run_index}")

    blended_rows: list[dict[str, Any]] = []
    for row_index, ref in enumerate(per_run_rows[0]):
        out: dict[str, Any] = dict(ref)
        for label in labels:
            values = [float(rows[row_index][f"prob_{label}"]) for rows in per_run_rows]
            prob = probability_blend(values, weights, blend)
            out[f"prob_{label}"] = float(prob)
            out[f"pred_{label}"] = int(prob >= thresholds[label])
        blended_rows.append(out)
    gt_events = load_gt_events(run_dirs[0] / video_id)
    return blended_rows, gt_events


def predictions_from_rows(
    rows: list[dict[str, Any]],
    labels: list[str],
    thresholds: dict[str, float],
    postprocess: str,
    nms_radius_sec: float,
    merge_gap_sec: float,
) -> list[dict[str, Any]]:
    if postprocess == "window_overlap":
        return window_overlap_predictions(rows, labels, thresholds)
    if postprocess == "point_nms":
        return point_nms_predictions(rows, labels, thresholds, nms_radius_sec)
    if postprocess == "interval_merge":
        return merge_predictions(rows, labels, thresholds, merge_gap_sec)
    raise ValueError(f"Unsupported postprocess={postprocess}")


def main() -> None:
    args = parse_args()
    run_dirs = [path.expanduser().resolve() for path in args.run_dirs]
    if len(run_dirs) < 2:
        raise ValueError("Ensemble requires at least two --run-dirs")
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
    video_ids = common_video_ids(run_dirs)
    if not video_ids:
        raise RuntimeError("No common videos with window_predictions.csv across all run dirs")

    first_rows = read_csv(run_dirs[0] / video_ids[0] / "window_predictions.csv")
    labels = [item.strip() for item in args.labels.split(",") if item.strip()]
    if not labels:
        labels = labels_from_rows(first_rows)
    if not labels:
        raise RuntimeError("Could not infer labels from prob_* columns")
    weights = parse_weights(args.weights, len(run_dirs))
    first_summary = read_json(run_dirs[0] / video_ids[0] / "summary.json")
    thresholds = parse_thresholds(args.thresholds, labels, first_summary)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    all_predictions: list[dict[str, Any]] = []
    all_gt_events: list[dict[str, Any]] = []
    per_video: list[dict[str, Any]] = []

    for video_id in video_ids:
        rows, gt_events = blend_video(run_dirs, video_id, labels, weights, args.blend, thresholds)
        predictions = predictions_from_rows(
            rows,
            labels,
            thresholds,
            args.prediction_postprocess,
            args.nms_radius_sec,
            args.merge_gap_sec,
        )
        metrics = compute_event_metrics(
            predictions,
            gt_events,
            args.match_tolerance_sec,
            matching_mode="window" if args.prediction_postprocess == "window_overlap" else "point",
            allow_many_predictions_per_gt=args.allow_many_predictions_per_gt
            or args.prediction_postprocess == "window_overlap",
        )
        video_dir = output_dir / video_id
        video_dir.mkdir(parents=True, exist_ok=True)
        write_csv(video_dir / "window_predictions.csv", rows, list(rows[0].keys()) if rows else [])
        (video_dir / "predicted_events.json").write_text(json.dumps(predictions, ensure_ascii=False, indent=2) + "\n")
        (video_dir / "summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
        all_predictions.extend(predictions)
        all_gt_events.extend(gt_events)
        per_video.append({"video_id": video_id, "metrics": metrics})

    aggregate = compute_event_metrics(
        all_predictions,
        all_gt_events,
        args.match_tolerance_sec,
        matching_mode="window" if args.prediction_postprocess == "window_overlap" else "point",
        allow_many_predictions_per_gt=args.allow_many_predictions_per_gt
        or args.prediction_postprocess == "window_overlap",
    )
    report = {
        "run_dirs": [str(path) for path in run_dirs],
        "video_ids": video_ids,
        "labels": labels,
        "weights": weights,
        "blend": args.blend,
        "thresholds": thresholds,
        "prediction_postprocess": args.prediction_postprocess,
        "match_tolerance_sec": args.match_tolerance_sec,
        "aggregate": aggregate,
        "per_video": per_video,
    }
    (output_dir / "ensemble_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
