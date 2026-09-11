#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_LABELS = ("shot", "save", "set_piece")
BRANCH_PREFIXES = {"fused": "", "global": "global_", "local": "local_"}


@dataclass(frozen=True)
class VideoLabelScores:
    video_id: str
    label: str
    scores: tuple[float, ...]
    matched_gt_indices: tuple[tuple[int, ...], ...]
    num_gt: int


def safe_div(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def finalize_counts(
    *, tp: int, fp: int, num_gt: int, num_matched_gt: int
) -> dict[str, Any]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(num_matched_gt, num_gt)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(num_gt - num_matched_gt),
        "num_pred": int(tp + fp),
        "num_gt": int(num_gt),
        "num_matched_gt": int(num_matched_gt),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def combine_metrics(items: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(items)
    return finalize_counts(
        tp=sum(int(item["tp"]) for item in rows),
        fp=sum(int(item["fp"]) for item in rows),
        num_gt=sum(int(item["num_gt"]) for item in rows),
        num_matched_gt=sum(int(item["num_matched_gt"]) for item in rows),
    )


def parse_sweep(raw: str) -> list[float]:
    if ":" not in raw:
        values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    else:
        parts = [float(item) for item in raw.split(":")]
        if len(parts) != 3:
            raise ValueError("--threshold-sweep must be start:end:step or comma-separated values")
        start, end, step = parts
        if step <= 0:
            raise ValueError("Threshold sweep step must be positive")
        values = []
        current = start
        while current <= end + step * 1e-6:
            values.append(current)
            current += step
    values = sorted(set(round(value, 10) for value in values))
    if not values or any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("Threshold sweep values must be in [0, 1]")
    return values


def parse_exclusions(values: Sequence[str]) -> set[tuple[str, str]]:
    exclusions: set[tuple[str, str]] = set()
    for value in values:
        if ":" not in value:
            raise ValueError(f"Invalid exclusion '{value}', expected VIDEO_ID:LABEL")
        video_id, label = value.rsplit(":", 1)
        exclusions.add((video_id.strip(), label.strip()))
    return exclusions


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def load_gt_events(path: Path, labels: Sequence[str]) -> dict[str, list[float]]:
    by_label = {label: [] for label in labels}
    if path.suffix == ".csv":
        rows: Sequence[dict[str, Any]] = read_csv(path)
    else:
        payload = json.loads(path.read_text())
        rows = payload.get("data", payload) if isinstance(payload, dict) else payload
    for row in rows:
        label = str(row.get("label", ""))
        if label not in by_label:
            continue
        raw_time = row.get("time_sec", row.get("start_sec", row.get("startTime", 0.0)))
        by_label[label].append(float(raw_time))
    for times in by_label.values():
        times.sort()
    return by_label


def score_column(label: str, branch: str) -> str:
    return f"{BRANCH_PREFIXES[branch]}prob_{label}"


def load_run_scores(
    run_dir: Path,
    *,
    labels: Sequence[str],
    branch: str,
    tolerance_sec: float,
    exclusions: set[tuple[str, str]],
    video_ids: Sequence[str] | None = None,
) -> tuple[list[VideoLabelScores], list[str]]:
    run_config_path = run_dir / "run_config.json"
    run_config = json.loads(run_config_path.read_text()) if run_config_path.exists() else {}
    configured_ids = [str(item) for item in run_config.get("video_ids", [])]
    selected_ids = list(video_ids or configured_ids)
    if not selected_ids:
        selected_ids = sorted(path.name for path in run_dir.iterdir() if path.is_dir())

    result: list[VideoLabelScores] = []
    loaded_ids: list[str] = []
    for video_id in selected_ids:
        video_dir = run_dir / video_id
        window_path = video_dir / "window_predictions.csv"
        gt_path = video_dir / "gt_events.csv"
        if not gt_path.exists():
            gt_path = video_dir / "gt_events.json"
        if not window_path.exists() or not gt_path.exists():
            continue
        windows = read_csv(window_path)
        gt_by_label = load_gt_events(gt_path, labels)
        loaded_ids.append(video_id)
        for label in labels:
            if (video_id, label) in exclusions:
                continue
            column = score_column(label, branch)
            if windows and column not in windows[0]:
                raise KeyError(f"Missing score column '{column}' in {window_path}")
            gt_times = gt_by_label[label]
            scores: list[float] = []
            matches: list[tuple[int, ...]] = []
            for window in windows:
                start = float(window["start_sec"]) - tolerance_sec
                end = float(window["end_sec"]) + tolerance_sec
                scores.append(float(window[column]))
                matches.append(
                    tuple(index for index, time_sec in enumerate(gt_times) if start <= time_sec <= end)
                )
            result.append(
                VideoLabelScores(
                    video_id=video_id,
                    label=label,
                    scores=tuple(scores),
                    matched_gt_indices=tuple(matches),
                    num_gt=len(gt_times),
                )
            )
    if not loaded_ids:
        raise FileNotFoundError(f"No complete per-video dense outputs found under {run_dir}")
    return result, loaded_ids


def evaluate_data(data: VideoLabelScores, threshold: float) -> dict[str, Any]:
    tp = fp = 0
    matched_gt: set[int] = set()
    for score, matches in zip(data.scores, data.matched_gt_indices):
        if score < threshold:
            continue
        if matches:
            tp += 1
            matched_gt.update(matches)
        else:
            fp += 1
    return finalize_counts(
        tp=tp, fp=fp, num_gt=data.num_gt, num_matched_gt=len(matched_gt)
    )


def aggregate_at_threshold(
    data: Sequence[VideoLabelScores], threshold: float
) -> dict[str, Any]:
    return combine_metrics(evaluate_data(item, threshold) for item in data)


def exact_curve(data: Sequence[VideoLabelScores]) -> list[dict[str, Any]]:
    candidates: list[tuple[float, bool, tuple[tuple[str, str, int], ...]]] = []
    num_gt = sum(item.num_gt for item in data)
    for item in data:
        for score, matches in zip(item.scores, item.matched_gt_indices):
            candidates.append(
                (
                    score,
                    bool(matches),
                    tuple((item.video_id, item.label, gt_index) for gt_index in matches),
                )
            )
    candidates.sort(key=lambda item: item[0], reverse=True)
    curve: list[dict[str, Any]] = []
    tp = fp = 0
    matched_gt: set[tuple[str, str, int]] = set()
    position = 0
    while position < len(candidates):
        threshold = candidates[position][0]
        end = position
        while end < len(candidates) and candidates[end][0] == threshold:
            _, is_tp, matches = candidates[end]
            if is_tp:
                tp += 1
                matched_gt.update(matches)
            else:
                fp += 1
            end += 1
        curve.append(
            {
                "threshold": threshold,
                **finalize_counts(
                    tp=tp,
                    fp=fp,
                    num_gt=num_gt,
                    num_matched_gt=len(matched_gt),
                ),
            }
        )
        position = end
    return curve


def pr_auc(curve: Sequence[dict[str, Any]]) -> float:
    if not curve:
        return 0.0
    precision_by_recall: dict[float, float] = {}
    for row in curve:
        recall = float(row["recall"])
        precision_by_recall[recall] = max(
            precision_by_recall.get(recall, 0.0), float(row["precision"])
        )
    points = sorted(precision_by_recall.items())
    precisions = [precision for _, precision in points]
    for index in range(len(precisions) - 2, -1, -1):
        precisions[index] = max(precisions[index], precisions[index + 1])
    area = 0.0
    previous_recall = 0.0
    for (recall, _), precision in zip(points, precisions):
        if recall > previous_recall:
            area += (recall - previous_recall) * precision
            previous_recall = recall
    return area


def best_row(
    rows: Iterable[dict[str, Any]],
    *,
    precision_floor: float | None = None,
    recall_floor: float | None = None,
    objective: str,
) -> dict[str, Any] | None:
    candidates = []
    for row in rows:
        if precision_floor is not None and float(row["precision"]) + 1e-12 < precision_floor:
            continue
        if recall_floor is not None and float(row["recall"]) + 1e-12 < recall_floor:
            continue
        candidates.append(row)
    if not candidates:
        return None
    if objective == "recall":
        key = lambda row: (float(row["recall"]), float(row["precision"]), float(row["f1"]))
    elif objective == "precision":
        key = lambda row: (float(row["precision"]), float(row["recall"]), float(row["f1"]))
    elif objective == "f1":
        key = lambda row: (float(row["f1"]), float(row["recall"]), float(row["precision"]))
    else:
        raise ValueError(f"Unsupported objective={objective}")
    return dict(max(candidates, key=key))


def thresholds_from_summaries(
    run_dir: Path, video_ids: Sequence[str], labels: Sequence[str]
) -> dict[str, float]:
    resolved: dict[str, float] | None = None
    for video_id in video_ids:
        summary_path = run_dir / video_id / "summary.json"
        if not summary_path.exists():
            continue
        payload = json.loads(summary_path.read_text())
        raw = payload.get("thresholds")
        if not isinstance(raw, dict):
            continue
        current = {label: float(raw.get(label, 0.5)) for label in labels}
        if resolved is not None and any(
            abs(resolved[label] - current[label]) > 1e-8 for label in labels
        ):
            raise ValueError("Checkpoint thresholds differ across per-video summaries")
        resolved = current
    return resolved or {label: 0.5 for label in labels}


def group_by_label(
    data: Sequence[VideoLabelScores], labels: Sequence[str]
) -> dict[str, list[VideoLabelScores]]:
    return {label: [item for item in data if item.label == label] for label in labels}


def metric_grid(
    data: Sequence[VideoLabelScores],
    labels: Sequence[str],
    thresholds: Sequence[float],
) -> dict[str, dict[float, dict[str, Any]]]:
    grouped = group_by_label(data, labels)
    return {
        label: {
            threshold: aggregate_at_threshold(grouped[label], threshold)
            for threshold in thresholds
        }
        for label in labels
    }


def search_per_class_thresholds(
    grid: dict[str, dict[float, dict[str, Any]]],
    labels: Sequence[str],
    thresholds: Sequence[float],
    *,
    precision_floor: float,
    recall_target: float,
) -> dict[str, dict[str, Any] | None]:
    best_recall: dict[str, Any] | None = None
    best_precision: dict[str, Any] | None = None
    best_f1_value: dict[str, Any] | None = None
    for values in itertools.product(thresholds, repeat=len(labels)):
        metrics = combine_metrics(
            grid[label][threshold] for label, threshold in zip(labels, values)
        )
        row = {
            "thresholds": {
                label: threshold for label, threshold in zip(labels, values)
            },
            **metrics,
        }
        if metrics["precision"] + 1e-12 >= precision_floor:
            if best_recall is None or (
                metrics["recall"], metrics["precision"], metrics["f1"]
            ) > (
                best_recall["recall"],
                best_recall["precision"],
                best_recall["f1"],
            ):
                best_recall = row
        if metrics["recall"] + 1e-12 >= recall_target:
            if best_precision is None or (
                metrics["precision"], metrics["recall"], metrics["f1"]
            ) > (
                best_precision["precision"],
                best_precision["recall"],
                best_precision["f1"],
            ):
                best_precision = row
        if best_f1_value is None or (
            metrics["f1"], metrics["recall"], metrics["precision"]
        ) > (
            best_f1_value["f1"],
            best_f1_value["recall"],
            best_f1_value["precision"],
        ):
            best_f1_value = row
    return {
        "max_recall_at_precision_floor": best_recall,
        "max_precision_at_recall_target": best_precision,
        "max_f1": best_f1_value,
    }


def evaluate_threshold_map(
    data: Sequence[VideoLabelScores], thresholds: dict[str, float]
) -> dict[str, Any]:
    return combine_metrics(evaluate_data(item, thresholds[item.label]) for item in data)


def selected_strategy_rows(
    data: Sequence[VideoLabelScores],
    strategy: str,
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    return [
        {
            "strategy": strategy,
            "video_id": item.video_id,
            "label": item.label,
            "threshold": thresholds[item.label],
            **evaluate_data(item, thresholds[item.label]),
        }
        for item in data
    ]


def per_video_oracle(
    data: Sequence[VideoLabelScores],
    labels: Sequence[str],
    thresholds: Sequence[float],
    *,
    precision_floor: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    detail_rows: list[dict[str, Any]] = []
    video_summaries: list[dict[str, Any]] = []
    for video_id in sorted({item.video_id for item in data}):
        video_data = [item for item in data if item.video_id == video_id]
        video_labels = [
            label for label in labels if any(item.label == label for item in video_data)
        ]
        grid = metric_grid(video_data, video_labels, thresholds)
        result = search_per_class_thresholds(
            grid,
            video_labels,
            thresholds,
            precision_floor=precision_floor,
            recall_target=0.0,
        )["max_recall_at_precision_floor"]
        if result is None:
            result = search_per_class_thresholds(
                grid,
                video_labels,
                thresholds,
                precision_floor=0.0,
                recall_target=0.0,
            )["max_f1"]
        assert result is not None
        threshold_map = {
            label: float(result["thresholds"][label]) for label in video_labels
        }
        video_summaries.append({"video_id": video_id, **result})
        for item in video_data:
            detail_rows.append(
                {
                    "strategy": "per_video_oracle",
                    "video_id": video_id,
                    "label": item.label,
                    "threshold": threshold_map[item.label],
                    **evaluate_data(item, threshold_map[item.label]),
                }
            )
    return video_summaries, detail_rows, combine_metrics(detail_rows)


def write_csv(
    path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the score-ranking ceiling of saved dense football window predictions "
            "under the window_overlap matching protocol."
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--video-ids", default="")
    parser.add_argument("--labels", default="")
    parser.add_argument("--branch", choices=sorted(BRANCH_PREFIXES), default="fused")
    parser.add_argument("--match-tolerance-sec", type=float, default=5.0)
    parser.add_argument("--precision-floor", type=float, default=0.60)
    parser.add_argument("--recall-target", type=float, default=0.8245)
    parser.add_argument("--threshold-sweep", default="0.05:0.95:0.01")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Exclude one video-label pair as VIDEO_ID:LABEL. May be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.match_tolerance_sec < 0:
        raise ValueError("--match-tolerance-sec must be non-negative")
    if not 0.0 <= args.precision_floor <= 1.0:
        raise ValueError("--precision-floor must be in [0, 1]")
    if not 0.0 <= args.recall_target <= 1.0:
        raise ValueError("--recall-target must be in [0, 1]")

    run_dir = Path(args.run_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else run_dir / f"dense_pr_ceiling_{args.branch}_tol{args.match_tolerance_sec:g}"
    )
    run_config_path = run_dir / "run_config.json"
    run_config = json.loads(run_config_path.read_text()) if run_config_path.exists() else {}
    labels = [item.strip() for item in args.labels.split(",") if item.strip()]
    labels = labels or [str(item) for item in run_config.get("labels", DEFAULT_LABELS)]
    video_ids = [item.strip() for item in args.video_ids.split(",") if item.strip()] or None
    exclusions = parse_exclusions(args.exclude)
    data, loaded_ids = load_run_scores(
        run_dir,
        labels=labels,
        branch=args.branch,
        tolerance_sec=args.match_tolerance_sec,
        exclusions=exclusions,
        video_ids=video_ids,
    )
    active_labels = [
        label for label in labels if any(item.label == label for item in data)
    ]
    checkpoint_thresholds = thresholds_from_summaries(
        run_dir, loaded_ids, active_labels
    )
    sweep_thresholds = sorted(
        set(parse_sweep(args.threshold_sweep))
        | {round(float(value), 10) for value in checkpoint_thresholds.values()}
    )

    grouped = group_by_label(data, active_labels)
    class_curves = {label: exact_curve(grouped[label]) for label in active_labels}
    class_summary = {
        label: {
            "pr_auc": pr_auc(curve),
            "max_recall_at_precision_floor": best_row(
                curve, precision_floor=args.precision_floor, objective="recall"
            ),
            "max_precision_at_recall_target": best_row(
                curve, recall_floor=args.recall_target, objective="precision"
            ),
            "max_f1": best_row(curve, objective="f1"),
        }
        for label, curve in class_curves.items()
    }
    shared_curve = exact_curve(data)
    shared_summary = {
        "pr_auc": pr_auc(shared_curve),
        "max_recall_at_precision_floor": best_row(
            shared_curve, precision_floor=args.precision_floor, objective="recall"
        ),
        "max_precision_at_recall_target": best_row(
            shared_curve, recall_floor=args.recall_target, objective="precision"
        ),
        "max_f1": best_row(shared_curve, objective="f1"),
    }

    grid = metric_grid(data, active_labels, sweep_thresholds)
    per_class_search = search_per_class_thresholds(
        grid,
        active_labels,
        sweep_thresholds,
        precision_floor=args.precision_floor,
        recall_target=args.recall_target,
    )
    checkpoint_metrics = evaluate_threshold_map(data, checkpoint_thresholds)
    oracle_videos, oracle_rows, oracle_micro = per_video_oracle(
        data,
        active_labels,
        sweep_thresholds,
        precision_floor=args.precision_floor,
    )

    strategy_rows = selected_strategy_rows(
        data, "checkpoint", checkpoint_thresholds
    )
    shared_selected = shared_summary["max_recall_at_precision_floor"]
    if shared_selected is not None:
        strategy_rows.extend(
            selected_strategy_rows(
                data,
                "shared_r_at_p60",
                {
                    label: float(shared_selected["threshold"])
                    for label in active_labels
                },
            )
        )
    per_class_selected = per_class_search["max_recall_at_precision_floor"]
    if per_class_selected is not None:
        strategy_rows.extend(
            selected_strategy_rows(
                data,
                "per_class_r_at_p60",
                {
                    label: float(per_class_selected["thresholds"][label])
                    for label in active_labels
                },
            )
        )
    strategy_rows.extend(oracle_rows)

    per_class_feasible = bool(
        per_class_selected is not None
        and float(per_class_selected["recall"]) + 1e-12 >= args.recall_target
    )
    oracle_feasible = bool(
        float(oracle_micro["precision"]) + 1e-12 >= args.precision_floor
        and float(oracle_micro["recall"]) + 1e-12 >= args.recall_target
    )
    if per_class_feasible:
        diagnosis = "threshold_or_calibration_limited"
    elif oracle_feasible:
        diagnosis = "cross_video_calibration_limited"
    else:
        diagnosis = "current_score_ranking_limited"

    summary = {
        "protocol": {
            "prediction_postprocess": "window_overlap",
            "matching_mode": "window",
            "allow_many_predictions_per_gt": True,
            "match_tolerance_sec": args.match_tolerance_sec,
            "branch": args.branch,
            "excluded_video_label_pairs": [
                list(item) for item in sorted(exclusions)
            ],
        },
        "run_dir": str(run_dir),
        "video_ids": loaded_ids,
        "labels": active_labels,
        "targets": {
            "precision_floor": args.precision_floor,
            "recall_target": args.recall_target,
        },
        "checkpoint": {
            "thresholds": checkpoint_thresholds,
            "metrics": checkpoint_metrics,
        },
        "per_class": class_summary,
        "shared_threshold": shared_summary,
        "per_class_threshold_search": per_class_search,
        "per_video_oracle": {"micro": oracle_micro, "videos": oracle_videos},
        "target_feasibility": {
            "per_class_thresholds": per_class_feasible,
            "per_video_oracle": oracle_feasible,
            "diagnosis": diagnosis,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "diagnostic_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    if exclusions:
        (output_dir / "summary_metrics_exclude_unlabeled_setpiece.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2)
        )
    curve_fields = [
        "label",
        "threshold",
        "tp",
        "fp",
        "fn",
        "num_pred",
        "num_gt",
        "num_matched_gt",
        "precision",
        "recall",
        "f1",
    ]
    write_csv(
        output_dir / "pr_curve_per_class.csv",
        [
            {"label": label, **row}
            for label in active_labels
            for row in class_curves[label]
        ],
        curve_fields,
    )
    write_csv(
        output_dir / "pr_curve_micro_shared_threshold.csv",
        [{"label": "micro", **row} for row in shared_curve],
        curve_fields,
    )
    write_csv(
        output_dir / "per_video_per_class.csv",
        strategy_rows,
        [
            "strategy",
            "video_id",
            "label",
            "threshold",
            "tp",
            "fp",
            "fn",
            "num_pred",
            "num_gt",
            "num_matched_gt",
            "precision",
            "recall",
            "f1",
        ],
    )
    write_csv(
        output_dir / "per_video_oracle_thresholds.csv",
        [
            {
                "video_id": item["video_id"],
                **{
                    f"threshold_{label}": item["thresholds"].get(label, "")
                    for label in active_labels
                },
                **{
                    key: item[key]
                    for key in ("precision", "recall", "f1", "tp", "fp", "fn")
                },
            }
            for item in oracle_videos
        ],
        [
            "video_id",
            *(f"threshold_{label}" for label in active_labels),
            "precision",
            "recall",
            "f1",
            "tp",
            "fp",
            "fn",
        ],
    )

    print(
        f"checkpoint P={checkpoint_metrics['precision']:.4f} "
        f"R={checkpoint_metrics['recall']:.4f}"
    )
    if per_class_selected is None:
        print(
            f"per-class thresholds: no point reaches "
            f"P>={args.precision_floor:.2f}"
        )
    else:
        print(
            f"per-class R@P{args.precision_floor:.2f}: "
            f"P={per_class_selected['precision']:.4f} "
            f"R={per_class_selected['recall']:.4f} "
            f"thresholds={per_class_selected['thresholds']}"
        )
    precision_at_recall = per_class_search["max_precision_at_recall_target"]
    if precision_at_recall is None:
        print(
            f"per-class P@R{args.recall_target:.4f}: "
            "target recall is unreachable"
        )
    else:
        print(
            f"per-class P@R{args.recall_target:.4f}: "
            f"P={precision_at_recall['precision']:.4f} "
            f"R={precision_at_recall['recall']:.4f} "
            f"thresholds={precision_at_recall['thresholds']}"
        )
    print(
        f"per-video oracle P={oracle_micro['precision']:.4f} "
        f"R={oracle_micro['recall']:.4f}"
    )
    print(f"diagnosis={diagnosis}")
    print(f"wrote {output_dir / 'diagnostic_summary.json'}")


if __name__ == "__main__":
    main()
