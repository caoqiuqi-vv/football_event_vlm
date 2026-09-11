#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from eval_long_video_checkpoint import pred_matches_gt  # noqa: E402
from recompute_football_eval_protocols import load_gt_events, normalize_score_columns, read_csv  # noqa: E402
from retune_all_dense_window_runs import maximum_matching  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select window-overlap thresholds on a calibration long-video run "
            "and optionally apply them to an untouched test run."
        )
    )
    parser.add_argument("--cal-run-dir", type=Path, required=True)
    parser.add_argument("--cal-video-id-file", type=Path, required=True)
    parser.add_argument("--test-run-dir", type=Path, default=None)
    parser.add_argument("--test-video-id-file", type=Path, default=None)
    parser.add_argument("--labels", default="shot,save,set_piece")
    parser.add_argument("--budgets", default="20,30,40,50,60,70")
    parser.add_argument("--recall-targets", default="0.85,0.88,0.90,0.93,0.95")
    parser.add_argument(
        "--recall-constraint", choices=("micro", "per_class"), default="micro",
        help="Require target recall on aggregate micro recall or independently for every class.",
    )
    parser.add_argument("--grid-size", type=int, default=31)
    parser.add_argument("--match-tolerance-sec", type=float, default=5.0)
    parser.add_argument(
        "--score-prefix",
        default="prob",
        help=(
            "Window score column prefix to threshold. For example: prob uses prob_shot, "
            "clip_prob uses clip_prob_shot, response_prob uses response_prob_shot, "
            "frame_max_prob uses frame_max_prob_shot."
        ),
    )
    parser.add_argument(
        "--review-mode",
        default="window",
        choices=("window", "window_tol5", "cap10_peak", "cap15_peak", "cap20_peak"),
        help="How to convert selected windows into human viewing intervals.",
    )
    parser.add_argument(
        "--manual-overhead-sec",
        type=float,
        default=0.0,
        help="Optional per-window operation overhead added to human minutes only.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_video_ids(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def metric(tp: int, fp: int, num_gt: int, **extra: Any) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / num_gt if num_gt else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(num_gt - tp),
        "num_gt": int(num_gt),
        "num_pred": int(tp + fp),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        **extra,
    }


def aggregate(per_class: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return metric(
        sum(int(v["tp"]) for v in per_class.values()),
        sum(int(v["fp"]) for v in per_class.values()),
        sum(int(v["num_gt"]) for v in per_class.values()),
    )


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    intervals = sorted((max(0.0, min(a, b)), max(0.0, max(a, b))) for a, b in intervals)
    intervals = [(a, b) for a, b in intervals if b > a]
    if not intervals:
        return []
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def load_data(run_dir: Path, video_ids: Sequence[str], labels: Sequence[str], score_prefix: str) -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = []
    for video_id in video_ids:
        video_dir = run_dir / video_id
        window_path = video_dir / "window_predictions.csv"
        gt_path = video_dir / "gt_events.csv"
        if not gt_path.exists():
            gt_path = video_dir / "gt_events.json"
        if not window_path.exists() or not gt_path.exists():
            raise FileNotFoundError(f"Missing dense eval outputs for {video_id}: {video_dir}")
        summary = json.loads((video_dir / "summary.json").read_text(encoding="utf-8"))
        rows = normalize_score_columns(read_csv(window_path), labels, score_prefix)
        gts = load_gt_events(gt_path, labels)
        data.append(
            {
                "video_id": video_id,
                "duration_sec": float(summary.get("duration_sec", 0.0) or 0.0),
                "thresholds": {
                    label: float(summary.get("thresholds", {}).get(label, 0.5))
                    for label in labels
                },
                "rows": rows,
                "gts": gts,
            }
        )
    return data


def choose_threshold_values(scores: list[float], grid_size: int, extra: Sequence[float]) -> list[float]:
    values = sorted({float(s) for s in scores if 0.0 <= float(s) <= 1.0})
    if not values:
        base = [1.0000001, 0.0]
    elif len(values) <= grid_size:
        base = values
    else:
        desc = list(reversed(values))
        idx = sorted({round(i * (len(desc) - 1) / max(grid_size - 1, 1)) for i in range(grid_size)})
        base = [desc[i] for i in idx]
    return sorted(set(base) | {0.0, 1.0000001} | {float(x) for x in extra}, reverse=True)


def make_predictions(rows: Sequence[dict[str, Any]], labels: Sequence[str], thresholds: dict[str, float]) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for row in rows:
        start = float(row["start_sec"])
        end = float(row["end_sec"])
        anchor = 0.5 * (start + end)
        for label in labels:
            score = float(row.get(f"prob_{label}", 0.0) or 0.0)
            if score < thresholds[label]:
                continue
            predictions.append(
                {
                    "label": label,
                    "score": score,
                    "time_sec": anchor,
                    "start_sec": start,
                    "end_sec": end,
                    "support_start_sec": start,
                    "support_end_sec": end,
                }
            )
    return predictions


def review_interval(pred: dict[str, Any], duration: float, mode: str, tolerance_sec: float) -> tuple[float, float]:
    start = float(pred["start_sec"])
    end = float(pred["end_sec"])
    anchor = float(pred["time_sec"])
    if mode == "window":
        left, right = start, end
    elif mode == "window_tol5":
        left, right = start - tolerance_sec, end + tolerance_sec
    elif mode == "cap10_peak":
        left, right = anchor - 5.0, anchor + 5.0
    elif mode == "cap15_peak":
        left, right = anchor - 7.5, anchor + 7.5
    elif mode == "cap20_peak":
        left, right = anchor - 10.0, anchor + 10.0
    else:
        raise ValueError(f"Unsupported review mode: {mode}")
    return max(0.0, left), min(duration, right)


def evaluate(
    data: Sequence[dict[str, Any]],
    labels: Sequence[str],
    thresholds: dict[str, float],
    tolerance_sec: float,
    review_mode: str,
    manual_overhead_sec: float,
) -> dict[str, Any]:
    totals = {label: {"tp": 0, "fp": 0, "gt": 0} for label in labels}
    total_duration = sum(float(item["duration_sec"]) for item in data)
    total_review_sec = 0.0
    total_raw_window_sec = 0.0
    total_candidates = 0
    per_video: list[dict[str, Any]] = []
    for item in data:
        predictions = make_predictions(item["rows"], labels, thresholds)
        total_candidates += len(predictions)
        intervals = [
            review_interval(pred, float(item["duration_sec"]), review_mode, tolerance_sec)
            for pred in predictions
        ]
        merged = merge_intervals(intervals)
        union_sec = sum(end - start for start, end in merged)
        raw_sec = sum(max(end - start, 0.0) for start, end in intervals)
        total_review_sec += union_sec
        total_raw_window_sec += raw_sec
        per_video.append(
            {
                "video_id": item["video_id"],
                "duration_sec": item["duration_sec"],
                "num_candidates": len(predictions),
                "review_sec": union_sec,
                "participation_pct": 100.0 * union_sec / item["duration_sec"] if item["duration_sec"] else 0.0,
            }
        )
        for label in labels:
            label_preds = [pred for pred in predictions if pred["label"] == label]
            label_gts = [gt for gt in item["gts"] if gt["label"] == label]
            edges = [
                [
                    idx
                    for idx, gt in enumerate(label_gts)
                    if pred_matches_gt(pred, gt, tolerance_sec, matching_mode="window")
                ]
                for pred in label_preds
            ]
            tp = maximum_matching(edges)
            totals[label]["tp"] += tp
            totals[label]["fp"] += len(label_preds) - tp
            totals[label]["gt"] += len(label_gts)
    per_class = {
        label: metric(values["tp"], values["fp"], values["gt"])
        for label, values in totals.items()
    }
    micro = aggregate(per_class)
    return {
        "thresholds": dict(thresholds),
        "threshold_string": ",".join(f"{label}={thresholds[label]:.9g}" for label in labels),
        "per_class": per_class,
        "micro": micro,
        "review_mode": review_mode,
        "candidate_view_minutes": total_review_sec / 60.0,
        "raw_candidate_minutes_unmerged": total_raw_window_sec / 60.0,
        "participation_pct": 100.0 * total_review_sec / total_duration if total_duration else 0.0,
        "human_minutes_with_overhead": (total_review_sec + manual_overhead_sec * total_candidates) / 60.0,
        "manual_overhead_sec": manual_overhead_sec,
        "num_candidates": total_candidates,
        "candidates_per_hour": total_candidates / (total_duration / 3600.0) if total_duration else 0.0,
        "total_video_minutes": total_duration / 60.0,
        "per_video": per_video,
    }


def better_for_budget(old: dict[str, Any] | None, new: dict[str, Any]) -> bool:
    if old is None:
        return True
    return (
        float(new["micro"]["recall"]),
        float(new["micro"]["precision"]),
        -float(new["participation_pct"]),
    ) > (
        float(old["micro"]["recall"]),
        float(old["micro"]["precision"]),
        -float(old["participation_pct"]),
    )


def better_for_recall(old: dict[str, Any] | None, new: dict[str, Any]) -> bool:
    if old is None:
        return True
    return (
        -float(new["participation_pct"]),
        float(new["micro"]["precision"]),
        float(new["micro"]["recall"]),
    ) > (
        -float(old["participation_pct"]),
        float(old["micro"]["precision"]),
        float(old["micro"]["recall"]),
    )


def meets_recall_target(item: dict[str, Any], target: float, constraint: str) -> bool:
    if constraint == "micro":
        return float(item["micro"]["recall"]) + 1e-12 >= target
    if constraint == "per_class":
        return all(
            float(metrics["recall"]) + 1e-12 >= target
            for metrics in item["per_class"].values()
        )
    raise ValueError(f"Unsupported recall constraint: {constraint}")


def rows_for_csv(section: str, items: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, item in items.items():
        rows.append(
            {
                "section": section,
                "key": key,
                **{f"thr_{label}": value for label, value in item["thresholds"].items()},
                "participation_pct": item["participation_pct"],
                "candidate_view_minutes": item["candidate_view_minutes"],
                "human_minutes_with_overhead": item["human_minutes_with_overhead"],
                "recall": item["micro"]["recall"],
                "precision": item["micro"]["precision"],
                "f1": item["micro"]["f1"],
                "num_candidates": item["num_candidates"],
                "candidates_per_hour": item["candidates_per_hour"],
                "threshold_string": item["threshold_string"],
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    labels = [item.strip() for item in args.labels.split(",") if item.strip()]
    budgets = [float(item) for item in args.budgets.split(",") if item.strip()]
    recall_targets = [float(item) for item in args.recall_targets.split(",") if item.strip()]
    cal_ids = read_video_ids(args.cal_video_id_file)
    cal_data = load_data(args.cal_run_dir, cal_ids, labels, args.score_prefix)
    checkpoint_thresholds = {
        label: float(cal_data[0]["thresholds"].get(label, 0.5))
        for label in labels
    }
    score_values = {label: [] for label in labels}
    for item in cal_data:
        for row in item["rows"]:
            for label in labels:
                score_values[label].append(float(row.get(f"prob_{label}", 0.0) or 0.0))
    grids = {
        label: choose_threshold_values(score_values[label], args.grid_size, [checkpoint_thresholds[label]])
        for label in labels
    }

    checkpoint_cal = evaluate(
        cal_data, labels, checkpoint_thresholds, args.match_tolerance_sec,
        args.review_mode, args.manual_overhead_sec
    )
    best_by_budget: dict[str, dict[str, Any] | None] = {f"{budget:g}": None for budget in budgets}
    min_cost_by_recall: dict[str, dict[str, Any] | None] = {f"{target:g}": None for target in recall_targets}
    combos = 0
    for values in itertools.product(*(grids[label] for label in labels)):
        thresholds = dict(zip(labels, values))
        item = evaluate(
            cal_data, labels, thresholds, args.match_tolerance_sec,
            args.review_mode, args.manual_overhead_sec
        )
        combos += 1
        for budget in budgets:
            key = f"{budget:g}"
            if float(item["participation_pct"]) <= budget + 1e-9 and better_for_budget(best_by_budget[key], item):
                best_by_budget[key] = item
        for target in recall_targets:
            key = f"{target:g}"
            if meets_recall_target(item, target, args.recall_constraint) and better_for_recall(min_cost_by_recall[key], item):
                min_cost_by_recall[key] = item

    selected_budget = {key: item for key, item in best_by_budget.items() if item is not None}
    selected_recall = {key: item for key, item in min_cost_by_recall.items() if item is not None}
    test_payload: dict[str, Any] = {}
    if args.test_run_dir is not None and args.test_video_id_file is not None:
        test_data = load_data(args.test_run_dir, read_video_ids(args.test_video_id_file), labels, args.score_prefix)
        applied: dict[str, dict[str, Any]] = {
            "checkpoint": evaluate(
                test_data, labels, checkpoint_thresholds, args.match_tolerance_sec,
                args.review_mode, args.manual_overhead_sec
            )
        }
        for key, item in selected_recall.items():
            applied[f"cal_min_cost_at_recall_{key}"] = evaluate(
                test_data, labels, item["thresholds"], args.match_tolerance_sec,
                args.review_mode, args.manual_overhead_sec
            )
        for key, item in selected_budget.items():
            applied[f"cal_max_recall_under_budget_{key}pct"] = evaluate(
                test_data, labels, item["thresholds"], args.match_tolerance_sec,
                args.review_mode, args.manual_overhead_sec
            )
        test_payload = {
            "test_run_dir": str(args.test_run_dir),
            "test_video_id_file": str(args.test_video_id_file),
            "applied": applied,
        }

    payload = {
        "protocol": "window_overlap_val_threshold_external_fixed_v1",
        "cal_run_dir": str(args.cal_run_dir),
        "cal_video_id_file": str(args.cal_video_id_file),
        "labels": labels,
        "review_mode": args.review_mode,
        "match_tolerance_sec": args.match_tolerance_sec,
        "manual_overhead_sec": args.manual_overhead_sec,
        "score_prefix": args.score_prefix,
        "budgets_pct": budgets,
        "recall_targets": recall_targets,
        "recall_constraint": args.recall_constraint,
        "grid_size": args.grid_size,
        "num_grid_combinations": combos,
        "calibration": {
            "checkpoint_thresholds": checkpoint_cal,
            "best_by_budget": selected_budget,
            "min_cost_by_recall": selected_recall,
        },
        "test": test_payload,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    csv_rows = [{"section": "cal_checkpoint", "key": "checkpoint", **{f"thr_{label}": value for label, value in checkpoint_cal["thresholds"].items()}, "participation_pct": checkpoint_cal["participation_pct"], "candidate_view_minutes": checkpoint_cal["candidate_view_minutes"], "human_minutes_with_overhead": checkpoint_cal["human_minutes_with_overhead"], "recall": checkpoint_cal["micro"]["recall"], "precision": checkpoint_cal["micro"]["precision"], "f1": checkpoint_cal["micro"]["f1"], "num_candidates": checkpoint_cal["num_candidates"], "candidates_per_hour": checkpoint_cal["candidates_per_hour"], "threshold_string": checkpoint_cal["threshold_string"]}]
    csv_rows.extend(rows_for_csv("cal_min_cost_by_recall", selected_recall))
    csv_rows.extend(rows_for_csv("cal_best_by_budget", selected_budget))
    if test_payload:
        csv_rows.extend(rows_for_csv("test_applied", test_payload["applied"]))
    csv_path = args.output.with_suffix(".csv")
    fields = list(csv_rows[0].keys())
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps({"output": str(args.output), "csv": str(csv_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
