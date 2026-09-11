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

from eval_long_video_checkpoint import point_nms_predictions, pred_matches_gt  # noqa: E402
from recompute_football_eval_protocols import load_gt_events, normalize_score_columns, read_csv  # noqa: E402
from retune_all_dense_window_runs import maximum_matching  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select PointNMS thresholds by max recall under a participation budget.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--video-id-file", type=Path, required=True)
    parser.add_argument("--labels", default="shot,save,set_piece")
    parser.add_argument("--budgets", default="5,10,15,20,25,30", help="Participation budgets in percent of video duration.")
    parser.add_argument("--recall-targets", default="0.80,0.85,0.90,0.95", help="Micro recall targets for min-cost operating points.")
    parser.add_argument("--grid-size", type=int, default=31, help="Approx threshold candidates per class from score quantiles.")
    parser.add_argument("--match-tolerance-sec", type=float, default=5.0)
    parser.add_argument("--nms-radius-sec", type=float, default=5.0)
    parser.add_argument("--exclude", action="append", default=["2027572406738604033:set_piece"])
    parser.add_argument("--manual-overhead-sec", type=float, default=0.0, help="Optional per-candidate operation overhead added to human-time minutes, not to participation%.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_video_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.strip().startswith("#")]


def parse_exclusions(values: Sequence[str]) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for value in values:
        if not value:
            continue
        video, label = value.rsplit(":", 1)
        out.add((video.strip(), label.strip()))
    return out


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted((min(a, b), max(a, b)) for a, b in intervals)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def metric(tp: int, fp: int, num_gt: int, **extra: Any) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / num_gt if num_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": num_gt - tp, "num_gt": num_gt, "num_pred": tp + fp, "precision": precision, "recall": recall, "f1": f1, **extra}


def aggregate(per_class: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return metric(
        sum(int(v["tp"]) for v in per_class.values()),
        sum(int(v["fp"]) for v in per_class.values()),
        sum(int(v["num_gt"]) for v in per_class.values()),
    )


def choose_threshold_values(scores: list[float], grid_size: int, extra: Sequence[float]) -> list[float]:
    values = sorted({float(s) for s in scores if 0.0 <= float(s) <= 1.0})
    if not values:
        base = [1.0000001, 0.0]
    elif len(values) <= grid_size:
        base = values
    else:
        # Quantiles over descending score distribution: keep both high-score and low-score tail coverage.
        desc = list(reversed(values))
        indices = sorted({round(i * (len(desc) - 1) / max(grid_size - 1, 1)) for i in range(grid_size)})
        base = [desc[i] for i in indices]
    all_values = set(base) | {0.0, 1.0000001} | {float(x) for x in extra if 0.0 <= float(x) <= 1.0000001}
    return sorted(all_values, reverse=True)


def load_data(run_dir: Path, video_ids: list[str], labels: list[str], exclusions: set[tuple[str, str]]) -> list[dict[str, Any]]:
    data = []
    for video_id in video_ids:
        video_dir = run_dir / video_id
        if not (video_dir / "window_predictions.csv").exists():
            raise FileNotFoundError(f"Missing window_predictions.csv for {video_id}: {video_dir}")
        summary = json.loads((video_dir / "summary.json").read_text())
        rows = normalize_score_columns(read_csv(video_dir / "window_predictions.csv"), labels, "prob")
        gts_all = load_gt_events(video_dir / "gt_events.csv", labels)
        gts = [gt for gt in gts_all if (video_id, gt.get("label", "")) not in exclusions]
        data.append({
            "video_id": video_id,
            "duration_sec": float(summary.get("duration_sec", 0.0) or 0.0),
            "thresholds": {label: float(summary.get("thresholds", {}).get(label, 0.5)) for label in labels},
            "rows": rows,
            "gts": gts,
        })
    return data


def evaluate(data: list[dict[str, Any]], labels: list[str], thresholds: dict[str, float], tolerance_sec: float, radius_sec: float, overhead_sec: float) -> dict[str, Any]:
    totals_by_label = {label: {"tp": 0, "fp": 0, "gt": 0, "pred": 0} for label in labels}
    total_duration = sum(float(d["duration_sec"]) for d in data)
    total_union_sec = 0.0
    total_raw_candidate_sec = 0.0
    total_candidates = 0
    duplicate_support_windows = 0
    per_video = []
    for d in data:
        preds = point_nms_predictions(d["rows"], labels, thresholds, radius_sec)
        total_candidates += len(preds)
        duplicate_support_windows += sum(max(int(p.get("num_windows", 1)) - 1, 0) for p in preds)
        intervals = [(float(p.get("support_start_sec", p.get("start_sec", 0.0))), float(p.get("support_end_sec", p.get("end_sec", 0.0)))) for p in preds]
        merged = merge_intervals(intervals)
        union_sec = sum(end - start for start, end in merged)
        raw_sec = sum(max(end - start, 0.0) for start, end in intervals)
        total_union_sec += union_sec
        total_raw_candidate_sec += raw_sec
        per_video.append({"video_id": d["video_id"], "duration_sec": d["duration_sec"], "num_candidates": len(preds), "participation_pct": 100.0 * union_sec / d["duration_sec"] if d["duration_sec"] else 0.0})
        for label in labels:
            label_preds = [p for p in preds if p["label"] == label]
            label_gts = [g for g in d["gts"] if g["label"] == label]
            edges = [[idx for idx, gt in enumerate(label_gts) if pred_matches_gt(pred, gt, tolerance_sec, matching_mode="point")] for pred in label_preds]
            tp = maximum_matching(edges)
            totals_by_label[label]["tp"] += tp
            totals_by_label[label]["fp"] += len(label_preds) - tp
            totals_by_label[label]["gt"] += len(label_gts)
            totals_by_label[label]["pred"] += len(label_preds)
    per_class = {label: metric(v["tp"], v["fp"], v["gt"], num_pred=v["pred"]) for label, v in totals_by_label.items()}
    micro = aggregate(per_class)
    participation_pct = 100.0 * total_union_sec / total_duration if total_duration else 0.0
    human_minutes = (total_union_sec + overhead_sec * total_candidates) / 60.0
    return {
        "thresholds": dict(thresholds),
        "threshold_string": ",".join(f"{label}={thresholds[label]:.9g}" for label in labels),
        "per_class": per_class,
        "micro": micro,
        "participation_pct": participation_pct,
        "candidate_view_minutes": total_union_sec / 60.0,
        "raw_candidate_minutes_unmerged": total_raw_candidate_sec / 60.0,
        "human_minutes_with_overhead": human_minutes,
        "manual_overhead_sec": overhead_sec,
        "num_candidates": total_candidates,
        "candidates_per_hour": total_candidates / (total_duration / 3600.0) if total_duration else 0.0,
        "duplicate_support_windows": duplicate_support_windows,
        "duplicate_ratio": duplicate_support_windows / max(total_candidates + duplicate_support_windows, 1),
        "total_video_minutes": total_duration / 60.0,
        "per_video": per_video,
    }


def better_for_budget(a: dict[str, Any] | None, b: dict[str, Any]) -> bool:
    if a is None:
        return True
    return (
        float(b["micro"]["recall"]),
        float(b["micro"]["precision"]),
        -float(b["participation_pct"]),
        float(b["micro"]["f1"]),
    ) > (
        float(a["micro"]["recall"]),
        float(a["micro"]["precision"]),
        -float(a["participation_pct"]),
        float(a["micro"]["f1"]),
    )


def better_for_recall_target(a: dict[str, Any] | None, b: dict[str, Any]) -> bool:
    if a is None:
        return True
    return (
        -float(b["participation_pct"]),
        -float(b["human_minutes_with_overhead"]),
        float(b["micro"]["precision"]),
        float(b["micro"]["recall"]),
    ) > (
        -float(a["participation_pct"]),
        -float(a["human_minutes_with_overhead"]),
        float(a["micro"]["precision"]),
        float(a["micro"]["recall"]),
    )


def main() -> None:
    args = parse_args()
    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    budgets = [float(x) for x in args.budgets.split(",") if x.strip()]
    recall_targets = [float(x) for x in args.recall_targets.split(",") if x.strip()]
    exclusions = parse_exclusions(args.exclude)
    video_ids = read_video_ids(args.video_id_file)
    data = load_data(args.run_dir, video_ids, labels, exclusions)
    checkpoint_thresholds = {label: data[0]["thresholds"].get(label, 0.5) for label in labels}
    score_values = {label: [] for label in labels}
    for d in data:
        for row in d["rows"]:
            for label in labels:
                score_values[label].append(float(row.get(f"prob_{label}", 0.0) or 0.0))
    grids = {label: choose_threshold_values(score_values[label], args.grid_size, [checkpoint_thresholds[label]]) for label in labels}

    checkpoint = evaluate(data, labels, checkpoint_thresholds, args.match_tolerance_sec, args.nms_radius_sec, args.manual_overhead_sec)
    best_by_budget: dict[str, dict[str, Any] | None] = {f"{b:g}": None for b in budgets}
    min_cost_by_recall: dict[str, dict[str, Any] | None] = {f"{r:g}": None for r in recall_targets}
    combos = 0
    for values in itertools.product(*(grids[label] for label in labels)):
        thresholds = dict(zip(labels, values))
        item = evaluate(data, labels, thresholds, args.match_tolerance_sec, args.nms_radius_sec, args.manual_overhead_sec)
        combos += 1
        for budget in budgets:
            if float(item["participation_pct"]) <= budget + 1e-9 and better_for_budget(best_by_budget[f"{budget:g}"], item):
                best_by_budget[f"{budget:g}"] = item
        for target in recall_targets:
            if float(item["micro"]["recall"]) + 1e-12 >= target and better_for_recall_target(min_cost_by_recall[f"{target:g}"], item):
                min_cost_by_recall[f"{target:g}"] = item
    selected = {budget: item for budget, item in best_by_budget.items() if item is not None}
    selected_recall = {target: item for target, item in min_cost_by_recall.items() if item is not None}
    payload = {
        "protocol": "point_nms_recall_at_participation_budget_v1",
        "run_dir": str(args.run_dir),
        "video_id_file": str(args.video_id_file),
        "labels": labels,
        "budgets_pct": budgets,
        "recall_targets": recall_targets,
        "grid_size": args.grid_size,
        "num_grid_combinations": combos,
        "match_tolerance_sec": args.match_tolerance_sec,
        "nms_radius_sec": args.nms_radius_sec,
        "exclusions": sorted(f"{a}:{b}" for a,b in exclusions),
        "participation_definition": "merged union of PointNMS candidate support intervals divided by total video duration; manual overhead reported separately",
        "checkpoint_thresholds": checkpoint,
        "best_by_budget": selected,
        "min_cost_by_recall": selected_recall,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    rows = []
    rows.append({"mode": "checkpoint", "budget_pct": "", "recall_target": "", **{f"thr_{k}": v for k, v in checkpoint["thresholds"].items()}, "participation_pct": checkpoint["participation_pct"], "recall": checkpoint["micro"]["recall"], "precision": checkpoint["micro"]["precision"], "f1": checkpoint["micro"]["f1"], "num_candidates": checkpoint["num_candidates"], "candidate_view_minutes": checkpoint["candidate_view_minutes"], "human_minutes_with_overhead": checkpoint["human_minutes_with_overhead"], "candidates_per_hour": checkpoint["candidates_per_hour"], "threshold_string": checkpoint["threshold_string"]})
    for budget, item in selected.items():
        rows.append({"mode": "max_recall_under_budget", "budget_pct": budget, "recall_target": "", **{f"thr_{k}": v for k, v in item["thresholds"].items()}, "participation_pct": item["participation_pct"], "recall": item["micro"]["recall"], "precision": item["micro"]["precision"], "f1": item["micro"]["f1"], "num_candidates": item["num_candidates"], "candidate_view_minutes": item["candidate_view_minutes"], "human_minutes_with_overhead": item["human_minutes_with_overhead"], "candidates_per_hour": item["candidates_per_hour"], "threshold_string": item["threshold_string"]})
    for target, item in selected_recall.items():
        rows.append({"mode": "min_cost_at_recall", "budget_pct": "", "recall_target": target, **{f"thr_{k}": v for k, v in item["thresholds"].items()}, "participation_pct": item["participation_pct"], "recall": item["micro"]["recall"], "precision": item["micro"]["precision"], "f1": item["micro"]["f1"], "num_candidates": item["num_candidates"], "candidate_view_minutes": item["candidate_view_minutes"], "human_minutes_with_overhead": item["human_minutes_with_overhead"], "candidates_per_hour": item["candidates_per_hour"], "threshold_string": item["threshold_string"]})
    csv_path = args.output.with_suffix(".csv")
    fields = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(args.output), "csv": str(csv_path), "checkpoint": {"participation_pct": checkpoint["participation_pct"], "recall": checkpoint["micro"]["recall"], "precision": checkpoint["micro"]["precision"]}, "budgets": {k: {"participation_pct": v["participation_pct"], "recall": v["micro"]["recall"], "precision": v["micro"]["precision"], "threshold_string": v["threshold_string"]} for k,v in selected.items()}, "recall_targets": {k: {"participation_pct": v["participation_pct"], "recall": v["micro"]["recall"], "precision": v["micro"]["precision"], "human_minutes_with_overhead": v["human_minutes_with_overhead"], "threshold_string": v["threshold_string"]} for k,v in selected_recall.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
