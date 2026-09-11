#!/usr/bin/env python
"""Exactly retune per-class thresholds on six dense videos.

This produces two explicitly-oracle threshold sets:
1. maximum duplicate-neutral window F1;
2. maximum precision subject to checkpoint GT recall not decreasing.

The selected window thresholds are also evaluated with merged proposals.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_long_video_checkpoint import point_nms_predictions, pred_matches_gt  # noqa: E402
from recompute_football_eval_protocols import load_gt_events, normalize_score_columns, read_csv  # noqa: E402

DEFAULT_VIDEOS = (
    "2027564888428580866",
    "2027572095412195330",
    "2027572406738604033",
    "2042125172076392450",
    "2042520971893485569",
    "2042525152494694401",
)


def maximum_matching(edges: list[list[int]]) -> int:
    gt_to_pred: dict[int, int] = {}

    def augment(pred_idx: int, seen: set[int]) -> bool:
        for gt_idx in edges[pred_idx]:
            if gt_idx in seen:
                continue
            seen.add(gt_idx)
            old = gt_to_pred.get(gt_idx)
            if old is None or augment(old, seen):
                gt_to_pred[gt_idx] = pred_idx
                return True
        return False

    for pred_idx in range(len(edges)):
        augment(pred_idx, set())
    return len(gt_to_pred)


def metric(tp: int, fp: int, num_gt: int, **extra: Any) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / num_gt if num_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": num_gt - tp, "num_gt": num_gt, "precision": precision, "recall": recall, "f1": f1, **extra}


def exact_window_curve(
    video_data: list[dict[str, Any]], label: str, tolerance_sec: float
) -> list[dict[str, Any]]:
    """Incremental exact curve under duplicate-neutral one-to-one matching."""
    states: list[dict[str, Any]] = []
    candidates: list[tuple[float, int, int, list[int]]] = []
    total_gt = 0
    for video_idx, data in enumerate(video_data):
        gts = [gt for gt in data["gts"] if gt["label"] == label]
        total_gt += len(gts)
        states.append({"edges": [], "gt_to_pred": {}})
        for row in data["rows"]:
            pred = {
                "label": label,
                "start_sec": float(row["start_sec"]),
                "end_sec": float(row["end_sec"]),
                "time_sec": (float(row["start_sec"]) + float(row["end_sec"])) * 0.5,
            }
            edges = [
                gt_idx
                for gt_idx, gt in enumerate(gts)
                if pred_matches_gt(pred, gt, tolerance_sec, matching_mode="window")
            ]
            candidates.append((float(row[f"prob_{label}"]), video_idx, len(states[video_idx]["edges"]), edges))
            states[video_idx]["edges"].append(edges)

    candidates.sort(key=lambda item: item[0], reverse=True)
    active: list[set[int]] = [set() for _ in states]
    matched_total = 0
    fp_total = 0
    positive_total = 0

    def augment(video_idx: int, pred_idx: int, seen_gt: set[int]) -> bool:
        state = states[video_idx]
        for gt_idx in state["edges"][pred_idx]:
            if gt_idx in seen_gt:
                continue
            seen_gt.add(gt_idx)
            old_pred = state["gt_to_pred"].get(gt_idx)
            if old_pred is None or augment(video_idx, old_pred, seen_gt):
                state["gt_to_pred"][gt_idx] = pred_idx
                return True
        return False

    curve: list[dict[str, Any]] = []
    index = 0
    while index < len(candidates):
        score = candidates[index][0]
        end = index
        while end < len(candidates) and candidates[end][0] == score:
            _, video_idx, pred_idx, edges = candidates[end]
            active[video_idx].add(pred_idx)
            positive_total += 1
            if not edges:
                fp_total += 1
            elif augment(video_idx, pred_idx, set()):
                matched_total += 1
            end += 1
        curve.append(metric(
            matched_total,
            fp_total,
            total_gt,
            threshold=score,
            num_positive_windows=positive_total,
            num_ignored_duplicates=positive_total - fp_total - matched_total,
        ))
        index = end
    curve.insert(0, metric(0, 0, total_gt, threshold=1.0000001, num_positive_windows=0, num_ignored_duplicates=0))
    return curve


def select_at_threshold(curve: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    eligible = [item for item in curve if float(item["threshold"]) >= threshold]
    return dict(eligible[-1] if eligible else curve[0])


def select_f1(curve: list[dict[str, Any]]) -> dict[str, Any]:
    return dict(max(curve, key=lambda item: (float(item["f1"]), float(item["precision"]), float(item["recall"]), float(item["threshold"]))))


def select_recall_preserving(curve: list[dict[str, Any]], recall_floor: float) -> dict[str, Any]:
    eligible = [item for item in curve if float(item["recall"]) + 1e-12 >= recall_floor]
    if not eligible:
        return dict(max(curve, key=lambda item: float(item["recall"])))
    return dict(max(eligible, key=lambda item: (float(item["precision"]), float(item["f1"]), float(item["threshold"]))))


def evaluate_merged(
    video_data: list[dict[str, Any]], labels: list[str], thresholds: dict[str, float], tolerance_sec: float, radius_sec: float
) -> dict[str, dict[str, Any]]:
    totals = {label: {"tp": 0, "fp": 0, "gt": 0, "pred": 0} for label in labels}
    for data in video_data:
        proposals = point_nms_predictions(data["rows"], labels, thresholds, radius_sec)
        for proposal in proposals:
            proposal["start_sec"] = float(proposal["support_start_sec"])
            proposal["end_sec"] = float(proposal["support_end_sec"])
        for label in labels:
            if data["excluded"].get(label, False):
                continue
            preds = [item for item in proposals if item["label"] == label]
            gts = [item for item in data["gts"] if item["label"] == label]
            edges = [[gt_idx for gt_idx, gt in enumerate(gts) if pred_matches_gt(pred, gt, tolerance_sec, matching_mode="window")] for pred in preds]
            tp = maximum_matching(edges)
            totals[label]["tp"] += tp
            totals[label]["fp"] += len(preds) - tp
            totals[label]["gt"] += len(gts)
            totals[label]["pred"] += len(preds)
    return {label: metric(item["tp"], item["fp"], item["gt"], num_proposals=item["pred"]) for label, item in totals.items()}


def aggregate(per_class: dict[str, dict[str, Any]]) -> dict[str, Any]:
    tp = sum(int(item["tp"]) for item in per_class.values())
    fp = sum(int(item["fp"]) for item in per_class.values())
    gt = sum(int(item["num_gt"]) for item in per_class.values())
    return metric(tp, fp, gt)


def checkpoint_thresholds(video_dirs: list[Path], labels: list[str]) -> dict[str, float]:
    values = []
    for video_dir in video_dirs:
        summary = json.loads((video_dir / "summary.json").read_text())
        values.append({label: float(summary["thresholds"][label]) for label in labels})
    if any(item != values[0] for item in values[1:]):
        raise ValueError("per-video checkpoint thresholds differ")
    return values[0]


def evaluate_run(run_dir: Path, videos: tuple[str, ...], tolerance_sec: float, radius_sec: float) -> dict[str, Any]:
    video_dirs = [run_dir / video_id for video_id in videos]
    first = json.loads((video_dirs[0] / "summary.json").read_text())
    labels = [str(label) for label in first["labels"]]
    thresholds = checkpoint_thresholds(video_dirs, labels)
    video_data = []
    for video_dir in video_dirs:
        rows = normalize_score_columns(read_csv(video_dir / "window_predictions.csv"), labels, "prob")
        gts = load_gt_events(video_dir / "gt_events.csv", labels)
        video_data.append({
            "video_id": video_dir.name,
            "rows": rows,
            "gts": gts,
            "excluded": {"set_piece": video_dir.name == "2027572406738604033"},
        })

    strategies: dict[str, Any] = {
        "checkpoint_tuned": {"thresholds": thresholds, "window": {"per_class": {}}},
        "dense_oracle_f1": {"thresholds": {}, "window": {"per_class": {}}},
        "dense_oracle_recall_preserving": {"thresholds": {}, "window": {"per_class": {}}},
    }
    for label in labels:
        filtered = [
            {**data, "gts": [] if data["excluded"].get(label, False) else data["gts"], "rows": [] if data["excluded"].get(label, False) else data["rows"]}
            for data in video_data
        ]
        curve = exact_window_curve(filtered, label, tolerance_sec)
        checkpoint = select_at_threshold(curve, thresholds[label])
        best_f1 = select_f1(curve)
        recall_preserving = select_recall_preserving(curve, float(checkpoint["recall"]))
        strategies["checkpoint_tuned"]["window"]["per_class"][label] = checkpoint
        strategies["dense_oracle_f1"]["thresholds"][label] = best_f1["threshold"]
        strategies["dense_oracle_f1"]["window"]["per_class"][label] = best_f1
        strategies["dense_oracle_recall_preserving"]["thresholds"][label] = recall_preserving["threshold"]
        strategies["dense_oracle_recall_preserving"]["window"]["per_class"][label] = recall_preserving

    for strategy in strategies.values():
        strategy["window"]["micro"] = aggregate(strategy["window"]["per_class"])
        strategy["merged"] = {"per_class": evaluate_merged(video_data, labels, strategy["thresholds"], tolerance_sec, radius_sec)}
        strategy["merged"]["micro"] = aggregate(strategy["merged"]["per_class"])
    return {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "video_ids": list(videos),
        "tolerance_sec": tolerance_sec,
        "merge_radius_sec": radius_sec,
        "note": "dense_oracle_* thresholds are selected on these same six videos and are not held-out scores",
        "strategies": strategies,
    }


def flatten(result: dict[str, Any], strategy: str, protocol: str) -> dict[str, Any]:
    item = result["strategies"][strategy]
    row: dict[str, Any] = {"run_name": result["run_name"], "run_dir": result["run_dir"], "strategy": strategy, "protocol": protocol}
    for label, value in item["thresholds"].items():
        row[f"threshold_{label}"] = value
    metrics = item[protocol]
    for label, values in metrics["per_class"].items():
        for key, value in values.items():
            row[f"{label}_{key}"] = value
    for key, value in metrics["micro"].items():
        row[f"micro_{key}"] = value
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/football_eval_runs")
    parser.add_argument("--videos", default=",".join(DEFAULT_VIDEOS))
    parser.add_argument("--tolerance-sec", type=float, default=2.0)
    parser.add_argument("--merge-radius-sec", type=float, default=5.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    videos = tuple(item.strip() for item in args.videos.split(",") if item.strip())
    results = []
    audit = []
    for run_dir in sorted(path for path in args.root.iterdir() if path.is_dir()):
        required = [run_dir / video_id / name for video_id in videos for name in ("summary.json", "window_predictions.csv", "gt_events.csv")]
        if not all(path.is_file() for path in required):
            continue
        output = run_dir / "retuned_new_protocols_six_videos_tol2.json"
        try:
            if output.exists() and not args.overwrite:
                result = json.loads(output.read_text())
                status = "existing"
            else:
                result = evaluate_run(run_dir, videos, args.tolerance_sec, args.merge_radius_sec)
                output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
                status = "completed"
            results.append(result)
            audit.append({"run_name": run_dir.name, "status": status})
            print(f"{status}: {run_dir.name}", flush=True)
        except Exception as exc:
            audit.append({"run_name": run_dir.name, "status": "error", "reason": f"{type(exc).__name__}: {exc}"})
            print(f"error: {run_dir.name}: {type(exc).__name__}: {exc}", flush=True)

    rows = [flatten(result, strategy, protocol) for result in results for strategy in result["strategies"] for protocol in ("window", "merged")]
    master = args.root / "all_runs_retuned_new_protocols_six_videos_tol2.csv"
    fields = sorted({key for row in rows for key in row})
    with master.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    audit_path = args.root / "all_runs_retuned_new_protocols_six_videos_tol2_audit.json"
    audit_path.write_text(json.dumps({"runs": audit}, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"runs": len(results), "rows": len(rows), "master": str(master), "audit": str(audit_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
