#!/usr/bin/env python
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

LABELS = ("shot", "save", "set_piece")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def point_box_distance(cx: float, cy: float, box: np.ndarray) -> float:
    dx = max(float(box[0]) - cx, 0.0, cx - float(box[2]))
    dy = max(float(box[1]) - cy, 0.0, cy - float(box[3]))
    return math.hypot(dx, dy)


def load_pair_observations(path: Path, goal_conf: float, max_gap_sec: float) -> tuple[np.ndarray, np.ndarray]:
    item = torch.load(path, map_location="cpu", weights_only=True)
    fps = float(item["fps"])
    width = float(item["image_size"]["width"])
    height = float(item["image_size"]["height"])
    diagonal = math.hypot(width, height)
    frame_ids = item["frame_ids"].numpy()
    offsets = item["frame_offsets"].numpy()
    classes = item["classes"].numpy()
    confidences = item["confidences"].float().numpy()
    boxes = item["boxes"].float().numpy()
    goals: dict[int, np.ndarray] = {}
    for index, frame_id in enumerate(frame_ids):
        start, end = int(offsets[index]), int(offsets[index + 1])
        mask = (classes[start:end] == 2) & (confidences[start:end] >= goal_conf)
        if mask.any():
            goals[int(frame_id)] = boxes[start:end][mask]
    goal_frames = sorted(goals)
    if not goal_frames:
        return np.empty(0), np.empty(0)
    max_gap_frames = max_gap_sec * fps
    observations: list[tuple[float, float]] = []
    for frame_id, box in zip(item["ball_frame_ids"].numpy(), item["ball_boxes"].float().numpy()):
        position = bisect.bisect_left(goal_frames, int(frame_id))
        candidates = goal_frames[max(position - 1, 0) : min(position + 1, len(goal_frames) - 1) + 1]
        if not candidates:
            continue
        nearest = min(candidates, key=lambda value: abs(value - int(frame_id)))
        if abs(nearest - int(frame_id)) > max_gap_frames:
            continue
        cx = 0.5 * (float(box[0]) + float(box[2]))
        cy = 0.5 * (float(box[1]) + float(box[3]))
        distance = min(point_box_distance(cx, cy, goal) for goal in goals[nearest]) / diagonal
        observations.append((float(frame_id) / fps, distance))
    observations.sort()
    if not observations:
        return np.empty(0), np.empty(0)
    return np.asarray([x[0] for x in observations]), np.asarray([x[1] for x in observations])


def evaluate(records: list[dict[str, Any]], labels: tuple[str, ...], thresholds: dict[str, float], gate_labels: set[str], statistic: str, distance_threshold: float, min_support: int) -> dict[str, Any]:
    output: dict[str, Any] = {"per_class": {}}
    for label in labels:
        tp = fp = matched = total_gt = filtered_tp = filtered_fp = 0
        for record in records:
            gt = record["gt"][label]
            total_gt += len(gt)
            hit: set[int] = set()
            for window in record["windows"]:
                if float(window[f"prob_{label}"]) < thresholds[label]:
                    continue
                gate = False
                if label in gate_labels and int(window["bg_support"]) >= min_support:
                    gate = float(window[f"bg_{statistic}"]) > distance_threshold
                matches = [i for i, time in enumerate(gt) if float(window["start_sec"]) - 5 <= time <= float(window["end_sec"]) + 5]
                if gate:
                    filtered_tp += int(bool(matches))
                    filtered_fp += int(not matches)
                    continue
                if matches:
                    tp += 1
                    hit.update(matches)
                else:
                    fp += 1
            matched += len(hit)
        recall = matched / total_gt if total_gt else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        output["per_class"][label] = {"tp": tp, "fp": fp, "fn": total_gt - matched, "num_gt": total_gt, "matched_gt": matched, "precision": precision, "recall": recall, "filtered_tp_windows": filtered_tp, "filtered_fp_windows": filtered_fp}
    values = output["per_class"].values()
    tp, fp = sum(x["tp"] for x in values), sum(x["fp"] for x in values)
    total_gt, matched = sum(x["num_gt"] for x in values), sum(x["matched_gt"] for x in values)
    output["micro"] = {"tp": tp, "fp": fp, "fn": total_gt - matched, "num_gt": total_gt, "matched_gt": matched, "precision": tp / (tp + fp) if tp + fp else 0.0, "recall": matched / total_gt if total_gt else 0.0}
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--index-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--goal-conf", type=float, default=0.5)
    parser.add_argument("--max-goal-frame-gap-sec", type=float, default=0.6)
    parser.add_argument("--exclude", action="append", default=[])
    args = parser.parse_args()
    run_dir, output_dir = Path(args.run_dir), Path(args.output_dir)
    config = json.loads((run_dir / "run_config.json").read_text())
    video_ids = [str(x) for x in config["video_ids"]]
    exclusions = {tuple(x.rsplit(":", 1)) for x in args.exclude}
    first_summary = json.loads((run_dir / video_ids[0] / "summary.json").read_text())
    thresholds = {label: float(first_summary["thresholds"][label]) for label in LABELS}
    records = []
    feature_rows = []
    for video_id in video_ids:
        times, distances = load_pair_observations(Path(args.index_root) / f"{video_id}.pt", args.goal_conf, args.max_goal_frame_gap_sec)
        windows = read_csv(run_dir / video_id / "window_predictions.csv")
        gt_rows = read_csv(run_dir / video_id / "gt_events.csv")
        gt = {label: ([] if (video_id, label) in exclusions else [float(x["time_sec"]) for x in gt_rows if x["label"] == label]) for label in LABELS}
        for window in windows:
            left = int(np.searchsorted(times, float(window["start_sec"]), side="left"))
            right = int(np.searchsorted(times, float(window["end_sec"]), side="right"))
            values = distances[left:right]
            window["bg_support"] = len(values)
            for name, quantile in (("min", 0.0), ("q10", 0.1), ("q25", 0.25), ("median", 0.5)):
                window[f"bg_{name}"] = float(np.quantile(values, quantile)) if len(values) else float("nan")
            feature_rows.append({"video_id": video_id, "index": window["index"], "start_sec": window["start_sec"], "end_sec": window["end_sec"], "support": len(values), **{name: window[f"bg_{name}"] for name in ("min", "q10", "q25", "median")}})
        records.append({"video_id": video_id, "windows": windows, "gt": gt})
    experiments = []
    for gate_name, gate_labels in (("shot_save", {"shot", "save"}), ("all", set(LABELS)), ("shot", {"shot"}), ("save", {"save"})):
        for statistic in ("min", "q10", "q25"):
            for min_support in (3, 5, 10, 20):
                for distance_threshold in np.arange(0.025, 0.401, 0.025):
                    metrics = evaluate(records, LABELS, thresholds, gate_labels, statistic, float(distance_threshold), min_support)
                    experiments.append({"gate_labels": gate_name, "statistic": statistic, "min_support": min_support, "distance_threshold": round(float(distance_threshold), 4), **metrics})
    baseline = evaluate(records, LABELS, thresholds, set(), "min", 1.0, 0)
    base_p, base_r = baseline["micro"]["precision"], baseline["micro"]["recall"]
    feasible = [x for x in experiments if x["micro"]["recall"] + 1e-12 >= base_r]
    safe_1pp = [x for x in experiments if x["micro"]["recall"] + 0.01 + 1e-12 >= base_r]
    best = max(feasible, key=lambda x: (x["micro"]["precision"], x["micro"]["recall"])) if feasible else None
    best_1pp = max(safe_1pp, key=lambda x: (x["micro"]["precision"], x["micro"]["recall"])) if safe_1pp else None
    summary = {"protocol": {"match_tolerance_sec": 5.0, "prediction_postprocess": "window_overlap", "missing_detection_policy": "keep", "distance": "ball center to nearest goal-box boundary / image diagonal", "excluded": sorted([list(x) for x in exclusions])}, "thresholds": thresholds, "baseline": baseline, "best_no_recall_loss": best, "best_recall_drop_le_1pp": best_1pp, "num_experiments": len(experiments)}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    with (output_dir / "window_ball_goal_features.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(feature_rows[0])); writer.writeheader(); writer.writerows(feature_rows)
    with (output_dir / "grid_results.jsonl").open("w") as file:
        for item in experiments: file.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
