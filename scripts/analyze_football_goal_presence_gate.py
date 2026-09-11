#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


LABELS = ("shot", "save", "set_piece")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def load_goal_observations(path: Path) -> tuple[np.ndarray, np.ndarray]:
    item = torch.load(path, map_location="cpu", weights_only=True)
    fps = float(item["fps"])
    frame_ids = item["frame_ids"].cpu().numpy()
    offsets = item["frame_offsets"].cpu().numpy()
    classes = item["classes"].cpu().numpy()
    confidences = item["confidences"].float().cpu().numpy()
    times: list[float] = []
    scores: list[float] = []
    for index, frame_id in enumerate(frame_ids):
        start, end = int(offsets[index]), int(offsets[index + 1])
        mask = classes[start:end] == 2
        if mask.any():
            times.append(float(frame_id) / fps)
            scores.append(float(confidences[start:end][mask].max()))
    return np.asarray(times, dtype=np.float64), np.asarray(scores, dtype=np.float32)


def evaluate(
    records: list[dict[str, Any]],
    thresholds: dict[str, float],
    gate_labels: set[str],
    goal_conf: float,
    min_goal_frames: int,
    tolerance_sec: float,
) -> dict[str, Any]:
    output: dict[str, Any] = {"per_class": {}}
    for label in LABELS:
        tp = fp = matched = total_gt = filtered_tp = filtered_fp = 0
        for record in records:
            if label in record["excluded_labels"]:
                continue
            gt = record["gt"][label]
            total_gt += len(gt)
            hit: set[int] = set()
            for window in record["windows"]:
                if float(window[f"prob_{label}"]) < thresholds[label]:
                    continue
                matches = [
                    index
                    for index, time_sec in enumerate(gt)
                    if float(window["start_sec"]) - tolerance_sec
                    <= time_sec
                    <= float(window["end_sec"]) + tolerance_sec
                ]
                gated = (
                    label in gate_labels
                    and not record["missing_detection"]
                    and int(window["goal_support_by_conf"][goal_conf]) < min_goal_frames
                )
                if gated:
                    filtered_tp += int(bool(matches))
                    filtered_fp += int(not matches)
                    continue
                if matches:
                    tp += 1
                    hit.update(matches)
                else:
                    fp += 1
            matched += len(hit)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = matched / total_gt if total_gt else 0.0
        output["per_class"][label] = {
            "tp": tp,
            "fp": fp,
            "fn": total_gt - matched,
            "num_gt": total_gt,
            "num_matched_gt": matched,
            "precision": precision,
            "recall": recall,
            "filtered_tp_windows": filtered_tp,
            "filtered_fp_windows": filtered_fp,
        }
    values = output["per_class"].values()
    tp = sum(item["tp"] for item in values)
    fp = sum(item["fp"] for item in values)
    total_gt = sum(item["num_gt"] for item in values)
    matched = sum(item["num_matched_gt"] for item in values)
    output["micro"] = {
        "tp": tp,
        "fp": fp,
        "fn": total_gt - matched,
        "num_gt": total_gt,
        "num_matched_gt": matched,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": matched / total_gt if total_gt else 0.0,
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test hard shot/save gating based on goal presence in each dense window."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--index-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--match-tolerance-sec", type=float, default=2.0)
    parser.add_argument("--goal-confidences", default="0.2,0.3,0.4,0.5,0.6,0.7")
    parser.add_argument("--min-goal-frames", default="1,2,3,5,10")
    parser.add_argument("--exclude", action="append", default=[])
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    index_root = Path(args.index_root)
    output_dir = Path(args.output_dir)
    goal_confidences = [float(value) for value in args.goal_confidences.split(",")]
    min_goal_frames_values = [int(value) for value in args.min_goal_frames.split(",")]
    exclusions = {tuple(value.rsplit(":", 1)) for value in args.exclude}

    config = json.loads((run_dir / "run_config.json").read_text())
    video_ids = [str(value) for value in config["video_ids"]]
    first_summary = json.loads((run_dir / video_ids[0] / "summary.json").read_text())
    thresholds = {label: float(first_summary["thresholds"][label]) for label in LABELS}

    records: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    missing_detection_videos: list[str] = []
    for video_id in video_ids:
        index_path = index_root / f"{video_id}.pt"
        missing_detection = not index_path.is_file()
        if missing_detection:
            times = np.empty(0, dtype=np.float64)
            scores = np.empty(0, dtype=np.float32)
            missing_detection_videos.append(video_id)
        else:
            times, scores = load_goal_observations(index_path)

        windows = read_csv(run_dir / video_id / "window_predictions.csv")
        gt_rows = read_csv(run_dir / video_id / "gt_events.csv")
        gt = {
            label: [float(row["time_sec"]) for row in gt_rows if row["label"] == label]
            for label in LABELS
        }
        excluded_labels = {label for item_video, label in exclusions if item_video == video_id}
        for window in windows:
            left = int(np.searchsorted(times, float(window["start_sec"]), side="left"))
            right = int(np.searchsorted(times, float(window["end_sec"]), side="right"))
            window_scores = scores[left:right]
            support_by_conf = {
                goal_conf: int((window_scores >= goal_conf).sum())
                for goal_conf in goal_confidences
            }
            window["goal_support_by_conf"] = support_by_conf
            feature_rows.append(
                {
                    "video_id": video_id,
                    "index": window["index"],
                    "start_sec": window["start_sec"],
                    "end_sec": window["end_sec"],
                    "missing_detection": missing_detection,
                    "max_goal_conf": float(window_scores.max()) if len(window_scores) else 0.0,
                    **{
                        f"goal_frames_conf_{goal_conf:g}": support
                        for goal_conf, support in support_by_conf.items()
                    },
                }
            )
        records.append(
            {
                "video_id": video_id,
                "windows": windows,
                "gt": gt,
                "excluded_labels": excluded_labels,
                "missing_detection": missing_detection,
            }
        )

    baseline = evaluate(
        records,
        thresholds,
        set(),
        goal_conf=goal_confidences[0],
        min_goal_frames=0,
        tolerance_sec=args.match_tolerance_sec,
    )
    experiments: list[dict[str, Any]] = []
    for gate_name, gate_labels in (
        ("shot_save", {"shot", "save"}),
        ("shot", {"shot"}),
        ("save", {"save"}),
    ):
        for goal_conf in goal_confidences:
            for min_goal_frames in min_goal_frames_values:
                metrics = evaluate(
                    records,
                    thresholds,
                    gate_labels,
                    goal_conf,
                    min_goal_frames,
                    args.match_tolerance_sec,
                )
                experiments.append(
                    {
                        "gate_labels": gate_name,
                        "goal_conf": goal_conf,
                        "min_goal_frames": min_goal_frames,
                        **metrics,
                    }
                )

    base_shot_recall = baseline["per_class"]["shot"]["recall"]
    base_save_recall = baseline["per_class"]["save"]["recall"]

    def recall_safe(item: dict[str, Any], max_drop: float) -> bool:
        return (
            item["per_class"]["shot"]["recall"] + max_drop + 1e-12 >= base_shot_recall
            and item["per_class"]["save"]["recall"] + max_drop + 1e-12 >= base_save_recall
        )

    no_loss = [item for item in experiments if recall_safe(item, 0.0)]
    within_half_pp = [item for item in experiments if recall_safe(item, 0.005)]
    best_no_loss = max(
        no_loss,
        key=lambda item: (item["micro"]["precision"], item["micro"]["recall"]),
        default=None,
    )
    best_within_half_pp = max(
        within_half_pp,
        key=lambda item: (item["micro"]["precision"], item["micro"]["recall"]),
        default=None,
    )
    strict_no_goal = next(
        item
        for item in experiments
        if item["gate_labels"] == "shot_save"
        and item["goal_conf"] == 0.5
        and item["min_goal_frames"] == 1
    )

    summary = {
        "protocol": {
            "match_tolerance_sec": args.match_tolerance_sec,
            "prediction_postprocess": "window_overlap",
            "missing_detection_policy": "keep",
            "gate": "remove selected DINO predictions when the window has fewer than min_goal_frames goal detections",
            "excluded": sorted([list(item) for item in exclusions]),
        },
        "thresholds": thresholds,
        "missing_detection_videos": missing_detection_videos,
        "baseline": baseline,
        "strict_no_goal_conf_0_5": strict_no_goal,
        "best_no_recall_loss": best_no_loss,
        "best_recall_drop_le_0_5pp": best_within_half_pp,
        "num_experiments": len(experiments),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    with (output_dir / "window_goal_features.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(feature_rows[0]))
        writer.writeheader()
        writer.writerows(feature_rows)
    with (output_dir / "grid_results.jsonl").open("w") as file:
        for item in experiments:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
