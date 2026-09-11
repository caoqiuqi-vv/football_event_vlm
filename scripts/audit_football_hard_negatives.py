#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def quantiles(values: Iterable[float]) -> dict[str, float | None]:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return {"min": None, "p25": None, "median": None, "p75": None, "max": None}

    def at(fraction: float) -> float:
        return ordered[round((len(ordered) - 1) * fraction)]

    return {
        "min": ordered[0],
        "p25": at(0.25),
        "median": at(0.5),
        "p75": at(0.75),
        "max": ordered[-1],
    }


def nearest_distance(center_sec: float, times: list[float]) -> float | None:
    if not times:
        return None
    return min(abs(center_sec - time_sec) for time_sec in times)


def load_gt_times(path: Path) -> tuple[list[float], dict[str, list[float]]]:
    all_times: list[float] = []
    by_label: dict[str, list[float]] = {}
    for row in read_csv(path):
        time_sec = float(row["time_sec"])
        label = str(row.get("label", row.get("event", ""))).strip()
        all_times.append(time_sec)
        if label:
            by_label.setdefault(label, []).append(time_sec)
    return all_times, by_label


def audit(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    items = manifest.get("hard_negatives", manifest.get("windows", []))
    run_dir = Path(args.eval_run_dir or manifest.get("eval_run_dir", ""))
    run_config_path = run_dir / "run_config.json" if run_dir else Path()
    run_config = json.loads(run_config_path.read_text()) if run_dir and run_config_path.exists() else {}
    mining_checkpoint = str(
        manifest.get("mining_checkpoint", run_config.get("checkpoint", ""))
    )
    min_scores = {
        str(label): float(score)
        for label, score in manifest.get("min_scores", {}).items()
    }
    combo_counts: Counter[str] = Counter()
    video_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    item_scores: list[float] = []
    label_scores: dict[str, list[float]] = {}
    branch_scores: dict[str, dict[str, list[float]]] = {}
    branch_retained: dict[str, dict[str, int]] = {}
    nearest_all_gt: list[float] = []
    nearest_same_gt: dict[str, list[float]] = {}
    missing_rows = 0

    rows_by_video: dict[str, dict[int, dict[str, str]]] = {}
    gt_by_video: dict[str, tuple[list[float], dict[str, list[float]]]] = {}

    for item in items:
        video_id = str(item["video_id"])
        labels = sorted(str(label) for label in item.get("labels", []))
        combo_counts["+".join(labels)] += 1
        video_counts[video_id] += 1
        item_scores.append(float(item.get("score", 0.0)))
        for label in labels:
            label_counts[label] += 1
            score = float(item.get("label_scores", {}).get(label, item.get("score", 0.0)))
            label_scores.setdefault(label, []).append(score)

        if not run_dir:
            continue
        if video_id not in rows_by_video:
            prediction_path = run_dir / video_id / "window_predictions.csv"
            gt_path = run_dir / video_id / "gt_events.csv"
            rows_by_video[video_id] = {
                int(row["index"]): row for row in read_csv(prediction_path)
            }
            gt_by_video[video_id] = load_gt_times(gt_path)

        row = rows_by_video[video_id].get(int(item["window_index"]))
        if row is None:
            missing_rows += 1
            continue
        center_sec = float(item["center_sec"])
        all_gt, gt_by_label = gt_by_video[video_id]
        distance = nearest_distance(center_sec, all_gt)
        if distance is not None:
            nearest_all_gt.append(distance)

        for label in labels:
            same_distance = nearest_distance(center_sec, gt_by_label.get(label, []))
            if same_distance is not None:
                nearest_same_gt.setdefault(label, []).append(same_distance)
            threshold = min_scores.get(label, 0.0)
            for branch, prefix in (("fused", ""), ("global", "global_"), ("local", "local_")):
                key = f"{prefix}prob_{label}"
                if key not in row or row[key] == "":
                    continue
                score = float(row[key])
                branch_scores.setdefault(label, {}).setdefault(branch, []).append(score)
                if score >= threshold:
                    branch_retained.setdefault(label, {}).setdefault(branch, 0)
                    branch_retained[label][branch] += 1

    branch_summary: dict[str, Any] = {}
    for label, branches in branch_scores.items():
        branch_summary[label] = {}
        total = label_counts[label]
        for branch, scores in branches.items():
            retained = branch_retained.get(label, {}).get(branch, 0)
            branch_summary[label][branch] = {
                "score_quantiles": quantiles(scores),
                "retained_at_mining_threshold": retained,
                "retained_rate": retained / total if total else 0.0,
            }

    supervision_count = sum(label_counts.values())
    output: dict[str, Any] = {
        "manifest": str(manifest_path),
        "eval_run_dir": str(run_dir) if run_dir else None,
        "mining_checkpoint": mining_checkpoint or None,
        "target_checkpoint": args.target_checkpoint or None,
        "checkpoint_matches_target": (
            Path(mining_checkpoint) == Path(args.target_checkpoint)
            if mining_checkpoint and args.target_checkpoint
            else None
        ),
        "num_items": len(items),
        "num_label_supervisions": supervision_count,
        "num_videos_with_items": len(video_counts),
        "item_fraction_of_train_clips": (
            len(items) / args.train_num_clips if args.train_num_clips else None
        ),
        "label_supervision_fraction_of_train_label_slots": (
            supervision_count / (args.train_num_clips * args.num_labels)
            if args.train_num_clips and args.num_labels
            else None
        ),
        "combination_counts": dict(sorted(combo_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "items_per_video": quantiles(video_counts.values()),
        "item_score_quantiles": quantiles(item_scores),
        "label_score_quantiles": {
            label: quantiles(scores) for label, scores in sorted(label_scores.items())
        },
        "source_model_cross_branch": branch_summary,
        "nearest_any_gt_sec": quantiles(nearest_all_gt),
        "nearest_same_label_gt_sec": {
            label: quantiles(distances)
            for label, distances in sorted(nearest_same_gt.items())
        },
        "missing_prediction_rows": missing_rows,
    }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a football hard-negative manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--eval-run-dir", default="")
    parser.add_argument("--train-num-clips", type=int, default=0)
    parser.add_argument("--num-labels", type=int, default=3)
    parser.add_argument("--target-checkpoint", default="")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit(args)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n")
        print(f"wrote {output}")
    print(rendered)


if __name__ == "__main__":
    main()
