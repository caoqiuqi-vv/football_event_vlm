#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_LABELS = ("shot", "save", "set_piece")


def parse_thresholds(raw: str | None, labels: tuple[str, ...], default: float) -> dict[str, float]:
    thresholds = {label: float(default) for label in labels}
    if not raw:
        return thresholds
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Threshold item must be label=value, got: {item}")
        label, value = item.split("=", 1)
        label = label.strip()
        if label not in thresholds:
            raise ValueError(f"Unknown label in thresholds: {label}")
        thresholds[label] = float(value)
    return thresholds


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def to_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def sigmoid_logit(prob: float) -> float:
    prob = min(max(float(prob), 1e-6), 1.0 - 1e-6)
    return math.log(prob / (1.0 - prob))


def nearest_gt(
    gt_events: list[dict[str, str]],
    label: str,
    start_sec: float,
    end_sec: float,
    tolerance_sec: float,
) -> tuple[int, float, str]:
    center_sec = 0.5 * (start_sec + end_sec)
    best_inside = (0, float("inf"), "")
    best_any = (0, float("inf"), "")
    for event in gt_events:
        if event.get("label") != label:
            continue
        event_time = to_float(event.get("time_sec"))
        distance = abs(center_sec - event_time)
        if distance < best_any[1]:
            best_any = (0, distance, event.get("event_id", ""))
        if start_sec - tolerance_sec <= event_time <= end_sec + tolerance_sec and distance < best_inside[1]:
            best_inside = (1, distance, event.get("event_id", ""))
    if best_inside[0]:
        return best_inside
    if best_any[1] != float("inf"):
        return best_any
    return 0, -1.0, ""


def load_window_frame_scores(video_dir: Path) -> dict[tuple[int, str, str], dict[str, float]]:
    rows = read_csv(video_dir / "frame_event_window_scores.csv")
    scores: dict[tuple[int, str, str], dict[str, float]] = {}
    for row in rows:
        window_index = to_int(row.get("window_index"), -1)
        branch = row.get("branch", "") or "global"
        label = row.get("label", "")
        if window_index < 0 or not label:
            continue
        scores[(window_index, branch, label)] = {
            "target": to_float(row.get("target")),
            "max_frame_prob": to_float(row.get("max_frame_prob")),
        }
    return scores


def load_frame_logit_stats(video_dir: Path, labels: tuple[str, ...]) -> dict[tuple[int, str], dict[str, float]]:
    rows = read_csv(video_dir / "frame_event_logits.csv")
    grouped: dict[tuple[int, str], list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        window_index = to_int(row.get("window_index"), -1)
        if window_index < 0:
            continue
        rel_time = to_float(row.get("relative_time_sec"))
        abs_time = to_float(row.get("frame_time_sec"))
        for label in labels:
            prob_col = f"frame_prob_{label}"
            local_prob_col = f"local_frame_prob_{label}"
            fused_prob_col = f"roi_fused_frame_prob_{label}"
            for prefix, col in (("", prob_col), ("local_", local_prob_col), ("roi_fused_", fused_prob_col)):
                if col in row and row.get(col) != "":
                    grouped[(window_index, prefix + label)].append(
                        {
                            "prob": to_float(row.get(col)),
                            "relative_time_sec": rel_time,
                            "frame_time_sec": abs_time,
                        }
                    )
    stats: dict[tuple[int, str], dict[str, float]] = {}
    for key, items in grouped.items():
        probs = sorted((item["prob"] for item in items), reverse=True)
        if not probs:
            continue
        max_idx = max(range(len(items)), key=lambda idx: items[idx]["prob"])
        top2 = probs[: min(2, len(probs))]
        top4 = probs[: min(4, len(probs))]
        stats[key] = {
            "frame_prob_max": probs[0],
            "frame_prob_mean": sum(probs) / len(probs),
            "frame_prob_top2_mean": sum(top2) / len(top2),
            "frame_prob_top4_mean": sum(top4) / len(top4),
            "frame_peak_relative_time_sec": items[max_idx]["relative_time_sec"],
            "frame_peak_time_sec": items[max_idx]["frame_time_sec"],
        }
    return stats


def discover_video_dirs(eval_run_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in eval_run_dir.iterdir()
        if path.is_dir() and (path / "window_predictions.csv").exists()
    )


def build_rows_for_video(
    eval_run_dir: Path,
    video_dir: Path,
    labels: tuple[str, ...],
    thresholds: dict[str, float],
    match_tolerance_sec: float,
    include_all_windows: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    video_id = video_dir.name
    window_rows = read_csv(video_dir / "window_predictions.csv")
    gt_events = read_csv(video_dir / "gt_events.csv")
    frame_window_scores = load_window_frame_scores(video_dir)
    frame_stats = load_frame_logit_stats(video_dir, labels)
    out_rows: list[dict[str, Any]] = []
    summary = {
        "video_id": video_id,
        "num_windows": len(window_rows),
        "num_gt_events": len(gt_events),
        "labels": {label: {"rows": 0, "candidates": 0, "tp_candidates": 0, "hard_fp_candidates": 0, "gt_windows": 0} for label in labels},
    }

    for window in window_rows:
        window_index = to_int(window.get("index"), -1)
        start_sec = to_float(window.get("start_sec"))
        end_sec = to_float(window.get("end_sec"))
        center_sec = 0.5 * (start_sec + end_sec)
        probs = {label: to_float(window.get(f"prob_{label}")) for label in labels}
        sorted_probs = sorted(probs.values(), reverse=True)
        max_other = {
            label: max((value for other, value in probs.items() if other != label), default=0.0)
            for label in labels
        }
        second_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
        for label in labels:
            clip_prob = probs[label]
            is_gt_window, nearest_distance, nearest_event_id = nearest_gt(
                gt_events,
                label,
                start_sec,
                end_sec,
                match_tolerance_sec,
            )
            is_candidate = int(clip_prob >= thresholds[label])
            if not include_all_windows and not is_candidate and not is_gt_window:
                continue
            summary["labels"][label]["rows"] += 1
            summary["labels"][label]["candidates"] += is_candidate
            summary["labels"][label]["tp_candidates"] += int(is_candidate and is_gt_window)
            summary["labels"][label]["hard_fp_candidates"] += int(is_candidate and not is_gt_window)
            summary["labels"][label]["gt_windows"] += int(is_gt_window)

            row: dict[str, Any] = {
                "source_run": eval_run_dir.name,
                "video_id": video_id,
                "window_index": window_index,
                "label": label,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "center_sec": center_sec,
                "clip_prob": clip_prob,
                "clip_logit": sigmoid_logit(clip_prob),
                "candidate_threshold": thresholds[label],
                "is_candidate": is_candidate,
                "is_gt_window": int(is_gt_window),
                "is_tp_candidate": int(is_candidate and is_gt_window),
                "is_hard_fp_candidate": int(is_candidate and not is_gt_window),
                "nearest_gt_distance_sec": nearest_distance,
                "nearest_gt_event_id": nearest_event_id,
                "max_other_prob": max_other[label],
                "margin_to_max_other": clip_prob - max_other[label],
                "margin_to_second_class": clip_prob - second_prob,
                "prob_shot": probs.get("shot", 0.0),
                "prob_save": probs.get("save", 0.0),
                "prob_set_piece": probs.get("set_piece", 0.0),
                "crop_area_ratio": to_float(window.get("crop_area_ratio")),
                "crop_goal_count": to_int(window.get("crop_goal_count")),
                "crop_ball_count": to_int(window.get("crop_ball_count")),
                "crop_person_count": to_int(window.get("crop_person_count")),
                "roi_valid": to_float(window.get("roi_valid")),
                "roi_confidence": to_float(window.get("roi_confidence")),
                "roi_goal_score": to_float(window.get("roi_goal_score")),
                "roi_ball_score": to_float(window.get("roi_ball_score")),
                "roi_center_circle_score": to_float(window.get("roi_center_circle_score")),
                "roi_person_support": to_float(window.get("roi_person_support")),
                "roi_frame_quality_mean": to_float(window.get("roi_frame_quality_mean")),
            }
            for label2 in labels:
                for prefix in ("global_", "local_", "roi_gate_", "roi_quality_prob_"):
                    col = f"{prefix}prob_{label2}" if prefix != "roi_gate_" else f"roi_gate_{label2}"
                    if prefix == "roi_quality_prob_":
                        col = f"roi_quality_prob_{label2}"
                    if col in window:
                        row[col] = to_float(window.get(col))
                global_col = f"global_prob_{label2}"
                local_col = f"local_prob_{label2}"
                if global_col in window and local_col in window:
                    row[f"local_minus_global_prob_{label2}"] = to_float(window.get(local_col)) - to_float(window.get(global_col))
                    row[f"fused_minus_global_prob_{label2}"] = probs[label2] - to_float(window.get(global_col))

            for branch in ("global", "local", "roi_fused", "fused"):
                score = frame_window_scores.get((window_index, branch, label))
                if score:
                    row[f"{branch}_frame_target"] = score["target"]
                    row[f"{branch}_frame_max_prob"] = score["max_frame_prob"]
            for stat_prefix, stat_key in (("", label), ("local_", f"local_{label}"), ("roi_fused_", f"roi_fused_{label}")):
                stats = frame_stats.get((window_index, stat_key))
                if stats:
                    for key, value in stats.items():
                        row[f"{stat_prefix}{key}"] = value
            out_rows.append(row)
    return out_rows, summary


def finalize_summary(summary: dict[str, Any]) -> dict[str, Any]:
    totals = {label: {"rows": 0, "candidates": 0, "tp_candidates": 0, "hard_fp_candidates": 0, "gt_windows": 0} for label in summary["labels"]}
    for video_summary in summary["videos"]:
        for label, stats in video_summary["labels"].items():
            for key in totals[label]:
                totals[label][key] += int(stats.get(key, 0))
    for label, stats in totals.items():
        candidates = stats["candidates"]
        gt_windows = stats["gt_windows"]
        stats["candidate_precision_vs_gt_window"] = stats["tp_candidates"] / candidates if candidates else 0.0
        stats["gt_window_coverage_by_candidate"] = stats["tp_candidates"] / gt_windows if gt_windows else 0.0
    summary["labels"] = totals
    summary["num_rows"] = sum(stats["rows"] for stats in totals.values())
    summary["num_candidate_rows"] = sum(stats["candidates"] for stats in totals.values())
    summary["num_hard_fp_candidate_rows"] = sum(stats["hard_fp_candidates"] for stats in totals.values())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a candidate-level dataset for long-video football event verifier/reranker experiments."
    )
    parser.add_argument("--eval-run-dir", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-summary", type=Path)
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--min-prob", type=float, default=0.10)
    parser.add_argument("--candidate-thresholds", default=None, help="Comma separated label=value list, e.g. shot=0.1,save=0.1,set_piece=0.1")
    parser.add_argument("--match-tolerance-sec", type=float, default=2.0)
    parser.add_argument("--include-all-windows", action="store_true")
    args = parser.parse_args()

    labels = tuple(label.strip() for label in args.labels.split(",") if label.strip())
    if not labels:
        raise ValueError("--labels cannot be empty")
    eval_run_dir = args.eval_run_dir
    if not eval_run_dir.exists():
        raise FileNotFoundError(eval_run_dir)
    thresholds = parse_thresholds(args.candidate_thresholds, labels, args.min_prob)
    video_dirs = discover_video_dirs(eval_run_dir)
    if not video_dirs:
        raise FileNotFoundError(f"No video eval directories found under {eval_run_dir}")

    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "eval_run_dir": str(eval_run_dir),
        "candidate_thresholds": thresholds,
        "match_tolerance_sec": args.match_tolerance_sec,
        "include_all_windows": bool(args.include_all_windows),
        "num_videos": len(video_dirs),
        "videos": [],
        "labels": {label: {} for label in labels},
    }
    for video_dir in video_dirs:
        video_rows, video_summary = build_rows_for_video(
            eval_run_dir,
            video_dir,
            labels,
            thresholds,
            args.match_tolerance_sec,
            args.include_all_windows,
        )
        rows.extend(video_rows)
        summary["videos"].append(video_summary)

    write_csv(args.output_csv, rows)
    summary = finalize_summary(summary)
    output_summary = args.output_summary or args.output_csv.with_suffix(".summary.json")
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(
        f"wrote rows={len(rows)} candidates={summary['num_candidate_rows']} "
        f"hard_fp={summary['num_hard_fp_candidate_rows']} csv={args.output_csv} summary={output_summary}"
    )
    for label, stats in summary["labels"].items():
        print(
            f"{label}: rows={stats['rows']} candidates={stats['candidates']} "
            f"tp_candidates={stats['tp_candidates']} hard_fp={stats['hard_fp_candidates']} "
            f"precision_vs_gt_window={stats['candidate_precision_vs_gt_window']:.4f} "
            f"gt_window_coverage={stats['gt_window_coverage_by_candidate']:.4f}"
        )


if __name__ == "__main__":
    main()
