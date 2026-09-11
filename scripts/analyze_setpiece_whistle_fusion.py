#!/usr/bin/env python3
"""Evaluate visual set-piece windows augmented with review-only whistle proposals.

The visual threshold is read from the existing per-video LOOV report.  Whistle
activity is never converted directly into a set-piece subtype: it only opens a
review interval in which a human (or a later visual classifier) may confirm
corner/freekick/kickoff.  Connected acoustic activities are kept independently;
no temporal NMS is applied.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SUBTYPE_ALIASES = {
    "corner": {"corner", "角球"},
    "freekick": {"freekick", "free_kick", "free kick", "任意球"},
    "kickoff": {"kickoff", "kick_off", "kick off", "中圈开球", "开球"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--adaptive-report", required=True, type=Path)
    parser.add_argument("--whistle-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tolerance-sec", type=float, default=3.0)
    parser.add_argument("--whistle-pre-sec", type=float, default=5.0)
    parser.add_argument("--whistle-post-sec", type=float, default=10.0)
    parser.add_argument("--target-recall", type=float, default=0.85)
    parser.add_argument(
        "--independent-calibration-threshold", type=float, default=1.8502276013294856,
        help="Threshold previously selected on calibration18, never test18.",
    )
    return parser.parse_args()


def logit(probability: float) -> float:
    probability = min(max(float(probability), 1e-8), 1.0 - 1e-8)
    return math.log(probability / (1.0 - probability))


def subtype(raw_label: str) -> str | None:
    normalized = str(raw_label).strip().lower()
    for name, aliases in SUBTYPE_ALIASES.items():
        if normalized in aliases:
            return name
    return None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    result: list[list[float]] = []
    for start, end in sorted(intervals):
        if not result or start > result[-1][1]:
            result.append([start, end])
        else:
            result[-1][1] = max(result[-1][1], end)
    return [(start, end) for start, end in result]


def interval_duration(intervals: Iterable[tuple[float, float]]) -> float:
    return sum(end - start for start, end in merge_intervals(intervals))


def covers(time_sec: float, interval: tuple[float, float], tolerance: float) -> bool:
    return interval[0] - tolerance <= time_sec <= interval[1] + tolerance


def load_visual(
    run_dir: Path, report: dict[str, Any]
) -> tuple[dict[str, list[dict[str, float]]], dict[str, dict[str, float]]]:
    folds = report["per_class"]["set_piece"]["adaptive_median_logit_loov"]["folds"]
    by_video: dict[str, list[dict[str, float]]] = {}
    threshold_meta: dict[str, dict[str, float]] = {}
    for fold in folds:
        video_id = str(fold["held_out_video_id"])
        alpha = float(fold["alpha"])
        median_logit = float(fold["video_median_logit"])
        threshold = float(fold["adaptive_threshold"])
        selected = []
        for row in read_csv(run_dir / video_id / "window_predictions.csv"):
            adjusted = logit(float(row["prob_set_piece"])) - alpha * median_logit
            if adjusted < threshold:
                continue
            start = float(row["start_sec"])
            end = float(row["end_sec"])
            selected.append({
                "score": adjusted, "start": start, "end": end,
                "time": 0.5 * (start + end),
            })
        by_video[video_id] = selected
        threshold_meta[video_id] = {
            "alpha": alpha, "video_median_logit": median_logit,
            "adaptive_threshold": threshold,
        }
    return by_video, threshold_meta


def load_gt(run_dir: Path, video_ids: Iterable[str]) -> tuple[
    dict[str, dict[str, list[float]]], Counter[str], Counter[str]
]:
    result: dict[str, dict[str, list[float]]] = {}
    recognized: Counter[str] = Counter()
    unknown: Counter[str] = Counter()
    for video_id in video_ids:
        per_subtype = {name: [] for name in SUBTYPE_ALIASES}
        for row in json.loads((run_dir / video_id / "gt_events.json").read_text()):
            if row.get("label") != "set_piece":
                continue
            raw = str(row.get("raw_label", ""))
            name = subtype(raw)
            if name is None:
                unknown[raw] += 1
                continue
            per_subtype[name].append(float(row["time_sec"]))
            recognized[name] += 1
        for values in per_subtype.values():
            values.sort()
        result[video_id] = per_subtype
    return result, recognized, unknown


def load_whistles(root: Path, video_ids: Iterable[str]) -> dict[str, list[dict[str, float]]]:
    result = {}
    for video_id in video_ids:
        path = root / f"{video_id}_whistles.csv"
        rows = [] if not path.is_file() else read_csv(path)
        result[video_id] = sorted(({
            "time": float(row["peak_time_sec"]),
            "score": float(row["peak_score"]),
        } for row in rows), key=lambda row: row["time"])
    return result


def best_audio_score_for_gt(
    gt_time: float, whistles: list[dict[str, float]], *, pre: float, post: float,
    tolerance: float,
) -> float:
    matches = [
        row["score"] for row in whistles
        if row["time"] - pre - tolerance <= gt_time <= row["time"] + post + tolerance
    ]
    return max(matches, default=-math.inf)


def event_evidence(
    video_ids: Iterable[str], gt: dict[str, dict[str, list[float]]],
    visual: dict[str, list[dict[str, float]]],
    whistles: dict[str, list[dict[str, float]]], *, pre: float, post: float,
    tolerance: float,
) -> dict[str, dict[str, list[tuple[bool, float]]]]:
    result: dict[str, dict[str, list[tuple[bool, float]]]] = {}
    for video_id in video_ids:
        per_subtype = {}
        for name, times in gt[video_id].items():
            values = []
            for time_sec in times:
                visual_hit = any(
                    covers(time_sec, (row["start"], row["end"]), tolerance)
                    for row in visual[video_id]
                )
                audio_score = best_audio_score_for_gt(
                    time_sec, whistles[video_id], pre=pre, post=post,
                    tolerance=tolerance,
                )
                values.append((visual_hit, audio_score))
            per_subtype[name] = values
        result[video_id] = per_subtype
    return result


def recall_at(
    evidence: dict[str, dict[str, list[tuple[bool, float]]]],
    video_ids: Iterable[str], threshold: float,
) -> dict[str, dict[str, float | int]]:
    output = {}
    for name in SUBTYPE_ALIASES:
        values = [item for video_id in video_ids for item in evidence[video_id][name]]
        baseline = sum(visual_hit for visual_hit, _ in values)
        fused = sum(visual_hit or score >= threshold for visual_hit, score in values)
        output[name] = {
            "gt": len(values), "visual_matched": baseline, "fused_matched": fused,
            "visual_recall": baseline / len(values) if values else 0.0,
            "fused_recall": fused / len(values) if values else 0.0,
            "additional_matched": fused - baseline,
        }
    return output


def choose_threshold(
    train_video_ids: list[str], evidence: dict[str, dict[str, list[tuple[bool, float]]]],
    whistles: dict[str, list[dict[str, float]]], target: float,
) -> tuple[float, dict[str, Any]]:
    thresholds = {math.inf}
    for video_id in train_video_ids:
        thresholds.update(row["score"] for row in whistles[video_id])
    best: tuple[float, dict[str, Any]] | None = None
    best_infeasible: tuple[tuple[float, float, int], float, dict[str, Any]] | None = None
    for threshold in sorted(thresholds, reverse=True):
        recalls = recall_at(evidence, train_video_ids, threshold)
        available = [value for value in recalls.values() if int(value["gt"]) > 0]
        minimum = min(float(value["fused_recall"]) for value in available)
        macro = sum(float(value["fused_recall"]) for value in available) / len(available)
        candidates = sum(
            row["score"] >= threshold
            for video_id in train_video_ids for row in whistles[video_id]
        )
        diagnostic = {
            "train_min_subtype_recall": minimum,
            "train_macro_subtype_recall": macro,
            "train_whistle_candidates": candidates,
            "train_recall": recalls,
            "target_met": minimum + 1e-12 >= target,
        }
        if diagnostic["target_met"]:
            best = (threshold, diagnostic)
            break
        key = (minimum, macro, -candidates)
        if best_infeasible is None or key > best_infeasible[0]:
            best_infeasible = (key, threshold, diagnostic)
    if best is not None:
        return best
    assert best_infeasible is not None
    return best_infeasible[1], best_infeasible[2]


def point_matches(
    gt_times: list[float], visual: list[dict[str, float]],
    whistles: list[dict[str, float]], threshold: float, tolerance: float,
) -> tuple[int, int]:
    unmatched = set(range(len(gt_times)))
    baseline = 0
    for row in sorted(visual, key=lambda item: item["score"], reverse=True):
        candidates = [
            index for index in unmatched
            if abs(row["time"] - gt_times[index]) <= tolerance
        ]
        if not candidates:
            continue
        index = min(candidates, key=lambda value: abs(row["time"] - gt_times[value]))
        unmatched.remove(index)
        baseline += 1
    fused = baseline
    for row in sorted(
        (item for item in whistles if item["score"] >= threshold),
        key=lambda item: item["score"], reverse=True,
    ):
        candidates = [
            index for index in unmatched
            if abs(row["time"] - gt_times[index]) <= tolerance
        ]
        if not candidates:
            continue
        index = min(candidates, key=lambda value: abs(row["time"] - gt_times[value]))
        unmatched.remove(index)
        fused += 1
    return baseline, fused


def evaluate_protocol(
    name: str, video_thresholds: dict[str, float], video_ids: list[str],
    evidence: dict[str, dict[str, list[tuple[bool, float]]]],
    gt: dict[str, dict[str, list[float]]], visual: dict[str, list[dict[str, float]]],
    whistles: dict[str, list[dict[str, float]]], run_dir: Path, *, pre: float,
    post: float, tolerance: float,
) -> dict[str, Any]:
    totals = {
        subtype_name: {
            "gt": 0, "visual_window_matched": 0, "fused_window_matched": 0,
            "visual_point_matched": 0, "fused_point_matched": 0,
        }
        for subtype_name in SUBTYPE_ALIASES
    }
    raw_visual_candidates = 0
    raw_whistle_candidates = 0
    visual_union = 0.0
    fused_union = 0.0
    total_video_sec = 0.0
    per_video = []
    for video_id in video_ids:
        threshold = video_thresholds[video_id]
        selected_whistles = [row for row in whistles[video_id] if row["score"] >= threshold]
        visual_intervals = [(row["start"], row["end"]) for row in visual[video_id]]
        audio_intervals = [
            (max(0.0, row["time"] - pre), row["time"] + post)
            for row in selected_whistles
        ]
        summary = json.loads((run_dir / video_id / "summary.json").read_text())
        duration = float(summary["duration_sec"])
        total_video_sec += duration
        visual_union += interval_duration(visual_intervals)
        fused_union += interval_duration(visual_intervals + audio_intervals)
        raw_visual_candidates += len(visual[video_id])
        raw_whistle_candidates += len(selected_whistles)
        video_metrics = {}
        for subtype_name, times in gt[video_id].items():
            values = evidence[video_id][subtype_name]
            visual_window = sum(item[0] for item in values)
            fused_window = sum(item[0] or item[1] >= threshold for item in values)
            visual_point, fused_point = point_matches(
                times, visual[video_id], whistles[video_id], threshold, tolerance,
            )
            target = totals[subtype_name]
            target["gt"] += len(times)
            target["visual_window_matched"] += visual_window
            target["fused_window_matched"] += fused_window
            target["visual_point_matched"] += visual_point
            target["fused_point_matched"] += fused_point
            video_metrics[subtype_name] = {
                "gt": len(times), "visual_window_matched": visual_window,
                "fused_window_matched": fused_window,
            }
        per_video.append({
            "video_id": video_id, "whistle_threshold": threshold,
            "visual_candidates": len(visual[video_id]),
            "whistle_candidates": len(selected_whistles), "subtypes": video_metrics,
        })
    aggregate = {}
    for subtype_name, value in totals.items():
        count = value["gt"]
        aggregate[subtype_name] = {
            **value,
            "visual_window_recall": value["visual_window_matched"] / count if count else 0.0,
            "fused_window_recall": value["fused_window_matched"] / count if count else 0.0,
            "window_recall_gain": (
                value["fused_window_matched"] - value["visual_window_matched"]
            ) / count if count else 0.0,
            "visual_point_recall": value["visual_point_matched"] / count if count else 0.0,
            "fused_point_recall": value["fused_point_matched"] / count if count else 0.0,
        }
    finite_thresholds = [value for value in video_thresholds.values() if math.isfinite(value)]
    return {
        "name": name,
        "aggregate": aggregate,
        "workload": {
            "visual_setpiece_candidates": raw_visual_candidates,
            "additional_whistle_candidates": raw_whistle_candidates,
            "visual_setpiece_review_union_sec": visual_union,
            "fused_setpiece_review_union_sec": fused_union,
            "additional_review_union_sec": fused_union - visual_union,
            "visual_setpiece_review_ratio": visual_union / total_video_sec,
            "fused_setpiece_review_ratio": fused_union / total_video_sec,
            "additional_review_ratio": (fused_union - visual_union) / total_video_sec,
            "total_video_sec": total_video_sec,
        },
        "threshold_summary": {
            "finite_videos": len(finite_thresholds),
            "min": min(finite_thresholds) if finite_thresholds else None,
            "median": statistics.median(finite_thresholds) if finite_thresholds else None,
            "max": max(finite_thresholds) if finite_thresholds else None,
        },
        "per_video": per_video,
    }


def main() -> None:
    args = parse_args()
    adaptive = json.loads(args.adaptive_report.read_text())
    visual, visual_thresholds = load_visual(args.run_dir, adaptive)
    video_ids = sorted(visual)
    gt, recognized, unknown = load_gt(args.run_dir, video_ids)
    whistles = load_whistles(args.whistle_dir, video_ids)
    evidence = event_evidence(
        video_ids, gt, visual, whistles, pre=args.whistle_pre_sec,
        post=args.whistle_post_sec, tolerance=args.tolerance_sec,
    )

    loov_thresholds = {}
    loov_selection = {}
    for held_out in video_ids:
        train_ids = [video_id for video_id in video_ids if video_id != held_out]
        threshold, diagnostic = choose_threshold(
            train_ids, evidence, whistles, args.target_recall,
        )
        loov_thresholds[held_out] = threshold
        loov_selection[held_out] = diagnostic

    protocols = [
        evaluate_protocol(
            "independent_calibration18_fixed", {
                video_id: args.independent_calibration_threshold for video_id in video_ids
            }, video_ids, evidence, gt, visual, whistles, args.run_dir,
            pre=args.whistle_pre_sec, post=args.whistle_post_sec,
            tolerance=args.tolerance_sec,
        ),
        evaluate_protocol(
            "test18_leave_one_video_out", loov_thresholds, video_ids, evidence,
            gt, visual, whistles, args.run_dir, pre=args.whistle_pre_sec,
            post=args.whistle_post_sec, tolerance=args.tolerance_sec,
        ),
    ]
    result = {
        "schema": "football_setpiece_whistle_fusion.v1",
        "model": "fromlast_e8/best.pt shared set_piece head",
        "protocol": {
            "visual_threshold": "adaptive_median_logit_loov",
            "whistle_role": "review-only subtype-agnostic proposal",
            "whistle_interval_sec": [-args.whistle_pre_sec, args.whistle_post_sec],
            "tolerance_sec": args.tolerance_sec,
            "temporal_nms": False,
            "acoustic_connected_activity_grouping": True,
            "loov_target_each_subtype": args.target_recall,
        },
        "setpiece_gt_counts": dict(recognized),
        "unmapped_setpiece_raw_labels": dict(unknown),
        "visual_thresholds": visual_thresholds,
        "loov_whistle_selection": loov_selection,
        "protocols": protocols,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({
        "setpiece_gt_counts": result["setpiece_gt_counts"],
        "unmapped_setpiece_raw_labels": result["unmapped_setpiece_raw_labels"],
        "protocols": [{
            "name": item["name"], "aggregate": item["aggregate"],
            "workload": item["workload"],
            "threshold_summary": item["threshold_summary"],
        } for item in protocols],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
