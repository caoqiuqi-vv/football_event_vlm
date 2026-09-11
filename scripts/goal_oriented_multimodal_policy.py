#!/usr/bin/env python
"""Calibrate/apply the deployable visual + whistle human-review policy.

Shot/save use independent visual candidates.  Set-piece uses a calibrated union
of visual candidates and connected whistle activity.  Audio proposals are
review-only: a whistle is evidence for a restart, never an automatic set-piece
label.  Event predictions are never NMSed or merged; only reviewer playback
intervals are grouped.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from scripts import goal_oriented_review_policy as gp


def read_whistles(root: Path, video_ids: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for video_id in video_ids:
        path = root / f"{video_id}_whistles.csv"
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for item in csv.DictReader(handle):
                score = float(item["peak_score"])
                activity_id = str(item["activity_id"])
                rows.append({
                    "candidate_id": f"whistle:{video_id}:{activity_id}:set_piece",
                    "row": -1,
                    "video_id": video_id,
                    "label": "set_piece",
                    "time_sec": float(item["peak_time_sec"]),
                    "score": score,
                    "raw_score": score,
                    "source": "whistle",
                })
    return rows


def selected_visual_rows(
    data: dict[str, Any], teacher: dict[str, Any] | None,
    label: str, item: dict[str, Any], *, review_only: bool,
) -> list[dict[str, Any]]:
    scores = gp.score_from_definition(data, teacher, label, item["score_definition"])
    rows = gp.candidate_rows(data, label, scores)
    low = float(item["review_boundary"])
    auto = float(item["auto_accept_boundary"])
    selected = []
    for row in rows:
        if float(row["score"]) < low:
            continue
        row["decision"] = "auto_accept" if float(row["score"]) >= auto else "review"
        row["review_boundary"] = low
        row["auto_accept_boundary"] = auto
        if not review_only or row["decision"] == "review":
            selected.append(row)
    return selected


def workload_ratio(
    rows: Sequence[dict[str, Any]], data: dict[str, Any], *,
    review_sec: float, max_segment_sec: float,
) -> tuple[float, int, float]:
    segments = gp.make_review_segments(
        rows, data["durations"], review_sec=review_sec,
        max_segment_sec=max_segment_sec,
    )
    intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for segment in segments:
        intervals[str(segment["video_id"])].append(
            (float(segment["start_sec"]), float(segment["end_sec"]))
        )
    union = sum(gp.intervals_union_duration(value) for value in intervals.values())
    total = sum(data["durations"].values())
    return union / max(total, 1e-12), len(segments), union


def multimodal_setpiece_calibration(
    data: dict[str, Any], teacher: dict[str, Any] | None,
    whistles: list[dict[str, Any]], fixed_review_rows: list[dict[str, Any]], *,
    recall_target: float, bootstrap_floor: float, tolerance_sec: float,
    grid_size: int, bootstrap_samples: int, alpha_grid: Sequence[float],
    review_sec: float, max_segment_sec: float, seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    label = "set_piece"
    gt = gp.class_gt(data, label)
    source_candidates = gp.build_calibration_score_candidates(
        data, teacher, label, alpha_grid,
    )
    whistle_scores = np.asarray([float(row["score"]) for row in whistles])
    whistle_thresholds = (
        np.concatenate(([math.inf], gp.threshold_grid(whistle_scores, grid_size)))
        if len(whistle_scores) else np.asarray([math.inf])
    )
    diagnostics: list[dict[str, Any]] = []
    feasible: list[dict[str, Any]] = []
    point_feasible: list[dict[str, Any]] = []
    best_ceiling = 0.0
    for source_index, source in enumerate(source_candidates):
        visual = gp.candidate_rows(data, label, source["scores"])
        visual_scores = np.asarray([float(row["score"]) for row in visual])
        # Infinity permits audio-only.  Quantiles keep the joint search bounded.
        visual_thresholds = np.concatenate((
            [math.inf], gp.threshold_grid(visual_scores, grid_size),
        ))
        source_best: dict[str, Any] | None = None
        for visual_threshold in visual_thresholds:
            selected_visual = [
                row for row in visual if float(row["score"]) >= visual_threshold
            ]
            # Give visual evidence a fixed priority over review-only audio.  This
            # is part of the frozen score definition, not per-video tuning.
            visual_eval = [{**row, "score": 2.0 + float(gp.sigmoid(np.asarray([row["score"]]))[0])} for row in selected_visual]
            for whistle_threshold in whistle_thresholds:
                selected_audio = [
                    {**row, "score": 1.0 + float(gp.sigmoid(np.asarray([row["score"]]))[0])}
                    for row in whistles if float(row["raw_score"]) >= whistle_threshold
                ]
                selected = visual_eval + selected_audio
                metrics = gp.match_selected(selected, gt, tolerance_sec)
                p05 = gp.bootstrap_recall_p05(
                    metrics["per_video"], samples=bootstrap_samples,
                    seed=seed + source_index,
                )
                best_ceiling = max(best_ceiling, float(metrics["recall"]))
                review_visual = [{**row, "decision": "review"} for row in selected_visual]
                review_audio = [
                    {**row, "decision": "review"} for row in whistles
                    if float(row["raw_score"]) >= whistle_threshold
                ]
                ratio, segments, union_sec = workload_ratio(
                    fixed_review_rows + review_visual + review_audio, data,
                    review_sec=review_sec, max_segment_sec=max_segment_sec,
                )
                result = {
                    "score_name": source["name"],
                    "score_definition": source["definition"],
                    "visual_boundary": float(visual_threshold),
                    "whistle_boundary": float(whistle_threshold),
                    "metrics": gp.compact_metrics(metrics),
                    "bootstrap_recall_p05": p05,
                    "review_time_ratio_all_classes": ratio,
                    "review_segments_all_classes": segments,
                    "review_union_sec_all_classes": union_sec,
                    "visual_candidates": len(selected_visual),
                    "whistle_candidates": len(selected_audio),
                }
                if source_best is None or (
                    float(result["metrics"]["recall"]),
                    -float(result["review_time_ratio_all_classes"]),
                    float(result["metrics"]["precision"]),
                ) > (
                    float(source_best["metrics"]["recall"]),
                    -float(source_best["review_time_ratio_all_classes"]),
                    float(source_best["metrics"]["precision"]),
                ):
                    source_best = result
                if float(metrics["recall"]) + 1e-12 >= recall_target:
                    point_feasible.append(result)
                    if p05 + 1e-12 >= bootstrap_floor:
                        feasible.append(result)
        if source_best is not None:
            diagnostics.append(source_best)
    pool = feasible or point_feasible or diagnostics
    if not pool:
        raise RuntimeError("no set-piece visual/audio policy candidates")
    chosen = min(
        pool,
        key=lambda item: (
            float(item["review_time_ratio_all_classes"]),
            -float(item["metrics"]["precision"]),
            int(item["visual_candidates"]) + int(item["whistle_candidates"]),
        ),
    )
    chosen["recall_target"] = recall_target
    chosen["bootstrap_floor"] = bootstrap_floor
    chosen["candidate_recall_ceiling"] = best_ceiling
    chosen["point_gate_pass"] = bool(float(chosen["metrics"]["recall"]) >= recall_target)
    chosen["bootstrap_gate_pass"] = bool(float(chosen["bootstrap_recall_p05"]) >= bootstrap_floor)
    return chosen, diagnostics


def build_policy(
    data: dict[str, Any], teacher: dict[str, Any] | None,
    whistles: list[dict[str, Any]], args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    # Shot/save retain the robust visual calibration.  Set-piece is calibrated
    # jointly with the whistle supplement below.
    visual_policy, visual_diagnostics = gp.calibrate_policy(
        data, teacher, labels=["shot", "save"],
        recall_targets={"shot": args.shot_recall_target, "save": args.save_recall_target},
        bootstrap_floors={"shot": args.shot_bootstrap_floor, "save": args.save_bootstrap_floor},
        auto_precision_floor=args.auto_precision_floor,
        tolerance_sec=args.tolerance_sec, grid_size=args.visual_grid_size,
        bootstrap_samples=args.bootstrap_samples, alphas=args.alpha_grid,
        seed=args.seed,
    )
    fixed_review: list[dict[str, Any]] = []
    for label in ("shot", "save"):
        fixed_review.extend(selected_visual_rows(
            data, teacher, label, visual_policy["labels"][label], review_only=True,
        ))
    setpiece, setpiece_diagnostics = multimodal_setpiece_calibration(
        data, teacher, whistles, fixed_review,
        recall_target=args.setpiece_recall_target,
        bootstrap_floor=args.setpiece_bootstrap_floor,
        tolerance_sec=args.tolerance_sec, grid_size=args.audio_grid_size,
        bootstrap_samples=args.bootstrap_samples, alpha_grid=args.alpha_grid,
        review_sec=args.review_sec, max_segment_sec=args.max_review_segment_sec,
        seed=args.seed + 91,
    )
    setpiece_scores = gp.score_from_definition(
        data, teacher, "set_piece", setpiece["score_definition"],
    )
    setpiece_rows = gp.candidate_rows(data, "set_piece", setpiece_scores)
    auto_boundary, auto_metrics = gp.choose_auto_boundary(
        setpiece_rows, gp.class_gt(data, "set_piece"), setpiece["visual_boundary"],
        precision_floor=args.auto_precision_floor, tolerance_sec=args.tolerance_sec,
        grid_size=args.visual_grid_size,
    )
    labels = dict(visual_policy["labels"])
    labels["set_piece"] = {
        "score_name": setpiece["score_name"],
        "score_definition": setpiece["score_definition"],
        "review_boundary": setpiece["visual_boundary"],
        "auto_accept_boundary": auto_boundary,
        "calibration_metrics_multimodal": setpiece,
        "auto_accept_calibration_visual_only": auto_metrics,
    }
    policy = {
        "schema": "football_goal_oriented_multimodal_policy.v1",
        "selection_split": "calibration18",
        "external_threshold_retuning": False,
        "no_temporal_nms": True,
        "labels": labels,
        "whistle": {
            "review_boundary": setpiece["whistle_boundary"],
            "automatic_event_prediction": False,
            "missing_audio_fallback": "visual_only",
            "connected_activity_grouping_is_not_event_nms": True,
        },
        "tolerance_sec": args.tolerance_sec,
        "primary_workload_kpi": "review_interval_union_duration / original_video_duration",
        "review_time_target": 0.35,
        "teacher": None if teacher is None else {
            key: teacher[key] for key in ("checkpoint", "labels", "offsets_sec")
        },
    }
    diagnostics = {
        "visual": visual_diagnostics,
        "setpiece_multimodal_source_best": setpiece_diagnostics,
        "setpiece_selected": setpiece,
    }
    return policy, diagnostics


def package(
    data: dict[str, Any], teacher: dict[str, Any] | None,
    whistles: list[dict[str, Any]], policy: dict[str, Any], output: Path,
    *, split_name: str, tolerance_sec: float, review_sec: float,
    max_segment_sec: float,
) -> dict[str, Any]:
    predictions: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    selected: dict[str, list[dict[str, Any]]] = {}
    for label, item in policy["labels"].items():
        rows = selected_visual_rows(data, teacher, label, item, review_only=False)
        selected[label] = rows
        predictions.extend(rows)
        review.extend(row for row in rows if row["decision"] == "review")
    whistle_boundary = float(policy["whistle"]["review_boundary"])
    selected_audio = []
    for row in whistles:
        if float(row["raw_score"]) < whistle_boundary:
            continue
        item = {**row, "decision": "review", "review_boundary": whistle_boundary}
        selected_audio.append(item)
        review.append(item)
        predictions.append(item)
    selected["set_piece"] = selected.get("set_piece", []) + selected_audio
    metrics: dict[str, Any] = {}
    for label, rows in selected.items():
        eval_rows = []
        for row in rows:
            priority = 1.0 if row["source"] == "whistle" else 2.0
            eval_rows.append({
                **row,
                "score": priority + float(gp.sigmoid(np.asarray([row["score"]]))[0]),
            })
        value = gp.match_selected(eval_rows, gp.class_gt(data, label), tolerance_sec)
        metrics[label] = {
            **gp.compact_metrics(value),
            "per_video": value["per_video"],
            "selected_candidates": len(rows),
            "auto_accept_candidates": sum(row["decision"] == "auto_accept" for row in rows),
            "review_candidates": sum(row["decision"] == "review" for row in rows),
            "whistle_review_candidates": sum(row["source"] == "whistle" for row in rows),
        }
    segments = gp.make_review_segments(
        review, data["durations"], review_sec=review_sec,
        max_segment_sec=max_segment_sec,
    )
    intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for segment in segments:
        intervals[str(segment["video_id"])].append(
            (float(segment["start_sec"]), float(segment["end_sec"]))
        )
    union_sec = sum(gp.intervals_union_duration(value) for value in intervals.values())
    total_sec = sum(data["durations"].values())
    workload = {
        "review_candidates": len(review),
        "review_segments": len(segments),
        "review_union_minutes": union_sec / 60.0,
        "total_video_minutes": total_sec / 60.0,
        "review_time_ratio": union_sec / max(total_sec, 1e-12),
        "review_time_target_lt_35pct": union_sec / max(total_sec, 1e-12) < 0.35,
        "primary_kpi": "review_interval_union_duration / original_video_duration",
        "prediction_slots_unchanged_by_interval_grouping": True,
    }
    output.mkdir(parents=True, exist_ok=True)
    gp.write_csv(output / "predictions_selected.csv", predictions)
    gp.write_csv(output / "review_segments.csv", segments)
    gp.write_csv(output / "review_annotations.csv", [{
        **segment, "review_status": "", "correct_labels": "",
        "correct_event_times_sec": "", "confounder": "", "notes": "",
    } for segment in segments])
    report = {
        "split": split_name,
        "protocol": {
            "no_temporal_nms": True,
            "one_to_one_tolerance_sec": tolerance_sec,
            "external_threshold_retuning": False,
            "whistle_is_review_only": True,
        },
        "videos": len(data["durations"]),
        "metrics": metrics,
        "workload": workload,
        "acceptance": {
            "shot_recall_ge_90pct": metrics["shot"]["recall"] >= 0.90,
            "save_recall_ge_85pct": metrics["save"]["recall"] >= 0.85,
            "setpiece_recall_ge_85pct": metrics["set_piece"]["recall"] >= 0.85,
            "review_time_lt_35pct": workload["review_time_target_lt_35pct"],
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--whistle-dir", required=True, type=Path)
    parser.add_argument("--teacher-shard-dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--split-name", default="calibration18")
    parser.add_argument("--shot-recall-target", type=float, default=0.93)
    parser.add_argument("--save-recall-target", type=float, default=0.88)
    parser.add_argument("--setpiece-recall-target", type=float, default=0.88)
    parser.add_argument("--shot-bootstrap-floor", type=float, default=0.90)
    parser.add_argument("--save-bootstrap-floor", type=float, default=0.85)
    parser.add_argument("--setpiece-bootstrap-floor", type=float, default=0.85)
    parser.add_argument("--auto-precision-floor", type=float, default=0.80)
    parser.add_argument("--tolerance-sec", type=float, default=3.0)
    parser.add_argument("--review-sec", type=float, default=10.0)
    parser.add_argument("--max-review-segment-sec", type=float, default=20.0)
    parser.add_argument("--visual-grid-size", type=int, default=201)
    parser.add_argument("--audio-grid-size", type=int, default=17)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--alpha-grid", type=lambda value: [float(item) for item in value.split(",")], default=[0.25, 0.5, 0.75])
    parser.add_argument("--seed", type=int, default=20260830)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = gp.load_dense(args.cache, args.metadata)
    teacher = (
        gp.load_teacher_shards(args.teacher_shard_dir, len(data["video_ids"]))
        if args.teacher_shard_dir else None
    )
    whistles = read_whistles(args.whistle_dir, sorted(data["durations"]))
    if args.policy:
        policy = json.loads(args.policy.read_text(encoding="utf-8"))
        if policy.get("schema") != "football_goal_oriented_multimodal_policy.v1":
            raise ValueError("unsupported multimodal policy schema")
    else:
        policy, diagnostics = build_policy(data, teacher, whistles, args)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "policy.json").write_text(
            json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (args.output / "calibration_diagnostics.json").write_text(
            json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    report = package(
        data, teacher, whistles, policy, args.output,
        split_name=args.split_name, tolerance_sec=args.tolerance_sec,
        review_sec=args.review_sec, max_segment_sec=args.max_review_segment_sec,
    )
    print(json.dumps({
        "report": str(args.output / "report.json"),
        "metrics": {label: {key: value for key, value in item.items() if key in {"precision", "recall", "tp", "fp", "fn"}} for label, item in report["metrics"].items()},
        "workload": report["workload"],
        "acceptance": report["acceptance"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
