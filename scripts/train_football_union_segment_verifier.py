#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


DEFAULT_LABELS = ("shot", "save", "set_piece")


@dataclass(frozen=True)
class Segment:
    video_id: str
    label: str
    index: int
    start_sec: float
    end_sec: float
    features: tuple[float, ...]
    target: int

    @property
    def key(self) -> tuple[str, str, int]:
        return self.video_id, self.label, self.index


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_label_floats(raw: str, labels: Sequence[str], default: float) -> dict[str, float]:
    values = {label: float(default) for label in labels}
    if not raw:
        return values
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Expected label=value item, got {item!r}")
        label, value = item.split("=", 1)
        label = label.strip()
        if label not in values:
            raise ValueError(f"Unknown label={label!r}")
        values[label] = float(value)
    return values


def probability_logit(probability: float, epsilon: float = 1e-6) -> float:
    probability = min(max(float(probability), epsilon), 1.0 - epsilon)
    return math.log(probability / (1.0 - probability))


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def load_video_ids(path: str) -> list[str]:
    return [
        line.strip()
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def load_gt(video_dir: Path, labels: Sequence[str]) -> dict[str, list[float]]:
    result = {label: [] for label in labels}
    for row in read_csv(video_dir / "gt_events.csv"):
        label = row.get("label", "")
        if label in result:
            result[label].append(float(row["time_sec"]))
    for times in result.values():
        times.sort()
    return result


def merge_raw(raw: list[dict[str, Any]], merge_gap_sec: float) -> list[list[dict[str, Any]]]:
    if not raw:
        return []
    ordered = sorted(raw, key=lambda row: (float(row["start_sec"]), float(row["end_sec"])))
    groups: list[list[dict[str, Any]]] = [[ordered[0]]]
    current_end = float(ordered[0]["end_sec"])
    for row in ordered[1:]:
        start = float(row["start_sec"])
        end = float(row["end_sec"])
        if start <= current_end + merge_gap_sec:
            groups[-1].append(row)
            current_end = max(current_end, end)
        else:
            groups.append([row])
            current_end = end
    return groups


def group_features(group: Sequence[dict[str, Any]], run_names: Sequence[str]) -> tuple[float, ...]:
    scores = [float(row["score"]) for row in group]
    starts = [float(row["start_sec"]) for row in group]
    ends = [float(row["end_sec"]) for row in group]
    duration = max(ends) - min(starts)
    ordered_scores = sorted(scores, reverse=True)
    top2 = ordered_scores[: min(2, len(ordered_scores))]
    top4 = ordered_scores[: min(4, len(ordered_scores))]
    active_runs = {str(row["run_name"]) for row in group}
    features: list[float] = [
        probability_logit(max(scores)),
        probability_logit(sum(scores) / len(scores)),
        probability_logit(sum(top2) / len(top2)),
        probability_logit(sum(top4) / len(top4)),
        max(scores) - min(scores),
        float(len(group)),
        float(len(active_runs)),
        float(duration),
    ]
    for run_name in run_names:
        run_scores = [float(row["score"]) for row in group if row["run_name"] == run_name]
        features.append(probability_logit(max(run_scores)) if run_scores else probability_logit(1e-6))
        features.append(float(len(run_scores)))
    return tuple(features)


def build_segments(
    run_dirs: Sequence[Path],
    video_ids: Sequence[str],
    labels: Sequence[str],
    thresholds: dict[str, float],
    merge_gap_sec: float,
) -> tuple[list[Segment], dict[str, dict[str, list[float]]], list[str], list[str]]:
    run_names = [path.name for path in run_dirs]
    feature_names = [
        "max_logit",
        "mean_logit",
        "top2_mean_logit",
        "top4_mean_logit",
        "score_span",
        "raw_window_count",
        "active_run_count",
        "segment_duration_sec",
    ]
    for run_name in run_names:
        feature_names.extend((f"{run_name}_max_logit", f"{run_name}_window_count"))

    all_segments: list[Segment] = []
    gt_by_video: dict[str, dict[str, list[float]]] = {}
    used_video_ids: list[str] = []
    for video_id in video_ids:
        gt_dir = next((run_dir / video_id for run_dir in run_dirs if (run_dir / video_id / "gt_events.csv").exists()), None)
        if gt_dir is None:
            continue
        used_video_ids.append(video_id)
        gt_by_video[video_id] = load_gt(gt_dir, labels)
        raw_by_label: dict[str, list[dict[str, Any]]] = {label: [] for label in labels}
        for run_dir, run_name in zip(run_dirs, run_names):
            window_path = run_dir / video_id / "window_predictions.csv"
            if not window_path.exists():
                continue
            for row in read_csv(window_path):
                start = float(row["start_sec"])
                end = float(row["end_sec"])
                for label in labels:
                    score = float(row.get(f"prob_{label}", 0.0) or 0.0)
                    if score >= thresholds[label]:
                        raw_by_label[label].append(
                            {
                                "start_sec": start,
                                "end_sec": end,
                                "score": score,
                                "run_name": run_name,
                            }
                        )
        for label in labels:
            groups = merge_raw(raw_by_label[label], merge_gap_sec)
            for index, group in enumerate(groups):
                start = min(float(row["start_sec"]) for row in group)
                end = max(float(row["end_sec"]) for row in group)
                target = int(any(start <= time_sec <= end for time_sec in gt_by_video[video_id][label]))
                all_segments.append(
                    Segment(
                        video_id=video_id,
                        label=label,
                        index=index,
                        start_sec=start,
                        end_sec=end,
                        features=group_features(group, run_names),
                        target=target,
                    )
                )
    return all_segments, gt_by_video, used_video_ids, feature_names


def metric(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def evaluate(
    segments: Sequence[Segment],
    scores: dict[tuple[str, str, int], float],
    thresholds: dict[str, float],
    gt_by_video: dict[str, dict[str, list[float]]],
    labels: Sequence[str],
) -> dict[str, Any]:
    totals = {label: {"tp": 0, "fp": 0, "fn": 0, "matched_gt": 0, "num_gt": 0, "coverage_seconds": 0.0, "segments": 0} for label in labels}
    per_video: list[dict[str, Any]] = []
    for video_id in sorted(gt_by_video):
        for label in labels:
            selected = [
                segment
                for segment in segments
                if segment.video_id == video_id
                and segment.label == label
                and scores.get(segment.key, 0.0) >= thresholds[label]
            ]
            matched_gt: set[int] = set()
            tp_segments = fp_segments = 0
            for segment in selected:
                hits = [
                    index
                    for index, time_sec in enumerate(gt_by_video[video_id][label])
                    if segment.start_sec <= time_sec <= segment.end_sec
                ]
                if hits:
                    tp_segments += 1
                    matched_gt.update(hits)
                else:
                    fp_segments += 1
            fn = len(gt_by_video[video_id][label]) - len(matched_gt)
            coverage = sum(max(0.0, segment.end_sec - segment.start_sec) for segment in selected)
            totals[label]["tp"] += tp_segments
            totals[label]["fp"] += fp_segments
            totals[label]["fn"] += fn
            totals[label]["matched_gt"] += len(matched_gt)
            totals[label]["num_gt"] += len(gt_by_video[video_id][label])
            totals[label]["coverage_seconds"] += coverage
            totals[label]["segments"] += len(selected)
            row = {
                "video_id": video_id,
                "label": label,
                **metric(tp_segments, fp_segments, fn),
                "matched_gt": len(matched_gt),
                "num_gt": len(gt_by_video[video_id][label]),
                "coverage_seconds": coverage,
            }
            per_video.append(row)
    per_class: dict[str, Any] = {}
    for label, values in totals.items():
        row = metric(values["tp"], values["fp"], values["fn"])
        row.update(values)
        row["event_recall"] = values["matched_gt"] / values["num_gt"] if values["num_gt"] else 0.0
        row["coverage_minutes"] = values["coverage_seconds"] / 60.0
        per_class[label] = row
    micro_values = {
        "tp": sum(values["tp"] for values in totals.values()),
        "fp": sum(values["fp"] for values in totals.values()),
        "fn": sum(values["fn"] for values in totals.values()),
        "matched_gt": sum(values["matched_gt"] for values in totals.values()),
        "num_gt": sum(values["num_gt"] for values in totals.values()),
        "coverage_seconds": sum(values["coverage_seconds"] for values in totals.values()),
        "segments": sum(values["segments"] for values in totals.values()),
    }
    micro = metric(micro_values["tp"], micro_values["fp"], micro_values["fn"])
    micro.update(micro_values)
    micro["event_recall"] = micro_values["matched_gt"] / micro_values["num_gt"] if micro_values["num_gt"] else 0.0
    micro["coverage_minutes"] = micro_values["coverage_seconds"] / 60.0
    return {"per_class": per_class, "micro": micro, "per_video": per_video}


def fit_oof(
    segments: Sequence[Segment],
    labels: Sequence[str],
    *,
    folds: int,
    regularization_c: float,
) -> tuple[dict[tuple[str, str, int], float], dict[str, Any]]:
    scores: dict[tuple[str, str, int], float] = {}
    fold_info: dict[str, Any] = {}
    for label in labels:
        rows = [segment for segment in segments if segment.label == label]
        groups = np.asarray([row.video_id for row in rows])
        x = np.asarray([row.features for row in rows], dtype=np.float64)
        y = np.asarray([row.target for row in rows], dtype=np.int64)
        split_count = min(max(2, folds), len(np.unique(groups)))
        label_folds: list[dict[str, Any]] = []
        for fold_index, (train_idx, val_idx) in enumerate(GroupKFold(n_splits=split_count).split(x, y, groups)):
            if len(np.unique(y[train_idx])) < 2:
                raise RuntimeError(f"label={label} fold={fold_index} lacks both classes")
            scaler = StandardScaler().fit(x[train_idx])
            model = LogisticRegression(class_weight="balanced", C=regularization_c, max_iter=2000, random_state=42)
            model.fit(scaler.transform(x[train_idx]), y[train_idx])
            pred = model.predict_proba(scaler.transform(x[val_idx]))[:, 1]
            for index, score in zip(val_idx, pred):
                scores[rows[int(index)].key] = float(score)
            label_folds.append({"fold": fold_index, "train_videos": sorted(set(groups[train_idx].tolist())), "val_videos": sorted(set(groups[val_idx].tolist()))})
        fold_info[label] = label_folds
    return scores, fold_info


def fit_final(segments: Sequence[Segment], labels: Sequence[str], feature_names: Sequence[str], *, regularization_c: float) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for label in labels:
        rows = [segment for segment in segments if segment.label == label]
        x = np.asarray([row.features for row in rows], dtype=np.float64)
        y = np.asarray([row.target for row in rows], dtype=np.int64)
        scaler = StandardScaler().fit(x)
        model = LogisticRegression(class_weight="balanced", C=regularization_c, max_iter=2000, random_state=42)
        model.fit(scaler.transform(x), y)
        models[label] = {
            "feature_names": list(feature_names),
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "coef": model.coef_[0].tolist(),
            "intercept": float(model.intercept_[0]),
            "num_samples": len(rows),
            "num_positive": int(y.sum()),
            "num_negative": int((1 - y).sum()),
        }
    return models


def apply_models(segments: Sequence[Segment], models: dict[str, Any]) -> dict[tuple[str, str, int], float]:
    scores: dict[tuple[str, str, int], float] = {}
    for segment in segments:
        model = models[segment.label]
        x = np.asarray(segment.features, dtype=np.float64)
        mean = np.asarray(model["mean"], dtype=np.float64)
        scale = np.asarray(model["scale"], dtype=np.float64)
        coef = np.asarray(model["coef"], dtype=np.float64)
        standardized = (x - mean) / np.where(scale == 0.0, 1.0, scale)
        scores[segment.key] = sigmoid(float(np.dot(coef, standardized) + model["intercept"]))
    return scores


def select_thresholds(
    segments: Sequence[Segment],
    scores: dict[tuple[str, str, int], float],
    gt_by_video: dict[str, dict[str, list[float]]],
    labels: Sequence[str],
    recall_floor: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    thresholds: dict[str, float] = {}
    selected: dict[str, Any] = {}
    for label in labels:
        values = sorted({scores[segment.key] for segment in segments if segment.label == label}, reverse=True)
        best: dict[str, Any] | None = None
        for threshold in values:
            result = evaluate(segments, scores, {item: threshold if item == label else 2.0 for item in labels}, gt_by_video, labels)["per_class"][label]
            if result["event_recall"] + 1e-12 < recall_floor:
                continue
            candidate = {"threshold": float(threshold), **result}
            if best is None or (candidate["precision"], -candidate["coverage_seconds"], candidate["threshold"]) > (best["precision"], -best["coverage_seconds"], best["threshold"]):
                best = candidate
        if best is None:
            # Fall back to keeping everything; the report still exposes that the floor is infeasible.
            best = {"threshold": -1.0, **evaluate(segments, scores, {item: -1.0 if item == label else 2.0 for item in labels}, gt_by_video, labels)["per_class"][label], "recall_floor_infeasible": True}
        thresholds[label] = float(best["threshold"])
        selected[label] = best
    return thresholds, selected


def row_output(segments: Sequence[Segment], scores: dict[tuple[str, str, int], float]) -> list[dict[str, Any]]:
    return [
        {
            "video_id": segment.video_id,
            "label": segment.label,
            "index": segment.index,
            "start_sec": segment.start_sec,
            "end_sec": segment.end_sec,
            "target": segment.target,
            "score": scores.get(segment.key, 0.0),
        }
        for segment in segments
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate a verifier over low-threshold union support segments.")
    parser.add_argument("--run-dirs", nargs="+", required=True, type=Path)
    parser.add_argument("--cal-video-id-file", required=True)
    parser.add_argument("--holdout-video-id-file", required=True)
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--candidate-thresholds", default="")
    parser.add_argument("--default-candidate-threshold", type=float, default=0.2)
    parser.add_argument("--merge-gap-sec", type=float, default=2.0)
    parser.add_argument("--recall-floor", type=float, default=0.90)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--regularization-c", type=float, default=0.1)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    labels = tuple(label.strip() for label in args.labels.split(",") if label.strip())
    thresholds = parse_label_floats(args.candidate_thresholds, labels, args.default_candidate_threshold)
    cal_ids = load_video_ids(args.cal_video_id_file)
    holdout_ids = load_video_ids(args.holdout_video_id_file)

    cal_segments, cal_gt, used_cal_ids, feature_names = build_segments(args.run_dirs, cal_ids, labels, thresholds, args.merge_gap_sec)
    holdout_segments, holdout_gt, used_holdout_ids, _ = build_segments(args.run_dirs, holdout_ids, labels, thresholds, args.merge_gap_sec)

    baseline_scores_cal = {segment.key: 1.0 for segment in cal_segments}
    baseline_scores_holdout = {segment.key: 1.0 for segment in holdout_segments}
    baseline_thresholds = {label: 0.0 for label in labels}
    baseline_cal = evaluate(cal_segments, baseline_scores_cal, baseline_thresholds, cal_gt, labels)
    baseline_holdout = evaluate(holdout_segments, baseline_scores_holdout, baseline_thresholds, holdout_gt, labels)

    oof_scores, folds = fit_oof(cal_segments, labels, folds=args.folds, regularization_c=args.regularization_c)
    thresholds_verifier, selected_oof = select_thresholds(cal_segments, oof_scores, cal_gt, labels, args.recall_floor)
    oof_metrics = evaluate(cal_segments, oof_scores, thresholds_verifier, cal_gt, labels)

    models = fit_final(cal_segments, labels, feature_names, regularization_c=args.regularization_c)
    holdout_scores = apply_models(holdout_segments, models)
    holdout_metrics = evaluate(holdout_segments, holdout_scores, thresholds_verifier, holdout_gt, labels)

    combined_segments = list(cal_segments) + list(holdout_segments)
    combined_gt = {**cal_gt, **holdout_gt}
    combined_baseline_scores = {segment.key: 1.0 for segment in combined_segments}
    combined_scores = {**oof_scores, **holdout_scores}
    combined_baseline = evaluate(combined_segments, combined_baseline_scores, baseline_thresholds, combined_gt, labels)
    combined_metrics = evaluate(combined_segments, combined_scores, thresholds_verifier, combined_gt, labels)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "run_dirs": [str(path) for path in args.run_dirs],
        "labels": list(labels),
        "candidate_thresholds": thresholds,
        "merge_gap_sec": args.merge_gap_sec,
        "recall_floor": args.recall_floor,
        "feature_names": feature_names,
        "cal_video_ids": used_cal_ids,
        "holdout_video_ids": used_holdout_ids,
        "baseline_cal": baseline_cal,
        "baseline_holdout": baseline_holdout,
        "baseline_35val": combined_baseline,
        "verifier_thresholds": thresholds_verifier,
        "selected_oof": selected_oof,
        "oof_cal": oof_metrics,
        "holdout": holdout_metrics,
        "combined_35val_leakage_safe": combined_metrics,
        "folds": folds,
        "models": models,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    write_csv(args.output_dir / "cal_oof_segments.csv", row_output(cal_segments, oof_scores))
    write_csv(args.output_dir / "holdout_segments.csv", row_output(holdout_segments, holdout_scores))
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "baseline_35val": combined_baseline["micro"],
        "verifier_35val": combined_metrics["micro"],
        "thresholds": thresholds_verifier,
    }, indent=2))


if __name__ == "__main__":
    main()
