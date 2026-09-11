#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = (
    "clip_logit",
    "frame_max_logit",
    "frame_top4_mean_logit",
    "frame_sharpness",
    "peak_offset_norm",
    "neighbor_clip_mean_logit",
    "neighbor_support_count",
)
DEFAULT_LABELS = ("shot", "save")
DEFAULT_BASELINE_THRESHOLDS = {"shot": 0.15, "save": 0.20, "set_piece": 0.60}
DEFAULT_PREFILTER_THRESHOLDS = {"shot": 0.02, "save": 0.02, "set_piece": 0.10}


@dataclass(frozen=True)
class Candidate:
    video_id: str
    label: str
    window_index: int
    window_start_sec: float
    window_end_sec: float
    center_time_sec: float
    peak_time_sec: float
    clip_prob: float
    features: tuple[float, ...]
    target: int | None
    nearest_gt_distance_sec: float | None

    @property
    def key(self) -> tuple[str, str, int]:
        return self.video_id, self.label, self.window_index


def parse_label_floats(raw: str, defaults: dict[str, float]) -> dict[str, float]:
    values = dict(defaults)
    if not raw.strip():
        return values
    for item in raw.split(","):
        if not item.strip():
            continue
        label, value = item.split("=", 1)
        values[label.strip()] = float(value)
    return values


def load_video_ids(run_dir: Path, path: str) -> list[str]:
    if path:
        return [
            line.strip()
            for line in Path(path).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    return sorted(
        item.name
        for item in run_dir.iterdir()
        if item.is_dir()
        and (item / "window_predictions.csv").exists()
        and (item / "frame_event_logits.csv").exists()
        and (item / "gt_events.csv").exists()
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def probability_logit(probability: float, epsilon: float = 1e-5) -> float:
    probability = min(max(float(probability), epsilon), 1.0 - epsilon)
    return math.log(probability / (1.0 - probability))


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _row_probability(row: dict[str, str], label: str) -> float:
    for key in (f"prob_{label}", f"global_prob_{label}"):
        value = row.get(key)
        if value not in (None, ""):
            return float(value)
    raise KeyError(f"Missing clip probability for label={label}; keys={sorted(row)}")


def _frame_probability(row: dict[str, str], label: str) -> float:
    for key in (f"frame_prob_{label}", f"global_frame_prob_{label}"):
        value = row.get(key)
        if value not in (None, ""):
            return float(value)
    raise KeyError(f"Missing frame probability for label={label}; keys={sorted(row)}")


def load_gt_times(video_dir: Path, labels: Sequence[str]) -> dict[str, list[float]]:
    result = {label: [] for label in labels}
    for row in read_csv(video_dir / "gt_events.csv"):
        label = row["label"]
        if label in result:
            result[label].append(float(row["time_sec"]))
    return result


def build_video_candidates(
    video_dir: Path,
    labels: Sequence[str],
    *,
    match_tolerance_sec: float,
    ignore_radius_sec: float,
    neighbor_radius_sec: float,
    support_thresholds: dict[str, float],
) -> list[Candidate]:
    video_id = video_dir.name
    windows = sorted(read_csv(video_dir / "window_predictions.csv"), key=lambda row: int(row["index"]))
    frame_rows = read_csv(video_dir / "frame_event_logits.csv")
    frames_by_window: dict[int, list[dict[str, str]]] = {}
    for row in frame_rows:
        frames_by_window.setdefault(int(row["window_index"]), []).append(row)
    gt_times = load_gt_times(video_dir, labels)

    centers = [
        (float(row["start_sec"]) + float(row["end_sec"])) * 0.5
        for row in windows
    ]
    clip_probs = {
        label: [_row_probability(row, label) for row in windows]
        for label in labels
    }
    candidates: list[Candidate] = []
    for row_pos, row in enumerate(windows):
        window_index = int(row["index"])
        start_sec = float(row["start_sec"])
        end_sec = float(row["end_sec"])
        center_sec = centers[row_pos]
        frame_values = frames_by_window.get(window_index, [])
        if not frame_values:
            continue
        for label in labels:
            ranked = sorted(
                (
                    (float(frame["frame_time_sec"]), _frame_probability(frame, label))
                    for frame in frame_values
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            peak_time, frame_max = ranked[0]
            top4 = ranked[: min(4, len(ranked))]
            top4_mean = sum(prob for _, prob in top4) / len(top4)
            neighbor_positions = [
                index
                for index, other_center in enumerate(centers)
                if index != row_pos and abs(other_center - center_sec) <= neighbor_radius_sec
            ]
            neighbor_values = [clip_probs[label][index] for index in neighbor_positions]
            neighbor_mean = (
                sum(neighbor_values) / len(neighbor_values)
                if neighbor_values
                else clip_probs[label][row_pos]
            )
            support_count = sum(
                value >= support_thresholds.get(label, 0.05)
                for value in neighbor_values
            )
            nearest_distance = (
                min(abs(peak_time - gt_time) for gt_time in gt_times[label])
                if gt_times[label]
                else None
            )
            if nearest_distance is not None and nearest_distance <= match_tolerance_sec:
                target: int | None = 1
            elif nearest_distance is not None and nearest_distance <= ignore_radius_sec:
                target = None
            else:
                target = 0
            duration = max(end_sec - start_sec, 1e-6)
            features = (
                probability_logit(clip_probs[label][row_pos]),
                probability_logit(frame_max),
                probability_logit(top4_mean),
                frame_max - top4_mean,
                abs(peak_time - center_sec) / (duration * 0.5),
                probability_logit(neighbor_mean),
                float(support_count),
            )
            candidates.append(
                Candidate(
                    video_id=video_id,
                    label=label,
                    window_index=window_index,
                    window_start_sec=start_sec,
                    window_end_sec=end_sec,
                    center_time_sec=center_sec,
                    peak_time_sec=peak_time,
                    clip_prob=clip_probs[label][row_pos],
                    features=features,
                    target=target,
                    nearest_gt_distance_sec=nearest_distance,
                )
            )
    return candidates


def load_candidates(
    run_dir: Path,
    video_ids: Sequence[str],
    labels: Sequence[str],
    **kwargs: Any,
) -> list[Candidate]:
    result: list[Candidate] = []
    for video_id in video_ids:
        video_dir = run_dir / video_id
        missing = [
            name
            for name in ("window_predictions.csv", "frame_event_logits.csv", "gt_events.csv")
            if not (video_dir / name).exists()
        ]
        if missing:
            raise FileNotFoundError(f"Missing {missing} under {video_dir}")
        result.extend(build_video_candidates(video_dir, labels, **kwargs))
    return result


def point_nms(
    candidates: Iterable[Candidate],
    scores: dict[tuple[str, str, int], float],
    *,
    radius_sec: float,
    use_peak_time: bool,
) -> list[Candidate]:
    kept: list[Candidate] = []
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            -scores[candidate.key],
            candidate.peak_time_sec if use_peak_time else candidate.center_time_sec,
        ),
    )
    for candidate in ordered:
        time_sec = candidate.peak_time_sec if use_peak_time else candidate.center_time_sec
        if any(
            abs(
                time_sec
                - (old.peak_time_sec if use_peak_time else old.center_time_sec)
            )
            <= radius_sec
            for old in kept
        ):
            continue
        kept.append(candidate)
    return kept


def evaluate_predictions(
    candidates: Sequence[Candidate],
    scores: dict[tuple[str, str, int], float],
    thresholds: dict[str, float],
    gt_by_video: dict[str, dict[str, list[float]]],
    labels: Sequence[str],
    *,
    nms_radius_sec: float,
    match_tolerance_sec: float,
    use_peak_time: bool,
) -> dict[str, Any]:
    totals = {label: {"tp": 0, "fp": 0, "fn": 0} for label in labels}
    per_video: list[dict[str, Any]] = []
    video_ids = sorted({candidate.video_id for candidate in candidates})
    for video_id in video_ids:
        for label in labels:
            selected = [
                candidate
                for candidate in candidates
                if candidate.video_id == video_id
                and candidate.label == label
                and scores[candidate.key] >= thresholds[label]
            ]
            predictions = point_nms(
                selected,
                scores,
                radius_sec=nms_radius_sec,
                use_peak_time=use_peak_time,
            )
            gts = gt_by_video[video_id][label]
            used_gt: set[int] = set()
            tp = fp = 0
            for prediction in predictions:
                pred_time = (
                    prediction.peak_time_sec
                    if use_peak_time
                    else prediction.center_time_sec
                )
                matches = [
                    (gt_index, abs(pred_time - gt_time))
                    for gt_index, gt_time in enumerate(gts)
                    if gt_index not in used_gt
                    and abs(pred_time - gt_time) <= match_tolerance_sec
                ]
                if matches:
                    gt_index, _ = min(matches, key=lambda item: item[1])
                    used_gt.add(gt_index)
                    tp += 1
                else:
                    fp += 1
            fn = len(gts) - len(used_gt)
            totals[label]["tp"] += tp
            totals[label]["fp"] += fp
            totals[label]["fn"] += fn
            per_video.append(_metric_row(video_id, label, tp, fp, fn))
    per_class = {
        label: _metric_row("", label, **values)
        for label, values in totals.items()
    }
    micro = _metric_row(
        "",
        "micro",
        sum(item["tp"] for item in totals.values()),
        sum(item["fp"] for item in totals.values()),
        sum(item["fn"] for item in totals.values()),
    )
    return {"per_class": per_class, "micro": micro, "per_video": per_video}


def _metric_row(video_id: str, label: str, tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "video_id": video_id,
        "label": label,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def collect_gt(
    run_dir: Path,
    video_ids: Sequence[str],
    labels: Sequence[str],
) -> dict[str, dict[str, list[float]]]:
    return {
        video_id: load_gt_times(run_dir / video_id, labels)
        for video_id in video_ids
    }


def fit_oof_models(
    candidates: Sequence[Candidate],
    labels: Sequence[str],
    *,
    folds: int,
    regularization_c: float,
) -> tuple[dict[tuple[str, str, int], float], dict[str, Any]]:
    oof_scores: dict[tuple[str, str, int], float] = {}
    fold_info: dict[str, Any] = {}
    for label in labels:
        rows = [candidate for candidate in candidates if candidate.label == label]
        groups = np.asarray([candidate.video_id for candidate in rows])
        unique_groups = np.unique(groups)
        split_count = min(max(folds, 2), len(unique_groups))
        if split_count < 2:
            raise ValueError(f"Need at least two calibration videos for label={label}")
        x = np.asarray([candidate.features for candidate in rows], dtype=np.float64)
        targets = np.asarray(
            [-1 if candidate.target is None else int(candidate.target) for candidate in rows],
            dtype=np.int64,
        )
        label_folds: list[dict[str, Any]] = []
        for fold_index, (raw_train_indices, val_indices) in enumerate(
            GroupKFold(n_splits=split_count).split(x, groups=groups)
        ):
            train_indices = raw_train_indices[targets[raw_train_indices] >= 0]
            if len(np.unique(targets[train_indices])) < 2:
                raise ValueError(
                    f"Fold {fold_index} for label={label} does not contain both classes"
                )
            scaler = StandardScaler().fit(x[train_indices])
            model = LogisticRegression(
                class_weight="balanced",
                C=regularization_c,
                max_iter=2000,
                random_state=42,
            ).fit(scaler.transform(x[train_indices]), targets[train_indices])
            predictions = model.predict_proba(scaler.transform(x[val_indices]))[:, 1]
            for index, score in zip(val_indices, predictions):
                oof_scores[rows[int(index)].key] = float(score)
            label_folds.append(
                {
                    "fold": fold_index,
                    "train_videos": sorted(set(groups[train_indices].tolist())),
                    "val_videos": sorted(set(groups[val_indices].tolist())),
                }
            )
        fold_info[label] = label_folds
    return oof_scores, fold_info


def fit_final_models(
    candidates: Sequence[Candidate],
    labels: Sequence[str],
    *,
    regularization_c: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label in labels:
        rows = [
            candidate
            for candidate in candidates
            if candidate.label == label and candidate.target is not None
        ]
        x = np.asarray([candidate.features for candidate in rows], dtype=np.float64)
        y = np.asarray([int(candidate.target) for candidate in rows], dtype=np.int64)
        scaler = StandardScaler().fit(x)
        model = LogisticRegression(
            class_weight="balanced",
            C=regularization_c,
            max_iter=2000,
            random_state=42,
        ).fit(scaler.transform(x), y)
        result[label] = {
            "feature_names": list(FEATURE_NAMES),
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "coef": model.coef_[0].tolist(),
            "intercept": float(model.intercept_[0]),
            "num_samples": len(rows),
            "num_positive": int(y.sum()),
            "num_negative": int((1 - y).sum()),
        }
    return result


def apply_models(
    candidates: Sequence[Candidate],
    models: dict[str, Any],
) -> dict[tuple[str, str, int], float]:
    scores: dict[tuple[str, str, int], float] = {}
    for candidate in candidates:
        model = models[candidate.label]
        values = np.asarray(candidate.features, dtype=np.float64)
        mean = np.asarray(model["mean"], dtype=np.float64)
        scale = np.asarray(model["scale"], dtype=np.float64)
        coef = np.asarray(model["coef"], dtype=np.float64)
        standardized = (values - mean) / np.where(scale == 0.0, 1.0, scale)
        scores[candidate.key] = sigmoid(float(np.dot(coef, standardized) + model["intercept"]))
    return scores


def select_thresholds(
    candidates: Sequence[Candidate],
    scores: dict[tuple[str, str, int], float],
    gt_by_video: dict[str, dict[str, list[float]]],
    labels: Sequence[str],
    recall_floors: dict[str, float],
    *,
    nms_radius_sec: float,
    match_tolerance_sec: float,
    max_thresholds: int = 501,
) -> tuple[dict[str, float], dict[str, Any]]:
    thresholds: dict[str, float] = {}
    selected_metrics: dict[str, Any] = {}
    for label in labels:
        values = np.asarray(
            sorted({scores[candidate.key] for candidate in candidates if candidate.label == label}),
            dtype=np.float64,
        )
        if len(values) > max_thresholds:
            indices = np.linspace(0, len(values) - 1, max_thresholds).round().astype(int)
            values = values[indices]
        best: dict[str, Any] | None = None
        for threshold in values:
            metrics = evaluate_predictions(
                candidates,
                scores,
                {item: float(threshold) if item == label else 2.0 for item in labels},
                gt_by_video,
                labels,
                nms_radius_sec=nms_radius_sec,
                match_tolerance_sec=match_tolerance_sec,
                use_peak_time=True,
            )["per_class"][label]
            if metrics["recall"] + 1e-12 < recall_floors[label]:
                continue
            candidate_result = {"threshold": float(threshold), **metrics}
            if best is None or (
                candidate_result["precision"],
                candidate_result["f1"],
                candidate_result["threshold"],
            ) > (best["precision"], best["f1"], best["threshold"]):
                best = candidate_result
        if best is None:
            raise RuntimeError(
                f"No threshold satisfies recall floor for {label}: {recall_floors[label]:.4f}"
            )
        thresholds[label] = float(best["threshold"])
        selected_metrics[label] = best
    return thresholds, selected_metrics


def clip_scores(candidates: Sequence[Candidate]) -> dict[tuple[str, str, int], float]:
    return {candidate.key: candidate.clip_prob for candidate in candidates}


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def candidate_rows(
    candidates: Sequence[Candidate],
    scores: dict[tuple[str, str, int], float] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        row = {
            "video_id": candidate.video_id,
            "label": candidate.label,
            "window_index": candidate.window_index,
            "center_time_sec": candidate.center_time_sec,
            "peak_time_sec": candidate.peak_time_sec,
            "clip_prob": candidate.clip_prob,
            "target": "" if candidate.target is None else candidate.target,
            "nearest_gt_distance_sec": (
                ""
                if candidate.nearest_gt_distance_sec is None
                else candidate.nearest_gt_distance_sec
            ),
        }
        row.update(dict(zip(FEATURE_NAMES, candidate.features)))
        if scores is not None:
            row["rerank_score"] = scores[candidate.key]
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a per-class frame-aware PointNMS candidate reranker."
    )
    parser.add_argument("--calibration-run-dir", required=True)
    parser.add_argument("--calibration-video-id-file", default="")
    parser.add_argument("--holdout-run-dir", default="")
    parser.add_argument("--holdout-video-id-file", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--labels", default="shot,save")
    parser.add_argument("--baseline-thresholds", default="shot=0.15,save=0.20")
    parser.add_argument("--prefilter-thresholds", default="shot=0.02,save=0.02")
    parser.add_argument("--match-tolerance-sec", type=float, default=5.0)
    parser.add_argument("--ignore-radius-sec", type=float, default=8.0)
    parser.add_argument("--neighbor-radius-sec", type=float, default=5.1)
    parser.add_argument("--nms-radius-sec", type=float, default=5.0)
    parser.add_argument("--recall-drop-tolerance", type=float, default=0.01)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--regularization-c", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = tuple(item.strip() for item in args.labels.split(",") if item.strip())
    calibration_run_dir = Path(args.calibration_run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_thresholds = parse_label_floats(
        args.baseline_thresholds, DEFAULT_BASELINE_THRESHOLDS
    )
    support_thresholds = parse_label_floats(
        args.prefilter_thresholds, DEFAULT_PREFILTER_THRESHOLDS
    )
    calibration_video_ids = load_video_ids(
        calibration_run_dir, args.calibration_video_id_file
    )
    calibration_candidates = load_candidates(
        calibration_run_dir,
        calibration_video_ids,
        labels,
        match_tolerance_sec=args.match_tolerance_sec,
        ignore_radius_sec=args.ignore_radius_sec,
        neighbor_radius_sec=args.neighbor_radius_sec,
        support_thresholds=support_thresholds,
    )
    calibration_gt = collect_gt(calibration_run_dir, calibration_video_ids, labels)
    baseline_scores = clip_scores(calibration_candidates)
    baseline_center = evaluate_predictions(
        calibration_candidates,
        baseline_scores,
        baseline_thresholds,
        calibration_gt,
        labels,
        nms_radius_sec=args.nms_radius_sec,
        match_tolerance_sec=args.match_tolerance_sec,
        use_peak_time=False,
    )
    baseline_peak = evaluate_predictions(
        calibration_candidates,
        baseline_scores,
        baseline_thresholds,
        calibration_gt,
        labels,
        nms_radius_sec=args.nms_radius_sec,
        match_tolerance_sec=args.match_tolerance_sec,
        use_peak_time=True,
    )
    recall_floors = {
        label: max(
            0.0,
            float(baseline_center["per_class"][label]["recall"])
            - args.recall_drop_tolerance,
        )
        for label in labels
    }

    oof_scores, fold_info = fit_oof_models(
        calibration_candidates,
        labels,
        folds=args.folds,
        regularization_c=args.regularization_c,
    )
    final_models = fit_final_models(
        calibration_candidates,
        labels,
        regularization_c=args.regularization_c,
    )
    thresholds, selected_oof = select_thresholds(
        calibration_candidates,
        oof_scores,
        calibration_gt,
        labels,
        recall_floors,
        nms_radius_sec=args.nms_radius_sec,
        match_tolerance_sec=args.match_tolerance_sec,
    )
    oof_metrics = evaluate_predictions(
        calibration_candidates,
        oof_scores,
        thresholds,
        calibration_gt,
        labels,
        nms_radius_sec=args.nms_radius_sec,
        match_tolerance_sec=args.match_tolerance_sec,
        use_peak_time=True,
    )

    artifact = {
        "version": 1,
        "labels": list(labels),
        "feature_names": list(FEATURE_NAMES),
        "models": final_models,
        "thresholds": thresholds,
        "baseline_thresholds": {
            label: baseline_thresholds[label] for label in labels
        },
        "support_thresholds": {
            label: support_thresholds[label] for label in labels
        },
        "match_tolerance_sec": args.match_tolerance_sec,
        "ignore_radius_sec": args.ignore_radius_sec,
        "neighbor_radius_sec": args.neighbor_radius_sec,
        "nms_radius_sec": args.nms_radius_sec,
        "recall_drop_tolerance": args.recall_drop_tolerance,
        "calibration_video_ids": calibration_video_ids,
    }
    (output_dir / "calibrator.json").write_text(
        json.dumps(artifact, indent=2) + "\n"
    )
    calibration_report = {
        "baseline_center": baseline_center,
        "baseline_frame_peak": baseline_peak,
        "recall_floors": recall_floors,
        "selected_oof": selected_oof,
        "oof_reranker": oof_metrics,
        "folds": fold_info,
    }
    (output_dir / "oof_metrics.json").write_text(
        json.dumps(calibration_report, indent=2) + "\n"
    )
    write_csv(
        output_dir / "candidate_features.csv",
        candidate_rows(calibration_candidates),
    )
    write_csv(
        output_dir / "oof_predictions.csv",
        candidate_rows(calibration_candidates, oof_scores),
    )
    write_csv(output_dir / "oof_per_video_metrics.csv", oof_metrics["per_video"])

    holdout_report = None
    if args.holdout_run_dir:
        holdout_run_dir = Path(args.holdout_run_dir)
        holdout_video_ids = load_video_ids(
            holdout_run_dir, args.holdout_video_id_file
        )
        overlap = sorted(set(calibration_video_ids) & set(holdout_video_ids))
        if overlap:
            raise ValueError(
                f"Calibration/holdout video leakage detected; first={overlap[:10]}"
            )
        holdout_candidates = load_candidates(
            holdout_run_dir,
            holdout_video_ids,
            labels,
            match_tolerance_sec=args.match_tolerance_sec,
            ignore_radius_sec=args.ignore_radius_sec,
            neighbor_radius_sec=args.neighbor_radius_sec,
            support_thresholds=support_thresholds,
        )
        holdout_gt = collect_gt(holdout_run_dir, holdout_video_ids, labels)
        holdout_baseline_scores = clip_scores(holdout_candidates)
        holdout_scores = apply_models(holdout_candidates, final_models)
        holdout_report = {
            "video_ids": holdout_video_ids,
            "baseline_center": evaluate_predictions(
                holdout_candidates,
                holdout_baseline_scores,
                baseline_thresholds,
                holdout_gt,
                labels,
                nms_radius_sec=args.nms_radius_sec,
                match_tolerance_sec=args.match_tolerance_sec,
                use_peak_time=False,
            ),
            "baseline_frame_peak": evaluate_predictions(
                holdout_candidates,
                holdout_baseline_scores,
                baseline_thresholds,
                holdout_gt,
                labels,
                nms_radius_sec=args.nms_radius_sec,
                match_tolerance_sec=args.match_tolerance_sec,
                use_peak_time=True,
            ),
            "reranker": evaluate_predictions(
                holdout_candidates,
                holdout_scores,
                thresholds,
                holdout_gt,
                labels,
                nms_radius_sec=args.nms_radius_sec,
                match_tolerance_sec=args.match_tolerance_sec,
                use_peak_time=True,
            ),
        }
        (output_dir / "holdout_metrics.json").write_text(
            json.dumps(holdout_report, indent=2) + "\n"
        )
        write_csv(
            output_dir / "holdout_predictions.csv",
            candidate_rows(holdout_candidates, holdout_scores),
        )
        write_csv(
            output_dir / "holdout_per_video_metrics.csv",
            holdout_report["reranker"]["per_video"],
        )

    for label in labels:
        baseline = baseline_center["per_class"][label]
        reranked = oof_metrics["per_class"][label]
        print(
            f"OOF {label}: baseline P/R={baseline['precision']:.4f}/{baseline['recall']:.4f} "
            f"rerank P/R={reranked['precision']:.4f}/{reranked['recall']:.4f} "
            f"threshold={thresholds[label]:.6f}",
            flush=True,
        )
    if holdout_report is not None:
        for label in labels:
            baseline = holdout_report["baseline_center"]["per_class"][label]
            reranked = holdout_report["reranker"]["per_class"][label]
            print(
                f"HOLDOUT {label}: baseline P/R={baseline['precision']:.4f}/{baseline['recall']:.4f} "
                f"rerank P/R={reranked['precision']:.4f}/{reranked['recall']:.4f}",
                flush=True,
            )
    print(f"wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
