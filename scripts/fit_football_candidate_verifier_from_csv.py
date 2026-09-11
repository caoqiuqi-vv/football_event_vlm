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
EXCLUDED_FEATURE_COLUMNS = {
    "source_run",
    "video_id",
    "window_index",
    "label",
    "start_sec",
    "end_sec",
    "center_sec",
    "candidate_threshold",
    "is_candidate",
    "is_gt_window",
    "is_tp_candidate",
    "is_hard_fp_candidate",
    "nearest_gt_distance_sec",
    "nearest_gt_event_id",
}


@dataclass(frozen=True)
class Row:
    video_id: str
    label: str
    window_index: int
    clip_prob: float
    target: int
    features: tuple[float, ...]

    @property
    def key(self) -> tuple[str, str, int]:
        return self.video_id, self.label, self.window_index


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
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


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def parse_label_floats(raw: str, labels: Sequence[str], default: float) -> dict[str, float]:
    values = {label: float(default) for label in labels}
    if not raw:
        return values
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Expected label=value item, got: {item}")
        label, value = item.split("=", 1)
        label = label.strip()
        if label not in values:
            raise ValueError(f"Unknown label in thresholds: {label}")
        values[label] = float(value)
    return values


def is_numeric_column(rows: Sequence[dict[str, str]], column: str) -> bool:
    seen = 0
    for row in rows:
        value = row.get(column, "")
        if value == "":
            continue
        try:
            float(value)
        except ValueError:
            return False
        seen += 1
    return seen > 0


def infer_feature_names(rows: Sequence[dict[str, str]]) -> list[str]:
    if not rows:
        return []
    result: list[str] = []
    for column in rows[0]:
        if column in EXCLUDED_FEATURE_COLUMNS:
            continue
        if "target" in column:
            continue
        if column.endswith("_time_sec") and column != "frame_peak_relative_time_sec":
            continue
        if column.endswith("_event_id"):
            continue
        if is_numeric_column(rows, column):
            result.append(column)
    if "clip_prob" not in result:
        raise ValueError("candidate CSV must contain clip_prob as a numeric feature")
    return result


def load_rows(
    csv_path: Path,
    labels: Sequence[str],
    feature_names: Sequence[str] | None,
    *,
    candidates_only: bool,
) -> tuple[list[Row], list[str]]:
    raw_rows = [
        row
        for row in read_csv(csv_path)
        if row.get("label") in labels
        and (not candidates_only or int(to_float(row.get("is_candidate"))) == 1)
    ]
    features = list(feature_names) if feature_names else infer_feature_names(raw_rows)
    rows: list[Row] = []
    for row in raw_rows:
        rows.append(
            Row(
                video_id=str(row["video_id"]),
                label=str(row["label"]),
                window_index=int(to_float(row["window_index"])),
                clip_prob=to_float(row.get("clip_prob")),
                target=int(to_float(row.get("is_gt_window"))),
                features=tuple(to_float(row.get(name)) for name in features),
            )
        )
    return rows, features


def metric(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def evaluate(
    rows: Sequence[Row],
    scores: dict[tuple[str, str, int], float],
    thresholds: dict[str, float],
    labels: Sequence[str],
) -> dict[str, Any]:
    per_class: dict[str, Any] = {}
    total_tp = total_fp = total_fn = 0
    for label in labels:
        label_rows = [row for row in rows if row.label == label]
        tp = fp = fn = 0
        for row in label_rows:
            selected = scores[row.key] >= thresholds[label]
            if selected and row.target:
                tp += 1
            elif selected and not row.target:
                fp += 1
            elif (not selected) and row.target:
                fn += 1
        per_class[label] = metric(tp, fp, fn)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    return {"per_class": per_class, "micro": metric(total_tp, total_fp, total_fn)}


def fit_oof(
    rows: Sequence[Row],
    labels: Sequence[str],
    *,
    folds: int,
    regularization_c: float,
) -> tuple[dict[tuple[str, str, int], float], dict[str, Any]]:
    scores: dict[tuple[str, str, int], float] = {}
    fold_info: dict[str, Any] = {}
    for label in labels:
        label_rows = [row for row in rows if row.label == label]
        groups = np.asarray([row.video_id for row in label_rows])
        unique_groups = np.unique(groups)
        split_count = min(max(2, folds), len(unique_groups))
        x = np.asarray([row.features for row in label_rows], dtype=np.float64)
        y = np.asarray([row.target for row in label_rows], dtype=np.int64)
        if len(np.unique(y)) < 2:
            raise ValueError(f"Need both positive and negative rows for label={label}")
        label_folds: list[dict[str, Any]] = []
        for fold_index, (train_indices, val_indices) in enumerate(
            GroupKFold(n_splits=split_count).split(x, y, groups)
        ):
            if len(np.unique(y[train_indices])) < 2:
                raise ValueError(f"Fold {fold_index} lacks both classes for label={label}")
            scaler = StandardScaler().fit(x[train_indices])
            model = LogisticRegression(
                class_weight="balanced",
                C=regularization_c,
                max_iter=2000,
                random_state=42,
            ).fit(scaler.transform(x[train_indices]), y[train_indices])
            pred = model.predict_proba(scaler.transform(x[val_indices]))[:, 1]
            for index, value in zip(val_indices, pred):
                scores[label_rows[int(index)].key] = float(value)
            label_folds.append(
                {
                    "fold": fold_index,
                    "train_videos": sorted(set(groups[train_indices].tolist())),
                    "val_videos": sorted(set(groups[val_indices].tolist())),
                }
            )
        fold_info[label] = label_folds
    return scores, fold_info


def fit_final(rows: Sequence[Row], labels: Sequence[str], feature_names: Sequence[str], regularization_c: float) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for label in labels:
        label_rows = [row for row in rows if row.label == label]
        x = np.asarray([row.features for row in label_rows], dtype=np.float64)
        y = np.asarray([row.target for row in label_rows], dtype=np.int64)
        scaler = StandardScaler().fit(x)
        model = LogisticRegression(
            class_weight="balanced",
            C=regularization_c,
            max_iter=2000,
            random_state=42,
        ).fit(scaler.transform(x), y)
        models[label] = {
            "feature_names": list(feature_names),
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "coef": model.coef_[0].tolist(),
            "intercept": float(model.intercept_[0]),
            "num_samples": len(label_rows),
            "num_positive": int(y.sum()),
            "num_negative": int((1 - y).sum()),
        }
    return models


def select_thresholds(
    rows: Sequence[Row],
    scores: dict[tuple[str, str, int], float],
    labels: Sequence[str],
    recall_floors: dict[str, float],
) -> tuple[dict[str, float], dict[str, Any]]:
    thresholds: dict[str, float] = {}
    selected: dict[str, Any] = {}
    for label in labels:
        values = sorted({scores[row.key] for row in rows if row.label == label}, reverse=True)
        if not values:
            raise ValueError(f"No scores for label={label}")
        best: dict[str, Any] | None = None
        for threshold in values:
            result = evaluate(rows, scores, {item: threshold if item == label else 2.0 for item in labels}, labels)["per_class"][label]
            if result["recall"] + 1e-12 < recall_floors[label]:
                continue
            candidate = {"threshold": float(threshold), **result}
            if best is None or (
                candidate["precision"],
                candidate["f1"],
                candidate["threshold"],
            ) > (best["precision"], best["f1"], best["threshold"]):
                best = candidate
        if best is None:
            raise RuntimeError(f"No verifier threshold satisfies recall floor for {label}")
        thresholds[label] = float(best["threshold"])
        selected[label] = best
    return thresholds, selected


def row_output(rows: Sequence[Row], scores: dict[tuple[str, str, int], float]) -> list[dict[str, Any]]:
    return [
        {
            "video_id": row.video_id,
            "label": row.label,
            "window_index": row.window_index,
            "clip_prob": row.clip_prob,
            "target": row.target,
            "verifier_score": scores[row.key],
        }
        for row in rows
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a lightweight candidate verifier from exported football candidate CSV.")
    parser.add_argument("--candidate-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--features", default="auto", help="auto or comma-separated feature names")
    parser.add_argument("--baseline-thresholds", default="", help="Optional label=value list; defaults to candidate_threshold from the CSV.")
    parser.add_argument("--recall-drop-tolerance", type=float, default=0.01)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--regularization-c", type=float, default=0.1)
    parser.add_argument("--candidates-only", action="store_true", help="Fit/evaluate only rows whose is_candidate=1.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = tuple(label.strip() for label in args.labels.split(",") if label.strip())
    feature_names = None if args.features == "auto" else [item.strip() for item in args.features.split(",") if item.strip()]
    rows, feature_names = load_rows(args.candidate_csv, labels, feature_names, candidates_only=args.candidates_only)
    if not rows:
        raise ValueError("No candidate rows loaded")
    csv_rows = read_csv(args.candidate_csv)
    threshold_defaults = {
        label: next(
            (
                to_float(row.get("candidate_threshold"))
                for row in csv_rows
                if row.get("label") == label and row.get("candidate_threshold") not in (None, "")
            ),
            0.1,
        )
        for label in labels
    }
    baseline_thresholds = threshold_defaults
    if args.baseline_thresholds:
        baseline_thresholds = parse_label_floats(args.baseline_thresholds, labels, 0.1)
    clip_scores = {row.key: row.clip_prob for row in rows}
    baseline = evaluate(rows, clip_scores, baseline_thresholds, labels)
    recall_floors = {
        label: max(0.0, baseline["per_class"][label]["recall"] - args.recall_drop_tolerance)
        for label in labels
    }
    oof_scores, folds = fit_oof(rows, labels, folds=args.folds, regularization_c=args.regularization_c)
    verifier_thresholds, selected = select_thresholds(rows, oof_scores, labels, recall_floors)
    verifier_metrics = evaluate(rows, oof_scores, verifier_thresholds, labels)
    models = fit_final(rows, labels, feature_names, args.regularization_c)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "candidate_csv": str(args.candidate_csv),
        "labels": list(labels),
        "feature_names": feature_names,
        "baseline_thresholds": baseline_thresholds,
        "verifier_thresholds": verifier_thresholds,
        "recall_floors": recall_floors,
        "baseline": baseline,
        "verifier_oof": verifier_metrics,
        "selected": selected,
        "folds": folds,
        "candidates_only": bool(args.candidates_only),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "verifier.json").write_text(
        json.dumps({"version": 1, "models": models, "thresholds": verifier_thresholds}, indent=2) + "\n"
    )
    write_csv(args.output_dir / "oof_predictions.csv", row_output(rows, oof_scores))
    for label in labels:
        base = baseline["per_class"][label]
        ver = verifier_metrics["per_class"][label]
        print(
            f"{label}: baseline P/R={base['precision']:.4f}/{base['recall']:.4f} "
            f"verifier P/R={ver['precision']:.4f}/{ver['recall']:.4f} "
            f"threshold={verifier_thresholds[label]:.6f}"
        )
    print(f"micro: baseline P/R={baseline['micro']['precision']:.4f}/{baseline['micro']['recall']:.4f} "
          f"verifier P/R={verifier_metrics['micro']['precision']:.4f}/{verifier_metrics['micro']['recall']:.4f}")
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
