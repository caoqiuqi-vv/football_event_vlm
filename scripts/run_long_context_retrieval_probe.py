#!/usr/bin/env python
"""Video-grouped OOF probe for 20-second retrieval with 3-minute context.

The probe tests a single hypothesis before an expensive long-context visual
model is built: do surrounding Stage-1 trajectories contain transferable
information that improves event-region ranking?  It is deliberately a small
tabular model and must not be reported as an external-test final system.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_hierarchical_retrieval_ceiling import (  # noqa: E402
    block_metrics,
    choose_score_threshold,
    oracle,
)


SOURCES = ("probs", "frame_peak_probs", "online_probs")


def probability_logit(values: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), eps, 1.0 - eps)
    return np.log(values / (1.0 - values))


def robust_z(values: np.ndarray) -> np.ndarray:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(1.4826 * mad, 1e-3)
    return (values - median) / scale


def aggregate(values: np.ndarray) -> list[float]:
    if not len(values):
        return [0.0, 0.0, 0.0]
    ordered = np.sort(values)
    return [float(ordered[-1]), float(values.mean()), float(ordered[-min(2, len(ordered)):].mean())]


def build_sequences(cache: Any, metadata: list[dict[str, Any]], block_sec: float) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, dict[str, list[float]], dict[str, float], list[str]]:
    labels = [str(value) for value in cache["labels"]]
    shot_index = labels.index("shot")
    video_ids = cache["video_ids"].astype(str)
    starts = cache["clip_starts"].astype(np.float64)
    ends = cache["clip_ends"].astype(np.float64)
    centres = 0.5 * (starts + ends)
    durations = {
        video_id: float(ends[video_ids == video_id].max())
        for video_id in sorted(set(video_ids.tolist()))
    }
    gt: dict[str, list[float]] = {}
    for index, video_id in enumerate(video_ids):
        if video_id not in gt:
            gt[video_id] = sorted(float(value) for value in metadata[index]["online_gt_anchors"][shot_index])

    blocks: list[dict[str, Any]] = []
    raw_features: list[list[float]] = []
    feature_names: list[str] = []
    for source in SOURCES:
        for label in labels:
            feature_names.extend((f"{source}_{label}_max", f"{source}_{label}_mean", f"{source}_{label}_top2"))
    feature_names.extend(("window_count", "duration_fraction"))

    video_slices: dict[str, tuple[int, int]] = {}
    for video_id, duration in durations.items():
        begin = len(blocks)
        rows = np.where(video_ids == video_id)[0]
        for start in np.arange(0.0, duration, block_sec):
            end = min(start + block_sec, duration)
            selected = rows[(centres[rows] >= start) & (centres[rows] < end)]
            values: list[float] = []
            for source in SOURCES:
                source_values = cache[source]
                for label_index in range(len(labels)):
                    values.extend(aggregate(source_values[selected, label_index]))
            values.extend((float(len(selected)), float((end - start) / block_sec)))
            targets = [value for value in gt[video_id] if start <= value < end]
            baseline = float(cache["online_probs"][selected, shot_index].max()) if len(selected) else 0.0
            blocks.append({
                "video_id": video_id, "start": float(start), "end": float(end),
                "score": baseline, "gt_count": len(targets),
            })
            raw_features.append(values)
        video_slices[video_id] = (begin, len(blocks))

    base = np.asarray(raw_features, dtype=np.float64)
    features: list[np.ndarray] = [base]
    context_names: list[str] = []
    online_base = len(labels) * 3 * 2  # probs and frame_peak_probs precede online_probs
    online_max_indices = [online_base + label_index * 3 for label_index in range(len(labels))]
    context_values = np.zeros((len(blocks), len(labels) * 8 + 2), dtype=np.float64)
    for video_id, (begin, end) in video_slices.items():
        sequence = base[begin:end]
        shot_max = sequence[:, online_max_indices[shot_index]]
        video_z = robust_z(probability_logit(shot_max))
        rank = np.argsort(np.argsort(shot_max, kind="stable"), kind="stable") / max(len(shot_max) - 1, 1)
        for local_index in range(len(sequence)):
            output: list[float] = []
            for radius in (1, 3, 9):
                left = max(0, local_index - radius)
                right = min(len(sequence), local_index + radius + 1)
                neighbourhood = sequence[left:right][:, online_max_indices]
                output.extend(neighbourhood.max(axis=0).tolist())
                output.extend(neighbourhood.mean(axis=0).tolist())
            before = sequence[max(0, local_index - 3):local_index, online_max_indices]
            after = sequence[local_index + 1:min(len(sequence), local_index + 4), online_max_indices]
            for neighbourhood in (before, after):
                output.extend((neighbourhood.mean(axis=0) if len(neighbourhood) else sequence[local_index, online_max_indices]).tolist())
            output.extend((float(video_z[local_index]), float(rank[local_index])))
            context_values[begin + local_index] = output
    for radius in (1, 3, 9):
        for stat in ("max", "mean"):
            context_names.extend(f"context_r{radius}_{label}_{stat}" for label in labels)
    context_names.extend(f"before_r3_{label}_mean" for label in labels)
    context_names.extend(f"after_r3_{label}_mean" for label in labels)
    context_names.extend(("video_robust_z_shot", "video_rank_shot"))
    features.append(context_values)
    return blocks, np.concatenate(features, axis=1), np.asarray([block["gt_count"] > 0 for block in blocks], dtype=np.int64), gt, durations, feature_names + context_names


def oof_scores(features: np.ndarray, targets: np.ndarray, blocks: Sequence[dict[str, Any]], folds: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    groups = np.asarray([block["video_id"] for block in blocks])
    unique = np.unique(groups)
    folds = min(max(folds, 2), len(unique))
    result = np.zeros(len(blocks), dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    for fold, (train_indices, val_indices) in enumerate(GroupKFold(folds).split(features, targets, groups)):
        positives = max(int(targets[train_indices].sum()), 1)
        negatives = max(int(len(train_indices) - positives), 1)
        weights = np.ones(len(train_indices), dtype=np.float64)
        weights[targets[train_indices] > 0] = negatives / positives
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=160,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=100 + fold,
        )
        model.fit(features[train_indices], targets[train_indices], sample_weight=weights)
        result[val_indices] = model.predict_proba(features[val_indices])[:, 1]
        fold_rows.append({
            "fold": fold,
            "train_videos": sorted(set(groups[train_indices].tolist())),
            "val_videos": sorted(set(groups[val_indices].tolist())),
            "train_blocks": len(train_indices),
            "val_blocks": len(val_indices),
        })
    return result, fold_rows


def scored_blocks(blocks: Sequence[dict[str, Any]], scores: np.ndarray) -> list[dict[str, Any]]:
    return [{**block, "score": float(score)} for block, score in zip(blocks, scores)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--block-sec", type=float, default=20.0)
    parser.add_argument("--context-sec", type=float, default=180.0)
    parser.add_argument("--recall-floor", type=float, default=0.95)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--blend-grid", default="0,0.25,0.5,0.75,1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not math.isclose(args.context_sec / args.block_sec, 9.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("this probe currently requires context-sec = 9 * block-sec")
    cache = np.load(args.cache, allow_pickle=True)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    blocks, features, targets, gt, durations, feature_names = build_sequences(cache, metadata, args.block_sec)
    oof, fold_rows = oof_scores(features, targets, blocks, args.folds)
    baseline_scores = np.asarray([block["score"] for block in blocks], dtype=np.float64)
    baseline = choose_score_threshold(blocks, gt, durations, args.recall_floor)
    probe = choose_score_threshold(scored_blocks(blocks, oof), gt, durations, args.recall_floor)
    baseline_z = robust_z(probability_logit(baseline_scores))
    probe_z = robust_z(probability_logit(oof))
    blends = []
    for alpha in (float(value) for value in args.blend_grid.split(",") if value.strip()):
        scores = (1.0 - alpha) * baseline_z + alpha * probe_z
        metrics = choose_score_threshold(scored_blocks(blocks, scores), gt, durations, args.recall_floor)
        blends.append({"alpha": alpha, **metrics})
    selected = min(
        blends,
        key=lambda row: (
            not row["recall_floor_reachable"], row["coverage_ratio"],
            -row["block_precision"], -row["macro_video_recall"],
        ),
    )
    report = {
        "protocol": {
            "video_grouped_oof": True,
            "probe_only_not_external_final": True,
            "block_sec": args.block_sec,
            "context_sec_each_side": args.context_sec,
            "recall_floor": args.recall_floor,
            "no_nms": True,
        },
        "data": {
            "videos": len(gt), "blocks": len(blocks),
            "positive_blocks": int(targets.sum()),
            "shot_gt": sum(len(values) for values in gt.values()),
            "feature_count": features.shape[1],
        },
        "folds": fold_rows,
        "baseline": baseline,
        "long_context_probe_oof": probe,
        "blend_grid": blends,
        "selected_blend": selected,
        "oracle_lower_bound": oracle(blocks, gt, durations, args.recall_floor),
        "feature_names": feature_names,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
