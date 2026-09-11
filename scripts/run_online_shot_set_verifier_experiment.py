#!/usr/bin/env python
"""Video-grouped OOF experiment for the dense online shot set verifier.

This is the leakage-safe scalar/context ablation before high-resolution visual
features are added.  It consumes the exact dense validation cache, trains only
on other videos in each fold, emits NMS-free event slots, tunes one global OOF
threshold at the requested recall floor, and reports review-time union.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "football_e2e_spotter" / "src"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from football_e2e_spotter.online_verifier import (  # noqa: E402
    CandidateContextSetVerifier,
    ShotTarget,
    decode_shot_slots,
    shot_set_verifier_loss,
)


FEATURE_NAMES = (
    "clip_logit_shot", "clip_logit_save", "clip_logit_set_piece",
    "frame_logit_shot", "frame_logit_save", "frame_logit_set_piece",
    "joint_logit_shot", "joint_logit_save", "joint_logit_set_piece",
    "shot_video_robust_z", "shot_neighbor_delta", "shot_local_prominence",
    "peak_offset_norm", "window_center_in_core",
)


def logit(values: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    clipped = np.clip(values.astype(np.float64), eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


@dataclass(frozen=True)
class CoreExample:
    video_id: str
    core_start: float
    duration: float
    features: np.ndarray
    candidate_times_normalized: np.ndarray
    targets: tuple[float, ...]
    supervision_weights: np.ndarray
    visual_features: np.ndarray | None = None


class CoreDataset(Dataset):
    def __init__(
        self,
        examples: Sequence[CoreExample],
        mean: np.ndarray,
        scale: np.ndarray,
    ) -> None:
        self.examples = list(examples)
        self.mean = mean.astype(np.float32)
        self.scale = np.maximum(scale.astype(np.float32), 1e-5)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        features = (example.features.astype(np.float32) - self.mean) / self.scale
        return {
            "features": torch.from_numpy(features),
            "baseline_logits": torch.from_numpy(
                example.features[:, 6].astype(np.float32)
            ),
            "supervision_weights": torch.from_numpy(
                example.supervision_weights.astype(np.float32)
            ),
            "times": torch.from_numpy(example.candidate_times_normalized.astype(np.float32)),
            "visual": (
                None if example.visual_features is None
                else torch.from_numpy(example.visual_features.astype(np.float32))
            ),
            "targets": [ShotTarget(value) for value in example.targets],
            "meta": example,
        }


def collate(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    max_candidates = max(item["features"].shape[0] for item in batch)
    feature_dim = batch[0]["features"].shape[1]
    features = torch.zeros(len(batch), max_candidates, feature_dim)
    times = torch.zeros(len(batch), max_candidates)
    baseline_logits = torch.zeros(len(batch), max_candidates)
    supervision_weights = torch.zeros(len(batch), max_candidates)
    mask = torch.zeros(len(batch), max_candidates, dtype=torch.bool)
    visual = None
    if batch[0]["visual"] is not None:
        visual = torch.zeros(
            len(batch), max_candidates, batch[0]["visual"].shape[-1]
        )
    for index, item in enumerate(batch):
        count = item["features"].shape[0]
        features[index, :count] = item["features"]
        times[index, :count] = item["times"]
        baseline_logits[index, :count] = item["baseline_logits"]
        supervision_weights[index, :count] = item["supervision_weights"]
        mask[index, :count] = True
        if visual is not None:
            visual[index, :count] = item["visual"]
    return {
        "features": features,
        "times": times,
        "baseline_logits": baseline_logits,
        "supervision_weights": supervision_weights,
        "mask": mask,
        "visual": visual,
        "targets": [item["targets"] for item in batch],
        "meta": [item["meta"] for item in batch],
    }


def load_examples(
    cache_path: Path,
    metadata_path: Path,
    *,
    core_sec: float,
    context_sec: float,
    visual_features: np.ndarray | None = None,
    ambiguity_negative_weight: float = 1.0,
) -> tuple[list[CoreExample], dict[str, list[float]], dict[str, float]]:
    cache = np.load(cache_path, allow_pickle=True)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if len(metadata) != len(cache["video_ids"]):
        raise ValueError("prediction metadata and arrays have different lengths")
    labels = [str(value) for value in cache["labels"]]
    shot_index = labels.index("shot")
    video_ids = cache["video_ids"].astype(str)
    starts = cache["clip_starts"].astype(np.float64)
    ends = cache["clip_ends"].astype(np.float64)
    candidate_times = cache["candidate_times"].astype(np.float64)
    clip_probs = cache["probs"].astype(np.float64)
    frame_probs = cache["frame_peak_probs"].astype(np.float64)
    joint_probs = cache["online_probs"].astype(np.float64)
    clip_logits, frame_logits, joint_logits = (
        logit(clip_probs), logit(frame_probs), logit(joint_probs)
    )
    durations = {
        video_id: float(ends[video_ids == video_id].max())
        for video_id in sorted(set(video_ids.tolist()))
    }
    gt_by_video: dict[str, list[float]] = {}
    auxiliary_by_video: dict[str, list[float]] = {}
    for index, video_id in enumerate(video_ids):
        if video_id not in gt_by_video:
            anchors = metadata[index].get("online_gt_anchors", ())
            gt_by_video[video_id] = sorted(float(value) for value in anchors[shot_index])
            auxiliary_by_video[video_id] = sorted(
                float(value)
                for label_index, values in enumerate(anchors)
                if label_index != shot_index
                for value in values
            )

    examples: list[CoreExample] = []
    for video_id in sorted(durations):
        rows = np.where(video_ids == video_id)[0]
        order = rows[np.argsort(starts[rows])]
        shot_values = joint_logits[order, shot_index]
        median = float(np.median(shot_values))
        mad = float(np.median(np.abs(shot_values - median)))
        robust_scale = max(1.4826 * mad, 0.1)
        centers = 0.5 * (starts[order] + ends[order])
        for core_start in np.arange(0.0, durations[video_id], core_sec):
            core_end = min(core_start + core_sec, durations[video_id])
            context_start, context_end = core_start - context_sec, core_end + context_sec
            keep_position = np.where((centers >= context_start) & (centers < context_end))[0]
            if not len(keep_position):
                continue
            selected = order[keep_position]
            local_shot = joint_logits[selected, shot_index]
            neighbor_mean = np.empty(len(selected), dtype=np.float64)
            prominence = np.empty(len(selected), dtype=np.float64)
            for local_index, row_index in enumerate(selected):
                distances = np.abs(centers[keep_position] - centers[keep_position[local_index]])
                neighbor_mask = (distances > 1e-6) & (distances <= 10.1)
                neighbor = local_shot[neighbor_mask]
                neighbor_mean[local_index] = float(neighbor.mean()) if len(neighbor) else local_shot[local_index]
                prominence[local_index] = local_shot[local_index] - (
                    float(neighbor.max()) if len(neighbor) else local_shot[local_index]
                )
            window_center_norm = (0.5 * (starts[selected] + ends[selected]) - core_start) / core_sec
            peak_offset = (
                candidate_times[selected, shot_index] - 0.5 * (starts[selected] + ends[selected])
            ) / np.maximum(0.5 * (ends[selected] - starts[selected]), 1e-5)
            features = np.column_stack((
                clip_logits[selected], frame_logits[selected], joint_logits[selected],
                (local_shot - median) / robust_scale,
                local_shot - neighbor_mean,
                prominence,
                peak_offset,
                window_center_norm,
            )).astype(np.float32)
            selected_times = candidate_times[selected, shot_index]
            supervision_weights = np.ones(len(selected), dtype=np.float32)
            for candidate_index, candidate_time in enumerate(selected_times):
                near_auxiliary = any(
                    abs(float(candidate_time) - value) <= 4.0
                    for value in auxiliary_by_video[video_id]
                )
                near_shot = any(
                    abs(float(candidate_time) - value) <= 3.0
                    for value in gt_by_video[video_id]
                )
                if near_auxiliary and not near_shot:
                    supervision_weights[candidate_index] = float(
                        ambiguity_negative_weight
                    )
            targets = tuple(
                (value - core_start) / core_sec
                for value in gt_by_video[video_id]
                if core_start <= value < core_end
            )
            examples.append(CoreExample(
                video_id=video_id,
                core_start=float(core_start),
                duration=float(core_end - core_start),
                features=features,
                candidate_times_normalized=(
                    (candidate_times[selected, shot_index] - core_start) / core_sec
                ).astype(np.float32),
                targets=targets,
                supervision_weights=supervision_weights,
                visual_features=(
                    None if visual_features is None
                    else visual_features[selected].astype(np.float16, copy=False)
                ),
            ))
    return examples, gt_by_video, durations


def load_visual_features(shard_dir: Path, rows: int) -> np.ndarray:
    paths = sorted(shard_dir.glob("visual_rank*_of_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no visual feature shards in {shard_dir}")
    result: np.ndarray | None = None
    filled = np.zeros(rows, dtype=bool)
    for path in paths:
        shard = np.load(path, allow_pickle=False)
        indices = shard["indices"].astype(np.int64)
        values = shard["visual_features"].astype(np.float16, copy=False)
        offsets = shard["offsets"].astype(np.float32)
        if values.ndim != 3 or values.shape[1] != len(offsets):
            raise ValueError(f"unexpected visual feature shape {values.shape}")
        if result is None:
            result = np.empty((rows, 4 * values.shape[-1]), dtype=np.float16)
        center_index = int(np.argmin(np.abs(offsets)))
        short = np.abs(offsets) <= 1.001
        before, after = offsets < 0.0, offsets > 0.0
        summarized = np.concatenate((
            values[:, center_index],
            values[:, short].mean(axis=1),
            values.mean(axis=1),
            values[:, after].mean(axis=1) - values[:, before].mean(axis=1),
        ), axis=1).astype(np.float16)
        if filled[indices].any():
            raise ValueError(f"duplicate visual rows in {path}")
        result[indices] = summarized
        filled[indices] = True
    if result is None or not filled.all():
        missing = np.where(~filled)[0][:10].tolist()
        raise ValueError(f"visual shards do not cover all rows; examples={missing}")
    return result


def feature_stats(examples: Sequence[CoreExample]) -> tuple[np.ndarray, np.ndarray]:
    values = np.concatenate([example.features for example in examples], axis=0)
    return values.mean(0), np.maximum(values.std(0), 1e-5)


def train_fold(
    train_examples: Sequence[CoreExample],
    val_examples: Sequence[CoreExample],
    *,
    epochs: int,
    batch_size: int,
    device: torch.device,
    seed: int,
    output_path: Path,
) -> list[dict[str, float | str | int]]:
    torch.manual_seed(seed)
    mean, scale = feature_stats(train_examples)
    train_loader = DataLoader(
        CoreDataset(train_examples, mean, scale), batch_size=batch_size,
        shuffle=True, collate_fn=collate,
    )
    val_loader = DataLoader(
        CoreDataset(val_examples, mean, scale), batch_size=batch_size,
        shuffle=False, collate_fn=collate,
    )
    model = CandidateContextSetVerifier(
        len(FEATURE_NAMES),
        visual_dim=(
            0 if train_examples[0].visual_features is None
            else train_examples[0].visual_features.shape[-1]
        ),
        hidden_dim=128, num_slots=8,
        encoder_layers=2, decoder_layers=2, heads=8, dropout=0.1,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.02)
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            output = model(
                batch["features"].to(device), batch["times"].to(device),
                batch["mask"].to(device),
                None if batch["visual"] is None else batch["visual"].to(device),
                baseline_logits=batch["baseline_logits"].to(device),
            )
            loss = shot_set_verifier_loss(
                output, batch["targets"],
                supervision_weights=batch["supervision_weights"].to(device),
            )["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "mean": mean, "scale": scale,
        "feature_names": FEATURE_NAMES, "epochs": epochs,
    }, output_path)
    model.eval()
    predictions: list[dict[str, float | str | int]] = []
    with torch.no_grad():
        for batch in val_loader:
            output = model(
                batch["features"].to(device), batch["times"].to(device),
                batch["mask"].to(device),
                None if batch["visual"] is None else batch["visual"].to(device),
                baseline_logits=batch["baseline_logits"].to(device),
            )
            for index, example in enumerate(batch["meta"]):
                sliced = {key: value[index:index + 1] for key, value in output.items()}
                for row in decode_shot_slots(
                    sliced, core_start_sec=example.core_start,
                    core_duration_sec=30.0, threshold=0.0,
                ):
                    predictions.append({"video_id": example.video_id, **row})
    return predictions


def one_to_one(
    predictions: Sequence[dict[str, Any]],
    gt_by_video: dict[str, list[float]],
    threshold: float,
    tolerance: float,
) -> dict[str, float | int]:
    tp = fp = fn = 0
    selected_by_video: dict[str, list[dict[str, Any]]] = {}
    for row in predictions:
        if float(row["score"]) >= threshold:
            selected_by_video.setdefault(str(row["video_id"]), []).append(row)
    for video_id, targets in gt_by_video.items():
        available = list(targets)
        selected = sorted(
            selected_by_video.get(video_id, ()),
            key=lambda row: float(row["score"]), reverse=True,
        )
        for row in selected:
            eligible = [
                index for index, target in enumerate(available)
                if abs(float(row["time_sec"]) - target) <= tolerance
            ]
            if eligible:
                best = min(eligible, key=lambda index: abs(float(row["time_sec"]) - available[index]))
                available.pop(best)
                tp += 1
            else:
                fp += 1
        fn += len(available)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "tp": tp, "fp": fp, "fn": fn, "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "threshold": float(threshold),
    }


def choose_threshold(
    predictions: Sequence[dict[str, Any]],
    gt_by_video: dict[str, list[float]],
    recall_floor: float,
    tolerance: float,
    grid_size: int,
) -> tuple[float, dict[str, Any]]:
    scores = np.asarray([float(row["score"]) for row in predictions])
    values = np.unique(np.quantile(
        scores, np.linspace(0.0, 1.0, min(max(grid_size, 2), len(scores)))
    ))
    rows = [one_to_one(predictions, gt_by_video, float(value), tolerance) for value in values]
    ceiling = max(float(row["recall"]) for row in rows)
    effective = min(recall_floor, ceiling)
    feasible = [row for row in rows if float(row["recall"]) + 1e-12 >= effective]
    best = max(feasible, key=lambda row: (float(row["precision"]), float(row["f1"]), float(row["threshold"])))
    best["candidate_recall_ceiling"] = ceiling
    best["requested_recall_floor"] = recall_floor
    best["recall_floor_reachable"] = ceiling + 1e-12 >= recall_floor
    return float(best["threshold"]), best


def merge_duration(intervals: Sequence[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    total = 0.0
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def workload(
    predictions: Sequence[dict[str, Any]],
    durations: dict[str, float],
    threshold: float,
    review_sec: float = 10.0,
) -> dict[str, float | int]:
    union = 0.0
    selected_count = 0
    half = 0.5 * review_sec
    for video_id, duration in durations.items():
        intervals = []
        for row in predictions:
            if row["video_id"] != video_id or float(row["score"]) < threshold:
                continue
            selected_count += 1
            time_sec = float(row["time_sec"])
            intervals.append((max(time_sec - half, 0.0), min(time_sec + half, duration)))
        union += merge_duration(intervals)
    total = sum(durations.values())
    return {
        "selected_slots": selected_count,
        "review_union_minutes": union / 60.0,
        "total_video_minutes": total / 60.0,
        "participation_ratio": union / max(total, 1e-12),
    }


def baseline_predictions(cache_path: Path) -> list[dict[str, Any]]:
    cache = np.load(cache_path, allow_pickle=True)
    labels = [str(value) for value in cache["labels"]]
    shot_index = labels.index("shot")
    return [
        {
            "video_id": str(cache["video_ids"][index]),
            "time_sec": float(cache["candidate_times"][index, shot_index]),
            "score": float(cache["online_probs"][index, shot_index]),
            "slot_index": index,
        }
        for index in range(len(cache["video_ids"]))
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--recall-floor", type=float, default=0.90)
    parser.add_argument("--tolerance-sec", type=float, default=3.0)
    parser.add_argument("--threshold-grid-size", type=int, default=201)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--visual-shard-dir", type=Path)
    parser.add_argument("--external-cache", type=Path)
    parser.add_argument("--external-metadata", type=Path)
    parser.add_argument("--external-visual-shard-dir", type=Path)
    parser.add_argument("--ambiguity-negative-weight", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    with np.load(args.cache, allow_pickle=True) as cache:
        cache_rows = len(cache["video_ids"])
    visual_features = (
        None if args.visual_shard_dir is None
        else load_visual_features(args.visual_shard_dir, cache_rows)
    )
    examples, gt_by_video, durations = load_examples(
        args.cache, args.metadata, core_sec=30.0, context_sec=15.0,
        visual_features=visual_features,
        ambiguity_negative_weight=args.ambiguity_negative_weight,
    )
    videos = np.asarray([example.video_id for example in examples])
    unique_videos = np.unique(videos)
    split_count = min(max(args.folds, 2), len(unique_videos))
    args.output.mkdir(parents=True, exist_ok=True)
    oof_predictions: list[dict[str, Any]] = []
    fold_report = []
    splitter = GroupKFold(n_splits=split_count)
    for fold, (train_indices, val_indices) in enumerate(
        splitter.split(np.arange(len(examples)), groups=videos)
    ):
        train_examples = [examples[int(index)] for index in train_indices]
        val_examples = [examples[int(index)] for index in val_indices]
        predictions = train_fold(
            train_examples, val_examples, epochs=args.epochs,
            batch_size=args.batch_size, device=torch.device(args.device),
            seed=42 + fold, output_path=args.output / f"fold_{fold}.pt",
        )
        oof_predictions.extend(predictions)
        fold_report.append({
            "fold": fold,
            "train_videos": sorted(set(example.video_id for example in train_examples)),
            "val_videos": sorted(set(example.video_id for example in val_examples)),
            "train_cores": len(train_examples), "val_cores": len(val_examples),
        })
        print(json.dumps({"fold": fold, "predictions": len(predictions)}), flush=True)

    threshold, verifier_metrics = choose_threshold(
        oof_predictions, gt_by_video, args.recall_floor, args.tolerance_sec,
        args.threshold_grid_size,
    )
    baseline = baseline_predictions(args.cache)
    baseline_threshold, baseline_metrics = choose_threshold(
        baseline, gt_by_video, args.recall_floor, args.tolerance_sec,
        args.threshold_grid_size,
    )
    report = {
        "protocol": {
            "video_grouped_oof": True,
            "no_temporal_nms": True,
            "core_sec": 30.0,
            "context_sec": 15.0,
            "tolerance_sec": args.tolerance_sec,
            "recall_floor": args.recall_floor,
            "feature_names": FEATURE_NAMES,
            "visual_features": visual_features is not None,
            "visual_feature_dim": (
                0 if visual_features is None else visual_features.shape[-1]
            ),
            "ambiguity_negative_weight": args.ambiguity_negative_weight,
        },
        "data": {
            "videos": len(unique_videos), "cores": len(examples),
            "shot_gt": sum(len(values) for values in gt_by_video.values()),
        },
        "folds": fold_report,
        "baseline": {
            "metrics": baseline_metrics,
            "workload": workload(baseline, durations, baseline_threshold),
        },
        "set_verifier_oof": {
            "metrics": verifier_metrics,
            "workload": workload(oof_predictions, durations, threshold),
        },
    }
    if args.external_cache is not None:
        if args.external_metadata is None:
            raise ValueError("--external-metadata is required with --external-cache")
        with np.load(args.external_cache, allow_pickle=True) as external_cache:
            external_rows = len(external_cache["video_ids"])
        external_visual = (
            None if args.external_visual_shard_dir is None
            else load_visual_features(args.external_visual_shard_dir, external_rows)
        )
        external_examples, external_gt, external_durations = load_examples(
            args.external_cache, args.external_metadata,
            core_sec=30.0, context_sec=15.0,
            visual_features=external_visual,
            ambiguity_negative_weight=args.ambiguity_negative_weight,
        )
        overlap = sorted(
            set(example.video_id for example in examples)
            & set(example.video_id for example in external_examples)
        )
        if overlap:
            raise ValueError(f"calibration/external video leakage: {overlap}")
        final_predictions = train_fold(
            examples, external_examples, epochs=args.epochs,
            batch_size=args.batch_size, device=torch.device(args.device),
            seed=2026, output_path=args.output / "final_all_calibration.pt",
        )
        external_baseline = baseline_predictions(args.external_cache)
        report["external_test"] = {
            "threshold_source": "calibration_oof_fixed",
            "videos": len(external_gt),
            "shot_gt": sum(len(values) for values in external_gt.values()),
            "baseline": {
                "metrics": one_to_one(
                    external_baseline, external_gt, baseline_threshold,
                    args.tolerance_sec,
                ),
                "workload": workload(
                    external_baseline, external_durations, baseline_threshold
                ),
            },
            "set_verifier": {
                "metrics": one_to_one(
                    final_predictions, external_gt, threshold,
                    args.tolerance_sec,
                ),
                "workload": workload(
                    final_predictions, external_durations, threshold
                ),
            },
        }
        (args.output / "external_predictions.jsonl").write_text(
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n"
                for row in final_predictions
            ), encoding="utf-8",
        )
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output / "oof_predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in oof_predictions),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
