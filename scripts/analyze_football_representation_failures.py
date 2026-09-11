#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_football_events as train_mod
from scripts.eval_long_video_checkpoint import (
    SlidingWindowVideoDataset,
    WindowRecord,
    collate_windows,
    load_checkpoint_model,
)


DEFAULT_LABELS = ("shot", "save", "set_piece")
REPRESENTATIONS = (
    "backbone_mean",
    "backbone_event",
    "projected_mean",
    "projected_event",
    "temporal",
)


@dataclass(frozen=True)
class Case:
    video_id: str
    label: str
    group: str
    window_index: int
    start_sec: float
    end_sec: float
    probability: float
    threshold: float
    nearest_gt_sec: float | None
    nearest_gt_distance_sec: float | None
    other_events: str


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_labels(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def find_video(video_root: Path, video_id: str) -> Path:
    for suffix in (".mp4", ".mov", ".mkv", ".avi"):
        path = video_root / f"{video_id}{suffix}"
        if path.exists():
            return path
    candidates = sorted(video_root.glob(f"{video_id}.*"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"Missing video_id={video_id} under {video_root}")


def load_exclusions(run_dir: Path) -> set[tuple[str, str]]:
    path = run_dir / "per_video_per_class.csv"
    if not path.exists():
        return set()
    return {
        (row["video_id"], row["label"])
        for row in read_csv(path)
        if str(row.get("excluded", "")).strip().lower() in {"1", "true", "yes"}
    }


def load_thresholds(run_dir: Path, video_ids: Sequence[str], labels: Sequence[str]) -> dict[str, float]:
    for video_id in video_ids:
        path = run_dir / video_id / "summary.json"
        if path.exists():
            data = json.loads(path.read_text())
            return {label: float(data["thresholds"][label]) for label in labels}
    raise FileNotFoundError(f"No per-video summary.json found under {run_dir}")


def interval_distance(time_sec: float, start_sec: float, end_sec: float) -> float:
    if start_sec <= time_sec <= end_sec:
        return 0.0
    return min(abs(time_sec - start_sec), abs(time_sec - end_sec))


def temporal_nms(rows: Sequence[dict[str, Any]], min_gap_sec: float) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: float(item["probability"]), reverse=True):
        center = 0.5 * (float(row["start_sec"]) + float(row["end_sec"]))
        if any(abs(center - float(item["center_sec"])) < min_gap_sec for item in kept):
            continue
        item = dict(row)
        item["center_sec"] = center
        kept.append(item)
    return kept


def round_robin_limit(cases: Sequence[Case], limit: int) -> list[Case]:
    if limit <= 0 or len(cases) <= limit:
        return list(cases)
    queues: dict[str, list[Case]] = defaultdict(list)
    for case in cases:
        queues[case.video_id].append(case)
    result: list[Case] = []
    while len(result) < limit and any(queues.values()):
        for video_id in sorted(queues):
            if queues[video_id] and len(result) < limit:
                result.append(queues[video_id].pop(0))
    return result


def build_cases(
    run_dir: Path,
    labels: Sequence[str],
    tolerance_sec: float,
    max_per_group: int,
    min_gap_sec: float,
) -> tuple[list[Case], dict[str, Any]]:
    run_config = json.loads((run_dir / "run_config.json").read_text())
    video_ids = [str(item) for item in run_config["video_ids"]]
    thresholds = load_thresholds(run_dir, video_ids, labels)
    exclusions = load_exclusions(run_dir)
    candidates: dict[tuple[str, str], list[Case]] = defaultdict(list)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for video_id in video_ids:
        video_dir = run_dir / video_id
        windows = read_csv(video_dir / "window_predictions.csv")
        gt_rows = read_csv(video_dir / "gt_events.csv")
        gt_by_label = {
            label: [float(row["time_sec"]) for row in gt_rows if row["label"] == label]
            for label in labels
        }
        for label in labels:
            if (video_id, label) in exclusions:
                continue
            threshold = thresholds[label]
            gt_times = gt_by_label[label]
            other_gt = [
                (row["label"], float(row["time_sec"]))
                for row in gt_rows
                if row["label"] != label
            ]
            tp_by_gt: dict[int, dict[str, Any]] = {}
            fp_rows: list[dict[str, Any]] = []
            tn_rows: list[dict[str, Any]] = []
            positive_windows_by_gt: dict[int, list[dict[str, Any]]] = defaultdict(list)
            all_windows_by_gt: dict[int, list[dict[str, Any]]] = defaultdict(list)

            for window in windows:
                start_sec = float(window["start_sec"])
                end_sec = float(window["end_sec"])
                probability = float(window[f"prob_{label}"])
                matches = [
                    index
                    for index, gt_time in enumerate(gt_times)
                    if start_sec - tolerance_sec <= gt_time <= end_sec + tolerance_sec
                ]
                row = {
                    "window_index": int(window["index"]),
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "probability": probability,
                }
                for gt_index in matches:
                    all_windows_by_gt[gt_index].append(row)
                if probability >= threshold and matches:
                    nearest = min(matches, key=lambda index: abs(gt_times[index] - 0.5 * (start_sec + end_sec)))
                    positive_windows_by_gt[nearest].append(row)
                    previous = tp_by_gt.get(nearest)
                    if previous is None or probability > float(previous["probability"]):
                        tp_by_gt[nearest] = row
                elif probability >= threshold:
                    fp_rows.append(row)
                elif not matches:
                    tn_rows.append(row)

            for gt_index, row in tp_by_gt.items():
                gt_time = gt_times[gt_index]
                candidates[(label, "TP")].append(
                    Case(
                        video_id=video_id,
                        label=label,
                        group="TP",
                        nearest_gt_sec=gt_time,
                        nearest_gt_distance_sec=interval_distance(
                            gt_time, float(row["start_sec"]), float(row["end_sec"])
                        ),
                        threshold=threshold,
                        other_events="",
                        **row,
                    )
                )

            for row in temporal_nms(fp_rows, min_gap_sec):
                start_sec = float(row["start_sec"])
                end_sec = float(row["end_sec"])
                nearby_other = sorted(
                    {
                        event_label
                        for event_label, event_time in other_gt
                        if start_sec - tolerance_sec <= event_time <= end_sec + tolerance_sec
                    }
                )
                nearest_gt = min(gt_times, key=lambda value: interval_distance(value, start_sec, end_sec), default=None)
                nearest_distance = (
                    interval_distance(nearest_gt, start_sec, end_sec)
                    if nearest_gt is not None
                    else None
                )
                candidates[(label, "FP")].append(
                    Case(
                        video_id=video_id,
                        label=label,
                        group="FP",
                        window_index=int(row["window_index"]),
                        start_sec=start_sec,
                        end_sec=end_sec,
                        probability=float(row["probability"]),
                        threshold=threshold,
                        nearest_gt_sec=nearest_gt,
                        nearest_gt_distance_sec=nearest_distance,
                        other_events="+".join(nearby_other),
                    )
                )

            for row in temporal_nms(tn_rows, min_gap_sec):
                candidates[(label, "hard_TN")].append(
                    Case(
                        video_id=video_id,
                        label=label,
                        group="hard_TN",
                        window_index=int(row["window_index"]),
                        start_sec=float(row["start_sec"]),
                        end_sec=float(row["end_sec"]),
                        probability=float(row["probability"]),
                        threshold=threshold,
                        nearest_gt_sec=None,
                        nearest_gt_distance_sec=None,
                        other_events="",
                    )
                )

            for gt_index, gt_time in enumerate(gt_times):
                if positive_windows_by_gt.get(gt_index):
                    continue
                covering = all_windows_by_gt.get(gt_index, [])
                if not covering:
                    continue
                row = max(covering, key=lambda item: float(item["probability"]))
                candidates[(label, "FN")].append(
                    Case(
                        video_id=video_id,
                        label=label,
                        group="FN",
                        nearest_gt_sec=gt_time,
                        nearest_gt_distance_sec=interval_distance(
                            gt_time, float(row["start_sec"]), float(row["end_sec"])
                        ),
                        threshold=threshold,
                        other_events="",
                        **row,
                    )
                )

    selected: list[Case] = []
    for (label, group), group_cases in sorted(candidates.items()):
        reverse = group in {"TP", "FP", "hard_TN"}
        group_cases = sorted(group_cases, key=lambda item: item.probability, reverse=reverse)
        limited = round_robin_limit(group_cases, max_per_group)
        selected.extend(limited)
        counts[label][group] = len(limited)

    return selected, {
        "video_ids": video_ids,
        "thresholds": thresholds,
        "exclusions": [list(item) for item in sorted(exclusions)],
        "selected_counts": {label: dict(groups) for label, groups in counts.items()},
    }


def normalized_weighted_mean(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.clamp_min(1e-5)
    return (features * weights.unsqueeze(-1)).sum(dim=1) / weights.sum(dim=1, keepdim=True)


def unique_windows_by_video(cases: Sequence[Case]) -> dict[str, list[WindowRecord]]:
    result: dict[str, dict[int, WindowRecord]] = defaultdict(dict)
    for case in cases:
        result[case.video_id][case.window_index] = WindowRecord(
            index=case.window_index,
            start_sec=case.start_sec,
            end_sec=case.end_sec,
        )
    return {
        video_id: [windows[index] for index in sorted(windows)]
        for video_id, windows in result.items()
    }


def extract_representations(
    model: torch.nn.Module,
    cfg: Any,
    labels: Sequence[str],
    cases: Sequence[Case],
    video_root: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    image_size = train_mod.parse_image_size(cfg.video.image_size)
    num_frames = train_mod.effective_num_frames(cfg)
    case_lookup: dict[tuple[str, int], list[int]] = defaultdict(list)
    for case_index, case in enumerate(cases):
        case_lookup[(case.video_id, case.window_index)].append(case_index)
    features: dict[str, list[np.ndarray | None]] = {
        name: [None] * len(cases) for name in REPRESENTATIONS
    }
    detail_rows: list[dict[str, Any] | None] = [None] * len(cases)
    label_to_index = {label: index for index, label in enumerate(labels)}

    model.eval()
    autocast_enabled = device.type == "cuda"
    for video_id, windows in unique_windows_by_video(cases).items():
        dataset = SlidingWindowVideoDataset(
            video_path=str(find_video(video_root, video_id)),
            video_id=video_id,
            windows=windows,
            num_frames=num_frames,
            image_size=image_size,
            normalize_on_cpu=True,
            decode_strategy="single_seek",
            video_reader_cache_size=1,
            window_cropper=None,
            view_mode="single",
            global_image_size=image_size,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=num_workers > 0,
            collate_fn=collate_windows,
        )
        with torch.inference_mode():
            for batch in loader:
                inputs = batch["inputs"].to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=autocast_enabled,
                ):
                    raw = model.encode_frames(inputs)
                    projected = model.frame_proj(raw)
                    outputs = model._projected_branch_outputs(
                        projected,
                        model.frame_event_head,
                        model.temporal,
                        model.head,
                    )
                frame_probs = torch.sigmoid(outputs["frame_event_logits"].float())
                logits = outputs["logits"].float()
                temporal = outputs["temporal"].float()
                for batch_index, meta in enumerate(batch["meta"]):
                    key = (video_id, int(meta["index"]))
                    for case_index in case_lookup[key]:
                        case = cases[case_index]
                        label_index = label_to_index[case.label]
                        weights = frame_probs[batch_index : batch_index + 1, :, label_index]
                        raw_item = raw[batch_index : batch_index + 1].float()
                        projected_item = projected[batch_index : batch_index + 1].float()
                        values = {
                            "backbone_mean": raw_item.mean(dim=1)[0],
                            "backbone_event": normalized_weighted_mean(raw_item, weights)[0],
                            "projected_mean": projected_item.mean(dim=1)[0],
                            "projected_event": normalized_weighted_mean(projected_item, weights)[0],
                            "temporal": temporal[batch_index],
                        }
                        for name, value in values.items():
                            features[name][case_index] = value.detach().cpu().numpy().astype(np.float32)
                        probs = frame_probs[batch_index, :, label_index]
                        detail_rows[case_index] = {
                            **asdict(case),
                            "model_probability": float(torch.sigmoid(logits[batch_index, label_index]).cpu()),
                            "frame_prob_max": float(probs.max().cpu()),
                            "frame_prob_mean": float(probs.mean().cpu()),
                            "frame_prob_top4_mean": float(torch.topk(probs, k=min(4, probs.numel())).values.mean().cpu()),
                            "frame_prob_std": float(probs.std(unbiased=False).cpu()),
                        }
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"features video={video_id} windows={len(windows)}", flush=True)

    if any(item is None for values in features.values() for item in values):
        raise RuntimeError("Some selected cases were not assigned representations")
    if any(item is None for item in detail_rows):
        raise RuntimeError("Some selected cases were not assigned detail rows")
    arrays = {
        name: np.stack([item for item in values if item is not None])
        for name, values in features.items()
    }
    return arrays, [item for item in detail_rows if item is not None]


def binary_auc(targets: np.ndarray, scores: np.ndarray) -> float:
    targets = targets.astype(np.int64)
    positives = int(targets.sum())
    negatives = int(len(targets) - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    rank_sum = float(ranks[targets == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def cosine_pair_stats(features: np.ndarray, targets: np.ndarray, videos: np.ndarray) -> dict[str, float]:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    normalized = features / np.maximum(norms, 1e-8)
    similarities = normalized @ normalized.T
    groups: dict[str, list[float]] = defaultdict(list)
    for left in range(len(features)):
        for right in range(left + 1, len(features)):
            if videos[left] == videos[right]:
                continue
            if targets[left] == 1 and targets[right] == 1:
                key = "tp_tp"
            elif targets[left] == 0 and targets[right] == 0:
                key = "fp_fp"
            else:
                key = "tp_fp"
            groups[key].append(float(similarities[left, right]))
    means = {
        key: float(np.mean(groups.get(key, [float("nan")])))
        for key in ("tp_tp", "fp_fp", "tp_fp")
    }
    means["cosine_cluster_gap"] = 0.5 * (means["tp_tp"] + means["fp_fp"]) - means["tp_fp"]
    return means


def fit_fold_ridge_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    max_components: int,
    ridge: float,
) -> np.ndarray:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    train = (train_x - mean) / np.maximum(std, 1e-5)
    test = (test_x - mean) / np.maximum(std, 1e-5)
    _, _, vh = np.linalg.svd(train, full_matrices=False)
    components = min(max_components, max(len(train) - 2, 1), vh.shape[0])
    basis = vh[:components].T
    train = train @ basis
    test = test @ basis
    train = np.concatenate([train, np.ones((len(train), 1), dtype=train.dtype)], axis=1)
    test = np.concatenate([test, np.ones((len(test), 1), dtype=test.dtype)], axis=1)
    target = train_y.astype(np.float64) * 2.0 - 1.0
    gram = train @ train.T
    alpha = np.linalg.solve(gram + ridge * np.eye(len(train)), target)
    weights = train.T @ alpha
    return test @ weights


def grouped_probe(
    features: np.ndarray,
    targets: np.ndarray,
    videos: np.ndarray,
    max_components: int,
    ridge: float,
) -> dict[str, float]:
    scores = np.full(len(features), np.nan, dtype=np.float64)
    folds = 0
    for video_id in sorted(set(videos.tolist())):
        test_mask = videos == video_id
        train_mask = ~test_mask
        if len(np.unique(targets[train_mask])) < 2 or not test_mask.any():
            continue
        scores[test_mask] = fit_fold_ridge_probe(
            features[train_mask], targets[train_mask], features[test_mask], max_components, ridge
        )
        folds += 1
    valid = np.isfinite(scores)
    predictions = scores[valid] >= 0.0
    positive_recall = float(predictions[targets[valid] == 1].mean()) if np.any(targets[valid] == 1) else float("nan")
    negative_recall = float((~predictions[targets[valid] == 0]).mean()) if np.any(targets[valid] == 0) else float("nan")
    return {
        "probe_auc": binary_auc(targets[valid], scores[valid]),
        "probe_balanced_accuracy": 0.5 * (positive_recall + negative_recall),
        "probe_tp_recall": positive_recall,
        "probe_fp_rejection": negative_recall,
        "probe_folds": folds,
        "probe_samples": int(valid.sum()),
    }


def representation_report(
    features: dict[str, np.ndarray],
    rows: Sequence[dict[str, Any]],
    labels: Sequence[str],
    max_components: int,
    ridge: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    report_rows: list[dict[str, Any]] = []
    diagnosis: dict[str, Any] = {}
    row_labels = np.asarray([row["label"] for row in rows])
    row_groups = np.asarray([row["group"] for row in rows])
    row_videos = np.asarray([row["video_id"] for row in rows])

    for label in labels:
        mask = (row_labels == label) & np.isin(row_groups, ["TP", "FP"])
        targets = (row_groups[mask] == "TP").astype(np.int64)
        videos = row_videos[mask]
        label_results: dict[str, dict[str, Any]] = {}
        for name, all_features in features.items():
            values = all_features[mask]
            if len(np.unique(targets)) < 2:
                continue
            pair = cosine_pair_stats(values, targets, videos)
            probe = grouped_probe(values, targets, videos, max_components, ridge)
            tp = values[targets == 1]
            fp = values[targets == 0]
            between = float(np.linalg.norm(tp.mean(axis=0) - fp.mean(axis=0)))
            within = 0.5 * (
                float(np.mean(np.linalg.norm(tp - tp.mean(axis=0), axis=1)))
                + float(np.mean(np.linalg.norm(fp - fp.mean(axis=0), axis=1)))
            )
            result = {
                "label": label,
                "representation": name,
                "num_tp": int((targets == 1).sum()),
                "num_fp": int((targets == 0).sum()),
                "centroid_distance": between,
                "mean_within_distance": within,
                "fisher_distance_ratio": between / max(within, 1e-8),
                **pair,
                **probe,
            }
            report_rows.append(result)
            label_results[name] = result

        score_mask = mask
        score_targets = (row_groups[score_mask] == "TP").astype(np.int64)
        diagnostics = {
            key: binary_auc(score_targets, np.asarray([float(row[key]) for row in rows])[score_mask])
            for key in ("model_probability", "frame_prob_max", "frame_prob_top4_mean", "frame_prob_mean")
        }
        backbone_auc = label_results.get("backbone_event", {}).get("probe_auc", float("nan"))
        projected_auc = label_results.get("projected_event", {}).get("probe_auc", float("nan"))
        temporal_auc = label_results.get("temporal", {}).get("probe_auc", float("nan"))
        score_auc = diagnostics["model_probability"]
        if math.isfinite(backbone_auc) and backbone_auc < 0.65:
            finding = "backbone_event_features_weak"
        elif math.isfinite(temporal_auc) and math.isfinite(projected_auc) and temporal_auc + 0.05 < projected_auc:
            finding = "temporal_aggregation_loses_separation"
        elif math.isfinite(temporal_auc) and math.isfinite(score_auc) and temporal_auc >= 0.72 and score_auc + 0.08 < temporal_auc:
            finding = "classifier_objective_or_calibration_bottleneck"
        else:
            finding = "overlapping_features_multiple_bottlenecks"
        diagnosis[label] = {
            "finding": finding,
            "score_auc": diagnostics,
            "backbone_event_probe_auc": backbone_auc,
            "projected_event_probe_auc": projected_auc,
            "temporal_probe_auc": temporal_auc,
        }
    return report_rows, diagnosis


class DetectionIndex:
    def __init__(self, path: Path):
        data = torch.load(path, map_location="cpu", weights_only=False)
        self.frame_ids = np.asarray(data["frame_ids"], dtype=np.int64)
        self.frame_offsets = np.asarray(data["frame_offsets"], dtype=np.int64)
        self.boxes = np.asarray(data["boxes"], dtype=np.float32)
        self.classes = np.asarray(data["classes"], dtype=np.int64)
        self.confidences = np.asarray(data["confidences"], dtype=np.float32)
        self.fps = float(data.get("fps", 0.0) or 0.0)
        size = data.get("image_size", {})
        self.width = float(size.get("width", 1.0))
        self.height = float(size.get("height", 1.0))

    def objects(self, frame_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        position = int(np.searchsorted(self.frame_ids, frame_id))
        candidates = [value for value in (position - 1, position) if 0 <= value < len(self.frame_ids)]
        if not candidates:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.int64), np.zeros(0, np.float32)
        nearest = min(candidates, key=lambda value: abs(int(self.frame_ids[value]) - frame_id))
        start, end = int(self.frame_offsets[nearest]), int(self.frame_offsets[nearest + 1])
        return self.boxes[start:end], self.classes[start:end], self.confidences[start:end]


def inverse_normalize(frame: torch.Tensor) -> np.ndarray:
    mean = train_mod.IMAGENET_MEAN.reshape(3, 1, 1)
    std = train_mod.IMAGENET_STD.reshape(3, 1, 1)
    rgb = (frame.detach().cpu() * std + mean).clamp(0, 1)
    rgb = (rgb.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def normalize_map(values: np.ndarray) -> np.ndarray:
    values = np.maximum(values, 0.0)
    high = float(np.percentile(values, 99.0))
    if high <= 1e-12:
        high = float(values.max())
    return np.clip(values / max(high, 1e-12), 0.0, 1.0).astype(np.float32)


def overlay_heatmap(frame: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)
    color = cv2.applyColorMap(np.uint8(np.clip(heatmap, 0, 1) * 255), cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 0.55, color, 0.45, 0)


def object_masks(
    index: DetectionIndex | None,
    frame_id: int,
    height: int,
    width: int,
) -> dict[str, np.ndarray]:
    masks = {name: np.zeros((height, width), dtype=np.uint8) for name in ("person", "ball", "goal")}
    if index is None:
        return masks
    boxes, classes, confidences = index.objects(frame_id)
    thresholds = {0: 0.30, 1: 0.05, 2: 0.20}
    names = {0: "person", 1: "ball", 2: "goal"}
    for box, class_id, confidence in zip(boxes, classes, confidences):
        class_id = int(class_id)
        if class_id not in names or float(confidence) < thresholds[class_id]:
            continue
        x1, y1, x2, y2 = map(float, box)
        x1, x2 = x1 / index.width * width, x2 / index.width * width
        y1, y2 = y1 / index.height * height, y2 / index.height * height
        if class_id == 1:
            center_x, center_y = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
            box_w, box_h = max(x2 - x1, width * 0.025), max(y2 - y1, height * 0.04)
            x1, x2 = center_x - box_w, center_x + box_w
            y1, y2 = center_y - box_h, center_y + box_h
        left = int(np.clip(math.floor(x1), 0, width - 1))
        top = int(np.clip(math.floor(y1), 0, height - 1))
        right = int(np.clip(math.ceil(x2), left + 1, width))
        bottom = int(np.clip(math.ceil(y2), top + 1, height))
        masks[names[class_id]][top:bottom, left:right] = 1
    return masks


def saliency_alignment(saliency: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, float]:
    saliency = np.maximum(saliency.astype(np.float64), 0.0)
    total = max(float(saliency.sum()), 1e-12)
    result: dict[str, float] = {}
    for name, mask in masks.items():
        area_fraction = float(mask.mean())
        mass_fraction = float(saliency[mask > 0].sum() / total) if mask.any() else 0.0
        result[f"{name}_area_fraction"] = area_fraction
        result[f"{name}_saliency_fraction"] = mass_fraction
        result[f"{name}_enrichment"] = mass_fraction / max(area_fraction, 1e-8) if mask.any() else 0.0
    return result


def draw_header(image: np.ndarray, text: str) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(image, text[:125], (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)


def render_attributions(
    model: torch.nn.Module,
    cfg: Any,
    labels: Sequence[str],
    cases: Sequence[Case],
    video_root: Path,
    roi_index_root: Path | None,
    output_dir: Path,
    device: torch.device,
    per_group: int,
    top_frames: int,
) -> list[dict[str, Any]]:
    if per_group <= 0:
        return []
    selected: list[Case] = []
    for label in labels:
        for group in ("TP", "FP"):
            group_cases = [case for case in cases if case.label == label and case.group == group]
            group_cases.sort(key=lambda item: item.probability, reverse=True)
            selected.extend(round_robin_limit(group_cases, per_group))
    image_size = train_mod.parse_image_size(cfg.video.image_size)
    num_frames = train_mod.effective_num_frames(cfg)
    label_to_index = {label: index for index, label in enumerate(labels)}
    index_cache: dict[str, DetectionIndex | None] = {}
    rows: list[dict[str, Any]] = []
    model.eval()

    for case_number, case in enumerate(selected, start=1):
        video_path = find_video(video_root, case.video_id)
        frames, frame_times = train_mod.read_video_segment(
            str(video_path),
            num_frames,
            image_size,
            False,
            0.0,
            start_sec=case.start_sec,
            end_sec=case.end_sec,
            normalize=True,
            decode_strategy="single_seek",
            return_frame_times=True,
        )
        inputs = frames.unsqueeze(0).to(device)
        label_index = label_to_index[case.label]
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            raw = model.encode_frames(inputs)
            projected = model.frame_proj(raw)
        token_input = projected.detach().float().requires_grad_(True)
        outputs = model._projected_branch_outputs(
            token_input, model.frame_event_head, model.temporal, model.head
        )
        clip_logit = outputs["logits"][0, label_index]
        token_gradient = torch.autograd.grad(clip_logit, token_input)[0][0].detach()
        signed_temporal = (token_input.detach()[0] * token_gradient).sum(dim=-1)
        temporal_sensitivity = token_gradient.norm(dim=-1)
        ranking = signed_temporal.clamp_min(0)
        if float(ranking.sum()) <= 1e-8:
            ranking = temporal_sensitivity
        selected_frames = torch.topk(ranking, k=min(top_frames, num_frames)).indices.cpu().tolist()

        if case.video_id not in index_cache:
            index_path = roi_index_root / f"{case.video_id}.pt" if roi_index_root is not None else None
            index_cache[case.video_id] = DetectionIndex(index_path) if index_path is not None and index_path.exists() else None
        detection_index = index_cache[case.video_id]
        cells: list[np.ndarray] = []
        for rank, frame_index in enumerate(selected_frames, start=1):
            frame_input = inputs[:, frame_index].detach().clone().requires_grad_(True)
            with torch.enable_grad():
                single_raw = model.encode_frames(frame_input.unsqueeze(1))[:, 0]
                single_projected = model.frame_proj(single_raw)
                scalar = (single_projected.float() * token_gradient[frame_index].reshape(1, -1)).sum()
                gradient = torch.autograd.grad(scalar, frame_input)[0][0]
            signed = (gradient * frame_input[0]).sum(dim=0).detach().float().cpu().numpy()
            sensitivity = gradient.abs().mean(dim=0).detach().float().cpu().numpy()
            positive = np.maximum(signed, 0.0)
            saliency = positive if float(positive.sum()) > 1e-10 else sensitivity
            normalized = normalize_map(saliency)
            frame_bgr = inverse_normalize(frame_input[0])
            overlay = overlay_heatmap(frame_bgr, normalized)
            frame_time = float(frame_times[frame_index])
            frame_id = int(
                round(frame_time * detection_index.fps)
                if detection_index is not None and detection_index.fps > 0
                else 0
            )
            masks = object_masks(detection_index, frame_id, saliency.shape[0], saliency.shape[1])
            alignment = saliency_alignment(saliency, masks)
            for name, mask in masks.items():
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                color = {"person": (0, 255, 0), "ball": (0, 255, 255), "goal": (255, 128, 0)}[name]
                scale_x = overlay.shape[1] / max(mask.shape[1], 1)
                scale_y = overlay.shape[0] / max(mask.shape[0], 1)
                for contour in contours:
                    x, y, width, height = cv2.boundingRect(contour)
                    cv2.rectangle(
                        overlay,
                        (int(x * scale_x), int(y * scale_y)),
                        (int((x + width) * scale_x), int((y + height) * scale_y)),
                        color,
                        1,
                    )
            header = (
                f"{case.group} {case.label} p={case.probability:.3f} rank={rank} "
                f"t={frame_time:.2f} temporal={float(signed_temporal[frame_index]):+.3f}"
            )
            draw_header(frame_bgr, header)
            draw_header(overlay, "class-logit attribution | green=person yellow=ball blue=goal")
            cells.append(np.concatenate([frame_bgr, overlay], axis=1))
            rows.append(
                {
                    **asdict(case),
                    "frame_rank": rank,
                    "frame_index": frame_index,
                    "frame_time_sec": frame_time,
                    "temporal_signed_contribution": float(signed_temporal[frame_index]),
                    "temporal_gradient_norm": float(temporal_sensitivity[frame_index]),
                    **alignment,
                }
            )
            del frame_input, gradient, single_raw, single_projected, scalar

        sheet = np.concatenate(cells, axis=0)
        relative = Path(case.group.lower()) / case.label / (
            f"{case.group.lower()}_{case.label}_{case.video_id}_w{case.window_index:05d}.jpg"
        )
        path = output_dir / "attributions" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
        for row in rows[-len(selected_frames) :]:
            row["image_path"] = relative.as_posix()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"attribution {case_number}/{len(selected)} {case.group} {case.label} {case.video_id}", flush=True)
    return rows


def summarize_alignment(rows: Sequence[dict[str, Any]], labels: Sequence[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for label in labels:
        for group in ("TP", "FP"):
            selected = [row for row in rows if row["label"] == label and row["group"] == group]
            if not selected:
                continue
            result.append(
                {
                    "label": label,
                    "group": group,
                    "num_frames": len(selected),
                    **{
                        f"mean_{name}_enrichment": float(np.mean([float(row[f"{name}_enrichment"]) for row in selected]))
                        for name in ("person", "ball", "goal")
                    },
                    "mean_temporal_signed_contribution": float(
                        np.mean([float(row["temporal_signed_contribution"]) for row in selected])
                    ),
                }
            )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose whether football FP errors originate in DINO features, temporal aggregation, or class attention."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="", help="Defaults to run_config.json checkpoint.")
    parser.add_argument("--video-root", default="", help="Defaults to run_config.json video_root.")
    parser.add_argument("--roi-index-root", default="outputs/football_roi_indices/robust_v2")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--match-tolerance-sec", type=float, default=-1.0, help="Negative uses run_config value.")
    parser.add_argument("--max-per-group", type=int, default=36)
    parser.add_argument("--min-fp-gap-sec", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--probe-components", type=int, default=32)
    parser.add_argument("--probe-ridge", type=float, default=1.0)
    parser.add_argument("--attribution-per-group", type=int, default=3)
    parser.add_argument("--attribution-top-frames", type=int, default=3)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = json.loads((run_dir / "run_config.json").read_text())
    checkpoint = args.checkpoint or str(run_config["checkpoint"])
    video_root = Path(args.video_root or run_config["video_root"]).expanduser().resolve()
    roi_index_root = Path(args.roi_index_root).expanduser().resolve() if args.roi_index_root else None
    labels = parse_labels(args.labels)
    tolerance_sec = (
        float(args.match_tolerance_sec)
        if args.match_tolerance_sec >= 0
        else float(run_config.get("match_tolerance_sec", 5.0))
    )
    device = torch.device(args.device)

    cases, case_summary = build_cases(
        run_dir,
        labels,
        tolerance_sec,
        args.max_per_group,
        args.min_fp_gap_sec,
    )
    write_csv(output_dir / "cases.csv", [asdict(case) for case in cases])
    print(json.dumps(case_summary["selected_counts"], ensure_ascii=False), flush=True)
    model, cfg, checkpoint_labels, _ = load_checkpoint_model(checkpoint, device, [])
    if labels != [label for label in checkpoint_labels if label in labels]:
        missing = [label for label in labels if label not in checkpoint_labels]
        if missing:
            raise ValueError(f"Requested labels missing from checkpoint: {missing}; checkpoint={checkpoint_labels}")
    model.eval()

    features, detail_rows = extract_representations(
        model,
        cfg,
        checkpoint_labels,
        cases,
        video_root,
        device,
        args.batch_size,
        args.num_workers,
    )
    write_csv(output_dir / "case_scores.csv", detail_rows)
    np.savez_compressed(output_dir / "representations.npz", **features)
    report_rows, diagnosis = representation_report(
        features,
        detail_rows,
        labels,
        args.probe_components,
        args.probe_ridge,
    )
    write_csv(output_dir / "representation_report.csv", report_rows)

    attribution_rows = render_attributions(
        model,
        cfg,
        checkpoint_labels,
        cases,
        video_root,
        roi_index_root,
        output_dir,
        device,
        args.attribution_per_group,
        args.attribution_top_frames,
    )
    write_csv(output_dir / "attribution_frames.csv", attribution_rows)
    alignment_rows = summarize_alignment(attribution_rows, labels)
    write_csv(output_dir / "attribution_alignment_summary.csv", alignment_rows)
    summary = {
        "run_dir": str(run_dir),
        "checkpoint": checkpoint,
        "video_root": str(video_root),
        "roi_index_root": str(roi_index_root) if roi_index_root is not None else "",
        "match_tolerance_sec": tolerance_sec,
        "case_selection": case_summary,
        "diagnosis": diagnosis,
        "attribution_alignment": alignment_rows,
        "interpretation": {
            "backbone_event_features_weak": "Even class-weighted DINO frame features cannot separate TP from FP across held-out videos.",
            "temporal_aggregation_loses_separation": "Projected frame evidence is separable, but the temporal embedding discards part of it.",
            "classifier_objective_or_calibration_bottleneck": "Temporal embeddings are separable, but the trained class score does not expose that separation.",
            "overlapping_features_multiple_bottlenecks": "TP and FP remain substantially overlapping; inspect attribution and FP subtypes before changing the network.",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({"diagnosis": diagnosis, "attribution_alignment": alignment_rows}, ensure_ascii=False, indent=2))
    print(f"wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
