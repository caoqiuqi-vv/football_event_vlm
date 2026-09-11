"""Full-timeline scanning and no-NMS evaluation for the goal retriever."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import Tensor

from .goal_annotations import LABELS, load_goal_annotations
from .goal_retriever import GoalLongContextRetriever


def load_timeline(base: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    paths = [base / name for name in ("timestamps.npy", "appearance.npy", "motion.npy", "audio.npy")]
    if all(path.is_file() for path in paths):
        return tuple(np.load(path, mmap_mode="r", allow_pickle=False) for path in paths)  # type: ignore[return-value]
    with np.load(base / "timeline.npz", allow_pickle=False) as payload:
        return tuple(np.asarray(payload[name]) for name in ("timestamps", "appearance", "motion", "audio"))  # type: ignore[return-value]


def intervals_union_seconds(intervals: Iterable[tuple[float, float]]) -> float:
    ordered = sorted((float(left), float(right)) for left, right in intervals if right > left)
    if not ordered:
        return 0.0
    total = 0.0
    left, right = ordered[0]
    for next_left, next_right in ordered[1:]:
        if next_left <= right:
            right = max(right, next_right)
        else:
            total += right - left
            left, right = next_left, next_right
    return total + right - left


def scan_video(
    model: GoalLongContextRetriever,
    *,
    feature_base: Path,
    video_id: str,
    device: torch.device,
    batch_size: int,
) -> tuple[list[dict], float]:
    _timestamps, appearance, motion, audio = load_timeline(feature_base)
    geometry = model.geometry
    duration = float(len(appearance))
    core_starts = np.arange(0.0, duration, geometry.core_seconds, dtype=np.float32)
    predictions: list[dict] = []
    for batch_begin in range(0, len(core_starts), batch_size):
        starts = core_starts[batch_begin:batch_begin + batch_size]
        appearance_batch, motion_batch, audio_batch, valid_batch = [], [], [], []
        for core_start in starts:
            context_start = float(core_start) - geometry.context_left_seconds
            source = np.floor(context_start + np.arange(geometry.context_steps)).astype(np.int64)
            valid = (source >= 0) & (source < len(appearance))
            clipped = np.clip(source, 0, max(len(appearance) - 1, 0))
            streams = [np.asarray(values[clipped], dtype=np.float32) for values in (appearance, motion, audio)]
            for stream in streams:
                stream[~valid] = 0.0
            appearance_batch.append(streams[0]); motion_batch.append(streams[1]); audio_batch.append(streams[2]); valid_batch.append(valid)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(
                torch.from_numpy(np.stack(appearance_batch)).to(device),
                torch.from_numpy(np.stack(motion_batch)).to(device),
                torch.from_numpy(np.stack(audio_batch)).to(device),
                torch.from_numpy(np.stack(valid_batch)).to(device),
            )
            probability = output["class_logits"].softmax(dim=-1)[..., :len(LABELS)]
            quality = output["quality_logit"].sigmoid().unsqueeze(-1)
            scores = (probability * quality.sqrt()).float().cpu().numpy()
            event_times = output["event_time"].float().cpu().numpy()
            durations = output["region_duration"].float().cpu().numpy()
            uncertainty = output["temporal_uncertainty"].float().cpu().numpy()
        for local_batch, core_start in enumerate(starts):
            for slot in range(scores.shape[1]):
                absolute_time = float(core_start + event_times[local_batch, slot])
                if not float(core_start) <= absolute_time < min(float(core_start + geometry.core_seconds), duration):
                    continue
                # Keep class hypotheses independently.  They are not merged or
                # suppressed; per-class calibration decides which are reviewed.
                for class_index, label in enumerate(LABELS):
                    predictions.append({
                        "video_id": video_id, "core_start": float(core_start), "slot": slot,
                        "label": label, "time": absolute_time,
                        "score": float(scores[local_batch, slot, class_index]),
                        "duration": float(durations[local_batch, slot]),
                        "uncertainty": float(uncertainty[local_batch, slot]),
                    })
    return predictions, duration


def greedy_prefix_curve(predictions: list[dict], gt: dict[str, list[float]], tolerance: float) -> list[dict]:
    ordered = sorted(predictions, key=lambda row: (-float(row["score"]), row["video_id"], row["time"]))
    matched = {video_id: np.zeros(len(times), dtype=bool) for video_id, times in gt.items()}
    total_gt = sum(len(times) for times in gt.values())
    tp = 0
    curve = []
    for rank, prediction in enumerate(ordered, start=1):
        video_id = str(prediction["video_id"])
        times = np.asarray(gt.get(video_id, []), dtype=np.float64)
        available = np.where(~matched.setdefault(video_id, np.zeros(len(times), dtype=bool)))[0]
        is_tp = False
        if len(available):
            errors = np.abs(times[available] - float(prediction["time"]))
            nearest = int(errors.argmin())
            if errors[nearest] <= tolerance:
                matched[video_id][available[nearest]] = True
                tp += 1
                is_tp = True
        curve.append({
            "threshold": float(prediction["score"]), "tp": tp, "fp": rank - tp,
            "precision": tp / rank, "recall": tp / max(total_gt, 1), "last_is_tp": is_tp,
        })
    return curve


def choose_threshold(predictions: list[dict], gt: dict[str, list[float]], recall_floor: float, tolerance: float = 3.0) -> dict:
    curve = greedy_prefix_curve(predictions, gt, tolerance)
    eligible = [row for row in curve if row["recall"] >= recall_floor]
    if eligible:
        chosen = max(eligible, key=lambda row: (row["precision"], row["threshold"]))
        reachable = True
    elif curve:
        chosen = max(curve, key=lambda row: (row["recall"], row["precision"]))
        reachable = False
    else:
        chosen = {"threshold": 1.0, "tp": 0, "fp": 0, "precision": 0.0, "recall": 0.0}
        reachable = False
    return {**chosen, "recall_floor": recall_floor, "recall_floor_reachable": reachable}


def evaluate_predictions(
    predictions: list[dict],
    *,
    manifest: str | Path,
    split: str,
    durations: dict[str, float],
    recall_floors: dict[str, float] | None = None,
) -> dict:
    recall_floors = recall_floors or {label: (0.97 if label == "shot" else 0.94) for label in LABELS}
    payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    manifest_items = {str(item["media_id"]): item for item in payload[split] if str(item["media_id"]) in durations}
    gt = {label: {} for label in LABELS}
    for video_id, item in manifest_items.items():
        events = load_goal_annotations(item["annotation_path"])
        for label in LABELS:
            gt[label][video_id] = [event.timestamp for event in events if event.accepted and event.label == label]
    selected = {}
    retained = []
    for label in LABELS:
        class_predictions = [row for row in predictions if row["label"] == label]
        selected[label] = choose_threshold(class_predictions, gt[label], recall_floors[label])
        threshold = selected[label]["threshold"]
        retained.extend(row for row in class_predictions if row["score"] >= threshold)
    intervals = {video_id: [] for video_id in durations}
    for row in retained:
        half = 0.5 * float(row["duration"])
        duration = durations[row["video_id"]]
        intervals[row["video_id"]].append((max(0.0, row["time"] - half), min(duration, row["time"] + half)))
    reviewed = {video_id: intervals_union_seconds(values) for video_id, values in intervals.items()}
    total_duration = sum(durations.values())
    report = {
        "protocol": {"no_nms": True, "one_to_one_tolerance_sec": 3.0, "review_interval_union": True},
        "classes": selected,
        "retained_candidates": len(retained),
        "review_seconds": sum(reviewed.values()),
        "total_video_seconds": total_duration,
        "review_ratio": sum(reviewed.values()) / max(total_duration, 1e-9),
        "per_video_review_ratio": {video_id: reviewed[video_id] / max(duration, 1e-9) for video_id, duration in durations.items()},
        "gate_pass": all(selected[label]["recall_floor_reachable"] for label in LABELS) and sum(reviewed.values()) / max(total_duration, 1e-9) < 0.35,
    }
    return report

