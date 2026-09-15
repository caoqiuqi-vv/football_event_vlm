#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Downstream football event training for DINOv3 video clip classification.

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import sys
import threading
import time
import warnings
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Union

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support, roc_auc_score
from football_dual_token_fusion import (
    DualViewTokenFusionTransformer,
    FullRoiCrossAttentionFusion,
    bounded_asymmetric_residual,
    staggered_local_indices,
    staggered_multi_roi_indices,
)
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from dinov3.hub.backbones import dinov3_vitb16, dinov3_vith16plus, dinov3_vitl16
from football_detection_aware import (
    ROI_META_DIM,
    DetectionHintRenderer,
    ROIProposal,
    RobustClipCropper,
    invalid_roi,
)
from football_evidence_provider import EvidenceProvider
from football_online_consistency import online_pair_consistency_loss
from football_no_evidence import balanced_no_evidence_loss
from football_saliency_rank import (
    bag_saliency_rank_loss,
    cross_window_saliency_consistency_loss,
    signed_causal_local_evidence_loss,
)
from football_frame_heatmap import positive_core_context_heatmap_loss
from football_online_evaluation import tune_online_event_thresholds
from football_object_spatial_aux import (
    ObjectSpatialAuxHead,
    ObjectTeacherTargetProvider,
    object_teacher_heatmap_loss,
)
from football_events.ball_goal_relation import (
    RELATION_NAMES,
    BallGoalRelationTargetProvider,
    ball_goal_relation_aux_loss,
)
from football_online_object_teacher import OnlineObjectTeacherTargeter
from football_events.mechanism_a import (
    gradient_alignment,
    shared_lora_named_parameters,
    should_measure_gradient_alignment,
    validate_mechanism_a_config,
)
from football_events.tracked_ball_teacher import TrackedBallTargetProvider


LABEL_SCHEMAS = {
    "five_way": {
        "labels": ["shot", "save", "corner", "freekick", "penalty"],
        "event_label_map": {
            "shot": "shot",
            "save": "save",
            "corner": "corner",
            "freekick": "freekick",
            "penalty": "penalty",
        },
    },
    "set_piece": {
        "labels": ["shot", "save", "set_piece"],
        "event_label_map": {
            "shot": "shot",
            "save": "save",
            "corner": "set_piece",
            "freekick": "set_piece",
            "penalty": "set_piece",
            "kickoff": "set_piece",
        },
    },
}
LABEL_SCHEMA = "five_way"
LABELS = list(LABEL_SCHEMAS[LABEL_SCHEMA]["labels"])
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
EVENT_LABEL_MAP = dict(LABEL_SCHEMAS[LABEL_SCHEMA]["event_label_map"])
SET_PIECE_SUBTYPES = ("corner", "freekick", "penalty", "kickoff")
SET_PIECE_SUBTYPE_TO_INDEX = {
    label: index for index, label in enumerate(SET_PIECE_SUBTYPES)
}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

BASE_TARGET_EVENT_LABELS = {
    "射门": "shot",
    "其他射门类型": "shot",
    "扑救": "save",
    "角球": "corner",
    "任意球": "freekick",
    "点球": "penalty",
    "中圈开球": "kickoff",
}
BASE_TARGET_EVENT_TYPES = {
    "S0199": "shot",
    "B0199": "shot",
    "S0401": "save",
    "B0401": "save",
    "S0201": "corner",
    "B0201": "corner",
    "S0202": "freekick",
    "B0202": "freekick",
    "S0101": "penalty",
    "B0101": "penalty",
    "S1004": "kickoff",
    "S06": "kickoff",
    "B1004": "kickoff",
    "B06": "kickoff",
}
VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".MP4", ".MOV", ".MKV", ".AVI")


class ConfigDict(dict):
    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def to_config(value: Any) -> Any:
    if isinstance(value, dict):
        return ConfigDict({key: to_config(item) for key, item in value.items()})
    if isinstance(value, list):
        return [to_config(item) for item in value]
    return value


def to_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: to_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain(item) for item in value]
    return value


def parse_override_value(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def apply_overrides(cfg: ConfigDict, overrides: list[str]) -> ConfigDict:
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override '{override}', expected key=value")
        key, raw_value = override.split("=", 1)
        parts = key.split(".")
        target: dict[str, Any] = cfg
        for part in parts[:-1]:
            target = target.setdefault(part, ConfigDict())
        target[parts[-1]] = to_config(parse_override_value(raw_value))
    return cfg


def load_config(path: str, overrides: list[str]) -> ConfigDict:
    with open(path) as f:
        cfg = to_config(yaml.safe_load(f))
    return apply_overrides(cfg, overrides)


def data_mode(cfg: Any) -> str:
    return str(cfg.data.get("mode", "clips"))


def configure_label_schema(cfg: Any) -> None:
    global LABEL_SCHEMA, LABELS, LABEL_TO_INDEX, EVENT_LABEL_MAP
    schema = str(cfg.get("task", ConfigDict()).get("label_schema", "five_way"))
    if schema not in LABEL_SCHEMAS:
        raise ValueError(f"Unsupported task.label_schema={schema}. Expected one of: {sorted(LABEL_SCHEMAS)}")
    LABEL_SCHEMA = schema
    LABELS = list(LABEL_SCHEMAS[schema]["labels"])
    LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
    EVENT_LABEL_MAP = dict(LABEL_SCHEMAS[schema]["event_label_map"])


def empty_label_vector() -> list[float]:
    return [0.0] * len(LABELS)


def full_label_mask() -> tuple[float, ...]:
    return tuple(1.0 for _ in LABELS)


def label_mask_from_counts(counts: dict[str, int], min_events_per_label: int) -> tuple[float, ...]:
    return tuple(1.0 if counts.get(label, 0) >= min_events_per_label else 0.0 for label in LABELS)


def label_mask_for_sample(
    video_label_mask: Sequence[float],
    labels: Sequence[float],
    *,
    positive_labels_always_trusted: bool = True,
) -> tuple[float, ...]:
    values = []
    for mask_value, label_value in zip(video_label_mask, labels):
        trusted = float(mask_value) > 0
        if positive_labels_always_trusted and float(label_value) > 0:
            trusted = True
        values.append(1.0 if trusted else 0.0)
    return tuple(values)


def set_output_label(labels: list[float], label: str | None) -> None:
    if label is not None and label in LABEL_TO_INDEX:
        labels[LABEL_TO_INDEX[label]] = 1.0


def map_base_event_label(base_label: str | None) -> str | None:
    if not base_label:
        return None
    return EVENT_LABEL_MAP.get(base_label)


def parse_image_size(value: Any) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if isinstance(value, str):
        text = value.strip()
        if "x" in text.lower():
            parts = text.lower().split("x", 1)
            return parse_image_size([parts[0], parts[1]])
        if "," in text and not text.startswith("["):
            return parse_image_size([part.strip() for part in text.split(",", 1)])
        parsed = yaml.safe_load(text)
        if parsed == value:
            raise ValueError(f"Invalid image_size={value}; expected int, [height,width], height,width, or heightxwidth")
        return parse_image_size(parsed)
    if isinstance(value, Sequence) and len(value) == 2:
        height, width = int(value[0]), int(value[1])
        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid image_size={value}")
        return height, width
    raise ValueError("video.image_size must be an int or [height, width]")


def normalize_on_device_enabled(cfg: Any) -> bool:
    preprocessing = cfg.data.get("preprocessing", ConfigDict())
    return bool(preprocessing.get("normalize_on_device", False))


@dataclass(frozen=True)
class ClipRecord:
    source: str
    root: str
    split: str
    video_id: str
    cluster_id: str
    clip_file: str
    clip_path: str
    is_negative: bool
    labels: tuple[float, ...]
    label_mask: tuple[float, ...]
    clip_start: float
    clip_end: float

    @property
    def cache_key(self) -> str:
        stem = Path(self.clip_file).stem
        return f"{self.source}/{self.split}/{stem}.pt"


@dataclass(frozen=True)
class FootballEvent:
    source: str
    video_id: str
    event_id: str
    event_type: str
    raw_label: str
    start_time: float
    end_time: float
    anchor_time: float
    labels: tuple[float, ...]
    # Reviewed timestamp remains the spotting anchor. Raw interval metadata is
    # separate because its midpoint is not the reviewed event time.
    context_start_time: float | None = None
    context_end_time: float | None = None
    is_ignored: bool = False


@dataclass(frozen=True)
class LongVideoRecord:
    source: str
    split: str
    video_id: str
    sample_id: str
    video_path: str
    annotation_path: str
    anchor_time: float
    base_clip_start: float
    base_clip_end: float
    video_duration: float
    is_negative: bool
    labels: tuple[float, ...]
    label_mask: tuple[float, ...]
    sample_loss_weight: float = 1.0
    focus_labels: tuple[float, ...] = ()
    save_cohort_kind: str = "none"
    save_cohort_weight: float = 0.0
    # Explicit online-simulation semantics. Do not encode supervision policy
    # in sample_id prefixes.
    online_chunk_id: str = ""
    online_chunk_mode: str = ""
    online_window_index: int = -1
    online_negative_kind: str = "none"
    online_positive_kind: str = "none"
    online_clean_negative_mask: tuple[float, ...] = ()
    online_clip_loss_weights: tuple[float, ...] = ()
    online_frame_loss_weights: tuple[float, ...] = ()
    online_label_loss_weights: tuple[float, ...] = ()
    online_gt_anchors: tuple[tuple[float, ...], ...] = ()
    # E1.3 central/edge pair metadata. A non-empty pair_id is globally unique
    # for one annotated event; both rows are kept in the same local batch.
    online_pair_id: str = ""
    online_pair_role: str = ""
    online_pair_class_mask: tuple[float, ...] = ()

    @property
    def cache_key(self) -> str:
        return f"{self.source}/{self.split}/{self.video_id}/{self.sample_id}.pt"


Record = Union[ClipRecord, LongVideoRecord]


def _as_float(value: str | float | int | None, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def _as_int(value: str | float | int | None, default: int = 0) -> int:
    if value is None or value == "":
        return default
    return int(float(value))


def parse_time_value(value: str | float | int | None, default: float = -1.0) -> float:
    if value is None or value == "":
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return default
    if ":" not in text:
        return float(text)
    parts = text.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600.0 + int(minutes) * 60.0 + float(seconds)
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60.0 + float(seconds)
    return default


def stable_int(value: str) -> int:
    return int(hashlib.md5(value.encode("utf-8")).hexdigest()[:8], 16)


def clamp(value: float, low: float, high: float) -> float:
    if high < low:
        return low
    return max(low, min(high, value))


def load_clip_records(roots: Iterable[str], split: str) -> list[ClipRecord]:
    records: list[ClipRecord] = []
    for root_str in roots:
        root = Path(root_str)
        csv_path = root / "annotations.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing annotations.csv: {csv_path}")
        source = root.name
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("split") != split:
                    continue
                is_negative = row.get("is_negative") == "1"
                subdir = "negative" if is_negative else "positive"
                clip_path = root / "clips" / subdir / row["clip_file"]
                labels = labels_from_clip_row(row)
                records.append(
                    ClipRecord(
                        source=source,
                        root=str(root),
                        split=split,
                        video_id=row["video_id"],
                        cluster_id=row["cluster_id"],
                        clip_file=row["clip_file"],
                        clip_path=str(clip_path),
                        is_negative=is_negative,
                        labels=labels,
                        label_mask=full_label_mask(),
                        clip_start=_as_float(row.get("clip_start")),
                        clip_end=_as_float(row.get("clip_end")),
                    )
                )
    if not records:
        raise RuntimeError(f"No records found for split={split}")
    return records


def read_video_id_files(paths: str | Sequence[str] | None) -> set[str]:
    if not paths:
        return set()
    if isinstance(paths, str):
        paths = [paths]
    video_ids: set[str] = set()
    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Missing split file: {path}")
        with path.open() as f:
            for line in f:
                item = line.strip()
                if item:
                    video_ids.add(Path(item).stem)
    return video_ids


def load_hard_negative_manifests(lv_cfg: Any, split: str) -> dict[tuple[str, str], list[dict[str, Any]]]:
    hard_cfg = lv_cfg.get("hard_negative", ConfigDict())
    if not bool(hard_cfg.get("enabled", False)):
        return {}
    if split != "train" and not bool(hard_cfg.get("include_in_val", False)):
        return {}

    paths = hard_cfg.get("manifests", hard_cfg.get("manifest", []))
    if isinstance(paths, str):
        paths = [paths]
    if not paths:
        raise ValueError("data.long_video.hard_negative.enabled=true requires manifest or manifests")

    min_score = float(hard_cfg.get("min_score", 0.0))
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Missing hard-negative manifest: {path}")
        with path.open() as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            items = payload.get("hard_negatives", payload.get("windows", []))
        else:
            items = payload
        if not isinstance(items, list):
            raise ValueError(f"Hard-negative manifest must contain a list: {path}")
        for item in items:
            if not isinstance(item, dict):
                continue
            video_id = str(item.get("video_id", "")).strip()
            if not video_id:
                continue
            score = float(item.get("score", 0.0) or 0.0)
            if score < min_score:
                continue
            source = str(item.get("source", "")).strip()
            grouped.setdefault((source, video_id), []).append(item)
    for items in grouped.values():
        items.sort(key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
    return grouped


def auto_split_video_keys(cfg: Any, split: str) -> set[str] | None:
    lv_cfg = cfg.data.long_video
    split_cfg = lv_cfg.get("auto_split", ConfigDict())
    if not bool(split_cfg.get("enabled", False)):
        return None

    val_ratio = float(split_cfg.get("val_ratio", 0.2))
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"data.long_video.auto_split.val_ratio must be between 0 and 1, got {val_ratio}")

    available_keys: set[str] = set()
    for root_cfg in lv_cfg.roots:
        source = str(root_cfg.get("source", Path(root_cfg.videos_dir).parent.name))
        videos_dir = Path(root_cfg.videos_dir)
        annotations_dir = Path(root_cfg.annotations_dir)
        if not videos_dir.exists() or not annotations_dir.exists():
            raise FileNotFoundError(f"Missing long-video dirs: videos={videos_dir} annotations={annotations_dir}")
        for annotation_path in sorted(annotations_dir.glob("*.json")):
            if annotation_path.name.startswith(".") or ".swp" in annotation_path.name:
                continue
            if find_video_path(videos_dir, annotation_path.stem) is not None:
                available_keys.add(f"{source}:{annotation_path.stem}")

    keys = sorted(available_keys, key=lambda key: stable_int(f"{int(split_cfg.get('seed', cfg.get('seed', 42)))}:{key}"))
    if len(keys) < 2:
        raise RuntimeError(f"auto_split needs at least 2 available annotated videos, found {len(keys)}")

    val_count = int(round(len(keys) * val_ratio))
    val_count = max(1, min(val_count, len(keys) - 1))
    val_keys = set(keys[:val_count])
    train_keys = set(keys[val_count:])
    if split == "train":
        selected = train_keys
    elif split == "val":
        selected = val_keys
    else:
        raise ValueError(f"auto_split only supports train/val splits, got {split}")
    print(
        f"auto_split split={split} selected_videos={len(selected)} total_available={len(keys)} val_ratio={val_ratio}",
        flush=True,
    )
    return selected


def find_video_path(videos_dir: Path, video_id: str) -> Path | None:
    for ext in VIDEO_EXTENSIONS:
        candidate = videos_dir / f"{video_id}{ext}"
        if candidate.exists():
            return candidate
    matches = sorted(path for path in videos_dir.glob(f"{video_id}*") if path.suffix in VIDEO_EXTENSIONS)
    return matches[0] if matches else None


def get_video_duration(path: Path) -> float:
    cap = _open_video_capture(path)
    if not cap.isOpened():
        return 0.0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    cap.release()
    if fps <= 0 or frames <= 0:
        return 0.0
    return frames / fps


def load_video_duration_cache(path: str | Path | None) -> dict[str, float]:
    if not path:
        return {}
    cache_path = Path(path)
    if not cache_path.is_file():
        return {}
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    raw = payload.get("durations_sec", payload) if isinstance(payload, dict) else {}
    if not isinstance(raw, dict):
        return {}
    return {str(key): float(value) for key, value in raw.items() if float(value) > 0}


def labels_from_clip_row(row: dict[str, str]) -> tuple[float, ...]:
    labels = empty_label_vector()
    for base_label, output_label in EVENT_LABEL_MAP.items():
        if _as_int(row.get(base_label), 0):
            set_output_label(labels, output_label)
    # Also support a pre-aggregated column such as set_piece in future CSVs.
    for output_label in LABELS:
        if _as_int(row.get(output_label), 0):
            set_output_label(labels, output_label)
    return tuple(labels)


def event_labels_from_item(item: dict[str, Any]) -> tuple[float, ...]:
    labels = empty_label_vector()
    event_label = str(item.get("label", item.get("event_label", "")))
    event_type = str(item.get("eventType", item.get("event_type", "")))
    base_label = BASE_TARGET_EVENT_LABELS.get(event_label) or BASE_TARGET_EVENT_TYPES.get(event_type)
    set_output_label(labels, map_base_event_label(base_label))
    # Goal annotations are treated as shot positives. The current downstream task
    # is key-event multi-label classification, not a separate goal subtype task.
    return tuple(labels)


def load_raw_interval_items(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    with path.open() as handle:
        payload = json.load(handle)
    items = payload.get("data", payload) if isinstance(payload, dict) else payload
    return [item for item in items if isinstance(item, dict)]


def match_raw_interval(
    item: dict[str, Any], raw_items: Sequence[dict[str, Any]], tolerance_sec: float = 0.05
) -> tuple[float | None, float | None]:
    anchor = parse_time_value(item.get("startTime", item.get("timestamp")), default=-1.0)
    if anchor < 0:
        return None, None
    label = str(item.get("label", item.get("event_label", "")))
    event_type = str(item.get("eventType", item.get("event_type", "")))
    best: tuple[float, float, float] | None = None
    for raw_item in raw_items:
        raw_label = str(raw_item.get("label", raw_item.get("event_label", "")))
        raw_type = str(raw_item.get("eventType", raw_item.get("event_type", "")))
        if label and raw_label and label != raw_label:
            continue
        if not label and event_type and raw_type and event_type != raw_type:
            continue
        raw_start = parse_time_value(
            raw_item.get("startTime", raw_item.get("timestamp")), default=-1.0
        )
        if raw_start < 0:
            continue
        delta = abs(raw_start - anchor)
        if delta > tolerance_sec:
            continue
        raw_end = parse_time_value(raw_item.get("endTime"), default=raw_start)
        raw_end = max(raw_end, raw_start)
        candidate = (delta, raw_start, raw_end)
        if best is None or candidate[0] < best[0]:
            best = candidate
    return (best[1], best[2]) if best is not None else (None, None)


def load_annotation_events(
    annotation_path: Path,
    source: str,
    video_id: str,
    *,
    raw_interval_path: Path | None = None,
    include_rejected_set_piece_as_ignored: bool = False,
) -> list[FootballEvent]:
    if annotation_path.name.startswith(".") or ".swp" in annotation_path.name:
        return []
    with annotation_path.open() as f:
        raw = json.load(f)
    items = raw.get("data", raw) if isinstance(raw, dict) else raw
    raw_items = load_raw_interval_items(raw_interval_path)
    events: list[FootballEvent] = []
    set_piece_index = LABEL_TO_INDEX.get("set_piece", -1)
    for item in items:
        if not isinstance(item, dict):
            continue
        labels = event_labels_from_item(item)
        if not any(labels):
            continue
        rejected = item.get("label_correct") is False
        rejected_set_piece = (
            rejected
            and set_piece_index >= 0
            and float(labels[set_piece_index]) > 0
        )
        if rejected and not (
            include_rejected_set_piece_as_ignored and rejected_set_piece
        ):
            continue
        start_time = parse_time_value(
            item.get("startTime", item.get("timestamp")), default=-1.0
        )
        end_time = parse_time_value(item.get("endTime"), default=start_time)
        if start_time < 0:
            continue
        if end_time < start_time:
            end_time = start_time
        raw_start, raw_end = match_raw_interval(item, raw_items)
        context_start = raw_start if raw_start is not None else start_time
        context_end = raw_end if raw_end is not None else end_time
        # A repair timestamp is the reviewed anchor and matches raw startTime.
        # Preserve legacy midpoint behavior only when directly loading an
        # interval annotation file without a separate reviewed/raw join.
        anchor_time = (
            start_time
            if raw_interval_path is not None or "timestamp" in item
            else ((start_time + end_time) * 0.5 if end_time > start_time else start_time)
        )
        events.append(
            FootballEvent(
                source=source,
                video_id=video_id,
                event_id=str(item.get("id", f"{video_id}_{len(events)}")),
                event_type=str(item.get("eventType", item.get("event_type", ""))),
                raw_label=str(item.get("label", "")),
                start_time=start_time,
                end_time=end_time,
                anchor_time=anchor_time,
                labels=labels,
                context_start_time=context_start,
                context_end_time=context_end,
                is_ignored=bool(rejected_set_piece),
            )
        )
    events.sort(key=lambda event: event.anchor_time)
    return events


def labels_for_window(events: Sequence[FootballEvent], start_sec: float, end_sec: float) -> tuple[float, ...]:
    labels = np.zeros(len(LABELS), dtype=np.float32)
    for event in events:
        if event.is_ignored:
            continue
        if start_sec <= event.anchor_time <= end_sec:
            labels = np.maximum(labels, np.asarray(event.labels, dtype=np.float32))
    return tuple(float(x) for x in labels.tolist())


def set_piece_subtypes_for_window(
    events: Sequence[FootballEvent], start_sec: float, end_sec: float
) -> tuple[float, ...]:
    """Return multi-hot set-piece subtypes without changing the 3-class target."""
    labels = [0.0] * len(SET_PIECE_SUBTYPES)
    for event in events:
        if event.is_ignored or not (start_sec <= event.anchor_time <= end_sec):
            continue
        subtype = (
            BASE_TARGET_EVENT_LABELS.get(event.raw_label)
            or BASE_TARGET_EVENT_TYPES.get(event.event_type)
        )
        index = SET_PIECE_SUBTYPE_TO_INDEX.get(str(subtype))
        if index is not None:
            labels[index] = 1.0
    return tuple(labels)


def resolve_negative_safety_margins(
    lv_cfg: Any, split: str
) -> tuple[float, dict[str, float]]:
    by_split = lv_cfg.get("negative_safety_margin_sec_by_split", ConfigDict())
    default_margin = max(
        float(by_split.get(split, lv_cfg.get("negative_safety_margin_sec", 0.0)) or 0.0),
        0.0,
    )
    label_by_split = lv_cfg.get(
        "negative_label_safety_margin_sec_by_split", ConfigDict()
    )
    raw = label_by_split.get(
        split, lv_cfg.get("negative_label_safety_margin_sec", ConfigDict())
    )
    if not isinstance(raw, dict):
        raw = ConfigDict()
    margins = {
        label: max(float(raw.get(label, default_margin) or 0.0), 0.0)
        for label in LABELS
    }
    return default_margin, margins


def event_negative_safety_margin(
    event: FootballEvent,
    default_margin_sec: float,
    label_margin_sec: dict[str, float],
) -> float:
    margins = [max(float(default_margin_sec), 0.0)]
    for index, value in enumerate(event.labels):
        if float(value) > 0 and index < len(LABELS):
            margins.append(float(label_margin_sec.get(LABELS[index], default_margin_sec)))
    return max(margins)


def is_safe_negative_window(
    events: Sequence[FootballEvent],
    start_sec: float,
    end_sec: float,
    *,
    default_margin_sec: float = 0.0,
    label_margin_sec: dict[str, float] | None = None,
) -> bool:
    margins = label_margin_sec or {}
    for event in events:
        if event.is_ignored:
            continue
        margin = event_negative_safety_margin(event, default_margin_sec, margins)
        if start_sec - margin <= event.anchor_time <= end_sec + margin:
            return False
    return True


def sample_coverage_balanced_negative_starts(
    events: Sequence[FootballEvent],
    video_duration: float,
    clip_duration: float,
    num_negatives: int,
    rng: random.Random,
    *,
    default_margin_sec: float = 0.0,
    label_margin_sec: dict[str, float] | None = None,
    candidate_stride_sec: float = 2.0,
    min_start_distance_sec: float = 2.0,
    temporal_bin_sec: float = 60.0,
) -> list[float]:
    if num_negatives <= 0:
        return []
    max_start = max(float(video_duration) - float(clip_duration), 0.0)
    stride = max(float(candidate_stride_sec), 0.25)
    seen: set[float] = set()
    candidates: list[float] = []

    def add_candidate(raw_start: float) -> None:
        start = clamp(float(raw_start), 0.0, max_start)
        key = round(start, 6)
        if key in seen:
            return
        if not is_safe_negative_window(
            events,
            start,
            start + clip_duration,
            default_margin_sec=default_margin_sec,
            label_margin_sec=label_margin_sec,
        ):
            return
        seen.add(key)
        candidates.append(start)

    add_candidate(0.0)
    add_candidate(max_start)
    grid_count = int(math.floor(max_start / stride)) + 1
    for index in range(grid_count + 1):
        base = min(index * stride, max_start)
        jitter = rng.uniform(-0.35 * stride, 0.35 * stride)
        add_candidate(base + jitter)

    # Dense random candidates avoid grid aliasing around annotation safety zones.
    max_attempts = max(num_negatives * 20, 1000)
    for _ in range(max_attempts):
        if len(candidates) >= max(num_negatives * 4, num_negatives + 128):
            break
        add_candidate(rng.uniform(0.0, max_start) if max_start > 0 else 0.0)

    bin_sec = max(float(temporal_bin_sec), clip_duration, 1.0)
    grouped: dict[int, list[float]] = {}
    for start in candidates:
        center = start + clip_duration * 0.5
        grouped.setdefault(int(center // bin_sec), []).append(start)
    for values in grouped.values():
        rng.shuffle(values)

    ordered: list[float] = []
    active_bins = sorted(grouped)
    while active_bins:
        next_bins: list[int] = []
        for bin_index in active_bins:
            values = grouped[bin_index]
            if values:
                ordered.append(values.pop())
            if values:
                next_bins.append(bin_index)
        active_bins = next_bins

    selected: list[float] = []
    selected_keys: set[float] = set()
    initial_distance = max(float(min_start_distance_sec), 0.0)
    distances = [initial_distance]
    if initial_distance > 0:
        distances.extend([initial_distance * 0.5, 0.0])
    for min_distance in distances:
        for start in ordered:
            key = round(start, 6)
            if key in selected_keys:
                continue
            if min_distance > 0 and any(
                abs(start - existing) < min_distance for existing in selected
            ):
                continue
            selected.append(start)
            selected_keys.add(key)
            if len(selected) >= num_negatives:
                return selected
    return selected


def online_negative_label_mask(
    events: Sequence[FootballEvent],
    start_sec: float,
    end_sec: float,
    label_mask: Sequence[float],
    safety_margin_sec: float,
) -> tuple[float, ...]:
    """Ignore near-event negatives only for online hard-negative losses."""
    if safety_margin_sec <= 0:
        return tuple(float(value) for value in label_mask)
    labels = labels_for_window(events, start_sec, end_sec)
    nearby_labels = labels_for_window(
        events,
        start_sec - safety_margin_sec,
        end_sec + safety_margin_sec,
    )
    return tuple(
        0.0 if labels[index] <= 0 and nearby_labels[index] > 0 else float(value)
        for index, value in enumerate(label_mask)
    )


def rejected_set_piece_label_mask(
    events: Sequence[FootballEvent],
    start_sec: float,
    end_sec: float,
    labels: Sequence[float],
    label_mask: Sequence[float],
    margin_sec: float,
) -> tuple[float, ...]:
    """Mask only set-piece negatives overlapping reviewed-rejected raw events."""
    result = [float(value) for value in label_mask]
    set_piece_index = LABEL_TO_INDEX.get("set_piece", -1)
    if set_piece_index < 0 or float(labels[set_piece_index]) > 0:
        return tuple(result)
    margin = max(float(margin_sec), 0.0)
    for event in events:
        if not event.is_ignored:
            continue
        if set_piece_index >= len(event.labels) or float(event.labels[set_piece_index]) <= 0:
            continue
        interval_start = float(
            event.context_start_time
            if event.context_start_time is not None
            else event.anchor_time
        )
        interval_end = float(
            event.context_end_time
            if event.context_end_time is not None
            else event.anchor_time
        )
        if start_sec <= interval_end + margin and end_sec >= interval_start - margin:
            result[set_piece_index] = 0.0
            break
    return tuple(result)


def set_piece_context_span_mask(
    events: Sequence[FootballEvent],
    start_sec: float,
    end_sec: float,
    frame_times: Sequence[float],
    min_duration_sec: float = 0.25,
) -> tuple[Tensor, Tensor]:
    """Return accepted raw context spans without moving the reviewed anchor."""
    times = torch.as_tensor(list(frame_times), dtype=torch.float32)
    mask = torch.zeros(len(times), dtype=torch.float32)
    set_piece_index = LABEL_TO_INDEX.get("set_piece", -1)
    if set_piece_index < 0 or times.numel() == 0:
        return mask, torch.tensor(0.0, dtype=torch.float32)
    valid_spans: list[tuple[float, float]] = []
    for event in events:
        if event.is_ignored or float(event.labels[set_piece_index]) <= 0:
            continue
        if not (start_sec <= float(event.anchor_time) <= end_sec):
            continue
        if event.context_start_time is None or event.context_end_time is None:
            continue
        span_start = max(float(event.context_start_time), float(start_sec))
        span_end = min(float(event.context_end_time), float(end_sec))
        if span_end - span_start < max(float(min_duration_sec), 0.0):
            continue
        valid_spans.append((span_start, span_end))
        mask = torch.maximum(
            mask, ((times >= span_start) & (times <= span_end)).to(torch.float32)
        )
    if not valid_spans:
        return mask, torch.tensor(0.0, dtype=torch.float32)
    if mask.sum() <= 0:
        centers = torch.tensor(
            [0.5 * (span_start + span_end) for span_start, span_end in valid_spans],
            dtype=torch.float32,
        )
        nearest = torch.abs(times[:, None] - centers[None, :]).min(dim=1).values.argmin()
        mask[nearest] = 1.0
    return mask, torch.tensor(1.0, dtype=torch.float32)


def labels_present_in_events(events: Sequence[FootballEvent]) -> set[str]:
    present: set[str] = set()
    for event in events:
        if event.is_ignored:
            continue
        for index, value in enumerate(event.labels):
            if value > 0:
                present.add(LABELS[index])
    return present


def label_counts_in_events(events: Sequence[FootballEvent]) -> dict[str, int]:
    counts = {label: 0 for label in LABELS}
    for event in events:
        if event.is_ignored:
            continue
        for index, value in enumerate(event.labels):
            if value > 0:
                counts[LABELS[index]] += 1
    return counts


def centered_window(anchor_time: float, duration: float, video_duration: float) -> tuple[float, float]:
    max_start = max(video_duration - duration, 0.0)
    start = clamp(anchor_time - duration * 0.5, 0.0, max_start)
    return start, start + duration


def nested_sampling_window(
    record: LongVideoRecord,
    context_start: float,
    context_end: float,
    sampling_duration: float,
    *,
    jitter_sec: float = 0.0,
) -> tuple[float, float]:
    """Choose a dense input window without changing the parent sample identity."""
    context_duration = max(float(context_end) - float(context_start), 0.0)
    duration = float(sampling_duration)
    if duration <= 0.0:
        raise ValueError("video.sampling_duration must be positive")
    if duration >= context_duration:
        return float(context_start), float(context_end)
    center = (
        float(record.anchor_time)
        if not record.is_negative
        else 0.5 * (float(context_start) + float(context_end))
    )
    jitter = max(float(jitter_sec), 0.0)
    if jitter > 0.0:
        center += random.uniform(-jitter, jitter)
    max_start = float(context_end) - duration
    start = clamp(center - 0.5 * duration, float(context_start), max_start)
    return start, start + duration



def classify_save_cohort(record: LongVideoRecord) -> str:
    if not record.focus_labels or "shot" not in LABEL_TO_INDEX or "save" not in LABEL_TO_INDEX:
        return "none"
    shot_index = LABEL_TO_INDEX["shot"]
    save_index = LABEL_TO_INDEX["save"]
    if len(record.focus_labels) <= max(shot_index, save_index):
        return "none"
    focus_shot = float(record.focus_labels[shot_index]) > 0
    focus_save = float(record.focus_labels[save_index]) > 0
    window_shot = float(record.labels[shot_index]) > 0
    window_save = float(record.labels[save_index]) > 0
    trusted = float(record.label_mask[shot_index]) > 0 and float(record.label_mask[save_index]) > 0
    if not trusted:
        return "none"
    if focus_save:
        return "save_joint" if window_shot else "save_without_shot"
    if focus_shot and window_shot and not window_save:
        return "shot_only"
    return "none"


def apply_dataset_level_save_cohort_weights(
    records: Sequence[LongVideoRecord], cfg: Any, split: str
) -> list[LongVideoRecord]:
    train_cfg = cfg.get("train", ConfigDict())
    if split != "train" or float(train_cfg.get("save_shot_cohort_loss_weight", 0.0) or 0.0) <= 0:
        return list(records)
    kinds = [classify_save_cohort(record) for record in records]
    counts = {kind: kinds.count(kind) for kind in ("save_joint", "save_without_shot", "shot_only")}
    mass_cfg = train_cfg.get("save_cohort_target_mass", ConfigDict())
    masses = {
        "save_joint": max(float(mass_cfg.get("save_joint", 0.4) or 0.0), 0.0),
        "save_without_shot": max(float(mass_cfg.get("save_without_shot", 0.2) or 0.0), 0.0),
        "shot_only": max(float(mass_cfg.get("shot_only", 0.4) or 0.0), 0.0),
    }
    active_mass = sum(masses[kind] for kind, count in counts.items() if count > 0)
    if active_mass <= 0:
        raise ValueError("save cohort target mass must be positive for an available cohort")
    included_count = sum(counts.values())
    max_weight = max(float(train_cfg.get("save_cohort_max_sample_weight", 4.0) or 4.0), 1.0)
    weights: dict[str, float] = {}
    for kind, count in counts.items():
        if count <= 0:
            weights[kind] = 0.0
            continue
        normalized_mass = masses[kind] / active_mass
        weights[kind] = min(normalized_mass * included_count / count, max_weight)
    print(f"split={split} save_cohort_counts={counts} target_mass={masses} resolved_weights={weights}", flush=True)
    return [
        replace(record, save_cohort_kind=kind, save_cohort_weight=weights.get(kind, 0.0))
        for record, kind in zip(records, kinds)
    ]


def load_long_video_records(cfg: Any, split: str) -> tuple[list[LongVideoRecord], dict[tuple[str, str], list[FootballEvent]]]:
    lv_cfg = cfg.data.long_video
    clip_duration = float(cfg.video.get("clip_duration", 9.0))
    seed = int(cfg.get("seed", 42)) + stable_int(split)
    split_files = lv_cfg.get("split_files", {}).get(split)
    split_ids = read_video_id_files(split_files)
    split_keys = auto_split_video_keys(cfg, split)
    records: list[LongVideoRecord] = []
    events_by_video: dict[tuple[str, str], list[FootballEvent]] = {}
    hard_negative_items = load_hard_negative_manifests(lv_cfg, split)
    hard_negative_cfg = lv_cfg.get("hard_negative", ConfigDict())
    hard_negative_max_per_video = int(hard_negative_cfg.get("max_per_video", 0))
    hard_negative_safety_margin_sec = float(hard_negative_cfg.get("safety_margin_sec", 2.0))
    hard_negative_repeat_factor = max(int(hard_negative_cfg.get("repeat_factor", 1)), 1)
    missing_annotation_paths: set[str] = set()
    seen_annotation_paths: set[str] = set()
    (
        negative_safety_margin_sec,
        negative_label_safety_margin_sec,
    ) = resolve_negative_safety_margins(lv_cfg, split)
    negative_skipped_incomplete_label_videos = 0
    hard_negatives_written = 0
    negative_require_all_labels = bool(lv_cfg.get("negative_require_all_labels", False))
    negative_min_events_per_label = int(lv_cfg.get("negative_min_events_per_label", 1))
    duration_cache_path = str(lv_cfg.get("duration_cache_path", "") or "")
    duration_cache = load_video_duration_cache(duration_cache_path)
    if duration_cache_path:
        print(
            f"split={split} video_duration_cache={duration_cache_path} entries={len(duration_cache)}",
            flush=True,
        )

    for root_cfg in lv_cfg.roots:
        source = str(root_cfg.get("source", Path(root_cfg.videos_dir).parent.name))
        positive_only_root = bool(root_cfg.get("positive_only", False))
        root_loss_weight = max(float(root_cfg.get("loss_weight", 1.0) or 1.0), 0.0)
        root_splits = root_cfg.get("splits", ["train"] if positive_only_root else None)
        if root_splits is not None:
            root_split_set = {str(item) for item in root_splits}
            if split not in root_split_set:
                continue
        videos_dir = Path(root_cfg.videos_dir)
        annotations_dir = Path(root_cfg.annotations_dir)
        raw_annotations_value = str(root_cfg.get("raw_annotations_dir", "") or "")
        raw_annotations_dir = Path(raw_annotations_value) if raw_annotations_value else None
        raw_supervision_cfg = cfg.get("raw_set_piece_supervision", ConfigDict())
        raw_supervision_enabled = bool(raw_supervision_cfg.get("enabled", False))
        if raw_supervision_enabled and raw_annotations_dir is not None and not raw_annotations_dir.is_dir():
            raise FileNotFoundError(f"Missing raw annotation dir: {raw_annotations_dir}")
        if not videos_dir.exists() or not annotations_dir.exists():
            raise FileNotFoundError(f"Missing long-video dirs: videos={videos_dir} annotations={annotations_dir}")
        for annotation_path in sorted(annotations_dir.glob("*.json")):
            if annotation_path.name.startswith(".") or ".swp" in annotation_path.name:
                continue
            video_id = annotation_path.stem
            if split_ids and not positive_only_root and video_id not in split_ids:
                continue
            if split_keys is not None and not positive_only_root and f"{source}:{video_id}" not in split_keys:
                continue
            annotation_key = str(annotation_path.resolve())
            if annotation_key in seen_annotation_paths:
                continue
            video_path = find_video_path(videos_dir, video_id)
            if video_path is None:
                missing_annotation_paths.add(annotation_key)
                continue
            path_overrides = root_cfg.get("video_path_overrides", ConfigDict())
            override_value = path_overrides.get(video_id) if isinstance(path_overrides, dict) else None
            if override_value:
                override_path = Path(str(override_value))
                if not override_path.is_file():
                    raise FileNotFoundError(
                        f"Configured video_path_overrides entry does not exist: "
                        f"video={video_id} path={override_path}"
                    )
                video_path = override_path
                print(
                    f"video_path_override split={split} video={video_id} path={video_path}",
                    flush=True,
                )
            seen_annotation_paths.add(annotation_key)
            missing_annotation_paths.discard(annotation_key)
            duration = float(duration_cache.get(video_id, 0.0) or 0.0)
            if duration <= 0:
                duration = get_video_duration(video_path)
            if duration <= 0:
                continue
            raw_interval_path = (
                raw_annotations_dir / annotation_path.name
                if raw_supervision_enabled and raw_annotations_dir is not None
                else None
            )
            events = load_annotation_events(
                annotation_path, source, video_id,
                raw_interval_path=raw_interval_path,
                include_rejected_set_piece_as_ignored=(
                    split == "train" and bool(raw_supervision_cfg.get("rejected_ignore_enabled", True))
                ),
            )
            key = (source, video_id)
            events_by_video[key] = events
            label_counts = label_counts_in_events(events)
            video_label_mask = (
                tuple(0.0 for _ in LABELS)
                if positive_only_root
                else label_mask_from_counts(label_counts, negative_min_events_per_label)
            )
            insufficient_labels = {
                label for label, count in label_counts.items() if count < negative_min_events_per_label
            }
            for event_index, event in enumerate(events):
                if event.is_ignored:
                    continue
                base_start, base_end = centered_window(event.anchor_time, clip_duration, duration)
                labels = labels_for_window(events, base_start, base_end)
                record_label_mask = (
                    tuple(1.0 if float(value) > 0 else 0.0 for value in labels)
                    if positive_only_root
                    else video_label_mask
                )
                records.append(
                    LongVideoRecord(
                        source=source,
                        split=split,
                        video_id=video_id,
                        sample_id=f"pos_{event_index:04d}_{event.event_id}",
                        video_path=str(video_path),
                        annotation_path=str(annotation_path),
                        anchor_time=event.anchor_time,
                        base_clip_start=base_start,
                        base_clip_end=base_end,
                        video_duration=duration,
                        is_negative=False,
                        labels=labels,
                        label_mask=record_label_mask,
                        sample_loss_weight=root_loss_weight,
                        focus_labels=event.labels,
                    )
                )

            if positive_only_root:
                continue

            if negative_require_all_labels and insufficient_labels:
                negative_skipped_incomplete_label_videos += 1
                continue

            ratio_by_split = lv_cfg.get("negative_ratio_by_split", ConfigDict())
            negative_ratio = float(
                ratio_by_split.get(split, lv_cfg.get("negative_ratio", 0.5))
            )
            negative_per_video = int(lv_cfg.get("negative_per_video", 0))
            negative_per_minute_by_split = lv_cfg.get(
                "negative_per_minute_by_split", ConfigDict()
            )
            negative_per_minute = float(
                negative_per_minute_by_split.get(
                    split, lv_cfg.get("negative_per_minute", 0.0)
                )
                or 0.0
            )
            negative_sampling_mode_by_split = lv_cfg.get(
                "negative_sampling_mode_by_split", ConfigDict()
            )
            negative_sampling_mode = str(
                negative_sampling_mode_by_split.get(
                    split, lv_cfg.get("negative_sampling_mode", "auto")
                )
            ).strip().lower()
            hybrid_event_ratio_by_split = lv_cfg.get(
                "negative_hybrid_event_ratio_by_split", ConfigDict()
            )
            hybrid_event_ratio = float(
                hybrid_event_ratio_by_split.get(
                    split,
                    lv_cfg.get("negative_hybrid_event_ratio", negative_ratio),
                )
            )
            if negative_per_minute < 0:
                raise ValueError("data.long_video.negative_per_minute must be non-negative")
            if negative_sampling_mode not in (
                "auto",
                "event_ratio",
                "duration",
                "hybrid",
                "coverage_balanced",
            ):
                raise ValueError(
                    "data.long_video.negative_sampling_mode must be auto, "
                    "event_ratio, duration, hybrid, or coverage_balanced"
                )
            if negative_per_video > 0:
                num_negatives = negative_per_video
            elif negative_sampling_mode == "coverage_balanced":
                event_target = int(round(len(events) * negative_ratio))
                duration_target = int(
                    round((duration / 60.0) * negative_per_minute)
                )
                num_negatives = max(event_target, duration_target)
            elif negative_sampling_mode == "hybrid":
                num_negatives = int(
                    round(
                        len(events) * hybrid_event_ratio
                        + (duration / 60.0) * negative_per_minute
                    )
                )
            elif negative_sampling_mode == "duration" or (
                negative_sampling_mode == "auto" and negative_per_minute > 0
            ):
                num_negatives = int(
                    round((duration / 60.0) * negative_per_minute)
                )
            else:
                num_negatives = int(round(len(events) * negative_ratio))
            if not any(video_label_mask):
                num_negatives = 0
            elif num_negatives > 0:
                num_negatives = max(num_negatives, 1)
            max_start = max(duration - clip_duration, 0.0)
            rng = random.Random(seed + stable_int(f"{source}:{video_id}"))
            if negative_sampling_mode == "coverage_balanced":
                negative_starts = sample_coverage_balanced_negative_starts(
                    events,
                    duration,
                    clip_duration,
                    num_negatives,
                    rng,
                    default_margin_sec=negative_safety_margin_sec,
                    label_margin_sec=negative_label_safety_margin_sec,
                    candidate_stride_sec=float(
                        lv_cfg.get("negative_candidate_stride_sec", 2.0) or 2.0
                    ),
                    min_start_distance_sec=float(
                        lv_cfg.get("negative_min_start_distance_sec", 2.0) or 0.0
                    ),
                    temporal_bin_sec=float(
                        lv_cfg.get("negative_temporal_bin_sec", 60.0) or 60.0
                    ),
                )
            else:
                negative_starts = []
                attempts = max(num_negatives * 50, 100)
                for _ in range(attempts):
                    if len(negative_starts) >= num_negatives:
                        break
                    start = rng.uniform(0.0, max_start) if max_start > 0 else 0.0
                    if is_safe_negative_window(
                        events,
                        start,
                        start + clip_duration,
                        default_margin_sec=negative_safety_margin_sec,
                        label_margin_sec=negative_label_safety_margin_sec,
                    ):
                        negative_starts.append(start)
            for negative_index, start in enumerate(negative_starts):
                end = start + clip_duration
                labels = labels_for_window(events, start, end)
                records.append(
                    LongVideoRecord(
                        source=source,
                        split=split,
                        video_id=video_id,
                        sample_id=f"neg_{negative_index:04d}",
                        video_path=str(video_path),
                        annotation_path=str(annotation_path),
                        anchor_time=-1.0,
                        base_clip_start=start,
                        base_clip_end=end,
                        video_duration=duration,
                        is_negative=True,
                        labels=labels,
                        label_mask=video_label_mask,
                        sample_loss_weight=root_loss_weight,
                    )
                )

            candidates = list(hard_negative_items.get((source, video_id), []))
            candidates.extend(hard_negative_items.get(("", video_id), []))
            candidates.sort(key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
            if hard_negative_max_per_video > 0:
                candidates = candidates[:hard_negative_max_per_video]
            for hard_index, item in enumerate(candidates):
                raw_start = parse_time_value(item.get("start_sec"), default=-1.0)
                raw_end = parse_time_value(item.get("end_sec"), default=raw_start)
                if raw_start < 0:
                    center = parse_time_value(item.get("center_sec", item.get("time_sec")), default=-1.0)
                else:
                    center = (raw_start + max(raw_end, raw_start)) * 0.5
                if center < 0:
                    continue
                raw_labels = item.get("labels", item.get("label", ""))
                if isinstance(raw_labels, str):
                    mined_labels = [label.strip() for label in raw_labels.split(",") if label.strip()]
                elif isinstance(raw_labels, Sequence):
                    mined_labels = [str(label).strip() for label in raw_labels if str(label).strip()]
                else:
                    mined_labels = []
                hard_label_mask = list(empty_label_vector())
                for mined_label in mined_labels:
                    if mined_label in LABEL_TO_INDEX:
                        label_index = LABEL_TO_INDEX[mined_label]
                        hard_label_mask[label_index] = float(video_label_mask[label_index])
                if not any(hard_label_mask):
                    hard_label_mask = list(video_label_mask)
                start, end = centered_window(center, clip_duration, duration)
                # Revalidate against current annotations so a stale mining manifest
                # cannot turn a newly repaired positive event into a negative. For
                # label-specific hard negatives, only reject if that mined label is
                # positive in the safety window; other labels are masked out below.
                safe_start = max(0.0, start - hard_negative_safety_margin_sec)
                safe_end = min(duration, end + hard_negative_safety_margin_sec)
                safe_labels = labels_for_window(events, safe_start, safe_end)
                if any(float(safe_labels[i]) > 0 and float(hard_label_mask[i]) > 0 for i in range(len(LABELS))):
                    continue
                item_id = stable_int(str(item))
                for repeat_index in range(hard_negative_repeat_factor):
                    records.append(
                        LongVideoRecord(
                            source=source,
                            split=split,
                            video_id=video_id,
                            sample_id=f"hard_neg_{hard_index:04d}_r{repeat_index:02d}_{item_id}",
                            video_path=str(video_path),
                            annotation_path=str(annotation_path),
                            anchor_time=-1.0,
                            base_clip_start=start,
                            base_clip_end=end,
                            video_duration=duration,
                            is_negative=True,
                            labels=tuple(empty_label_vector()),
                            label_mask=tuple(hard_label_mask),
                            sample_loss_weight=root_loss_weight,
                        )
                    )
                    hard_negatives_written += 1

    missing_videos = len(missing_annotation_paths - seen_annotation_paths)
    records = apply_dataset_level_save_cohort_weights(records, cfg, split)
    if not records:
        raise RuntimeError(f"No long-video records found for split={split}; missing_videos={missing_videos}")
    if missing_videos:
        print(f"warning: split={split} skipped {missing_videos} annotations with missing videos", flush=True)
    if negative_skipped_incomplete_label_videos:
        print(
            f"split={split} skipped negative sampling for {negative_skipped_incomplete_label_videos} videos "
            f"because negative_require_all_labels=true and at least one label count was "
            f"below negative_min_events_per_label={negative_min_events_per_label}",
            flush=True,
        )
    if hard_negative_items:
        print(
            f"split={split} loaded hard_negatives={hard_negatives_written} "
            f"from_manifest_candidates={sum(len(items) for items in hard_negative_items.values())}",
            flush=True,
        )
    return records, events_by_video


def summarize_records(records: list[Record]) -> dict[str, Any]:
    targets = np.asarray([r.labels for r in records], dtype=np.float32)
    masks = np.asarray([label_mask_for_sample(r.label_mask, r.labels) for r in records], dtype=np.float32)
    positive_records = [record for record in records if not record.is_negative]
    anchor_keys_by_label: dict[str, set[tuple[str, str, str]]] = {
        label: set() for label in LABELS
    }
    video_keys_by_label: dict[str, set[tuple[str, str]]] = {
        label: set() for label in LABELS
    }
    anchor_clip_counts: dict[tuple[str, str, str], int] = {}
    for record in positive_records:
        video_key = (record.source, record.video_id)
        anchor_key = (
            record.source,
            record.video_id,
            getattr(record, "sample_id", record.cache_key),
        )
        anchor_clip_counts[anchor_key] = anchor_clip_counts.get(anchor_key, 0) + 1
        focus_labels = getattr(record, "focus_labels", ())
        anchor_labels = focus_labels if focus_labels else record.labels
        for label_index, label in enumerate(LABELS):
            if float(anchor_labels[label_index]) > 0:
                anchor_keys_by_label[label].add(anchor_key)
                video_keys_by_label[label].add(video_key)
    clips_per_anchor = np.asarray(list(anchor_clip_counts.values()), dtype=np.float32)
    missing = 0
    for record in records:
        path = getattr(record, "clip_path", getattr(record, "video_path", ""))
        missing += int(not Path(path).exists())
    return {
        "num_clips": len(records),
        "num_videos": len({r.video_id for r in records}),
        "num_negative": sum(r.is_negative for r in records),
        "num_hard_negative": sum(
            isinstance(r, LongVideoRecord) and r.sample_id.startswith("hard_neg_")
            for r in records
        ),
        "num_positive": sum(not r.is_negative for r in records),
        "num_multilabel": int((targets.sum(axis=1) > 1).sum()),
        "label_counts": {label: int(targets[:, i].sum()) for i, label in enumerate(LABELS)},
        "unique_event_anchors": {
            label: len(anchor_keys_by_label[label]) for label in LABELS
        },
        "unique_videos_by_label": {
            label: len(video_keys_by_label[label]) for label in LABELS
        },
        "clips_per_event_anchor": {
            "mean": float(clips_per_anchor.mean()) if clips_per_anchor.size else 0.0,
            "p50": float(np.quantile(clips_per_anchor, 0.50)) if clips_per_anchor.size else 0.0,
            "p90": float(np.quantile(clips_per_anchor, 0.90)) if clips_per_anchor.size else 0.0,
            "max": int(clips_per_anchor.max()) if clips_per_anchor.size else 0,
        },
        "trusted_label_counts": {label: int(masks[:, i].sum()) for i, label in enumerate(LABELS)},
        "unknown_label_counts": {label: int((masks[:, i] <= 0).sum()) for i, label in enumerate(LABELS)},
        "missing_files": missing,
    }


def compute_pos_weight(records: list[Record], max_weight: float) -> Tensor:
    targets = torch.tensor([r.labels for r in records], dtype=torch.float32)
    masks = torch.tensor([label_mask_for_sample(r.label_mask, r.labels) for r in records], dtype=torch.float32)
    pos = (targets * masks).sum(dim=0)
    neg = ((1.0 - targets) * masks).sum(dim=0)
    weights = neg / pos.clamp_min(1.0)
    return weights.clamp(max=max_weight)


def resolve_pos_weight(records: list[Record], train_cfg: Any) -> tuple[Tensor, str]:
    """Resolve an explicit fixed positive weight or the legacy automatic weight."""
    raw = train_cfg.get("pos_weight", "auto")
    if raw is None or (isinstance(raw, str) and raw.strip().lower() == "auto"):
        max_weight = float(train_cfg.get("pos_weight_max", 50.0))
        return compute_pos_weight(records, max_weight), "auto"

    if isinstance(raw, dict):
        missing = [label for label in LABELS if label not in raw]
        extra = [label for label in raw if label not in LABELS]
        if missing or extra:
            raise ValueError(f"train.pos_weight label mismatch: missing={missing} extra={extra}")
        values = [float(raw[label]) for label in LABELS]
    elif isinstance(raw, (int, float)):
        values = [float(raw)] * len(LABELS)
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = [float(value) for value in raw]
        if len(values) != len(LABELS):
            raise ValueError(
                f"train.pos_weight must have {len(LABELS)} values for labels={LABELS}, got {len(values)}"
            )
    else:
        raise ValueError(
            "train.pos_weight must be 'auto', a positive scalar, a label mapping, or a label-sized list"
        )

    weights = torch.tensor(values, dtype=torch.float32)
    if not torch.isfinite(weights).all() or bool((weights <= 0).any()):
        raise ValueError(f"train.pos_weight values must be finite and > 0, got {values}")
    return weights, "fixed"


def sample_frame_indices(frame_count: int, num_frames: int, is_train: bool) -> list[int]:
    if frame_count <= 0:
        return [0] * num_frames
    if frame_count < num_frames:
        return [min(round(i * (frame_count - 1) / max(num_frames - 1, 1)), frame_count - 1) for i in range(num_frames)]
    if not is_train:
        return np.linspace(0, frame_count - 1, num_frames).round().astype(int).tolist()
    edges = np.linspace(0, frame_count, num_frames + 1).astype(int)
    indices = []
    for start, end in zip(edges[:-1], edges[1:]):
        if end <= start:
            indices.append(min(start, frame_count - 1))
        else:
            indices.append(random.randrange(start, end))
    return indices


def segment_frame_indices(
    total_frames: int,
    fps: float,
    num_frames: int,
    is_train: bool,
    *,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> list[int]:
    if total_frames <= 0:
        return [0] * num_frames
    if start_sec is None or end_sec is None or fps <= 0:
        start_frame = 0
        end_frame = total_frames - 1
    else:
        start_frame = int(clamp(round(start_sec * fps), 0, total_frames - 1))
        end_frame = int(clamp(round(end_sec * fps) - 1, start_frame, total_frames - 1))
    segment_count = max(end_frame - start_frame + 1, 1)
    return [start_frame + index for index in sample_frame_indices(segment_count, num_frames, is_train)]


def frame_times_from_indices(indices: Sequence[int], fps: float) -> list[float]:
    if fps <= 0:
        return [0.0 for _ in indices]
    return [float(index) / fps for index in indices]


def offline_provider_absolute_times(
    decoded_frame_times: Tensor | Sequence[float] | None,
    *,
    start_sec: float,
    clip_duration: float,
    num_frames: int,
) -> list[float]:
    """Return absolute video timestamps for offline evidence/teacher queries.

    Video readers derive timestamps directly from absolute source frame ids, so
    their values must not be offset by the sampled window start a second time.
    The synthetic fallback is relative and therefore receives the offset here.
    """
    if decoded_frame_times is not None:
        if torch.is_tensor(decoded_frame_times):
            values = decoded_frame_times.detach().cpu().tolist()
        else:
            values = list(decoded_frame_times)
        if len(values) != int(num_frames):
            raise ValueError(
                "decoded frame time count must match num_frames: "
                f"times={len(values)} num_frames={num_frames}"
            )
        return [float(value) for value in values]
    relative_times = torch.linspace(
        0.0, float(clip_duration), int(num_frames)
    ).tolist()
    return [float(start_sec) + float(value) for value in relative_times]


def canonical_temporal_fusion(value: str) -> str:
    fusion = str(value)
    if fusion == "transformer":
        return "cls_transformer"
    return fusion


def effective_num_frames(cfg: Any) -> int:
    fusion = canonical_temporal_fusion(str(cfg.model.get("temporal_fusion", "cls_transformer")))
    if fusion in {
        "event_topk_transformer",
        "event_anchor_transformer",
        "uniform_event_dual_transformer",
    }:
        return int(cfg.video.get("candidate_num_frames", cfg.video.num_frames))
    return int(cfg.video.num_frames)


def frame_supervision_enabled(cfg: Any) -> bool:
    train_cfg = cfg.get("train", ConfigDict())
    return any(
        float(train_cfg.get(key, 0.0) or 0.0) > 0.0
        for key in (
            "frame_det_loss_weight",
            "spatial_temporal_localization_loss_weight",
            "structured_frame_temporal_localization_loss_weight",
            "object_heatmap_loss_weight",
            "frame_tail_rank_loss_weight",
        )
    )


def object_teacher_provider_from_config(
    cfg: Any,
) -> ObjectTeacherTargetProvider | TrackedBallTargetProvider | BallGoalRelationTargetProvider | None:
    aux_cfg = cfg.model.get("object_spatial_aux", ConfigDict())
    if not bool(aux_cfg.get("enabled", False)):
        return None
    object_weight = float(cfg.train.get("object_heatmap_loss_weight", 0.0) or 0.0)
    relation_weight = float(
        cfg.train.get("ball_goal_relation_loss_weight", 0.0) or 0.0
    )
    if object_weight <= 0.0 and relation_weight <= 0.0:
        return None
    index_root = str(aux_cfg.get("teacher_index_root", "") or "")
    if not index_root:
        raise ValueError("model.object_spatial_aux.teacher_index_root is required")
    if float(cfg.video.get("hflip_prob", 0.0) or 0.0) > 0:
        raise ValueError(
            "object teacher heatmaps currently require video.hflip_prob=0 so "
            "teacher boxes and resized full-image patches remain aligned"
        )
    if str(cfg.get("spatial_crop", ConfigDict()).get("mode", "none")) != "none":
        raise ValueError("object teacher heatmaps require the full-image path (spatial_crop.mode=none)")
    teacher_format = str(aux_cfg.get("teacher_format", "indexed_boxes_v3"))
    if teacher_format == "tracked_ball_goal_relation_v1":
        goal_index_root = str(aux_cfg.get("goal_teacher_index_root", "") or "")
        if not goal_index_root:
            raise ValueError(
                "tracked_ball_goal_relation_v1 requires goal_teacher_index_root"
            )
        return BallGoalRelationTargetProvider(
            index_root,
            goal_index_root,
            image_size=parse_image_size(cfg.video.image_size),
            patch_size=int(aux_cfg.get("patch_size", 16)),
            max_ball_frame_gap_sec=float(aux_cfg.get("max_frame_gap_sec", 0.10)),
            max_goal_frame_gap_sec=float(aux_cfg.get("goal_max_frame_gap_sec", 0.12)),
            ball_sigma_patches=float(aux_cfg.get("ball_sigma_patches", 1.25)),
            goal_dilation_patches=float(aux_cfg.get("goal_dilation_patches", 0.5)),
            ball_min_confidence=float(aux_cfg.get("min_confidence", 0.0)),
            ball_min_quality=float(aux_cfg.get("min_quality", 0.0)),
            relation_min_ball_confidence=float(
                aux_cfg.get("relation_min_ball_confidence", 0.20)
            ),
            relation_min_ball_quality=float(
                aux_cfg.get("relation_min_ball_quality", 0.75)
            ),
            goal_confidence=float(aux_cfg.get("goal_confidence", 0.25)),
            goal_class_id=int(aux_cfg.get("goal_class_id", 2)),
            goal_negative_weight=float(aux_cfg.get("goal_negative_weight", 0.05)),
            teacher_time_offset_sec=float(aux_cfg.get("teacher_time_offset_sec", 0.0)),
            max_cached_videos=int(aux_cfg.get("max_cached_videos", 4)),
        )
    if teacher_format == "tracked_ball_npz_v1":
        return TrackedBallTargetProvider(
            index_root,
            image_size=parse_image_size(cfg.video.image_size),
            patch_size=int(aux_cfg.get("patch_size", 16)),
            max_frame_gap_sec=float(aux_cfg.get("max_frame_gap_sec", 0.10)),
            ball_sigma_patches=float(aux_cfg.get("ball_sigma_patches", 1.25)),
            min_confidence=float(aux_cfg.get("min_confidence", 0.0)),
            min_quality=float(aux_cfg.get("min_quality", 0.0)),
            teacher_time_offset_sec=float(
                aux_cfg.get("teacher_time_offset_sec", 0.0)
            ),
            max_cached_videos=int(aux_cfg.get("max_cached_videos", 4)),
        )
    if teacher_format != "indexed_boxes_v3":
        raise ValueError(
            "model.object_spatial_aux.teacher_format must be indexed_boxes_v3 "
            "tracked_ball_npz_v1, or tracked_ball_goal_relation_v1"
        )
    return ObjectTeacherTargetProvider(
        index_root,
        image_size=parse_image_size(cfg.video.image_size),
        patch_size=int(aux_cfg.get("patch_size", 16)),
        ball_class_id=int(aux_cfg.get("ball_class_id", 1)),
        goal_class_id=int(aux_cfg.get("goal_class_id", 2)),
        ball_confidence=float(aux_cfg.get("ball_confidence", 0.10)),
        goal_confidence=float(aux_cfg.get("goal_confidence", 0.20)),
        max_frame_gap_sec=float(aux_cfg.get("max_frame_gap_sec", 0.65)),
        ball_sigma_patches=float(aux_cfg.get("ball_sigma_patches", 1.25)),
        goal_dilation_patches=float(aux_cfg.get("goal_dilation_patches", 0.5)),
        max_cached_videos=int(aux_cfg.get("max_cached_videos", 2)),
    )



def online_object_teacher_from_config(
    cfg: Any, device: torch.device
) -> OnlineObjectTeacherTargeter | None:
    aux_cfg = cfg.model.get("object_spatial_aux", ConfigDict())
    online_cfg = aux_cfg.get("online_missing_teacher", ConfigDict())
    if not bool(aux_cfg.get("enabled", False)) or not bool(
        online_cfg.get("enabled", False)
    ):
        return None
    if not normalize_on_device_enabled(cfg):
        raise ValueError(
            "online object Teacher requires uint8 dataloader inputs; "
            "set data.preprocessing.normalize_on_device=true"
        )
    return OnlineObjectTeacherTargeter(
        device=device,
        ball_checkpoint=str(aux_cfg.get("ball_teacher_checkpoint", "")),
        goal_checkpoint=str(aux_cfg.get("goal_teacher_checkpoint", "")),
        patch_size=int(aux_cfg.get("patch_size", 16)),
        ball_confidence=float(aux_cfg.get("ball_confidence", 0.10)),
        goal_confidence=float(aux_cfg.get("goal_confidence", 0.25)),
        ball_sigma_patches=float(aux_cfg.get("ball_sigma_patches", 1.25)),
        goal_dilation_patches=float(aux_cfg.get("goal_dilation_patches", 0.5)),
        input_size=online_cfg.get("input_size", [1088, 1920]),
        batch_size=int(online_cfg.get("batch_size", 16)),
        half=bool(online_cfg.get("half", True)),
        iou=float(online_cfg.get("iou", 0.70)),
    )

def label_float_map(raw: Any, default: dict[str, float]) -> dict[str, float]:
    if raw is None:
        return {label: float(default[label]) for label in LABELS}
    if isinstance(raw, (int, float)):
        return {label: float(raw) for label in LABELS}
    if isinstance(raw, dict):
        result = {label: float(default.get(label, next(iter(default.values())))) for label in LABELS}
        for label, value in raw.items():
            if label in result:
                result[label] = float(value)
        return result
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = [float(value) for value in raw]
        if len(values) != len(LABELS):
            raise ValueError(f"Expected {len(LABELS)} values, got {len(values)}")
        return {label: values[index] for index, label in enumerate(LABELS)}
    raise ValueError(f"Unsupported label float map: {raw}")


def frame_label_sigma_seconds(cfg: Any) -> dict[str, float]:
    defaults = {"shot": 0.8, "save": 0.8, "set_piece": 1.5}
    fallback = {label: defaults.get(label, 1.0) for label in LABELS}
    return label_float_map(cfg.get("train", ConfigDict()).get("frame_label_sigma_sec"), fallback)


def frame_label_ignore_radius_seconds(cfg: Any) -> dict[str, float]:
    defaults = {"shot": 3.0, "save": 3.0, "set_piece": 5.0}
    fallback = {label: defaults.get(label, 3.0) for label in LABELS}
    return label_float_map(cfg.get("train", ConfigDict()).get("frame_label_ignore_radius_sec"), fallback)

def frame_label_time_jitter_seconds(cfg: Any) -> dict[str, float]:
    fallback = {label: 0.0 for label in LABELS}
    return label_float_map(
        cfg.get("train", ConfigDict()).get("frame_label_time_jitter_sec"), fallback
    )



def gaussian_frame_targets(
    events: Sequence[FootballEvent],
    start_sec: float,
    end_sec: float,
    frame_times: Sequence[float],
    label_mask: Sequence[float],
    sigma_sec: dict[str, float],
    ignore_radius_sec: dict[str, float],
) -> tuple[Tensor, Tensor]:
    times = torch.tensor(list(frame_times), dtype=torch.float32)
    targets = torch.zeros((len(times), len(LABELS)), dtype=torch.float32)
    masks = torch.zeros_like(targets)
    events_by_label: dict[str, list[float]] = {label: [] for label in LABELS}
    for event in events:
        if event.is_ignored or not (start_sec <= float(event.anchor_time) <= end_sec):
            continue
        for label_index, value in enumerate(event.labels):
            if float(value) > 0:
                events_by_label[LABELS[label_index]].append(float(event.anchor_time))

    for label_index, label in enumerate(LABELS):
        if label_index >= len(label_mask) or float(label_mask[label_index]) <= 0:
            continue
        anchors = events_by_label[label]
        if not anchors:
            masks[:, label_index] = 1.0
            continue
        sigma = max(float(sigma_sec.get(label, 1.0)), 1e-6)
        radius = max(float(ignore_radius_sec.get(label, 3.0)), 0.0)
        anchor_tensor = torch.tensor(anchors, dtype=torch.float32)
        distances = torch.abs(times[:, None] - anchor_tensor[None, :])
        min_distances = distances.min(dim=1).values
        heatmap = torch.exp(-0.5 * (min_distances / sigma).pow(2))
        targets[:, label_index] = heatmap
        masks[:, label_index] = (min_distances <= radius).float()

    return targets, masks

def jitter_frame_supervision_events(
    events: Sequence[FootballEvent],
    start_sec: float,
    end_sec: float,
    video_duration: float,
    jitter_sec: dict[str, float],
) -> Sequence[FootballEvent]:
    if not any(float(value) > 0.0 for value in jitter_sec.values()):
        return events
    jittered: list[FootballEvent] = []
    for event in events:
        if event.is_ignored:
            jittered.append(event)
            continue
        anchor = float(event.anchor_time)
        new_anchor = anchor
        if start_sec <= anchor <= end_sec:
            max_jitter = 0.0
            for label_index, value in enumerate(event.labels):
                if float(value) > 0.0 and label_index < len(LABELS):
                    max_jitter = max(max_jitter, float(jitter_sec.get(LABELS[label_index], 0.0)))
            if max_jitter > 0.0:
                new_anchor = clamp(
                    anchor + random.uniform(-max_jitter, max_jitter),
                    max(float(start_sec), 0.0),
                    min(float(end_sec), float(video_duration)),
                )
        if abs(new_anchor - anchor) <= 1e-6:
            jittered.append(event)
            continue
        delta = new_anchor - anchor
        jittered.append(
            FootballEvent(
                source=event.source,
                video_id=event.video_id,
                event_id=event.event_id,
                event_type=event.event_type,
                raw_label=event.raw_label,
                start_time=max(float(event.start_time) + delta, 0.0),
                end_time=max(float(event.end_time) + delta, 0.0),
                anchor_time=new_anchor,
                labels=event.labels,
                context_start_time=event.context_start_time,
                context_end_time=event.context_end_time,
                is_ignored=event.is_ignored,
            )
        )
    return jittered


def _zero_video(num_frames: int, image_size: tuple[int, int], *, normalize: bool = True) -> Tensor:
    height, width = image_size
    dtype = torch.float32 if normalize else torch.uint8
    return torch.zeros(num_frames, 3, height, width, dtype=dtype)


class VideoDecodeError(RuntimeError):
    """Raised when a sampled video segment is not usable for supervision."""


def _open_video_capture(path: str | Path) -> cv2.VideoCapture:
    # Open/read timeouts must be supplied to the FFMPEG constructor. Setting
    # CAP_PROP_OPEN_TIMEOUT_MSEC after VideoCapture(path) returns is too late:
    # a corrupt or large MP4 can already be blocked inside the constructor.
    params: list[int] = []
    for prop_name, value in (
        ("CAP_PROP_OPEN_TIMEOUT_MSEC", 10000),
        ("CAP_PROP_READ_TIMEOUT_MSEC", 10000),
    ):
        prop = getattr(cv2, prop_name, None)
        if prop is not None:
            params.extend([int(prop), int(value)])
    try:
        return cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, params)
    except (TypeError, cv2.error):
        # Older OpenCV builds may not accept constructor parameters. Keep a
        # compatibility fallback, although its open timeout cannot be enforced.
        cap = cv2.VideoCapture(str(path))
        for index in range(0, len(params), 2):
            try:
                cap.set(params[index], params[index + 1])
            except cv2.error:
                pass
        return cap


# Wall-clock decode guard.  DataLoader workers are separate single-threaded
# processes, so a thread-local deadline is enough to bound the total time a
# single decode attempt may spend (cap.set/cap.read plus geometry probes).
# Not set -> no deadline -> no-op.
_decode_deadline = threading.local()


def set_decode_deadline(timeout_sec: float) -> None:
    timeout = float(timeout_sec or 0.0)
    if timeout > 0.0:
        _decode_deadline.deadline = time.monotonic() + timeout
    else:
        _decode_deadline.deadline = None


def check_decode_deadline() -> None:
    deadline = getattr(_decode_deadline, "deadline", None)
    if deadline is None:
        return
    overdue = time.monotonic() - deadline
    if overdue > 0.0:
        raise VideoDecodeError(
            f"decode attempt exceeded wall-clock deadline by {overdue:.1f}s"
        )


def _ensure_usable_decode(
    decoded_frames: Sequence[np.ndarray | None],
    *,
    path: str,
    min_valid_ratio: float = 0.5,
) -> None:
    total = len(decoded_frames)
    valid = sum(frame is not None for frame in decoded_frames)
    required = max(1, int(math.ceil(float(total) * float(min_valid_ratio)))) if total > 0 else 1
    if valid < required:
        raise VideoDecodeError(
            f"decoded too few valid frames for {path}: valid={valid}/{total} required={required}"
        )


def _is_video_decode_exception(exc: BaseException) -> bool:
    if isinstance(exc, (VideoDecodeError, FileNotFoundError, OSError, cv2.error)):
        return True
    text = str(exc).lower()
    needles = (
        "video",
        "decode",
        "opencv",
        "ffmpeg",
        "nal",
        "partial file",
        "missing picture",
        "invalid video geometry",
        "could not open",
    )
    return isinstance(exc, RuntimeError) and any(needle in text for needle in needles)


class VideoCaptureCache:
    def __init__(self, max_size: int):
        self.max_size = max(0, int(max_size))
        self._items: OrderedDict[str, cv2.VideoCapture] = OrderedDict()

    def get(self, path: str) -> cv2.VideoCapture:
        cap = self._items.get(path)
        if cap is not None and cap.isOpened():
            self._items.move_to_end(path)
            return cap
        if cap is not None:
            cap.release()
            del self._items[path]

        cap = _open_video_capture(path)
        if self.max_size > 0 and cap.isOpened():
            self._items[path] = cap
            self._items.move_to_end(path)
            while len(self._items) > self.max_size:
                _, old_cap = self._items.popitem(last=False)
                old_cap.release()
        return cap

    def close(self) -> None:
        for cap in self._items.values():
            cap.release()
        self._items.clear()

    def __del__(self):
        self.close()


def _decode_frames_multi_seek(cap: cv2.VideoCapture, indices: Sequence[int]) -> list[np.ndarray | None]:
    frames: list[np.ndarray | None] = []
    for index in indices:
        check_decode_deadline()
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
        except cv2.error:
            ok, frame = False, None
        frames.append(frame if ok and frame is not None else None)
    return frames


def _decode_frames_single_seek(cap: cv2.VideoCapture, indices: Sequence[int]) -> list[np.ndarray | None]:
    if not indices:
        return []
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(indices[0]))
    current = int(indices[0])
    last_frame: np.ndarray | None = None
    frames: list[np.ndarray | None] = []
    for raw_index in indices:
        check_decode_deadline()
        index = int(raw_index)
        if last_frame is not None and index < current:
            frames.append(last_frame.copy())
            continue
        ok = True
        while current < index:
            try:
                ok = cap.grab()
            except cv2.error:
                ok = False
            current += 1
            if not ok:
                break
        if not ok:
            frames.append(None)
            continue
        try:
            ok, frame = cap.read()
        except cv2.error:
            ok, frame = False, None
        current += 1
        if ok and frame is not None:
            last_frame = frame
            frames.append(frame)
        else:
            frames.append(None)
    return frames


def _decode_video_frames(cap: cv2.VideoCapture, indices: Sequence[int], strategy: str) -> list[np.ndarray | None]:
    if strategy == "single_seek":
        return _decode_frames_single_seek(cap, indices)
    if strategy == "multi_seek":
        return _decode_frames_multi_seek(cap, indices)
    raise ValueError("data.video_decode_strategy must be one of: multi_seek, single_seek")


class DetectorAwareCropper:
    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.manifest_root = Path(cfg.manifest_root).expanduser()
        self.ball_conf = float(cfg.get("ball_conf", 0.5))
        self.goal_conf = float(cfg.get("goal_conf", 0.5))
        self.person_conf = float(cfg.get("person_conf", 0.4))
        self.padding = float(cfg.get("padding", 0.12))
        self.min_crop_area_ratio = float(cfg.get("min_crop_area_ratio", 0.15))
        self.max_crop_area_ratio = float(cfg.get("max_crop_area_ratio", 0.85))
        self.max_frame_gap = int(cfg.get("max_frame_gap", 5))
        self.use_detection_goals = bool(cfg.get("use_detection_goals", False))
        self._cache: dict[str, dict[int, list[dict[str, Any]]]] = {}
        self._metadata_cache: dict[str, dict[str, Any]] = {}

    @classmethod
    def from_config(cls, cfg: Any) -> "DetectorAwareCropper | None":
        spatial_cfg = cfg.get("spatial_crop", ConfigDict())
        if str(spatial_cfg.get("mode", "none")) != "detector_aware":
            return None
        return cls(spatial_cfg)

    @staticmethod
    def _dedupe_existing_paths(paths: Sequence[Path]) -> list[Path]:
        deduped: list[Path] = []
        seen: set[Path] = set()
        for path in paths:
            if path.exists() and path not in seen:
                deduped.append(path)
                seen.add(path)
        return deduped

    def _candidate_paths(self, video_id: str) -> list[Path]:
        root = self.manifest_root / video_id
        paths = [root / "tracked_objects.json"]
        paths.extend(sorted((root / "detection_tracking").glob("*/tracked_objects.json")))
        paths.extend(sorted(root.glob("**/tracked_objects.json")))
        return self._dedupe_existing_paths(paths)

    def _detection_candidate_paths(self, video_id: str) -> list[Path]:
        root = self.manifest_root / video_id
        paths = [root / "detections.json"]
        paths.extend(sorted((root / "detection_tracking").glob("*/detections.json")))
        paths.extend(sorted(root.glob("**/detections.json")))
        return self._dedupe_existing_paths(paths)

    def _metadata_paths(self, video_id: str) -> list[Path]:
        root = self.manifest_root / video_id
        paths = [root / "metadata.json", root / "trajectory_results.json"]
        paths.extend(sorted(root.glob("**/metadata.json")))
        paths.extend(sorted(root.glob("**/trajectory_results.json")))
        deduped: list[Path] = []
        seen: set[Path] = set()
        for path in paths:
            if path.exists() and path not in seen:
                deduped.append(path)
                seen.add(path)
        return deduped

    def _load_metadata(self, video_id: str) -> dict[str, Any]:
        if video_id in self._metadata_cache:
            return self._metadata_cache[video_id]
        metadata: dict[str, Any] = {}
        for path in self._metadata_paths(video_id):
            with path.open() as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                metadata = payload
                break
        self._metadata_cache[video_id] = metadata
        return metadata

    def _frame_lookup_gap(self, video_id: str) -> int:
        metadata = self._load_metadata(video_id)
        sampling = metadata.get("sampling", {}) if isinstance(metadata, dict) else {}
        stride = int(sampling.get("frame_stride", 1) or 1) if isinstance(sampling, dict) else 1
        return max(self.max_frame_gap, stride // 2 + 1)

    def _scale_bbox(self, video_id: str, bbox: Sequence[float], width: int, height: int) -> list[float]:
        metadata = self._load_metadata(video_id)
        image_size = metadata.get("image_size", {}) if isinstance(metadata, dict) else {}
        det_w = float(image_size.get("width", width) or width) if isinstance(image_size, dict) else float(width)
        det_h = float(image_size.get("height", height) or height) if isinstance(image_size, dict) else float(height)
        if det_w <= 0 or det_h <= 0 or (abs(det_w - width) < 1e-3 and abs(det_h - height) < 1e-3):
            return [float(v) for v in bbox[:4]]
        sx = float(width) / det_w
        sy = float(height) / det_h
        return [float(bbox[0]) * sx, float(bbox[1]) * sy, float(bbox[2]) * sx, float(bbox[3]) * sy]

    @staticmethod
    def _item_objects(item: dict[str, Any]) -> list[dict[str, Any]]:
        objects = item.get("objects")
        if objects is None:
            objects = item.get("detections")
        if objects is None:
            objects = item.get("Detect4in1")
        return objects or []

    def _load_frame_objects_from_paths(self, paths: Sequence[Path], keep_classes: set[int] | None = None) -> dict[int, list[dict[str, Any]]]:
        frames: dict[int, list[dict[str, Any]]] = {}
        offset = 0
        for path in paths:
            with path.open() as f:
                payload = json.load(f)
            items = payload.get("frames", payload) if isinstance(payload, dict) else payload
            local_max = -1
            for item in items:
                if not isinstance(item, dict):
                    continue
                frame_id = int(item.get("frame_id", item.get("frame", 0)))
                local_max = max(local_max, frame_id)
                objects = []
                for obj in self._item_objects(item):
                    if not isinstance(obj, dict):
                        continue
                    if keep_classes is not None:
                        cls_id = int(obj.get("cls", obj.get("class", -1)))
                        if cls_id not in keep_classes:
                            continue
                    objects.append(obj)
                if objects:
                    frames.setdefault(offset + frame_id, []).extend(objects)
            if len(paths) > 1 and local_max >= 0:
                offset += local_max + 1
        return frames

    def _load_video(self, video_id: str) -> dict[int, list[dict[str, Any]]]:
        if video_id in self._cache:
            return self._cache[video_id]
        frames = self._load_frame_objects_from_paths(self._candidate_paths(video_id))
        if self.use_detection_goals:
            # Optional fallback for legacy outputs where tracked_objects.json has no goal boxes.
            for frame_id, objects in self._load_frame_objects_from_paths(self._detection_candidate_paths(video_id), keep_classes={2}).items():
                frames.setdefault(frame_id, []).extend(objects)
        self._cache[video_id] = frames
        return frames

    def _objects_for_frame(self, video_id: str, frame_id: int) -> list[dict[str, Any]]:
        frames = self._load_video(video_id)
        if frame_id in frames:
            return frames[frame_id]
        lookup_gap = self._frame_lookup_gap(video_id)
        if not frames or lookup_gap <= 0:
            return []
        nearest = min(frames, key=lambda key: abs(key - frame_id))
        if abs(nearest - frame_id) <= lookup_gap:
            return frames[nearest]
        return []

    @staticmethod
    def _area(box: Sequence[float]) -> float:
        return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))

    @staticmethod
    def _union(boxes: Sequence[Sequence[float]]) -> list[float]:
        return [
            min(float(box[0]) for box in boxes),
            min(float(box[1]) for box in boxes),
            max(float(box[2]) for box in boxes),
            max(float(box[3]) for box in boxes),
        ]

    @staticmethod
    def _intersects(a: Sequence[float], b: Sequence[float]) -> bool:
        return min(float(a[2]), float(b[2])) > max(float(a[0]), float(b[0])) and min(float(a[3]), float(b[3])) > max(float(a[1]), float(b[1]))

    @staticmethod
    def _expand(box: Sequence[float], ratio: float, width: int, height: int) -> list[float]:
        x1, y1, x2, y2 = map(float, box)
        pad_x = (x2 - x1) * ratio
        pad_y = (y2 - y1) * ratio
        return [max(0.0, x1 - pad_x), max(0.0, y1 - pad_y), min(float(width), x2 + pad_x), min(float(height), y2 + pad_y)]

    def get_roi(self, video_id: str, frame_id: int, width: int, height: int) -> tuple[int, int, int, int] | None:
        objects = self._objects_for_frame(video_id, frame_id)
        if not objects:
            return None
        persons, balls, goals = [], [], []
        for obj in objects:
            bbox = obj.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            bbox = self._scale_bbox(video_id, bbox, width, height)
            cls_id = int(obj.get("cls", obj.get("class", -1)))
            conf = float(obj.get("conf", obj.get("score", 0.0)))
            if cls_id == 0 and conf >= self.person_conf:
                persons.append(bbox)
            elif cls_id == 1 and conf >= self.ball_conf:
                balls.append(bbox)
            elif cls_id == 2 and conf >= self.goal_conf:
                goals.append(bbox)
        if not goals:
            return None

        goal = max(goals, key=self._area)
        boxes = [goal]
        boxes.extend(balls)
        reference = self._expand(self._union(boxes), 0.35, width, height)
        for person in persons:
            cx = (float(person[0]) + float(person[2])) * 0.5
            cy = (float(person[1]) + float(person[3])) * 0.5
            if self._intersects(person, reference) or (reference[0] <= cx <= reference[2] and reference[1] <= cy <= reference[3]):
                boxes.append(person)

        roi = self._expand(self._union(boxes), self.padding, width, height)
        area_ratio = self._area(roi) / max(float(width * height), 1.0)
        if area_ratio > self.max_crop_area_ratio:
            return None
        if area_ratio < self.min_crop_area_ratio:
            cx = (roi[0] + roi[2]) * 0.5
            cy = (roi[1] + roi[3]) * 0.5
            target_area = self.min_crop_area_ratio * width * height
            aspect = max((roi[2] - roi[0]) / max(roi[3] - roi[1], 1.0), 1e-3)
            target_w = min(width, max(roi[2] - roi[0], (target_area * aspect) ** 0.5))
            target_h = min(height, max(roi[3] - roi[1], target_w / aspect))
            roi = [cx - target_w * 0.5, cy - target_h * 0.5, cx + target_w * 0.5, cy + target_h * 0.5]
        x1 = int(clamp(round(roi[0]), 0, width - 1))
        y1 = int(clamp(round(roi[1]), 0, height - 1))
        x2 = int(clamp(round(roi[2]), x1 + 1, width))
        y2 = int(clamp(round(roi[3]), y1 + 1, height))
        return x1, y1, x2, y2


class FixedCropProvider:
    """Apply one immutable ROI to every frame in a clip."""
    def __init__(self, roi: tuple[int, int, int, int] | None):
        self.roi = roi
    def get_roi(self, video_id: str, frame_id: int, width: int, height: int) -> tuple[int, int, int, int] | None:
        return self.roi


class FrameCropProvider:
    """Apply frame-aligned ROIs using the exact sampled source frame ids."""

    def __init__(
        self,
        frame_indices: Sequence[int],
        rois: Sequence[tuple[int, int, int, int] | None],
    ):
        if len(frame_indices) != len(rois):
            raise ValueError("frame_indices and rois must have the same length")
        self.rois = {int(frame_id): roi for frame_id, roi in zip(frame_indices, rois)}

    def get_roi(
        self,
        video_id: str,
        frame_id: int,
        width: int,
        height: int,
    ) -> tuple[int, int, int, int] | None:
        return self.rois.get(int(frame_id))


class TopBandCropProvider:
    """Apply one resolution-independent top crop to every frame."""

    def __init__(self, top_crop_ratio: float):
        ratio = float(top_crop_ratio)
        if not 0.0 < ratio < 0.5:
            raise ValueError("spatial_crop.top_crop_ratio must be in (0, 0.5)")
        self.top_crop_ratio = ratio

    @classmethod
    def from_config(cls, cfg: Any) -> "TopBandCropProvider | None":
        spatial_cfg = cfg.get("spatial_crop", ConfigDict())
        if str(spatial_cfg.get("mode", "none")).strip().lower() != "top_fixed":
            return None
        return cls(float(spatial_cfg.get("top_crop_ratio", 0.0) or 0.0))

    def get_roi(
        self,
        video_id: str,
        frame_id: int,
        width: int,
        height: int,
    ) -> tuple[int, int, int, int]:
        del video_id, frame_id
        if width <= 0 or height <= 1:
            raise ValueError(f"invalid frame geometry for top crop: {width}x{height}")
        crop_y = int(round(float(height) * self.top_crop_ratio))
        crop_y = min(max(crop_y, 1), height - 1)
        return 0, crop_y, width, height


def read_video_segment(
    path: str,
    num_frames: int,
    image_size: tuple[int, int],
    is_train: bool,
    hflip_prob: float,
    *,
    start_sec: float | None = None,
    end_sec: float | None = None,
    frame_indices: Sequence[int] | None = None,
    video_id: str = "",
    crop_provider: DetectorAwareCropper | TopBandCropProvider | None = None,
    detection_hint_renderer: DetectionHintRenderer | None = None,
    normalize: bool = True,
    cap_cache: VideoCaptureCache | None = None,
    decode_strategy: str = "multi_seek",
    return_frame_times: bool = False,
    hflip_override: bool | None = None,
) -> Tensor | tuple[Tensor, Tensor]:
    cap = cap_cache.get(path) if cap_cache is not None else _open_video_capture(path)
    release_cap = cap_cache is None
    if not cap.isOpened():
        cap.release()
        raise VideoDecodeError(f"could not open video: {path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if total_frames <= 0 or fps <= 0:
        if release_cap:
            cap.release()
        raise VideoDecodeError(f"invalid video metadata for {path}: frames={total_frames} fps={fps}")

    if frame_indices is None:
        indices = segment_frame_indices(
            total_frames, fps, num_frames, is_train, start_sec=start_sec, end_sec=end_sec
        )
    else:
        if len(frame_indices) != num_frames:
            raise ValueError(f"frame_indices length={len(frame_indices)} must equal num_frames={num_frames}")
        indices = [int(clamp(index, 0, total_frames - 1)) for index in frame_indices]
    frame_times = frame_times_from_indices(indices, fps)
    out_h, out_w = image_size
    frames: list[Tensor] = []
    do_hflip = (
        bool(hflip_override)
        if hflip_override is not None
        else is_train and random.random() < hflip_prob
    )

    decoded_frames = _decode_video_frames(cap, indices, decode_strategy)
    try:
        _ensure_usable_decode(decoded_frames, path=path)
    except Exception:
        if release_cap:
            cap.release()
        raise
    for index, frame in zip(indices, decoded_frames):
        if frame is None:
            dtype = torch.float32 if normalize else torch.uint8
            frames.append(torch.zeros(3, out_h, out_w, dtype=dtype))
            continue
        if detection_hint_renderer is not None:
            frame = detection_hint_renderer.render(
                frame, video_id, index, is_train=is_train
            )
        if crop_provider is not None:
            height, width = frame.shape[:2]
            roi = crop_provider.get_roi(video_id, index, width, height)
            if roi is not None:
                x1, y1, x2, y2 = roi
                frame = frame[y1:y2, x1:x2]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
        if do_hflip:
            frame = cv2.flip(frame, 1)
        tensor = torch.from_numpy(frame).permute(2, 0, 1)
        if normalize:
            tensor = tensor.float().div_(255.0)
            tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        frames.append(tensor)
    if release_cap:
        cap.release()
    stacked = torch.stack(frames, dim=0)
    if return_frame_times:
        return stacked, torch.tensor(frame_times, dtype=torch.float32)
    return stacked


def read_video_global_highres_pool(
    path: str,
    global_frames: int,
    pool_frames: int,
    global_size: tuple[int, int],
    pool_size: tuple[int, int],
    is_train: bool,
    hflip_prob: float,
    *,
    start_sec: float,
    end_sec: float,
    video_id: str = "",
    detection_hint_renderer: DetectionHintRenderer | None = None,
    normalize_global: bool = True,
    cap_cache: VideoCaptureCache | None = None,
    decode_strategy: str = "single_seek",
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Decode both views from one union of source indices.

    The dense pool stays uint8 until the model selects eight frames, avoiding
    normalization and DINO encoding for the unselected source pixels.
    """
    cap = cap_cache.get(path) if cap_cache is not None else _open_video_capture(path)
    release_cap = cap_cache is None
    if not cap.isOpened():
        cap.release()
        raise VideoDecodeError(f"could not open video: {path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if total_frames <= 0 or fps <= 0:
        if release_cap:
            cap.release()
        raise VideoDecodeError(f"invalid video metadata for {path}: frames={total_frames} fps={fps}")
    global_indices = segment_frame_indices(
        total_frames, fps, global_frames, is_train,
        start_sec=start_sec, end_sec=end_sec,
    )
    pool_indices = segment_frame_indices(
        total_frames, fps, pool_frames, is_train,
        start_sec=start_sec, end_sec=end_sec,
    )
    union_indices = sorted(set(global_indices + pool_indices))
    decoded = _decode_video_frames(cap, union_indices, decode_strategy)
    try:
        _ensure_usable_decode(decoded, path=path)
    except Exception:
        if release_cap:
            cap.release()
        raise
    if release_cap:
        cap.release()
    decoded_by_index = dict(zip(union_indices, decoded))
    do_hflip = is_train and random.random() < hflip_prob

    def render(indices: Sequence[int], size: tuple[int, int], *, global_view: bool) -> Tensor:
        out_h, out_w = size
        output: list[Tensor] = []
        for frame_index in indices:
            frame = decoded_by_index.get(frame_index)
            if frame is None:
                dtype = torch.float32 if global_view and normalize_global else torch.uint8
                output.append(torch.zeros(3, out_h, out_w, dtype=dtype))
                continue
            if global_view and detection_hint_renderer is not None:
                frame = detection_hint_renderer.render(
                    frame.copy(), video_id, frame_index, is_train=is_train
                )
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
            if do_hflip:
                frame = cv2.flip(frame, 1)
            tensor = torch.from_numpy(frame).permute(2, 0, 1)
            if global_view and normalize_global:
                tensor = tensor.float().div_(255.0)
                tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
            output.append(tensor)
        return torch.stack(output)

    return (
        render(global_indices, global_size, global_view=True),
        torch.tensor(frame_times_from_indices(global_indices, fps), dtype=torch.float32),
        render(pool_indices, pool_size, global_view=False),
        torch.tensor(frame_times_from_indices(pool_indices, fps), dtype=torch.float32),
    )


def read_video_segment_views(
    path: str,
    num_frames: int,
    global_image_size: tuple[int, int],
    local_image_size: tuple[int, int],
    roi: tuple[int, int, int, int] | Sequence[tuple[int, int, int, int] | None] | None,
    is_train: bool,
    hflip_prob: float,
    *,
    start_sec: float,
    end_sec: float,
    frame_indices: Sequence[int] | None = None,
    normalize: bool = True,
    cap_cache: VideoCaptureCache | None = None,
    decode_strategy: str = "multi_seek",
    return_frame_times: bool = False,
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
    """Decode once and create temporally aligned global/local clip views."""
    cap = cap_cache.get(path) if cap_cache is not None else _open_video_capture(path)
    release_cap = cap_cache is None
    if not cap.isOpened():
        if release_cap:
            cap.release()
        raise VideoDecodeError(f"could not open video: {path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if total_frames <= 0 or fps <= 0:
        if release_cap:
            cap.release()
        raise VideoDecodeError(f"invalid video metadata for {path}: frames={total_frames} fps={fps}")
    if frame_indices is None:
        indices = segment_frame_indices(
            total_frames, fps, num_frames, is_train, start_sec=start_sec, end_sec=end_sec
        )
    else:
        if len(frame_indices) != num_frames:
            raise ValueError(f"frame_indices length={len(frame_indices)} must equal num_frames={num_frames}")
        indices = [int(clamp(index, 0, total_frames - 1)) for index in frame_indices]
    frame_times = frame_times_from_indices(indices, fps)
    if roi is not None and isinstance(roi, Sequence) and len(roi) == num_frames and (
        len(roi) == 0 or roi[0] is None or isinstance(roi[0], (tuple, list))
    ):
        frame_rois = list(roi)
    else:
        frame_rois = [roi for _ in indices]
    do_hflip = is_train and random.random() < hflip_prob
    decoded_frames = _decode_video_frames(cap, indices, decode_strategy)
    try:
        _ensure_usable_decode(decoded_frames, path=path)
    except Exception:
        if release_cap:
            cap.release()
        raise
    global_frames: list[Tensor] = []
    local_frames: list[Tensor] = []

    def convert(frame: np.ndarray, image_size: tuple[int, int]) -> Tensor:
        out_h, out_w = image_size
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
        if do_hflip:
            rgb = cv2.flip(rgb, 1)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1)
        if normalize:
            tensor = tensor.float().div_(255.0)
            tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        return tensor

    for frame, frame_roi in zip(decoded_frames, frame_rois):
        if frame is None:
            global_frames.append(_zero_video(1, global_image_size, normalize=normalize)[0])
            local_frames.append(_zero_video(1, local_image_size, normalize=normalize)[0])
            continue
        global_frames.append(convert(frame, global_image_size))
        local_frame = frame
        if frame_roi is not None:
            height, width = frame.shape[:2]
            x1 = int(clamp(frame_roi[0], 0, width - 1))
            y1 = int(clamp(frame_roi[1], 0, height - 1))
            x2 = int(clamp(frame_roi[2], x1 + 1, width))
            y2 = int(clamp(frame_roi[3], y1 + 1, height))
            local_frame = frame[y1:y2, x1:x2]
        local_frames.append(convert(local_frame, local_image_size))
    if release_cap:
        cap.release()
    global_stacked = torch.stack(global_frames)
    local_stacked = torch.stack(local_frames)
    if return_frame_times:
        return global_stacked, local_stacked, torch.tensor(frame_times, dtype=torch.float32)
    return global_stacked, local_stacked


def read_video_segment_multi_roi_views(
    path: str,
    num_frames: int,
    global_image_size: tuple[int, int],
    local_image_size: tuple[int, int],
    roi_a: Sequence[tuple[int, int, int, int] | None],
    roi_b: Sequence[tuple[int, int, int, int] | None],
    is_train: bool,
    hflip_prob: float,
    *,
    start_sec: float,
    end_sec: float,
    frame_indices: Sequence[int],
    roi_a_frame_indices: Sequence[int] | None = None,
    roi_b_frame_indices: Sequence[int] | None = None,
    normalize: bool = True,
    cap_cache: VideoCaptureCache | None = None,
    decode_strategy: str = "multi_seek",
    return_frame_times: bool = False,
) -> tuple[Tensor, ...]:
    """Decode global and two ROI timelines once, with shared augmentation."""
    del start_sec, end_sec
    roi_a_frame_indices = (
        frame_indices if roi_a_frame_indices is None else roi_a_frame_indices
    )
    roi_b_frame_indices = (
        frame_indices if roi_b_frame_indices is None else roi_b_frame_indices
    )
    if (
        len(frame_indices) != num_frames
        or len(roi_a_frame_indices) != num_frames
        or len(roi_b_frame_indices) != num_frames
        or len(roi_a) != num_frames
        or len(roi_b) != num_frames
    ):
        raise ValueError("global/ROI frame indices and ROI boxes must match num_frames")
    cap = cap_cache.get(path) if cap_cache is not None else _open_video_capture(path)
    release_cap = cap_cache is None

    def empty_outputs() -> tuple[Tensor, ...]:
        outputs: tuple[Tensor, ...] = (
            _zero_video(num_frames, global_image_size, normalize=normalize),
            _zero_video(num_frames, local_image_size, normalize=normalize),
            _zero_video(num_frames, local_image_size, normalize=normalize),
        )
        if return_frame_times:
            zero_times = torch.zeros(num_frames, dtype=torch.float32)
            return *outputs, zero_times, zero_times.clone(), zero_times.clone()
        return outputs

    if not cap.isOpened():
        if release_cap:
            cap.release()
        raise VideoDecodeError(f"could not open video: {path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if total_frames <= 0 or fps <= 0:
        if release_cap:
            cap.release()
        raise VideoDecodeError(f"invalid video metadata for {path}: frames={total_frames} fps={fps}")

    def clamp_indices(raw: Sequence[int]) -> list[int]:
        return [int(clamp(index, 0, total_frames - 1)) for index in raw]

    global_indices = clamp_indices(frame_indices)
    local_a_indices = clamp_indices(roi_a_frame_indices)
    local_b_indices = clamp_indices(roi_b_frame_indices)
    unique_indices = sorted(set(global_indices + local_a_indices + local_b_indices))
    decoded = _decode_video_frames(cap, unique_indices, decode_strategy)
    try:
        _ensure_usable_decode(decoded, path=path)
    except Exception:
        if release_cap:
            cap.release()
        raise
    decoded_by_index = dict(zip(unique_indices, decoded))
    do_hflip = is_train and random.random() < hflip_prob

    def convert(frame: np.ndarray, image_size: tuple[int, int]) -> Tensor:
        out_h, out_w = image_size
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
        if do_hflip:
            rgb = cv2.flip(rgb, 1)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1)
        if normalize:
            tensor = tensor.float().div_(255.0)
            tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        return tensor

    def crop(
        frame: np.ndarray, region: tuple[int, int, int, int] | None
    ) -> np.ndarray | None:
        if region is None:
            return None
        height, width = frame.shape[:2]
        x1 = int(clamp(region[0], 0, width - 1))
        y1 = int(clamp(region[1], 0, height - 1))
        x2 = int(clamp(region[2], x1 + 1, width))
        y2 = int(clamp(region[3], y1 + 1, height))
        return frame[y1:y2, x1:x2]

    global_frames: list[Tensor] = []
    local_a_frames: list[Tensor] = []
    local_b_frames: list[Tensor] = []
    for index in global_indices:
        frame = decoded_by_index.get(index)
        global_frames.append(
            convert(frame, global_image_size)
            if frame is not None
            else _zero_video(1, global_image_size, normalize=normalize)[0]
        )
    for index, region in zip(local_a_indices, roi_a):
        frame = decoded_by_index.get(index)
        region_frame = crop(frame, region) if frame is not None else None
        local_a_frames.append(
            convert(region_frame, local_image_size)
            if region_frame is not None
            else _zero_video(1, local_image_size, normalize=normalize)[0]
        )
    for index, region in zip(local_b_indices, roi_b):
        frame = decoded_by_index.get(index)
        region_frame = crop(frame, region) if frame is not None else None
        local_b_frames.append(
            convert(region_frame, local_image_size)
            if region_frame is not None
            else _zero_video(1, local_image_size, normalize=normalize)[0]
        )
    if release_cap:
        cap.release()
    outputs = (
        torch.stack(global_frames),
        torch.stack(local_a_frames),
        torch.stack(local_b_frames),
    )
    if return_frame_times:
        return (
            *outputs,
            torch.tensor(frame_times_from_indices(global_indices, fps), dtype=torch.float32),
            torch.tensor(frame_times_from_indices(local_a_indices, fps), dtype=torch.float32),
            torch.tensor(frame_times_from_indices(local_b_indices, fps), dtype=torch.float32),
        )
    return outputs


def read_video_staggered_views(
    path: str,
    global_frame_indices: Sequence[int],
    local_frame_indices: Sequence[int],
    global_image_size: tuple[int, int],
    local_image_size: tuple[int, int],
    local_rois: Sequence[tuple[int, int, int, int] | None],
    is_train: bool,
    hflip_prob: float,
    *,
    video_id: str = "",
    normalize: bool = True,
    cap_cache: VideoCaptureCache | None = None,
    decode_strategy: str = "multi_seek",
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Decode staggered views while applying one shared horizontal flip."""
    if len(local_frame_indices) != len(local_rois):
        raise ValueError(
            "local_frame_indices and local_rois must have the same length"
        )
    global_result = read_video_segment(
        path,
        len(global_frame_indices),
        global_image_size,
        False,
        0.0,
        frame_indices=global_frame_indices,
        normalize=normalize,
        cap_cache=cap_cache,
        decode_strategy=decode_strategy,
        return_frame_times=True,
    )
    local_result = read_video_segment(
        path,
        len(local_frame_indices),
        local_image_size,
        False,
        0.0,
        frame_indices=local_frame_indices,
        video_id=video_id,
        crop_provider=FrameCropProvider(local_frame_indices, local_rois),
        normalize=normalize,
        cap_cache=cap_cache,
        decode_strategy=decode_strategy,
        return_frame_times=True,
    )
    global_frames, global_times = global_result
    local_frames, local_times = local_result
    if is_train and random.random() < hflip_prob:
        global_frames = torch.flip(global_frames, dims=(-1,))
        local_frames = torch.flip(local_frames, dims=(-1,))
    return global_frames, local_frames, global_times, local_times

def read_video_clip(
    path: str,
    num_frames: int,
    image_size: int | tuple[int, int],
    is_train: bool,
    hflip_prob: float,
    *,
    normalize: bool = True,
) -> Tensor:
    return read_video_segment(path, num_frames, parse_image_size(image_size), is_train, hflip_prob, normalize=normalize)


class FootballClipDataset(Dataset):
    def __init__(
        self,
        records: list[ClipRecord],
        *,
        num_frames: int,
        image_size: int | tuple[int, int],
        is_train: bool,
        hflip_prob: float,
        normalize_on_cpu: bool = True,
        decode_strategy: str = "multi_seek",
        video_reader_cache_size: int = 0,
    ):
        self.records = records
        self.num_frames = num_frames
        self.image_size = parse_image_size(image_size)
        self.is_train = is_train
        self.hflip_prob = hflip_prob
        self.normalize_on_cpu = normalize_on_cpu
        self.decode_strategy = decode_strategy
        self.cap_cache = VideoCaptureCache(video_reader_cache_size) if video_reader_cache_size > 0 else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        try:
            frames = read_video_segment(
                record.clip_path,
                self.num_frames,
                self.image_size,
                self.is_train,
                self.hflip_prob,
                normalize=self.normalize_on_cpu,
                cap_cache=self.cap_cache,
                decode_strategy=self.decode_strategy,
            )
            return {
                "inputs": frames,
                "targets": torch.tensor(record.labels, dtype=torch.float32),
                "label_masks": torch.tensor(record.label_mask, dtype=torch.float32),
                "cache_key": record.cache_key,
                "meta": asdict(record),
            }
        except Exception as exc:
            if not _is_video_decode_exception(exc):
                raise
            if self.cap_cache is not None:
                self.cap_cache.close()
            print(
                f"[video_decode_recover] split={'train' if self.is_train else 'val'} "
                f"clip={record.clip_path} error={type(exc).__name__}: {exc}",
                flush=True,
            )
            meta = asdict(record)
            meta.update({"decode_failed": True, "decode_error": str(exc)[:240]})
            return {
                "inputs": _zero_video(self.num_frames, self.image_size, normalize=self.normalize_on_cpu),
                "targets": torch.zeros(len(LABELS), dtype=torch.float32),
                "label_masks": torch.zeros(len(LABELS), dtype=torch.float32),
                "cache_key": record.cache_key,
                "meta": meta,
            }


class FootballLongVideoDataset(Dataset):
    def __init__(
        self,
        records: list[LongVideoRecord],
        events_by_video: dict[tuple[str, str], list[FootballEvent]],
        *,
        num_frames: int,
        image_size: int | tuple[int, int],
        clip_duration: float,
        sampling_duration: float | None = None,
        sampling_temporal_jitter_sec: float = 0.0,
        event_margin: float,
        temporal_jitter_sec: float,
        is_train: bool,
        hflip_prob: float,
        hard_negative_temporal_jitter_sec: float | None = None,
        crop_provider: DetectorAwareCropper | RobustClipCropper | TopBandCropProvider | None = None,
        detector_view_mode: str = "roi_only",
        global_image_size: int | tuple[int, int] | None = None,
        roi_noise_cfg: Any | None = None,
        normalize_on_cpu: bool = True,
        decode_strategy: str = "multi_seek",
        video_reader_cache_size: int = 0,
        frame_supervision: bool = False,
        frame_label_sigma_sec: dict[str, float] | None = None,
        frame_label_ignore_radius_sec: dict[str, float] | None = None,
        frame_label_time_jitter_sec: dict[str, float] | None = None,
        online_hard_negative_safety_margin_sec: float = 0.0,
        negative_safety_margin_sec: float = 0.0,
        negative_label_safety_margin_sec: dict[str, float] | None = None,
        dual_sampling: str = "aligned",
        roi_overlap_frames: int = 8,
        num_rois: int = 1,
        detection_hint_renderer: DetectionHintRenderer | None = None,
        highres_pool_frames: int = 0,
        highres_pool_size: int | tuple[int, int] = (720, 1280),
        positive_window_strategy: str = "legacy_random",
        positive_anchor_min_sec: float | None = None,
        positive_anchor_max_sec: float | None = None,
        decode_max_attempts_train: int = 4,
        decode_attempt_timeout_sec: float = 60.0,
        decode_same_record_retries: int = 1,
        raw_rejected_ignore_margin_sec: float = 0.0,
        raw_context_span_min_duration_sec: float = 0.25,
        evidence_provider: Any | None = None,
        object_teacher_provider: Any | None = None,
    ):
        self.records = records
        self.events_by_video = events_by_video
        self.num_frames = num_frames
        self.image_size = parse_image_size(image_size)
        self.clip_duration = clip_duration
        self.sampling_duration = float(sampling_duration or clip_duration)
        if self.sampling_duration <= 0.0 or self.sampling_duration > self.clip_duration:
            raise ValueError(
                "video.sampling_duration must be in (0, video.clip_duration]"
            )
        self.sampling_temporal_jitter_sec = max(
            float(sampling_temporal_jitter_sec), 0.0
        )
        self.event_margin = event_margin
        self.positive_window_strategy = str(positive_window_strategy).strip().lower()
        if self.positive_window_strategy in ("", "legacy"):
            self.positive_window_strategy = "legacy_random"
        if self.positive_window_strategy not in ("legacy_random", "centered", "anchor_range_jitter", "anchor3_7_jitter"):
            raise ValueError(
                "video.positive_window_strategy must be legacy_random, centered, "
                "anchor_range_jitter, or anchor3_7_jitter"
            )
        default_min = float(positive_anchor_min_sec if positive_anchor_min_sec is not None else event_margin)
        default_max = float(
            positive_anchor_max_sec
            if positive_anchor_max_sec is not None
            else self.clip_duration - event_margin
        )
        if self.positive_window_strategy == "anchor3_7_jitter":
            default_min, default_max = 3.0, 7.0
        self.positive_anchor_min_sec = max(0.0, min(default_min, self.clip_duration))
        self.positive_anchor_max_sec = max(0.0, min(default_max, self.clip_duration))
        if self.positive_anchor_max_sec < self.positive_anchor_min_sec:
            raise ValueError(
                "positive anchor range is invalid: "
                f"min={self.positive_anchor_min_sec} max={self.positive_anchor_max_sec}"
            )
        self.temporal_jitter_sec = temporal_jitter_sec
        self.hard_negative_temporal_jitter_sec = (
            temporal_jitter_sec
            if hard_negative_temporal_jitter_sec is None
            else float(hard_negative_temporal_jitter_sec)
        )
        self.is_train = is_train
        self.hflip_prob = hflip_prob
        self.crop_provider = crop_provider
        self.detector_view_mode = str(detector_view_mode)
        if self.detector_view_mode not in ("roi_only", "dual"):
            raise ValueError("detector_view_mode must be roi_only or dual")
        self.global_image_size = parse_image_size(global_image_size or image_size)
        self.dual_sampling = str(dual_sampling).strip().lower()
        if self.dual_sampling not in ("aligned", "staggered", "multi_staggered"):
            raise ValueError(
                "dual_sampling must be aligned, staggered, or multi_staggered"
            )
        if self.dual_sampling == "staggered" and self.detector_view_mode != "dual":
            raise ValueError("staggered dual sampling requires detector_view_mode=dual")
        self.roi_overlap_frames = min(max(int(roi_overlap_frames), 0), self.num_frames)
        self.num_rois = int(num_rois)
        self.detection_hint_renderer = detection_hint_renderer
        self.highres_pool_frames = max(int(highres_pool_frames), 0)
        self.highres_pool_size = parse_image_size(highres_pool_size)
        if self.highres_pool_frames and self.crop_provider is not None:
            raise ValueError(
                "learned high-resolution glimpses require spatial_crop.mode=none"
            )
        if self.detection_hint_renderer is not None and isinstance(
            self.crop_provider, RobustClipCropper
        ):
            raise ValueError(
                "detection_hint v1 supports only the full-image input path"
            )
        if self.num_rois not in (1, 2):
            raise ValueError("spatial_crop.num_rois must be 1 or 2")
        if self.num_rois == 2:
            if self.detector_view_mode != "dual":
                raise ValueError("two ROI inputs require spatial_crop.output_view=dual")
            if self.dual_sampling == "staggered":
                raise ValueError(
                    "two ROI inputs use dual_sampling=aligned or multi_staggered"
                )
            if not isinstance(self.crop_provider, RobustClipCropper):
                raise ValueError("two ROI inputs require RobustClipCropper")
            if self.crop_provider.temporal_mode != "clip":
                raise ValueError("two ROI inputs currently require spatial_crop.temporal_mode=clip")
        self.roi_noise_cfg = roi_noise_cfg or ConfigDict()
        self.normalize_on_cpu = normalize_on_cpu
        self.decode_strategy = decode_strategy
        self.cap_cache = VideoCaptureCache(video_reader_cache_size) if video_reader_cache_size > 0 else None
        self.frame_supervision = bool(frame_supervision)
        self.frame_label_sigma_sec = frame_label_sigma_sec or {label: 1.0 for label in LABELS}
        self.frame_label_ignore_radius_sec = frame_label_ignore_radius_sec or {label: 3.0 for label in LABELS}
        self.frame_label_time_jitter_sec = {
            label: max(float((frame_label_time_jitter_sec or {}).get(label, 0.0)), 0.0)
            for label in LABELS
        }
        self.online_hard_negative_safety_margin_sec = max(
            float(online_hard_negative_safety_margin_sec), 0.0
        )
        self.raw_rejected_ignore_margin_sec = max(
            float(raw_rejected_ignore_margin_sec), 0.0
        )
        self.raw_context_span_min_duration_sec = max(
            float(raw_context_span_min_duration_sec), 0.0
        )
        self.evidence_provider = evidence_provider
        self.object_teacher_provider = object_teacher_provider
        self.negative_safety_margin_sec = max(
            float(negative_safety_margin_sec), 0.0
        )
        self.negative_label_safety_margin_sec = {
            label: max(
                float(
                    (negative_label_safety_margin_sec or {}).get(
                        label, self.negative_safety_margin_sec
                    )
                ),
                0.0,
            )
            for label in LABELS
        }
        self._geometry_cache: dict[str, tuple[int, int, float, int]] = {}
        self.decode_max_attempts_train = max(int(decode_max_attempts_train), 1)
        self.decode_attempt_timeout_sec = max(
            float(decode_attempt_timeout_sec or 0.0), 0.0
        )
        self.decode_same_record_retries = max(
            min(int(decode_same_record_retries), self.decode_max_attempts_train - 1),
            0,
        )

    def __len__(self) -> int:
        return len(self.records)

    def _video_geometry(self, path: str) -> tuple[int, int, float, int]:
        if path in self._geometry_cache:
            return self._geometry_cache[path]
        cap = _open_video_capture(path)
        if not cap.isOpened():
            cap.release()
            raise FileNotFoundError(f"Could not open video for ROI geometry: {path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0:
            raise RuntimeError(
                f"Invalid video geometry for {path}: {width}x{height} fps={fps} frames={frame_count}"
            )
        self._geometry_cache[path] = (width, height, fps, frame_count)
        return width, height, fps, frame_count

    def _sample_window(self, record: LongVideoRecord, events: Sequence[FootballEvent]) -> tuple[float, float]:
        max_start = max(record.video_duration - self.clip_duration, 0.0)
        # Online-simulation records already encode deliberate central and edge
        # windows in base_clip_start. Re-applying anchor jitter here silently
        # turns both roles into ordinary central positives.
        if self.is_train and record.online_pair_id:
            base_start = clamp(record.base_clip_start, 0.0, max_start)
            return base_start, base_start + self.clip_duration
        if record.is_negative:
            base_start = clamp(record.base_clip_start, 0.0, max_start)
            jitter_sec = self.temporal_jitter_sec
            if record.sample_id.startswith("hard_neg_"):
                jitter_sec = self.hard_negative_temporal_jitter_sec
            if not self.is_train or jitter_sec <= 0:
                return base_start, base_start + self.clip_duration
            for _ in range(10):
                start = base_start + random.uniform(-jitter_sec, jitter_sec)
                start = clamp(start, 0.0, max_start)
                end = start + self.clip_duration
                if is_safe_negative_window(
                    events,
                    start,
                    end,
                    default_margin_sec=self.negative_safety_margin_sec,
                    label_margin_sec=self.negative_label_safety_margin_sec,
                ):
                    return start, end
            return base_start, base_start + self.clip_duration

        if self.is_train:
            if self.positive_window_strategy == "centered":
                start, _ = centered_window(record.anchor_time, self.clip_duration, record.video_duration)
            elif self.positive_window_strategy in ("anchor_range_jitter", "anchor3_7_jitter"):
                start = record.anchor_time - random.uniform(
                    self.positive_anchor_min_sec, self.positive_anchor_max_sec
                )
            else:
                low = record.anchor_time - self.clip_duration + self.event_margin
                high = record.anchor_time - self.event_margin
                low = max(low, 0.0)
                high = min(high, max_start)
                if high >= low:
                    start = random.uniform(low, high)
                else:
                    start, _ = centered_window(record.anchor_time, self.clip_duration, record.video_duration)
        else:
            start, _ = centered_window(record.anchor_time, self.clip_duration, record.video_duration)
        start = clamp(start, 0.0, max_start)
        return start, start + self.clip_duration

    def __getitem__(self, index: int) -> dict[str, Any]:
        current_index = int(index)
        max_attempts = self.decode_max_attempts_train if self.is_train else 1
        last_exc: BaseException | None = None
        for attempt in range(max_attempts):
            try:
                set_decode_deadline(self.decode_attempt_timeout_sec)
                result = self._getitem_impl(current_index)
                check_decode_deadline()
                return result
            except Exception as exc:
                if not _is_video_decode_exception(exc):
                    raise
                last_exc = exc
                record = self.records[current_index]
                if self.cap_cache is not None:
                    self.cap_cache.close()
                self._warn_decode_recover(record, exc, attempt + 1, max_attempts)
                if not self.is_train:
                    break
                # A retry may re-jitter the same positive record, but deterministic
                # corruption must switch records quickly so all loader workers do
                # not block on the same truncated MP4.
                current_index = (
                    int(index)
                    if attempt < self.decode_same_record_retries
                    else random.randrange(len(self.records))
                )
        record = self.records[current_index if 0 <= current_index < len(self.records) else int(index)]
        return self._decode_failure_item(record, last_exc)

    def _warn_decode_recover(
        self,
        record: LongVideoRecord,
        exc: BaseException,
        attempt: int,
        max_attempts: int,
    ) -> None:
        count = int(getattr(self, "_decode_warning_count", 0))
        self._decode_warning_count = count + 1
        if count >= 80:
            return
        action = "retry" if self.is_train and attempt < max_attempts else "mask"
        print(
            f"[video_decode_recover] split={'train' if self.is_train else 'val'} "
            f"action={action} attempt={attempt}/{max_attempts} "
            f"video={record.video_id} sample={record.sample_id} "
            f"path={record.video_path} error={type(exc).__name__}: {str(exc)[:240]}",
            flush=True,
        )

    def _decode_failure_item(
        self,
        record: LongVideoRecord,
        exc: BaseException | None,
    ) -> dict[str, Any]:
        frames = _zero_video(
            self.num_frames,
            self.global_image_size if self.detector_view_mode == "dual" else self.image_size,
            normalize=self.normalize_on_cpu,
        )
        label_count = len(LABELS)
        meta = asdict(record)
        meta.update({
            "decode_failed": True,
            "decode_error": "" if exc is None else str(exc)[:240],
            "sampled_clip_start": float(record.base_clip_start),
            "sampled_clip_end": float(record.base_clip_start) + float(self.clip_duration),
            "context_clip_start": float(record.base_clip_start),
            "context_clip_end": float(record.base_clip_start) + float(self.clip_duration),
            "dynamic_labels": tuple(0.0 for _ in LABELS),
            "label_mask": tuple(0.0 for _ in LABELS),
            "roi": invalid_roi("decode_failed").to_dict(),
            "frame_rois": [invalid_roi("decode_failed").to_dict() for _ in range(self.num_frames)],
            "roi_b": invalid_roi("decode_failed").to_dict(),
            "frame_rois_b": [invalid_roi("decode_failed").to_dict() for _ in range(self.num_frames)],
        })
        result: dict[str, Any] = {
            "inputs": frames,
            "targets": torch.zeros(label_count, dtype=torch.float32),
            "label_masks": torch.zeros(label_count, dtype=torch.float32),
            "set_piece_subtype_targets": torch.zeros(
                len(SET_PIECE_SUBTYPES), dtype=torch.float32
            ),
            "set_piece_subtype_mask": torch.tensor(0.0, dtype=torch.float32),
            "set_piece_context_span_mask": torch.zeros(self.num_frames, dtype=torch.float32),
            "set_piece_context_span_row_mask": torch.tensor(0.0, dtype=torch.float32),
            "save_cohort_targets": torch.tensor(0.0, dtype=torch.float32),
            "save_cohort_masks": torch.tensor(0.0, dtype=torch.float32),
            "save_cohort_weights": torch.tensor(0.0, dtype=torch.float32),
            "sample_loss_weights": torch.tensor(0.0, dtype=torch.float32),
            "online_negative_masks": torch.zeros(label_count, dtype=torch.float32),
            "cache_key": record.cache_key,
            "meta": meta,
            "roi_valid": torch.tensor(0.0, dtype=torch.float32),
            "roi_meta": invalid_roi("decode_failed").meta_vector(),
            "roi_frame_valid": torch.zeros(self.num_frames, dtype=torch.float32),
            "roi_frame_meta": torch.stack([invalid_roi("decode_failed").meta_vector() for _ in range(self.num_frames)]),
            "roi_valid_b": torch.tensor(0.0, dtype=torch.float32),
            "roi_meta_b": invalid_roi("decode_failed").meta_vector(),
            "roi_frame_valid_b": torch.zeros(self.num_frames, dtype=torch.float32),
            "roi_frame_meta_b": torch.stack([invalid_roi("decode_failed").meta_vector() for _ in range(self.num_frames)]),
        }
        if self.detector_view_mode == "dual":
            result["roi_inputs"] = _zero_video(
                self.num_frames, self.image_size, normalize=self.normalize_on_cpu
            )
            if self.num_rois == 2:
                result["roi_inputs_b"] = _zero_video(
                    self.num_frames, self.image_size, normalize=self.normalize_on_cpu
                )
        if self.highres_pool_frames:
            result["highres_pool_inputs"] = _zero_video(
                self.highres_pool_frames, self.highres_pool_size, normalize=False
            )
            result["highres_pool_times"] = torch.zeros(self.highres_pool_frames, dtype=torch.float32)
            result["highres_pool_targets"] = torch.zeros(
                self.highres_pool_frames, label_count, dtype=torch.float32
            )
            result["highres_pool_target_masks"] = torch.zeros(
                self.highres_pool_frames, label_count, dtype=torch.float32
            )
        if self.frame_supervision:
            frame_times = torch.linspace(0.0, float(self.clip_duration), self.num_frames)
            frame_targets = torch.zeros(self.num_frames, label_count, dtype=torch.float32)
            frame_masks = torch.zeros(self.num_frames, label_count, dtype=torch.float32)
            result["frame_times"] = frame_times
            result["frame_targets"] = frame_targets
            result["frame_target_masks"] = frame_masks
            if self.detector_view_mode == "dual":
                result["local_frame_times"] = frame_times.clone()
                result["local_frame_times_b"] = frame_times.clone()
                result["local_frame_targets"] = frame_targets.clone()
                result["local_frame_target_masks"] = frame_masks.clone()
        if self.object_teacher_provider is not None:
            if hasattr(self.object_teacher_provider, "empty_bundle"):
                result.update(self.object_teacher_provider.empty_bundle(self.num_frames))
            else:
                object_targets, object_masks = self.object_teacher_provider.empty(self.num_frames)
                result["object_heatmap_targets"] = object_targets
                result["object_heatmap_masks"] = object_masks
        return result

    def _getitem_impl(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        events = self.events_by_video[(record.source, record.video_id)]
        context_start, context_end = self._sample_window(record, events)
        start, end = nested_sampling_window(
            record,
            context_start,
            context_end,
            self.sampling_duration,
            jitter_sec=(self.sampling_temporal_jitter_sec if self.is_train else 0.0),
        )
        labels = labels_for_window(events, start, end)
        label_mask = label_mask_for_sample(record.label_mask, labels)
        label_mask = rejected_set_piece_label_mask(
            events, start, end, labels, label_mask, self.raw_rejected_ignore_margin_sec
        )
        set_piece_subtype_targets = set_piece_subtypes_for_window(
            events, start, end
        )
        set_piece_index = LABEL_TO_INDEX.get("set_piece", -1)
        set_piece_subtype_mask = (
            1.0
            if set_piece_index >= 0
            and float(labels[set_piece_index]) > 0
            and float(label_mask[set_piece_index]) > 0
            else 0.0
        )
        online_negative_mask = online_negative_label_mask(
            events,
            start,
            end,
            label_mask,
            self.online_hard_negative_safety_margin_sec,
        )
        save_index = LABEL_TO_INDEX.get("save", -1)
        save_cohort_target = 1.0 if record.save_cohort_kind.startswith("save_") else 0.0
        save_cohort_mask = (
            1.0
            if record.save_cohort_weight > 0
            and not (
                record.save_cohort_kind == "shot_only"
                and save_index >= 0
                and float(labels[save_index]) > 0
            )
            else 0.0
        )
        proposal = invalid_roi("spatial_crop_disabled")
        secondary_proposal = invalid_roi("second_roi_disabled")
        roi_inputs: Tensor | None = None
        roi_inputs_b: Tensor | None = None
        frame_times: Tensor | None = None
        local_frame_times: Tensor | None = None
        local_frame_times_b: Tensor | None = None

        frame_proposals = [proposal for _ in range(self.num_frames)]
        secondary_frame_proposals = [secondary_proposal for _ in range(self.num_frames)]
        sampled_frame_times: Tensor | None = None
        sampled_local_frame_times: Tensor | None = None
        sampled_local_frame_times_b: Tensor | None = None
        needs_decoded_frame_times = bool(
            self.frame_supervision
            or self.evidence_provider is not None
            or self.object_teacher_provider is not None
        )
        if isinstance(self.crop_provider, RobustClipCropper):
            width, height, fps, frame_count = self._video_geometry(record.video_path)
            frame_indices = segment_frame_indices(
                frame_count,
                fps,
                self.num_frames,
                self.is_train,
                start_sec=start,
                end_sec=end,
            )
            global_frame_indices = list(frame_indices)
            local_frame_indices = list(frame_indices)
            local_frame_indices_b = list(frame_indices)
            if self.detector_view_mode == "dual" and self.dual_sampling == "staggered":
                local_frame_indices = staggered_local_indices(
                    frame_indices, self.roi_overlap_frames
                )
            elif (
                self.detector_view_mode == "dual"
                and self.dual_sampling == "multi_staggered"
            ):
                segment_start_frame = int(
                    clamp(round(start * fps), 0, frame_count - 1)
                )
                segment_end_frame = int(
                    clamp(round(end * fps) - 1, segment_start_frame, frame_count - 1)
                )
                local_frame_indices, local_frame_indices_b = (
                    staggered_multi_roi_indices(
                        frame_indices,
                        segment_start_frame,
                        segment_end_frame,
                    )
                )
            global_frame_time_values = frame_times_from_indices(
                global_frame_indices, fps
            )
            frame_time_values = frame_times_from_indices(local_frame_indices, fps)
            frame_time_values_b = frame_times_from_indices(
                local_frame_indices_b, fps
            )
            sampled_frame_times = torch.tensor(
                global_frame_time_values, dtype=torch.float32
            )
            sampled_local_frame_times = torch.tensor(
                frame_time_values, dtype=torch.float32
            )
            sampled_local_frame_times_b = torch.tensor(
                frame_time_values_b, dtype=torch.float32
            )
            fixed_fallback = None
            if self.crop_provider.temporal_mode == "dynamic":
                fixed_fallback = self.crop_provider.get_window_roi(
                    record.video_id,
                    start,
                    end,
                    width,
                    height,
                    self.image_size,
                )
            frame_proposals, proposal = self.crop_provider.get_clip_rois(
                record.video_id,
                start,
                end,
                frame_time_values,
                width,
                height,
                self.image_size,
                fixed_fallback=fixed_fallback,
            )
            if self.num_rois == 2:
                secondary_proposal = self.crop_provider.get_complementary_window_roi(
                    record.video_id,
                    start,
                    end,
                    width,
                    height,
                    self.image_size,
                    proposal,
                )
                secondary_frame_proposals = [
                    secondary_proposal for _ in local_frame_indices_b
                ]
            if self.is_train:
                if self.crop_provider.temporal_mode == "dynamic":
                    frame_proposals, proposal = self.crop_provider.augment_sequence(
                        frame_proposals,
                        width,
                        height,
                        self.roi_noise_cfg,
                        self.image_size,
                        fixed_fallback=fixed_fallback,
                    )
                else:
                    proposal = self.crop_provider.augment(
                        proposal, width, height, self.roi_noise_cfg, self.image_size
                    )
                    frame_proposals = [proposal for _ in local_frame_indices]
                    if self.num_rois == 2:
                        secondary_proposal = self.crop_provider.augment(
                            secondary_proposal,
                            width,
                            height,
                            self.roi_noise_cfg,
                            self.image_size,
                        )
                        secondary_frame_proposals = [
                            secondary_proposal for _ in local_frame_indices_b
                        ]
            if self.num_rois == 2 and not proposal.valid and secondary_proposal.valid:
                proposal, secondary_proposal = secondary_proposal, proposal
                frame_proposals, secondary_frame_proposals = (
                    secondary_frame_proposals,
                    frame_proposals,
                )
            frame_rois = [item.bbox if item.valid else None for item in frame_proposals]
            secondary_frame_rois = [
                item.bbox if item.valid else None
                for item in secondary_frame_proposals
            ]
            if self.detector_view_mode == "dual":
                if self.dual_sampling == "staggered":
                    (
                        frames,
                        roi_inputs,
                        frame_times,
                        local_frame_times,
                    ) = read_video_staggered_views(
                        record.video_path,
                        global_frame_indices,
                        local_frame_indices,
                        self.global_image_size,
                        self.image_size,
                        frame_rois,
                        self.is_train,
                        self.hflip_prob,
                        video_id=record.video_id,
                        normalize=self.normalize_on_cpu,
                        cap_cache=self.cap_cache,
                        decode_strategy=self.decode_strategy,
                    )
                else:
                    if self.num_rois == 2:
                        view_result = read_video_segment_multi_roi_views(
                            record.video_path,
                            self.num_frames,
                            self.global_image_size,
                            self.image_size,
                            frame_rois,
                            secondary_frame_rois,
                            self.is_train,
                            self.hflip_prob,
                            start_sec=start,
                            end_sec=end,
                            frame_indices=global_frame_indices,
                            roi_a_frame_indices=local_frame_indices,
                            roi_b_frame_indices=local_frame_indices_b,
                            normalize=self.normalize_on_cpu,
                            cap_cache=self.cap_cache,
                            decode_strategy=self.decode_strategy,
                            return_frame_times=needs_decoded_frame_times,
                        )
                        if needs_decoded_frame_times:
                            (
                                frames,
                                roi_inputs,
                                roi_inputs_b,
                                frame_times,
                                local_frame_times,
                                local_frame_times_b,
                            ) = view_result
                        else:
                            frames, roi_inputs, roi_inputs_b = view_result
                    else:
                        view_result = read_video_segment_views(
                            record.video_path,
                            self.num_frames,
                            self.global_image_size,
                            self.image_size,
                            frame_rois,
                            self.is_train,
                            self.hflip_prob,
                            start_sec=start,
                            end_sec=end,
                            frame_indices=frame_indices,
                            normalize=self.normalize_on_cpu,
                            cap_cache=self.cap_cache,
                            decode_strategy=self.decode_strategy,
                            return_frame_times=needs_decoded_frame_times,
                        )
                        if needs_decoded_frame_times:
                            frames, roi_inputs, frame_times = view_result
                            local_frame_times = frame_times
                        else:
                            frames, roi_inputs = view_result
            else:
                segment_result = read_video_segment(
                    record.video_path,
                    self.num_frames,
                    self.image_size,
                    self.is_train,
                    self.hflip_prob,
                    start_sec=start,
                    end_sec=end,
                    frame_indices=frame_indices,
                    video_id=record.video_id,
                    crop_provider=FrameCropProvider(frame_indices, frame_rois),
                    normalize=self.normalize_on_cpu,
                    cap_cache=self.cap_cache,
                    decode_strategy=self.decode_strategy,
                    return_frame_times=needs_decoded_frame_times,
                )
                if needs_decoded_frame_times:
                    frames, frame_times = segment_result
                else:
                    frames = segment_result
        else:
            if self.highres_pool_frames:
                (
                    frames,
                    frame_times,
                    dense_frames,
                    dense_frame_times,
                ) = read_video_global_highres_pool(
                    record.video_path,
                    self.num_frames,
                    self.highres_pool_frames,
                    self.image_size,
                    self.highres_pool_size,
                    self.is_train,
                    self.hflip_prob,
                    start_sec=start,
                    end_sec=end,
                    video_id=record.video_id,
                    detection_hint_renderer=self.detection_hint_renderer,
                    normalize_global=self.normalize_on_cpu,
                    cap_cache=self.cap_cache,
                    decode_strategy=self.decode_strategy,
                )
            else:
                segment_result = read_video_segment(
                    record.video_path,
                    self.num_frames,
                    self.image_size,
                    self.is_train,
                    self.hflip_prob,
                    start_sec=start,
                    end_sec=end,
                    video_id=record.video_id,
                    crop_provider=self.crop_provider,
                    detection_hint_renderer=self.detection_hint_renderer,
                    normalize=self.normalize_on_cpu,
                    cap_cache=self.cap_cache,
                    decode_strategy=self.decode_strategy,
                    return_frame_times=needs_decoded_frame_times,
                )
                if needs_decoded_frame_times:
                    frames, frame_times = segment_result
                else:
                    frames = segment_result

        effective_proposal = proposal if proposal.valid else secondary_proposal
        effective_frame_proposals = [
            first if first.valid else second
            for first, second in zip(frame_proposals, secondary_frame_proposals)
        ]
        meta = asdict(record)
        meta.update({
            "sampled_clip_start": start,
            "sampled_clip_end": end,
            "context_clip_start": context_start,
            "context_clip_end": context_end,
            "dynamic_labels": labels,
            "label_mask": label_mask,
            "roi": proposal.to_dict(),
            "frame_rois": [item.to_dict() for item in frame_proposals],
            "roi_b": secondary_proposal.to_dict(),
            "frame_rois_b": [item.to_dict() for item in secondary_frame_proposals],
        })
        result = {
            "inputs": frames,
            "targets": torch.tensor(labels, dtype=torch.float32),
            "label_masks": torch.tensor(label_mask, dtype=torch.float32),
            "set_piece_subtype_targets": torch.tensor(
                set_piece_subtype_targets, dtype=torch.float32
            ),
            "set_piece_subtype_mask": torch.tensor(
                set_piece_subtype_mask, dtype=torch.float32
            ),
            "set_piece_context_span_mask": torch.zeros(self.num_frames, dtype=torch.float32),
            "set_piece_context_span_row_mask": torch.tensor(0.0, dtype=torch.float32),
            "save_cohort_targets": torch.tensor(save_cohort_target, dtype=torch.float32),
            "save_cohort_masks": torch.tensor(save_cohort_mask, dtype=torch.float32),
            "save_cohort_weights": torch.tensor(record.save_cohort_weight, dtype=torch.float32),
            "sample_loss_weights": torch.tensor(record.sample_loss_weight, dtype=torch.float32),
            "online_negative_masks": torch.tensor(
                online_negative_mask, dtype=torch.float32
            ),
            "cache_key": record.cache_key,
            "meta": meta,
            "roi_valid": torch.tensor(
                sum(float(item.valid) for item in effective_frame_proposals)
                / max(len(effective_frame_proposals), 1),
                dtype=torch.float32,
            ),
            "roi_meta": effective_proposal.meta_vector(),
            "roi_frame_valid": torch.tensor(
                [float(item.valid) for item in effective_frame_proposals],
                dtype=torch.float32,
            ),
            "roi_frame_meta": torch.stack(
                [item.meta_vector() for item in effective_frame_proposals]
            ),
            "roi_valid_b": torch.tensor(
                sum(float(item.valid) for item in secondary_frame_proposals)
                / max(len(secondary_frame_proposals), 1),
                dtype=torch.float32,
            ),
            "roi_meta_b": secondary_proposal.meta_vector(),
            "roi_frame_valid_b": torch.tensor(
                [float(item.valid) for item in secondary_frame_proposals],
                dtype=torch.float32,
            ),
            "roi_frame_meta_b": torch.stack(
                [item.meta_vector() for item in secondary_frame_proposals]
            ),
        }
        frame_supervision_events = events
        if self.is_train and self.frame_supervision:
            frame_supervision_events = jitter_frame_supervision_events(
                events,
                start,
                end,
                record.video_duration,
                self.frame_label_time_jitter_sec,
            )

        if roi_inputs is not None:
            result["roi_inputs"] = roi_inputs
        if roi_inputs_b is not None:
            result["roi_inputs_b"] = roi_inputs_b
        if self.highres_pool_frames:
            result["highres_pool_inputs"] = dense_frames
            result["highres_pool_times"] = dense_frame_times
            dense_targets, dense_target_masks = gaussian_frame_targets(
                frame_supervision_events,
                start,
                end,
                dense_frame_times.tolist(),
                label_mask,
                self.frame_label_sigma_sec,
                self.frame_label_ignore_radius_sec,
            )
            result["highres_pool_targets"] = dense_targets
            result["highres_pool_target_masks"] = dense_target_masks
        if self.frame_supervision and frame_times is None:
            frame_times = sampled_frame_times
        if self.frame_supervision and frame_times is not None:
            frame_targets, frame_target_masks = gaussian_frame_targets(
                frame_supervision_events,
                start,
                end,
                frame_times.tolist(),
                label_mask,
                self.frame_label_sigma_sec,
                self.frame_label_ignore_radius_sec,
            )
            result["frame_times"] = frame_times
            result["frame_targets"] = frame_targets
            result["frame_target_masks"] = frame_target_masks
            context_span_mask, context_span_row_mask = set_piece_context_span_mask(
                events,
                start,
                end,
                frame_times.tolist(),
                self.raw_context_span_min_duration_sec,
            )
            result["set_piece_context_span_mask"] = context_span_mask
            result["set_piece_context_span_row_mask"] = context_span_row_mask
            if roi_inputs is not None:
                if local_frame_times is None:
                    local_frame_times = (
                        sampled_local_frame_times
                        if sampled_local_frame_times is not None else frame_times
                    )
                local_targets, local_target_masks = gaussian_frame_targets(
                    frame_supervision_events,
                    start,
                    end,
                    local_frame_times.tolist(),
                    label_mask,
                    self.frame_label_sigma_sec,
                    self.frame_label_ignore_radius_sec,
                )
                result["local_frame_times"] = local_frame_times
                if local_frame_times_b is None:
                    local_frame_times_b = (
                        sampled_local_frame_times_b
                        if sampled_local_frame_times_b is not None
                        else local_frame_times
                    )
                result["local_frame_times_b"] = local_frame_times_b
                result["local_frame_targets"] = local_targets
                result["local_frame_target_masks"] = local_target_masks
        if self.evidence_provider is not None:
            absolute_times = offline_provider_absolute_times(
                frame_times,
                start_sec=start,
                clip_duration=self.clip_duration,
                num_frames=self.num_frames,
            )
            result["evidence_features"] = self.evidence_provider.features(
                record.video_id, absolute_times
            )
        if self.object_teacher_provider is not None:
            if frame_times is None:
                raise RuntimeError(
                    "object teacher supervision requires aligned frame_times; "
                    "enable train.object_heatmap_loss_weight"
                )
            absolute_times = offline_provider_absolute_times(
                frame_times,
                start_sec=start,
                clip_duration=self.clip_duration,
                num_frames=self.num_frames,
            )
            if hasattr(self.object_teacher_provider, "target_bundle"):
                result.update(
                    self.object_teacher_provider.target_bundle(
                        record.video_id, absolute_times
                    )
                )
            else:
                object_targets, object_masks = self.object_teacher_provider.targets(record.video_id, absolute_times)
                result["object_heatmap_targets"] = object_targets
                result["object_heatmap_masks"] = object_masks
        return result


class CachedFootballFeatureDataset(Dataset):
    def __init__(self, records: list[ClipRecord], cache_dir: str):
        self.records = records
        self.cache_dir = Path(cache_dir)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        cache_path = self.cache_dir / record.cache_key
        if not cache_path.exists():
            raise FileNotFoundError(f"Missing cached feature: {cache_path}")
        item = torch.load(cache_path, map_location="cpu")
        return {
            "inputs": item["features"].float(),
            "targets": torch.tensor(record.labels, dtype=torch.float32),
            "label_masks": torch.tensor(item.get("label_mask", record.label_mask), dtype=torch.float32),
            "cache_key": record.cache_key,
            "meta": asdict(record),
        }


def football_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    result = {
        "inputs": torch.stack([item["inputs"] for item in batch], dim=0),
        "targets": torch.stack([item["targets"] for item in batch], dim=0),
        "label_masks": torch.stack([item.get("label_masks", torch.ones_like(item["targets"])) for item in batch], dim=0),
        "cache_key": [item["cache_key"] for item in batch],
        "meta": [item["meta"] for item in batch],
    }
    for key in (
        "object_heatmap_targets",
        "object_heatmap_masks",
        "ball_goal_relation_targets",
        "ball_goal_relation_masks",
        "ball_context_masks",
    ):
        if key in batch[0]:
            result[key] = torch.stack([item[key] for item in batch], dim=0)
    if "roi_inputs" in batch[0]:
        result["roi_inputs"] = torch.stack([item["roi_inputs"] for item in batch], dim=0)
    if "roi_inputs_b" in batch[0]:
        result["roi_inputs_b"] = torch.stack([item["roi_inputs_b"] for item in batch], dim=0)
    for key in (
        "highres_pool_inputs",
        "highres_pool_times",
        "highres_pool_targets",
        "highres_pool_target_masks",
    ):
        if key in batch[0]:
            result[key] = torch.stack([item[key] for item in batch], dim=0)
    if "roi_valid" in batch[0]:
        result["roi_valid"] = torch.stack([item["roi_valid"] for item in batch], dim=0)
    if "roi_meta" in batch[0]:
        result["roi_meta"] = torch.stack([item["roi_meta"] for item in batch], dim=0)
    if "roi_frame_valid" in batch[0]:
        result["roi_frame_valid"] = torch.stack([item["roi_frame_valid"] for item in batch], dim=0)
    if "roi_frame_meta" in batch[0]:
        result["roi_frame_meta"] = torch.stack([item["roi_frame_meta"] for item in batch], dim=0)
    for key in ("roi_valid_b", "roi_meta_b", "roi_frame_valid_b", "roi_frame_meta_b"):
        if key in batch[0]:
            result[key] = torch.stack([item[key] for item in batch], dim=0)
    if "frame_times" in batch[0]:
        result["frame_times"] = torch.stack([item["frame_times"] for item in batch], dim=0)
    if any("evidence_features" in item for item in batch):
        reference = next(
            item["evidence_features"] for item in batch if "evidence_features" in item
        )
        result["evidence_features"] = torch.stack(
            [
                item.get("evidence_features", torch.zeros_like(reference))
                for item in batch
            ],
            dim=0,
        )
    if "local_frame_times" in batch[0]:
        result["local_frame_times"] = torch.stack(
            [item["local_frame_times"] for item in batch], dim=0
        )
    if "local_frame_times_b" in batch[0]:
        result["local_frame_times_b"] = torch.stack(
            [item["local_frame_times_b"] for item in batch], dim=0
        )
    if "local_frame_targets" in batch[0]:
        result["local_frame_targets"] = torch.stack([item["local_frame_targets"] for item in batch], dim=0)
    if "local_frame_target_masks" in batch[0]:
        result["local_frame_target_masks"] = torch.stack([item["local_frame_target_masks"] for item in batch], dim=0)
    if "frame_targets" in batch[0]:
        result["frame_targets"] = torch.stack([item["frame_targets"] for item in batch], dim=0)
    if "frame_target_masks" in batch[0]:
        result["frame_target_masks"] = torch.stack([item["frame_target_masks"] for item in batch], dim=0)
    if "set_piece_subtype_targets" in batch[0]:
        result["set_piece_subtype_targets"] = torch.stack(
            [item["set_piece_subtype_targets"] for item in batch], dim=0
        )
        result["set_piece_subtype_mask"] = torch.stack(
            [item["set_piece_subtype_mask"] for item in batch], dim=0
        )
    for key in (
        "set_piece_context_span_mask",
        "set_piece_context_span_row_mask",
        "save_cohort_targets",
        "save_cohort_masks",
        "save_cohort_weights",
    ):
        if key in batch[0]:
            result[key] = torch.stack([item[key] for item in batch], dim=0)
    if any("sample_loss_weights" in item for item in batch):
        result["sample_loss_weights"] = torch.stack(
            [item.get("sample_loss_weights", torch.tensor(1.0, dtype=torch.float32)) for item in batch], dim=0
        )
    if "online_negative_masks" in batch[0]:
        result["online_negative_masks"] = torch.stack(
            [item["online_negative_masks"] for item in batch], dim=0
        )
    return result


def build_backbone(cfg: Any) -> nn.Module:
    arch = cfg.model.backbone
    weights = cfg.model.get("weights", "")
    pretrained = bool(cfg.model.get("pretrained", True))
    if pretrained and not weights:
        raise ValueError("model.weights must point to a local backbone checkpoint or URL when pretrained=true")
    if str(arch).startswith("videomaev2_"):
        from football_videomaev2 import build_videomaev2_backbone

        return build_videomaev2_backbone(cfg)
    factories = {
        "dinov3_vitb16": dinov3_vitb16,
        "dinov3_vitl16": dinov3_vitl16,
        "dinov3_vith16plus": dinov3_vith16plus,
    }
    if arch not in factories:
        supported = sorted([*factories, "videomaev2_vitb16", "videomaev2_vitl16"])
        raise ValueError(f"Unsupported backbone {arch}. Expected one of {supported}")
    kwargs: dict[str, Any] = {"pretrained": pretrained}
    if weights:
        kwargs["weights"] = weights
    return factories[arch](**kwargs)


def _looks_like_football_checkpoint(path: str) -> bool:
    return bool(path) and Path(path).suffix.lower() == ".pt" and Path(path).exists()


def _as_plain_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def resolve_model_init_checkpoint(cfg: Any) -> tuple[str, dict[str, Any] | None]:
    init_checkpoint = str(cfg.model.get("init_checkpoint", cfg.get("init_checkpoint", "")) or "")
    if init_checkpoint:
        return init_checkpoint, None

    weights = str(cfg.model.get("weights", "") or "")
    if not _looks_like_football_checkpoint(weights):
        return "", None

    checkpoint = torch.load(weights, map_location="cpu")
    if not (isinstance(checkpoint, dict) and "model" in checkpoint and "config" in checkpoint):
        return "", None

    checkpoint_cfg = _as_plain_dict(checkpoint.get("config"))
    checkpoint_model_cfg = _as_plain_dict(checkpoint_cfg.get("model"))
    backbone_weights = str(checkpoint_model_cfg.get("weights", "") or "")
    if not backbone_weights:
        raise ValueError(
            f"model.weights points to a football checkpoint ({weights}), but its saved config has no model.weights "
            "for building the DINO backbone. Put the official DINO .pth in model.weights and the football "
            "checkpoint in model.init_checkpoint."
        )
    cfg.model["weights"] = backbone_weights
    cfg.model["init_checkpoint"] = weights
    print(
        "WARN: model.weights points to a full football checkpoint; treating it as model.init_checkpoint "
        f"and using backbone weights from that checkpoint config: {backbone_weights}",
        flush=True,
    )
    return weights, checkpoint


def strip_module_prefix(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    return {key[7:] if key.startswith("module.") else key: value for key, value in state_dict.items()}


def resize_temporal_pos_embed(source: Tensor, target_shape: torch.Size | tuple[int, ...]) -> Tensor | None:
    if source.ndim != 3 or len(target_shape) != 3:
        return None
    if int(source.shape[0]) != int(target_shape[0]) or int(source.shape[2]) != int(target_shape[2]):
        return None
    target_tokens = int(target_shape[1])
    if target_tokens <= 0:
        return None
    if int(source.shape[1]) == target_tokens:
        return source
    if target_tokens == 1:
        return source[:, :1, :]

    cls_pos = source[:, :1, :]
    frame_pos = source[:, 1:, :]
    if frame_pos.shape[1] <= 0:
        return None
    resized = F.interpolate(
        frame_pos.transpose(1, 2),
        size=target_tokens - 1,
        mode="linear",
        align_corners=False,
    ).transpose(1, 2)
    return torch.cat([cls_pos, resized], dim=1)


def load_model_init_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    *,
    checkpoint: dict[str, Any] | None = None,
    expected_backbone: str = "",
    strict: bool = False,
) -> set[str]:
    checkpoint = checkpoint or torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported init checkpoint format: {checkpoint_path}")
    raw_state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not isinstance(raw_state, dict):
        raise ValueError(f"Init checkpoint has no model/state_dict: {checkpoint_path}")

    checkpoint_cfg = _as_plain_dict(checkpoint.get("config"))
    checkpoint_model_cfg = _as_plain_dict(checkpoint_cfg.get("model"))
    source_backbone = str(checkpoint_model_cfg.get("backbone", "") or "")
    if expected_backbone and source_backbone and source_backbone != expected_backbone:
        print(
            f"WARN: init checkpoint backbone={source_backbone} differs from current backbone={expected_backbone}; "
            "shape-mismatched weights will be skipped",
            flush=True,
        )

    source_labels = checkpoint.get("labels")
    source_schema = checkpoint.get("label_schema")
    if source_labels is not None and list(source_labels) != LABELS:
        print(f"WARN: init checkpoint labels={source_labels} differ from current labels={LABELS}; mismatched head weights will be skipped", flush=True)
    if source_schema is not None and str(source_schema) != LABEL_SCHEMA:
        print(f"WARN: init checkpoint label_schema={source_schema} differs from current label_schema={LABEL_SCHEMA}", flush=True)

    state = strip_module_prefix(raw_state)
    target_state = model.state_dict()
    matched: dict[str, Tensor] = {}
    skipped: list[str] = []
    resized_pos_embed: list[str] = []
    unexpected: list[str] = []
    for key, value in state.items():
        target_key = key
        if target_key not in target_state:
            # Support progressive LoRA expansion. A checkpoint produced with
            # fewer adapted blocks stores an untouched Linear as
            # ``...qkv.weight`` while the expanded model exposes the same
            # pretrained tensor as ``...qkv.base.weight`` inside LoRALinear.
            # Remapping the base tensor makes the warm start explicit and
            # leaves only the newly introduced LoRA A/B tensors missing.
            if key.endswith(".weight") or key.endswith(".bias"):
                suffix = ".weight" if key.endswith(".weight") else ".bias"
                candidate = key[: -len(suffix)] + ".base" + suffix
                if candidate in target_state:
                    target_key = candidate
            if target_key not in target_state:
                unexpected.append(key)
                continue
        if tuple(value.shape) != tuple(target_state[target_key].shape):
            if target_key.endswith("pos_embed"):
                resized = resize_temporal_pos_embed(value, target_state[target_key].shape)
                if resized is not None and tuple(resized.shape) == tuple(target_state[target_key].shape):
                    matched[target_key] = resized.to(dtype=target_state[target_key].dtype)
                    resized_pos_embed.append(f"{target_key}: checkpoint{tuple(value.shape)} -> model{tuple(target_state[target_key].shape)}")
                    continue
            skipped.append(f"{target_key}: checkpoint{tuple(value.shape)} != model{tuple(target_state[target_key].shape)}")
            continue
        matched[target_key] = value

    missing, load_unexpected = model.load_state_dict(matched, strict=False)
    unexpected.extend(load_unexpected)
    print(
        f"Loaded init checkpoint {checkpoint_path}: matched={len(matched)} "
        f"missing={len(missing)} unexpected={len(unexpected)} skipped_shape={len(skipped)}",
        flush=True,
    )
    if skipped:
        print("Skipped shape-mismatched init weights:\n  " + "\n  ".join(skipped[:20]), flush=True)
        if len(skipped) > 20:
            print(f"  ... {len(skipped) - 20} more", flush=True)
    if resized_pos_embed:
        print("Resized temporal positional embeddings:\n  " + "\n  ".join(resized_pos_embed), flush=True)
    if strict and (missing or unexpected or skipped):
        raise RuntimeError(
            f"Strict init checkpoint load failed: missing={len(missing)} unexpected={len(unexpected)} skipped={len(skipped)}"
        )
    return set(matched)


def load_uniform_branch_init_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
) -> set[str]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Unsupported uniform init checkpoint format: {checkpoint_path}"
        )
    raw_state = checkpoint.get(
        "model", checkpoint.get("state_dict", checkpoint)
    )
    if not isinstance(raw_state, dict):
        raise ValueError(
            f"Uniform init checkpoint has no model/state_dict: {checkpoint_path}"
        )
    source = strip_module_prefix(raw_state)
    target = model.state_dict()
    matched: dict[str, Tensor] = {}
    resized: list[str] = []
    for source_prefix, target_prefix in (
        ("temporal.", "uniform_temporal."),
        ("head.", "uniform_head."),
    ):
        for key, value in source.items():
            if not key.startswith(source_prefix):
                continue
            target_key = target_prefix + key[len(source_prefix) :]
            if target_key not in target:
                continue
            target_value = target[target_key]
            if tuple(value.shape) == tuple(target_value.shape):
                matched[target_key] = value
                continue
            if target_key.endswith("pos_embed"):
                resized_value = resize_temporal_pos_embed(
                    value, target_value.shape
                )
                if (
                    resized_value is not None
                    and tuple(resized_value.shape) == tuple(target_value.shape)
                ):
                    matched[target_key] = resized_value.to(
                        dtype=target_value.dtype
                    )
                    resized.append(
                        f"{target_key}: checkpoint{tuple(value.shape)} "
                        f"-> model{tuple(target_value.shape)}"
                    )
    if not matched:
        raise RuntimeError(
            f"No uniform temporal/head weights matched from {checkpoint_path}"
        )
    model.load_state_dict(matched, strict=False)
    print(
        f"Loaded uniform branch init checkpoint {checkpoint_path}: "
        f"matched={len(matched)}",
        flush=True,
    )
    if resized:
        print(
            "Resized uniform temporal positional embeddings:\n  "
            + "\n  ".join(resized),
            flush=True,
        )
    return set(matched)


def load_event_branch_init_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
) -> set[str]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(
            f"Unsupported event init checkpoint format: {checkpoint_path}"
        )
    raw_state = checkpoint.get(
        "model", checkpoint.get("state_dict", checkpoint)
    )
    if not isinstance(raw_state, dict):
        raise ValueError(
            f"Event init checkpoint has no model/state_dict: {checkpoint_path}"
        )
    source = strip_module_prefix(raw_state)
    target = model.state_dict()
    matched: dict[str, Tensor] = {}
    resized: list[str] = []
    for prefix in ("temporal.", "head."):
        for key, value in source.items():
            if not key.startswith(prefix) or key not in target:
                continue
            target_value = target[key]
            if tuple(value.shape) == tuple(target_value.shape):
                matched[key] = value
                continue
            if key.endswith("pos_embed"):
                resized_value = resize_temporal_pos_embed(
                    value, target_value.shape
                )
                if (
                    resized_value is not None
                    and tuple(resized_value.shape) == tuple(target_value.shape)
                ):
                    matched[key] = resized_value.to(dtype=target_value.dtype)
                    resized.append(
                        f"{key}: checkpoint{tuple(value.shape)} "
                        f"-> model{tuple(target_value.shape)}"
                    )
    if not matched:
        raise RuntimeError(
            f"No event temporal/head weights matched from {checkpoint_path}"
        )
    model.load_state_dict(matched, strict=False)
    print(
        f"Loaded event branch init checkpoint {checkpoint_path}: "
        f"matched={len(matched)}",
        flush=True,
    )
    if resized:
        print(
            "Resized event temporal positional embeddings:\n  "
            + "\n  ".join(resized),
            flush=True,
        )
    return set(matched)


def configure_backbone_trainability(backbone: nn.Module, *, freeze: bool, finetune_last_blocks: int) -> None:
    if not freeze:
        for param in backbone.parameters():
            param.requires_grad = True
        return

    for param in backbone.parameters():
        param.requires_grad = False
    if finetune_last_blocks <= 0:
        return

    blocks = getattr(backbone, "blocks", None)
    if blocks is None:
        raise ValueError("finetune_last_blocks requires a ViT-style backbone with .blocks")
    for block in list(blocks)[-finetune_last_blocks:]:
        for param in block.parameters():
            param.requires_grad = True
    for module_name in ("norm", "cls_norm", "fc_norm"):
        module = getattr(backbone, module_name, None)
        if module is not None:
            for param in module.parameters():
                param.requires_grad = True


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Module, rank: int, alpha: float, dropout: float):
        super().__init__()
        if not hasattr(base, "in_features") or not hasattr(base, "out_features"):
            raise TypeError(f"LoRA target must expose in_features/out_features, got {type(base)}")
        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False
        self.in_features = int(base.in_features)
        self.out_features = int(base.out_features)
        self.rank = rank
        self.scaling = alpha / max(rank, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_a = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_b = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=5 ** 0.5)

    @property
    def weight(self) -> Tensor:
        return self.base.weight

    @property
    def bias(self) -> Tensor | None:
        return self.base.bias

    def forward(self, x: Tensor) -> Tensor:
        base_out = self.base(x)
        lora_out = F.linear(F.linear(self.dropout(x), self.lora_a), self.lora_b) * self.scaling
        return base_out + lora_out


def _resolve_child_module(module: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    parent = module
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_lora(backbone: nn.Module, lora_cfg: Any) -> int:
    for param in backbone.parameters():
        param.requires_grad = False
    blocks = getattr(backbone, "blocks", None)
    if blocks is None:
        raise ValueError("LoRA injection requires a ViT-style backbone with .blocks")
    rank = int(lora_cfg.get("rank", 8))
    alpha = float(lora_cfg.get("alpha", rank * 2))
    dropout = float(lora_cfg.get("dropout", 0.0))
    target_last_blocks = int(lora_cfg.get("target_last_blocks", 4))
    target_modules = list(lora_cfg.get("target_modules", ["attn.qkv", "attn.proj"]))
    selected_blocks = list(blocks)[-target_last_blocks:] if target_last_blocks > 0 else list(blocks)
    injected = 0
    for block in selected_blocks:
        for target_name in target_modules:
            parent, attr = _resolve_child_module(block, target_name)
            old_module = getattr(parent, attr)
            if isinstance(old_module, LoRALinear):
                continue
            setattr(parent, attr, LoRALinear(old_module, rank=rank, alpha=alpha, dropout=dropout))
            injected += 1
    if bool(lora_cfg.get("train_norm", False)):
        for module_name in ("norm", "cls_norm"):
            module = getattr(backbone, module_name, None)
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = True
    return injected


def extract_dino_frame_features(backbone: nn.Module, frames: Tensor) -> Tensor:
    features = backbone.forward_features(frames)
    cls = features["x_norm_clstoken"]
    patch_mean = features["x_norm_patchtokens"].mean(dim=1)
    return torch.cat([cls, patch_mean], dim=-1)



def get_dino_intermediate_layers_trainable_checkpointed(
    backbone: nn.Module,
    frames: Tensor,
    feature_layers: tuple[int, ...],
) -> tuple[tuple[Tensor, Tensor], ...]:
    """DINO intermediate layers with checkpointing only on trainable blocks.

    Frozen prefix blocks run exactly once. Blocks containing LoRA/trainable
    parameters are recomputed during backward, preserving the original state
    dict and the numerical forward path.
    """
    required = (
        "prepare_tokens_with_masks",
        "blocks",
        "norm",
        "n_storage_tokens",
    )
    missing = [name for name in required if not hasattr(backbone, name)]
    if missing:
        raise ValueError(
            "trainable_blocks checkpointing requires a DINO ViT backbone; "
            f"missing={missing}"
        )
    x, (height, width) = backbone.prepare_tokens_with_masks(frames)
    requested = set(int(index) for index in feature_layers)
    captured: list[Tensor] = []
    for index, block in enumerate(backbone.blocks):
        rope_sincos = (
            backbone.rope_embed(H=height, W=width)
            if getattr(backbone, "rope_embed", None) is not None
            else None
        )
        block_trainable = any(parameter.requires_grad for parameter in block.parameters())
        if backbone.training and block_trainable:
            def run_block(value: Tensor, module: nn.Module = block, rope: Any = rope_sincos) -> Tensor:
                return module(value, rope)

            x = activation_checkpoint(
                run_block,
                x,
                use_reentrant=False,
                preserve_rng_state=True,
            )
        else:
            x = block(x, rope_sincos)
        if index in requested:
            captured.append(x)
    if len(captured) != len(requested):
        raise ValueError(
            f"captured {len(captured)}/{len(requested)} requested DINO layers"
        )

    normalized: list[Tensor] = []
    for output in captured:
        if bool(getattr(backbone, "untie_cls_and_patch_norms", False)):
            cls_and_storage = backbone.cls_norm(
                output[:, : backbone.n_storage_tokens + 1]
            )
            patch_tokens = backbone.norm(
                output[:, backbone.n_storage_tokens + 1 :]
            )
            normalized.append(torch.cat((cls_and_storage, patch_tokens), dim=1))
        else:
            normalized.append(backbone.norm(output))
    return tuple(
        (
            output[:, backbone.n_storage_tokens + 1 :],
            output[:, 0],
        )
        for output in normalized
    )


def extract_dino_frame_features_and_patches(
    backbone: nn.Module,
    frames: Tensor,
    feature_mode: str = "last_cls_patch_mean",
    feature_layers: tuple[int, ...] = (-12, -8, -4, -1),
    patch_pool: nn.Module | None = None,
    checkpoint_trainable_blocks: bool = False,
) -> tuple[Tensor, Tensor]:
    """Extract frame features AND last-layer patch tokens for spatial attention.

    When *feature_mode* is ``multi_layer_cls_patch_attn`` the frame features
    are built from the specified intermediate layers while the patch tokens
    still come from the final layer so that spatial attention works
    unchanged.
    """
    if feature_mode == "last_cls_patch_mean":
        features = backbone.forward_features(frames)
        cls = features["x_norm_clstoken"]
        patches = features["x_norm_patchtokens"]
        pooled = torch.cat([cls, patches.mean(dim=1)], dim=-1)
        return pooled, patches

    # multi_layer_cls_patch_attn — one forward pass for all layers
    outputs = (
        get_dino_intermediate_layers_trainable_checkpointed(
            backbone, frames, feature_layers
        )
        if checkpoint_trainable_blocks
        else backbone.get_intermediate_layers(
            frames,
            n=feature_layers,
            return_class_token=True,
            norm=True,
        )
    )
    pieces: list[Tensor] = []
    for patches, cls in outputs:
        pieces.append(cls)
        if patch_pool is not None:
            weights = patch_pool(patches).softmax(dim=1)
            pieces.append((patches * weights).sum(dim=1))
        else:
            pieces.append(patches.mean(dim=1))
    multi_layer_features = torch.cat(pieces, dim=-1)
    # spatial attention always consumes the final-layer patch tokens
    last_patches, _last_cls = outputs[-1]
    return multi_layer_features, last_patches


def _coerce_frame_feature_layers(raw_layers: Any) -> tuple[int, ...]:
    if raw_layers is None:
        return (-12, -8, -4, -1)
    if isinstance(raw_layers, str):
        text = raw_layers.strip()
        if not text:
            return (-12, -8, -4, -1)
        text = text.strip("[]")
        values = [item.strip() for item in text.split(",") if item.strip()]
        return tuple(int(item) for item in values)
    return tuple(int(item) for item in raw_layers)


class TemporalTransformer(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, num_heads: int, dropout: float, max_frames: int):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_frames + 1, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: Tensor, position_indices: Tensor | None = None) -> Tensor:
        bsz, frames, _ = x.shape
        cls = self.cls_token.expand(bsz, -1, -1)
        x = torch.cat([cls, x], dim=1)
        if position_indices is None:
            position = self.pos_embed[:, : frames + 1]
        else:
            if position_indices.shape != (bsz, frames):
                raise ValueError(
                    f"position_indices shape={tuple(position_indices.shape)} "
                    f"must be {(bsz, frames)}"
                )
            frame_position = self.pos_embed[:, 1:].expand(bsz, -1, -1)
            frame_position = frame_position.gather(
                1,
                position_indices.unsqueeze(-1).expand(-1, -1, frame_position.shape[-1]),
            )
            position = torch.cat(
                [self.pos_embed[:, :1].expand(bsz, -1, -1), frame_position],
                dim=1,
            )
        x = x + position
        x = self.encoder(x)
        return self.norm(x[:, 0])


class TemporalStemBottleneckTransformer(nn.Module):
    """Local-first temporal head for football event spotting.

    It strengthens short local motion with depthwise temporal convolutions,
    aggregates multi-scale temporal context, runs a compact bottleneck
    transformer, then decodes back to frame resolution. The returned clip
    token feeds the clip-context head; decoded frame tokens feed dense/frame
    supervision.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        max_frames: int,
        *,
        stem_kernel_size: int = 3,
        multiscale_dilations: Sequence[int] = (1, 2, 4),
        bottleneck_frames: int = 8,
    ) -> None:
        super().__init__()
        kernel = max(int(stem_kernel_size), 1)
        if kernel % 2 == 0:
            kernel += 1
        self.max_frames = max(int(max_frames), 1)
        self.bottleneck_frames = max(int(bottleneck_frames), 1)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.stem_depthwise = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=kernel, padding=kernel // 2, groups=hidden_dim
        )
        self.stem_pointwise = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1),
        )
        dilations = tuple(max(int(item), 1) for item in multiscale_dilations) or (1, 2, 4)
        self.multiscale = nn.ModuleList(
            [
                nn.Conv1d(
                    hidden_dim, hidden_dim, kernel_size=3, padding=dilation,
                    dilation=dilation, groups=hidden_dim,
                )
                for dilation in dilations
            ]
        )
        self.multiscale_mix = nn.Sequential(
            nn.Conv1d(hidden_dim * len(dilations), hidden_dim * 2, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_frames + 1, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(int(num_layers), 1))
        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _resize_bottleneck(self, x: Tensor, frames: int) -> Tensor:
        target = min(max(self.bottleneck_frames, 1), max(frames, 1), self.max_frames)
        if frames == target:
            return x
        return F.interpolate(x.transpose(1, 2), size=target, mode="linear", align_corners=False).transpose(1, 2)

    def forward(self, x: Tensor, position_indices: Tensor | None = None) -> dict[str, Tensor]:
        del position_indices
        if x.ndim != 3:
            raise ValueError("TemporalStemBottleneckTransformer expects [batch, frames, hidden]")
        bsz, frames, _hidden = x.shape
        stem_input = self.input_norm(x).transpose(1, 2)
        stem_delta = self.stem_pointwise(F.gelu(self.stem_depthwise(stem_input)))
        stem = x + stem_delta.transpose(1, 2)
        stem_t = stem.transpose(1, 2)
        multiscale_delta = self.multiscale_mix(
            torch.cat([F.gelu(conv(stem_t)) for conv in self.multiscale], dim=1)
        ).transpose(1, 2)
        encoded_frames = stem + multiscale_delta
        bottleneck = self._resize_bottleneck(encoded_frames, frames)
        bottleneck_len = bottleneck.shape[1]
        sequence = torch.cat([self.cls_token.expand(bsz, -1, -1), bottleneck], dim=1)
        sequence = self.encoder(sequence + self.pos_embed[:, : bottleneck_len + 1])
        clip_token = self.output_norm(sequence[:, 0])
        decoded = F.interpolate(
            sequence[:, 1:].transpose(1, 2), size=frames, mode="linear", align_corners=False
        ).transpose(1, 2)
        dense_tokens = self.output_norm(
            encoded_frames + self.decoder(torch.cat([encoded_frames, decoded], dim=-1))
        )
        return {"clip_token": clip_token, "frame_tokens": dense_tokens}


class AttentionPoolTransformer(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, num_heads: int, dropout: float, max_frames: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_frames, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.attn = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        self.norm = nn.LayerNorm(hidden_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        frames = x.shape[1]
        x = x + self.pos_embed[:, :frames]
        x = self.encoder(x)
        weights = self.attn(x).softmax(dim=1)
        pooled = (x * weights).sum(dim=1)
        return self.norm(pooled), weights.squeeze(-1)


class ClassQueryTransformer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        max_frames: int,
        num_labels: int,
    ):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_frames, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.class_queries = nn.Parameter(torch.zeros(1, num_labels, hidden_dim))
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.class_queries, std=0.02)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        bsz, frames, _ = x.shape
        x = x + self.pos_embed[:, :frames]
        x = self.encoder(x)
        queries = self.class_queries.expand(bsz, -1, -1)
        attended, weights = self.cross_attn(
            queries,
            x,
            x,
            need_weights=True,
            average_attn_weights=False,
        )
        attended = self.norm(attended)
        logits = self.classifier(attended).squeeze(-1)
        attention = weights.mean(dim=1).transpose(1, 2)
        return attended.mean(dim=1), logits, attention


class AttentionBiGRU(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        gru_hidden = max(hidden_dim // 2, 1)
        self.gru = nn.GRU(hidden_dim, gru_hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.attn = nn.Sequential(nn.LayerNorm(gru_hidden * 2), nn.Linear(gru_hidden * 2, 1))
        self.out = nn.Sequential(nn.Dropout(dropout), nn.Linear(gru_hidden * 2, hidden_dim), nn.LayerNorm(hidden_dim))

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.gru(x)
        weights = self.attn(x).softmax(dim=1)
        pooled = (x * weights).sum(dim=1)
        return self.out(pooled)

class TemporalDifferenceAdapter(nn.Module):
    """Add explicit first/second-order temporal changes as a residual token update."""

    def __init__(
        self,
        hidden_dim: int,
        mode: str = "delta2",
        dropout: float = 0.1,
        gate_init: float = 0.25,
    ):
        super().__init__()
        self.mode = str(mode).strip().lower()
        if self.mode not in ("delta1", "delta2"):
            raise ValueError("model.temporal_difference.mode must be delta1 or delta2")
        input_multiplier = 2 if self.mode == "delta1" else 3
        self.adapter = nn.Sequential(
            nn.LayerNorm(hidden_dim * input_multiplier),
            nn.Linear(hidden_dim * input_multiplier, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        gate_init = min(max(float(gate_init), 1e-4), 1.0 - 1e-4)
        self.gate_logit = nn.Parameter(torch.logit(torch.tensor(gate_init)))
        # Loading a baseline checkpoint starts from the exact baseline output.
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)

    @staticmethod
    def temporal_differences(x: Tensor) -> tuple[Tensor, Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Temporal tokens must be BxTxD, got {tuple(x.shape)}")
        first = torch.cat(
            [x.new_zeros((x.shape[0], 1, x.shape[2])), x[:, 1:] - x[:, :-1]],
            dim=1,
        )
        if x.shape[1] < 3:
            second = torch.zeros_like(x)
        else:
            second = torch.cat(
                [
                    x.new_zeros((x.shape[0], 2, x.shape[2])),
                    x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2],
                ],
                dim=1,
            )
        return first, second

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        first, second = self.temporal_differences(x)
        pieces = [x, first]
        if self.mode == "delta2":
            pieces.append(second)
        residual = self.adapter(torch.cat(pieces, dim=-1))
        gate = torch.sigmoid(self.gate_logit).to(dtype=x.dtype)
        return x + gate * residual, residual, gate



class _TemporalConvBlock(nn.Module):
    """Single TCN residual block with dilated causal conv1d."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.causal_conv = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            )
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.residual_conv = nn.Conv1d(channels, channels, 1)
        self.downstream_conv = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.causal_conv(x)
        if out.shape[-1] > x.shape[-1]:
            out = out[..., : x.shape[-1]]
        out = self.relu(out)
        out = self.dropout(out)
        out = self.downstream_conv(out)
        if out.shape[-1] > x.shape[-1]:
            out = out[..., : x.shape[-1]]
        out = self.relu(out)
        out = self.dropout(out)
        residual = self.residual_conv(x)
        return self.relu(out + residual)


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network for clip-level event classification.

    Uses dilated causal 1D convolutions to capture multi-scale temporal
    patterns with an exponentially growing receptive field. Designed as a
    drop-in replacement for ``TemporalTransformer``.

    Parameters
    ----------
    hidden_dim : int
        Feature dimension per frame token.
    num_layers : int
        Number of stacked TCN residual blocks (controls receptive field depth).
    kernel_size : int
        Temporal kernel size (default 3).
    dropout : float
        Dropout probability applied after each activation.
    base_dilation : int
        Starting dilation rate; doubles every block.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 5,
        kernel_size: int = 3,
        dropout: float = 0.1,
        base_dilation: int = 1,
    ):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        blocks = []
        for layer_idx in range(num_layers):
            dilation = base_dilation * (2 ** layer_idx)
            blocks.append(
                _TemporalConvBlock(
                    hidden_dim,
                    kernel_size,
                    dilation,
                    dropout,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, F, D) -> conv1d expects (B, D, F)
        x = self.proj(x)
        x = x.transpose(1, 2)  # (B, D, F)
        for block in self.blocks:
            x = block(x)
        x = x.transpose(1, 2)  # (B, F, D)
        # Pool over the temporal dimension: mean of all frame tokens
        x = x.mean(dim=1)
        return self.norm(x)


class TemporalConditionedSpatialAttention(nn.Module):
    """Read class-specific regions using queries conditioned on clip dynamics."""

    def __init__(
        self,
        *,
        patch_dim: int,
        hidden_dim: int,
        attention_dim: int,
        num_labels: int,
        queries_per_class: int,
        context_layers: int,
        temporal_layers: int,
        num_heads: int,
        dropout: float,
        max_frames: int,
        gate_init: float,
        dynamic_query_scale_init: float = 0.0,
        attention_mode: str = "softmax",
        topk_ratio: float = 0.03,
        residual_max_delta: float = 1.0,
        actionness_query_scale_init: float = 0.0,
        return_attention_maps: bool = False,
    ):
        super().__init__()
        if attention_dim % num_heads != 0:
            raise ValueError(
                "model.spatial_attention.attention_dim must be divisible by "
                "model.spatial_attention.num_heads"
            )
        if hidden_dim % num_heads != 0:
            raise ValueError(
                "model.hidden_dim must be divisible by "
                "model.spatial_attention.num_heads"
            )
        self.num_labels = int(num_labels)
        self.queries_per_class = max(int(queries_per_class), 1)
        self.attention_dim = int(attention_dim)
        self.return_attention_maps = bool(return_attention_maps)
        self.attention_mode = str(attention_mode).strip().lower()
        if self.attention_mode not in {"softmax", "topk_softmax"}:
            raise ValueError(
                "model.spatial_attention.attention_mode must be softmax or topk_softmax"
            )
        self.topk_ratio = float(topk_ratio)
        if not 0.0 < self.topk_ratio <= 1.0:
            raise ValueError("model.spatial_attention.topk_ratio must be in (0, 1]")
        self.residual_max_delta = float(residual_max_delta)
        if self.residual_max_delta <= 0.0:
            raise ValueError("model.spatial_attention.residual_max_delta must be positive")

        self.context_pos_embed = nn.Parameter(
            torch.zeros(1, max_frames, hidden_dim)
        )
        context_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(
            context_layer,
            num_layers=max(int(context_layers), 1),
        )
        self.context_norm = nn.LayerNorm(hidden_dim)

        self.base_queries = nn.Parameter(
            torch.zeros(
                1,
                1,
                self.num_labels,
                self.queries_per_class,
                attention_dim,
            )
        )
        dynamic_query_dim = (
            self.num_labels * self.queries_per_class * attention_dim
        )
        self.context_query = nn.Linear(hidden_dim, dynamic_query_dim)
        self.motion_query = nn.Linear(hidden_dim, dynamic_query_dim)
        self.context_query_scale = nn.Parameter(
            torch.tensor(float(dynamic_query_scale_init))
        )
        self.motion_query_scale = nn.Parameter(
            torch.tensor(float(dynamic_query_scale_init))
        )
        self.actionness_queries = nn.Parameter(
            torch.zeros(
                1,
                1,
                self.num_labels,
                self.queries_per_class,
                attention_dim,
            )
        )
        self.actionness_query_scale = nn.Parameter(
            torch.tensor(float(actionness_query_scale_init))
        )
        self.query_norm = nn.LayerNorm(attention_dim)
        self.patch_norm = nn.LayerNorm(patch_dim)
        self.patch_key = nn.Linear(patch_dim, attention_dim, bias=False)
        self.key_norm = nn.LayerNorm(attention_dim)
        self.patch_value = nn.Linear(patch_dim, hidden_dim, bias=False)
        self.query_pool_logits = nn.Parameter(
            torch.zeros(self.num_labels, self.queries_per_class)
        )
        self.region_adapter = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.region_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, max(hidden_dim // 2, 64)),
            nn.GELU(),
            nn.Linear(max(hidden_dim // 2, 64), 1),
        )
        gate_init = min(max(float(gate_init), 1e-4), 1.0 - 1e-4)
        nn.init.zeros_(self.region_gate[-1].weight)
        nn.init.constant_(
            self.region_gate[-1].bias,
            torch.logit(torch.tensor(gate_init)).item(),
        )

        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, max_frames + 1, hidden_dim)
        )
        self.class_tokens = nn.Parameter(
            torch.zeros(1, self.num_labels, hidden_dim)
        )
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer,
            num_layers=max(int(temporal_layers), 1),
        )
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        self.clip_classifier = nn.Linear(hidden_dim, 1)
        self.clip_residual_head = nn.Linear(hidden_dim, 1)
        self.frame_event_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

        nn.init.trunc_normal_(self.base_queries, std=0.02)
        nn.init.trunc_normal_(self.actionness_queries, std=0.02)
        nn.init.trunc_normal_(self.context_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.class_tokens, std=0.02)
        nn.init.zeros_(self.clip_residual_head.weight)
        nn.init.zeros_(self.clip_residual_head.bias)

    def query_diversity_loss(self) -> Tensor:
        if self.queries_per_class <= 1:
            return self.base_queries.new_zeros(())
        queries = F.normalize(self.base_queries[0, 0], dim=-1)
        similarity = torch.einsum("cqd,ckd->cqk", queries, queries)
        eye = torch.eye(
            self.queries_per_class,
            device=similarity.device,
            dtype=similarity.dtype,
        ).unsqueeze(0)
        off_diagonal = similarity * (1.0 - eye)
        denominator = float(
            self.num_labels
            * self.queries_per_class
            * (self.queries_per_class - 1)
        )
        return off_diagonal.square().sum() / max(denominator, 1.0)

    def forward(
        self,
        global_frame_tokens: Tensor,
        patch_tokens: Tensor,
        frame_actionness_logits: Tensor | None = None,
    ) -> dict[str, Tensor]:
        bsz, frames, hidden_dim = global_frame_tokens.shape
        if patch_tokens.shape[:2] != (bsz, frames):
            raise ValueError(
                f"patch_tokens shape={tuple(patch_tokens.shape)} must start "
                f"with {(bsz, frames)}"
            )
        if frames > self.context_pos_embed.shape[1]:
            raise ValueError(
                f"Received {frames} frames, but spatial attention max_frames="
                f"{self.context_pos_embed.shape[1]}"
            )

        context = global_frame_tokens + self.context_pos_embed[:, :frames]
        context = self.context_norm(self.context_encoder(context))
        motion = torch.zeros_like(context)
        if frames > 1:
            motion[:, 1:] = context[:, 1:] - context[:, :-1]

        dynamic_query = (
            self.context_query_scale * self.context_query(context)
            + self.motion_query_scale * self.motion_query(motion)
        ).reshape(
            bsz,
            frames,
            self.num_labels,
            self.queries_per_class,
            self.attention_dim,
        )
        if frame_actionness_logits is not None:
            expected_actionness_shape = (bsz, frames, self.num_labels)
            if tuple(frame_actionness_logits.shape) != expected_actionness_shape:
                raise ValueError(
                    "frame_actionness_logits shape="
                    f"{tuple(frame_actionness_logits.shape)} must equal "
                    f"{expected_actionness_shape}"
                )
            actionness = torch.sigmoid(
                frame_actionness_logits.detach().to(dynamic_query.dtype)
            ).unsqueeze(-1).unsqueeze(-1)
            dynamic_query = dynamic_query + (
                self.actionness_query_scale
                * actionness
                * self.actionness_queries
            )
        queries = self.query_norm(self.base_queries + dynamic_query)
        normalized_patches = self.patch_norm(patch_tokens)
        keys = self.key_norm(self.patch_key(normalized_patches))
        values = self.patch_value(normalized_patches)
        scores = torch.einsum(
            "btcqa,btna->btcqn", queries, keys
        ) / math.sqrt(float(self.attention_dim))
        if self.attention_mode == "topk_softmax":
            patch_count = scores.shape[-1]
            topk_count = min(
                patch_count,
                max(1, int(math.ceil(self.topk_ratio * patch_count))),
            )
            topk_scores, topk_indices = torch.topk(
                scores, k=topk_count, dim=-1, sorted=False
            )
            # softmax may be promoted to fp32 under AMP while scores remain
            # bf16; scatter requires src and destination to share a dtype.
            topk_attention = topk_scores.softmax(dim=-1).to(scores.dtype)
            attention = torch.zeros_like(scores).scatter(
                -1, topk_indices, topk_attention
            )
            active_patch_fraction = scores.new_tensor(
                float(topk_count) / float(patch_count)
            )
        else:
            attention = scores.softmax(dim=-1)
            active_patch_fraction = scores.new_tensor(1.0)
        region_queries = torch.einsum(
            "btcqn,btnh->btcqh", attention, values
        )
        query_pool = self.query_pool_logits.softmax(dim=-1)
        region_tokens = (
            region_queries
            * query_pool.reshape(
                1, 1, self.num_labels, self.queries_per_class, 1
            )
        ).sum(dim=3)
        region_delta = self.region_adapter(region_tokens)

        expanded_context = context.unsqueeze(2).expand(
            -1, -1, self.num_labels, -1
        )
        gate_input = torch.cat(
            [
                expanded_context,
                region_delta,
                (region_delta - expanded_context).abs(),
            ],
            dim=-1,
        )
        gate = torch.sigmoid(self.region_gate(gate_input))
        # The residual classifier receives attended region evidence only.
        # Global context conditions query selection and the confidence gate,
        # while the unchanged global classifier remains the reference path.
        spatial_tokens = region_delta

        class_sequences = spatial_tokens.permute(0, 2, 1, 3).reshape(
            bsz * self.num_labels, frames, hidden_dim
        )
        class_tokens = self.class_tokens.expand(bsz, -1, -1).reshape(
            bsz * self.num_labels, 1, hidden_dim
        )
        class_sequences = torch.cat([class_tokens, class_sequences], dim=1)
        class_sequences = (
            class_sequences + self.temporal_pos_embed[:, : frames + 1]
        )
        class_sequences = self.temporal_encoder(class_sequences)
        class_features = self.temporal_norm(class_sequences[:, 0]).reshape(
            bsz, self.num_labels, hidden_dim
        )
        clip_logits = self.clip_classifier(class_features).squeeze(-1)
        unbounded_residual_logits = self.clip_residual_head(class_features).squeeze(-1)
        raw_residual_logits = self.residual_max_delta * torch.tanh(
            unbounded_residual_logits
        )
        frame_gate = gate.squeeze(-1)
        gate_topk = min(3, frames)
        clip_gate = torch.topk(frame_gate, k=gate_topk, dim=1).values.mean(dim=1)
        residual_logits = clip_gate * raw_residual_logits
        frame_event_logits = self.frame_event_head(region_delta).squeeze(-1)

        probabilities = attention.clamp_min(1e-8)
        entropy = -(probabilities * probabilities.log()).sum(dim=-1)
        entropy_support = (
            max(1, int(math.ceil(self.topk_ratio * patch_tokens.shape[2])))
            if self.attention_mode == "topk_softmax"
            else patch_tokens.shape[2]
        )
        entropy = entropy / max(math.log(float(entropy_support)), 1e-6)
        if self.queries_per_class > 1:
            normalized_attention = F.normalize(attention, dim=-1)
            attention_similarity = torch.einsum(
                "btcqn,btckn->btcqk",
                normalized_attention,
                normalized_attention,
            )
            query_eye = torch.eye(
                self.queries_per_class,
                device=attention.device,
                dtype=attention.dtype,
            ).reshape(1, 1, 1, self.queries_per_class, self.queries_per_class)
            attention_overlap = (
                attention_similarity * (1.0 - query_eye)
            ).sum(dim=(-1, -2)) / float(
                self.queries_per_class * (self.queries_per_class - 1)
            )
        else:
            attention_overlap = entropy.new_zeros(
                (bsz, frames, self.num_labels)
            )
        result = {
            "clip_logits": clip_logits,
            "residual_logits": residual_logits,
            "raw_residual_logits": raw_residual_logits,
            "frame_event_logits": frame_event_logits,
            "gate": frame_gate,
            "clip_gate": clip_gate,
            "attention_entropy": entropy,
            "attention_overlap": attention_overlap,
            "query_diversity_loss": self.query_diversity_loss(),
            "context_query_scale": self.context_query_scale,
            "class_features": class_features,
            "roi_frame_features": region_delta,
            "motion_query_scale": self.motion_query_scale,
            "actionness_query_scale": self.actionness_query_scale,
            "active_patch_fraction": active_patch_fraction,
        }
        if self.return_attention_maps:
            result["attention_maps"] = attention
        return result


class AdaptiveSpatialTokenPool(nn.Module):
    """Compress all frame patches into distinct, temporally persistent slots."""

    def __init__(
        self,
        *,
        patch_dim: int,
        hidden_dim: int,
        attention_dim: int,
        num_queries: int,
        num_labels: int,
        slot_dropout: float,
        gate_init: float,
        return_attention_maps: bool = False,
    ):
        super().__init__()
        self.num_queries = max(int(num_queries), 1)
        self.attention_dim = int(attention_dim)
        self.slot_dropout = float(np.clip(slot_dropout, 0.0, 0.95))
        self.return_attention_maps = bool(return_attention_maps)

        self.base_queries = nn.Parameter(
            torch.empty(self.num_queries, self.attention_dim)
        )
        self.context_query = nn.Linear(
            hidden_dim, self.num_queries * self.attention_dim
        )
        self.context_scale = nn.Parameter(torch.tensor(0.1))
        self.query_norm = nn.LayerNorm(self.attention_dim)
        self.patch_norm = nn.LayerNorm(patch_dim)
        self.patch_key = nn.Linear(patch_dim, self.attention_dim, bias=False)
        self.key_norm = nn.LayerNorm(self.attention_dim)
        self.patch_value = nn.Linear(patch_dim, hidden_dim, bias=False)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.slot_type_embed = nn.Parameter(
            torch.zeros(1, 1, self.num_queries + 1, hidden_dim)
        )
        self.slot_event_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_labels),
        )

        gate_init = float(np.clip(gate_init, 1e-4, 1.0 - 1e-4))
        self.slot_gate_logits = nn.Parameter(
            torch.full(
                (self.num_queries,),
                torch.logit(torch.tensor(gate_init)).item(),
            )
        )
        nn.init.trunc_normal_(self.base_queries, std=0.02)
        nn.init.zeros_(self.context_query.weight)
        nn.init.zeros_(self.context_query.bias)
        nn.init.trunc_normal_(self.slot_type_embed[:, :, 1:], std=0.02)

    def query_diversity_loss(self) -> Tensor:
        if self.num_queries <= 1:
            return self.base_queries.new_zeros(())
        queries = F.normalize(self.base_queries, dim=-1)
        similarity = queries @ queries.transpose(0, 1)
        eye = torch.eye(
            self.num_queries,
            device=similarity.device,
            dtype=similarity.dtype,
        )
        off_diagonal = similarity * (1.0 - eye)
        return off_diagonal.square().sum() / float(
            self.num_queries * (self.num_queries - 1)
        )

    def forward(
        self,
        global_frame_tokens: Tensor,
        patch_tokens: Tensor,
    ) -> dict[str, Tensor]:
        bsz, frames, _hidden_dim = global_frame_tokens.shape
        if patch_tokens.shape[:2] != (bsz, frames):
            raise ValueError(
                f"patch_tokens shape={tuple(patch_tokens.shape)} must start "
                f"with {(bsz, frames)}"
            )

        dynamic_queries = self.context_query(global_frame_tokens).reshape(
            bsz, frames, self.num_queries, self.attention_dim
        )
        queries = self.query_norm(
            self.base_queries.reshape(1, 1, self.num_queries, self.attention_dim)
            + self.context_scale * dynamic_queries
        )
        normalized_patches = self.patch_norm(patch_tokens)
        keys = self.key_norm(self.patch_key(normalized_patches))
        values = self.patch_value(normalized_patches)
        scores = torch.einsum(
            "btqa,btna->btqn", queries, keys
        ) / math.sqrt(float(self.attention_dim))
        attention = scores.softmax(dim=-1)
        local_tokens = self.output_norm(
            torch.einsum("btqn,btnh->btqh", attention, values)
        )

        slot_gate = torch.sigmoid(self.slot_gate_logits).reshape(
            1, 1, self.num_queries, 1
        )
        gated_tokens = local_tokens * slot_gate
        temporal_local_tokens = gated_tokens
        if self.training and self.slot_dropout > 0.0:
            keep_probability = 1.0 - self.slot_dropout
            # Keep a slot present or absent for the whole clip. Per-frame slot
            # dropout changes the identity of a temporal slot and introduces
            # artificial flicker into the temporal encoder.
            keep = torch.empty(
                bsz,
                1,
                self.num_queries,
                1,
                device=local_tokens.device,
                dtype=local_tokens.dtype,
            ).bernoulli_(keep_probability)
            temporal_local_tokens = temporal_local_tokens * (
                keep / keep_probability
            )

        sequence = torch.cat(
            [global_frame_tokens.unsqueeze(2), temporal_local_tokens], dim=2
        )
        sequence = sequence + self.slot_type_embed
        slot_logits = self.slot_event_head(local_tokens)

        probabilities = attention.clamp_min(1e-8)
        entropy = -(probabilities * probabilities.log()).sum(dim=-1)
        entropy = entropy / max(math.log(float(patch_tokens.shape[2])), 1e-6)
        if self.num_queries > 1:
            normalized_attention = F.normalize(attention, dim=-1)
            similarity = torch.einsum(
                "btqn,btkn->btqk", normalized_attention, normalized_attention
            )
            eye = torch.eye(
                self.num_queries,
                device=similarity.device,
                dtype=similarity.dtype,
            ).reshape(1, 1, self.num_queries, self.num_queries)
            overlap = (similarity * (1.0 - eye)).sum(dim=(-1, -2)) / float(
                self.num_queries * (self.num_queries - 1)
            )
        else:
            overlap = entropy.new_zeros((bsz, frames))

        result = {
            "sequence": sequence,
            "local_tokens": local_tokens,
            "gated_local_tokens": gated_tokens,
            "temporal_local_tokens": temporal_local_tokens,
            "slot_logits": slot_logits,
            "attention_entropy": entropy,
            "attention_overlap": overlap,
            "query_diversity_loss": self.query_diversity_loss(),
            "slot_gate": torch.sigmoid(self.slot_gate_logits),
            "context_scale": self.context_scale,
        }
        if self.return_attention_maps:
            result["attention_maps"] = attention
        return result


class JointSpatialTemporalTransformer(nn.Module):
    """Directly model global and K4 local tokens in one temporal sequence."""

    def __init__(
        self,
        hidden_dim: int,
        num_slots: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        max_frames: int,
        include_global_token: bool = True,
        modality_dropout: bool = False,
    ) -> None:
        super().__init__()
        self.num_slots = int(num_slots)
        self.include_global_token = bool(include_global_token)
        self.modality_dropout = bool(modality_dropout)
        self.tokens_per_frame = self.num_slots + int(self.include_global_token)
        self.event_cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_frames, self.tokens_per_frame, hidden_dim))
        self.type_embed = nn.Parameter(torch.zeros(1, 1, self.tokens_per_frame, hidden_dim))
        layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        nn.init.trunc_normal_(self.event_cls, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.type_embed, std=0.02)

    def forward(self, global_tokens: Tensor, local_tokens: Tensor) -> dict[str, Tensor]:
        bsz, frames, hidden_dim = global_tokens.shape
        if local_tokens.shape[:3] != (bsz, frames, self.num_slots):
            raise ValueError(f"local token shape={tuple(local_tokens.shape)} incompatible with global={tuple(global_tokens.shape)}")
        if self.modality_dropout and self.training:
            global_keep = (torch.rand(bsz, frames, 1, 1, device=global_tokens.device) > 0.30).to(global_tokens.dtype)
            local_keep = (torch.rand(bsz, frames, self.num_slots, 1, device=local_tokens.device) > 0.15).to(local_tokens.dtype)
            global_tokens = global_tokens * global_keep.squeeze(2)
            local_tokens = local_tokens * local_keep
        if self.include_global_token:
            frame_tokens = torch.cat([global_tokens.unsqueeze(2), local_tokens], dim=2)
        else:
            frame_tokens = local_tokens
        frame_tokens = frame_tokens + self.pos_embed[:, :frames] + self.type_embed
        sequence = frame_tokens.reshape(bsz, frames * self.tokens_per_frame, hidden_dim)
        sequence = torch.cat([self.event_cls.expand(bsz, -1, -1), sequence], dim=1)
        encoded = self.norm(self.encoder(sequence))
        event = encoded[:, 0]
        frame_encoded = encoded[:, 1:].reshape(bsz, frames, self.tokens_per_frame, hidden_dim)
        if self.include_global_token:
            global_frame_tokens = frame_encoded[:, :, 0]
            local_frame_tokens = frame_encoded[:, :, 1:]
        else:
            local_frame_tokens = frame_encoded
            global_frame_tokens = local_frame_tokens.mean(dim=2)
        return {"event": event, "global_frame_tokens": global_frame_tokens, "local_frame_tokens": local_frame_tokens}


class LightweightClassEvidenceHead(nn.Module):
    """Class-specific patch evidence that safely corrects a frozen baseline."""
    def __init__(self, *, patch_dim: int, hidden_dim: int, attention_dim: int,
                 evidence_dim: int, num_labels: int, queries_per_class: int,
                 topk_per_class: Sequence[int], context_frames_per_class: Sequence[int],
                 positive_max_per_class: Sequence[float],
                 negative_max_per_class: Sequence[float], dropout: float,
                 return_attention_maps: bool = False):
        super().__init__()
        self.num_labels = int(num_labels)
        self.queries_per_class = max(int(queries_per_class), 1)
        self.attention_dim = int(attention_dim)
        self.return_attention_maps = bool(return_attention_maps)

        def class_values(values: Sequence[Any], cast: Any, name: str) -> tuple[Any, ...]:
            result = tuple(cast(value) for value in values)
            if len(result) != self.num_labels:
                raise ValueError(f"{name} must contain one value per label")
            return result

        self.topk_per_class = class_values(
            topk_per_class, lambda value: max(int(value), 1), "topk_per_class"
        )
        self.context_frames_per_class = class_values(
            context_frames_per_class, lambda value: max(int(value), 0),
            "context_frames_per_class",
        )
        positive_max = class_values(
            positive_max_per_class, float, "positive_max_per_class"
        )
        negative_max = class_values(
            negative_max_per_class, float, "negative_max_per_class"
        )
        self.register_buffer("positive_max", torch.tensor(positive_max).reshape(1, -1))
        self.register_buffer("negative_max", torch.tensor(negative_max).reshape(1, -1))
        self.base_queries = nn.Parameter(
            torch.empty(self.num_labels, self.queries_per_class, self.attention_dim)
        )
        self.global_query = nn.Linear(
            hidden_dim, self.queries_per_class * self.attention_dim
        )
        self.query_scale = nn.Parameter(torch.tensor(0.1))
        self.query_norm = nn.LayerNorm(self.attention_dim)
        self.patch_norm = nn.LayerNorm(patch_dim)
        self.patch_key = nn.Linear(patch_dim, self.attention_dim, bias=False)
        self.patch_value = nn.Linear(patch_dim, evidence_dim, bias=False)
        self.query_pool_logits = nn.Parameter(
            torch.zeros(self.num_labels, self.queries_per_class)
        )
        self.global_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, evidence_dim)
        )
        self.frame_score_proj = nn.Sequential(
            nn.Linear(self.num_labels + 1, evidence_dim), nn.GELU()
        )
        self.frame_fusion = nn.Sequential(
            nn.LayerNorm(evidence_dim * 3), nn.Linear(evidence_dim * 3, evidence_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.temporal_mlp = nn.Sequential(
            nn.LayerNorm(evidence_dim * 3), nn.Linear(evidence_dim * 3, evidence_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.temporal_score = nn.Linear(evidence_dim, 1)
        self.correction_weight = nn.Parameter(
            torch.zeros(self.num_labels, evidence_dim)
        )
        self.correction_bias = nn.Parameter(torch.zeros(self.num_labels))
        nn.init.trunc_normal_(self.base_queries, std=0.02)
        nn.init.zeros_(self.global_query.weight)
        nn.init.zeros_(self.global_query.bias)

    def _selected_indices(self, frame_logits: Tensor) -> tuple[Tensor, Tensor]:
        bsz, frames, labels = frame_logits.shape
        if labels != self.num_labels:
            raise ValueError(f"frame logits labels={labels}, expected {self.num_labels}")
        max_selected = max(
            min(topk, frames) + min(context, frames)
            for topk, context in zip(
                self.topk_per_class, self.context_frames_per_class
            )
        )
        selections: list[Tensor] = []
        valid_masks: list[Tensor] = []
        for label_index, (topk, context_count) in enumerate(
            zip(self.topk_per_class, self.context_frames_per_class)
        ):
            indices = torch.topk(
                frame_logits[:, :, label_index].detach(),
                k=min(topk, frames), dim=1,
            ).indices
            if context_count > 0:
                context = torch.linspace(
                    0, frames - 1, min(context_count, frames),
                    device=frame_logits.device,
                ).round().long().reshape(1, -1).expand(bsz, -1)
                indices = torch.cat([indices, context], dim=1)
            indices = indices.sort(dim=1).values
            valid = torch.ones_like(indices, dtype=torch.bool)
            if indices.shape[1] < max_selected:
                pad_count = max_selected - indices.shape[1]
                indices = torch.cat(
                    [indices, indices[:, -1:].expand(-1, pad_count)],
                    dim=1,
                )
                valid = torch.cat(
                    [valid, torch.zeros(bsz, pad_count, dtype=torch.bool, device=indices.device)],
                    dim=1,
                )
            selections.append(indices)
            valid_masks.append(valid)
        return torch.stack(selections, dim=1), torch.stack(valid_masks, dim=1)

    def forward(self, global_frame_tokens: Tensor, patch_tokens: Tensor,
                frame_logits: Tensor) -> dict[str, Tensor]:
        bsz, frames, _ = global_frame_tokens.shape
        if patch_tokens.shape[:2] != (bsz, frames):
            raise ValueError("patch tokens and global frame tokens must align")
        selected, selected_valid = self._selected_indices(frame_logits)
        labels, selected_frames = selected.shape[1:]
        batch_index = torch.arange(bsz, device=selected.device).reshape(
            bsz, 1, 1
        ).expand_as(selected)
        selected_global = global_frame_tokens[batch_index, selected]
        selected_patches = patch_tokens[batch_index, selected]
        selected_scores = frame_logits[batch_index, selected]
        dynamic_queries = self.global_query(selected_global).reshape(
            bsz, labels, selected_frames, self.queries_per_class, self.attention_dim
        )
        queries = self.query_norm(
            self.base_queries.reshape(
                1, labels, 1, self.queries_per_class, self.attention_dim
            ) + self.query_scale * dynamic_queries
        )
        keys = self.patch_key(self.patch_norm(selected_patches))
        values = self.patch_value(selected_patches)
        attention = (
            torch.einsum("bcsqa,bcsna->bcsqn", queries, keys)
            / math.sqrt(float(self.attention_dim))
        ).softmax(dim=-1)
        region_queries = torch.einsum("bcsqn,bcsne->bcsqe", attention, values)
        query_weights = self.query_pool_logits.softmax(dim=-1).reshape(
            1, labels, 1, self.queries_per_class, 1
        )
        region = (region_queries * query_weights).sum(dim=3)
        relative_time = selected.to(selected_scores.dtype) / max(frames - 1, 1)
        score_features = torch.cat(
            [selected_scores, relative_time.unsqueeze(-1)], dim=-1
        )
        frame_evidence = self.frame_fusion(torch.cat([
            region, self.global_proj(selected_global),
            self.frame_score_proj(score_features),
        ], dim=-1))
        first = torch.zeros_like(frame_evidence)
        first[:, :, 1:] = frame_evidence[:, :, 1:] - frame_evidence[:, :, :-1]
        second = torch.zeros_like(frame_evidence)
        if selected_frames > 2:
            second[:, :, 2:] = (
                frame_evidence[:, :, 2:] - 2.0 * frame_evidence[:, :, 1:-1]
                + frame_evidence[:, :, :-2]
            )
        temporal = self.temporal_mlp(
            torch.cat([frame_evidence, first, second], dim=-1)
        )
        temporal_scores = self.temporal_score(temporal).squeeze(-1)
        temporal_scores = temporal_scores.masked_fill(
            ~selected_valid, torch.finfo(temporal_scores.dtype).min
        )
        temporal_weights = temporal_scores.softmax(dim=2).unsqueeze(-1)
        summary = (temporal * temporal_weights).sum(dim=2)
        raw_correction = (
            torch.einsum("bce,ce->bc", summary, self.correction_weight)
            + self.correction_bias.reshape(1, -1)
        )
        correction = bounded_asymmetric_residual(
            raw_correction, positive_max=self.positive_max,
            negative_max=self.negative_max,
        )
        probabilities = attention.clamp_min(1e-8)
        entropy = -(probabilities * probabilities.log()).sum(dim=-1)
        result = {
            "correction": correction,
            "raw_correction": raw_correction,
            "selected_indices": selected,
            "selected_valid": selected_valid,
            "temporal_weights": temporal_weights.squeeze(-1),
            "attention_entropy": entropy / max(
                math.log(float(patch_tokens.shape[2])), 1e-6
            ),
        }
        if self.return_attention_maps:
            result["attention_maps"] = attention
        return result


class GlobalLocalEvidenceTemporalHead(nn.Module):
    """Build a new temporal representation from global and local evidence tokens.

    Unlike the legacy class-evidence head, this module does not correct frozen
    logits.  Dense class-specific patch evidence is fused with every global
    frame token before a transferred temporal encoder makes the event decision.
    """

    def __init__(
        self,
        *,
        patch_dim: int,
        hidden_dim: int,
        attention_dim: int,
        num_labels: int,
        queries_per_class: int,
        temporal: nn.Module,
        classifier: nn.Module,
        frame_event_head: nn.Module,
        dropout: float,
        return_attention_maps: bool = False,
    ):
        super().__init__()
        self.num_labels = int(num_labels)
        self.queries_per_class = max(int(queries_per_class), 1)
        self.attention_dim = int(attention_dim)
        self.return_attention_maps = bool(return_attention_maps)
        self.base_queries = nn.Parameter(
            torch.empty(
                self.num_labels, self.queries_per_class, self.attention_dim
            )
        )
        self.global_query = nn.Linear(
            hidden_dim, self.num_labels * self.queries_per_class * self.attention_dim
        )
        self.dynamic_query_scale = nn.Parameter(torch.tensor(0.1))
        self.query_norm = nn.LayerNorm(self.attention_dim)
        self.patch_norm = nn.LayerNorm(patch_dim)
        self.patch_key = nn.Linear(patch_dim, self.attention_dim, bias=False)
        self.appearance_value = nn.Linear(patch_dim, hidden_dim, bias=False)
        # A local event is not just an object/region. The signed central
        # difference describes direction while the absolute neighbour change
        # describes motion magnitude. Keeping both prevents a static penalty
        # area or touchline from being sufficient evidence.
        self.motion_value = nn.Linear(patch_dim * 2, hidden_dim, bias=False)
        self.relation_fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2 * self.queries_per_class),
            nn.Linear(hidden_dim * 2 * self.queries_per_class, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.class_embeddings = nn.Parameter(
            torch.zeros(self.num_labels, hidden_dim)
        )
        # Every class may explicitly choose "no decisive local action" instead
        # of forcing a field edge or penalty area to explain every frame.
        self.no_evidence_tokens = nn.Parameter(
            torch.zeros(self.num_labels, hidden_dim)
        )
        self.presence_norm = nn.LayerNorm(hidden_dim)
        self.presence_weight = nn.Parameter(
            torch.empty(self.num_labels, hidden_dim)
        )
        self.presence_bias = nn.Parameter(torch.zeros(self.num_labels))
        # Concatenation is projected into a new frame representation.  The
        # global block starts as identity and local blocks as zero so the
        # transferred temporal head begins at the strong baseline without a
        # logit-level skip connection.  Local blocks receive gradients on the
        # first update; patch queries receive them from the following update.
        self.fusion_proj = nn.Linear(
            hidden_dim * (self.num_labels + 1), hidden_dim
        )
        self.local_dropout = nn.Dropout(dropout)
        self.temporal = copy.deepcopy(temporal)
        self.classifier = copy.deepcopy(classifier)
        self.frame_event_head = copy.deepcopy(frame_event_head)

        nn.init.trunc_normal_(self.base_queries, std=0.02)
        nn.init.zeros_(self.global_query.weight)
        nn.init.zeros_(self.global_query.bias)
        nn.init.trunc_normal_(self.class_embeddings, std=0.02)
        nn.init.trunc_normal_(self.no_evidence_tokens, std=0.02)
        nn.init.trunc_normal_(self.presence_weight, std=0.02)
        nn.init.zeros_(self.fusion_proj.weight)
        nn.init.zeros_(self.fusion_proj.bias)
        with torch.no_grad():
            self.fusion_proj.weight[:, :hidden_dim].copy_(
                torch.eye(hidden_dim)
            )

    def initialize_from_primary(
        self,
        temporal: nn.Module,
        classifier: nn.Module,
        frame_event_head: nn.Module,
    ) -> None:
        self.temporal.load_state_dict(temporal.state_dict())
        self.classifier.load_state_dict(classifier.state_dict())
        self.frame_event_head.load_state_dict(frame_event_head.state_dict())

    def query_diversity_loss(self) -> Tensor:
        if self.queries_per_class <= 1:
            return self.base_queries.new_zeros(())
        queries = F.normalize(self.base_queries, dim=-1)
        similarity = torch.einsum("cqa,cka->cqk", queries, queries)
        eye = torch.eye(
            self.queries_per_class,
            device=similarity.device,
            dtype=similarity.dtype,
        ).unsqueeze(0)
        denominator = float(
            self.num_labels
            * self.queries_per_class
            * (self.queries_per_class - 1)
        )
        return (similarity * (1.0 - eye)).square().sum() / max(
            denominator, 1.0
        )

    def forward(
        self,
        global_frame_tokens: Tensor,
        patch_tokens: Tensor,
    ) -> dict[str, Tensor]:
        bsz, frames, hidden_dim = global_frame_tokens.shape
        if patch_tokens.shape[:2] != (bsz, frames):
            raise ValueError("patch tokens and global frame tokens must align")
        dynamic_queries = self.global_query(global_frame_tokens).reshape(
            bsz,
            frames,
            self.num_labels,
            self.queries_per_class,
            self.attention_dim,
        )
        queries = self.query_norm(
            self.base_queries.reshape(
                1,
                1,
                self.num_labels,
                self.queries_per_class,
                self.attention_dim,
            )
            + self.dynamic_query_scale * dynamic_queries
        )
        normalized_patches = self.patch_norm(patch_tokens)
        keys = self.patch_key(normalized_patches)
        appearance_values = self.appearance_value(normalized_patches)
        previous = torch.cat(
            [normalized_patches[:, :1], normalized_patches[:, :-1]], dim=1
        )
        following = torch.cat(
            [normalized_patches[:, 1:], normalized_patches[:, -1:]], dim=1
        )
        signed_motion = following - previous
        motion_magnitude = 0.5 * (
            (normalized_patches - previous).abs()
            + (following - normalized_patches).abs()
        )
        motion_values = self.motion_value(
            torch.cat([signed_motion, motion_magnitude], dim=-1)
        )
        attention = (
            torch.einsum("btcqa,btna->btcqn", queries, keys)
            / math.sqrt(float(self.attention_dim))
        ).softmax(dim=-1)
        appearance_queries = torch.einsum(
            "btcqn,btnh->bt c q h".replace(" ", ""),
            attention,
            appearance_values,
        )
        motion_queries = torch.einsum(
            "btcqn,btnh->bt c q h".replace(" ", ""),
            attention,
            motion_values,
        )
        # Preserve all query slots until relation fusion. Premature averaging
        # discards actor/ball/target distinctions needed to separate a shot
        # from a generic attack near the goal.
        relation_input = torch.cat(
            [appearance_queries, motion_queries], dim=-1
        ).flatten(start_dim=3)
        raw_local_evidence = self.relation_fusion(relation_input)
        raw_local_evidence = raw_local_evidence + self.class_embeddings.reshape(
            1, 1, self.num_labels, hidden_dim
        )
        normalized_evidence = self.presence_norm(raw_local_evidence)
        presence_logits = torch.einsum(
            "btch,ch->btc", normalized_evidence, self.presence_weight
        ) + self.presence_bias.reshape(1, 1, -1)
        presence = torch.sigmoid(presence_logits).unsqueeze(-1)
        no_evidence = self.no_evidence_tokens.reshape(
            1, 1, self.num_labels, hidden_dim
        ).expand(bsz, frames, -1, -1)
        local_evidence = (
            presence * raw_local_evidence + (1.0 - presence) * no_evidence
        )
        local_evidence = self.local_dropout(local_evidence)
        fusion_input = torch.cat(
            [global_frame_tokens, local_evidence.flatten(start_dim=2)], dim=-1
        )
        fused_frame_tokens = self.fusion_proj(fusion_input)
        temporal_representation = self.temporal(fused_frame_tokens)
        logits = self.classifier(temporal_representation)
        frame_event_logits = self.frame_event_head(fused_frame_tokens)

        # Differentiable counterfactual: keep the global video unchanged but
        # replace all local evidence by no-evidence. This adds only a second
        # lightweight temporal pass; DINO is not recomputed.
        no_evidence_input = torch.cat(
            [global_frame_tokens, no_evidence.flatten(start_dim=2)], dim=-1
        )
        no_evidence_frame_tokens = self.fusion_proj(no_evidence_input)
        no_evidence_representation = self.temporal(no_evidence_frame_tokens)
        no_evidence_logits = self.classifier(no_evidence_representation)
        probabilities = attention.clamp_min(1e-8)
        entropy = -(probabilities * probabilities.log()).sum(dim=-1)
        result = {
            "logits": logits,
            "frame_event_logits": frame_event_logits,
            "local_frame_event_logits": presence_logits,
            "fused_frame_tokens": fused_frame_tokens,
            "local_evidence_tokens": local_evidence,
            "raw_local_evidence_tokens": raw_local_evidence,
            "local_evidence_presence": presence.squeeze(-1),
            "no_evidence_logits": no_evidence_logits,
            "temporal_representation": temporal_representation,
            "attention_entropy": entropy
            / max(math.log(float(patch_tokens.shape[2])), 1e-6),
            "query_diversity_loss": self.query_diversity_loss(),
        }
        if self.return_attention_maps:
            result["attention_maps"] = attention
        return result



class IndependentPreprojectionEvidenceHead(nn.Module):
    """Class-conditioned patch routing before the shared 6144->hidden bottleneck.

    Multi-layer CLS context stays structured, final-layer patches are pooled by
    independent event/query scorers with an explicit no-evidence option, and
    every event owns its frame adapter, temporal encoder, dense head and clip
    head.  The DINO trunk is shared; event representation spaces are not.
    """

    def __init__(
        self,
        *,
        patch_dim: int,
        raw_feature_dim: int,
        num_feature_layers: int,
        class_dim: int,
        num_labels: int,
        queries_per_class: int,
        temporal_layers: int,
        temporal_heads: int,
        dropout: float,
        max_frames: int,
        bottleneck_frames: int,
        anti_erasure_no_evidence: bool = False,
        causal_local_counterfactual: bool = False,
        set_piece_index: int = -1,
        set_piece_subtype_count: int = 0,
        save_shot_conditioning: bool = False,
        save_shot_condition_hidden_dim: int = 256,
        save_shot_condition_window_frames: int = 8,
        save_shot_condition_shot_index: int = 0,
        save_shot_condition_save_index: int = 1,
        evidence_feature_dim: int = 0,
        evidence_gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_dim = int(patch_dim)
        self.num_feature_layers = int(num_feature_layers)
        self.num_labels = int(num_labels)
        self.queries_per_class = max(int(queries_per_class), 1)
        self.anti_erasure_no_evidence = bool(anti_erasure_no_evidence)
        self.causal_local_counterfactual = bool(causal_local_counterfactual)
        self.set_piece_index = int(set_piece_index)
        self.save_shot_condition_shot_index = int(save_shot_condition_shot_index)
        self.save_shot_condition_save_index = int(save_shot_condition_save_index)
        self.save_shot_condition_window_frames = max(
            int(save_shot_condition_window_frames), 1
        )
        expected = self.patch_dim * 2 * self.num_feature_layers
        if int(raw_feature_dim) != expected:
            raise ValueError(
                f"independent_preprojection expects raw feature dim={expected}, "
                f"got {raw_feature_dim}"
            )
        class_dim = max(int(class_dim), 64)
        if class_dim % max(int(temporal_heads), 1) != 0:
            raise ValueError("class evidence hidden_dim must be divisible by temporal_heads")
        global_slot_dim = max(class_dim // 2, 64)
        evidence_dim = max(class_dim // 2, 64)
        self.global_layer_projections = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.patch_dim),
                nn.Linear(self.patch_dim, global_slot_dim),
                nn.GELU(),
            )
            for _ in range(self.num_feature_layers)
        ])
        self.patch_norm = nn.LayerNorm(self.patch_dim)
        self.patch_scores = nn.Linear(
            self.patch_dim, self.num_labels * self.queries_per_class
        )
        self.patch_values = nn.Linear(self.patch_dim, evidence_dim, bias=False)
        # One no-evidence slot competes with O(10^3) spatial patches.  A zero
        # logit would give it only ~1/N mass at initialization and make the
        # advertised fallback effectively unreachable.  Start at the patch
        # multiplicity prior; training can then move it class/query-wise.
        self.no_evidence_logits = nn.Parameter(
            torch.full(
                (self.num_labels, self.queries_per_class),
                math.log(1024.0),
            )
        )
        self.no_evidence_values = nn.Parameter(
            torch.zeros(
                self.num_labels, self.queries_per_class, evidence_dim
            )
        )
        self.no_evidence_gate_weight: nn.Parameter | None = None
        self.no_evidence_gate_bias: nn.Parameter | None = None
        if self.anti_erasure_no_evidence:
            # Monotone saliency gate (user design, 2026-08-26): 4 parameters.
            #   saliency = log1p([score_max_gap, score_topk_gap, score_std])
            #   null_logit = bias - sum(softplus(w_i) * saliency_i)
            #   null_mass = sigmoid(null_logit)
            # softplus keeps every saliency coefficient non-negative, so
            # stronger saliency can only lower null mass: the gate preserves
            # absolute saliency strength, cannot learn to erase appearance,
            # and pushes evidence-free background to high null via bias.
            # Init zero weight / zero bias keeps the warm-started behaviour
            # unchanged at start (mass=0.5); a fresh optimizer group
            # (class_evidence_no_evidence_gate) gives it a real LR.
            self.no_evidence_gate_weight = nn.Parameter(torch.zeros(3))
            # bias +1.5: with realistic saliency scales (sum of log1p stats
            # ~3.3), zero bias puts null_logit at ~-2.3 (mass ~0.09) inside
            # the saturated sigmoid region where gradients are compressed
            # ~12x and the gate never learns. +1.5 lands at null_logit ~-0.8
            # (mass ~0.31), the linear working region, with the semantics
            # "evidence-free frames start moderately high on null".
            self.no_evidence_gate_bias = nn.Parameter(
                torch.full((1,), 1.5)
            )
            # E1.3.2 (2026-08-26): the frame-level null-mass supervision was
            # removed (class_evidence_no_evidence_loss_weight: 0.0) because
            # "not an event frame" != "no local evidence" in football; the
            # E1.3 gate run proved the gate learns a compromise constant.
            # The gate is now frozen as a pure diagnostic (mass/saliency
            # outputs still logged) - keeping it trainable would trip DDP's
            # unused-parameter reduction.
            self.no_evidence_gate_weight.requires_grad_(False)
            self.no_evidence_gate_bias.requires_grad_(False)
            # Retain legacy tensors for checkpoint shape compatibility, but do
            # not let their inherited global prior/value erase patch content.
            self.no_evidence_logits.requires_grad_(False)
            self.no_evidence_values.requires_grad_(False)
        global_dim = global_slot_dim * self.num_feature_layers
        evidence_flat_dim = evidence_dim * self.queries_per_class
        adapter_input_dim = global_dim + evidence_flat_dim * 3
        self.class_adapters = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(adapter_input_dim),
                nn.Linear(adapter_input_dim, class_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(class_dim * 2, class_dim),
                nn.LayerNorm(class_dim),
            )
            for _ in range(self.num_labels)
        ])
        self.temporals = nn.ModuleList([
            TemporalStemBottleneckTransformer(
                hidden_dim=class_dim,
                num_layers=max(int(temporal_layers), 1),
                num_heads=max(int(temporal_heads), 1),
                dropout=dropout,
                max_frames=max_frames,
                stem_kernel_size=3 if index < 2 else 5,
                multiscale_dilations=(1, 2) if index < 2 else (1, 2, 4),
                bottleneck_frames=bottleneck_frames,
            )
            for index in range(self.num_labels)
        ])
        self.clip_heads = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(class_dim), nn.Linear(class_dim, 1))
            for _ in range(self.num_labels)
        ])
        self.frame_heads = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(class_dim), nn.Linear(class_dim, 1))
            for _ in range(self.num_labels)
        ])
        self.set_piece_subtype_head: nn.Module | None = None
        if int(set_piece_subtype_count) > 0:
            if not 0 <= self.set_piece_index < self.num_labels:
                raise ValueError(
                    "set-piece subtype supervision requires a valid set_piece class index"
                )
            self.set_piece_subtype_head = nn.Sequential(
                nn.LayerNorm(class_dim),
                nn.Linear(class_dim, max(class_dim // 2, 64)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(max(class_dim // 2, 64), int(set_piece_subtype_count)),
            )
        # Save-conditioned-on-Shot: shot evidence enters the save head as an
        # additive residual computed from shot's own (detached) frame tokens.
        # The last MLP layer is zero-initialized so warm-started save behavior
        # stays bitwise identical at init; the MLP itself is trainable.
        self.save_shot_condition_mlp: nn.Module | None = None
        if save_shot_conditioning:
            if not 0 <= self.save_shot_condition_shot_index < self.num_labels:
                raise ValueError(
                    "save-shot conditioning requires a valid shot class index"
                )
            if not 0 <= self.save_shot_condition_save_index < self.num_labels:
                raise ValueError(
                    "save-shot conditioning requires a valid save class index"
                )
            if self.save_shot_condition_shot_index >= self.save_shot_condition_save_index:
                raise ValueError(
                    "save-shot conditioning requires the shot class index to precede "
                    "the save class index in the per-class loop"
                )
            condition_hidden = max(int(save_shot_condition_hidden_dim), 16)
            condition_dim = class_dim + 2
            self.save_shot_condition_mlp = nn.Sequential(
                nn.LayerNorm(condition_dim),
                nn.Linear(condition_dim, condition_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(condition_hidden, class_dim),
            )
            nn.init.zeros_(self.save_shot_condition_mlp[-1].weight)
            nn.init.zeros_(self.save_shot_condition_mlp[-1].bias)
        # Geometric evidence injection (E4): per-frame ball/goal/keeper/crowd
        # features are projected per class and added to the class frame tokens
        # through a zero-initialized tanh gate, so a warm-started checkpoint is
        # bitwise identical at init and the model can learn how much detector
        # evidence each event class actually uses.
        self.evidence_feature_dim = max(int(evidence_feature_dim), 0)
        self.evidence_projs: nn.ModuleList | None = None
        self.evidence_gates: nn.Parameter | None = None
        if self.evidence_feature_dim > 0:
            self.evidence_projs = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(self.evidence_feature_dim),
                    nn.Linear(self.evidence_feature_dim, class_dim),
                )
                for _ in range(self.num_labels)
            ])
            # gate_init 0.0 = strict identity at warm start, but with the small
            # head LR the gate effectively never opens within a 2-epoch budget
            # (E2.0 epoch-1 measured |gate| < 1e-3).  A small positive init
            # (e.g. 0.2 -> tanh ~0.2) lets evidence actually flow while the
            # classifier learns how much to trust it.
            self.evidence_gates = nn.Parameter(
                torch.full((self.num_labels,), float(evidence_gate_init))
            )
        nn.init.trunc_normal_(self.no_evidence_values, std=0.02)

    def forward(
        self, raw_features: Tensor, patch_tokens: Tensor,
        evidence_features: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if raw_features.ndim != 3 or patch_tokens.ndim != 4:
            raise ValueError(
                "independent preprojection expects raw [B,T,F] and patches [B,T,N,D]"
            )
        if self.evidence_projs is not None:
            expected = (raw_features.shape[0], raw_features.shape[1], self.evidence_feature_dim)
            if evidence_features is None or tuple(evidence_features.shape) != expected:
                raise ValueError(
                    f"evidence_features must have shape {expected}, got "
                    f"{None if evidence_features is None else tuple(evidence_features.shape)}"
                )
        bsz, frames, _ = raw_features.shape
        if patch_tokens.shape[:2] != (bsz, frames):
            raise ValueError("raw features and patch tokens must align")
        structured = raw_features.reshape(
            bsz, frames, self.num_feature_layers, 2, self.patch_dim
        )
        cls_layers = structured[:, :, :, 0]
        global_context = torch.cat(
            [
                projection(cls_layers[:, :, layer_index])
                for layer_index, projection in enumerate(
                    self.global_layer_projections
                )
            ],
            dim=-1,
        )

        normalized_patches = self.patch_norm(patch_tokens)
        scores = self.patch_scores(normalized_patches).reshape(
            bsz, frames, patch_tokens.shape[2], self.num_labels, self.queries_per_class
        ).permute(0, 1, 3, 4, 2)
        patch_values = self.patch_values(normalized_patches)
        no_evidence_content_logits: Tensor | None = None
        if self.anti_erasure_no_evidence:
            if self.no_evidence_gate_weight is None:
                raise RuntimeError("anti-erasure no-evidence gate is missing")
            score_mean = scores.mean(dim=-1)
            score_max_gap = scores.amax(dim=-1) - score_mean
            topk = min(4, scores.shape[-1])
            score_topk_gap = scores.topk(k=topk, dim=-1).values.mean(dim=-1) - score_mean
            score_std = scores.float().std(dim=-1, unbiased=False).to(scores.dtype)
            saliency_statistics = torch.stack(
                [score_max_gap, score_topk_gap, score_std], dim=-1
            )
            saliency = torch.log1p(saliency_statistics.clamp_min(0.0))
            no_evidence_content_logits = self.no_evidence_gate_bias - (
                F.softplus(self.no_evidence_gate_weight) * saliency
            ).sum(dim=-1)
            no_evidence_mass = torch.sigmoid(no_evidence_content_logits)
            no_evidence_saliency = saliency.mean(dim=-2)
            # Conditional normalization over real patches prevents the null slot
            # from erasing visual evidence. The gate remains a supervised quality
            # diagnostic and does not scale or replace appearance.
            patch_attention = scores.softmax(dim=-1)
            appearance = torch.einsum(
                "btcqn,btne->btcqe", patch_attention, patch_values
            )
            probabilities = torch.cat(
                [
                    # stop-grad: the gate must be driven only by the
                    # supervised no-evidence loss. Main-task gradients flowing
                    # back through the routed attention push every frame's
                    # null mass toward 0 (content retention is the safest way
                    # to reduce the main loss), which hijacked the gate and
                    # pinned background null at ~0.1 regardless of hinge/BCE.
                    (1.0 - no_evidence_mass.detach()).unsqueeze(-1) * patch_attention,
                    no_evidence_mass.detach().unsqueeze(-1),
                ],
                dim=-1,
            )
        else:
            no_scores = self.no_evidence_logits.reshape(
                1, 1, self.num_labels, self.queries_per_class, 1
            ).expand(bsz, frames, -1, -1, -1)
            attention = torch.cat([scores, no_scores], dim=-1).softmax(dim=-1)
            patch_attention = attention[..., :-1]
            no_evidence_mass = attention[..., -1]
            appearance = torch.einsum(
                "btcqn,btne->btcqe", patch_attention, patch_values
            )
            appearance = appearance + no_evidence_mass.unsqueeze(-1) * (
                self.no_evidence_values.reshape(
                    1, 1, self.num_labels, self.queries_per_class, -1
                )
            )
            probabilities = attention
        routed_patch_appearance = appearance
        appearance = appearance.flatten(start_dim=3)
        previous = torch.cat([appearance[:, :1], appearance[:, :-1]], dim=1)
        following = torch.cat([appearance[:, 1:], appearance[:, -1:]], dim=1)
        signed_motion = following - previous
        motion_magnitude = 0.5 * (
            (appearance - previous).abs() + (following - appearance).abs()
        )

        clip_logits: list[Tensor] = []
        counterfactual_clip_logits: list[Tensor] = []
        causal_full_clip_logits: list[Tensor] = []
        frame_logits: list[Tensor] = []
        class_frame_tokens: list[Tensor] = []
        class_clip_tokens: list[Tensor] = []
        evidence_gate_values = (
            torch.tanh(self.evidence_gates)
            if self.evidence_gates is not None
            else None
        )
        for class_index in range(self.num_labels):
            adapter_input = torch.cat(
                [
                    global_context,
                    appearance[:, :, class_index],
                    signed_motion[:, :, class_index],
                    motion_magnitude[:, :, class_index],
                ],
                dim=-1,
            )
            event_frames = self.class_adapters[class_index](adapter_input)
            if self.evidence_projs is not None:
                if evidence_gate_values is None:
                    raise RuntimeError("evidence projections require evidence gates")
                gate = evidence_gate_values[class_index]
                event_frames = event_frames + gate * self.evidence_projs[
                    class_index
                ](evidence_features)
            if (
                self.save_shot_condition_mlp is not None
                and class_index == self.save_shot_condition_save_index
            ):
                # Shot evidence conditions the save head. Stop-grad keeps both
                # the shot tokens and shot classifier untouched by save loss.
                # Each save frame only sees the trailing shot window, avoiding
                # leakage from a future shot elsewhere in the 10-second clip.
                shot_frames = class_frame_tokens[
                    self.save_shot_condition_shot_index
                ].detach()
                shot_scores = self.frame_heads[self.save_shot_condition_shot_index](
                    shot_frames
                ).squeeze(-1).detach()
                window = min(self.save_shot_condition_window_frames, frames)
                padded_shot = F.pad(
                    shot_scores, (window - 1, 0), value=float("-inf")
                )
                trailing_shot = padded_shot.unfold(-1, window, 1)
                # Tensor.max(dim) is an amax alias (single value); torch.max
                # returns (values, indices).
                max_shot, peak_offset = torch.max(trailing_shot, dim=-1)
                time_since_peak = (window - 1 - peak_offset).to(
                    shot_scores.dtype
                ) / max(window - 1, 1)
                condition_input = torch.cat(
                    [
                        event_frames,
                        max_shot.unsqueeze(-1),
                        time_since_peak.unsqueeze(-1),
                    ],
                    dim=-1,
                )
                event_frames = event_frames + self.save_shot_condition_mlp(
                    condition_input
                )
            temporal = self.temporals[class_index](event_frames)
            dense = temporal["frame_tokens"]
            clip = temporal["clip_token"]
            class_frame_tokens.append(dense)
            class_clip_tokens.append(clip)
            frame_logits.append(
                self.frame_heads[class_index](dense).squeeze(-1)
            )
            clip_logits.append(
                self.clip_heads[class_index](clip).squeeze(-1)
            )
            if self.causal_local_counterfactual and self.training:
                # Causal-only path: freeze the adapter/temporal/classifier
                # parameters and detach global context. Consequently this loss
                # can update only sample-specific patch routing/values; it
                # cannot be satisfied by shifting the global representation or
                # classifier bias. Eval mode removes dropout noise between the
                # full-local and zero-local counterfactual passes.
                causal_modules = (
                    self.class_adapters[class_index],
                    self.temporals[class_index],
                    self.clip_heads[class_index],
                )
                training_states = [module.training for module in causal_modules]
                for module in causal_modules:
                    module.eval()
                try:
                    def frozen_call(module: nn.Module, value: Tensor) -> Any:
                        detached_state = {
                            name: parameter.detach()
                            for name, parameter in module.named_parameters()
                        }
                        detached_state.update({
                            name: buffer.detach()
                            for name, buffer in module.named_buffers()
                        })
                        return torch.func.functional_call(
                            module, detached_state, (value,), strict=True
                        )

                    causal_input = torch.cat(
                        [
                            global_context.detach(),
                            appearance[:, :, class_index],
                            signed_motion[:, :, class_index],
                            motion_magnitude[:, :, class_index],
                        ],
                        dim=-1,
                    )
                    causal_frames = frozen_call(
                        causal_modules[0], causal_input
                    )
                    causal_temporal = frozen_call(
                        causal_modules[1], causal_frames
                    )
                    causal_full_clip_logits.append(
                        frozen_call(
                            causal_modules[2], causal_temporal["clip_token"]
                        ).squeeze(-1)
                    )
                    with torch.no_grad():
                        zero_local = torch.zeros_like(
                            appearance[:, :, class_index]
                        )
                        counterfactual_input = torch.cat(
                            [
                                global_context.detach(),
                                zero_local,
                                zero_local,
                                zero_local,
                            ],
                            dim=-1,
                        )
                        counterfactual_frames = frozen_call(
                            causal_modules[0], counterfactual_input
                        )
                        counterfactual_temporal = frozen_call(
                            causal_modules[1], counterfactual_frames
                        )
                        counterfactual_clip_logits.append(
                            frozen_call(
                                causal_modules[2],
                                counterfactual_temporal["clip_token"],
                            ).squeeze(-1)
                        )
                finally:
                    for module, was_training in zip(
                        causal_modules, training_states
                    ):
                        module.train(was_training)
        probabilities = probabilities.clamp_min(1e-8)
        entropy = -(probabilities * probabilities.log()).sum(dim=-1)
        class_clip_tensor = torch.stack(class_clip_tokens, dim=1)
        result = {
            "logits": torch.stack(clip_logits, dim=-1),
            "frame_event_logits": torch.stack(frame_logits, dim=-1),
            "class_frame_tokens": torch.stack(class_frame_tokens, dim=2),
            "class_clip_tokens": class_clip_tensor,
            "no_evidence_mass": no_evidence_mass.mean(dim=-1),
            "attention_entropy": entropy
            / max(math.log(float(patch_tokens.shape[2] + 1)), 1e-6),
            "attention_maps": patch_attention,
            "patch_appearance": routed_patch_appearance,
        }
        if counterfactual_clip_logits:
            result["counterfactual_logits"] = torch.stack(
                counterfactual_clip_logits, dim=-1
            )
            result["causal_full_logits"] = torch.stack(
                causal_full_clip_logits, dim=-1
            )
        if no_evidence_content_logits is not None:
            result["no_evidence_content_logits"] = no_evidence_content_logits
        if no_evidence_content_logits is not None:
            result["no_evidence_saliency"] = no_evidence_saliency
        if evidence_gate_values is not None:
            # Expose the tiny gate tensor so silent zero-gate runs are visible
            # in both single-process and DDP health logs.
            result["external_evidence_gate_values"] = evidence_gate_values
        if self.set_piece_subtype_head is not None:
            result["set_piece_subtype_logits"] = self.set_piece_subtype_head(
                class_clip_tensor[:, self.set_piece_index]
            )
        return result


class ClassSpecificResponseCurveHead(nn.Module):
    """Per-class temporal response heads for dense action-spotting curves.

    The legacy dense-response v1 path reused the shared frame_event_head for all
    labels, then pooled the frame logits into clip logits.  That is compact, but
    it couples shot/save/set_piece evidence too tightly: a strong set-piece
    boundary cue can shape the same projection space used by shot/save.  This
    head keeps the input frame tokens shared while giving each event class its
    own small bottleneck and local temporal filter.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_labels: int,
        curve_hidden_dim: int = 128,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_labels = int(num_labels)
        self.norm = nn.LayerNorm(hidden_dim)
        bottleneck = max(int(curve_hidden_dim), 16)
        kernel = max(int(kernel_size), 1)
        if kernel % 2 == 0:
            kernel += 1
        padding = kernel // 2
        self.adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, bottleneck),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(bottleneck, bottleneck),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for _ in range(self.num_labels)
            ]
        )
        self.temporal_filters = nn.ModuleList(
            [
                nn.Conv1d(
                    bottleneck,
                    bottleneck,
                    kernel_size=kernel,
                    padding=padding,
                )
                for _ in range(self.num_labels)
            ]
        )
        self.outputs = nn.ModuleList(
            [nn.Linear(bottleneck, 1) for _ in range(self.num_labels)]
        )
        for conv in self.temporal_filters:
            nn.init.kaiming_uniform_(conv.weight, a=math.sqrt(5))
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

    def forward(self, frame_tokens: Tensor) -> Tensor:
        if frame_tokens.ndim != 3:
            raise ValueError(
                "ClassSpecificResponseCurveHead expects [batch, frames, hidden]"
            )
        tokens = self.norm(frame_tokens)
        logits: list[Tensor] = []
        for adapter, temporal_filter, output in zip(
            self.adapters, self.temporal_filters, self.outputs
        ):
            class_tokens = adapter(tokens)
            temporal_delta = temporal_filter(class_tokens.transpose(1, 2)).transpose(
                1, 2
            )
            class_tokens = class_tokens + F.gelu(temporal_delta)
            logits.append(output(class_tokens).squeeze(-1))
        return torch.stack(logits, dim=-1)



class VideoEventClassifier(nn.Module):
    def __init__(
        self,
        *,
        backbone: nn.Module | None,
        frame_feature_dim: int,
        local_backbone: nn.Module | None = None,
        hidden_dim: int,
        num_labels: int,
        fusion: str,
        num_layers: int,
        num_heads: int,
        dropout: float,
        max_frames: int,
        view_fusion: str = "single",
        roi_meta_dim: int = ROI_META_DIM,
        roi_count: int = 1,
        roi_verifier_min_confidence: float = 0.6,
        roi_verifier_class_indices: Sequence[int] = (),
        view_fusion_layers: int = 2,
        view_fusion_heads: int = 8,
        view_fusion_positive_delta: float = 0.5,
        view_fusion_negative_delta: float = 2.0,
        view_fusion_direct_delta_gate_init: float = 0.05,
        view_fusion_residual_gate_init: float = 0.0,
        view_fusion_positive_delta_per_class: Any | None = None,
        view_fusion_negative_delta_per_class: Any | None = None,
        event_topk: int = 12,
        context_frames: int = 4,
        event_topk_strategy: str = "shared_max",
        event_topk_per_class: int = 4,
        event_topk_class_indices: Sequence[int] = (0, 1),
        event_topk_gradient: str = "detached",
        event_topk_temperature: float = 1.0,
        event_topk_gradient_scale: float = 1.0,
        event_anchor_topk_per_class: Sequence[int] = (2, 2, 1),
        event_anchor_class_indices: Sequence[int] = (0, 1, 2),
        event_anchor_offsets: Sequence[int] = (-2, -1, 0, 1, 2),
        event_anchor_nms_radius: int = 2,
        event_anchor_max_frames: int = 16,
        uniform_frames: int = 16,
        uniform_event_gate_init: Sequence[float] = (0.25, 0.5, 0.25),
        uniform_event_gate_mode: str = "static",
        uniform_event_gate_hidden: int = 32,
        gradient_checkpointing: bool = False,
        gradient_checkpointing_mode: str = "full_extractor",
        frame_feature_mode: str = "last_cls_patch_mean",
        backbone_frame_chunk_size: int = 0,
        frame_feature_layers: Sequence[int] = (-12, -8, -4, -1),
        frame_patch_pool: str = "mean",
        temporal_difference_enabled: bool = False,
        temporal_difference_mode: str = "delta2",
        temporal_difference_gate_init: float = 0.25,
        object_spatial_aux_enabled: bool = False,
        object_spatial_aux_temporal_layers: int = 1,
        object_spatial_aux_topk_ratio: float = 0.03,
        object_spatial_aux_temperature: float = 0.7,
        object_spatial_aux_residual_max_delta: float = 1.0,
        object_spatial_aux_representation_only: bool = False,
        ball_goal_relation_aux_enabled: bool = False,
        ball_goal_relation_aux_targets: int = 10,
        ball_goal_relation_aux_topk_candidates: int = 3,
        ball_goal_relation_aux_candidate_nms_radius: int = 2,
        ball_goal_relation_aux_grid_size: Sequence[int] = (),
        ball_goal_relation_aux_context_radii: Sequence[float] = (1.5, 4.0, 8.0),
        spatial_attention_enabled: bool = False,
        spatial_attention_dim: int = 256,
        spatial_attention_queries_per_class: int = 2,
        spatial_attention_context_layers: int = 1,
        spatial_attention_temporal_layers: int = 2,
        spatial_attention_heads: int = 8,
        spatial_attention_gate_init: float = 0.05,
        spatial_attention_dynamic_query_scale_init: float = 0.0,
        spatial_attention_actionness_query_scale_init: float = 0.0,
        spatial_attention_attention_mode: str = "softmax",
        spatial_attention_topk_ratio: float = 0.03,
        spatial_attention_residual_max_delta: float = 1.0,
        spatial_attention_mode: str = "residual",
        spatial_attention_patch_mode: str = "last_layer",
        spatial_attention_feature_layers: Sequence[int] = (-8, -4, -1),
        spatial_attention_fusion_gate_init: float = 0.15,
        spatial_attention_correction_dim: int = 128,
        spatial_attention_correction_gate_init: float = 0.25,
        spatial_attention_correction_max_delta: float = 2.0,
        spatial_attention_return_maps: bool = False,
        spatial_token_pooling_enabled: bool = False,
        spatial_token_pooling_num_queries: int = 4,
        spatial_token_pooling_attention_dim: int = 256,
        spatial_token_pooling_slot_dropout: float = 0.1,
        spatial_token_pooling_gate_init: float = 0.25,
        spatial_token_pooling_fusion_mode: str = "legacy_sequence",
        spatial_token_pooling_relation_layers: int = 1,
        spatial_token_pooling_patch_layers: Sequence[int] = (),
        spatial_token_pooling_legacy_patch_layers: bool = True,
        spatial_token_pooling_return_maps: bool = False,
        class_evidence_enabled: bool = False,
        class_evidence_mode: str = "logit_residual",
        class_evidence_attention_dim: int = 128,
        class_evidence_hidden_dim: int = 192,
        class_evidence_queries_per_class: int = 2,
        class_evidence_temporal_layers: int = 1,
        class_evidence_temporal_heads: int = 4,
        class_evidence_bottleneck_frames: int = 8,
        class_evidence_anti_erasure_no_evidence: bool = False,
        class_evidence_causal_local_counterfactual: bool = False,
        save_shot_conditioning: bool = False,
        save_shot_condition_hidden_dim: int = 256,
        save_shot_condition_window_frames: int = 8,
        set_piece_subtype_count: int = 0,
        evidence_feature_dim: int = 0,
        evidence_gate_init: float = 0.0,
        class_evidence_topk_per_class: Sequence[int] = (4, 4, 4),
        class_evidence_context_frames_per_class: Sequence[int] = (2, 2, 4),
        class_evidence_positive_max_per_class: Sequence[float] = (0.5, 0.6, 1.2),
        class_evidence_negative_max_per_class: Sequence[float] = (1.5, 1.8, 0.6),
        class_evidence_return_maps: bool = False,
        highres_glimpse_enabled: bool = False,
        highres_glimpse_candidates: int = 2,
        highres_glimpse_frames_per_candidate: int = 4,
        highres_glimpse_crop_size: int = 384,
        highres_glimpse_attention_dim: int = 128,
        highres_glimpse_min_scale: Sequence[float] = (0.24, 0.30),
        highres_glimpse_max_scale: Sequence[float] = (0.58, 0.72),
        highres_glimpse_nms_radius: int = 2,
        highres_glimpse_fusion_init: float = 0.05,
        highres_glimpse_fusion_mode: str = "joint_replace",
        highres_glimpse_residual_max_delta: float = 1.0,
        highres_glimpse_residual_gate_init: float = 0.10,
        highres_glimpse_motion_prior_weight: float = 0.0,
        highres_glimpse_attention_temperature: float = 1.0,
        videomae_temporal_gate_init: float = 0.25,
        response_curve_primary_enabled: bool = False,
        response_curve_pooling: str = "topk_lse",
        response_curve_topk: int = 3,
        response_curve_blend_weight: float = 1.0,
        response_curve_head: str = "shared_frame",
        response_curve_hidden_dim: int = 128,
        response_curve_kernel_size: int = 3,
        response_curve_dropout: float | None = None,
        temporal_stem_kernel_size: int = 3,
        temporal_stem_dilations: Sequence[int] = (1, 2, 4),
        temporal_stem_bottleneck_frames: int = 8,
    ):
        super().__init__()
        self.backbone = backbone
        self.local_backbone = local_backbone
        self.register_buffer("input_mean", IMAGENET_MEAN.clone(), persistent=False)
        self.register_buffer("input_std", IMAGENET_STD.clone(), persistent=False)
        self.backbone_has_trainable_params = False
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.gradient_checkpointing_mode = str(gradient_checkpointing_mode).strip().lower()
        if self.gradient_checkpointing_mode not in {"full_extractor", "trainable_blocks"}:
            raise ValueError(
                "model.gradient_checkpointing_mode must be full_extractor or trainable_blocks"
            )
        self.backbone_frame_chunk_size = int(backbone_frame_chunk_size)
        if self.backbone_frame_chunk_size < 0:
            raise ValueError("model.backbone_frame_chunk_size must be non-negative")
        self.frame_feature_mode = str(frame_feature_mode).strip().lower()
        if self.frame_feature_mode not in (
            "last_cls_patch_mean",
            "multi_layer_cls_patch_attn",
            "residual_multilayer_cls_patch_attn",
        ):
            raise ValueError(
                "model.frame_feature_mode must be last_cls_patch_mean, "
                "multi_layer_cls_patch_attn, or residual_multilayer_cls_patch_attn"
            )
        self.frame_feature_layers = _coerce_frame_feature_layers(frame_feature_layers)
        if not self.frame_feature_layers:
            raise ValueError("model.frame_feature_layers must not be empty")
        self.frame_patch_pool = str(frame_patch_pool).strip().lower()
        if self.frame_patch_pool not in ("mean", "attn"):
            raise ValueError("model.frame_patch_pool must be mean or attn")
        backbone_feature_dim = None
        video_backbone_feature_dim = None
        if backbone is not None:
            backbone_feature_dim = int(backbone.num_features)
            if bool(getattr(backbone, "is_video_backbone", False)):
                video_backbone_feature_dim = int(
                    getattr(backbone, "output_feature_dim", backbone_feature_dim)
                )
        elif local_backbone is not None:
            backbone_feature_dim = int(local_backbone.num_features)
            if bool(getattr(local_backbone, "is_video_backbone", False)):
                video_backbone_feature_dim = int(
                    getattr(local_backbone, "output_feature_dim", backbone_feature_dim)
                )
        if backbone_feature_dim is not None:
            if video_backbone_feature_dim is not None:
                if self.frame_feature_mode != "last_cls_patch_mean":
                    raise ValueError(
                        "VideoMAEv2 currently requires "
                        "model.frame_feature_mode=last_cls_patch_mean"
                    )
                frame_feature_dim = video_backbone_feature_dim
            elif self.frame_feature_mode == "multi_layer_cls_patch_attn":
                frame_feature_dim = backbone_feature_dim * 2 * len(self.frame_feature_layers)
            else:
                frame_feature_dim = backbone_feature_dim * 2
        if (
            self.frame_feature_mode
            in ("multi_layer_cls_patch_attn", "residual_multilayer_cls_patch_attn")
            and backbone_feature_dim is None
        ):
            raise ValueError(
                f"{self.frame_feature_mode} requires a DINO backbone"
            )
        if (
            self.frame_feature_mode
            in ("multi_layer_cls_patch_attn", "residual_multilayer_cls_patch_attn")
            and self.frame_patch_pool == "attn"
        ):
            self.frame_patch_attn = nn.Sequential(
                nn.LayerNorm(backbone_feature_dim),
                nn.Linear(backbone_feature_dim, 1),
            )
        else:
            self.frame_patch_attn = None
        self.backbone_has_trainable_params = any(
            p.requires_grad
            for module in (backbone, local_backbone)
            if module is not None
            for p in module.parameters()
        )

        self.num_labels = int(num_labels)
        self.hidden_dim = int(hidden_dim)
        self.fusion = canonical_temporal_fusion(fusion)
        self.response_curve_primary_enabled = bool(response_curve_primary_enabled)
        self.response_curve_pooling = str(response_curve_pooling).strip().lower()
        if self.response_curve_pooling not in ("topk_lse", "max"):
            raise ValueError(
                "model.response_curve_primary.pooling must be topk_lse or max"
            )
        self.response_curve_topk = max(int(response_curve_topk), 1)
        self.response_curve_blend_weight = clamp(
            float(response_curve_blend_weight), 0.0, 1.0
        )
        self.response_curve_head_mode = str(response_curve_head).strip().lower()
        if self.response_curve_head_mode in ("shared", "shared_frame_event"):
            self.response_curve_head_mode = "shared_frame"
        if self.response_curve_head_mode not in ("shared_frame", "class_specific"):
            raise ValueError(
                "model.response_curve_primary.curve_head must be shared_frame or class_specific"
            )
        self.response_curve_hidden_dim = max(int(response_curve_hidden_dim), 16)
        self.response_curve_kernel_size = max(int(response_curve_kernel_size), 1)
        self.response_curve_dropout = (
            float(dropout)
            if response_curve_dropout is None
            else float(response_curve_dropout)
        )
        self.response_curve_logit_scale = nn.Parameter(torch.ones(num_labels))
        self.response_curve_logit_bias = nn.Parameter(torch.zeros(num_labels))
        self.response_curve_head: nn.Module | None = None
        self.video_global_feature_dim = int(
            getattr(backbone, "global_feature_dim", 0)
            if backbone is not None
            else 0
        )
        frame_projection_dim = int(frame_feature_dim)
        if self.fusion == "videomae_residual_transformer":
            if self.video_global_feature_dim <= 0:
                raise ValueError(
                    "videomae_residual_transformer requires "
                    "model.videomaev2.preserve_global_feature=true"
                )
            frame_projection_dim -= self.video_global_feature_dim
            if frame_projection_dim <= 0:
                raise ValueError(
                    "VideoMAE tubelet feature dimension must be positive after "
                    "splitting the preserved global feature"
                )
        self.temporal_difference_adapter: TemporalDifferenceAdapter | None = None
        if bool(temporal_difference_enabled):
            self.temporal_difference_adapter = TemporalDifferenceAdapter(
                hidden_dim=hidden_dim,
                mode=temporal_difference_mode,
                dropout=dropout,
                gate_init=temporal_difference_gate_init,
            )
        self.frame_proj = nn.Sequential(
            nn.LayerNorm(frame_projection_dim),
            nn.Linear(frame_projection_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.base_frame_feature_dim = int(frame_projection_dim)
        self.multi_layer_feature_dim = 0
        self.multi_layer_frame_adapter: nn.Module | None = None
        self.multi_layer_patch_adapter: nn.Module | None = None
        self.multi_layer_patch_mix_logits: nn.Parameter | None = None
        if self.frame_feature_mode == "residual_multilayer_cls_patch_attn":
            if backbone_feature_dim is None:
                raise ValueError(
                    "residual_multilayer_cls_patch_attn requires a DINO backbone"
                )
            self.multi_layer_feature_dim = (
                backbone_feature_dim * 2 * len(self.frame_feature_layers)
            )
            self.multi_layer_frame_adapter = nn.Sequential(
                nn.LayerNorm(self.multi_layer_feature_dim),
                nn.Linear(self.multi_layer_feature_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            patch_bottleneck = max(hidden_dim // 2, 128)
            self.multi_layer_patch_adapter = nn.Sequential(
                nn.LayerNorm(backbone_feature_dim),
                nn.Linear(backbone_feature_dim, patch_bottleneck),
                nn.GELU(),
                nn.Linear(patch_bottleneck, backbone_feature_dim),
            )
            self.multi_layer_patch_mix_logits = nn.Parameter(
                torch.zeros(len(self.frame_feature_layers))
            )
            nn.init.zeros_(self.multi_layer_frame_adapter[-1].weight)
            nn.init.zeros_(self.multi_layer_frame_adapter[-1].bias)
            nn.init.zeros_(self.multi_layer_patch_adapter[-1].weight)
            nn.init.zeros_(self.multi_layer_patch_adapter[-1].bias)
        self.frame_event_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_labels),
        )
        if self.response_curve_head_mode == "class_specific":
            self.response_curve_head = ClassSpecificResponseCurveHead(
                hidden_dim=hidden_dim,
                num_labels=num_labels,
                curve_hidden_dim=self.response_curve_hidden_dim,
                kernel_size=self.response_curve_kernel_size,
                dropout=self.response_curve_dropout,
            )
        self.highres_glimpse_enabled = bool(highres_glimpse_enabled)
        self.highres_glimpse_candidates = max(int(highres_glimpse_candidates), 1)
        self.highres_glimpse_frames_per_candidate = max(
            int(highres_glimpse_frames_per_candidate), 1
        )
        self.highres_glimpse_crop_size = max(int(highres_glimpse_crop_size), 32)
        self.highres_glimpse_nms_radius = max(int(highres_glimpse_nms_radius), 0)
        self.highres_glimpse_fusion_mode = str(
            highres_glimpse_fusion_mode
        ).strip().lower()
        if self.highres_glimpse_fusion_mode not in {
            "joint_replace", "bounded_residual"
        }:
            raise ValueError(
                "highres glimpse fusion_mode must be joint_replace or bounded_residual"
            )
        self.highres_glimpse_residual_max_delta = max(
            float(highres_glimpse_residual_max_delta), 0.0
        )
        residual_gate_init = clamp(
            float(highres_glimpse_residual_gate_init), 1e-4, 1.0 - 1e-4
        )
        self.highres_glimpse_motion_prior_weight = max(
            float(highres_glimpse_motion_prior_weight), 0.0
        )
        self.highres_glimpse_attention_temperature = max(
            float(highres_glimpse_attention_temperature), 1e-3
        )
        self.highres_glimpse_min_scale = tuple(float(x) for x in highres_glimpse_min_scale)
        self.highres_glimpse_max_scale = tuple(float(x) for x in highres_glimpse_max_scale)
        if len(self.highres_glimpse_min_scale) != 2 or len(self.highres_glimpse_max_scale) != 2:
            raise ValueError("highres glimpse scales must contain [width, height]")
        self.highres_crop_query: nn.Module | None = None
        self.highres_crop_key: nn.Module | None = None
        self.highres_crop_scale: nn.Module | None = None
        self.highres_token_fusion: nn.Module | None = None
        self.highres_event_query_embedding: nn.Parameter | None = None
        self.highres_local_motion_fusion: nn.Module | None = None
        self.highres_local_attention: nn.Module | None = None
        self.highres_local_classifier: nn.Module | None = None
        self.highres_evidence_gate: nn.Module | None = None
        self.highres_no_evidence_token: nn.Parameter | None = None
        self.highres_residual_head: nn.Module | None = None
        self.highres_residual_gate_logits: nn.Parameter | None = None
        if self.highres_glimpse_enabled:
            if backbone_feature_dim is None:
                raise ValueError("highres glimpses require an uncached DINO backbone")
            attention_dim = max(int(highres_glimpse_attention_dim), 32)
            self.highres_crop_query = nn.Sequential(
                nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, attention_dim)
            )
            self.highres_crop_key = nn.Sequential(
                nn.LayerNorm(backbone_feature_dim), nn.Linear(backbone_feature_dim, attention_dim)
            )
            self.highres_crop_scale = nn.Sequential(
                nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 2)
            )
            # The transferred frame detector supplies a stable event condition.
            # This lets shot/save/set-piece learn different spatial queries while
            # keeping candidate-time selection frozen.
            self.highres_event_query_embedding = nn.Parameter(
                torch.zeros(num_labels, attention_dim)
            )
            nn.init.normal_(self.highres_event_query_embedding, std=0.02)
            # Preserve the short-range action inside every dense glimpse instead
            # of destroying it with a plain temporal mean.
            self.highres_local_motion_fusion = nn.Sequential(
                nn.LayerNorm(hidden_dim * 3),
                nn.Linear(hidden_dim * 3, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            # The local auxiliary task must not update the primary temporal/head:
            # its role is to make the crop itself discriminative.
            self.highres_local_attention = nn.Sequential(
                nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, num_labels)
            )
            self.highres_local_classifier = nn.Sequential(
                nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, num_labels)
            )
            self.highres_evidence_gate = nn.Sequential(
                nn.LayerNorm(hidden_dim * 4),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            self.highres_no_evidence_token = nn.Parameter(
                torch.zeros(1, 1, hidden_dim)
            )
            nn.init.normal_(self.highres_no_evidence_token, std=0.02)
            # Deliberately no identity/global-residual initialization.  Every
            # output token is a newly learned joint representation, constructed
            # from global context, local evidence, and their interactions.
            self.highres_token_fusion = nn.Sequential(
                nn.LayerNorm(hidden_dim * 4),
                nn.Linear(hidden_dim * 4, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            # Guard the proven global classifier instead of replacing it. The
            # native-pixel crop can only contribute a bounded class correction,
            # and zero initialization makes the first output exactly the anchor.
            self.highres_residual_head = nn.Sequential(
                nn.LayerNorm(hidden_dim * 4),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            self.highres_residual_gate_logits = nn.Parameter(
                torch.full(
                    (num_labels,),
                    math.log(residual_gate_init / (1.0 - residual_gate_init)),
                )
            )
            nn.init.zeros_(self.highres_residual_head[-1].weight)
            nn.init.zeros_(self.highres_residual_head[-1].bias)
        self.event_topk = max(int(event_topk), 0)
        self.context_frames = max(int(context_frames), 0)
        self.event_topk_strategy = str(event_topk_strategy)
        if self.event_topk_strategy not in ("shared_max", "class_union"):
            raise ValueError(
                "model.event_topk_strategy must be shared_max or class_union"
            )
        self.event_topk_per_class = max(int(event_topk_per_class), 0)
        self.event_topk_class_indices = tuple(
            int(index) for index in event_topk_class_indices
        )
        self.event_topk_gradient = str(event_topk_gradient)
        if self.event_topk_gradient not in ("detached", "straight_through"):
            raise ValueError(
                "model.event_topk_gradient must be detached or straight_through"
            )
        self.event_topk_temperature = float(event_topk_temperature)
        if self.event_topk_temperature <= 0:
            raise ValueError("model.event_topk_temperature must be positive")
        self.event_topk_gradient_scale = float(event_topk_gradient_scale)
        if self.event_topk_gradient_scale < 0:
            raise ValueError("model.event_topk_gradient_scale must be non-negative")
        anchor_counts = [0 for _ in range(self.num_labels)]
        for index, value in enumerate(event_anchor_topk_per_class):
            if index >= self.num_labels:
                break
            anchor_counts[index] = max(int(value), 0)
        self.event_anchor_topk_per_class = tuple(anchor_counts)
        self.event_anchor_class_indices = tuple(
            int(index) for index in event_anchor_class_indices
        )
        self.event_anchor_offsets = tuple(
            sorted({int(offset) for offset in event_anchor_offsets})
        )
        if not self.event_anchor_offsets:
            self.event_anchor_offsets = (0,)
        self.event_anchor_nms_radius = max(int(event_anchor_nms_radius), 0)
        self.event_anchor_max_frames = max(int(event_anchor_max_frames), 1)
        self.uniform_frames = max(int(uniform_frames), 1)
        self.temporal = self._make_temporal_module(
            self.fusion,
            hidden_dim,
            num_layers,
            num_heads,
            dropout,
            max_frames,
            num_labels,
            stem_kernel_size=int(temporal_stem_kernel_size),
            stem_dilations=tuple(int(item) for item in temporal_stem_dilations),
            stem_bottleneck_frames=int(temporal_stem_bottleneck_frames),
        )
        if self.fusion == "class_query_transformer":
            self.head = nn.Identity()
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_labels),
            )
        self.video_global_proj: nn.Module | None = None
        self.video_temporal_delta_head: nn.Module | None = None
        self.video_temporal_gate_logits: nn.Parameter | None = None
        if self.fusion == "videomae_residual_transformer":
            self.video_global_proj = nn.Sequential(
                nn.LayerNorm(self.video_global_feature_dim),
                nn.Linear(self.video_global_feature_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            delta_hidden = max(hidden_dim // 2, 64)
            self.video_temporal_delta_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, delta_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(delta_hidden, num_labels),
            )
            # Begin as an exact global-action classifier. The temporal path
            # learns only the correction needed for football event context.
            nn.init.zeros_(self.video_temporal_delta_head[-1].weight)
            nn.init.zeros_(self.video_temporal_delta_head[-1].bias)
            gate_init = min(max(float(videomae_temporal_gate_init), 1e-4), 1.0 - 1e-4)
            self.video_temporal_gate_logits = nn.Parameter(
                torch.full(
                    (num_labels,),
                    torch.logit(torch.tensor(gate_init)).item(),
                    dtype=torch.float32,
                )
            )
        if self.fusion == "uniform_event_dual_transformer":
            self.uniform_temporal = copy.deepcopy(self.temporal)
            self.uniform_head = copy.deepcopy(self.head)
            self.uniform_event_gate_mode = str(
                uniform_event_gate_mode
            ).strip().lower()
            if self.uniform_event_gate_mode not in ("static", "adaptive"):
                raise ValueError(
                    "model.uniform_event_gate_mode must be static or adaptive"
                )
            gate_init = list(uniform_event_gate_init)
            if len(gate_init) != self.num_labels:
                raise ValueError(
                    "model.uniform_event_gate_init must have one value per label"
                )
            gate_prob = torch.tensor(gate_init, dtype=torch.float32).clamp(
                1e-4, 1.0 - 1e-4
            )
            self.uniform_event_gate_logits = nn.Parameter(torch.logit(gate_prob))
            self.uniform_event_gate_adapter: nn.Module | None = None
            if self.uniform_event_gate_mode == "adaptive":
                gate_hidden = max(int(uniform_event_gate_hidden), 4)
                self.uniform_event_gate_adapter = nn.Sequential(
                    nn.LayerNorm(self.num_labels * 3),
                    nn.Linear(self.num_labels * 3, gate_hidden),
                    nn.GELU(),
                    nn.Linear(gate_hidden, self.num_labels),
                )
                nn.init.zeros_(self.uniform_event_gate_adapter[-1].weight)
                nn.init.zeros_(self.uniform_event_gate_adapter[-1].bias)
        self.view_fusion = str(view_fusion)
        if self.view_fusion not in (
            "single",
            "dual_gate",
            "dual_feature_quality",
            "dual_multi_roi_gain",
            "dual_multi_roi_memory",
            "dual_verifier",
            "dual_token_fusion",
            "dual_cross_attention",
            "dual_class_query_fusion",
            "dual_direct_class_query_fusion",
        ):
            raise ValueError(
                "model.view_fusion must be single, dual_gate, dual_feature_quality, "
                "dual_multi_roi_gain, dual_multi_roi_memory, dual_token_fusion, "
                "dual_cross_attention, "
                "dual_class_query_fusion, "
                "dual_direct_class_query_fusion, or dual_verifier"
            )
        self.dual_view = self.view_fusion != "single"
        if self.fusion == "videomae_residual_transformer" and self.dual_view:
            raise ValueError(
                "videomae_residual_transformer currently requires "
                "model.view_fusion=single"
            )
        if self.dual_view and self.fusion == "uniform_event_dual_transformer":
            raise ValueError(
                "uniform_event_dual_transformer currently supports model.view_fusion=single only"
            )
        self.object_spatial_aux_enabled = bool(object_spatial_aux_enabled)
        self.object_spatial_aux_representation_only = bool(
            object_spatial_aux_representation_only
        )
        self.object_spatial_aux: ObjectSpatialAuxHead | None = None
        if self.object_spatial_aux_enabled:
            if self.dual_view:
                raise ValueError("model.object_spatial_aux requires view_fusion=single")
            if backbone_feature_dim is None or video_backbone_feature_dim is not None:
                raise ValueError(
                    "model.object_spatial_aux requires an uncached DINO image backbone"
                )
            self.object_spatial_aux = ObjectSpatialAuxHead(
                patch_dim=backbone_feature_dim,
                hidden_dim=hidden_dim,
                num_labels=num_labels,
                temporal_layers=int(object_spatial_aux_temporal_layers),
                dropout=dropout,
                topk_ratio=float(object_spatial_aux_topk_ratio),
                temperature=float(object_spatial_aux_temperature),
                residual_max_delta=float(object_spatial_aux_residual_max_delta),
                relation_aux_enabled=bool(ball_goal_relation_aux_enabled),
                relation_targets=int(ball_goal_relation_aux_targets),
                relation_topk_candidates=int(ball_goal_relation_aux_topk_candidates),
                relation_candidate_nms_radius=int(ball_goal_relation_aux_candidate_nms_radius),
                relation_grid_size=tuple(ball_goal_relation_aux_grid_size),
                relation_context_radii=tuple(ball_goal_relation_aux_context_radii),
            )
        self.spatial_token_pooling_enabled = bool(spatial_token_pooling_enabled)
        self.spatial_token_pooling_fusion_mode = str(
            spatial_token_pooling_fusion_mode
        ).strip().lower()
        supported_spatial_token_fusions = {
            "legacy_sequence",
            "joint_token_temporal",
            "slot_only_temporal",
            "joint_token_temporal_dropout",
        }
        if self.spatial_token_pooling_fusion_mode not in supported_spatial_token_fusions:
            raise ValueError(
                "model.spatial_token_pooling.fusion_mode must be one of "
                f"{sorted(supported_spatial_token_fusions)}"
            )
        self.spatial_token_pooling_patch_layers = tuple(
            int(layer) for layer in spatial_token_pooling_patch_layers
        )
        self.spatial_token_pooling_legacy_patch_layers = bool(
            spatial_token_pooling_legacy_patch_layers
        )
        self.spatial_token_pool: AdaptiveSpatialTokenPool | None = None
        self.spatial_token_joint_temporal: JointSpatialTemporalTransformer | None = None
        if self.spatial_token_pooling_enabled:
            if self.dual_view:
                raise ValueError(
                    "model.spatial_token_pooling requires model.view_fusion=single"
                )
            if self.fusion != "cls_transformer":
                raise ValueError(
                    "model.spatial_token_pooling requires "
                    "model.temporal_fusion=cls_transformer"
                )
            if self.frame_feature_mode != "last_cls_patch_mean":
                raise ValueError(
                    "model.spatial_token_pooling currently requires "
                    "model.frame_feature_mode=last_cls_patch_mean"
                )
            if backbone_feature_dim is None:
                raise ValueError(
                    "model.spatial_token_pooling requires uncached DINO patch tokens"
                )
            if bool(spatial_attention_enabled):
                raise ValueError(
                    "model.spatial_token_pooling and model.spatial_attention "
                    "cannot be enabled together"
                )
            self.spatial_token_pool = AdaptiveSpatialTokenPool(
                patch_dim=backbone_feature_dim,
                hidden_dim=hidden_dim,
                attention_dim=int(spatial_token_pooling_attention_dim),
                num_queries=int(spatial_token_pooling_num_queries),
                num_labels=num_labels,
                slot_dropout=float(spatial_token_pooling_slot_dropout),
                gate_init=float(spatial_token_pooling_gate_init),
                return_attention_maps=bool(spatial_token_pooling_return_maps),
            )
            if self.spatial_token_pooling_fusion_mode in {
                "joint_token_temporal",
                "slot_only_temporal",
                "joint_token_temporal_dropout",
            }:
                self.spatial_token_joint_temporal = JointSpatialTemporalTransformer(
                    hidden_dim=hidden_dim,
                    num_slots=int(spatial_token_pooling_num_queries),
                    num_layers=int(spatial_token_pooling_relation_layers),
                    num_heads=num_heads,
                    dropout=dropout,
                    max_frames=max_frames,
                    include_global_token=(
                        self.spatial_token_pooling_fusion_mode
                        != "slot_only_temporal"
                    ),
                    modality_dropout=(
                        self.spatial_token_pooling_fusion_mode
                        == "joint_token_temporal_dropout"
                    ),
                )
        self.class_evidence_enabled = bool(class_evidence_enabled)
        self.class_evidence_return_maps = bool(class_evidence_return_maps)
        self.class_evidence_mode = str(class_evidence_mode).strip().lower()
        if self.class_evidence_mode not in (
            "logit_residual",
            "global_local_temporal",
            "independent_preprojection",
        ):
            raise ValueError(
                "model.class_evidence.mode must be logit_residual, "
                "global_local_temporal, or independent_preprojection"
            )
        self.class_evidence_head: (
            LightweightClassEvidenceHead
            | GlobalLocalEvidenceTemporalHead
            | IndependentPreprojectionEvidenceHead
            | None
        ) = None
        if self.class_evidence_enabled:
            if self.dual_view:
                raise ValueError(
                    "model.class_evidence requires model.view_fusion=single"
                )
            if backbone_feature_dim is None or video_backbone_feature_dim is not None:
                raise ValueError(
                    "model.class_evidence requires an uncached DINO image backbone"
                )
            if self.spatial_token_pooling_enabled or bool(spatial_attention_enabled):
                raise ValueError(
                    "model.class_evidence cannot be combined with existing spatial modules"
                )
            if self.class_evidence_mode == "independent_preprojection":
                if self.frame_feature_mode != "multi_layer_cls_patch_attn":
                    raise ValueError(
                        "independent_preprojection requires "
                        "model.frame_feature_mode=multi_layer_cls_patch_attn"
                    )
                self.class_evidence_head = IndependentPreprojectionEvidenceHead(
                    patch_dim=backbone_feature_dim,
                    raw_feature_dim=frame_feature_dim,
                    num_feature_layers=len(self.frame_feature_layers),
                    class_dim=int(class_evidence_hidden_dim),
                    num_labels=num_labels,
                    queries_per_class=int(class_evidence_queries_per_class),
                    temporal_layers=int(class_evidence_temporal_layers),
                    temporal_heads=int(class_evidence_temporal_heads),
                    dropout=dropout,
                    max_frames=max_frames,
                    bottleneck_frames=int(class_evidence_bottleneck_frames),
                    anti_erasure_no_evidence=bool(
                        class_evidence_anti_erasure_no_evidence
                    ),
                    causal_local_counterfactual=bool(
                        class_evidence_causal_local_counterfactual
                    ),
                    set_piece_index=LABEL_TO_INDEX.get("set_piece", -1),
                    set_piece_subtype_count=int(set_piece_subtype_count),
                    save_shot_conditioning=bool(save_shot_conditioning),
                    save_shot_condition_hidden_dim=int(save_shot_condition_hidden_dim),
                    save_shot_condition_window_frames=int(
                        save_shot_condition_window_frames
                    ),
                    save_shot_condition_shot_index=LABEL_TO_INDEX.get("shot", 0),
                    save_shot_condition_save_index=LABEL_TO_INDEX.get("save", 1),
                    evidence_feature_dim=int(evidence_feature_dim),
                    evidence_gate_init=float(evidence_gate_init),
                )
            elif self.class_evidence_mode == "global_local_temporal":
                if self.fusion != "cls_transformer":
                    raise ValueError(
                        "global_local_temporal class evidence requires "
                        "model.temporal_fusion=cls_transformer"
                    )
                self.class_evidence_head = GlobalLocalEvidenceTemporalHead(
                    patch_dim=backbone_feature_dim,
                    hidden_dim=hidden_dim,
                    attention_dim=int(class_evidence_attention_dim),
                    num_labels=num_labels,
                    queries_per_class=int(class_evidence_queries_per_class),
                    temporal=self.temporal,
                    classifier=self.head,
                    frame_event_head=self.frame_event_head,
                    dropout=dropout,
                    return_attention_maps=bool(class_evidence_return_maps),
                )
            else:
                self.class_evidence_head = LightweightClassEvidenceHead(
                    patch_dim=backbone_feature_dim,
                    hidden_dim=hidden_dim,
                    attention_dim=int(class_evidence_attention_dim),
                    evidence_dim=int(class_evidence_hidden_dim),
                    num_labels=num_labels,
                    queries_per_class=int(class_evidence_queries_per_class),
                    topk_per_class=class_evidence_topk_per_class,
                    context_frames_per_class=class_evidence_context_frames_per_class,
                    positive_max_per_class=class_evidence_positive_max_per_class,
                    negative_max_per_class=class_evidence_negative_max_per_class,
                    dropout=dropout,
                    return_attention_maps=bool(class_evidence_return_maps),
                )
        self.spatial_attention_mode = str(spatial_attention_mode).strip().lower()
        if self.spatial_attention_mode not in (
            "residual",
            "probe",
            "adaptive_fusion",
            "conditioned_residual",
            "adaspot_feature_fusion",
        ):
            raise ValueError(
                "model.spatial_attention.mode must be residual, probe, "
                "adaptive_fusion, conditioned_residual, or "
                "adaspot_feature_fusion"
            )
        self.spatial_attention_patch_mode = str(spatial_attention_patch_mode).strip().lower()
        if self.spatial_attention_patch_mode not in ("last_layer", "residual_multilayer"):
            raise ValueError(
                "model.spatial_attention.patch_mode must be last_layer or residual_multilayer"
            )
        self.spatial_attention_feature_layers = _coerce_frame_feature_layers(
            spatial_attention_feature_layers
        )
        if not self.spatial_attention_feature_layers:
            raise ValueError("model.spatial_attention.feature_layers must not be empty")
        self.spatial_patch_adapter: nn.Module | None = None
        self.spatial_patch_mix_logits: nn.Parameter | None = None
        if self.spatial_attention_patch_mode == "residual_multilayer":
            if backbone_feature_dim is None:
                raise ValueError("multi-layer spatial patches require a DINO backbone")
            patch_bottleneck = max(hidden_dim // 2, 128)
            self.spatial_patch_adapter = nn.Sequential(
                nn.LayerNorm(backbone_feature_dim),
                nn.Linear(backbone_feature_dim, patch_bottleneck),
                nn.GELU(),
                nn.Linear(patch_bottleneck, backbone_feature_dim),
            )
            self.spatial_patch_mix_logits = nn.Parameter(
                torch.zeros(len(self.spatial_attention_feature_layers))
            )
            nn.init.zeros_(self.spatial_patch_adapter[-1].weight)
            nn.init.zeros_(self.spatial_patch_adapter[-1].bias)
        self.spatial_attention_enabled = bool(spatial_attention_enabled)
        if self.spatial_attention_enabled:
            if self.dual_view:
                raise ValueError(
                    "model.spatial_attention currently requires model.view_fusion=single"
                )
            if backbone_feature_dim is None:
                raise ValueError(
                    "model.spatial_attention requires uncached DINO patch tokens"
                )
            if self.frame_feature_mode not in (
                "last_cls_patch_mean",
                "multi_layer_cls_patch_attn",
                "residual_multilayer_cls_patch_attn",
            ):
                raise ValueError(
                    "model.spatial_attention currently requires "
                    "a DINO frame feature mode"
                )
            self.spatial_attention = TemporalConditionedSpatialAttention(
                patch_dim=backbone_feature_dim,
                hidden_dim=hidden_dim,
                attention_dim=int(spatial_attention_dim),
                num_labels=num_labels,
                queries_per_class=int(spatial_attention_queries_per_class),
                context_layers=int(spatial_attention_context_layers),
                temporal_layers=int(spatial_attention_temporal_layers),
                num_heads=int(spatial_attention_heads),
                dropout=dropout,
                max_frames=max_frames,
                gate_init=float(spatial_attention_gate_init),
                dynamic_query_scale_init=float(
                    spatial_attention_dynamic_query_scale_init
                ),
                actionness_query_scale_init=float(
                    spatial_attention_actionness_query_scale_init
                ),
                attention_mode=str(spatial_attention_attention_mode),
                topk_ratio=float(spatial_attention_topk_ratio),
                residual_max_delta=float(spatial_attention_residual_max_delta),
                return_attention_maps=bool(spatial_attention_return_maps),
            )
        else:
            self.spatial_attention = None
        self.spatial_fusion_class_embedding: nn.Parameter | None = None
        self.spatial_fusion_delta: nn.Module | None = None
        self.spatial_fusion_gate: nn.Module | None = None
        if self.spatial_attention_enabled and self.spatial_attention_mode == "adaptive_fusion":
            fusion_class_dim = 8
            fusion_hidden = 32
            self.spatial_fusion_class_embedding = nn.Parameter(
                torch.zeros(self.num_labels, fusion_class_dim)
            )
            fusion_input_dim = 5 + fusion_class_dim
            self.spatial_fusion_delta = nn.Sequential(
                nn.LayerNorm(fusion_input_dim),
                nn.Linear(fusion_input_dim, fusion_hidden),
                nn.GELU(),
                nn.Linear(fusion_hidden, 1),
            )
            self.spatial_fusion_gate = nn.Sequential(
                nn.LayerNorm(fusion_input_dim),
                nn.Linear(fusion_input_dim, fusion_hidden),
                nn.GELU(),
                nn.Linear(fusion_hidden, 1),
            )
            gate_init = min(
                max(float(spatial_attention_fusion_gate_init), 1e-4),
                1.0 - 1e-4,
            )
            nn.init.zeros_(self.spatial_fusion_delta[-1].weight)
            nn.init.zeros_(self.spatial_fusion_delta[-1].bias)
            nn.init.zeros_(self.spatial_fusion_gate[-1].weight)
            nn.init.constant_(
                self.spatial_fusion_gate[-1].bias,
                torch.logit(torch.tensor(gate_init)).item(),
            )
        self.spatial_conditioned_global_proj: nn.Module | None = None
        self.spatial_conditioned_local_proj: nn.Module | None = None
        self.spatial_conditioned_class_embedding: nn.Parameter | None = None
        self.spatial_conditioned_delta: nn.Module | None = None
        self.spatial_conditioned_gate: nn.Module | None = None
        self.spatial_conditioned_max_delta = float(
            spatial_attention_correction_max_delta
        )
        if self.spatial_conditioned_max_delta <= 0:
            raise ValueError(
                "model.spatial_attention.correction_max_delta must be positive"
            )
        if (
            self.spatial_attention_enabled
            and self.spatial_attention_mode == "conditioned_residual"
        ):
            correction_dim = max(int(spatial_attention_correction_dim), 16)
            class_dim = 16
            self.spatial_conditioned_global_proj = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, correction_dim),
            )
            self.spatial_conditioned_local_proj = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, correction_dim),
            )
            self.spatial_conditioned_class_embedding = nn.Parameter(
                torch.zeros(self.num_labels, class_dim)
            )
            correction_input_dim = correction_dim * 4 + 8 + class_dim
            correction_hidden = max(correction_dim, 64)
            self.spatial_conditioned_delta = nn.Sequential(
                nn.LayerNorm(correction_input_dim),
                nn.Linear(correction_input_dim, correction_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(correction_hidden, 1),
            )
            self.spatial_conditioned_gate = nn.Sequential(
                nn.LayerNorm(correction_input_dim),
                nn.Linear(correction_input_dim, correction_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(correction_hidden, 1),
            )
            gate_init = min(
                max(float(spatial_attention_correction_gate_init), 1e-4),
                1.0 - 1e-4,
            )
            nn.init.zeros_(self.spatial_conditioned_delta[-1].weight)
            nn.init.zeros_(self.spatial_conditioned_delta[-1].bias)
            nn.init.zeros_(self.spatial_conditioned_gate[-1].weight)
            nn.init.constant_(
                self.spatial_conditioned_gate[-1].bias,
                torch.logit(torch.tensor(gate_init)).item(),
            )
        self.spatial_feature_global_align: nn.Module | None = None
        self.spatial_feature_local_align: nn.Module | None = None
        self.spatial_feature_output: nn.Module | None = None
        if (
            self.spatial_attention_enabled
            and self.spatial_attention_mode == "adaspot_feature_fusion"
        ):
            if self.fusion != "cls_transformer":
                raise ValueError(
                    "AdaSpot feature fusion currently requires "
                    "model.temporal_fusion=cls_transformer"
                )
            # AdaSpot aligns the global/local streams before channel-wise max.
            # The global token remains an immutable skip connection; the final
            # correction is zero-initialized for exact baseline identity.
            align_hidden = max(hidden_dim // 2, 128)
            self.spatial_feature_global_align = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, align_hidden),
                nn.GELU(),
                nn.Linear(align_hidden, hidden_dim),
            )
            self.spatial_feature_local_align = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, align_hidden),
                nn.GELU(),
                nn.Linear(align_hidden, hidden_dim),
            )
            self.spatial_feature_local_align.load_state_dict(
                self.spatial_feature_global_align.state_dict()
            )
            self.spatial_feature_output = nn.Sequential(
                nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim)
            )
            nn.init.zeros_(self.spatial_feature_output[-1].weight)
            nn.init.zeros_(self.spatial_feature_output[-1].bias)
        self.roi_meta_dim = int(roi_meta_dim)
        self.roi_count = int(roi_count)
        if self.roi_count not in (1, 2):
            raise ValueError("model.roi_count must be 1 or 2")
        if self.roi_count == 2 and not self.dual_view:
            raise ValueError("model.roi_count=2 requires a dual view_fusion")
        self.roi_verifier_min_confidence = float(roi_verifier_min_confidence)
        if not 0.0 <= self.roi_verifier_min_confidence < 1.0:
            raise ValueError("model.roi_verifier_min_confidence must be in [0, 1)")
        verifier_indices = tuple(
            dict.fromkeys(int(index) for index in roi_verifier_class_indices)
        )
        if any(index < 0 or index >= self.num_labels for index in verifier_indices):
            raise ValueError(
                "model.roi_verifier_class_indices contains an invalid label index: "
                f"{verifier_indices} for num_labels={self.num_labels}"
            )
        verifier_mask = torch.ones(self.num_labels, dtype=torch.float32)
        if verifier_indices:
            verifier_mask.zero_()
            verifier_mask[list(verifier_indices)] = 1.0
        self.register_buffer(
            "roi_verifier_class_mask", verifier_mask, persistent=False
        )
        self.freeze_global_branch = False
        self.freeze_spatial_probe = False
        self.freeze_uniform_reference = False
        self.freeze_event_branch_for_gate = False
        if self.dual_view:
            self.local_frame_proj = copy.deepcopy(self.frame_proj)
            self.local_frame_event_head = copy.deepcopy(self.frame_event_head)
            self.local_temporal = copy.deepcopy(self.temporal)
            self.local_head = copy.deepcopy(self.head)
            self.multi_roi_frame_residual: nn.Module | None = None
            self.multi_roi_frame_gate: nn.Module | None = None
            if self.roi_count == 2:
                pair_dim = hidden_dim * 3 + roi_meta_dim * 2
                pair_hidden = max(hidden_dim // 2, 64)
                self.multi_roi_frame_residual = nn.Sequential(
                    nn.LayerNorm(pair_dim),
                    nn.Linear(pair_dim, pair_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(pair_hidden, hidden_dim),
                )
                self.multi_roi_frame_gate = nn.Sequential(
                    nn.LayerNorm(pair_dim),
                    nn.Linear(pair_dim, pair_hidden),
                    nn.GELU(),
                    nn.Linear(pair_hidden, 1),
                )
                nn.init.zeros_(self.multi_roi_frame_residual[-1].weight)
                nn.init.zeros_(self.multi_roi_frame_residual[-1].bias)
                nn.init.zeros_(self.multi_roi_frame_gate[-1].weight)
                nn.init.constant_(self.multi_roi_frame_gate[-1].bias, -2.0)
            self.multi_roi_gain_adapter: nn.Module | None = None
            self.multi_roi_gain_frame_head: nn.Module | None = None
            self.multi_roi_gain_clip_head: nn.Module | None = None
            if self.view_fusion == "dual_multi_roi_gain":
                if self.roi_count != 2:
                    raise ValueError(
                        "model.view_fusion=dual_multi_roi_gain requires model.roi_count=2"
                    )
                gain_frame_dim = hidden_dim * 3 + roi_meta_dim
                gain_hidden = max(hidden_dim // 2, 64)
                self.multi_roi_gain_adapter = nn.Sequential(
                    nn.LayerNorm(gain_frame_dim),
                    nn.Linear(gain_frame_dim, gain_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(gain_hidden, hidden_dim),
                )
                self.multi_roi_gain_frame_head = nn.Sequential(
                    nn.LayerNorm(gain_frame_dim),
                    nn.Linear(gain_frame_dim, max(hidden_dim // 4, 32)),
                    nn.GELU(),
                    nn.Linear(max(hidden_dim // 4, 32), 1),
                )
                gain_clip_dim = hidden_dim * 3 + roi_meta_dim * 2 + 2
                self.multi_roi_gain_clip_head = nn.Sequential(
                    nn.LayerNorm(gain_clip_dim),
                    nn.Linear(gain_clip_dim, gain_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(gain_hidden, num_labels),
                )
                nn.init.normal_(
                    self.multi_roi_gain_adapter[-1].weight, mean=0.0, std=1e-3
                )
                nn.init.zeros_(self.multi_roi_gain_adapter[-1].bias)
                nn.init.zeros_(self.multi_roi_gain_frame_head[-1].weight)
                nn.init.zeros_(self.multi_roi_gain_frame_head[-1].bias)
                nn.init.zeros_(self.multi_roi_gain_clip_head[-1].weight)
                nn.init.constant_(self.multi_roi_gain_clip_head[-1].bias, -1.5)
            gate_hidden = max(hidden_dim // 4, 32)
            if self.view_fusion == "dual_gate":
                self.roi_gate = nn.Sequential(
                    nn.LayerNorm(hidden_dim * 2 + roi_meta_dim),
                    nn.Linear(hidden_dim * 2 + roi_meta_dim, gate_hidden),
                    nn.GELU(),
                    nn.Linear(gate_hidden, num_labels),
                )
                nn.init.zeros_(self.roi_gate[-1].weight)
                nn.init.constant_(self.roi_gate[-1].bias, -3.0)
            elif self.view_fusion == "dual_verifier":
                verifier_dim = hidden_dim * 3 + roi_meta_dim
                self.roi_gate = nn.Sequential(
                    nn.LayerNorm(verifier_dim),
                    nn.Linear(verifier_dim, gate_hidden),
                    nn.GELU(),
                    nn.Linear(gate_hidden, num_labels),
                )
                self.roi_residual_head = nn.Sequential(
                    nn.LayerNorm(verifier_dim),
                    nn.Linear(verifier_dim, max(hidden_dim // 2, 64)),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(max(hidden_dim // 2, 64), num_labels),
                )
                nn.init.zeros_(self.roi_gate[-1].weight)
                nn.init.constant_(self.roi_gate[-1].bias, -2.0)
                nn.init.zeros_(self.roi_residual_head[-1].weight)
                nn.init.constant_(self.roi_residual_head[-1].bias, -4.0)
            elif self.view_fusion == "dual_multi_roi_gain":
                pass
            else:
                frame_quality_dim = hidden_dim * 3 + roi_meta_dim
                clip_quality_dim = hidden_dim * 3 + roi_meta_dim
                self.roi_feature_adapter = nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                nn.init.zeros_(self.roi_feature_adapter[-1].weight)
                nn.init.zeros_(self.roi_feature_adapter[-1].bias)
                self.roi_frame_quality_head = nn.Sequential(
                    nn.LayerNorm(frame_quality_dim),
                    nn.Linear(frame_quality_dim, gate_hidden),
                    nn.GELU(),
                    nn.Linear(gate_hidden, 1),
                )
                self.roi_quality_head = nn.Sequential(
                    nn.LayerNorm(clip_quality_dim),
                    nn.Linear(clip_quality_dim, gate_hidden),
                    nn.GELU(),
                    nn.Linear(gate_hidden, num_labels),
                )
                self.roi_residual_head = nn.Sequential(
                    nn.LayerNorm(clip_quality_dim),
                    nn.Linear(clip_quality_dim, max(hidden_dim // 2, 64)),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(max(hidden_dim // 2, 64), num_labels),
                )
                nn.init.zeros_(self.roi_frame_quality_head[-1].weight)
                nn.init.constant_(self.roi_frame_quality_head[-1].bias, -1.0)
                nn.init.zeros_(self.roi_quality_head[-1].weight)
                nn.init.constant_(self.roi_quality_head[-1].bias, -1.5)
                nn.init.zeros_(self.roi_residual_head[-1].weight)
                nn.init.zeros_(self.roi_residual_head[-1].bias)

                if self.view_fusion in (
                    "dual_token_fusion",
                    "dual_cross_attention",
                    "dual_class_query_fusion",
                    "dual_direct_class_query_fusion",
                ):
                    def normalize_delta_values(raw: Any, default: float) -> tuple[float, ...]:
                        values = list(per_label_float_tuple(raw, default=default))
                        if len(values) < self.num_labels:
                            values.extend([float(default)] * (self.num_labels - len(values)))
                        return tuple(float(value) for value in values[: self.num_labels])

                    positive_delta_values = normalize_delta_values(
                        view_fusion_positive_delta_per_class
                        if view_fusion_positive_delta_per_class is not None
                        else view_fusion_positive_delta,
                        float(view_fusion_positive_delta),
                    )
                    negative_delta_values = normalize_delta_values(
                        view_fusion_negative_delta_per_class
                        if view_fusion_negative_delta_per_class is not None
                        else view_fusion_negative_delta,
                        float(view_fusion_negative_delta),
                    )
                    self.register_buffer(
                        "view_fusion_positive_delta",
                        torch.tensor(positive_delta_values, dtype=torch.float32).clamp_min(1e-6),
                        persistent=False,
                    )
                    self.register_buffer(
                        "view_fusion_negative_delta",
                        torch.tensor(negative_delta_values, dtype=torch.float32).clamp_min(1e-6),
                        persistent=False,
                    )
                if self.view_fusion in (
                    "dual_token_fusion",
                    "dual_multi_roi_memory",
                    "dual_class_query_fusion",
                    "dual_direct_class_query_fusion",
                ):
                    # ROI and global projections share one representation. Keep
                    # ROI details intact before temporal memory attention.
                    self.roi_feature_adapter = nn.Identity()
                if self.view_fusion in (
                    "dual_token_fusion",
                    "dual_class_query_fusion",
                    "dual_direct_class_query_fusion",
                ):
                    self.dual_token_temporal = DualViewTokenFusionTransformer(
                        hidden_dim=hidden_dim,
                        num_layers=max(int(view_fusion_layers), 1),
                        num_heads=max(int(view_fusion_heads), 1),
                        dropout=dropout,
                        num_labels=num_labels,
                    )
                    self.dual_token_residual = nn.Linear(hidden_dim, 1)
                    nn.init.zeros_(self.dual_token_residual.weight)
                    nn.init.zeros_(self.dual_token_residual.bias)
                    if self.view_fusion == "dual_direct_class_query_fusion":
                        direct_hidden = max(hidden_dim // 2, 64)
                        self.dual_direct_head = nn.Sequential(
                            nn.LayerNorm(hidden_dim),
                            nn.Linear(hidden_dim, direct_hidden),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(direct_hidden, 1),
                        )
                        gate_init = float(view_fusion_direct_delta_gate_init)
                        gate_init = min(max(gate_init, 1e-4), 1.0 - 1e-4)
                        self.dual_direct_delta_gate_logits = nn.Parameter(
                            torch.full(
                                (num_labels,),
                                torch.logit(torch.tensor(gate_init)).item(),
                                dtype=torch.float32,
                            )
                        )
                        nn.init.zeros_(self.dual_direct_head[-1].weight)
                        nn.init.zeros_(self.dual_direct_head[-1].bias)
                if self.view_fusion in (
                    "dual_cross_attention",
                    "dual_multi_roi_memory",
                ):
                    self.dual_cross_attention = FullRoiCrossAttentionFusion(
                        hidden_dim=hidden_dim,
                        num_heads=max(int(view_fusion_heads), 1),
                        dropout=dropout,
                        num_layers=max(int(view_fusion_layers), 1),
                        residual_gate_init=float(view_fusion_residual_gate_init),
                    )

    @staticmethod
    def _make_temporal_module(
        fusion: str,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        max_frames: int,
        num_labels: int,
        stem_kernel_size: int = 3,
        stem_dilations: Sequence[int] = (1, 2, 4),
        stem_bottleneck_frames: int = 8,
    ) -> nn.Module:
        if fusion in {"cls_transformer", "videomae_residual_transformer"}:
            return TemporalTransformer(hidden_dim, num_layers, num_heads, dropout, max_frames)
        if fusion in {
            "event_topk_transformer",
            "event_anchor_transformer",
            "uniform_event_dual_transformer",
        }:
            return TemporalTransformer(
                hidden_dim, num_layers, num_heads, dropout, max_frames
            )
        if fusion == "temporal_stem_bottleneck_transformer":
            return TemporalStemBottleneckTransformer(
                hidden_dim,
                num_layers,
                num_heads,
                dropout,
                max_frames,
                stem_kernel_size=stem_kernel_size,
                multiscale_dilations=stem_dilations,
                bottleneck_frames=stem_bottleneck_frames,
            )
        if fusion == "attn_pool_transformer":
            return AttentionPoolTransformer(hidden_dim, num_layers, num_heads, dropout, max_frames)
        if fusion == "class_query_transformer":
            return ClassQueryTransformer(hidden_dim, num_layers, num_heads, dropout, max_frames, num_labels)
        if fusion == "mean":
            return nn.Sequential(nn.LayerNorm(hidden_dim))
        if fusion == "bigru":
            return AttentionBiGRU(hidden_dim, dropout)
        if fusion == "tcn":
            return TemporalConvNet(
                hidden_dim,
                num_layers=num_layers,
                kernel_size=3,
                dropout=dropout,
                base_dilation=1,
            )
        raise ValueError(
            "model.temporal_fusion must be one of: transformer, cls_transformer, "
            "attn_pool_transformer, class_query_transformer, event_topk_transformer, "
            "event_anchor_transformer, uniform_event_dual_transformer, temporal_stem_bottleneck_transformer, mean, bigru, tcn"
        )

    def initialize_uniform_from_primary(self) -> None:
        if self.fusion != "uniform_event_dual_transformer":
            return
        self.uniform_temporal.load_state_dict(self.temporal.state_dict())
        self.uniform_head.load_state_dict(self.head.state_dict())

    def initialize_class_evidence_temporal_from_primary(self) -> None:
        if not (
            self.class_evidence_enabled
            and self.class_evidence_mode == "global_local_temporal"
            and isinstance(
                self.class_evidence_head, GlobalLocalEvidenceTemporalHead
            )
        ):
            return
        self.class_evidence_head.initialize_from_primary(
            self.temporal,
            self.head,
            self.frame_event_head,
        )

    def initialize_local_from_global(self) -> None:
        if not self.dual_view:
            return
        self.local_frame_proj.load_state_dict(self.frame_proj.state_dict())
        self.local_frame_event_head.load_state_dict(self.frame_event_head.state_dict())
        self.local_temporal.load_state_dict(self.temporal.state_dict())
        self.local_head.load_state_dict(self.head.state_dict())

    def freeze_backbone_parameters(self) -> None:
        if self.backbone is not None:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
        self.backbone_has_trainable_params = any(
            p.requires_grad
            for module in (self.backbone, self.local_backbone)
            if module is not None
            for p in module.parameters()
        )

    def freeze_all_backbone_parameters(self) -> None:
        for module in (self.backbone, self.local_backbone):
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad = False
        self.backbone_has_trainable_params = False

    def initialize_local_backbone_from_global(self) -> None:
        if self.backbone is None or self.local_backbone is None:
            return
        self.local_backbone.load_state_dict(self.backbone.state_dict())
        self.backbone_has_trainable_params = any(
            p.requires_grad
            for module in (self.backbone, self.local_backbone)
            if module is not None
            for p in module.parameters()
        )

    def freeze_uniform_reference_parameters(self) -> None:
        if self.fusion != "uniform_event_dual_transformer":
            raise ValueError(
                "freeze_uniform_reference requires uniform_event_dual_transformer"
            )
        self.freeze_uniform_reference = True
        self.freeze_backbone_parameters()
        for module in (
            self.frame_proj,
            self.frame_event_head,
            self.uniform_temporal,
            self.uniform_head,
        ):
            for parameter in module.parameters():
                parameter.requires_grad = False

    def freeze_temporal_branches_for_gate_parameters(self) -> None:
        if self.fusion != "uniform_event_dual_transformer":
            raise ValueError(
                "freeze_event_branch_for_gate requires uniform_event_dual_transformer"
            )
        self.freeze_event_branch_for_gate = True
        self.freeze_uniform_reference_parameters()
        for module in (self.temporal, self.head):
            for parameter in module.parameters():
                parameter.requires_grad = False

    def freeze_global_parameters(self) -> None:
        self.freeze_global_branch = True
        self.freeze_backbone_parameters()
        for module in (
            self.frame_proj,
            self.frame_event_head,
            self.temporal,
            self.head,
            self.frame_patch_attn,
            self.multi_layer_frame_adapter,
        ):
            if module is None:
                continue
            for parameter in module.parameters():
                parameter.requires_grad = False

    def freeze_spatial_probe_parameters(self) -> None:
        if self.spatial_attention is None:
            raise ValueError("freeze_spatial_probe requires spatial attention")
        self.freeze_spatial_probe = True
        for parameter in self.spatial_attention.parameters():
            parameter.requires_grad = False
        if self.spatial_patch_adapter is not None:
            for parameter in self.spatial_patch_adapter.parameters():
                parameter.requires_grad = False
        if self.spatial_patch_mix_logits is not None:
            self.spatial_patch_mix_logits.requires_grad = False
        if (
            self.frame_feature_mode == "residual_multilayer_cls_patch_attn"
            and self.multi_layer_patch_adapter is not None
        ):
            for parameter in self.multi_layer_patch_adapter.parameters():
                parameter.requires_grad = False
            if self.multi_layer_patch_mix_logits is not None:
                self.multi_layer_patch_mix_logits.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        for backbone_module in (self.backbone, self.local_backbone):
            if backbone_module is not None and not any(p.requires_grad for p in backbone_module.parameters()):
                backbone_module.eval()
        if self.freeze_global_branch:
            self.frame_proj.eval()
            self.frame_event_head.eval()
            self.temporal.eval()
            self.head.eval()
            if self.frame_patch_attn is not None:
                self.frame_patch_attn.eval()
            if self.multi_layer_frame_adapter is not None:
                self.multi_layer_frame_adapter.eval()
        if self.freeze_spatial_probe and self.spatial_attention is not None:
            self.spatial_attention.eval()
            if self.spatial_patch_adapter is not None:
                self.spatial_patch_adapter.eval()
            if (
                self.frame_feature_mode == "residual_multilayer_cls_patch_attn"
                and self.multi_layer_patch_adapter is not None
            ):
                self.multi_layer_patch_adapter.eval()
        if self.freeze_uniform_reference:
            self.frame_proj.eval()
            self.frame_event_head.eval()
            self.uniform_temporal.eval()
            self.uniform_head.eval()
        if self.freeze_event_branch_for_gate:
            self.temporal.eval()
            self.head.eval()
        return self

    def preprocess_inputs(self, inputs: Tensor) -> Tensor:
        if inputs.dtype == torch.uint8:
            inputs = inputs.float().div_(255.0)
            return (inputs - self.input_mean) / self.input_std
        return inputs

    def _resolved_frame_feature_layers(self, backbone: nn.Module) -> tuple[int, ...]:
        blocks = getattr(backbone, "blocks", None)
        if blocks is None:
            raise ValueError("multi_layer_cls_patch_attn requires a ViT-style backbone with .blocks")
        total = len(blocks)
        resolved: list[int] = []
        for layer in self.frame_feature_layers:
            index = total + int(layer) if int(layer) < 0 else int(layer)
            if index < 0 or index >= total:
                raise ValueError(
                    f"model.frame_feature_layers contains out-of-range layer {layer} "
                    f"for backbone with {total} blocks"
                )
            resolved.append(index)
        return tuple(dict.fromkeys(resolved))

    def _resolved_spatial_feature_layers(
        self, backbone: nn.Module
    ) -> tuple[int, ...]:
        blocks = getattr(backbone, "blocks", None)
        if blocks is None:
            raise ValueError("multi-layer spatial patches require a ViT backbone")
        total = len(blocks)
        resolved: list[int] = []
        for layer in self.spatial_attention_feature_layers:
            index = total + int(layer) if int(layer) < 0 else int(layer)
            if index < 0 or index >= total:
                raise ValueError(
                    "model.spatial_attention.feature_layers contains "
                    f"out-of-range layer {layer} for {total} blocks"
                )
            resolved.append(index)
        resolved_layers = tuple(dict.fromkeys(resolved))
        if resolved_layers[-1] != total - 1:
            raise ValueError(
                "model.spatial_attention.feature_layers must include the final block"
            )
        return resolved_layers

    def _extract_spatial_multilayer_patches(
        self, backbone: nn.Module, flat: Tensor
    ) -> tuple[Tensor, Tensor]:
        if self.spatial_patch_adapter is None or self.spatial_patch_mix_logits is None:
            raise RuntimeError("multi-layer spatial patch adapter is not initialized")
        outputs = backbone.get_intermediate_layers(
            flat,
            n=self._resolved_spatial_feature_layers(backbone),
            return_class_token=True,
            norm=True,
        )
        patch_mix = self.spatial_patch_mix_logits.softmax(dim=0)
        mixed_patches: Tensor | None = None
        for layer_weight, (patches, _cls) in zip(patch_mix, outputs):
            weighted = layer_weight * patches
            mixed_patches = weighted if mixed_patches is None else mixed_patches + weighted
        if mixed_patches is None:
            raise RuntimeError("DINO returned no intermediate patch tokens")
        last_patches, last_cls = outputs[-1]
        base_features = torch.cat(
            (last_cls, last_patches.mean(dim=1)), dim=-1
        )
        patch_delta = self.spatial_patch_adapter(mixed_patches)
        return base_features, last_patches + patch_delta

    def _pool_patch_tokens(self, patches: Tensor) -> Tensor:
        if self.frame_patch_pool == "mean" or self.frame_patch_attn is None:
            return patches.mean(dim=1)
        weights = self.frame_patch_attn(patches).softmax(dim=1)
        return (patches * weights).sum(dim=1)

    def _resolved_spatial_token_pooling_layers(
        self, backbone: nn.Module
    ) -> tuple[int, ...]:
        # Remote checkpoints before this compatibility merge accidentally used
        # frame_feature_layers whenever patch_layers was non-empty. Preserve
        # that behavior by default, while allowing new experiments to opt into
        # the configured patch_layers explicitly.
        raw_layers = (
            self.frame_feature_layers
            if self.spatial_token_pooling_legacy_patch_layers
            else self.spatial_token_pooling_patch_layers
        )
        blocks = getattr(backbone, "blocks", None)
        if blocks is None:
            raise ValueError("spatial-token patch layers require a ViT backbone")
        total = len(blocks)
        resolved: list[int] = []
        for layer in raw_layers:
            index = total + int(layer) if int(layer) < 0 else int(layer)
            if index < 0 or index >= total:
                raise ValueError(
                    "model.spatial_token_pooling.patch_layers contains "
                    f"out-of-range layer {layer} for {total} blocks"
                )
            resolved.append(index)
        resolved_layers = tuple(dict.fromkeys(resolved))
        if not resolved_layers or resolved_layers[-1] != total - 1:
            raise ValueError(
                "spatial-token patch layers must include the final block"
            )
        return resolved_layers

    def _extract_spatial_token_multilayer_patches(
        self, backbone: nn.Module, flat: Tensor
    ) -> tuple[Tensor, Tensor]:
        outputs = backbone.get_intermediate_layers(
            flat,
            n=self._resolved_spatial_token_pooling_layers(backbone),
            return_class_token=True,
            norm=True,
        )
        last_patches, last_cls = outputs[-1]
        fused_patches = torch.stack(
            [patches for patches, _cls in outputs], dim=0
        ).mean(dim=0)
        base_features = torch.cat(
            (last_cls, last_patches.mean(dim=1)), dim=-1
        )
        return base_features, fused_patches

    def _extract_residual_multilayer_features_and_patches(
        self,
        backbone: nn.Module,
        flat: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if (
            self.multi_layer_frame_adapter is None
            or self.multi_layer_patch_adapter is None
            or self.multi_layer_patch_mix_logits is None
        ):
            raise RuntimeError("Residual multi-layer adapters are not initialized")
        layers = self._resolved_frame_feature_layers(backbone)
        blocks = getattr(backbone, "blocks")
        if layers[-1] != len(blocks) - 1:
            raise ValueError(
                "residual_multilayer_cls_patch_attn requires the final DINO "
                "block in model.frame_feature_layers"
            )
        outputs = backbone.get_intermediate_layers(
            flat,
            n=layers,
            return_class_token=True,
            norm=True,
        )
        pieces: list[Tensor] = []
        patch_mix = self.multi_layer_patch_mix_logits.softmax(dim=0)
        mixed_patches: Tensor | None = None
        for layer_weight, (patches, cls) in zip(patch_mix, outputs):
            pieces.extend((cls, self._pool_patch_tokens(patches)))
            weighted_patches = layer_weight * patches
            mixed_patches = (
                weighted_patches
                if mixed_patches is None
                else mixed_patches + weighted_patches
            )
        if mixed_patches is None:
            raise RuntimeError("DINO returned no intermediate patch tokens")
        last_patches, last_cls = outputs[-1]
        base_features = torch.cat(
            (last_cls, last_patches.mean(dim=1)), dim=-1
        )
        multi_layer_features = torch.cat(pieces, dim=-1)
        patch_delta = self.multi_layer_patch_adapter(mixed_patches)
        return (
            torch.cat((base_features, multi_layer_features), dim=-1),
            last_patches + patch_delta,
        )

    def extract_frame_features_from_flat(self, backbone: nn.Module, flat: Tensor) -> Tensor:
        if self.frame_feature_mode == "last_cls_patch_mean":
            return extract_dino_frame_features(backbone, flat)
        if self.frame_feature_mode == "residual_multilayer_cls_patch_attn":
            features, _patches = (
                self._extract_residual_multilayer_features_and_patches(
                    backbone, flat
                )
            )
            return features
        layers = self._resolved_frame_feature_layers(backbone)
        outputs = backbone.get_intermediate_layers(
            flat,
            n=layers,
            return_class_token=True,
            norm=True,
        )
        pieces: list[Tensor] = []
        for patches, cls in outputs:
            pieces.append(cls)
            pieces.append(self._pool_patch_tokens(patches))
        return torch.cat(pieces, dim=-1)

    def encode_frames(
        self,
        inputs: Tensor,
        backbone: nn.Module | None = None,
        *,
        require_input_grad: bool = False,
    ) -> Tensor:
        selected_backbone = self.backbone if backbone is None else backbone
        if selected_backbone is None:
            return inputs
        inputs = self.preprocess_inputs(inputs)
        bsz, frames, channels, height, width = inputs.shape
        if bool(getattr(selected_backbone, "is_video_backbone", False)):
            has_trainable_params = any(
                parameter.requires_grad
                for parameter in selected_backbone.parameters()
            )
            if has_trainable_params:
                return selected_backbone.forward_temporal_features(
                    inputs, output_frames=frames
                )
            with torch.no_grad():
                return selected_backbone.forward_temporal_features(
                    inputs, output_frames=frames
                )
        flat = inputs.reshape(bsz * frames, channels, height, width)
        has_trainable_params = any(p.requires_grad for p in selected_backbone.parameters())
        def extract_chunk(chunk: Tensor) -> Tensor:
            if has_trainable_params or require_input_grad:
                if self.training and self.gradient_checkpointing:
                    return activation_checkpoint(
                        lambda x: self.extract_frame_features_from_flat(selected_backbone, x),
                        chunk,
                        use_reentrant=False,
                        preserve_rng_state=True,
                    )
                return self.extract_frame_features_from_flat(selected_backbone, chunk)
            with torch.no_grad():
                return self.extract_frame_features_from_flat(selected_backbone, chunk)

        chunk_size = self.backbone_frame_chunk_size
        if chunk_size > 0 and flat.shape[0] > chunk_size:
            features = torch.cat(
                [extract_chunk(chunk) for chunk in flat.split(chunk_size, dim=0)], dim=0
            )
        else:
            features = extract_chunk(flat)
        return features.reshape(bsz, frames, -1)

    def _select_highres_candidates(self, frame_logits: Tensor) -> Tensor:
        """Select separated candidate times; the discrete indices are not differentiated."""
        scores = frame_logits.sigmoid().amax(dim=-1).detach()
        selected: list[Tensor] = []
        working = scores.clone()
        frames = scores.shape[1]
        for _ in range(min(self.highres_glimpse_candidates, frames)):
            index = working.argmax(dim=1)
            selected.append(index)
            positions = torch.arange(frames, device=scores.device).unsqueeze(0)
            suppress = (positions - index.unsqueeze(1)).abs() <= self.highres_glimpse_nms_radius
            working = working.masked_fill(suppress, -1.0)
        return torch.stack(selected, dim=1).sort(dim=1).values

    def _highres_glimpse_outputs(
        self,
        global_outputs: dict[str, Tensor],
        patch_tokens: Tensor,
        highres_pool_inputs: Tensor,
        global_frame_times: Tensor | None = None,
        highres_pool_times: Tensor | None = None,
        clip_targets: Tensor | None = None,
        clip_label_masks: Tensor | None = None,
    ) -> dict[str, Tensor]:
        modules = (
            self.highres_crop_query,
            self.highres_crop_key,
            self.highres_crop_scale,
            self.highres_token_fusion,
            self.highres_local_motion_fusion,
            self.highres_local_attention,
            self.highres_local_classifier,
            self.highres_evidence_gate,
            self.highres_residual_head,
        )
        if any(module is None for module in modules) or self.highres_event_query_embedding is None or self.highres_no_evidence_token is None:
            raise RuntimeError("high-resolution glimpse modules are missing")
        bsz, global_frames, hidden = global_outputs["frame_tokens"].shape
        _, pool_frames, channels, pool_h, pool_w = highres_pool_inputs.shape
        candidate_indices = self._select_highres_candidates(
            global_outputs["frame_event_logits"]
        )
        candidates = candidate_indices.shape[1]
        patch_count = patch_tokens.shape[2]
        grid_h = max(int(round(math.sqrt(patch_count * pool_h / max(pool_w, 1)))), 1)
        while grid_h > 1 and patch_count % grid_h:
            grid_h -= 1
        grid_w = patch_count // grid_h

        gather_frame = candidate_indices.unsqueeze(-1).expand(-1, -1, hidden)
        candidate_tokens = global_outputs["frame_tokens"].gather(1, gather_frame)
        gather_patch = candidate_indices[:, :, None, None].expand(
            -1, -1, patch_count, patch_tokens.shape[-1]
        )
        candidate_patches = patch_tokens.gather(1, gather_patch)
        queries = self.highres_crop_query(candidate_tokens)
        keys = self.highres_crop_key(candidate_patches)
        candidate_class_logits = global_outputs["frame_event_logits"].gather(
            1, candidate_indices.unsqueeze(-1).expand(-1, -1, self.num_labels)
        )
        queries = queries + candidate_class_logits.sigmoid() @ self.highres_event_query_embedding
        attention_logits = torch.einsum("bcd,bcpd->bcp", queries, keys)
        attention_logits = attention_logits / math.sqrt(float(keys.shape[-1]))
        # Camera motion changes most patches similarly. Standardizing the
        # two-sided patch change per frame suppresses that global component and
        # retains local action boundaries as a soft detector-free prior.
        prev_indices = (candidate_indices - 1).clamp_min(0)
        next_indices = (candidate_indices + 1).clamp_max(global_frames - 1)
        gather_prev = prev_indices[:, :, None, None].expand(
            -1, -1, patch_count, patch_tokens.shape[-1]
        )
        gather_next = next_indices[:, :, None, None].expand(
            -1, -1, patch_count, patch_tokens.shape[-1]
        )
        prev_patches = patch_tokens.gather(1, gather_prev)
        next_patches = patch_tokens.gather(1, gather_next)
        motion_prior = 0.5 * (
            (candidate_patches - prev_patches).square().mean(dim=-1)
            + (next_patches - candidate_patches).square().mean(dim=-1)
        )
        motion_prior = motion_prior.detach()
        motion_prior = (
            motion_prior - motion_prior.mean(dim=-1, keepdim=True)
        ) / motion_prior.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        attention_logits = (
            attention_logits
            + self.highres_glimpse_motion_prior_weight * motion_prior
        ) / self.highres_glimpse_attention_temperature
        attention = attention_logits.softmax(dim=-1)
        xs = torch.linspace(0.0, 1.0, grid_w, device=attention.device, dtype=attention.dtype)
        ys = torch.linspace(0.0, 1.0, grid_h, device=attention.device, dtype=attention.dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        center_x = (attention * xx.flatten()).sum(dim=-1)
        center_y = (attention * yy.flatten()).sum(dim=-1)
        raw_scale = self.highres_crop_scale(candidate_tokens).sigmoid()
        min_scale = raw_scale.new_tensor(self.highres_glimpse_min_scale)
        max_scale = raw_scale.new_tensor(self.highres_glimpse_max_scale)
        scales = min_scale + raw_scale * (max_scale - min_scale)
        # Keep every crop inside the source frame without clipping, preserving gradients.
        center_x = scales[..., 0] * 0.5 + center_x * (1.0 - scales[..., 0])
        center_y = scales[..., 1] * 0.5 + center_y * (1.0 - scales[..., 1])

        if global_frame_times is not None and highres_pool_times is not None:
            candidate_times = global_frame_times.gather(1, candidate_indices)
            center_pool = (
                candidate_times.unsqueeze(-1) - highres_pool_times.unsqueeze(1)
            ).abs().argmin(dim=-1)
        else:
            center_pool = torch.round(
                candidate_indices.float() * float(max(pool_frames - 1, 0))
                / float(max(global_frames - 1, 1))
            ).long()
        count = self.highres_glimpse_frames_per_candidate
        offsets = torch.arange(count, device=center_pool.device) - (count - 1) // 2
        selected_pool_indices = (
            center_pool.unsqueeze(-1) + offsets.view(1, 1, -1)
        ).clamp(0, max(pool_frames - 1, 0))
        flat_indices = selected_pool_indices.flatten(1)
        gather_pixels = flat_indices[:, :, None, None, None].expand(
            -1, -1, channels, pool_h, pool_w
        )
        selected_pixels = highres_pool_inputs.gather(1, gather_pixels).float().div(255.0)
        repeated_centers = torch.stack([center_x, center_y], dim=-1)
        repeated_centers = repeated_centers.unsqueeze(2).expand(-1, -1, count, -1).flatten(1, 2)
        repeated_scales = scales.unsqueeze(2).expand(-1, -1, count, -1).flatten(1, 2)
        theta = selected_pixels.new_zeros((bsz, candidates * count, 2, 3))
        theta[..., 0, 0] = repeated_scales[..., 0]
        theta[..., 1, 1] = repeated_scales[..., 1]
        theta[..., 0, 2] = repeated_centers[..., 0] * 2.0 - 1.0
        theta[..., 1, 2] = repeated_centers[..., 1] * 2.0 - 1.0
        flat_pixels = selected_pixels.flatten(0, 1)
        flat_theta = theta.flatten(0, 1)
        grid = F.affine_grid(
            flat_theta,
            (flat_pixels.shape[0], channels, self.highres_glimpse_crop_size, self.highres_glimpse_crop_size),
            align_corners=False,
        )
        crops = F.grid_sample(
            flat_pixels, grid, mode="bilinear", padding_mode="border", align_corners=False
        ).reshape(
            bsz, candidates * count, channels,
            self.highres_glimpse_crop_size, self.highres_glimpse_crop_size,
        )
        local_features = self.encode_frames(crops, require_input_grad=True)
        # Reuse the checkpoint's exact frame projection (including its learned
        # residual multi-layer adapter) without introducing a random local head.
        local_branch = self._global_branch_outputs(local_features)
        local_tokens = local_branch["frame_tokens"]
        local_frame_logits = local_branch["frame_event_logits"]

        local_groups = local_tokens.reshape(bsz, candidates, count, hidden)
        local_appearance = local_groups.mean(dim=2)
        if count > 1:
            local_direction = local_groups[:, :, -1] - local_groups[:, :, 0]
            local_motion = (
                local_groups[:, :, 1:] - local_groups[:, :, :-1]
            ).abs().mean(dim=2)
        else:
            local_direction = torch.zeros_like(local_appearance)
            local_motion = torch.zeros_like(local_appearance)
        candidate_local = self.highres_local_motion_fusion(
            torch.cat([local_appearance, local_direction, local_motion], dim=-1)
        )

        # This auxiliary head shapes the crop without updating the primary
        # temporal model on a different sequence length/distribution.
        local_attention = self.highres_local_attention(candidate_local)
        local_attention = local_attention.transpose(1, 2).softmax(dim=-1)
        local_pooled = torch.einsum(
            "blc,bch->blh", local_attention, candidate_local
        )
        local_logit_matrix = self.highres_local_classifier(local_pooled)
        local_logits = local_logit_matrix.diagonal(dim1=1, dim2=2)

        def guarded_residual(local_evidence: Tensor) -> tuple[Tensor, Tensor]:
            if self.highres_residual_head is None or self.highres_residual_gate_logits is None:
                raise RuntimeError("high-resolution residual modules are missing")
            class_global = global_outputs["temporal"].unsqueeze(1).expand(-1, self.num_labels, -1)
            residual_features = torch.cat(
                [
                    class_global,
                    local_evidence,
                    class_global * local_evidence,
                    (class_global - local_evidence).abs(),
                ],
                dim=-1,
            )
            raw_delta = self.highres_residual_head(residual_features).squeeze(-1)
            gates = self.highres_residual_gate_logits.sigmoid().unsqueeze(0)
            delta = (
                gates
                * self.highres_glimpse_residual_max_delta
                * raw_delta.tanh()
            )
            return global_outputs["logits"] + delta, delta

        def gated_candidates(local: Tensor) -> tuple[Tensor, Tensor]:
            gate_features = torch.cat(
                [
                    candidate_tokens,
                    local,
                    candidate_tokens * local,
                    (candidate_tokens - local).abs(),
                ],
                dim=-1,
            )
            gates = self.highres_evidence_gate(gate_features).sigmoid()
            no_evidence = self.highres_no_evidence_token.expand(
                bsz, candidates, -1
            )
            return gates * local + (1.0 - gates) * no_evidence, gates

        candidate_evidence, evidence_gates = gated_candidates(candidate_local)
        positions = torch.arange(global_frames, device=local_tokens.device, dtype=local_tokens.dtype)
        distance = positions.view(1, global_frames, 1) - candidate_indices.to(local_tokens.dtype).unsqueeze(1)
        weights = torch.exp(-0.5 * distance.square() / 1.5**2)
        local_support = torch.einsum("btc,bch->bth", weights, candidate_evidence)
        local_support = local_support / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        global_tokens = global_outputs["frame_tokens"]

        def joint_fusion(support: Tensor) -> Tensor:
            return self.highres_token_fusion(
                torch.cat(
                    [
                        global_tokens,
                        support,
                        global_tokens * support,
                        (global_tokens - support).abs(),
                    ],
                    dim=-1,
                )
            )

        residual_delta = torch.zeros_like(global_outputs["logits"])
        if self.highres_glimpse_fusion_mode == "bounded_residual":
            fused_tokens = global_tokens
            temporal = global_outputs["temporal"]
            logits, residual_delta = guarded_residual(local_pooled)
        else:
            fused_tokens = joint_fusion(local_support)
            temporal = self.temporal(fused_tokens)
            logits = self.head(temporal)
        shuffled_logits: Tensor | None = None
        causal_valid_mask: Tensor | None = None
        if self.training and bsz > 1:
            shuffled_outputs = []
            causal_masks = []
            for shift in range(1, bsz):
                if self.highres_glimpse_fusion_mode == "bounded_residual":
                    shuffled_pooled = local_pooled.roll(shifts=shift, dims=0)
                    shuffled_outputs.append(guarded_residual(shuffled_pooled)[0])
                else:
                    shuffled_local = candidate_local.roll(shifts=shift, dims=0)
                    shuffled_evidence, _ = gated_candidates(shuffled_local)
                    shuffled_support = torch.einsum(
                        "btc,bch->bth", weights, shuffled_evidence
                    )
                    shuffled_support = shuffled_support / weights.sum(
                        dim=-1, keepdim=True
                    ).clamp_min(1e-6)
                    shuffled_outputs.append(
                        self.head(self.temporal(joint_fusion(shuffled_support)))
                    )
                if clip_targets is not None and clip_label_masks is not None:
                    donor_targets = clip_targets.roll(shifts=shift, dims=0)
                    donor_masks = clip_label_masks.roll(shifts=shift, dims=0)
                    causal_masks.append(
                        (donor_targets < 0.5).to(logits.dtype) * donor_masks
                    )
            shuffled_logits = torch.stack(shuffled_outputs, dim=1)
            if causal_masks:
                causal_valid_mask = torch.stack(causal_masks, dim=1)
        result = {
            "logits": logits,
            "frame_event_logits": self.frame_event_head(fused_tokens),
            "local_frame_event_logits": local_frame_logits,
            "local_logits": local_logits,
            "candidate_indices": candidate_indices,
            "selected_pool_indices": flat_indices,
            "crop_centers": torch.stack([center_x, center_y], dim=-1),
            "crop_scales": scales,
            "crop_attention": attention.reshape(bsz, candidates, grid_h, grid_w),
            "crops": crops,
            "fused_frame_tokens": fused_tokens,
            "evidence_gates": evidence_gates,
            "local_attention": local_attention,
            "residual_delta": residual_delta,
            "residual_gate": self.highres_residual_gate_logits.sigmoid(),
            "motion_prior": motion_prior,
        }
        if shuffled_logits is not None:
            result["shuffled_local_logits"] = shuffled_logits
        if causal_valid_mask is not None:
            result["causal_valid_mask"] = causal_valid_mask
        return result

    def encode_frames_with_patch_tokens(
        self,
        inputs: Tensor,
        backbone: nn.Module | None = None,
    ) -> tuple[Tensor, Tensor]:
        selected_backbone = self.backbone if backbone is None else backbone
        if selected_backbone is None:
            raise ValueError(
                "Spatial attention cannot run from cached frame features because "
                "raw DINO patch tokens are required"
            )
        inputs = self.preprocess_inputs(inputs)
        bsz, frames, channels, height, width = inputs.shape
        flat = inputs.reshape(bsz * frames, channels, height, width)
        has_trainable_params = any(
            parameter.requires_grad
            for parameter in selected_backbone.parameters()
        )
        post_backbone_modules = (
            self.frame_patch_attn,
            self.multi_layer_patch_adapter,
            self.spatial_patch_adapter,
        )
        has_trainable_postprocess = any(
            parameter.requires_grad
            for module in post_backbone_modules
            if module is not None
            for parameter in module.parameters()
        ) or any(
            parameter is not None and parameter.requires_grad
            for parameter in (
                self.multi_layer_patch_mix_logits,
                self.spatial_patch_mix_logits,
            )
        )

        # Resolve negative layer indices to positive before capture
        _resolved_layers = (
            ()
            if self.frame_feature_mode == "last_cls_patch_mean"
            else self._resolved_frame_feature_layers(selected_backbone)
        )
        supports_trainable_block_checkpoint = (
            self.frame_feature_mode == "multi_layer_cls_patch_attn"
            and not self.spatial_token_pooling_patch_layers
            and self.spatial_attention_patch_mode != "residual_multilayer"
        )
        use_trainable_block_checkpoint = (
            self.training
            and self.gradient_checkpointing
            and self.gradient_checkpointing_mode == "trainable_blocks"
            and has_trainable_params
            and supports_trainable_block_checkpoint
        )

        def extract(flat_inputs: Tensor) -> tuple[Tensor, Tensor]:
            if self.spatial_token_pooling_patch_layers:
                return self._extract_spatial_token_multilayer_patches(
                    selected_backbone, flat_inputs
                )
            if self.frame_feature_mode == "residual_multilayer_cls_patch_attn":
                return self._extract_residual_multilayer_features_and_patches(
                    selected_backbone, flat_inputs
                )
            if self.spatial_attention_patch_mode == "residual_multilayer":
                if self.frame_feature_mode != "last_cls_patch_mean":
                    raise ValueError(
                        "separate multi-layer spatial patches require "
                        "model.frame_feature_mode=last_cls_patch_mean"
                    )
                return self._extract_spatial_multilayer_patches(
                    selected_backbone, flat_inputs
                )
            return extract_dino_frame_features_and_patches(
                selected_backbone,
                flat_inputs,
                feature_mode=self.frame_feature_mode,
                feature_layers=_resolved_layers,
                patch_pool=self.frame_patch_attn if self.frame_patch_pool == "attn" else None,
                checkpoint_trainable_blocks=use_trainable_block_checkpoint,
            )

        def extract_chunk(flat_inputs: Tensor) -> tuple[Tensor, Tensor]:
            if not (has_trainable_params or has_trainable_postprocess):
                with torch.no_grad():
                    return extract(flat_inputs)
            use_outer_checkpoint = (
                self.training
                and self.gradient_checkpointing
                and has_trainable_params
                and not use_trainable_block_checkpoint
            )
            if use_outer_checkpoint:
                return activation_checkpoint(
                    extract,
                    flat_inputs,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            return extract(flat_inputs)

        chunk_size = self.backbone_frame_chunk_size
        if chunk_size > 0 and flat.shape[0] > chunk_size:
            chunk_outputs = [
                extract_chunk(chunk) for chunk in flat.split(chunk_size, dim=0)
            ]
            features = torch.cat([output[0] for output in chunk_outputs], dim=0)
            patches = torch.cat([output[1] for output in chunk_outputs], dim=0)
        else:
            features, patches = extract_chunk(flat)
        return (
            features.reshape(bsz, frames, -1),
            patches.reshape(bsz, frames, patches.shape[1], patches.shape[2]),
        )

    def _selector_soft_importance(
        self,
        frame_event_logits: Tensor,
        valid_class_indices: Sequence[int],
    ) -> Tensor:
        frames = frame_event_logits.shape[1]
        temperature = self.event_topk_temperature
        if valid_class_indices:
            selector_logits = frame_event_logits[..., list(valid_class_indices)]
            per_class_weights = torch.softmax(selector_logits / temperature, dim=1)
            return per_class_weights.mean(dim=-1) * float(frames)
        selector_logits = torch.logsumexp(frame_event_logits, dim=-1)
        return torch.softmax(selector_logits / temperature, dim=1) * float(frames)

    def _response_curve_clip_logits(self, frame_event_logits: Tensor) -> Tensor:
        if frame_event_logits.ndim != 3:
            raise ValueError(
                "frame_event_logits must be [batch, frames, labels] for "
                "response-curve primary pooling"
            )
        frames = frame_event_logits.shape[1]
        if frames <= 0:
            raise ValueError("response-curve primary pooling requires frames > 0")
        if self.response_curve_pooling == "max":
            pooled = frame_event_logits.amax(dim=1)
        else:
            topk = min(self.response_curve_topk, frames)
            values = torch.topk(frame_event_logits, k=topk, dim=1).values
            pooled = values.logsumexp(dim=1) - math.log(float(topk))
        scale = self.response_curve_logit_scale.to(
            device=pooled.device, dtype=pooled.dtype
        )
        bias = self.response_curve_logit_bias.to(
            device=pooled.device, dtype=pooled.dtype
        )
        return pooled * scale.reshape(1, -1) + bias.reshape(1, -1)

    def _apply_response_curve_primary(
        self, logits: Tensor, frame_event_logits: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        # Always expose the pooled dense-response clip score for diagnostics and
        # offline fusion.  response_curve_primary only controls whether that
        # score replaces/blends into the main training logits.
        response_logits = self._response_curve_clip_logits(frame_event_logits)
        if not self.response_curve_primary_enabled:
            return logits, response_logits
        blend = self.response_curve_blend_weight
        if blend >= 1.0:
            return response_logits, response_logits
        if blend <= 0.0:
            return logits, response_logits
        return logits + blend * (response_logits - logits), response_logits

    def _select_event_topk(self, x: Tensor, frame_event_logits: Tensor) -> tuple[Tensor, Tensor]:
        bsz, frames, hidden = x.shape
        if frames <= 0:
            return x, x.new_zeros((bsz, 0), dtype=torch.long)
        event_k = min(max(self.event_topk, 0), frames)
        context_k = min(max(self.context_frames, 0), frames)
        desired = min(frames, max(event_k + context_k, 1))
        class_probabilities = torch.sigmoid(frame_event_logits.detach())
        if self.event_topk_strategy == "class_union":
            valid_class_indices = [
                index
                for index in self.event_topk_class_indices
                if 0 <= index < class_probabilities.shape[-1]
            ]
            if not valid_class_indices:
                raise ValueError("event_topk_class_indices has no valid class indices")
            scores = class_probabilities[..., valid_class_indices].amax(dim=-1)
        else:
            valid_class_indices = []
            scores = class_probabilities.amax(dim=-1)
        context_indices = []
        if context_k > 0:
            context_indices = torch.linspace(0, frames - 1, context_k, device=x.device).round().long().tolist()
        all_indices = []
        for batch_index in range(bsz):
            selected: list[int] = []
            if event_k > 0:
                if self.event_topk_strategy == "class_union":
                    per_class_k = min(self.event_topk_per_class, frames)
                    for class_index in valid_class_indices:
                        selected.extend(
                            torch.topk(
                                class_probabilities[batch_index, :, class_index],
                                k=per_class_k,
                            ).indices.tolist()
                        )
                else:
                    selected.extend(
                        torch.topk(scores[batch_index], k=event_k).indices.tolist()
                    )
            selected.extend(int(index) for index in context_indices)
            deduped: list[int] = []
            seen: set[int] = set()
            for index in selected:
                index = int(index)
                if index not in seen:
                    seen.add(index)
                    deduped.append(index)
            if len(deduped) < desired:
                for index in torch.argsort(scores[batch_index], descending=True).tolist():
                    index = int(index)
                    if index in seen:
                        continue
                    seen.add(index)
                    deduped.append(index)
                    if len(deduped) >= desired:
                        break
            deduped = sorted(deduped[:desired])
            all_indices.append(torch.tensor(deduped, dtype=torch.long, device=x.device))
        indices = torch.stack(all_indices, dim=0)
        selected_x = x.gather(1, indices.unsqueeze(-1).expand(-1, -1, hidden))
        if self.event_topk_gradient == "straight_through" and self.event_topk_gradient_scale > 0:
            soft_importance = self._selector_soft_importance(
                frame_event_logits, valid_class_indices
            )
            selected_importance = soft_importance.gather(1, indices)
            gate = 1.0 + self.event_topk_gradient_scale * (
                selected_importance - selected_importance.detach()
            )
            selected_x = selected_x * gate.unsqueeze(-1)
        return selected_x, indices

    def _select_event_anchor_frames(self, x: Tensor, frame_event_logits: Tensor) -> tuple[Tensor, Tensor]:
        bsz, frames, hidden = x.shape
        if frames <= 0:
            return x, x.new_zeros((bsz, 0), dtype=torch.long)
        context_k = min(max(self.context_frames, 0), frames)
        desired = min(frames, max(self.event_anchor_max_frames, 1))
        class_probabilities = torch.sigmoid(frame_event_logits.detach())
        valid_class_indices = [
            index
            for index in self.event_anchor_class_indices
            if 0 <= index < class_probabilities.shape[-1]
            and index < len(self.event_anchor_topk_per_class)
            and self.event_anchor_topk_per_class[index] > 0
        ]
        if not valid_class_indices:
            raise ValueError("event_anchor_class_indices has no valid class indices with positive topk")
        scores = class_probabilities[..., valid_class_indices].amax(dim=-1)
        context_indices: list[int] = []
        if context_k > 0:
            context_indices = [
                int(index)
                for index in torch.linspace(0, frames - 1, context_k, device=x.device).round().long().tolist()
            ]

        all_indices = []
        center_offsets = [offset for offset in self.event_anchor_offsets if offset == 0]
        if not center_offsets:
            center_offsets = [0]
        neighbor_offsets = [
            offset
            for offset in sorted(self.event_anchor_offsets, key=lambda value: (abs(value), value))
            if offset != 0
        ]
        for batch_index in range(bsz):
            selected_by_priority: list[int] = []
            anchors_by_priority: list[int] = []
            for class_index in valid_class_indices:
                anchors: list[int] = []
                class_scores = class_probabilities[batch_index, :, class_index]
                max_anchors = min(self.event_anchor_topk_per_class[class_index], frames)
                for candidate in torch.argsort(class_scores, descending=True).tolist():
                    candidate = int(candidate)
                    if any(abs(candidate - anchor) <= self.event_anchor_nms_radius for anchor in anchors):
                        continue
                    anchors.append(candidate)
                    if len(anchors) >= max_anchors:
                        break
                anchors_by_priority.extend(anchors)

            for anchor in anchors_by_priority:
                for offset in center_offsets:
                    selected_by_priority.append(min(max(anchor + offset, 0), frames - 1))
            selected_by_priority.extend(context_indices)
            for offset in neighbor_offsets:
                for anchor in anchors_by_priority:
                    selected_by_priority.append(min(max(anchor + offset, 0), frames - 1))
            deduped: list[int] = []
            seen: set[int] = set()
            for index in selected_by_priority:
                index = int(index)
                if index in seen:
                    continue
                seen.add(index)
                deduped.append(index)
                if len(deduped) >= desired:
                    break
            if len(deduped) < desired:
                for index in torch.argsort(scores[batch_index], descending=True).tolist():
                    index = int(index)
                    if index in seen:
                        continue
                    seen.add(index)
                    deduped.append(index)
                    if len(deduped) >= desired:
                        break
            deduped = sorted(deduped[:desired])
            all_indices.append(torch.tensor(deduped, dtype=torch.long, device=x.device))
        indices = torch.stack(all_indices, dim=0)
        selected_x = x.gather(1, indices.unsqueeze(-1).expand(-1, -1, hidden))
        if self.event_topk_gradient == "straight_through" and self.event_topk_gradient_scale > 0:
            soft_importance = self._selector_soft_importance(
                frame_event_logits, valid_class_indices
            )
            selected_importance = soft_importance.gather(1, indices)
            gate = 1.0 + self.event_topk_gradient_scale * (
                selected_importance - selected_importance.detach()
            )
            selected_x = selected_x * gate.unsqueeze(-1)
        return selected_x, indices

    def _select_uniform_frames(self, x: Tensor) -> tuple[Tensor, Tensor]:
        bsz, frames, hidden = x.shape
        if frames <= 0:
            return x, x.new_zeros((bsz, 0), dtype=torch.long)
        desired = min(self.uniform_frames, frames)
        indices = torch.linspace(
            0, frames - 1, desired, device=x.device
        ).round().long()
        indices = indices.unsqueeze(0).expand(bsz, -1)
        selected_x = x.gather(
            1, indices.unsqueeze(-1).expand(-1, -1, hidden)
        )
        return selected_x, indices

    def _projected_branch_outputs(
        self,
        x: Tensor,
        frame_event_head: nn.Module,
        temporal: nn.Module,
        head: nn.Module,
    ) -> dict[str, Tensor]:
        dense_frame_tokens = x
        frame_event_logits = frame_event_head(dense_frame_tokens)
        response_curve_logits: Tensor | None = None
        if self.response_curve_head is not None:
            response_curve_logits = self.response_curve_head(dense_frame_tokens)
        response_curve_pool_logits = (
            response_curve_logits
            if response_curve_logits is not None
            else frame_event_logits
        )
        difference_residual: Tensor | None = None
        difference_gate: Tensor | None = None
        if self.temporal_difference_adapter is not None:
            x, difference_residual, difference_gate = (
                self.temporal_difference_adapter(x)
            )
        frame_attention: Tensor | None = None
        topk_indices: Tensor | None = None
        if self.fusion == "mean":
            temporal_features = temporal(x.mean(dim=1))
            logits = head(temporal_features)
        elif self.fusion == "event_topk_transformer":
            selected_x, topk_indices = self._select_event_topk(x, frame_event_logits)
            temporal_features = temporal(
                selected_x,
                position_indices=topk_indices,
            )
            logits = head(temporal_features)
        elif self.fusion == "event_anchor_transformer":
            selected_x, topk_indices = self._select_event_anchor_frames(x, frame_event_logits)
            temporal_features = temporal(
                selected_x,
                position_indices=topk_indices,
            )
            logits = head(temporal_features)
        elif self.fusion == "uniform_event_dual_transformer":
            selected_x, topk_indices = self._select_event_topk(
                x, frame_event_logits
            )
            event_temporal = temporal(
                selected_x, position_indices=topk_indices
            )
            event_logits = head(event_temporal)
            uniform_x, uniform_indices = self._select_uniform_frames(x)
            uniform_temporal = self.uniform_temporal(
                uniform_x, position_indices=uniform_indices
            )
            uniform_logits = self.uniform_head(uniform_temporal)
            gate_logits = self.uniform_event_gate_logits.unsqueeze(0)
            if self.uniform_event_gate_adapter is not None:
                frame_evidence = torch.sigmoid(frame_event_logits).amax(dim=1)
                gate_features = torch.cat(
                    [
                        uniform_logits.detach(),
                        event_logits.detach(),
                        frame_evidence.detach(),
                    ],
                    dim=-1,
                )
                gate_logits = gate_logits + self.uniform_event_gate_adapter(
                    gate_features
                )
            temporal_gate = torch.sigmoid(gate_logits)
            logits = uniform_logits + temporal_gate * (
                event_logits - uniform_logits
            )
            temporal_features = 0.5 * (
                uniform_temporal + event_temporal
            )
        elif self.fusion == "temporal_stem_bottleneck_transformer":
            temporal_outputs = temporal(x)
            if not isinstance(temporal_outputs, dict):
                raise RuntimeError("temporal_stem_bottleneck_transformer must return a dict")
            temporal_features = temporal_outputs["clip_token"]
            dense_frame_tokens = temporal_outputs["frame_tokens"]
            frame_event_logits = frame_event_head(dense_frame_tokens)
            response_curve_logits = (
                self.response_curve_head(dense_frame_tokens)
                if self.response_curve_head is not None
                else None
            )
            response_curve_pool_logits = (
                response_curve_logits
                if response_curve_logits is not None
                else frame_event_logits
            )
            logits = head(temporal_features)
        elif self.fusion == "attn_pool_transformer":
            temporal_features, frame_attention = temporal(x)
            logits = head(temporal_features)
        elif self.fusion == "class_query_transformer":
            temporal_features, logits, frame_attention = temporal(x)
        else:
            temporal_features = temporal(x)
            logits = head(temporal_features)
        temporal_logits = logits
        logits, response_clip_logits = self._apply_response_curve_primary(
            logits, response_curve_pool_logits
        )
        result: dict[str, Tensor] = {
            "frame_tokens": dense_frame_tokens,
            "temporal": temporal_features,
            "logits": logits,
            "temporal_logits": temporal_logits,
            "frame_event_logits": frame_event_logits,
        }
        if response_curve_logits is not None:
            result["response_curve_logits"] = response_curve_logits
        if response_clip_logits is not None:
            result["response_clip_logits"] = response_clip_logits
        if difference_residual is not None and difference_gate is not None:
            result["temporal_difference_gate"] = difference_gate.reshape(1).expand(
                x.shape[0]
            )
            result["temporal_difference_residual_abs_mean"] = (
                difference_residual.abs().mean(dim=(1, 2))
            )
            result["temporal_difference_residual_norm"] = difference_residual.norm(
                dim=-1
            ).mean(dim=1)
        if frame_attention is not None:
            result["frame_attention"] = frame_attention
        if topk_indices is not None:
            result["topk_indices"] = topk_indices
        if self.fusion == "uniform_event_dual_transformer":
            result.update(
                {
                    "uniform_logits": uniform_logits,
                    "event_logits": event_logits,
                    "uniform_indices": uniform_indices,
                    "temporal_gate": temporal_gate.expand(logits.shape[0], -1),
                    "retention_reference_logits": uniform_logits,
                }
            )
        return result

    def _global_branch_outputs(self, features: Tensor) -> dict[str, Tensor]:
        if self.fusion == "videomae_residual_transformer":
            modules = (
                self.video_global_proj,
                self.video_temporal_delta_head,
            )
            if any(module is None for module in modules) or (
                self.video_temporal_gate_logits is None
            ):
                raise RuntimeError("VideoMAE residual temporal modules are missing")
            expected_dim = self.video_global_feature_dim + self.base_frame_feature_dim
            if features.shape[-1] != expected_dim:
                raise ValueError(
                    f"VideoMAE feature dim={features.shape[-1]} must be {expected_dim}"
                )
            global_feature = features[:, 0, : self.video_global_feature_dim]
            tubelet_features = features[..., self.video_global_feature_dim :]
            frame_tokens = self.frame_proj(tubelet_features)
            frame_event_logits = self.frame_event_head(frame_tokens)
            temporal_features = self.temporal(frame_tokens)
            global_action_features = self.video_global_proj(global_feature)
            global_action_logits = self.head(global_action_features)
            temporal_delta_logits = self.video_temporal_delta_head(temporal_features)
            temporal_residual_gate = torch.sigmoid(
                self.video_temporal_gate_logits
            ).unsqueeze(0).expand_as(temporal_delta_logits)
            logits = global_action_logits + temporal_residual_gate * temporal_delta_logits
            return {
                "frame_tokens": frame_tokens,
                "temporal": temporal_features,
                "logits": logits,
                "frame_event_logits": frame_event_logits,
                "global_action_logits": global_action_logits,
                "temporal_delta_logits": temporal_delta_logits,
                "temporal_residual_gate": temporal_residual_gate,
            }
        if self.frame_feature_mode != "residual_multilayer_cls_patch_attn":
            return self._branch_outputs(
                features,
                self.frame_proj,
                self.frame_event_head,
                self.temporal,
                self.head,
            )
        if self.multi_layer_frame_adapter is None:
            raise RuntimeError("Residual multi-layer frame adapter is missing")
        expected_dim = self.base_frame_feature_dim + self.multi_layer_feature_dim
        if features.shape[-1] != expected_dim:
            raise ValueError(
                f"Residual multi-layer feature dim={features.shape[-1]} "
                f"does not match expected={expected_dim}"
            )
        base_features = features[..., : self.base_frame_feature_dim]
        multi_layer_features = features[..., self.base_frame_feature_dim :]
        base_tokens = self.frame_proj(base_features)
        residual_tokens = self.multi_layer_frame_adapter(multi_layer_features)
        outputs = self._projected_branch_outputs(
            base_tokens + residual_tokens,
            self.frame_event_head,
            self.temporal,
            self.head,
        )
        outputs["multi_layer_frame_residual"] = residual_tokens
        return outputs

    def _global_spatial_token_outputs(
        self,
        features: Tensor,
        patch_tokens: Tensor,
    ) -> dict[str, Tensor]:
        if self.spatial_token_pool is None:
            raise RuntimeError("Spatial token pool is not initialized")
        global_frame_tokens = self.frame_proj(features)
        pooled = self.spatial_token_pool(global_frame_tokens, patch_tokens)
        bsz, frames, _hidden_dim = global_frame_tokens.shape

        if self.spatial_token_pooling_fusion_mode in {
            "joint_token_temporal",
            "slot_only_temporal",
            "joint_token_temporal_dropout",
        }:
            if self.spatial_token_joint_temporal is None:
                raise RuntimeError("Joint token temporal fusion is not initialized")
            joint = self.spatial_token_joint_temporal(
                global_frame_tokens, pooled["temporal_local_tokens"]
            )
            encoded_global = joint["global_frame_tokens"]
            local_summary = joint["local_frame_tokens"].mean(dim=2)
            result = {
                "frame_tokens": encoded_global,
                "global_frame_tokens": global_frame_tokens,
                "temporal": joint["event"],
                "logits": self.head(joint["event"]),
                "frame_event_logits": self.frame_event_head(encoded_global),
                "spatial_token_joint_global_norm": (
                    global_frame_tokens.norm(dim=-1).mean(dim=1)
                ),
                "spatial_token_joint_local_norm": (
                    local_summary.norm(dim=-1).mean(dim=1)
                ),
            }
        else:
            # Preserve the original local implementation as the default path.
            token_grid = pooled["sequence"]
            _, _, slots, hidden_dim = token_grid.shape
            temporal_tokens = token_grid.reshape(
                bsz, frames * slots, hidden_dim
            )
            position_indices = (
                torch.arange(frames, device=features.device)
                .reshape(1, frames, 1)
                .expand(bsz, frames, slots)
                .reshape(bsz, frames * slots)
            )
            temporal_features = self.temporal(
                temporal_tokens,
                position_indices=position_indices,
            )
            local_summary = pooled["gated_local_tokens"].mean(dim=2)
            frame_tokens = global_frame_tokens + local_summary
            result = {
                "frame_tokens": frame_tokens,
                "global_frame_tokens": global_frame_tokens,
                "temporal": temporal_features,
                "logits": self.head(temporal_features),
                "frame_event_logits": self.frame_event_head(frame_tokens),
            }

        result.update(
            {
                "spatial_token_local_summary_norm": (
                    local_summary.norm(dim=-1).mean(dim=1)
                ),
                "spatial_token_slot_logits": pooled["slot_logits"],
                "spatial_token_attention_entropy": pooled[
                    "attention_entropy"
                ],
                "spatial_token_attention_overlap": pooled[
                    "attention_overlap"
                ],
                "spatial_token_query_diversity_loss": pooled[
                    "query_diversity_loss"
                ],
                "spatial_token_slot_gate": pooled["slot_gate"],
                "spatial_token_context_scale": pooled["context_scale"],
            }
        )
        if "attention_maps" in pooled:
            result["spatial_token_attention_maps"] = pooled["attention_maps"]
        return result


    def _branch_outputs(
        self,
        features: Tensor,
        frame_proj: nn.Module,
        frame_event_head: nn.Module,
        temporal: nn.Module,
        head: nn.Module,
    ) -> dict[str, Tensor]:
        return self._projected_branch_outputs(
            frame_proj(features), frame_event_head, temporal, head
        )

    def _adaptive_spatial_fusion(
        self,
        global_logits: Tensor,
        spatial_outputs: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor]:
        if (
            self.spatial_fusion_class_embedding is None
            or self.spatial_fusion_delta is None
            or self.spatial_fusion_gate is None
        ):
            raise RuntimeError("Adaptive spatial fusion modules are missing")
        spatial_logits = spatial_outputs["clip_logits"]
        frame_probabilities = torch.sigmoid(
            spatial_outputs["frame_event_logits"]
        )
        topk = min(3, frame_probabilities.shape[1])
        frame_evidence = torch.topk(
            frame_probabilities, k=topk, dim=1
        ).values.mean(dim=1)
        entropy_quality = 1.0 - spatial_outputs[
            "attention_entropy"
        ].mean(dim=(1, 3))
        scalar_features = torch.stack(
            [
                global_logits,
                spatial_logits,
                spatial_logits - global_logits,
                frame_evidence,
                entropy_quality,
            ],
            dim=-1,
        )
        class_features = self.spatial_fusion_class_embedding.unsqueeze(0).expand(
            global_logits.shape[0], -1, -1
        )
        fusion_features = torch.cat(
            [scalar_features, class_features], dim=-1
        )
        delta = self.spatial_fusion_delta(fusion_features).squeeze(-1)
        gate = torch.sigmoid(
            self.spatial_fusion_gate(fusion_features).squeeze(-1)
        )
        return global_logits + gate * delta, delta, gate

    def _conditioned_spatial_residual(
        self,
        global_logits: Tensor,
        global_temporal: Tensor,
        spatial_outputs: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        modules = (
            self.spatial_conditioned_global_proj,
            self.spatial_conditioned_local_proj,
            self.spatial_conditioned_delta,
            self.spatial_conditioned_gate,
        )
        if any(module is None for module in modules) or (
            self.spatial_conditioned_class_embedding is None
        ):
            raise RuntimeError("Conditioned spatial residual modules are missing")
        spatial_features = spatial_outputs["class_features"]
        if global_temporal.ndim == 2:
            global_features = global_temporal.unsqueeze(1).expand(
                -1, self.num_labels, -1
            )
        elif global_temporal.ndim == 3:
            if global_temporal.shape[1] != self.num_labels:
                raise ValueError(
                    "class-aware global temporal features must have one token per label"
                )
            global_features = global_temporal
        else:
            raise ValueError(
                f"unsupported global temporal shape: {tuple(global_temporal.shape)}"
            )
        global_features = global_features.detach()
        reference_logits = global_logits.detach()
        global_projected = self.spatial_conditioned_global_proj(global_features)
        local_projected = self.spatial_conditioned_local_proj(spatial_features)
        frame_probabilities = torch.sigmoid(
            spatial_outputs["frame_event_logits"]
        )
        topk = min(3, frame_probabilities.shape[1])
        frame_evidence = torch.topk(
            frame_probabilities, k=topk, dim=1
        ).values.mean(dim=1)
        entropy_quality = 1.0 - spatial_outputs[
            "attention_entropy"
        ].mean(dim=(1, 3))
        overlap_quality = 1.0 - spatial_outputs[
            "attention_overlap"
        ].mean(dim=1)
        scalar_features = torch.stack(
            [
                reference_logits,
                torch.sigmoid(reference_logits),
                spatial_outputs["raw_residual_logits"],
                spatial_outputs["residual_logits"],
                frame_evidence,
                entropy_quality,
                spatial_outputs["clip_gate"],
                overlap_quality,
            ],
            dim=-1,
        )
        class_features = self.spatial_conditioned_class_embedding.unsqueeze(0).expand(
            global_logits.shape[0], -1, -1
        )
        correction_features = torch.cat(
            [
                global_projected,
                local_projected,
                global_projected * local_projected,
                (global_projected - local_projected).abs(),
                scalar_features,
                class_features,
            ],
            dim=-1,
        )
        bounded_delta = self.spatial_conditioned_max_delta * torch.tanh(
            self.spatial_conditioned_delta(correction_features).squeeze(-1)
        )
        gate = torch.sigmoid(
            self.spatial_conditioned_gate(correction_features).squeeze(-1)
        )
        correction = gate * bounded_delta
        return reference_logits + correction, correction, bounded_delta, gate

    def _adaspot_spatial_feature_fusion(
        self,
        global_outputs: dict[str, Tensor],
        spatial_outputs: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Fuse class-aware ROI evidence before the global temporal model."""
        modules = (
            self.spatial_feature_global_align,
            self.spatial_feature_local_align,
            self.spatial_feature_output,
        )
        if any(module is None for module in modules):
            raise RuntimeError("AdaSpot feature fusion modules are missing")
        local_tokens = spatial_outputs.get("roi_frame_features")
        if local_tokens is None:
            raise ValueError("AdaSpot feature fusion requires ROI frame features")
        global_tokens = global_outputs["frame_tokens"].detach()
        bsz, frames, hidden_dim = global_tokens.shape
        expected = (bsz, frames, self.num_labels, hidden_dim)
        if tuple(local_tokens.shape) != expected:
            raise ValueError(
                f"ROI frame feature shape={tuple(local_tokens.shape)} must be {expected}"
            )
        expanded_global = global_tokens.unsqueeze(2).expand(
            -1, -1, self.num_labels, -1
        )
        global_aligned = self.spatial_feature_global_align(expanded_global)
        local_aligned = self.spatial_feature_local_align(local_tokens)
        competitive_detail = torch.maximum(
            global_aligned, local_aligned
        ) - global_aligned
        feature_delta = self.spatial_feature_output(competitive_detail)
        fused_tokens = expanded_global + feature_delta

        reference_tokens = expanded_global.permute(0, 2, 1, 3).reshape(
            bsz * self.num_labels, frames, hidden_dim
        )
        fused_flat_tokens = fused_tokens.permute(0, 2, 1, 3).reshape(
            bsz * self.num_labels, frames, hidden_dim
        )
        paired_tokens = torch.cat([reference_tokens, fused_flat_tokens], dim=0)
        paired_temporal = self.temporal(paired_tokens)
        paired_logits = self.head(paired_temporal).reshape(
            2, bsz, self.num_labels, self.num_labels
        )
        class_index = torch.arange(self.num_labels, device=paired_logits.device)
        reference_logits = paired_logits[0].diagonal(dim1=1, dim2=2)
        fused_logits = paired_logits[1].diagonal(dim1=1, dim2=2)
        logits = global_outputs["logits"].detach() + (
            fused_logits - reference_logits
        )

        paired_frame_logits = self.frame_event_head(paired_tokens).reshape(
            2, bsz, self.num_labels, frames, self.num_labels
        )
        reference_frame = paired_frame_logits[0].diagonal(dim1=1, dim2=3)
        fused_frame = paired_frame_logits[1].diagonal(dim1=1, dim2=3)
        fused_frame_logits = global_outputs["frame_event_logits"].detach() + (
            fused_frame - reference_frame
        )
        return (
            logits,
            fused_frame_logits,
            feature_delta,
            logits - global_outputs["logits"].detach(),
        )


    def forward(
        self,
        inputs: Tensor,
        *,
        roi_inputs: Tensor | None = None,
        roi_inputs_b: Tensor | None = None,
        roi_meta: Tensor | None = None,
        roi_valid: Tensor | None = None,
        roi_frame_meta: Tensor | None = None,
        roi_frame_valid: Tensor | None = None,
        roi_meta_b: Tensor | None = None,
        roi_valid_b: Tensor | None = None,
        roi_frame_meta_b: Tensor | None = None,
        roi_frame_valid_b: Tensor | None = None,
        global_frame_times: Tensor | None = None,
        local_frame_times: Tensor | None = None,
        local_frame_times_b: Tensor | None = None,
        evidence_features: Tensor | None = None,
        highres_pool_inputs: Tensor | None = None,
        highres_pool_times: Tensor | None = None,
        clip_targets: Tensor | None = None,
        clip_label_masks: Tensor | None = None,
        return_aux: bool = False,
    ) -> Tensor | dict[str, Tensor]:
        global_patch_tokens: Tensor | None = None
        if (
            self.object_spatial_aux_enabled
            or self.spatial_attention_enabled
            or self.spatial_token_pooling_enabled
            or self.class_evidence_enabled
            or self.highres_glimpse_enabled
        ):
            global_features, global_patch_tokens = (
                self.encode_frames_with_patch_tokens(inputs)
            )
        else:
            global_features = self.encode_frames(inputs)
        if (
            self.class_evidence_enabled
            and self.class_evidence_mode == "independent_preprojection"
        ):
            if not isinstance(
                self.class_evidence_head, IndependentPreprojectionEvidenceHead
            ) or global_patch_tokens is None:
                raise RuntimeError(
                    "independent preprojection evidence head or patch tokens are missing"
                )
            independent = self.class_evidence_head(
                global_features, global_patch_tokens,
                evidence_features=evidence_features,
            )
            if not return_aux:
                return independent["logits"]
            frame_logits = independent["frame_event_logits"]
            result = {
                "logits": independent["logits"],
                "frame_event_logits": frame_logits,
                "response_curve_logits": frame_logits,
                "response_clip_logits": self._response_curve_clip_logits(
                    frame_logits
                ),
                "class_evidence_frame_tokens": independent[
                    "class_frame_tokens"
                ],
                "class_evidence_clip_tokens": independent[
                    "class_clip_tokens"
                ],
                "class_evidence_no_evidence_mass": independent[
                    "no_evidence_mass"
                ],
                "class_evidence_attention_entropy": independent[
                    "attention_entropy"
                ],
                "class_evidence_patch_appearance": independent[
                    "patch_appearance"
                ],
            }
            if "external_evidence_gate_values" in independent:
                result["external_evidence_gate_values"] = independent[
                    "external_evidence_gate_values"
                ]
            if "counterfactual_logits" in independent:
                result["class_evidence_counterfactual_logits"] = independent[
                    "counterfactual_logits"
                ]
                result["class_evidence_causal_full_logits"] = independent[
                    "causal_full_logits"
                ]
            if "no_evidence_content_logits" in independent:
                result["class_evidence_no_evidence_content_logits"] = independent[
                    "no_evidence_content_logits"
                ]
                result["class_evidence_no_evidence_saliency"] = independent[
                    "no_evidence_saliency"
                ]
            if self.class_evidence_return_maps:
                result["class_evidence_attention_maps"] = independent[
                    "attention_maps"
                ]
            if "set_piece_subtype_logits" in independent:
                result["set_piece_subtype_logits"] = independent[
                    "set_piece_subtype_logits"
                ]
            return result

        if self.spatial_token_pooling_enabled:
            if global_patch_tokens is None:
                raise RuntimeError(
                    "Spatial token pooling was enabled but patch tokens are missing"
                )
            global_outputs = self._global_spatial_token_outputs(
                global_features, global_patch_tokens
            )
        else:
            global_outputs = self._global_branch_outputs(global_features)
        global_temporal = global_outputs["temporal"]
        global_logits = global_outputs["logits"]
        spatial_outputs: dict[str, Tensor] | None = None
        logits = global_logits
        spatial_fused_logits = global_logits
        spatial_fusion_delta: Tensor | None = None
        spatial_fusion_gate: Tensor | None = None
        spatial_conditioned_raw_delta: Tensor | None = None
        spatial_feature_delta: Tensor | None = None
        spatial_feature_frame_logits: Tensor | None = None
        class_evidence_outputs: dict[str, Tensor] | None = None
        object_spatial_outputs: dict[str, Tensor] | None = None
        highres_outputs: dict[str, Tensor] | None = None
        if self.highres_glimpse_enabled:
            if global_patch_tokens is None or highres_pool_inputs is None:
                raise ValueError(
                    "highres glimpse mode requires DINO patch tokens and highres_pool_inputs"
                )
            highres_outputs = self._highres_glimpse_outputs(
                global_outputs,
                global_patch_tokens,
                highres_pool_inputs,
                global_frame_times=global_frame_times,
                highres_pool_times=highres_pool_times,
                clip_targets=clip_targets,
                clip_label_masks=clip_label_masks,
            )
            logits = highres_outputs["logits"]
            spatial_fused_logits = logits
        if self.spatial_attention_enabled:
            if self.spatial_attention is None or global_patch_tokens is None:
                raise RuntimeError("Spatial attention was enabled but patch tokens are missing")
            spatial_outputs = self.spatial_attention(
                global_outputs["frame_tokens"],
                global_patch_tokens,
                global_outputs.get("frame_event_logits"),
            )
            legacy_fused_logits = global_logits + spatial_outputs["residual_logits"]
            if self.spatial_attention_mode == "probe":
                logits = spatial_outputs["clip_logits"]
                spatial_fused_logits = legacy_fused_logits
            elif self.spatial_attention_mode == "conditioned_residual":
                (
                    logits,
                    spatial_fusion_delta,
                    spatial_conditioned_raw_delta,
                    spatial_fusion_gate,
                ) = self._conditioned_spatial_residual(
                    global_logits, global_temporal, spatial_outputs
                )
                spatial_fused_logits = logits
            elif self.spatial_attention_mode == "adaptive_fusion":
                logits, spatial_fusion_delta, spatial_fusion_gate = (
                    self._adaptive_spatial_fusion(global_logits, spatial_outputs)
                )
                spatial_fused_logits = logits
            elif self.spatial_attention_mode == "adaspot_feature_fusion":
                (
                    logits,
                    spatial_feature_frame_logits,
                    spatial_feature_delta,
                    spatial_fusion_delta,
                ) = self._adaspot_spatial_feature_fusion(
                    global_outputs, spatial_outputs
                )
                spatial_fused_logits = logits
            else:
                logits = legacy_fused_logits
                spatial_fused_logits = logits
        if self.class_evidence_enabled:
            if self.class_evidence_head is None or global_patch_tokens is None:
                raise RuntimeError("Class evidence was enabled but patch tokens are missing")
            if self.class_evidence_mode == "global_local_temporal":
                if not isinstance(
                    self.class_evidence_head,
                    GlobalLocalEvidenceTemporalHead,
                ):
                    raise RuntimeError(
                        "global_local_temporal evidence head was not initialized"
                    )
                class_evidence_outputs = self.class_evidence_head(
                    global_outputs["frame_tokens"], global_patch_tokens
                )
                logits = class_evidence_outputs["logits"]
            else:
                if not isinstance(
                    self.class_evidence_head, LightweightClassEvidenceHead
                ):
                    raise RuntimeError("residual class evidence head was not initialized")
                class_evidence_outputs = self.class_evidence_head(
                    global_outputs["frame_tokens"],
                    global_patch_tokens,
                    global_outputs["frame_event_logits"],
                )
                spatial_fusion_delta = class_evidence_outputs["correction"]
                logits = global_logits + spatial_fusion_delta
            spatial_fused_logits = logits
        if self.object_spatial_aux_enabled:
            if self.object_spatial_aux is None or global_patch_tokens is None:
                raise RuntimeError(
                    "object spatial aux was enabled but its head or patch tokens are missing"
                )
            if self.object_spatial_aux_representation_only:
                object_spatial_outputs = self.object_spatial_aux.predict_representation_aux(
                    global_outputs["frame_tokens"], global_patch_tokens
                )
            else:
                object_spatial_outputs = self.object_spatial_aux(global_patch_tokens)
                logits = logits + object_spatial_outputs["residual"]
                spatial_fused_logits = logits
        if not self.dual_view:
            if return_aux:
                result = {
                    "logits": logits,
                    "global_logits": global_logits,
                    "global_temporal": global_temporal,
                    "global_frame_tokens": global_outputs["frame_tokens"],
                    "frame_event_logits": (
                        global_outputs["frame_event_logits"]
                        if highres_outputs is not None
                        else spatial_feature_frame_logits
                        if spatial_feature_frame_logits is not None
                        else global_outputs["frame_event_logits"]
                    ),
                }
                if object_spatial_outputs is not None:
                    result["object_heatmap_logits"] = object_spatial_outputs[
                        "heatmap_logits"
                    ]
                    relation_output_names = {
                        "relation_logits": "ball_goal_relation_logits",
                        "context_predictions": "ball_context_predictions",
                        "context_targets": "ball_context_targets",
                        "candidate_entropy": "ball_candidate_entropy",
                        "candidate_indices": "ball_candidate_indices",
                        "candidate_weights": "ball_candidate_weights",
                    }
                    for source_name, output_name in relation_output_names.items():
                        if source_name in object_spatial_outputs:
                            result[output_name] = object_spatial_outputs[
                                source_name
                            ]
                    if not self.object_spatial_aux_representation_only:
                        result.update({
                            "object_attention_maps": object_spatial_outputs["attention_maps"],
                            "object_tokens": object_spatial_outputs["object_tokens"],
                            "object_class_attention": object_spatial_outputs["class_attention"],
                            "object_spatial_raw_residual": object_spatial_outputs["raw_residual"],
                            "object_spatial_residual": object_spatial_outputs["residual"],
                            "retention_reference_logits": global_logits,
                        })
                if highres_outputs is not None:
                    result.update({
                        "highres_local_frame_event_logits": highres_outputs[
                            "local_frame_event_logits"
                        ],
                        "highres_candidate_indices": highres_outputs[
                            "candidate_indices"
                        ],
                        "highres_selected_pool_indices": highres_outputs[
                            "selected_pool_indices"
                        ],
                        "highres_crop_centers": highres_outputs["crop_centers"],
                        "highres_crop_scales": highres_outputs["crop_scales"],
                        "highres_crop_attention": highres_outputs["crop_attention"],
                        "highres_local_logits": highres_outputs["local_logits"],
                        "highres_evidence_gates": highres_outputs[
                            "evidence_gates"
                        ],
                        "highres_local_attention": highres_outputs["local_attention"],
                        "highres_residual_delta": highres_outputs["residual_delta"],
                        "highres_residual_gate": highres_outputs["residual_gate"],
                        "highres_motion_prior": highres_outputs["motion_prior"],
                        "highres_fused_frame_event_logits": highres_outputs[
                            "frame_event_logits"
                        ],
                    })
                    if "shuffled_local_logits" in highres_outputs:
                        result["highres_shuffled_local_logits"] = highres_outputs[
                            "shuffled_local_logits"
                        ]
                    if "causal_valid_mask" in highres_outputs:
                        result["highres_causal_valid_mask"] = highres_outputs[
                            "causal_valid_mask"
                        ]
                if "multi_layer_frame_residual" in global_outputs:
                    result["multi_layer_frame_residual"] = global_outputs[
                        "multi_layer_frame_residual"
                    ]
                if "frame_attention" in global_outputs:
                    result["frame_attention"] = global_outputs["frame_attention"]
                if "topk_indices" in global_outputs:
                    result["topk_indices"] = global_outputs["topk_indices"]
                if class_evidence_outputs is not None:
                    if self.class_evidence_mode == "global_local_temporal":
                        result.update({
                            "frame_event_logits": class_evidence_outputs[
                                "frame_event_logits"
                            ],
                            "class_evidence_fused_frame_tokens": class_evidence_outputs[
                                "fused_frame_tokens"
                            ],
                            "class_evidence_local_tokens": class_evidence_outputs[
                                "local_evidence_tokens"
                            ],
                            "class_evidence_raw_local_tokens": class_evidence_outputs[
                                "raw_local_evidence_tokens"
                            ],
                            "class_evidence_presence": class_evidence_outputs[
                                "local_evidence_presence"
                            ],
                            "class_evidence_frame_event_logits": class_evidence_outputs[
                                "local_frame_event_logits"
                            ],
                            "class_evidence_no_evidence_logits": class_evidence_outputs[
                                "no_evidence_logits"
                            ],
                            "class_evidence_temporal_representation": class_evidence_outputs[
                                "temporal_representation"
                            ],
                            "class_evidence_attention_entropy": class_evidence_outputs[
                                "attention_entropy"
                            ],
                            "class_evidence_query_diversity_loss": class_evidence_outputs[
                                "query_diversity_loss"
                            ],
                        })
                    else:
                        result.update({
                            "spatial_fusion_delta": class_evidence_outputs["correction"],
                            "class_evidence_raw_correction": class_evidence_outputs[
                                "raw_correction"
                            ],
                            "class_evidence_selected_indices": class_evidence_outputs[
                                "selected_indices"
                            ],
                            "class_evidence_selected_valid": class_evidence_outputs[
                                "selected_valid"
                            ],
                            "class_evidence_temporal_weights": class_evidence_outputs[
                                "temporal_weights"
                            ],
                            "class_evidence_attention_entropy": class_evidence_outputs[
                                "attention_entropy"
                            ],
                        })
                    if "attention_maps" in class_evidence_outputs:
                        result["class_evidence_attention_maps"] = (
                            class_evidence_outputs["attention_maps"]
                        )
                for key in (
                    "spatial_token_slot_logits",
                    "spatial_token_attention_entropy",
                    "spatial_token_attention_overlap",
                    "spatial_token_query_diversity_loss",
                    "spatial_token_slot_gate",
                    "spatial_token_context_scale",
                    "spatial_token_attention_maps",
                    "temporal_difference_gate",
                    "temporal_difference_residual_abs_mean",
                    "temporal_difference_residual_norm",
                    "uniform_logits",
                    "event_logits",
                    "uniform_indices",
                    "temporal_gate",
                    "retention_reference_logits",
                    "global_action_logits",
                    "temporal_delta_logits",
                    "temporal_residual_gate",
                    "temporal_logits",
                    "response_clip_logits",
                    "response_curve_logits",
                ):
                    if key in global_outputs:
                        result[key] = global_outputs[key]
                if spatial_outputs is not None:
                    result.update(
                        {
                            "spatial_residual_logits": spatial_outputs[
                                "residual_logits"
                            ],
                            "spatial_clip_logits": spatial_outputs["clip_logits"],
                            "spatial_fused_logits": spatial_fused_logits,
                            "spatial_frame_event_logits": spatial_outputs[
                                "frame_event_logits"
                            ],
                            "spatial_gate": spatial_outputs["gate"],
                            "spatial_attention_entropy": spatial_outputs[
                                "attention_entropy"
                            ],
                            "spatial_attention_overlap": spatial_outputs[
                                "attention_overlap"
                            ],
                            "spatial_clip_gate": spatial_outputs["clip_gate"],
                            "spatial_query_diversity_loss": spatial_outputs[
                                "query_diversity_loss"
                            ],
                            "spatial_context_query_scale": spatial_outputs[
                                "context_query_scale"
                            ],
                            "spatial_motion_query_scale": spatial_outputs[
                                "motion_query_scale"
                            ],
                            "spatial_actionness_query_scale": spatial_outputs[
                                "actionness_query_scale"
                            ],
                            "spatial_active_patch_fraction": spatial_outputs[
                                "active_patch_fraction"
                            ],
                            "retention_reference_logits": global_logits,
                        }
                    )
                    if "erased_clip_logits" in spatial_outputs:
                        result["spatial_erased_clip_logits"] = spatial_outputs[
                            "erased_clip_logits"
                        ]
                    if spatial_fusion_delta is not None:
                        result["spatial_fusion_delta"] = spatial_fusion_delta
                    if spatial_fusion_gate is not None:
                        result["spatial_fusion_gate"] = spatial_fusion_gate
                    if spatial_feature_delta is not None:
                        result["spatial_feature_delta"] = spatial_feature_delta
                    if spatial_feature_frame_logits is not None:
                        # Preserve the corrected path under an explicit key so
                        # it can be supervised independently from the raw probe.
                        result["structured_frame_event_logits"] = (
                            spatial_feature_frame_logits
                        )
                    if "attention_maps" in spatial_outputs:
                        result["spatial_attention_maps"] = spatial_outputs[
                            "attention_maps"
                        ]
                return result
            return logits
        if roi_inputs is None:
            raise ValueError(f"roi_inputs are required when model.view_fusion={self.view_fusion}")
        local_features = self.encode_frames(
            roi_inputs,
            backbone=self.local_backbone if self.local_backbone is not None else self.backbone,
        )
        roi_pair_gate: Tensor | None = None
        roi_pair_delta: Tensor | None = None
        local_tokens_a_for_gain: Tensor | None = None
        local_tokens_b_for_gain: Tensor | None = None
        roi_gain_weight_a: Tensor | None = None
        roi_gain_weight_b: Tensor | None = None
        if self.roi_count == 2:
            if roi_inputs_b is None:
                raise ValueError("roi_inputs_b is required when model.roi_count=2")
            if self.multi_roi_frame_residual is None or self.multi_roi_frame_gate is None:
                raise RuntimeError("multi-ROI aggregation modules are missing")
            local_features_b = self.encode_frames(
                roi_inputs_b,
                backbone=self.local_backbone if self.local_backbone is not None else self.backbone,
            )
            local_tokens_a = self.local_frame_proj(local_features)
            local_tokens_b = self.local_frame_proj(local_features_b)
            local_tokens_a_for_gain = local_tokens_a
            local_tokens_b_for_gain = local_tokens_b
            batch_size, local_frames, _ = local_tokens_a.shape
            if roi_frame_meta is None:
                meta_a = local_tokens_a.new_zeros(
                    (batch_size, local_frames, self.roi_meta_dim)
                )
            else:
                meta_a = roi_frame_meta.to(local_tokens_a.dtype)
            if roi_frame_meta_b is None:
                meta_b = local_tokens_a.new_zeros(
                    (batch_size, local_frames, self.roi_meta_dim)
                )
            else:
                meta_b = roi_frame_meta_b.to(local_tokens_a.dtype)
            if tuple(meta_a.shape[:2]) != (batch_size, local_frames):
                raise ValueError("roi_frame_meta has incompatible multi-ROI shape")
            if tuple(meta_b.shape[:2]) != (batch_size, local_frames):
                raise ValueError("roi_frame_meta_b has incompatible multi-ROI shape")
            if roi_frame_valid_b is None:
                if roi_valid_b is None:
                    valid_b = local_tokens_a.new_ones((batch_size, local_frames))
                else:
                    valid_b = roi_valid_b.to(local_tokens_a.dtype).reshape(-1, 1).expand(-1, local_frames)
            else:
                valid_b = roi_frame_valid_b.to(local_tokens_a.dtype)
            if self.view_fusion in (
                "dual_multi_roi_gain",
                "dual_multi_roi_memory",
            ):
                local_outputs = self._projected_branch_outputs(
                    local_tokens_a,
                    self.local_frame_event_head,
                    self.local_temporal,
                    self.local_head,
                )
            else:
                pair_input = torch.cat(
                    [
                        local_tokens_a,
                        local_tokens_b,
                        (local_tokens_b - local_tokens_a).abs(),
                        meta_a,
                        meta_b,
                    ],
                    dim=-1,
                )
                roi_pair_gate = torch.sigmoid(
                    self.multi_roi_frame_gate(pair_input).squeeze(-1)
                ) * valid_b
                roi_pair_delta = (
                    roi_pair_gate.unsqueeze(-1)
                    * self.multi_roi_frame_residual(pair_input)
                )
                local_tokens = local_tokens_a + roi_pair_delta
                local_outputs = self._projected_branch_outputs(
                    local_tokens,
                    self.local_frame_event_head,
                    self.local_temporal,
                    self.local_head,
                )
        else:
            local_outputs = self._branch_outputs(
                local_features,
                self.local_frame_proj,
                self.local_frame_event_head,
                self.local_temporal,
                self.local_head,
            )
        local_temporal = local_outputs["temporal"]
        local_logits = local_outputs["logits"]
        if roi_meta is None:
            roi_meta = global_logits.new_zeros((len(global_logits), self.roi_meta_dim))
        else:
            roi_meta = roi_meta.to(global_logits.dtype)
        if roi_valid is None:
            roi_valid = global_logits.new_ones((len(global_logits),))
        else:
            roi_valid = roi_valid.to(global_logits.dtype)

        feature_outputs: dict[str, Tensor] | None = None
        roi_quality_logits: Tensor | None = None
        roi_frame_quality_logits: Tensor | None = None
        roi_frame_quality: Tensor | None = None
        roi_residual_logits: Tensor | None = None
        if self.view_fusion == "dual_gate":
            gate_input = torch.cat([global_temporal, local_temporal, roi_meta], dim=-1)
            learned_gate = torch.sigmoid(self.roi_gate(gate_input))
            confidence = roi_meta[:, 1:2].clamp(0.0, 1.0)
            alpha = learned_gate * confidence * roi_valid.reshape(-1, 1)
            logits = global_logits + alpha * (local_logits - global_logits)
        elif self.view_fusion == "dual_verifier":
            detached_global_temporal = global_temporal.detach()
            fusion_input = torch.cat(
                [
                    detached_global_temporal,
                    local_temporal,
                    (local_temporal - detached_global_temporal).abs(),
                    roi_meta,
                ],
                dim=-1,
            )
            learned_gate = torch.sigmoid(self.roi_gate(fusion_input))
            confidence = roi_meta[:, 1:2].clamp(0.0, 1.0)
            confidence_gate = (
                (confidence - self.roi_verifier_min_confidence)
                / max(1.0 - self.roi_verifier_min_confidence, 1e-6)
            ).clamp(0.0, 1.0)
            alpha = learned_gate * confidence_gate * roi_valid.reshape(-1, 1)
            alpha = alpha * self.roi_verifier_class_mask.to(alpha.dtype)
            # A verifier is suppression-only: ROI evidence may remove false
            # positives, but it cannot create detections absent from global.
            roi_residual_logits = -F.softplus(self.roi_residual_head(fusion_input))
            logits = global_logits + alpha * roi_residual_logits
        elif self.view_fusion == "dual_feature_quality":
            global_tokens = global_outputs["frame_tokens"]
            local_tokens = local_outputs["frame_tokens"]
            bsz, frames, _ = global_tokens.shape
            if roi_frame_meta is None:
                roi_frame_meta = global_tokens.new_zeros((bsz, frames, self.roi_meta_dim))
            else:
                roi_frame_meta = roi_frame_meta.to(global_tokens.dtype)
                if tuple(roi_frame_meta.shape[:2]) != (bsz, frames):
                    raise ValueError(
                        f"roi_frame_meta shape={tuple(roi_frame_meta.shape)} must start with {(bsz, frames)}"
                    )
            if roi_frame_valid is None:
                roi_frame_valid = (roi_valid > 0).to(global_tokens.dtype).reshape(-1, 1).expand(-1, frames)
            else:
                roi_frame_valid = roi_frame_valid.to(global_tokens.dtype)
                if tuple(roi_frame_valid.shape) != (bsz, frames):
                    raise ValueError(
                        f"roi_frame_valid shape={tuple(roi_frame_valid.shape)} must equal {(bsz, frames)}"
                    )
            frame_quality_input = torch.cat(
                [global_tokens, local_tokens, (local_tokens - global_tokens).abs(), roi_frame_meta],
                dim=-1,
            )
            roi_frame_quality_logits = self.roi_frame_quality_head(frame_quality_input).squeeze(-1)
            roi_frame_quality = torch.sigmoid(roi_frame_quality_logits) * roi_frame_valid
            adapted_local = self.roi_feature_adapter(local_tokens)
            fused_tokens = global_tokens + roi_frame_quality.unsqueeze(-1) * adapted_local
            feature_outputs = self._projected_branch_outputs(
                fused_tokens, self.frame_event_head, self.temporal, self.head
            )
            fused_temporal = feature_outputs["temporal"]
            fusion_input = torch.cat(
                [global_temporal, local_temporal, fused_temporal, roi_meta], dim=-1
            )
            roi_quality_logits = self.roi_quality_head(fusion_input)
            roi_residual_logits = self.roi_residual_head(fusion_input)
            learned_quality = torch.sigmoid(roi_quality_logits)
            frame_availability = roi_frame_quality.mean(dim=1, keepdim=True)
            alpha = (
                learned_quality
                * frame_availability
                * (roi_valid > 0).to(global_logits.dtype).reshape(-1, 1)
            )
            logits = global_logits + alpha * roi_residual_logits
        elif self.view_fusion == "dual_multi_roi_memory":
            if (
                local_tokens_a_for_gain is None
                or local_tokens_b_for_gain is None
                or not hasattr(self, "dual_cross_attention")
                or not hasattr(self, "roi_frame_quality_head")
                or not hasattr(self, "roi_feature_adapter")
            ):
                raise RuntimeError(
                    "dual_multi_roi_memory modules or ROI tokens are missing"
                )
            global_tokens = global_outputs["frame_tokens"]
            batch_size, global_frames, hidden_dim = global_tokens.shape

            if global_frame_times is None:
                global_frame_times = torch.linspace(
                    0.0,
                    1.0,
                    global_frames,
                    device=global_tokens.device,
                    dtype=torch.float32,
                ).unsqueeze(0).expand(batch_size, -1)
            else:
                global_frame_times = global_frame_times.to(global_tokens.device)

            def resolve_local_times(
                raw_times: Tensor | None, frame_count: int
            ) -> Tensor:
                if raw_times is not None:
                    resolved = raw_times.to(global_tokens.device)
                    if resolved.shape != (batch_size, frame_count):
                        raise ValueError(
                            "multi-ROI memory timestamps have incompatible shape"
                        )
                    return resolved
                start_time = global_frame_times[:, :1]
                duration = (
                    global_frame_times[:, -1:] - start_time
                ).clamp_min(1e-6)
                fractions = torch.linspace(
                    0.0,
                    1.0,
                    frame_count,
                    device=global_tokens.device,
                    dtype=global_frame_times.dtype,
                ).unsqueeze(0)
                return start_time + fractions * duration

            local_times_a = resolve_local_times(
                local_frame_times, local_tokens_a_for_gain.shape[1]
            )
            local_times_b = resolve_local_times(
                local_frame_times_b, local_tokens_b_for_gain.shape[1]
            )

            def prepare_memory_path(
                local_tokens: Tensor,
                local_times: Tensor,
                raw_meta: Tensor | None,
                raw_valid: Tensor | None,
                clip_valid: Tensor | None,
            ) -> tuple[Tensor, Tensor, Tensor]:
                local_frames = local_tokens.shape[1]
                if raw_meta is None:
                    frame_meta = global_tokens.new_zeros(
                        (batch_size, local_frames, self.roi_meta_dim)
                    )
                else:
                    frame_meta = raw_meta.to(global_tokens.dtype)
                    if tuple(frame_meta.shape[:2]) != (
                        batch_size,
                        local_frames,
                    ):
                        raise ValueError(
                            "multi-ROI memory metadata has incompatible shape"
                        )
                if raw_valid is not None:
                    frame_valid = raw_valid.to(global_tokens.dtype)
                elif clip_valid is not None:
                    frame_valid = clip_valid.to(global_tokens.dtype).reshape(
                        -1, 1
                    ).expand(-1, local_frames)
                else:
                    frame_valid = global_tokens.new_ones(
                        (batch_size, local_frames)
                    )
                if tuple(frame_valid.shape) != (batch_size, local_frames):
                    raise ValueError(
                        "multi-ROI memory validity has incompatible shape"
                    )
                nearest_indices = (
                    local_times.unsqueeze(-1)
                    - global_frame_times.unsqueeze(1)
                ).abs().argmin(dim=-1)
                nearest_global = global_tokens.gather(
                    1,
                    nearest_indices.unsqueeze(-1).expand(
                        -1, -1, hidden_dim
                    ),
                )
                quality_input = torch.cat(
                    [
                        nearest_global,
                        local_tokens,
                        (local_tokens - nearest_global).abs(),
                        frame_meta,
                    ],
                    dim=-1,
                )
                quality_logits = self.roi_frame_quality_head(
                    quality_input
                ).squeeze(-1)
                quality = torch.sigmoid(quality_logits) * frame_valid
                return (
                    self.roi_feature_adapter(local_tokens),
                    quality_logits,
                    quality,
                )

            (
                adapted_local_a,
                quality_logits_a,
                quality_a,
            ) = prepare_memory_path(
                local_tokens_a_for_gain,
                local_times_a,
                roi_frame_meta,
                roi_frame_valid,
                roi_valid,
            )
            (
                adapted_local_b,
                quality_logits_b,
                quality_b,
            ) = prepare_memory_path(
                local_tokens_b_for_gain,
                local_times_b,
                roi_frame_meta_b,
                roi_frame_valid_b,
                roi_valid_b,
            )
            memory_tokens = torch.cat(
                [adapted_local_a, adapted_local_b], dim=1
            )
            memory_times = torch.cat(
                [local_times_a, local_times_b], dim=1
            )
            memory_quality = torch.cat([quality_a, quality_b], dim=1)
            memory_view_ids = torch.cat(
                [
                    torch.zeros_like(quality_a, dtype=torch.long),
                    torch.ones_like(quality_b, dtype=torch.long),
                ],
                dim=1,
            )
            token_outputs = self.dual_cross_attention(
                global_tokens,
                memory_tokens,
                global_frame_times,
                memory_times,
                memory_quality,
                memory_view_ids,
            )
            fused_tokens = token_outputs["tokens"]
            feature_outputs = self._projected_branch_outputs(
                fused_tokens,
                self.frame_event_head,
                self.temporal,
                self.head,
            )
            logits = feature_outputs["logits"]
            roi_residual_logits = logits - global_logits
            availability = (
                memory_quality.sum(dim=1, keepdim=True) > 0
            ).to(global_logits.dtype)
            memory_gate = token_outputs["residual_gates"].abs().mean()
            alpha = (
                availability
                * memory_gate.to(global_logits.dtype)
            ).expand(-1, self.num_labels)
            roi_frame_quality_logits = torch.cat(
                [quality_logits_a, quality_logits_b], dim=1
            )
            roi_frame_quality = memory_quality
            feature_outputs["frame_times"] = global_frame_times
            feature_outputs["frame_view_ids"] = torch.zeros(
                batch_size,
                global_frames,
                dtype=torch.long,
                device=global_tokens.device,
            )
            feature_outputs["roi_cross_attention"] = token_outputs[
                "attention"
            ]
            feature_outputs["roi_memory_residual_gates"] = token_outputs[
                "residual_gates"
            ]
            feature_outputs["roi_memory_quality_a"] = quality_a
            feature_outputs["roi_memory_quality_b"] = quality_b
        elif self.view_fusion == "dual_multi_roi_gain":
            if (
                local_tokens_a_for_gain is None
                or local_tokens_b_for_gain is None
                or self.multi_roi_gain_adapter is None
                or self.multi_roi_gain_frame_head is None
                or self.multi_roi_gain_clip_head is None
            ):
                raise RuntimeError("dual_multi_roi_gain modules or ROI tokens are missing")
            global_tokens = global_outputs["frame_tokens"]
            bsz, frames, _ = global_tokens.shape

            def prepare_frame_meta(
                raw_meta: Tensor | None, raw_valid: Tensor | None, clip_valid: Tensor | None
            ) -> tuple[Tensor, Tensor]:
                if raw_meta is None:
                    meta = global_tokens.new_zeros((bsz, frames, self.roi_meta_dim))
                else:
                    meta = raw_meta.to(global_tokens.dtype)
                    if tuple(meta.shape[:2]) != (bsz, frames):
                        raise ValueError("multi-ROI gain metadata has incompatible shape")
                if raw_valid is not None:
                    valid = raw_valid.to(global_tokens.dtype)
                elif clip_valid is not None:
                    valid = clip_valid.to(global_tokens.dtype).reshape(-1, 1).expand(-1, frames)
                else:
                    valid = global_tokens.new_ones((bsz, frames))
                if tuple(valid.shape) != (bsz, frames):
                    raise ValueError("multi-ROI gain validity has incompatible shape")
                return meta, valid

            meta_a, valid_a = prepare_frame_meta(
                roi_frame_meta, roi_frame_valid, roi_valid
            )
            meta_b, valid_b = prepare_frame_meta(
                roi_frame_meta_b, roi_frame_valid_b, roi_valid_b
            )
            gain_input_a = torch.cat(
                [
                    global_tokens,
                    local_tokens_a_for_gain,
                    (local_tokens_a_for_gain - global_tokens).abs(),
                    meta_a,
                ],
                dim=-1,
            )
            gain_input_b = torch.cat(
                [
                    global_tokens,
                    local_tokens_b_for_gain,
                    (local_tokens_b_for_gain - global_tokens).abs(),
                    meta_b,
                ],
                dim=-1,
            )
            relevance_logits = torch.stack(
                [
                    self.multi_roi_gain_frame_head(gain_input_a).squeeze(-1),
                    self.multi_roi_gain_frame_head(gain_input_b).squeeze(-1),
                ],
                dim=-1,
            )
            valid_pair = torch.stack([valid_a, valid_b], dim=-1)
            masked_relevance = relevance_logits.masked_fill(valid_pair <= 0, -1e4)
            relevance = torch.softmax(masked_relevance, dim=-1) * valid_pair
            relevance = relevance / relevance.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            roi_gain_weight_a = relevance[..., 0]
            roi_gain_weight_b = relevance[..., 1]
            delta_a = self.multi_roi_gain_adapter(gain_input_a)
            delta_b = self.multi_roi_gain_adapter(gain_input_b)
            roi_delta = (
                roi_gain_weight_a.unsqueeze(-1) * delta_a
                + roi_gain_weight_b.unsqueeze(-1) * delta_b
            )
            any_valid = (valid_pair.sum(dim=-1) > 0).to(global_tokens.dtype)
            fused_tokens = global_tokens + any_valid.unsqueeze(-1) * roi_delta
            feature_outputs = self._projected_branch_outputs(
                fused_tokens, self.frame_event_head, self.temporal, self.head
            )
            fused_temporal = feature_outputs["temporal"]
            mean_weight_a = roi_gain_weight_a.mean(dim=1, keepdim=True)
            mean_weight_b = roi_gain_weight_b.mean(dim=1, keepdim=True)
            clip_gain_input = torch.cat(
                [
                    global_temporal,
                    fused_temporal,
                    (fused_temporal - global_temporal).abs(),
                    roi_meta,
                    roi_meta_b.to(global_logits.dtype)
                    if roi_meta_b is not None
                    else global_logits.new_zeros((bsz, self.roi_meta_dim)),
                    mean_weight_a,
                    mean_weight_b,
                ],
                dim=-1,
            )
            roi_quality_logits = self.multi_roi_gain_clip_head(clip_gain_input)
            availability = (any_valid.mean(dim=1, keepdim=True) > 0).to(
                global_logits.dtype
            )
            alpha = torch.sigmoid(roi_quality_logits) * availability
            roi_residual_logits = feature_outputs["logits"] - global_logits
            logits = global_logits + alpha * roi_residual_logits
            roi_frame_quality_logits = relevance_logits.max(dim=-1).values
            roi_frame_quality = relevance.max(dim=-1).values * any_valid
        else:
            global_tokens = global_outputs["frame_tokens"]
            local_tokens = local_outputs["frame_tokens"]
            batch_size, global_frames, hidden_dim = global_tokens.shape
            local_frames = local_tokens.shape[1]
            if roi_frame_meta is None:
                roi_frame_meta = global_tokens.new_zeros(
                    (batch_size, local_frames, self.roi_meta_dim)
                )
            else:
                roi_frame_meta = roi_frame_meta.to(global_tokens.dtype)
                if tuple(roi_frame_meta.shape[:2]) != (batch_size, local_frames):
                    raise ValueError(
                        f"roi_frame_meta shape={tuple(roi_frame_meta.shape)} must "
                        f"start with {(batch_size, local_frames)}"
                    )
            if roi_frame_valid is None:
                roi_frame_valid = (
                    (roi_valid > 0)
                    .to(global_tokens.dtype)
                    .reshape(-1, 1)
                    .expand(-1, local_frames)
                )
            else:
                roi_frame_valid = roi_frame_valid.to(global_tokens.dtype)
                if tuple(roi_frame_valid.shape) != (batch_size, local_frames):
                    raise ValueError(
                        f"roi_frame_valid shape={tuple(roi_frame_valid.shape)} "
                        f"must equal {(batch_size, local_frames)}"
                    )
            if global_frame_times is None:
                global_frame_times = torch.linspace(
                    0.0, 1.0, global_frames, device=global_tokens.device
                ).expand(batch_size, -1)
            else:
                global_frame_times = global_frame_times.to(global_tokens.device)
            if local_frame_times is None:
                if local_frames == global_frames:
                    local_frame_times = global_frame_times.clone()
                else:
                    fractions = torch.linspace(
                        0.0, 1.0, local_frames, device=local_tokens.device
                    ).reshape(1, -1)
                    start_time = global_frame_times.amin(dim=1, keepdim=True)
                    duration = (
                        global_frame_times.amax(dim=1, keepdim=True) - start_time
                    )
                    local_frame_times = start_time + fractions * duration
            else:
                local_frame_times = local_frame_times.to(local_tokens.device)

            nearest_indices = (
                local_frame_times.unsqueeze(-1)
                - global_frame_times.unsqueeze(1)
            ).abs().argmin(dim=-1)
            nearest_global = global_tokens.gather(
                1,
                nearest_indices.unsqueeze(-1).expand(-1, -1, hidden_dim),
            )
            frame_quality_input = torch.cat(
                [
                    nearest_global,
                    local_tokens,
                    (local_tokens - nearest_global).abs(),
                    roi_frame_meta,
                ],
                dim=-1,
            )
            roi_frame_quality_logits = self.roi_frame_quality_head(
                frame_quality_input
            ).squeeze(-1)
            roi_frame_quality = (
                torch.sigmoid(roi_frame_quality_logits) * roi_frame_valid
            )
            adapted_local = self.roi_feature_adapter(local_tokens)
            if self.view_fusion == "dual_cross_attention":
                token_outputs = self.dual_cross_attention(
                    global_tokens,
                    adapted_local,
                    global_frame_times,
                    local_frame_times,
                    roi_frame_quality,
                )
                fused_tokens = token_outputs["tokens"]
                feature_outputs = self._projected_branch_outputs(
                    fused_tokens,
                    self.frame_event_head,
                    self.temporal,
                    self.head,
                )
                raw_feature_logits = feature_outputs["logits"]
                alpha = roi_frame_quality.mean(dim=1, keepdim=True) * (
                    roi_valid > 0
                ).to(global_logits.dtype).reshape(-1, 1)
                roi_residual_logits = bounded_asymmetric_residual(
                    raw_feature_logits - global_logits,
                    positive_max=self.view_fusion_positive_delta.to(
                        device=global_logits.device, dtype=global_logits.dtype
                    ),
                    negative_max=self.view_fusion_negative_delta.to(
                        device=global_logits.device, dtype=global_logits.dtype
                    ),
                )
                logits = global_logits + alpha * roi_residual_logits
                feature_outputs["logits"] = logits
                feature_outputs["frame_times"] = global_frame_times
                feature_outputs["frame_view_ids"] = torch.zeros(
                    batch_size,
                    global_frames,
                    dtype=torch.long,
                    device=global_tokens.device,
                )
                if token_outputs.get("attention") is not None:
                    feature_outputs["roi_cross_attention"] = token_outputs["attention"]
            else:
                token_outputs = self.dual_token_temporal(
                    global_tokens,
                    adapted_local,
                    global_frame_times,
                    local_frame_times,
                    roi_frame_quality,
                )
                fused_temporal = token_outputs["temporal"]
                frame_availability = roi_frame_quality.mean(dim=1, keepdim=True)
                if self.view_fusion == "dual_direct_class_query_fusion":
                    direct_delta_gate = torch.sigmoid(
                        self.dual_direct_delta_gate_logits
                    ).to(device=global_logits.device, dtype=global_logits.dtype)
                    alpha = (
                        frame_availability
                        * direct_delta_gate.reshape(1, -1)
                        * (roi_valid > 0).to(global_logits.dtype).reshape(-1, 1)
                    )
                    raw_residual = self.dual_direct_head(
                        token_outputs["query_features"]
                    ).squeeze(-1)
                    roi_residual_logits = bounded_asymmetric_residual(
                        raw_residual,
                        positive_max=self.view_fusion_positive_delta.to(
                            device=raw_residual.device, dtype=raw_residual.dtype
                        ),
                        negative_max=self.view_fusion_negative_delta.to(
                            device=raw_residual.device, dtype=raw_residual.dtype
                        ),
                    )
                    logits = global_logits + alpha * roi_residual_logits
                else:
                    raw_residual = self.dual_token_residual(
                        token_outputs["query_features"]
                    ).squeeze(-1)
                    roi_residual_logits = bounded_asymmetric_residual(
                        raw_residual,
                        positive_max=self.view_fusion_positive_delta.to(
                            device=raw_residual.device, dtype=raw_residual.dtype
                        ),
                        negative_max=self.view_fusion_negative_delta.to(
                            device=raw_residual.device, dtype=raw_residual.dtype
                        ),
                    )
                    if self.view_fusion == "dual_class_query_fusion":
                        alpha = frame_availability * (roi_valid > 0).to(
                            global_logits.dtype
                        ).reshape(-1, 1)
                    else:
                        fusion_input = torch.cat(
                            [global_temporal, local_temporal, fused_temporal, roi_meta],
                            dim=-1,
                        )
                        roi_quality_logits = self.roi_quality_head(fusion_input)
                        learned_quality = torch.sigmoid(roi_quality_logits)
                        alpha = (
                            learned_quality
                            * frame_availability
                            * (roi_valid > 0).to(global_logits.dtype).reshape(-1, 1)
                        )
                    logits = global_logits + alpha * roi_residual_logits
                # Frame detection targets are defined on the original full-image
                # timestamps. Mixed-token fusion sorts global and ROI tokens
                # together, so recover the encoded global-view tokens by their
                # original concat indices instead of taking the first N tokens.
                sorted_order = token_outputs["order"]
                global_token_mask = sorted_order < global_frames
                if not bool(global_token_mask.sum(dim=1).eq(global_frames).all()):
                    raise RuntimeError(
                        "mixed-token fusion did not preserve exactly "
                        f"{global_frames} global tokens per sample"
                    )
                selected_original_indices = sorted_order[global_token_mask].reshape(
                    batch_size, global_frames
                )
                selected_global_tokens = token_outputs["encoded_tokens"][
                    global_token_mask
                ].reshape(batch_size, global_frames, hidden_dim)
                restore_order = selected_original_indices.argsort(dim=1)
                fused_global_tokens = selected_global_tokens.gather(
                    1,
                    restore_order.unsqueeze(-1).expand(-1, -1, hidden_dim),
                )
                fused_frame_logits = self.frame_event_head(fused_global_tokens)
                feature_outputs = {
                    "temporal": fused_temporal,
                    "frame_event_logits": fused_frame_logits,
                    "frame_attention": token_outputs["attention"],
                    "frame_times": global_frame_times,
                    "frame_view_ids": torch.zeros(
                        batch_size,
                        global_frames,
                        dtype=torch.long,
                        device=global_tokens.device,
                    ),
                }

        if return_aux:
            early_fused_frame_loss = self.view_fusion in (
                "dual_cross_attention",
                "dual_multi_roi_memory",
                "dual_class_query_fusion",
                "dual_direct_class_query_fusion",
            ) and feature_outputs is not None
            primary_frame_event_logits = (
                feature_outputs["frame_event_logits"]
                if early_fused_frame_loss
                else global_outputs["frame_event_logits"]
            )
            result = {
                "logits": logits,
                "global_logits": global_logits,
                "local_logits": local_logits,
                "roi_gate": alpha,
                "frame_event_logits": primary_frame_event_logits,
                "global_frame_event_logits": global_outputs["frame_event_logits"],
            }
            for key in ("temporal_logits", "response_clip_logits", "response_curve_logits"):
                if key in global_outputs:
                    result[key] = global_outputs[key]
            if not early_fused_frame_loss:
                result["local_frame_event_logits"] = local_outputs["frame_event_logits"]
            if roi_pair_gate is not None:
                result["roi_pair_gate"] = roi_pair_gate
            if roi_pair_delta is not None:
                result["roi_pair_delta"] = roi_pair_delta
            if roi_gain_weight_a is not None and roi_gain_weight_b is not None:
                result["roi_gain_weight_a"] = roi_gain_weight_a
                result["roi_gain_weight_b"] = roi_gain_weight_b
            if roi_residual_logits is not None:
                result["roi_residual_logits"] = roi_residual_logits
            if feature_outputs is not None and roi_residual_logits is not None:
                result["roi_feature_logits"] = global_logits + roi_residual_logits
                result["fused_frame_event_logits"] = feature_outputs["frame_event_logits"]
                if "frame_times" in feature_outputs:
                    result["fused_frame_times"] = feature_outputs["frame_times"]
                if "frame_view_ids" in feature_outputs:
                    result["fused_frame_view_ids"] = feature_outputs["frame_view_ids"]

            if roi_quality_logits is not None:
                result["roi_quality_logits"] = roi_quality_logits
            if roi_frame_quality_logits is not None and roi_frame_quality is not None:
                result["roi_frame_quality_logits"] = roi_frame_quality_logits
                result["roi_frame_quality"] = roi_frame_quality
            if "frame_attention" in global_outputs:
                result["frame_attention"] = global_outputs["frame_attention"]
                result["global_frame_attention"] = global_outputs["frame_attention"]
            if "frame_attention" in local_outputs:
                result["local_frame_attention"] = local_outputs["frame_attention"]
            if feature_outputs is not None and "frame_attention" in feature_outputs:
                result["fused_frame_attention"] = feature_outputs["frame_attention"]
            if feature_outputs is not None and "roi_cross_attention" in feature_outputs:
                result["roi_cross_attention"] = feature_outputs["roi_cross_attention"]
            if (
                feature_outputs is not None
                and "roi_memory_residual_gates" in feature_outputs
            ):
                result["roi_memory_residual_gates"] = feature_outputs[
                    "roi_memory_residual_gates"
                ]
                result["roi_memory_quality_a"] = feature_outputs[
                    "roi_memory_quality_a"
                ]
                result["roi_memory_quality_b"] = feature_outputs[
                    "roi_memory_quality_b"
                ]
            if "topk_indices" in global_outputs:
                result["topk_indices"] = global_outputs["topk_indices"]
                result["global_topk_indices"] = global_outputs["topk_indices"]
            if "topk_indices" in local_outputs:
                result["local_topk_indices"] = local_outputs["topk_indices"]
            if feature_outputs is not None and "topk_indices" in feature_outputs:
                result["fused_topk_indices"] = feature_outputs["topk_indices"]
            return result
        return logits

def make_model(cfg: Any, *, use_cached_features: bool, device: torch.device) -> VideoEventClassifier:
    init_checkpoint, init_checkpoint_data = resolve_model_init_checkpoint(cfg)
    backbone = None
    local_backbone = None
    frame_feature_dim = int(cfg.model.get("frame_feature_dim", 2048))
    if not use_cached_features:
        backbone = build_backbone(cfg)
        lora_cfg = cfg.model.get("lora", cfg.get("lora", ConfigDict()))
        if bool(lora_cfg.get("enabled", False)):
            injected = inject_lora(backbone, lora_cfg)
            print(f"LoRA injected into {injected} DINO linear modules", flush=True)
        else:
            configure_backbone_trainability(
                backbone,
                freeze=bool(cfg.model.get("freeze_backbone", True)),
                finetune_last_blocks=int(cfg.model.get("finetune_last_blocks", 0)),
            )
        if bool(cfg.model.get("separate_local_backbone", False)):
            local_backbone = build_backbone(cfg)
            if bool(lora_cfg.get("enabled", False)):
                local_injected = inject_lora(local_backbone, lora_cfg)
                print(
                    f"Local ROI LoRA injected into {local_injected} DINO linear modules",
                    flush=True,
                )
            else:
                configure_backbone_trainability(
                    local_backbone,
                    freeze=bool(cfg.model.get("freeze_backbone", True)),
                    finetune_last_blocks=int(cfg.model.get("finetune_last_blocks", 0)),
                )
        if bool(getattr(backbone, "is_video_backbone", False)):
            frame_feature_dim = int(backbone.output_feature_dim)
        else:
            frame_feature_dim = backbone.num_features * 2
    spatial_attention_cfg = cfg.model.get(
        "spatial_attention", ConfigDict()
    )
    temporal_difference_cfg = cfg.model.get(
        "temporal_difference", ConfigDict()
    )
    spatial_token_pooling_cfg = cfg.model.get(
        "spatial_token_pooling", ConfigDict()
    )
    class_evidence_cfg = cfg.model.get(
        "class_evidence", ConfigDict()
    )
    highres_glimpse_cfg = cfg.model.get(
        "highres_glimpse", ConfigDict()
    )
    object_spatial_aux_cfg = cfg.model.get(
        "object_spatial_aux", ConfigDict()
    )
    ball_goal_relation_aux_cfg = cfg.model.get(
        "ball_goal_relation_aux", ConfigDict()
    )
    response_curve_primary_cfg = cfg.model.get(
        "response_curve_primary", ConfigDict()
    )
    model = VideoEventClassifier(
        backbone=backbone,
        frame_feature_dim=frame_feature_dim,
        local_backbone=local_backbone,
        hidden_dim=int(cfg.model.hidden_dim),
        num_labels=len(LABELS),
        fusion=str(cfg.model.temporal_fusion),
        num_layers=int(cfg.model.temporal_layers),
        num_heads=int(cfg.model.temporal_heads),
        dropout=float(cfg.model.dropout),
        max_frames=max(int(cfg.video.num_frames), int(cfg.video.get("candidate_num_frames", cfg.video.num_frames))),
        view_fusion=str(cfg.model.get("view_fusion", "single")),
        roi_meta_dim=int(cfg.model.get("roi_meta_dim", ROI_META_DIM)),
        roi_count=int(cfg.model.get("roi_count", 1)),
        roi_verifier_min_confidence=float(
            cfg.model.get("roi_verifier_min_confidence", 0.6)
        ),
        roi_verifier_class_indices=tuple(
            cfg.model.get("roi_verifier_class_indices", [])
        ),
        view_fusion_layers=int(cfg.model.get("view_fusion_layers", 2)),
        view_fusion_heads=int(cfg.model.get("view_fusion_heads", cfg.model.temporal_heads)),
        view_fusion_positive_delta=float(
            cfg.model.get("view_fusion_positive_delta", 0.5)
        ),
        view_fusion_negative_delta=float(cfg.model.get("view_fusion_negative_delta", 2.0)),
        view_fusion_direct_delta_gate_init=float(
            cfg.model.get("view_fusion_direct_delta_gate_init", 0.05)
        ),
        view_fusion_residual_gate_init=float(
            cfg.model.get("view_fusion_residual_gate_init", 0.0)
        ),
        view_fusion_positive_delta_per_class=cfg.model.get(
            "view_fusion_positive_delta_per_class", None
        ),
        view_fusion_negative_delta_per_class=cfg.model.get(
            "view_fusion_negative_delta_per_class", None
        ),
        event_topk=int(cfg.model.get("event_topk", 12)),
        context_frames=int(cfg.model.get("context_frames", 4)),
        event_topk_strategy=str(
            cfg.model.get("event_topk_strategy", "shared_max")
        ),
        event_topk_per_class=int(cfg.model.get("event_topk_per_class", 4)),
        event_topk_class_indices=tuple(
            cfg.model.get("event_topk_class_indices", [0, 1])
        ),
        event_topk_gradient=str(cfg.model.get("event_topk_gradient", "detached")),
        event_topk_temperature=float(cfg.model.get("event_topk_temperature", 1.0)),
        event_topk_gradient_scale=float(
            cfg.model.get("event_topk_gradient_scale", 1.0)
        ),
        event_anchor_topk_per_class=per_label_int_tuple(
            cfg.model.get("event_anchor_topk_per_class", {"shot": 2, "save": 2, "set_piece": 1}),
            default=0,
        ),
        event_anchor_class_indices=tuple(
            cfg.model.get("event_anchor_class_indices", list(range(len(LABELS))))
        ),
        event_anchor_offsets=tuple(
            cfg.model.get("event_anchor_offsets", [-2, -1, 0, 1, 2])
        ),
        event_anchor_nms_radius=int(cfg.model.get("event_anchor_nms_radius", 2)),
        event_anchor_max_frames=int(cfg.model.get("event_anchor_max_frames", cfg.video.num_frames)),
        uniform_frames=int(cfg.model.get("uniform_frames", cfg.video.num_frames)),
        uniform_event_gate_init=per_label_float_tuple(
            cfg.model.get(
                "uniform_event_gate_init",
                {"shot": 0.25, "save": 0.5, "set_piece": 0.25},
            ),
            default=0.25,
        ),
        uniform_event_gate_mode=str(
            cfg.model.get("uniform_event_gate_mode", "static")
        ),
        uniform_event_gate_hidden=int(
            cfg.model.get("uniform_event_gate_hidden", 32)
        ),
        gradient_checkpointing=bool(cfg.model.get("gradient_checkpointing", False)),
        gradient_checkpointing_mode=str(
            cfg.model.get("gradient_checkpointing_mode", "full_extractor")
        ),
        backbone_frame_chunk_size=int(
            cfg.model.get("backbone_frame_chunk_size", 0)
        ),
        frame_feature_mode=str(
            cfg.model.get("frame_feature_mode", "last_cls_patch_mean")
        ),
        frame_feature_layers=_coerce_frame_feature_layers(
            cfg.model.get("frame_feature_layers", [-12, -8, -4, -1])
        ),
        frame_patch_pool=str(cfg.model.get("frame_patch_pool", "mean")),
        temporal_difference_enabled=bool(
            temporal_difference_cfg.get("enabled", False)
        ),
        temporal_difference_mode=str(
            temporal_difference_cfg.get("mode", "delta2")
        ),
        temporal_difference_gate_init=float(
            temporal_difference_cfg.get("gate_init", 0.25)
        ),
        object_spatial_aux_enabled=bool(
            object_spatial_aux_cfg.get("enabled", False)
        ),
        object_spatial_aux_temporal_layers=int(
            object_spatial_aux_cfg.get("temporal_layers", 1)
        ),
        object_spatial_aux_topk_ratio=float(
            object_spatial_aux_cfg.get("topk_ratio", 0.03)
        ),
        object_spatial_aux_temperature=float(
            object_spatial_aux_cfg.get("temperature", 0.7)
        ),
        object_spatial_aux_residual_max_delta=float(
            object_spatial_aux_cfg.get("residual_max_delta", 1.0)
        ),
        object_spatial_aux_representation_only=bool(
            object_spatial_aux_cfg.get("representation_only", False)
        ),
        ball_goal_relation_aux_enabled=bool(
            ball_goal_relation_aux_cfg.get("enabled", False)
        ),
        ball_goal_relation_aux_targets=len(RELATION_NAMES),
        ball_goal_relation_aux_topk_candidates=int(
            ball_goal_relation_aux_cfg.get("topk_candidates", 3)
        ),
        ball_goal_relation_aux_candidate_nms_radius=int(
            ball_goal_relation_aux_cfg.get("candidate_nms_radius", 2)
        ),
        ball_goal_relation_aux_grid_size=(
            parse_image_size(cfg.video.image_size)[0]
            // int(object_spatial_aux_cfg.get("patch_size", 16)),
            parse_image_size(cfg.video.image_size)[1]
            // int(object_spatial_aux_cfg.get("patch_size", 16)),
        ),
        ball_goal_relation_aux_context_radii=tuple(
            ball_goal_relation_aux_cfg.get(
                "context_radii_patches", [1.5, 4.0, 8.0]
            )
        ),
        spatial_attention_enabled=bool(
            spatial_attention_cfg.get("enabled", False)
        ),
        spatial_attention_dim=int(
            spatial_attention_cfg.get("attention_dim", 256)
        ),
        spatial_attention_queries_per_class=int(
            spatial_attention_cfg.get("queries_per_class", 2)
        ),
        spatial_attention_context_layers=int(
            spatial_attention_cfg.get("context_layers", 1)
        ),
        spatial_attention_temporal_layers=int(
            spatial_attention_cfg.get("temporal_layers", 2)
        ),
        spatial_attention_heads=int(
            spatial_attention_cfg.get(
                "num_heads", cfg.model.temporal_heads
            )
        ),
        spatial_attention_gate_init=float(
            spatial_attention_cfg.get("gate_init", 0.05)
        ),
        spatial_attention_dynamic_query_scale_init=float(
            spatial_attention_cfg.get("dynamic_query_scale_init", 0.0)
        ),
        spatial_attention_actionness_query_scale_init=float(
            spatial_attention_cfg.get("actionness_query_scale_init", 0.0)
        ),
        spatial_attention_attention_mode=str(
            spatial_attention_cfg.get("attention_mode", "softmax")
        ),
        spatial_attention_topk_ratio=float(
            spatial_attention_cfg.get("topk_ratio", 0.03)
        ),
        spatial_attention_residual_max_delta=float(
            spatial_attention_cfg.get("residual_max_delta", 1.0)
        ),
        spatial_attention_mode=str(
            spatial_attention_cfg.get("mode", "residual")
        ),
        spatial_attention_patch_mode=str(
            spatial_attention_cfg.get("patch_mode", "last_layer")
        ),
        spatial_attention_feature_layers=_coerce_frame_feature_layers(
            spatial_attention_cfg.get("feature_layers", [-8, -4, -1])
        ),
        spatial_attention_fusion_gate_init=float(
            spatial_attention_cfg.get("fusion_gate_init", 0.15)
        ),
        spatial_attention_correction_dim=int(
            spatial_attention_cfg.get("correction_dim", 128)
        ),
        spatial_attention_correction_gate_init=float(
            spatial_attention_cfg.get("correction_gate_init", 0.25)
        ),
        spatial_attention_correction_max_delta=float(
            spatial_attention_cfg.get("correction_max_delta", 2.0)
        ),
        spatial_attention_return_maps=bool(
            spatial_attention_cfg.get("return_attention_maps", False)
        ),
        spatial_token_pooling_enabled=bool(
            spatial_token_pooling_cfg.get("enabled", False)
        ),
        spatial_token_pooling_num_queries=int(
            spatial_token_pooling_cfg.get("num_queries", 4)
        ),
        spatial_token_pooling_attention_dim=int(
            spatial_token_pooling_cfg.get("attention_dim", 256)
        ),
        spatial_token_pooling_slot_dropout=float(
            spatial_token_pooling_cfg.get("slot_dropout", 0.1)
        ),
        spatial_token_pooling_gate_init=float(
            spatial_token_pooling_cfg.get("gate_init", 0.25)
        ),
        spatial_token_pooling_fusion_mode=str(
            spatial_token_pooling_cfg.get("fusion_mode", "legacy_sequence")
        ),
        spatial_token_pooling_relation_layers=int(
            spatial_token_pooling_cfg.get("relation_layers", 1)
        ),
        spatial_token_pooling_patch_layers=_coerce_frame_feature_layers(
            spatial_token_pooling_cfg.get("patch_layers", [])
        ),
        spatial_token_pooling_legacy_patch_layers=bool(
            spatial_token_pooling_cfg.get("legacy_patch_layers", True)
        ),
        spatial_token_pooling_return_maps=bool(
            spatial_token_pooling_cfg.get("return_attention_maps", False)
        ),
        class_evidence_enabled=bool(
            class_evidence_cfg.get("enabled", False)
        ),
        class_evidence_mode=str(
            class_evidence_cfg.get("mode", "logit_residual")
        ),
        class_evidence_attention_dim=int(
            class_evidence_cfg.get("attention_dim", 128)
        ),
        class_evidence_hidden_dim=int(
            class_evidence_cfg.get("hidden_dim", 192)
        ),
        class_evidence_queries_per_class=int(
            class_evidence_cfg.get("queries_per_class", 2)
        ),
        class_evidence_temporal_layers=int(
            class_evidence_cfg.get("temporal_layers", 1)
        ),
        class_evidence_temporal_heads=int(
            class_evidence_cfg.get("temporal_heads", 4)
        ),
        class_evidence_bottleneck_frames=int(
            class_evidence_cfg.get("bottleneck_frames", 8)
        ),
        class_evidence_anti_erasure_no_evidence=bool(
            class_evidence_cfg.get("anti_erasure_no_evidence", False)
        ),
        class_evidence_causal_local_counterfactual=(
            float(cfg.train.get("class_evidence_signed_causal_loss_weight", 0.0) or 0.0) > 0
        ),
        save_shot_conditioning=bool(
            class_evidence_cfg.get("save_shot_conditioning", False)
        ),
        evidence_feature_dim=int(
            int(cfg.model.get("evidence_features", ConfigDict()).get("feature_dim", 19))
            + (
                5
                if cfg.model.get("evidence_features", ConfigDict()).get(
                    "audio_index_dir", None
                )
                else 0
            )
        )
        if bool(cfg.model.get("evidence_features", ConfigDict()).get("enabled", False))
        else 0,
        evidence_gate_init=float(
            cfg.model.get("evidence_features", ConfigDict()).get("gate_init", 0.0)
        ),
        save_shot_condition_hidden_dim=int(
            class_evidence_cfg.get("save_shot_condition_hidden_dim", 256)
        ),
        save_shot_condition_window_frames=int(
            class_evidence_cfg.get("save_shot_condition_window_frames", 8)
        ),
        set_piece_subtype_count=(
            len(SET_PIECE_SUBTYPES)
            if float(cfg.train.get("set_piece_subtype_loss_weight", 0.0) or 0.0) > 0
            else 0
        ),
        class_evidence_topk_per_class=per_label_int_tuple(
            class_evidence_cfg.get(
                "topk_per_class", {"shot": 4, "save": 4, "set_piece": 4}
            ),
            default=4,
        ),
        class_evidence_context_frames_per_class=per_label_int_tuple(
            class_evidence_cfg.get(
                "context_frames_per_class", {"shot": 2, "save": 2, "set_piece": 4}
            ),
            default=2,
        ),
        class_evidence_positive_max_per_class=per_label_float_tuple(
            class_evidence_cfg.get(
                "positive_max_per_class", {"shot": 0.5, "save": 0.6, "set_piece": 1.2}
            ),
            default=0.5,
        ),
        class_evidence_negative_max_per_class=per_label_float_tuple(
            class_evidence_cfg.get(
                "negative_max_per_class", {"shot": 1.5, "save": 1.8, "set_piece": 0.6}
            ),
            default=1.0,
        ),
        class_evidence_return_maps=bool(
            class_evidence_cfg.get("return_attention_maps", False)
        ),
        highres_glimpse_enabled=bool(
            highres_glimpse_cfg.get("enabled", False)
        ),
        highres_glimpse_candidates=int(
            highres_glimpse_cfg.get("candidates", 2)
        ),
        highres_glimpse_frames_per_candidate=int(
            highres_glimpse_cfg.get("frames_per_candidate", 4)
        ),
        highres_glimpse_crop_size=int(
            highres_glimpse_cfg.get("crop_size", 384)
        ),
        highres_glimpse_attention_dim=int(
            highres_glimpse_cfg.get("attention_dim", 128)
        ),
        highres_glimpse_min_scale=tuple(
            highres_glimpse_cfg.get("min_scale", [0.24, 0.30])
        ),
        highres_glimpse_max_scale=tuple(
            highres_glimpse_cfg.get("max_scale", [0.58, 0.72])
        ),
        highres_glimpse_nms_radius=int(
            highres_glimpse_cfg.get("nms_radius", 2)
        ),
        highres_glimpse_fusion_init=float(
            highres_glimpse_cfg.get("fusion_init", 0.05)
        ),
        highres_glimpse_fusion_mode=str(
            highres_glimpse_cfg.get("fusion_mode", "joint_replace")
        ),
        highres_glimpse_residual_max_delta=float(
            highres_glimpse_cfg.get("residual_max_delta", 1.0)
        ),
        highres_glimpse_residual_gate_init=float(
            highres_glimpse_cfg.get("residual_gate_init", 0.10)
        ),
        highres_glimpse_motion_prior_weight=float(
            highres_glimpse_cfg.get("motion_prior_weight", 0.0)
        ),
        highres_glimpse_attention_temperature=float(
            highres_glimpse_cfg.get("attention_temperature", 1.0)
        ),
        videomae_temporal_gate_init=float(
            cfg.model.get("videomaev2", ConfigDict()).get(
                "temporal_gate_init", 0.25
            )
        ),
        response_curve_primary_enabled=bool(
            response_curve_primary_cfg.get("enabled", False)
        ),
        response_curve_pooling=str(
            response_curve_primary_cfg.get("pooling", "topk_lse")
        ),
        response_curve_topk=int(response_curve_primary_cfg.get("topk", 3)),
        response_curve_blend_weight=float(
            response_curve_primary_cfg.get("blend_weight", 1.0)
        ),
        response_curve_head=str(
            response_curve_primary_cfg.get("curve_head", "shared_frame")
        ),
        response_curve_hidden_dim=int(
            response_curve_primary_cfg.get("curve_hidden_dim", 128)
        ),
        response_curve_kernel_size=int(
            response_curve_primary_cfg.get("curve_kernel_size", 3)
        ),
        response_curve_dropout=(
            None
            if response_curve_primary_cfg.get("curve_dropout", None) is None
            else float(response_curve_primary_cfg.get("curve_dropout"))
        ),
        temporal_stem_kernel_size=int(
            cfg.model.get("temporal_stem", ConfigDict()).get("kernel_size", 3)
        ),
        temporal_stem_dilations=tuple(
            cfg.model.get("temporal_stem", ConfigDict()).get("dilations", [1, 2, 4])
        ),
        temporal_stem_bottleneck_frames=int(
            cfg.model.get("temporal_stem", ConfigDict()).get("bottleneck_frames", 8)
        ),
    )
    matched_keys: set[str] = set()
    if init_checkpoint:
        matched_keys = load_model_init_checkpoint(
            model,
            init_checkpoint,
            checkpoint=init_checkpoint_data,
            expected_backbone=str(cfg.model.get("backbone", "")),
            strict=bool(cfg.model.get("init_checkpoint_strict", False)),
        )
    if (
        model.class_evidence_enabled
        and model.class_evidence_mode == "independent_preprojection"
    ):
        # The direct class-conditioned branch bypasses the legacy shared
        # 6144->hidden projection and shared temporal/heads. Keep them only for
        # checkpoint compatibility; do not waste optimizer state on dead paths.
        for module in (
            model.frame_proj,
            model.frame_patch_attn,
            model.temporal,
            model.head,
            model.frame_event_head,
            model.response_curve_head,
        ):
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad = False
        model.response_curve_logit_scale.requires_grad = False
        model.response_curve_logit_bias.requires_grad = False
        print(
            "Froze legacy shared projection/temporal/heads; training direct "
            "independent preprojection evidence spaces",
            flush=True,
        )
    if model.highres_glimpse_enabled and bool(
        highres_glimpse_cfg.get("freeze_selector", True)
    ):
        for module in (
            model.frame_proj,
            model.frame_event_head,
            model.frame_patch_attn,
            model.multi_layer_frame_adapter,
            model.multi_layer_patch_adapter,
        ):
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad = False
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False
        model.backbone_has_trainable_params = False
        print(
            "Froze transferred global frame selector; training crop/fusion and temporal task head",
            flush=True,
        )
    if (
        model.class_evidence_enabled
        and model.class_evidence_mode == "global_local_temporal"
        and not any(
            key.startswith("class_evidence_head.temporal.")
            or key.startswith("class_evidence_head.classifier.")
            or key.startswith("class_evidence_head.frame_event_head.")
            for key in matched_keys
        )
    ):
        model.initialize_class_evidence_temporal_from_primary()
        print(
            "Initialized global-local evidence temporal/head/frame head from "
            "the strong primary checkpoint branch",
            flush=True,
        )
    if model.fusion == "uniform_event_dual_transformer":
        if not any(
            key.startswith("uniform_temporal.")
            or key.startswith("uniform_head.")
            for key in matched_keys
        ):
            model.initialize_uniform_from_primary()
            print(
                "Initialized uniform temporal branch from primary checkpoint branch",
                flush=True,
            )
        uniform_init_checkpoint = str(
            cfg.model.get("uniform_init_checkpoint", "") or ""
        )
        if uniform_init_checkpoint:
            load_uniform_branch_init_checkpoint(model, uniform_init_checkpoint)
        event_init_checkpoint = str(
            cfg.model.get("event_init_checkpoint", "") or ""
        )
        if event_init_checkpoint:
            load_event_branch_init_checkpoint(model, event_init_checkpoint)
        if bool(cfg.model.get("freeze_event_branch_for_gate", False)):
            model.freeze_temporal_branches_for_gate_parameters()
            print(
                "Froze uniform/event temporal branches; training adaptive gate only",
                flush=True,
            )
        elif bool(cfg.model.get("freeze_uniform_reference", False)):
            model.freeze_uniform_reference_parameters()
            print(
                "Froze E1 uniform reference path; training event temporal/head and gate only",
                flush=True,
            )
    if model.dual_view and model.local_backbone is not None and not any(
        key.startswith("local_backbone.") for key in matched_keys
    ):
        model.initialize_local_backbone_from_global()
        print("Initialized local ROI backbone from global checkpoint backbone", flush=True)
    if model.dual_view and not any(key.startswith("local_") for key in matched_keys):
        model.initialize_local_from_global()
        print("Initialized local ROI branch from global checkpoint branch", flush=True)
    if bool(cfg.model.get("freeze_loaded_backbone", False)):
        model.freeze_all_backbone_parameters()
        print("Froze loaded DINO/LoRA backbone for resource-controlled fine-tuning", flush=True)
    controlled_scope = str(
        cfg.model.get("controlled_online_train_scope", "all") or "all"
    ).strip().lower()
    if controlled_scope not in {
        "all",
        "temporal_heads",
        "lora_temporal_heads",
        "spatial_residual",
        "highres_residual",
        "object_spatial_aux",
        "external_evidence_only",
        "mechanism_a",
    }:
        raise ValueError(
            "model.controlled_online_train_scope must be all, temporal_heads, "
            "lora_temporal_heads, spatial_residual, highres_residual, "
            "object_spatial_aux, external_evidence_only, or mechanism_a"
        )
    if controlled_scope != "all":
        temporal_head_prefixes = (
            "class_evidence_head.temporal.",
            "class_evidence_head.temporals.",
            "class_evidence_head.classifier.",
            "class_evidence_head.frame_event_head.",
            "class_evidence_head.clip_heads.",
            "class_evidence_head.frame_heads.",
            "class_evidence_head.set_piece_subtype_head.",
            "class_evidence_head.save_shot_condition_mlp.",
            "class_evidence_head.evidence_projs.",
            "class_evidence_head.evidence_gates",
            "temporal.",
            "head.",
            "frame_event_head.",
            "response_curve_head.",
        )
        mechanism_a_event_prefixes = (
            "frame_proj.",
            "temporal.",
            "head.",
            "frame_event_head.",
        )
        mechanism_a_aux_prefixes = [
            "object_spatial_aux.heatmap_head.",
        ]
        if bool(
            getattr(model.object_spatial_aux, "relation_aux_enabled", False)
        ):
            mechanism_a_aux_prefixes.extend(
                ["object_spatial_aux.relation_head.", "object_spatial_aux.context_predictor."]
            )
        trainable_total = 0
        temporal_head_total = 0
        lora_total = 0
        for name, parameter in model.named_parameters():
            is_temporal_head = name.startswith(temporal_head_prefixes)
            is_lora = (
                name.startswith(("backbone.", "local_backbone."))
                and name.endswith((".lora_a", ".lora_b"))
            )
            is_spatial_residual = name.startswith(
                ("spatial_attention.", "spatial_patch_adapter.")
            ) or name == "spatial_patch_mix_logits"
            is_object_spatial_aux = name.startswith("object_spatial_aux.")
            is_highres_residual = name.startswith(
                (
                    "highres_crop_query.",
                    "highres_crop_key.",
                    "highres_crop_scale.",
                    "highres_event_query_embedding",
                    "highres_local_motion_fusion.",
                    "highres_local_attention.",
                    "highres_local_classifier.",
                    "highres_residual_head.",
                    "highres_residual_gate_logits",
                )
            )
            if controlled_scope == "spatial_residual":
                allowed = is_spatial_residual
            elif controlled_scope == "highres_residual":
                allowed = is_highres_residual
            elif controlled_scope == "object_spatial_aux":
                allowed = is_object_spatial_aux
            elif controlled_scope == "external_evidence_only":
                allowed = name.startswith(
                    (
                        "class_evidence_head.evidence_projs.",
                        "class_evidence_head.evidence_gates",
                    )
                )
            elif controlled_scope == "mechanism_a":
                allowed = (
                    is_lora
                    or name.startswith(mechanism_a_event_prefixes)
                    or name.startswith(tuple(mechanism_a_aux_prefixes))
                )
            else:
                allowed = is_temporal_head or (
                    controlled_scope == "lora_temporal_heads" and is_lora
                )
            if not allowed:
                parameter.requires_grad = False
                continue
            if not parameter.requires_grad:
                continue
            trainable_total += parameter.numel()
            if is_temporal_head and controlled_scope != "spatial_residual":
                temporal_head_total += parameter.numel()
            if is_lora:
                lora_total += parameter.numel()
        if trainable_total <= 0:
            raise RuntimeError(
                f"controlled online scope={controlled_scope} left no trainable parameters"
            )
        if controlled_scope == "lora_temporal_heads" and lora_total <= 0:
            raise RuntimeError("stage2 requested trainable LoRA but no LoRA parameters are active")
        if controlled_scope == "mechanism_a":
            trainable_names = [
                name
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            ]
            for required_prefix in mechanism_a_event_prefixes + tuple(mechanism_a_aux_prefixes):
                if not any(
                    name.startswith(required_prefix) for name in trainable_names
                ):
                    raise RuntimeError(
                        f"mechanism_a has no trainable {required_prefix} parameters"
                    )
            if lora_total <= 0:
                raise RuntimeError(
                    "mechanism_a requires trainable shared DINO LoRA parameters"
                )
        if controlled_scope == "external_evidence_only":
            evidence_names = [
                name
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            ]
            if not evidence_names or not any(
                name == "class_evidence_head.evidence_gates"
                for name in evidence_names
            ):
                raise RuntimeError(
                    "external_evidence_only requires trainable evidence projections and gates"
                )
            unexpected = [
                name
                for name in evidence_names
                if not name.startswith(
                    (
                        "class_evidence_head.evidence_projs.",
                        "class_evidence_head.evidence_gates",
                    )
                )
            ]
            if unexpected:
                raise RuntimeError(
                    "external_evidence_only trainable allowlist violation: "
                    f"{unexpected[:20]}"
                )
        if controlled_scope in {"spatial_residual", "highres_residual", "object_spatial_aux"}:
            model.freeze_global_branch = True
        model.backbone_has_trainable_params = lora_total > 0
        print(
            f"controlled_online_train_scope={controlled_scope} "
            f"trainable_params={trainable_total} temporal_head_params={temporal_head_total} "
            f"lora_params={lora_total}",
            flush=True,
        )
    if bool(cfg.model.get("freeze_global_branch", False)):
        model.freeze_global_parameters()
        if model.dual_view:
            message = "Froze global branch; training local ROI branch and class gate only"
        else:
            message = "Froze global DINO/temporal reference; training auxiliary branch only"
        print(message, flush=True)
    if bool(cfg.model.get("freeze_spatial_probe", False)):
        model.freeze_spatial_probe_parameters()
        print("Froze learned spatial probe; training fusion parameters only", flush=True)
    return model.to(device)


def resolve_runtime_topology(cfg: Any, device: torch.device) -> dict[str, Any]:
    gpu_ids = [int(gpu_id) for gpu_id in cfg.get("gpu_ids", [])]
    if device.type == "cuda":
        world_size = max(len(gpu_ids), 1)
    else:
        world_size = 1

    train_cfg = cfg.train
    data_cfg = cfg.data
    eval_cfg = cfg.eval
    is_distributed = distributed_training_active()
    uses_per_gpu_config = False

    if train_cfg.get("per_gpu_batch_size") is not None:
        per_gpu_batch_size = int(train_cfg.per_gpu_batch_size)
        if per_gpu_batch_size <= 0:
            raise ValueError("train.per_gpu_batch_size must be positive")
        train_cfg.batch_size = (per_gpu_batch_size if is_distributed else per_gpu_batch_size * world_size)
        uses_per_gpu_config = True
    else:
        per_gpu_batch_size = float(train_cfg.batch_size) / world_size

    if train_cfg.get("lr_per_gpu") is not None:
        lr_per_gpu = float(train_cfg.lr_per_gpu)
        train_cfg.lr = lr_per_gpu * world_size
        uses_per_gpu_config = True
    else:
        lr_per_gpu = float(train_cfg.lr) / world_size

    if train_cfg.get("backbone_lr_per_gpu") is not None:
        backbone_lr_per_gpu = float(train_cfg.backbone_lr_per_gpu)
        train_cfg.backbone_lr = backbone_lr_per_gpu * world_size
        uses_per_gpu_config = True
    else:
        backbone_lr_per_gpu = float(train_cfg.backbone_lr) / world_size

    global_backbone_lr_per_gpu = float(
        train_cfg.get("global_backbone_lr_per_gpu", backbone_lr_per_gpu)
    )
    local_backbone_lr_per_gpu = float(
        train_cfg.get("local_backbone_lr_per_gpu", backbone_lr_per_gpu)
    )
    train_cfg["global_backbone_lr"] = global_backbone_lr_per_gpu * world_size
    train_cfg["local_backbone_lr"] = local_backbone_lr_per_gpu * world_size
    if (
        train_cfg.get("global_backbone_lr_per_gpu") is not None
        or train_cfg.get("local_backbone_lr_per_gpu") is not None
    ):
        uses_per_gpu_config = True

    if eval_cfg.get("per_gpu_batch_size") is not None:
        eval_per_gpu_batch_size = int(eval_cfg.per_gpu_batch_size)
        if eval_per_gpu_batch_size <= 0:
            raise ValueError("eval.per_gpu_batch_size must be positive")
        eval_cfg.batch_size = (eval_per_gpu_batch_size if is_distributed else eval_per_gpu_batch_size * world_size)
        uses_per_gpu_config = True
    else:
        eval_per_gpu_batch_size = float(eval_cfg.batch_size) / world_size

    if data_cfg.get("num_workers_per_gpu") is not None:
        num_workers_per_gpu = int(data_cfg.num_workers_per_gpu)
        if num_workers_per_gpu < 0:
            raise ValueError("data.num_workers_per_gpu must be non-negative")
        data_cfg.num_workers = (num_workers_per_gpu if is_distributed else num_workers_per_gpu * world_size)
        uses_per_gpu_config = True
    else:
        num_workers_per_gpu = float(data_cfg.num_workers) / world_size

    grad_accum_steps = int(train_cfg.grad_accum_steps)
    if grad_accum_steps <= 0:
        raise ValueError("train.grad_accum_steps must be positive")
    topology = {
        "world_size": world_size,
        "gpu_ids": gpu_ids,
        "uses_per_gpu_config": uses_per_gpu_config,
        "train_per_gpu_batch_size": per_gpu_batch_size,
        "train_global_batch_size": int(train_cfg.batch_size) * (world_size if is_distributed else 1),
        "grad_accum_steps": grad_accum_steps,
        "effective_batch_size": int(train_cfg.batch_size) * (world_size if is_distributed else 1) * grad_accum_steps,
        "lr_per_gpu": lr_per_gpu,
        "lr_global": float(train_cfg.lr),
        "backbone_lr_per_gpu": backbone_lr_per_gpu,
        "backbone_lr_global": float(train_cfg.backbone_lr),
        "global_backbone_lr_per_gpu": global_backbone_lr_per_gpu,
        "global_backbone_lr_global": float(train_cfg.global_backbone_lr),
        "local_backbone_lr_per_gpu": local_backbone_lr_per_gpu,
        "local_backbone_lr_global": float(train_cfg.local_backbone_lr),
        "eval_per_gpu_batch_size": eval_per_gpu_batch_size,
        "eval_global_batch_size": int(eval_cfg.batch_size) * (world_size if is_distributed else 1),
        "num_workers_per_gpu": num_workers_per_gpu,
        "num_workers_global": int(data_cfg.num_workers) * (world_size if is_distributed else 1),
        "distributed": is_distributed,
    }
    cfg.runtime_topology = to_config(topology)
    print(
        "runtime_topology "
        f"world_size={world_size} gpu_ids={gpu_ids} "
        f"train_batch_per_gpu/global/effective={per_gpu_batch_size}/{topology['train_global_batch_size']}/{topology['effective_batch_size']} "
        f"grad_accum={grad_accum_steps} lr_per_gpu/global={lr_per_gpu:.8g}/{float(train_cfg.lr):.8g} "
        f"backbone_lr_per_gpu/global={backbone_lr_per_gpu:.8g}/{float(train_cfg.backbone_lr):.8g} "
        f"global_backbone_lr_per_gpu/global={global_backbone_lr_per_gpu:.8g}/{float(train_cfg.global_backbone_lr):.8g} "
        f"local_backbone_lr_per_gpu/global={local_backbone_lr_per_gpu:.8g}/{float(train_cfg.local_backbone_lr):.8g} "
        f"eval_batch_per_gpu/global={eval_per_gpu_batch_size}/{topology['eval_global_batch_size']} "
        f"workers_per_gpu/global={num_workers_per_gpu}/{topology['num_workers_global']}",
        flush=True,
    )
    resume_cfg = get_resume_cfg(cfg)
    if (
        uses_per_gpu_config
        and bool(resume_cfg.get("enabled", False))
        and bool(resume_cfg.get("load_optimizer", True))
    ):
        print(
            "WARN resume.load_optimizer=true: checkpoint optimizer/scheduler LR may override "
            "the topology-resolved LR; use resume=false for a newly scaled run.",
            flush=True,
        )
    return topology


def configure_runtime_threads(cfg: Any) -> None:
    data_cfg = cfg.get("data", ConfigDict())
    for env_name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(env_name, str(data_cfg.get("cpu_thread_env", 1)))
    cv2_threads = int(data_cfg.get("opencv_num_threads", 0))
    try:
        cv2.setNumThreads(cv2_threads)
    except cv2.error:
        pass


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed model, augmentation, CUDA, and DataLoader RNG sources."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def football_worker_init(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    try:
        cv2.setNumThreads(0)
    except cv2.error:
        pass
    torch.set_num_threads(1)


def distributed_training_active() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def gather_objects_to_main(payload: Any) -> list[Any] | None:
    """Gather small diagnostic payloads without duplicating console output."""
    if not distributed_training_active():
        return [payload]
    gathered: list[Any] | None = (
        [None for _ in range(dist.get_world_size())]
        if dist.get_rank() == 0 else None
    )
    dist.gather_object(payload, gathered, dst=0)
    return gathered


def sum_float_mappings(mappings: Iterable[dict[str, float]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for mapping in mappings:
        for key, value in mapping.items():
            totals[key] = totals.get(key, 0.0) + float(value)
    return totals


def add_slot_weighted_no_evidence_metrics(values: dict[str, float]) -> None:
    """Derive true mass from accumulated numerator/slots diagnostics."""
    stems = [
        "class_evidence_no_evidence_event_mass",
        "class_evidence_no_evidence_background_mass",
    ]
    stems.extend(
        f"class_evidence_no_evidence_{kind}_mass_{label}"
        for kind in ("event", "background")
        for label in LABELS
    )
    for stem in stems:
        numerator = float(values.get(f"{stem}_numerator", 0.0))
        slots = float(values.get(f"{stem}_valid_slots", 0.0))
        if slots > 0:
            values[f"{stem}_slot_weighted"] = numerator / slots


class DistributedEvalSampler(Sampler[int]):
    """Shard validation without padding or duplicating any sample."""

    def __init__(self, dataset: Dataset, *, rank: int, world_size: int) -> None:
        self.dataset = dataset
        self.rank = int(rank)
        self.world_size = int(world_size)
        if self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise ValueError(
                f"invalid distributed eval rank/world={self.rank}/{self.world_size}"
            )

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        remaining = len(self.dataset) - self.rank
        return 0 if remaining <= 0 else (remaining + self.world_size - 1) // self.world_size


class VideoEventBalancedSampler(Sampler[int]):
    """Hierarchical long-video sampler with DDP-consistent batch diversity.

    A global effective-batch window first selects a configurable number of
    distinct long videos, then draws an event anchor or background record from
    each selected video. Every rank deterministically builds the same global
    order and consumes a strided shard, so concurrent DDP microbatches retain
    the intended video diversity without cross-rank communication.
    """

    def __init__(
        self,
        dataset: Dataset,
        records: Sequence[Record],
        *,
        num_replicas: int = 1,
        rank: int = 0,
        local_batch_size: int = 1,
        grad_accum_steps: int = 1,
        drop_last: bool = True,
        seed: int = 0,
        positive_fraction: float | None = None,
        unique_videos_per_effective_batch: int = 0,
    ) -> None:
        if len(dataset) != len(records):
            raise ValueError(
                "video/event-balanced sampler requires one record per dataset item"
            )
        self.dataset = dataset
        self.records = records
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.local_batch_size = max(int(local_batch_size), 1)
        self.grad_accum_steps = max(int(grad_accum_steps), 1)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        if self.num_replicas <= 0 or not 0 <= self.rank < self.num_replicas:
            raise ValueError(
                f"invalid sampler rank/world={self.rank}/{self.num_replicas}"
            )

        self.global_batch_size = self.local_batch_size * self.num_replicas
        self.effective_batch_size = (
            self.global_batch_size * self.grad_accum_steps
        )
        if self.drop_last:
            self.total_size = (
                len(self.dataset) // self.global_batch_size
            ) * self.global_batch_size
        else:
            self.total_size = math.ceil(
                len(self.dataset) / self.global_batch_size
            ) * self.global_batch_size
        self.num_samples = self.total_size // self.num_replicas

        pools: dict[tuple[str, str], dict[str, list[int]]] = {}
        for index, record in enumerate(records):
            video_key = (record.source, record.video_id)
            video_pools = pools.setdefault(
                video_key, {"event": [], "background": []}
            )
            pool_name = "background" if record.is_negative else "event"
            video_pools[pool_name].append(index)
        if not pools:
            raise ValueError("video/event-balanced sampler received no videos")
        self.pools = pools
        self.video_keys = sorted(pools)

        observed_positive_fraction = (
            sum(not record.is_negative for record in records) / max(len(records), 1)
        )
        self.positive_fraction = (
            observed_positive_fraction
            if positive_fraction is None
            else float(positive_fraction)
        )
        if not 0.0 <= self.positive_fraction <= 1.0:
            raise ValueError("sampler positive_fraction must be in [0, 1]")

        requested_unique = int(unique_videos_per_effective_batch)
        if requested_unique <= 0:
            requested_unique = min(
                len(self.video_keys), self.effective_batch_size
            )
        self.unique_videos_per_effective_batch = min(
            max(requested_unique, 1),
            len(self.video_keys),
            self.effective_batch_size,
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_samples

    def summary(self) -> dict[str, Any]:
        return {
            "mode": "video_event_balanced",
            "num_videos": len(self.video_keys),
            "positive_fraction": self.positive_fraction,
            "local_batch_size": self.local_batch_size,
            "global_batch_size": self.global_batch_size,
            "effective_batch_size": self.effective_batch_size,
            "unique_videos_per_effective_batch": (
                self.unique_videos_per_effective_batch
            ),
            "samples_per_rank": self.num_samples,
            "total_samples": self.total_size,
        }

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        pool_orders: dict[tuple[tuple[str, str], str], list[int]] = {}
        pool_offsets: dict[tuple[tuple[str, str], str], int] = {}

        def draw(video_key: tuple[str, str], prefer_event: bool) -> int:
            preferred = "event" if prefer_event else "background"
            fallback = "background" if prefer_event else "event"
            pool_name = preferred if self.pools[video_key][preferred] else fallback
            state_key = (video_key, pool_name)
            order = pool_orders.get(state_key)
            offset = pool_offsets.get(state_key, 0)
            if order is None or offset >= len(order):
                order = list(self.pools[video_key][pool_name])
                rng.shuffle(order)
                pool_orders[state_key] = order
                offset = 0
            index = order[offset]
            pool_offsets[state_key] = offset + 1
            return index

        global_indices: list[int] = []
        while len(global_indices) < self.total_size:
            window_size = min(
                self.effective_batch_size,
                self.total_size - len(global_indices),
            )
            unique_count = min(
                self.unique_videos_per_effective_batch, window_size
            )
            selected_videos = rng.sample(self.video_keys, unique_count)
            video_sequence: list[tuple[str, str]] = []
            while len(video_sequence) < window_size:
                round_videos = list(selected_videos)
                rng.shuffle(round_videos)
                video_sequence.extend(
                    round_videos[: window_size - len(video_sequence)]
                )

            positive_count = int(round(window_size * self.positive_fraction))
            event_choices = [True] * positive_count + [False] * (
                window_size - positive_count
            )
            rng.shuffle(event_choices)
            global_indices.extend(
                draw(video_key, prefer_event)
                for video_key, prefer_event in zip(
                    video_sequence, event_choices
                )
            )

        if len(global_indices) != self.total_size:
            raise RuntimeError("video/event-balanced sampler built wrong epoch size")
        return iter(global_indices[self.rank : self.total_size : self.num_replicas])


def make_loader(
    dataset: Dataset,
    cfg: Any,
    *,
    is_train: bool,
    batch_size: int | None = None,
    distributed: bool = False,
) -> DataLoader:
    num_workers = int(cfg.data.num_workers)
    rank = dist.get_rank() if distributed_training_active() else 0
    local_batch_size = batch_size or int(cfg.train.batch_size)
    generator = torch.Generator()
    generator.manual_seed(
        int(cfg.get("seed", 42)) + (0 if is_train else 1) + rank * 100003
    )
    sampler = None
    train_drop_last = bool(cfg.train.get("drop_last", True)) if is_train else False
    sampler_cfg = cfg.train.get("sampler", ConfigDict()) if is_train else ConfigDict()
    sampler_mode = str(sampler_cfg.get("mode", "global_random")).strip().lower()
    if sampler_mode in ("", "random", "shuffle"):
        sampler_mode = "global_random"
    if sampler_mode not in ("global_random", "video_event_balanced"):
        raise ValueError(
            "train.sampler.mode must be global_random or video_event_balanced"
        )
    if is_train and sampler_mode == "video_event_balanced":
        records = getattr(dataset, "records", None)
        if records is None:
            raise ValueError(
                "video_event_balanced sampler requires a dataset.records sequence"
            )
        sampler = VideoEventBalancedSampler(
            dataset,
            records,
            num_replicas=(dist.get_world_size() if distributed else 1),
            rank=(dist.get_rank() if distributed else 0),
            local_batch_size=local_batch_size,
            grad_accum_steps=int(cfg.train.get("grad_accum_steps", 1)),
            drop_last=train_drop_last,
            seed=int(cfg.get("seed", 42)),
            positive_fraction=(
                None
                if sampler_cfg.get("positive_fraction", None) is None
                else float(sampler_cfg.get("positive_fraction"))
            ),
            unique_videos_per_effective_batch=int(
                sampler_cfg.get("unique_videos_per_effective_batch", 0)
            ),
        )
        if rank == 0:
            print(f"train_sampler {sampler.summary()}", flush=True)
    elif distributed:
        if not distributed_training_active():
            raise RuntimeError("distributed DataLoader requested before process-group initialization")
        if is_train:
            sampler = DistributedSampler(
                dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=True,
                seed=int(cfg.get("seed", 42)),
                drop_last=train_drop_last,
            )
        else:
            sampler = DistributedEvalSampler(
                dataset, rank=dist.get_rank(), world_size=dist.get_world_size()
            )
    loader_kwargs: dict[str, Any] = {
        "batch_size": local_batch_size,
        "shuffle": is_train and sampler is None,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": bool(cfg.data.pin_memory),
        "drop_last": train_drop_last,
        "collate_fn": football_collate,
        "generator": generator,
    }
    if num_workers > 0:
        loader_kwargs["worker_init_fn"] = football_worker_init
        loader_kwargs["persistent_workers"] = bool(cfg.data.get("persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(cfg.data.get("prefetch_factor", 4))
    return DataLoader(dataset, **loader_kwargs)


def build_optimizer(model: nn.Module, cfg: Any) -> torch.optim.Optimizer:
    global_backbone_params = []
    local_backbone_params = []
    spatial_probe_params = []
    spatial_fusion_params = []
    class_evidence_local_params = []
    class_evidence_temporal_params = []
    class_evidence_classifier_params = []
    class_evidence_no_evidence_gate_params = []
    highres_glimpse_params = []
    temporal_params = []
    classifier_head_params = []
    head_params = []
    spatial_probe_lr_per_gpu = cfg.train.get("spatial_probe_lr_per_gpu")
    spatial_fusion_lr_per_gpu = cfg.train.get("spatial_fusion_lr_per_gpu")
    use_spatial_lr_groups = any(
        value is not None
        for value in (spatial_probe_lr_per_gpu, spatial_fusion_lr_per_gpu)
    )
    class_evidence_local_lr_per_gpu = cfg.train.get(
        "class_evidence_local_lr_per_gpu"
    )
    class_evidence_temporal_lr_per_gpu = cfg.train.get(
        "class_evidence_temporal_lr_per_gpu"
    )
    class_evidence_head_lr_per_gpu = cfg.train.get(
        "class_evidence_head_lr_per_gpu"
    )
    use_class_evidence_lr_groups = any(
        value is not None
        for value in (
            class_evidence_local_lr_per_gpu,
            class_evidence_temporal_lr_per_gpu,
        )
    )
    highres_glimpse_lr_per_gpu = cfg.train.get("highres_glimpse_lr_per_gpu")
    temporal_lr_per_gpu = cfg.train.get("temporal_lr_per_gpu")
    head_lr_per_gpu = cfg.train.get("head_lr_per_gpu")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        normalized_name = name.removeprefix("module.")
        if (
            highres_glimpse_lr_per_gpu is not None
            and normalized_name.startswith("highres_")
        ):
            highres_glimpse_params.append(param)
        elif use_spatial_lr_groups and normalized_name.startswith("spatial_attention."):
            spatial_probe_params.append(param)
        elif use_spatial_lr_groups and normalized_name.startswith("spatial_feature_"):
            spatial_fusion_params.append(param)
        elif (
            use_class_evidence_lr_groups
            and normalized_name.startswith(
                (
                    "class_evidence_head.classifier.",
                    "class_evidence_head.frame_event_head.",
                    "class_evidence_head.clip_heads.",
                    "class_evidence_head.frame_heads.",
                    "class_evidence_head.set_piece_subtype_head.",
                )
            )
        ):
            class_evidence_classifier_params.append(param)
        elif (
            use_class_evidence_lr_groups
            and normalized_name.startswith(
                (
                    "class_evidence_head.temporal.",
                    "class_evidence_head.temporals.",
                )
            )
        ):
            class_evidence_temporal_params.append(param)
        elif (
            use_class_evidence_lr_groups
            and normalized_name.startswith(
                (
                    "class_evidence_head.no_evidence_gate_",
                    "class_evidence_head.evidence_gates",
                    "class_evidence_head.evidence_projs.",
                )
            )
        ):
            # The 4-param monotone saliency gate is freshly initialized, so it
            # needs a much higher LR than the warm-started evidence head: at
            # the local-group LR its logits only move ~1e-3/epoch and never
            # differentiate event vs background.
            class_evidence_no_evidence_gate_params.append(param)
        elif (
            use_class_evidence_lr_groups
            and normalized_name.startswith("class_evidence_head.")
        ):
            class_evidence_local_params.append(param)
        elif normalized_name.startswith("local_backbone."):
            local_backbone_params.append(param)
        elif normalized_name.startswith("backbone."):
            global_backbone_params.append(param)
        elif temporal_lr_per_gpu is not None and normalized_name.startswith(
            ("temporal.", "uniform_temporal.", "local_temporal.")
        ):
            temporal_params.append(param)
        elif head_lr_per_gpu is not None and normalized_name.startswith(
            (
                "head.",
                "frame_event_head.",
                "response_curve_head.",
                "uniform_head.",
                "local_head.",
                "local_frame_event_head.",
            )
        ):
            classifier_head_params.append(param)
        else:
            head_params.append(param)

    groups = []
    gpu_ids_cfg = cfg.get("gpu_ids", [])
    fallback_world_size = len(gpu_ids_cfg) if isinstance(gpu_ids_cfg, (list, tuple)) and gpu_ids_cfg else 1
    world_size = int(cfg.get("runtime_topology", {}).get("world_size", fallback_world_size))
    if head_params:
        groups.append({"name": "head", "params": head_params, "lr": float(cfg.train.lr)})
    if temporal_params:
        groups.append(
            {
                "name": "temporal",
                "params": temporal_params,
                "lr": float(temporal_lr_per_gpu) * world_size,
            }
        )
    if classifier_head_params:
        groups.append(
            {
                "name": "classifier_head",
                "params": classifier_head_params,
                "lr": float(head_lr_per_gpu) * world_size,
            }
        )
    if highres_glimpse_params:
        groups.append(
            {
                "name": "highres_glimpse",
                "params": highres_glimpse_params,
                "lr": float(highres_glimpse_lr_per_gpu) * int(
                    cfg.get("runtime_topology", {}).get("world_size", 1)
                ),
            }
        )
    if global_backbone_params:
        groups.append(
            {
                "name": "global_backbone",
                "params": global_backbone_params,
                "lr": float(cfg.train.get("global_backbone_lr", cfg.train.backbone_lr)),
            }
        )
    if local_backbone_params:
        groups.append(
            {
                "name": "local_backbone",
                "params": local_backbone_params,
                "lr": float(cfg.train.get("local_backbone_lr", cfg.train.backbone_lr)),
            }
        )
    if class_evidence_local_params:
        per_gpu_lr = (
            class_evidence_local_lr_per_gpu
            if class_evidence_local_lr_per_gpu is not None
            else cfg.train.lr_per_gpu
        )
        groups.append(
            {
                "name": "class_evidence_local",
                "params": class_evidence_local_params,
                "lr": float(per_gpu_lr) * world_size,
            }
        )
    if class_evidence_no_evidence_gate_params:
        gate_lr_per_gpu = cfg.train.get("class_evidence_gate_lr_per_gpu")
        groups.append(
            {
                "name": "class_evidence_no_evidence_gate",
                "params": class_evidence_no_evidence_gate_params,
                "lr": float(gate_lr_per_gpu) * world_size,
            }
        )
    if class_evidence_temporal_params:
        per_gpu_lr = (
            class_evidence_temporal_lr_per_gpu
            if class_evidence_temporal_lr_per_gpu is not None
            else cfg.train.lr_per_gpu
        )
        groups.append(
            {
                "name": "class_evidence_temporal",
                "params": class_evidence_temporal_params,
                "lr": float(per_gpu_lr) * world_size,
            }
        )
    if class_evidence_classifier_params:
        per_gpu_lr = (
            class_evidence_head_lr_per_gpu
            if class_evidence_head_lr_per_gpu is not None
            else (
                class_evidence_temporal_lr_per_gpu
                if class_evidence_temporal_lr_per_gpu is not None
                else cfg.train.lr_per_gpu
            )
        )
        groups.append(
            {
                "name": "class_evidence_classifier",
                "params": class_evidence_classifier_params,
                "lr": float(per_gpu_lr) * world_size,
            }
        )
    if spatial_probe_params:
        per_gpu_lr = (
            spatial_probe_lr_per_gpu
            if spatial_probe_lr_per_gpu is not None
            else cfg.train.lr_per_gpu
        )
        groups.append(
            {
                "name": "spatial_probe",
                "params": spatial_probe_params,
                "lr": float(per_gpu_lr) * world_size,
            }
        )
    if spatial_fusion_params:
        per_gpu_lr = (
            spatial_fusion_lr_per_gpu
            if spatial_fusion_lr_per_gpu is not None
            else cfg.train.lr_per_gpu
        )
        groups.append(
            {
                "name": "spatial_fusion",
                "params": spatial_fusion_params,
                "lr": float(per_gpu_lr) * world_size,
            }
        )
    if not groups:
        raise RuntimeError("No trainable parameters found")
    return torch.optim.AdamW(groups, weight_decay=float(cfg.train.weight_decay))


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Any, steps_per_epoch: int) -> torch.optim.lr_scheduler.LRScheduler:
    epochs = max(
        int(cfg.train.get("scheduler_epochs", cfg.train.epochs)), 1
    )
    total_steps = max(epochs * max(int(steps_per_epoch), 1), 1)
    warmup_cfg = cfg.train.get("warmup", ConfigDict())
    warmup_ratio = float(warmup_cfg.get("ratio", cfg.train.get("warmup_ratio", 0.0)) or 0.0)
    warmup_steps = int(warmup_cfg.get("steps", 0) or 0)
    if warmup_steps <= 0 and warmup_ratio > 0:
        warmup_steps = int(round(total_steps * warmup_ratio))
    warmup_steps = max(0, min(warmup_steps, total_steps - 1))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps + 1:
            return 1.0
        progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def optimizer_group_summary(optimizer: torch.optim.Optimizer) -> str:
    parts = []
    for index, group in enumerate(optimizer.param_groups):
        parameter_count = sum(parameter.numel() for parameter in group["params"])
        name = str(group.get("name", f"group{index}"))
        parts.append(f"{name}:lr={float(group['lr']):.8g},params={parameter_count}")
    return " ".join(parts)

def autocast_context(device: torch.device, enabled: bool, dtype_name: str):
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled and device.type == "cuda")


def safe_metrics(
    targets: np.ndarray,
    probs: np.ndarray,
    thresholds: np.ndarray,
    masks: np.ndarray | None = None,
    *,
    fbeta_beta: float = 1.0,
) -> dict[str, Any]:
    if masks is None:
        masks = np.ones_like(targets, dtype=np.float32)
    masks_bool = masks.astype(bool)
    preds = (probs >= thresholds.reshape(1, -1)).astype(np.int32)

    precision = np.zeros(targets.shape[1], dtype=np.float32)
    recall = np.zeros(targets.shape[1], dtype=np.float32)
    f1 = np.zeros(targets.shape[1], dtype=np.float32)
    fbeta = np.zeros(targets.shape[1], dtype=np.float32)
    beta = max(float(fbeta_beta), 1e-6)
    beta_sq = beta * beta
    support = np.zeros(targets.shape[1], dtype=np.int64)
    true_positives = np.zeros(targets.shape[1], dtype=np.int64)
    false_positives = np.zeros(targets.shape[1], dtype=np.int64)
    false_negatives = np.zeros(targets.shape[1], dtype=np.int64)
    ap_values = []
    auc_values = []
    for i in range(targets.shape[1]):
        valid = masks_bool[:, i]
        if not valid.any():
            ap_values.append(float("nan"))
            auc_values.append(float("nan"))
            continue
        y = targets[valid, i]
        pred = preds[valid, i]
        p_i, r_i, f_i, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        precision[i] = float(p_i)
        recall[i] = float(r_i)
        f1[i] = float(f_i)
        fbeta[i] = float(
            (1.0 + beta_sq) * p_i * r_i / max(beta_sq * p_i + r_i, 1e-12)
        )
        support[i] = int(y.sum())
        true_positives[i] = int(((y == 1) & (pred == 1)).sum())
        false_positives[i] = int(((y == 0) & (pred == 1)).sum())
        false_negatives[i] = int(((y == 1) & (pred == 0)).sum())
        if y.max() == y.min():
            ap_values.append(float("nan"))
            auc_values.append(float("nan"))
        else:
            ap_values.append(float(average_precision_score(y, probs[valid, i])))
            auc_values.append(float(roc_auc_score(y, probs[valid, i])))

    flat_valid = masks_bool.reshape(-1)
    if flat_valid.any():
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
            targets.reshape(-1)[flat_valid], preds.reshape(-1)[flat_valid], average="binary", zero_division=0
        )
    else:
        micro_precision = micro_recall = micro_f1 = 0.0
    micro_fbeta = float(
        (1.0 + beta_sq) * micro_precision * micro_recall
        / max(beta_sq * micro_precision + micro_recall, 1e-12)
    )
    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))
    macro_f1 = float(np.mean(f1))
    macro_fbeta = float(np.mean(fbeta))

    return {
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "micro_fbeta": micro_fbeta,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "macro_fbeta": macro_fbeta,
        "fbeta_beta": beta,
        "mAP": float(np.nanmean(ap_values)) if not np.isnan(ap_values).all() else 0.0,
        "mAUROC": float(np.nanmean(auc_values)) if not np.isnan(auc_values).all() else 0.0,
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "fbeta": float(fbeta[i]),
                "tp": int(true_positives[i]),
                "fp": int(false_positives[i]),
                "fn": int(false_negatives[i]),
                "support": int(support[i]),
                "valid_count": int(masks_bool[:, i].sum()),
                "unknown_count": int((~masks_bool[:, i]).sum()),
                "ap": ap_values[i],
                "auroc": auc_values[i],
                "threshold": float(thresholds[i]),
            }
            for i, label in enumerate(LABELS)
        },
    }


def combine_fixed_threshold_metrics(
    validation: dict[str, Any], external: dict[str, Any]
) -> dict[str, Any]:
    """Combine TP/FP/FN only; ranking metrics cannot be reconstructed from aggregates."""
    beta = float(validation.get("fbeta_beta", 1.0) or 1.0)
    beta_sq = beta * beta
    per_class: dict[str, Any] = {}
    totals = {"tp": 0, "fp": 0, "fn": 0}
    for label in LABELS:
        left = validation["per_class"][label]
        right = external["per_class"][label]
        tp = int(left["tp"]) + int(right["tp"]); fp = int(left["fp"]) + int(right["fp"]); fn = int(left["fn"]) + int(right["fn"])
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        fbeta = (1.0 + beta_sq) * precision * recall / max(beta_sq * precision + recall, 1e-12)
        per_class[label] = {
            "precision": precision, "recall": recall, "f1": f1,
            "fbeta": fbeta, "tp": tp, "fp": fp, "fn": fn,
            "support": int(left["support"]) + int(right["support"]),
            "threshold": float(left["threshold"]),
        }
        totals["tp"] += tp; totals["fp"] += fp; totals["fn"] += fn
    micro_p = totals["tp"] / max(totals["tp"] + totals["fp"], 1)
    micro_r = totals["tp"] / max(totals["tp"] + totals["fn"], 1)
    micro_f1 = 2.0 * micro_p * micro_r / max(micro_p + micro_r, 1e-12)
    return {
        "micro_precision": micro_p, "micro_recall": micro_r, "micro_f1": micro_f1,
        "macro_precision": float(np.mean([v["precision"] for v in per_class.values()])),
        "macro_recall": float(np.mean([v["recall"] for v in per_class.values()])),
        "macro_f1": float(np.mean([v["f1"] for v in per_class.values()])),
        "per_class": per_class,
        "ranking_metrics_available": False,
        "threshold_source": "validation15_frozen",
    }


def confidence_separation_statistics(
    targets: np.ndarray,
    probs: np.ndarray,
    masks: np.ndarray,
) -> dict[str, dict[str, float | int | None]]:
    masks_bool = masks.astype(bool)
    stats: dict[str, dict[str, float | int | None]] = {}
    for label_index, label in enumerate(LABELS):
        valid = masks_bool[:, label_index]
        positive = probs[valid & (targets[:, label_index] > 0), label_index]
        negative = probs[valid & (targets[:, label_index] <= 0), label_index]
        if positive.size == 0 or negative.size == 0:
            stats[label] = {
                "positive_count": int(positive.size),
                "negative_count": int(negative.size),
                "positive_mean": None,
                "negative_mean": None,
                "mean_gap": None,
                "positive_p10": None,
                "negative_p90": None,
                "tail_gap": None,
            }
            continue
        positive_mean = float(positive.mean())
        negative_mean = float(negative.mean())
        positive_p10 = float(np.quantile(positive, 0.10))
        negative_p90 = float(np.quantile(negative, 0.90))
        stats[label] = {
            "positive_count": int(positive.size),
            "negative_count": int(negative.size),
            "positive_mean": positive_mean,
            "negative_mean": negative_mean,
            "mean_gap": positive_mean - negative_mean,
            "positive_p10": positive_p10,
            "negative_p90": negative_p90,
            "tail_gap": positive_p10 - negative_p90,
        }
    return stats

def normalize_threshold_objective(value: Any) -> str:
    normalized = str(value).strip().lower()
    if normalized in ("f2", "fbeta", "f_beta"):
        return "fbeta"
    if normalized in ("precision_at_recall_floor", "precision@recall"):
        return "precision"
    if normalized in ("precision", "f1"):
        return normalized
    raise ValueError(
        "eval.tuned_objective must be precision, "
        "precision_at_recall_floor, f1, fbeta, or f2"
    )


def resolve_threshold_tuning_policy(
    eval_cfg: dict[str, Any],
    labels: Sequence[str],
) -> tuple[list[str], list[float | None]]:
    """Resolve legacy global and new per-class threshold tuning settings."""
    global_objective = str(eval_cfg.get("tuned_objective", "precision")).strip().lower()
    raw_objectives = eval_cfg.get("tuned_objective_by_class")
    if raw_objectives is not None and not isinstance(raw_objectives, dict):
        raise ValueError("eval.tuned_objective_by_class must be a label mapping")
    objectives = [
        normalize_threshold_objective(
            raw_objectives.get(label, global_objective)
            if raw_objectives is not None
            else global_objective
        )
        for label in labels
    ]

    raw_legacy_floors = eval_cfg.get("tuned_min_recall")
    if raw_legacy_floors is None:
        floors: list[float | None] = [None] * len(labels)
    elif isinstance(raw_legacy_floors, dict):
        # Preserve the old missing-key behavior for legacy configurations.
        floors = [
            None
            if raw_legacy_floors.get(label, 0.0) is None
            else float(raw_legacy_floors.get(label, 0.0))
            for label in labels
        ]
    else:
        floors = [float(raw_legacy_floors)] * len(labels)

    raw_class_floors = eval_cfg.get("tuned_min_recall_by_class")
    if raw_class_floors is not None:
        if not isinstance(raw_class_floors, dict):
            raise ValueError("eval.tuned_min_recall_by_class must be a label mapping")
        floors = [
            (
                None
                if raw_class_floors[label] is None
                else float(raw_class_floors[label])
            )
            if label in raw_class_floors
            else floors[index]
            for index, label in enumerate(labels)
        ]
    return objectives, floors


def apply_training_label_semantics(
    targets: Tensor,
    label_masks: Tensor,
    batch: dict[str, Any],
    train_cfg: Any,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """Apply conservative multi-label semantics to training supervision only.

    ``positive_implications`` turns a logically implied label positive, while
    ``ambiguous_negative_masks`` prevents an absent label from becoming a
    negative when another positive label makes the visual evidence ambiguous.
    The same transformation is applied to frame supervision so clip and
    spotting losses cannot provide contradictory gradients.
    """
    semantics = train_cfg.get("label_semantics", ConfigDict())
    if not isinstance(semantics, dict) or not bool(semantics.get("enabled", False)):
        return targets, label_masks, {}

    targets = targets.clone()
    label_masks = label_masks.clone()
    frame_targets = batch.get("frame_targets")
    frame_masks = batch.get("frame_target_masks")
    if torch.is_tensor(frame_targets):
        frame_targets = frame_targets.to(targets.device, non_blocking=True).clone()
    if torch.is_tensor(frame_masks):
        frame_masks = frame_masks.to(targets.device, non_blocking=True).clone()

    diagnostics: dict[str, float] = {}
    implications = semantics.get("positive_implications", ConfigDict())
    if isinstance(implications, dict):
        for source_label, destination_labels in implications.items():
            if source_label not in LABELS:
                raise ValueError(f"Unknown label_semantics source: {source_label}")
            if isinstance(destination_labels, str):
                destination_labels = [destination_labels]
            source_index = LABELS.index(source_label)
            source_positive = (
                (label_masks[:, source_index] > 0)
                & (targets[:, source_index] > 0.5)
            )
            for destination_label in destination_labels or ():
                if destination_label not in LABELS:
                    raise ValueError(
                        f"Unknown label_semantics destination: {destination_label}"
                    )
                destination_index = LABELS.index(destination_label)
                changed = source_positive & (targets[:, destination_index] <= 0.5)
                targets[source_positive, destination_index] = 1.0
                label_masks[source_positive, destination_index] = 1.0
                if frame_targets is not None and frame_masks is not None:
                    frame_targets[source_positive, :, destination_index] = torch.maximum(
                        frame_targets[source_positive, :, destination_index],
                        frame_targets[source_positive, :, source_index],
                    )
                    frame_masks[source_positive, :, destination_index] = torch.maximum(
                        frame_masks[source_positive, :, destination_index],
                        frame_masks[source_positive, :, source_index],
                    )
                diagnostics[
                    f"label_semantics_implied_{source_label}_to_{destination_label}"
                ] = float(changed.sum().detach().cpu())

    ambiguous = semantics.get("ambiguous_negative_masks", ConfigDict())
    if isinstance(ambiguous, dict):
        for source_label, destination_labels in ambiguous.items():
            if source_label not in LABELS:
                raise ValueError(f"Unknown label_semantics source: {source_label}")
            if isinstance(destination_labels, str):
                destination_labels = [destination_labels]
            source_index = LABELS.index(source_label)
            source_positive = (
                (label_masks[:, source_index] > 0)
                & (targets[:, source_index] > 0.5)
            )
            for destination_label in destination_labels or ():
                if destination_label not in LABELS:
                    raise ValueError(
                        f"Unknown label_semantics destination: {destination_label}"
                    )
                destination_index = LABELS.index(destination_label)
                ambiguous_rows = source_positive & (
                    targets[:, destination_index] <= 0.5
                )
                label_masks[ambiguous_rows, destination_index] = 0.0
                if frame_masks is not None:
                    frame_masks[ambiguous_rows, :, destination_index] = 0.0
                diagnostics[
                    f"label_semantics_masked_{source_label}_to_{destination_label}"
                ] = float(ambiguous_rows.sum().detach().cpu())

    batch["targets"] = targets
    batch["label_masks"] = label_masks
    if frame_targets is not None:
        batch["frame_targets"] = frame_targets
    if frame_masks is not None:
        batch["frame_target_masks"] = frame_masks
    return targets, label_masks, diagnostics


def tune_thresholds(
    targets: np.ndarray,
    probs: np.ndarray,
    masks: np.ndarray | None = None,
    min_recalls: np.ndarray | Sequence[float | None] | None = None,
    *,
    objective: str | None = None,
    objectives: Sequence[str] | None = None,
    fbeta_beta: float = 1.0,
) -> np.ndarray:
    if masks is None:
        masks = np.ones_like(targets, dtype=np.float32)
    masks_bool = masks.astype(bool)
    num_classes = targets.shape[1]
    if min_recalls is None:
        recall_floors: list[float | None] = [None] * num_classes
    else:
        raw_floors = np.asarray(min_recalls, dtype=object).reshape(-1).tolist()
        recall_floors = [None if value is None else float(value) for value in raw_floors]
    if len(recall_floors) != num_classes:
        raise ValueError(
            f"Expected {num_classes} recall floors, got {len(recall_floors)}"
        )
    default_objective = (
        "precision" if min_recalls is not None else "f1"
    ) if objective is None else str(objective).strip().lower()
    raw_objectives = (
        [default_objective] * num_classes
        if objectives is None
        else [str(value).strip().lower() for value in objectives]
    )
    if len(raw_objectives) != num_classes:
        raise ValueError(
            f"Expected {num_classes} threshold objectives, got {len(raw_objectives)}"
        )
    class_objectives = [normalize_threshold_objective(value) for value in raw_objectives]
    beta = max(float(fbeta_beta), 1e-6)
    beta_sq = beta * beta
    thresholds = np.full(num_classes, 0.5, dtype=np.float32)
    for i in range(num_classes):
        valid = masks_bool[:, i]
        if not valid.any():
            continue
        y = targets[valid, i]
        class_probs = probs[valid, i]
        if y.max() == y.min():
            continue
        # A prediction set changes only when the threshold crosses an observed
        # score. Distinct scores are therefore an exact search space, including
        # for collapsed probabilities far below the old 0.05 grid.
        order = np.argsort(-class_probs, kind="stable")
        sorted_probs = class_probs[order]
        sorted_targets = y[order].astype(np.int64)
        cumulative_tp = np.cumsum(sorted_targets)
        distinct_ends = np.flatnonzero(
            np.r_[sorted_probs[:-1] != sorted_probs[1:], True]
        )
        tp = cumulative_tp[distinct_ends].astype(np.float64)
        predicted = (distinct_ends + 1).astype(np.float64)
        precision = tp / np.maximum(predicted, 1.0)
        recall = tp / max(float(sorted_targets.sum()), 1.0)
        f1 = 2.0 * precision * recall / np.maximum(
            precision + recall, 1e-12
        )
        fbeta = (1.0 + beta_sq) * precision * recall / np.maximum(
            beta_sq * precision + recall, 1e-12
        )
        class_objective = class_objectives[i]
        recall_floor = recall_floors[i]
        if recall_floor is None:
            candidate_indices = np.arange(distinct_ends.size)
        else:
            eligible = np.flatnonzero(
                recall + 1e-12 >= recall_floor
            )
            # An unreachable floor must not silently select zero or an
            # arbitrary grid point. Maximize achievable recall first.
            candidate_indices = (
                eligible
                if eligible.size > 0
                else np.flatnonzero(recall == recall.max())
            )

        def selection_key(index: int) -> tuple[float, ...]:
            threshold = float(sorted_probs[distinct_ends[index]])
            if class_objective == "fbeta":
                return (
                    float(fbeta[index]), float(recall[index]),
                    float(precision[index]), threshold,
                )
            if class_objective == "f1":
                return (
                    float(f1[index]), float(recall[index]),
                    float(precision[index]), threshold,
                )
            return (
                float(precision[index]), float(f1[index]),
                float(recall[index]), threshold,
            )

        best_index = max(candidate_indices.tolist(), key=selection_key)
        thresholds[i] = float(sorted_probs[distinct_ends[best_index]])
    return thresholds


def per_video_metrics(
    targets: np.ndarray,
    probs: np.ndarray,
    masks: np.ndarray,
    metas: Sequence[dict[str, Any]],
    thresholds: np.ndarray,
) -> list[dict[str, Any]]:
    if len(metas) != len(targets):
        raise ValueError(f"metadata/target length mismatch: metas={len(metas)} targets={len(targets)}")
    grouped: OrderedDict[tuple[str, str], list[int]] = OrderedDict()
    for index, meta in enumerate(metas):
        source = str(meta.get("source", ""))
        video_id = str(meta.get("video_id", ""))
        grouped.setdefault((source, video_id), []).append(index)

    rows: list[dict[str, Any]] = []
    for (source, video_id), indices in sorted(grouped.items()):
        values = safe_metrics(targets[indices], probs[indices], thresholds, masks[indices])
        rows.append(
            {
                "source": source,
                "video_id": video_id,
                "num_samples": len(indices),
                "per_class": {
                    label: {
                        key: values["per_class"][label][key]
                        for key in (
                            "tp",
                            "fp",
                            "fn",
                            "precision",
                            "recall",
                            "f1",
                            "support",
                            "valid_count",
                            "unknown_count",
                            "threshold",
                        )
                    }
                    for label in LABELS
                },
            }
        )
    return rows


def data_parallel_active_device_count(batch_size: int, device_count: int) -> int:
    if batch_size <= 0 or device_count <= 0:
        return 0
    chunk_size = (batch_size + device_count - 1) // device_count
    return (batch_size + chunk_size - 1) // chunk_size


def forward_model_batch(
    model: nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    *,
    return_aux: bool = False,
) -> Tensor | dict[str, Tensor]:
    inputs = batch["inputs"].to(device, non_blocking=True)
    kwargs: dict[str, Any] = {"return_aux": return_aux}
    if "roi_inputs" in batch:
        kwargs["roi_inputs"] = batch["roi_inputs"].to(device, non_blocking=True)
        kwargs["roi_valid"] = batch["roi_valid"].to(device, non_blocking=True)
        kwargs["roi_meta"] = batch["roi_meta"].to(device, non_blocking=True)
        if "roi_frame_valid" in batch:
            kwargs["roi_frame_valid"] = batch["roi_frame_valid"].to(device, non_blocking=True)
        if "roi_frame_meta" in batch:
            kwargs["roi_frame_meta"] = batch["roi_frame_meta"].to(device, non_blocking=True)
        if "roi_inputs_b" in batch:
            kwargs["roi_inputs_b"] = batch["roi_inputs_b"].to(device, non_blocking=True)
            for key in ("roi_valid_b", "roi_meta_b", "roi_frame_valid_b", "roi_frame_meta_b"):
                if key in batch:
                    kwargs[key] = batch[key].to(device, non_blocking=True)
    if "frame_times" in batch:
        kwargs["global_frame_times"] = batch["frame_times"].to(device, non_blocking=True)
    if "evidence_features" in batch:
        kwargs["evidence_features"] = batch["evidence_features"].to(
            device, non_blocking=True
        )
    if "local_frame_times" in batch:
        kwargs["local_frame_times"] = batch["local_frame_times"].to(
            device, non_blocking=True
        )
    if "local_frame_times_b" in batch:
        kwargs["local_frame_times_b"] = batch["local_frame_times_b"].to(
            device, non_blocking=True
        )
    if "highres_pool_inputs" in batch:
        kwargs["highres_pool_inputs"] = batch["highres_pool_inputs"].to(
            device, non_blocking=True
        )
        kwargs["highres_pool_times"] = batch["highres_pool_times"].to(
            device, non_blocking=True
        )
        if "targets" in batch and "label_masks" in batch:
            kwargs["clip_targets"] = batch["targets"].to(
                device, non_blocking=True
            )
            kwargs["clip_label_masks"] = batch["label_masks"].to(
                device, non_blocking=True
            )
    forward_model = model
    if isinstance(model, nn.DataParallel):
        active_devices = data_parallel_active_device_count(
            int(inputs.shape[0]), len(model.device_ids)
        )
        if active_devices < len(model.device_ids):
            # Tensor.chunk may create fewer chunks than requested (for example,
            # batch=7 across 5 devices creates 4). Match kwargs replicas to the
            # actual input chunks so no replica is called without `inputs`.
            if active_devices <= 1:
                forward_model = model.module
            else:
                forward_model = nn.DataParallel(
                    model.module,
                    device_ids=model.device_ids[:active_devices],
                    output_device=model.device_ids[0],
                )
    return forward_model(inputs, **kwargs)


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def focal_bce_with_logits(
    logits: Tensor,
    targets: Tensor,
    gamma: float = 2.0,
    *,
    pos_weight: Tensor | None = None,
    alpha: Tensor | None = None,
) -> Tensor:
    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=pos_weight,
    )
    probs = torch.sigmoid(logits)
    pt = probs * targets + (1.0 - probs) * (1.0 - targets)
    loss = bce * (1.0 - pt).clamp_min(1e-6).pow(gamma)
    if alpha is not None:
        alpha = alpha.to(device=logits.device, dtype=logits.dtype)
        alpha_factor = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = loss * alpha_factor
    return loss


def training_requires_aux_outputs(
    batch: dict[str, Any],
    cfg: Any,
) -> bool:
    if "roi_inputs" in batch or "frame_targets" in batch:
        return True
    train_cfg = cfg.get("train", ConfigDict())
    return any(
        float(train_cfg.get(key, 0.0) or 0.0) > 0.0
        for key in (
            "uniform_temporal_loss_weight",
            "event_temporal_loss_weight",
            "positive_retention_loss_weight",
            "negative_teacher_guard_loss_weight",
            "temporal_gate_quality_loss_weight",
            "hard_negative_rank_loss_weight",
            "online_hard_negative_loss_weight",
            "online_hard_negative_rank_loss_weight",
            "roi_quality_loss_weight",
            "local_loss_weight",
            "spatial_clip_loss_weight",
            "spatial_attention_mil_loss_weight",
            "spatial_attention_query_diversity_loss_weight",
            "spatial_attention_concentration_loss_weight",
            "spatial_attention_overlap_loss_weight",
            "spatial_temporal_localization_loss_weight",
            "structured_frame_temporal_localization_loss_weight",
            "spatial_counterfactual_loss_weight",
            "global_conditioned_correction_loss_weight",
            "class_evidence_query_diversity_loss_weight",
            "class_evidence_counterfactual_loss_weight",
            "class_evidence_no_evidence_loss_weight",
            "class_evidence_signed_causal_loss_weight",
            "save_given_shot_loss_weight",
            "save_shot_cohort_loss_weight",
            "set_piece_subtype_loss_weight",
            "raw_context_span_loss_weight",
            "object_heatmap_loss_weight",
            "ball_goal_relation_loss_weight",
            "frame_tail_rank_loss_weight",
        )
    )


def positive_retention_loss(
    outputs: dict[str, Tensor],
    targets: Tensor,
    label_masks: Tensor,
    *,
    margin: float = 0.0,
    branch: str = "fused",
) -> Tensor:
    """Keep labeled-positive logits from falling below the frozen reference."""
    if "global_logits" not in outputs:
        return targets.new_zeros(())
    branch = str(branch).strip().lower()
    candidate_key = {"fused": "logits", "event": "event_logits"}.get(branch)
    if candidate_key is None:
        raise ValueError("positive retention branch must be fused or event")
    if candidate_key not in outputs:
        raise ValueError(
            f"positive retention branch={branch} requires outputs[{candidate_key!r}]"
        )
    reference_logits = outputs.get(
        "retention_reference_logits", outputs["global_logits"]
    ).detach()
    candidate_logits = outputs[candidate_key]
    positive_mask = (
        (targets > 0.5).to(candidate_logits.dtype)
        * label_masks.to(candidate_logits.dtype)
    )
    shortfall = F.relu(
        reference_logits + float(margin) - candidate_logits
    )
    return masked_mean(shortfall.square(), positive_mask)


def clean_positive_logit_margin_loss(
    outputs: dict[str, Tensor],
    targets: Tensor,
    label_masks: Tensor,
    *,
    margins: Sequence[float],
    label_indices: list[int] | tuple[int, ...] | None = None,
    branch: str = "fused",
) -> tuple[Tensor, dict[str, float]]:
    """Push trusted positive clip logits above an absolute per-class margin."""
    branch = str(branch).strip().lower()
    candidate_key = {"fused": "logits", "event": "event_logits"}.get(branch)
    if candidate_key is None:
        raise ValueError("clean positive logit margin branch must be fused or event")
    if candidate_key not in outputs:
        raise ValueError(
            f"clean positive logit margin branch={branch} requires outputs[{candidate_key!r}]"
        )
    logits = outputs[candidate_key]
    if label_indices is None:
        label_indices = tuple(range(logits.shape[1]))
    margins_tensor = logits.new_tensor(list(margins)).reshape(1, -1)
    if margins_tensor.shape[1] != logits.shape[1]:
        raise ValueError(
            f"clean positive margins length {margins_tensor.shape[1]} != logits classes {logits.shape[1]}"
        )
    selected = torch.zeros_like(label_masks, dtype=torch.bool)
    for label_index in label_indices:
        idx = int(label_index)
        if not 0 <= idx < logits.shape[1]:
            raise ValueError(f"invalid clean positive margin label index: {idx}")
        selected[:, idx] = (label_masks[:, idx] > 0) & (targets[:, idx] > 0.5)
    selected_f = selected.to(logits.dtype)
    shortfall = F.softplus(margins_tensor - logits)
    loss = masked_mean(shortfall, selected_f)
    selected_count = int(selected.detach().sum().cpu())
    if selected_count > 0:
        selected_logits = logits.detach()[selected]
        selected_margins = margins_tensor.expand_as(logits).detach()[selected]
        mean_logit = float(selected_logits.float().mean().cpu())
        violation = float((selected_logits.float() < selected_margins.float()).float().mean().cpu())
    else:
        mean_logit = 0.0
        violation = 0.0
    return loss, {
        "clean_positive_margin_loss": float(loss.detach().cpu()),
        "clean_positive_margin_positive_logit": mean_logit,
        "clean_positive_margin_violation_fraction": violation,
        "clean_positive_margin_slots": float(selected_count),
    }





def positive_low_tail_bce_mining_loss(
    loss_matrix: Tensor,
    logits: Tensor,
    targets: Tensor,
    label_masks: Tensor,
    *,
    label_indices: list[int] | tuple[int, ...] | None = None,
    quantile: float = 0.25,
    min_count: int = 1,
) -> tuple[Tensor, dict[str, float]]:
    """Extra BCE pressure on the lowest-scoring trusted positive clips.

    This is the positive analogue of hard-negative mining: for each class in the
    current batch, select the bottom-q positive logits and re-apply their BCE
    loss. It directly targets recall-tail positives instead of increasing all
    positive gradients uniformly.
    """
    if loss_matrix.shape != logits.shape or logits.shape != targets.shape:
        raise ValueError("positive low-tail BCE mining expects loss/logits/targets with matching [B,C] shapes")
    if label_indices is None:
        label_indices = tuple(range(logits.shape[1]))
    q = min(max(float(quantile), 0.0), 1.0)
    min_keep = max(int(min_count), 1)
    losses: list[Tensor] = []
    selected_logits: list[Tensor] = []
    selected_losses: list[Tensor] = []
    active_classes = 0
    total_slots = 0
    for label_index in label_indices:
        idx = int(label_index)
        if not 0 <= idx < logits.shape[1]:
            raise ValueError(f"invalid positive low-tail BCE label index: {idx}")
        positive_mask = (label_masks[:, idx] > 0) & (targets[:, idx] > 0.5)
        values = logits[positive_mask, idx]
        if values.numel() == 0:
            continue
        keep = min(values.numel(), max(min_keep, int(math.ceil(values.numel() * q))))
        tail = torch.topk(values.detach(), k=keep, largest=False, sorted=False)
        class_losses = loss_matrix[positive_mask, idx][tail.indices]
        losses.append(class_losses.mean())
        selected_logits.append(values.detach()[tail.indices])
        selected_losses.append(class_losses.detach())
        total_slots += int(keep)
        active_classes += 1
    if not losses:
        zero = logits.sum() * 0.0
        return zero, {
            "positive_low_tail_bce_loss": 0.0,
            "positive_low_tail_bce_logit": 0.0,
            "positive_low_tail_bce_raw_loss": 0.0,
            "positive_low_tail_bce_slots": 0.0,
            "positive_low_tail_bce_active_classes": 0.0,
        }
    loss = torch.stack(losses).mean()
    return loss, {
        "positive_low_tail_bce_loss": float(loss.detach().cpu()),
        "positive_low_tail_bce_logit": float(torch.cat(selected_logits).float().mean().cpu()),
        "positive_low_tail_bce_raw_loss": float(torch.cat(selected_losses).float().mean().cpu()),
        "positive_low_tail_bce_slots": float(total_slots),
        "positive_low_tail_bce_active_classes": float(active_classes),
    }


def positive_low_tail_logit_margin_loss(
    outputs: dict[str, Tensor],
    targets: Tensor,
    label_masks: Tensor,
    *,
    margins: Sequence[float],
    label_indices: list[int] | tuple[int, ...] | None = None,
    branch: str = "fused",
    quantile: float = 0.25,
    min_count: int = 1,
) -> tuple[Tensor, dict[str, float]]:
    """Protect the lowest-scoring trusted positives from calibration collapse.

    Unlike the average clean-positive margin loss, this term only uses the
    bottom q-fraction of positive logits in the current batch. It targets the
    failure mode where recall-floor threshold tuning is forced near zero by a
    small positive low tail even though average positive logits look healthy.
    """
    branch = str(branch).strip().lower()
    candidate_key = {"fused": "logits", "event": "event_logits"}.get(branch)
    if candidate_key is None:
        raise ValueError("positive low-tail branch must be fused or event")
    if candidate_key not in outputs:
        raise ValueError(
            f"positive low-tail branch={branch} requires outputs[{candidate_key!r}]"
        )
    logits = outputs[candidate_key]
    if label_indices is None:
        label_indices = tuple(range(logits.shape[1]))
    margins_tensor = logits.new_tensor(list(margins)).reshape(-1)
    if margins_tensor.numel() != logits.shape[1]:
        raise ValueError(
            f"positive low-tail margins length {margins_tensor.numel()} != logits classes {logits.shape[1]}"
        )
    q = min(max(float(quantile), 0.0), 1.0)
    min_keep = max(int(min_count), 1)
    losses: list[Tensor] = []
    selected_values: list[Tensor] = []
    selected_margins: list[Tensor] = []
    selected_total = 0
    active_classes = 0
    for label_index in label_indices:
        idx = int(label_index)
        if not 0 <= idx < logits.shape[1]:
            raise ValueError(f"invalid positive low-tail label index: {idx}")
        positive_mask = (label_masks[:, idx] > 0) & (targets[:, idx] > 0.5)
        values = logits[positive_mask, idx]
        if values.numel() == 0:
            continue
        keep = min(values.numel(), max(min_keep, int(math.ceil(values.numel() * q))))
        tail_values = torch.topk(values, k=keep, largest=False, sorted=False).values
        margin = margins_tensor[idx]
        losses.append(F.softplus(margin - tail_values).mean())
        selected_values.append(tail_values.detach())
        selected_margins.append(margin.detach().expand_as(tail_values))
        selected_total += int(tail_values.numel())
        active_classes += 1
    if not losses:
        zero = logits.sum() * 0.0
        return zero, {
            "positive_low_tail_margin_loss": 0.0,
            "positive_low_tail_logit": 0.0,
            "positive_low_tail_violation_fraction": 0.0,
            "positive_low_tail_slots": 0.0,
            "positive_low_tail_active_classes": 0.0,
        }
    selected_logits = torch.cat(selected_values).float()
    selected_margin_values = torch.cat(selected_margins).float()
    loss = torch.stack(losses).mean()
    return loss, {
        "positive_low_tail_margin_loss": float(loss.detach().cpu()),
        "positive_low_tail_logit": float(selected_logits.mean().cpu()),
        "positive_low_tail_violation_fraction": float(
            (selected_logits < selected_margin_values).float().mean().cpu()
        ),
        "positive_low_tail_slots": float(selected_total),
        "positive_low_tail_active_classes": float(active_classes),
    }


def hard_positive_tail_margin_loss(
    logits: Tensor,
    targets: Tensor,
    label_masks: Tensor,
    *,
    floors: Sequence[float],
    margins: Sequence[float],
    label_indices: list[int] | tuple[int, ...] | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Pull up trusted positives that sit below an EMA floor (hard positives).

    Unlike positive_low_tail_logit_margin_loss, which selects the bottom-q
    fraction per micro-batch (statistically dead at batch=2 where a step often
    has zero or one positive), this term fires on every below-floor positive in
    every step. The floors come from an EMA of the per-class positive quantile
    maintained in the training loop, so the target adapts as the distribution
    moves; the loss itself stays a pure function of tensors.
    """
    if logits.ndim != 2 or targets.ndim != 2 or label_masks.ndim != 2:
        raise ValueError("hard positive tail margin expects [B,C] tensors")
    if label_indices is None:
        label_indices = tuple(range(logits.shape[1]))
    floors_tensor = logits.new_tensor(list(floors)).reshape(-1)
    margins_tensor = logits.new_tensor(list(margins)).reshape(-1)
    if floors_tensor.numel() != logits.shape[1]:
        raise ValueError(
            f"hard positive tail floors length {floors_tensor.numel()} != logits classes {logits.shape[1]}"
        )
    if margins_tensor.numel() != logits.shape[1]:
        raise ValueError(
            f"hard positive tail margins length {margins_tensor.numel()} != logits classes {logits.shape[1]}"
        )
    losses: list[Tensor] = []
    selected_values: list[Tensor] = []
    selected_floors: list[Tensor] = []
    selected_targets: list[Tensor] = []
    selected_total = 0
    active_classes = 0
    for label_index in label_indices:
        idx = int(label_index)
        if not 0 <= idx < logits.shape[1]:
            raise ValueError(f"invalid hard positive tail label index: {idx}")
        floor = floors_tensor[idx]
        selected = (
            (label_masks[:, idx] > 0)
            & (targets[:, idx] > 0.5)
            & (logits[:, idx] < floor)
        )
        if not bool(selected.any().item()):
            continue
        selected_f = selected.to(logits.dtype)
        margin = margins_tensor[idx]
        shortfall = F.softplus((floor + margin) - logits[:, idx])
        losses.append(masked_mean(shortfall, selected_f))
        selected_values.append(logits[selected, idx].detach())
        selected_floors.append((floor + margin).detach().expand_as(logits[selected, idx]))
        selected_total += int(selected_f.sum().item())
        active_classes += 1
    if not losses:
        zero = logits.sum() * 0.0
        return zero, {
            "hard_positive_tail_margin_loss": 0.0,
            "hard_positive_tail_logit": 0.0,
            "hard_positive_tail_violation_fraction": 0.0,
            "hard_positive_tail_slots": 0.0,
            "hard_positive_tail_active_classes": 0.0,
        }
    selected_logits = torch.cat(selected_values).float()
    selected_target_values = torch.cat(selected_floors).float()
    loss = torch.stack(losses).mean()
    return loss, {
        "hard_positive_tail_margin_loss": float(loss.detach().cpu()),
        "hard_positive_tail_logit": float(selected_logits.mean().cpu()),
        "hard_positive_tail_violation_fraction": float(
            (selected_logits < selected_target_values).float().mean().cpu()
        ),
        "hard_positive_tail_slots": float(selected_total),
        "hard_positive_tail_active_classes": float(active_classes),
    }


def tail_separation_paired_loss(
    logits: Tensor,
    targets: Tensor,
    label_masks: Tensor,
    metas: Sequence[Mapping[str, Any]],
    *,
    labels: Sequence[str],
    margin: float = 0.15,
    positive_quantile: float = 0.25,
    negative_quantile: float = 0.25,
    min_positive: int = 2,
    min_negative: int = 2,
    label_indices: list[int] | tuple[int, ...] | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Online tail separation: bottom positives must beat top negatives.

    E1.5: instead of sampling fixed random pair rankings, every class
    dynamically pairs the weakest trusted positives in the current batch /
    same video against the strongest trusted negatives and optimizes
    softplus(margin + top_negative - bottom_positive).  Only trusted label
    masks (label_mask > 0) are used; no event-time anchor is consulted, so
    the supervision survives annotation noise at the manual event time.

    Positives/negatives are grouped by video (meta video_path); a video
    group qualifies with at least min_positive positives and min_negative
    negatives.  When no video group qualifies, the class falls back to the
    whole batch (all rows) so the term stays alive on small batches.
    """
    if logits.ndim != 2 or targets.ndim != 2 or label_masks.ndim != 2:
        raise ValueError("tail separation expects [B,C] tensors")
    if targets.shape != logits.shape or label_masks.shape != logits.shape:
        raise ValueError("targets/label_masks must match logits [B,C]")
    if label_indices is None:
        label_indices = tuple(range(logits.shape[1]))
    video_ids = [
        str(meta.get("video_path", "")) if isinstance(meta, Mapping) else ""
        for meta in metas
    ]
    q_pos = min(max(float(positive_quantile), 0.0), 1.0)
    q_neg = min(max(float(negative_quantile), 0.0), 1.0)
    min_pos = max(int(min_positive), 1)
    min_neg = max(int(min_negative), 1)
    device = logits.device
    losses: list[Tensor] = []
    bottom_pos_logits: list[Tensor] = []
    top_neg_logits: list[Tensor] = []
    gaps: list[Tensor] = []
    violations: list[Tensor] = []
    per_class_gaps: list[Tensor | None] = [None] * logits.shape[1]
    active_classes = 0
    active_groups = 0
    positive_slots = 0
    negative_slots = 0
    for class_index in label_indices:
        idx = int(class_index)
        if not 0 <= idx < logits.shape[1]:
            raise ValueError(f"invalid tail separation label index: {idx}")
        trusted = label_masks[:, idx] > 0
        positive_mask = trusted & (targets[:, idx] > 0.5)
        negative_mask = trusted & (targets[:, idx] <= 0.5)
        # Group rows by video; fall back to the whole batch when no video
        # group has enough trusted positives AND negatives.
        groups: list[tuple[Tensor, Tensor]] = []
        if any(video_ids) and positive_mask.any() and negative_mask.any():
            for video in dict.fromkeys(video_ids):
                if not video:
                    continue
                rows = torch.tensor(
                    [video_id == video for video_id in video_ids],
                    dtype=torch.bool,
                    device=device,
                )
                pos_rows = rows & positive_mask
                neg_rows = rows & negative_mask
                if (
                    int(pos_rows.sum().item()) >= min_pos
                    and int(neg_rows.sum().item()) >= min_neg
                ):
                    groups.append((pos_rows, neg_rows))
        if not groups:
            if not positive_mask.any() or not negative_mask.any():
                continue
            groups.append((positive_mask, negative_mask))
        class_gaps: list[Tensor] = []
        for pos_rows, neg_rows in groups:
            pos_vals = logits[pos_rows, idx]
            neg_vals = logits[neg_rows, idx]
            keep_pos = max(1, int(math.ceil(int(pos_vals.numel()) * q_pos)))
            keep_neg = max(1, int(math.ceil(int(neg_vals.numel()) * q_neg)))
            bottom_pos = torch.topk(
                pos_vals, k=min(keep_pos, int(pos_vals.numel())), largest=False
            ).values
            top_neg = torch.topk(
                neg_vals, k=min(keep_neg, int(neg_vals.numel())), largest=True
            ).values
            gap_matrix = (
                bottom_pos.reshape(-1, 1) - top_neg.reshape(1, -1)
            )  # [k_pos, k_neg]
            class_gaps.append(gap_matrix)
            losses.append(
                F.softplus(gap_matrix.new_tensor(float(margin)) - gap_matrix).mean()
            )
            bottom_pos_logits.append(bottom_pos.detach().mean())
            top_neg_logits.append(top_neg.detach().mean())
            gaps.append(gap_matrix.detach().mean())
            violations.append((gap_matrix.detach() < float(margin)).float().mean())
            positive_slots += int(bottom_pos.numel())
            negative_slots += int(top_neg.numel())
        if class_gaps:
            combined = torch.cat([gap.reshape(-1) for gap in class_gaps])
            per_class_gaps[idx] = combined.mean()
            active_classes += 1
            active_groups += len(class_gaps)
    if not losses:
        zero = logits.sum() * 0.0
        return zero, {
            "tail_separation_loss": 0.0,
            "tail_separation_bottom_positive_logit": 0.0,
            "tail_separation_top_negative_logit": 0.0,
            "tail_separation_gap": 0.0,
            "tail_separation_violation_fraction": 0.0,
            "tail_separation_active_classes": 0.0,
            "tail_separation_active_groups": 0.0,
            "tail_separation_positive_slots": 0.0,
            "tail_separation_negative_slots": 0.0,
        }
    loss = torch.stack(losses).mean()
    diagnostics = {
        "tail_separation_loss": float(loss.detach().cpu()),
        "tail_separation_bottom_positive_logit": float(
            torch.stack(bottom_pos_logits).mean().cpu()
        ),
        "tail_separation_top_negative_logit": float(
            torch.stack(top_neg_logits).mean().cpu()
        ),
        "tail_separation_gap": float(torch.stack(gaps).mean().cpu()),
        "tail_separation_violation_fraction": float(
            torch.stack(violations).mean().cpu()
        ),
        "tail_separation_active_classes": float(active_classes),
        "tail_separation_active_groups": float(active_groups),
        "tail_separation_positive_slots": float(positive_slots),
        "tail_separation_negative_slots": float(negative_slots),
    }
    for class_index, label in enumerate(labels):
        if class_index >= logits.shape[1]:
            continue
        class_gap = per_class_gaps[class_index]
        diagnostics[f"tail_separation_gap_{label}"] = float(
            class_gap.cpu() if class_gap is not None else 0.0
        )
    return loss, diagnostics


def online_global_tail_ranking_loss(
    logits: Tensor,
    targets: Tensor,
    label_masks: Tensor,
    metas: Sequence[Mapping[str, Any]],
    *,
    labels: Sequence[str],
    margin: float = 0.5,
    positive_topk: int = 4,
    negative_topk: int = 4,
    label_indices: list[int] | tuple[int, ...] | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Rank reliable central positives above the global near-event tail.

    Every E1.3 local batch contains only one linked near-event negative, so a
    local top-k loss mostly sees an easy pair. This loss gathers detached
    scores and supervision masks across DDP ranks, selects the global bottom-k
    central positives and top-k true near-event negatives, and applies
    symmetric ranking pressure to the local rows that own those two tails.

    Edge positives are excluded because their event support is clipped and
    their label time is uncertain. Random/fallback negatives are excluded so
    trivially separable backgrounds cannot make this objective look healthy.
    """
    if logits.ndim != 2 or targets.shape != logits.shape or label_masks.shape != logits.shape:
        raise ValueError("online global tail ranking expects matching [B,C] tensors")
    if len(metas) != logits.shape[0]:
        raise ValueError("online global tail ranking requires one meta row per logit row")
    if label_indices is None:
        label_indices = tuple(range(logits.shape[1]))

    device = logits.device
    central = torch.tensor(
        [
            bool(isinstance(meta, Mapping) and meta.get("online_pair_role") == "central")
            for meta in metas
        ], dtype=torch.bool, device=device,
    )
    near = torch.tensor(
        [
            bool(isinstance(meta, Mapping) and meta.get("online_negative_kind") == "near_event_context")
            for meta in metas
        ], dtype=torch.bool, device=device,
    )

    gathered_logits = [logits.detach()]
    gathered_targets = [targets.detach()]
    gathered_masks = [label_masks.detach()]
    gathered_central = [central]
    gathered_near = [near]
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        world = dist.get_world_size()

        def gather_detached(value: Tensor) -> list[Tensor]:
            result = [torch.empty_like(value) for _ in range(world)]
            dist.all_gather(result, value.detach())
            return result

        gathered_logits = gather_detached(logits)
        gathered_targets = gather_detached(targets)
        gathered_masks = gather_detached(label_masks)
        gathered_central = gather_detached(central)
        gathered_near = gather_detached(near)

    global_logits = torch.cat(gathered_logits, dim=0)
    global_targets = torch.cat(gathered_targets, dim=0)
    global_masks = torch.cat(gathered_masks, dim=0)
    global_central = torch.cat(gathered_central, dim=0).bool()
    global_near = torch.cat(gathered_near, dim=0).bool()

    losses: list[Tensor] = []
    gaps: list[Tensor] = []
    violations: list[Tensor] = []
    bottom_values: list[Tensor] = []
    top_values: list[Tensor] = []
    active_classes = 0
    local_positive_slots = 0
    local_negative_slots = 0
    per_class_gap: dict[str, float] = {}
    for class_index in label_indices:
        idx = int(class_index)
        if not 0 <= idx < logits.shape[1]:
            raise ValueError(f"invalid online global tail label index: {idx}")
        global_trusted = global_masks[:, idx] > 0
        global_pos_mask = global_central & global_trusted & (global_targets[:, idx] > 0.5)
        global_neg_mask = global_near & global_trusted & (global_targets[:, idx] <= 0.5)
        if not global_pos_mask.any() or not global_neg_mask.any():
            continue
        global_pos = global_logits[global_pos_mask, idx]
        global_neg = global_logits[global_neg_mask, idx]
        keep_pos = min(max(int(positive_topk), 1), int(global_pos.numel()))
        keep_neg = min(max(int(negative_topk), 1), int(global_neg.numel()))
        bottom_pos = torch.topk(global_pos, k=keep_pos, largest=False).values
        top_neg = torch.topk(global_neg, k=keep_neg, largest=True).values
        bottom_boundary = bottom_pos.max().detach()
        top_boundary = top_neg.min().detach()

        local_trusted = label_masks[:, idx] > 0
        local_pos = central & local_trusted & (targets[:, idx] > 0.5)
        local_neg = near & local_trusted & (targets[:, idx] <= 0.5)
        local_pos = local_pos & (logits[:, idx].detach() <= bottom_boundary)
        local_neg = local_neg & (logits[:, idx].detach() >= top_boundary)
        class_losses: list[Tensor] = []
        if local_pos.any():
            class_losses.append(
                F.softplus(float(margin) + top_neg.mean().detach() - logits[local_pos, idx]).mean()
            )
            local_positive_slots += int(local_pos.sum().item())
        if local_neg.any():
            class_losses.append(
                F.softplus(float(margin) + logits[local_neg, idx] - bottom_pos.mean().detach()).mean()
            )
            local_negative_slots += int(local_neg.sum().item())
        losses.append(
            torch.stack(class_losses).mean()
            if class_losses else logits[:, idx].sum() * 0.0
        )
        gap_matrix = bottom_pos.reshape(-1, 1) - top_neg.reshape(1, -1)
        gaps.append(gap_matrix.mean())
        violations.append((gap_matrix < float(margin)).float().mean())
        bottom_values.append(bottom_pos.mean())
        top_values.append(top_neg.mean())
        per_class_gap[labels[idx]] = float(gap_matrix.mean().cpu())
        active_classes += 1

    if not losses:
        zero = logits.sum() * 0.0
        return zero, {
            "online_global_tail_loss": 0.0,
            "online_global_tail_gap": 0.0,
            "online_global_tail_violation_fraction": 0.0,
            "online_global_tail_active_classes": 0.0,
            "online_global_tail_positive_slots": 0.0,
            "online_global_tail_negative_slots": 0.0,
        }
    loss = torch.stack(losses).mean()
    diagnostics = {
        "online_global_tail_loss": float(loss.detach().cpu()),
        "online_global_tail_bottom_positive_logit": float(torch.stack(bottom_values).mean().cpu()),
        "online_global_tail_top_negative_logit": float(torch.stack(top_values).mean().cpu()),
        "online_global_tail_gap": float(torch.stack(gaps).mean().cpu()),
        "online_global_tail_violation_fraction": float(torch.stack(violations).mean().cpu()),
        "online_global_tail_active_classes": float(active_classes),
        "online_global_tail_positive_slots": float(local_positive_slots),
        "online_global_tail_negative_slots": float(local_negative_slots),
    }
    for label in labels:
        diagnostics[f"online_global_tail_gap_{label}"] = per_class_gap.get(label, 0.0)
    return loss, diagnostics


def ema_positive_retention_loss(
    logits: Tensor,
    teacher_logits: Tensor,
    targets: Tensor,
    label_masks: Tensor,
    *,
    labels: Sequence[str],
    margin: float = 0.05,
    temperature: float = 0.5,
    label_indices: list[int] | tuple[int, ...] | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """EMA teacher positive retention: the student must not degrade positives.

    The EMA of the trainable parameters is the reference the new model must
    not regress on: for every trusted positive, the student logit must stay
    at least margin above the (detached) teacher logit.  A temperature-scaled
    softplus hinge keeps the constraint smooth, so positives that already
    satisfy it receive a gentle (not zero) gradient.
    """
    if logits.ndim != 2 or teacher_logits.ndim != 2:
        raise ValueError("EMA retention expects [B,C] logits")
    if logits.shape != teacher_logits.shape:
        raise ValueError("student/teacher logits must have identical [B,C] shape")
    if targets.shape != logits.shape or label_masks.shape != logits.shape:
        raise ValueError("targets/label_masks must match logits [B,C]")
    if label_indices is None:
        label_indices = tuple(range(logits.shape[1]))
    tau = max(float(temperature), 1e-6)
    gap = logits - teacher_logits.detach()
    losses: list[Tensor] = []
    positive_gaps: list[Tensor] = []
    violations: list[Tensor] = []
    per_class_gaps: list[Tensor | None] = [None] * logits.shape[1]
    positive_slots = 0
    active_classes = 0
    for class_index in label_indices:
        idx = int(class_index)
        if not 0 <= idx < logits.shape[1]:
            raise ValueError(f"invalid EMA retention label index: {idx}")
        selected = (label_masks[:, idx] > 0) & (targets[:, idx] > 0.5)
        if not bool(selected.any().item()):
            continue
        class_gap = gap[selected, idx]
        losses.append(
            tau
            * F.softplus(
                (class_gap.new_tensor(float(margin)) - class_gap) / tau
            ).mean()
        )
        positive_gaps.append(class_gap.detach())
        violations.append(
            (class_gap.detach() < float(margin)).float().mean()
        )
        per_class_gaps[idx] = class_gap.detach().mean()
        positive_slots += int(class_gap.numel())
        active_classes += 1
    if not losses:
        zero = logits.sum() * 0.0
        return zero, {
            "ema_positive_retention_loss": 0.0,
            "ema_positive_retention_gap": 0.0,
            "ema_positive_retention_violation_fraction": 0.0,
            "ema_positive_retention_positive_slots": 0.0,
            "ema_positive_retention_active_classes": 0.0,
        }
    loss = torch.stack(losses).mean()
    diagnostics = {
        "ema_positive_retention_loss": float(loss.detach().cpu()),
        "ema_positive_retention_gap": float(
            torch.cat(positive_gaps).mean().cpu()
        ),
        "ema_positive_retention_violation_fraction": float(
            torch.stack(violations).mean().cpu()
        ),
        "ema_positive_retention_positive_slots": float(positive_slots),
        "ema_positive_retention_active_classes": float(active_classes),
    }
    for class_index, label in enumerate(labels):
        if class_index >= logits.shape[1]:
            continue
        class_gap = per_class_gaps[class_index]
        diagnostics[f"ema_positive_retention_gap_{label}"] = float(
            class_gap.cpu() if class_gap is not None else 0.0
        )
    return loss, diagnostics


def fixed_teacher_logit_guard_loss(
    logits: Tensor,
    teacher_logits: Tensor,
    targets: Tensor,
    label_masks: Tensor,
    *,
    labels: Sequence[str],
    positive_margin: float = 0.0,
    negative_margin: float = 0.0,
    label_indices: list[int] | tuple[int, ...] | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Preserve a fixed initialization's ranking while allowing improvements.

    Positives may move above the teacher but not below it. Negatives may move
    below the teacher but not above it. This one-sided guard permits margin
    improvement while blocking recall erosion and dense false-positive
    broadening during online adaptation.
    """
    if logits.ndim != 2 or teacher_logits.shape != logits.shape:
        raise ValueError("fixed teacher guard expects matching [B,C] logits")
    if targets.shape != logits.shape or label_masks.shape != logits.shape:
        raise ValueError("targets/label_masks must match fixed teacher logits")
    if label_indices is None:
        label_indices = tuple(range(logits.shape[1]))

    losses: list[Tensor] = []
    positive_violations: list[Tensor] = []
    negative_violations: list[Tensor] = []
    active_classes = 0
    positive_slots = 0
    negative_slots = 0
    per_class: dict[str, dict[str, float]] = {}
    teacher = teacher_logits.detach()
    for class_index in label_indices:
        idx = int(class_index)
        if not 0 <= idx < logits.shape[1]:
            raise ValueError(f"invalid fixed teacher guard label index: {idx}")
        trusted = label_masks[:, idx] > 0
        positive = trusted & (targets[:, idx] > 0.5)
        negative = trusted & (targets[:, idx] <= 0.5)
        class_terms: list[Tensor] = []
        class_pos_violation = logits.new_zeros(())
        class_neg_violation = logits.new_zeros(())
        if positive.any():
            shortfall = F.relu(
                teacher[positive, idx]
                + float(positive_margin)
                - logits[positive, idx]
            )
            class_terms.append(shortfall.square().mean())
            class_pos_violation = (shortfall.detach() > 0).float().mean()
            positive_violations.append(class_pos_violation)
            positive_slots += int(positive.sum().item())
        if negative.any():
            excess = F.relu(
                logits[negative, idx]
                - teacher[negative, idx]
                - float(negative_margin)
            )
            class_terms.append(excess.square().mean())
            class_neg_violation = (excess.detach() > 0).float().mean()
            negative_violations.append(class_neg_violation)
            negative_slots += int(negative.sum().item())
        if not class_terms:
            continue
        losses.append(torch.stack(class_terms).mean())
        active_classes += 1
        label = labels[idx] if idx < len(labels) else str(idx)
        per_class[label] = {
            "positive_violation": float(class_pos_violation.cpu()),
            "negative_violation": float(class_neg_violation.cpu()),
        }

    if not losses:
        zero = logits.sum() * 0.0
        return zero, {
            "fixed_teacher_logit_guard_loss": 0.0,
            "fixed_teacher_positive_violation_fraction": 0.0,
            "fixed_teacher_negative_violation_fraction": 0.0,
            "fixed_teacher_positive_slots": 0.0,
            "fixed_teacher_negative_slots": 0.0,
            "fixed_teacher_active_classes": 0.0,
        }
    loss = torch.stack(losses).mean()
    diagnostics = {
        "fixed_teacher_logit_guard_loss": float(loss.detach().cpu()),
        "fixed_teacher_positive_violation_fraction": float(
            torch.stack(positive_violations).mean().cpu()
            if positive_violations else 0.0
        ),
        "fixed_teacher_negative_violation_fraction": float(
            torch.stack(negative_violations).mean().cpu()
            if negative_violations else 0.0
        ),
        "fixed_teacher_positive_slots": float(positive_slots),
        "fixed_teacher_negative_slots": float(negative_slots),
        "fixed_teacher_active_classes": float(active_classes),
    }
    for label, values in per_class.items():
        diagnostics[f"fixed_teacher_positive_violation_{label}"] = values[
            "positive_violation"
        ]
        diagnostics[f"fixed_teacher_negative_violation_{label}"] = values[
            "negative_violation"
        ]
    return loss, diagnostics


def global_conditioned_correction_loss(
    outputs: dict[str, Tensor],
    targets: Tensor,
    label_masks: Tensor,
    *,
    threshold_probs: Sequence[float],
    margin_logit: float = 0.25,
    fp_weight: float = 1.0,
    fn_weight: float = 1.25,
    tp_guard_weight: float = 0.5,
    tn_guard_weight: float = 0.25,
    stability_weight: float = 0.02,
    target_max_delta: float = 2.0,
    smooth_l1_beta: float = 0.5,
) -> tuple[Tensor, dict[str, float]]:
    """Teach a bounded residual to fix frozen-global FP and FN decisions."""
    required = {"logits", "global_logits", "spatial_fusion_delta"}
    missing = sorted(required.difference(outputs))
    if missing:
        raise ValueError(
            f"conditioned correction loss requires output keys: {missing}"
        )
    fused_logits = outputs["logits"]
    global_logits = outputs["global_logits"].detach()
    correction = outputs["spatial_fusion_delta"]
    thresholds = torch.as_tensor(
        list(threshold_probs), device=fused_logits.device, dtype=fused_logits.dtype
    )
    if thresholds.numel() != fused_logits.shape[1]:
        raise ValueError(
            "global correction thresholds must have one value per label"
        )
    thresholds = thresholds.clamp(1e-5, 1.0 - 1e-5)
    threshold_logits = torch.logit(thresholds).reshape(1, -1)
    valid = label_masks.to(fused_logits.dtype)
    positive = targets > 0.5
    global_positive = global_logits >= threshold_logits
    fp_mask = valid * ((~positive) & global_positive).to(valid.dtype)
    fn_mask = valid * (positive & (~global_positive)).to(valid.dtype)
    tp_mask = valid * (positive & global_positive).to(valid.dtype)
    tn_mask = valid * ((~positive) & (~global_positive)).to(valid.dtype)

    margin = float(margin_logit)
    max_delta = max(float(target_max_delta), 1e-6)
    target_correction = torch.zeros_like(correction)
    fp_target = ((threshold_logits - margin) - global_logits).clamp(
        min=-max_delta, max=0.0
    )
    fn_target = ((threshold_logits + margin) - global_logits).clamp(
        min=0.0, max=max_delta
    )
    target_correction = torch.where(fp_mask > 0, fp_target, target_correction)
    target_correction = torch.where(fn_mask > 0, fn_target, target_correction)
    correction_error = F.smooth_l1_loss(
        correction,
        target_correction,
        reduction="none",
        beta=max(float(smooth_l1_beta), 1e-6),
    )
    fp_loss = masked_mean(correction_error, fp_mask)
    fn_loss = masked_mean(correction_error, fn_mask)
    tp_guard = masked_mean(F.relu(global_logits - fused_logits).square(), tp_mask)
    tn_guard = masked_mean(F.relu(fused_logits - global_logits).square(), tn_mask)
    correct_mask = tp_mask + tn_mask
    stability = masked_mean(correction.square(), correct_mask)
    total = (
        float(fp_weight) * fp_loss
        + float(fn_weight) * fn_loss
        + float(tp_guard_weight) * tp_guard
        + float(tn_guard_weight) * tn_guard
        + float(stability_weight) * stability
    )

    components = {
        "global_correction_loss": float(total.detach().cpu()),
        "global_correction_fp_loss": float(fp_loss.detach().cpu()),
        "global_correction_fn_loss": float(fn_loss.detach().cpu()),
        "global_correction_tp_guard_loss": float(tp_guard.detach().cpu()),
        "global_correction_tn_guard_loss": float(tn_guard.detach().cpu()),
        "global_correction_stability_loss": float(stability.detach().cpu()),
        "global_correction_fp_slots": float(fp_mask.sum().detach().cpu()),
        "global_correction_fn_slots": float(fn_mask.sum().detach().cpu()),
        "global_correction_tp_slots": float(tp_mask.sum().detach().cpu()),
        "global_correction_tn_slots": float(tn_mask.sum().detach().cpu()),
        "global_correction_fp_delta": float(
            masked_mean(correction, fp_mask).detach().cpu()
        ),
        "global_correction_fn_delta": float(
            masked_mean(correction, fn_mask).detach().cpu()
        ),
    }
    return total, components


def per_class_equal_pos_neg_loss(
    loss_matrix: Tensor,
    targets: Tensor,
    masks: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    positive_mask = masks * (targets > 0.5).to(masks.dtype)
    negative_mask = masks * (targets <= 0.5).to(masks.dtype)
    positive_count = positive_mask.sum(dim=0)
    negative_count = negative_mask.sum(dim=0)
    positive_loss = (loss_matrix * positive_mask).sum(dim=0) / positive_count.clamp_min(1.0)
    negative_loss = (loss_matrix * negative_mask).sum(dim=0) / negative_count.clamp_min(1.0)
    positive_present = (positive_count > 0).to(loss_matrix.dtype)
    negative_present = (negative_count > 0).to(loss_matrix.dtype)
    term_count = positive_present + negative_present
    class_loss = (
        positive_loss * positive_present + negative_loss * negative_present
    ) / term_count.clamp_min(1.0)
    valid_classes = (term_count > 0).to(loss_matrix.dtype)
    total = (class_loss * valid_classes).sum() / valid_classes.sum().clamp_min(1.0)
    return (
        total,
        masked_mean(loss_matrix, positive_mask),
        masked_mean(loss_matrix, negative_mask),
    )


def save_given_shot_classification_loss(
    logits: Tensor,
    targets: Tensor,
    masks: Tensor,
) -> tuple[Tensor, dict[str, float]]:
    """Discriminate saves from shot-only clips instead of using shot as a cue."""
    if "shot" not in LABEL_TO_INDEX or "save" not in LABEL_TO_INDEX:
        return logits.new_zeros(()), {}
    shot_index = LABEL_TO_INDEX["shot"]
    save_index = LABEL_TO_INDEX["save"]
    condition = (
        masks[:, shot_index]
        * masks[:, save_index]
        * (targets[:, shot_index] > 0.5).to(masks.dtype)
    )
    save_targets = targets[:, save_index]
    matrix = F.binary_cross_entropy_with_logits(
        logits[:, save_index], save_targets, reduction="none"
    )
    positive_mask = condition * (save_targets > 0.5).to(masks.dtype)
    negative_mask = condition * (save_targets <= 0.5).to(masks.dtype)
    positive_present = (positive_mask.sum() > 0).to(logits.dtype)
    negative_present = (negative_mask.sum() > 0).to(logits.dtype)
    positive_loss = masked_mean(matrix, positive_mask)
    negative_loss = masked_mean(matrix, negative_mask)
    term_count = positive_present + negative_present
    total = (
        positive_loss * positive_present + negative_loss * negative_present
    ) / term_count.clamp_min(1.0)
    return total, {
        "save_given_shot_loss": float(total.detach().cpu()),
        "save_given_shot_positive_loss": float(positive_loss.detach().cpu()),
        "save_given_shot_negative_loss": float(negative_loss.detach().cpu()),
        "save_given_shot_positive_slots": float(positive_mask.sum().detach().cpu()),
        "save_given_shot_negative_slots": float(negative_mask.sum().detach().cpu()),
    }


def save_shot_cohort_classification_loss(
    logits: Tensor,
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[Tensor, dict[str, float]]:
    """Dataset-weighted save-focus versus trusted shot-only supervision.

    The denominator is the local microbatch size, not the weight sum. Therefore
    fixed dataset-level weights retain their intended mass under DDP and gradient
    accumulation even when a per-rank microbatch contains only one cohort.
    """
    if "save" not in LABEL_TO_INDEX:
        return logits.new_zeros(()), {}
    targets = batch.get("save_cohort_targets")
    masks = batch.get("save_cohort_masks")
    weights = batch.get("save_cohort_weights")
    if targets is None or masks is None or weights is None:
        return logits.new_zeros(()), {}
    targets = targets.to(device=device, dtype=logits.dtype, non_blocking=True)
    masks = masks.to(device=device, dtype=logits.dtype, non_blocking=True)
    weights = weights.to(device=device, dtype=logits.dtype, non_blocking=True)
    save_logits = logits[:, LABEL_TO_INDEX["save"]]
    matrix = F.binary_cross_entropy_with_logits(save_logits, targets, reduction="none")
    effective = masks * weights
    total = (matrix * effective).sum() / max(int(logits.shape[0]), 1)
    positive = masks * (targets > 0.5).to(logits.dtype)
    negative = masks * (targets <= 0.5).to(logits.dtype)
    return total, {
        "save_shot_cohort_loss": float(total.detach().cpu()),
        "save_cohort_positive_slots": float(positive.sum().detach().cpu()),
        "save_cohort_negative_slots": float(negative.sum().detach().cpu()),
        "save_cohort_weight_mass": float(effective.sum().detach().cpu()),
        "save_cohort_positive_loss": float(masked_mean(matrix, positive).detach().cpu()),
        "save_cohort_negative_loss": float(masked_mean(matrix, negative).detach().cpu()),
    }


def set_piece_subtype_classification_loss(
    outputs: dict[str, Tensor],
    batch: dict[str, Any],
    device: torch.device,
    cfg: Any,
) -> tuple[Tensor, dict[str, float]]:
    logits = outputs.get("set_piece_subtype_logits")
    targets = batch.get("set_piece_subtype_targets")
    row_mask = batch.get("set_piece_subtype_mask")
    if logits is None or targets is None or row_mask is None:
        return next(iter(outputs.values())).new_zeros(()), {}
    targets = targets.to(device=device, dtype=logits.dtype, non_blocking=True)
    row_mask = row_mask.to(device=device, dtype=logits.dtype, non_blocking=True)
    if logits.shape != targets.shape:
        raise ValueError(
            f"set-piece subtype logits{tuple(logits.shape)} != targets{tuple(targets.shape)}"
        )
    raw_pos_weight = cfg.train.get("set_piece_subtype_pos_weight", ConfigDict())
    pos_weight = torch.tensor(
        [float(raw_pos_weight.get(label, 1.0)) for label in SET_PIECE_SUBTYPES],
        device=device,
        dtype=logits.dtype,
    )
    matrix = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pos_weight
    )
    expanded_mask = row_mask.reshape(-1, 1).expand_as(matrix)
    balance = str(
        cfg.train.get("set_piece_subtype_balance", "masked_bce")
    ).strip().lower()
    if balance == "masked_bce":
        total = masked_mean(matrix, expanded_mask)
    elif balance == "per_class_equal_pos_neg":
        # Every valid row is a reviewed set-piece clip. Balancing positive and
        # negative subtype slots per class prevents common corner/freekick
        # rows from drowning rare penalty/kickoff gradients without changing
        # the aggregate set_piece target.
        total, _, _ = per_class_equal_pos_neg_loss(
            matrix, targets, expanded_mask
        )
    else:
        raise ValueError(
            "train.set_piece_subtype_balance must be masked_bce or "
            "per_class_equal_pos_neg"
        )
    with torch.no_grad():
        predicted = logits.argmax(dim=1)
        target_index = targets.argmax(dim=1)
        single_label = (targets.sum(dim=1) == 1).to(logits.dtype) * row_mask
        top1 = masked_mean(
            (predicted == target_index).to(logits.dtype), single_label
        )
    diagnostics = {
        "set_piece_subtype_loss": float(total.detach().cpu()),
        "set_piece_subtype_slots": float(row_mask.sum().detach().cpu()),
        "set_piece_subtype_top1": float(top1.detach().cpu()),
    }
    with torch.no_grad():
        for index, subtype in enumerate(SET_PIECE_SUBTYPES):
            subtype_mask = expanded_mask[:, index]
            positive_mask = subtype_mask * (targets[:, index] > 0.5).to(logits.dtype)
            negative_mask = subtype_mask * (targets[:, index] <= 0.5).to(logits.dtype)
            diagnostics[f"set_piece_subtype_positive_slots_{subtype}"] = float(
                positive_mask.sum().detach().cpu()
            )
            diagnostics[f"set_piece_subtype_negative_slots_{subtype}"] = float(
                negative_mask.sum().detach().cpu()
            )
    return total, diagnostics


def raw_set_piece_context_span_coverage_loss(
    outputs: dict[str, Tensor],
    batch: dict[str, Any],
    device: torch.device,
    cfg: Any,
) -> tuple[Tensor, dict[str, float]]:
    logits = outputs.get("frame_event_logits")
    span_mask = batch.get("set_piece_context_span_mask")
    row_mask = batch.get("set_piece_context_span_row_mask")
    set_piece_index = LABEL_TO_INDEX.get("set_piece", -1)
    if logits is None or span_mask is None or row_mask is None or set_piece_index < 0:
        return next(iter(outputs.values())).new_zeros(()), {}
    span_mask = span_mask.to(device=device, dtype=logits.dtype, non_blocking=True)
    row_mask = row_mask.to(device=device, dtype=logits.dtype, non_blocking=True)
    if logits.shape[:2] != span_mask.shape:
        raise ValueError(
            f"context span mask{tuple(span_mask.shape)} != frame logits{tuple(logits.shape[:2])}"
        )
    objective = str(
        cfg.train.get("raw_context_span_objective", "softmax_mass")
    ).strip().lower()
    span_frames = span_mask.sum(dim=1).clamp_min(1.0)
    frame_probabilities = torch.sigmoid(logits[:, :, set_piece_index])
    span_mean_probability = (
        frame_probabilities * span_mask
    ).sum(dim=1) / span_frames
    effective_row_mask = row_mask
    relative_diagnostics: dict[str, Tensor] = {}
    if objective == "softmax_mass":
        temperature = max(
            float(cfg.train.get("raw_context_span_temperature", 1.0) or 1.0),
            1e-4,
        )
        attention = torch.softmax(
            logits[:, :, set_piece_index] / temperature, dim=1
        )
        covered_mass = (attention * span_mask).sum(dim=1).clamp_min(1e-6)
        per_row = -covered_mass.log()
    elif objective == "mean_probability_floor":
        minimum_probability = float(
            cfg.train.get("raw_context_span_min_probability", 0.25) or 0.25
        )
        if not 0.0 < minimum_probability < 1.0:
            raise ValueError(
                "train.raw_context_span_min_probability must be in (0, 1)"
            )
        softness = max(
            float(cfg.train.get("raw_context_span_softness", 0.05) or 0.05),
            1e-4,
        )
        violation = minimum_probability - span_mean_probability
        per_row = (
            F.softplus(violation / softness) * softness
            - math.log(2.0) * softness
        ).clamp_min(0.0)
        covered_mass = span_mean_probability
    elif objective == "relative_logit_rank":
        # Weak relative supervision: at least one informative part of the raw
        # span should outrank the same clip's outside context. A global logit
        # shift cancels out and therefore cannot inflate absolute probability.
        temperature = max(
            float(
                cfg.train.get(
                    "raw_context_span_relative_temperature", 0.5
                ) or 0.5
            ),
            1e-4,
        )
        margin = float(
            cfg.train.get("raw_context_span_relative_margin", 0.15) or 0.15
        )
        softness = max(
            float(
                cfg.train.get(
                    "raw_context_span_relative_softness", 0.05
                ) or 0.05
            ),
            1e-4,
        )
        set_piece_logits = logits[:, :, set_piece_index]
        outside_mask = (span_mask <= 0.5).to(logits.dtype)
        outside_frames = outside_mask.sum(dim=1)
        effective_row_mask = row_mask * (outside_frames > 0).to(logits.dtype)
        finite_min = torch.finfo(logits.dtype).min

        def normalized_logsumexp(
            values: Tensor, mask: Tensor, count: Tensor
        ) -> Tensor:
            scaled = (values / temperature).masked_fill(
                mask <= 0.5, finite_min
            )
            return temperature * (
                torch.logsumexp(scaled, dim=1)
                - count.clamp_min(1.0).log()
            )

        inside_score = normalized_logsumexp(
            set_piece_logits, span_mask, span_frames
        )
        outside_score = normalized_logsumexp(
            set_piece_logits, outside_mask, outside_frames
        )
        relative_gap = inside_score - outside_score
        violation = margin - relative_gap
        per_row = (
            F.softplus(violation / softness) * softness
            - math.log(2.0) * softness
        ).clamp_min(0.0)
        covered_mass = span_mean_probability
        relative_diagnostics = {
            "raw_context_span_relative_inside_logit": masked_mean(
                inside_score, effective_row_mask
            ),
            "raw_context_span_relative_outside_logit": masked_mean(
                outside_score, effective_row_mask
            ),
            "raw_context_span_relative_gap": masked_mean(
                relative_gap, effective_row_mask
            ),
            "raw_context_span_relative_violation_fraction": masked_mean(
                (relative_gap < margin).to(logits.dtype),
                effective_row_mask,
            ),
        }
    else:
        raise ValueError(
            "train.raw_context_span_objective must be softmax_mass, "
            "mean_probability_floor or relative_logit_rank"
        )
    total = masked_mean(per_row, effective_row_mask)
    diagnostics = {
        "raw_context_span_loss": float(total.detach().cpu()),
        "raw_context_span_slots": float(
            effective_row_mask.sum().detach().cpu()
        ),
        "raw_context_span_covered_mass": float(
            masked_mean(covered_mass, effective_row_mask).detach().cpu()
        ),
        "raw_context_span_mean_probability": float(
            masked_mean(
                span_mean_probability, effective_row_mask
            ).detach().cpu()
        ),
        "raw_context_span_mean_frames": float(
            masked_mean(span_frames, effective_row_mask).detach().cpu()
        ),
    }
    diagnostics.update(
        {
            name: float(value.detach().cpu())
            for name, value in relative_diagnostics.items()
        }
    )
    return total, diagnostics

def spatial_attention_auxiliary_loss(
    outputs: dict[str, Tensor],
    targets: Tensor,
    label_masks: Tensor,
    cfg: Any,
) -> tuple[Tensor, dict[str, float]]:
    frame_logits = outputs.get("spatial_frame_event_logits")
    if frame_logits is None:
        return targets.new_zeros(()), {}
    train_cfg = cfg.get("train", ConfigDict())
    frame_logits = frame_logits.to(targets.dtype)
    topk = min(
        max(int(train_cfg.get("spatial_attention_mil_topk", 3)), 1),
        frame_logits.shape[1],
    )
    topk_result = torch.topk(frame_logits, k=topk, dim=1)
    pooled_logits = (
        topk_result.values.logsumexp(dim=1) - math.log(float(topk))
    )
    mil_matrix = F.binary_cross_entropy_with_logits(
        pooled_logits,
        targets.to(pooled_logits.dtype),
        reduction="none",
    )
    masks = label_masks.to(pooled_logits.dtype)
    positive_mask = masks * (targets > 0.5).to(masks.dtype)
    negative_mask = masks * (targets <= 0.5).to(masks.dtype)
    mil_balance = str(
        train_cfg.get("spatial_attention_mil_balance", "all")
    ).strip().lower()
    if mil_balance == "per_class_equal_pos_neg":
        positive_count = positive_mask.sum(dim=0)
        negative_count = negative_mask.sum(dim=0)
        positive_loss_per_class = (
            (mil_matrix * positive_mask).sum(dim=0)
            / positive_count.clamp_min(1.0)
        )
        negative_loss_per_class = (
            (mil_matrix * negative_mask).sum(dim=0)
            / negative_count.clamp_min(1.0)
        )
        positive_present = (positive_count > 0).to(mil_matrix.dtype)
        negative_present = (negative_count > 0).to(mil_matrix.dtype)
        class_terms = positive_present + negative_present
        class_loss = (
            positive_loss_per_class * positive_present
            + negative_loss_per_class * negative_present
        ) / class_terms.clamp_min(1.0)
        mil_loss = (class_loss * (class_terms > 0)).sum() / (
            (class_terms > 0).sum().clamp_min(1)
        )
    elif mil_balance == "equal_pos_neg":
        positive_present = (positive_mask.sum() > 0).to(mil_matrix.dtype)
        negative_present = (negative_mask.sum() > 0).to(mil_matrix.dtype)
        mil_loss = (
            masked_mean(mil_matrix, positive_mask) * positive_present
            + masked_mean(mil_matrix, negative_mask) * negative_present
        ) / (positive_present + negative_present).clamp_min(1.0)
    elif mil_balance == "all":
        mil_loss = masked_mean(mil_matrix, masks)
    else:
        raise ValueError(
            "train.spatial_attention_mil_balance must be all, "
            "equal_pos_neg, or per_class_equal_pos_neg"
        )

    diversity = outputs.get("spatial_query_diversity_loss")
    diversity_loss = (
        mil_loss.new_zeros(()) if diversity is None else diversity.mean()
    )
    entropy = outputs.get("spatial_attention_entropy")
    overlap = outputs.get("spatial_attention_overlap")
    concentration_loss = mil_loss.new_zeros(())
    overlap_loss = mil_loss.new_zeros(())
    selected_entropy = mil_loss.new_zeros(())
    selected_overlap = mil_loss.new_zeros(())
    if entropy is not None:
        entropy_per_frame = entropy.to(mil_loss.dtype).mean(dim=-1)
        selected_entropy_per_class = torch.gather(
            entropy_per_frame,
            1,
            topk_result.indices.detach(),
        ).mean(dim=1)
        selected_entropy = masked_mean(
            selected_entropy_per_class, positive_mask
        )
        entropy_target = float(
            train_cfg.get("spatial_attention_entropy_target", 0.85)
        )
        concentration_loss = masked_mean(
            F.relu(selected_entropy_per_class - entropy_target),
            positive_mask,
        )
    if overlap is not None:
        selected_overlap_per_class = torch.gather(
            overlap.to(mil_loss.dtype),
            1,
            topk_result.indices.detach(),
        ).mean(dim=1)
        selected_overlap = masked_mean(
            selected_overlap_per_class, positive_mask
        )
        overlap_loss = selected_overlap

    mil_weight = float(
        train_cfg.get("spatial_attention_mil_loss_weight", 0.0) or 0.0
    )
    diversity_weight = float(
        train_cfg.get(
            "spatial_attention_query_diversity_loss_weight", 0.0
        )
        or 0.0
    )
    concentration_weight = float(
        train_cfg.get(
            "spatial_attention_concentration_loss_weight", 0.0
        )
        or 0.0
    )
    overlap_weight = float(
        train_cfg.get("spatial_attention_overlap_loss_weight", 0.0)
        or 0.0
    )
    total = (
        mil_weight * mil_loss
        + diversity_weight * diversity_loss
        + concentration_weight * concentration_loss
        + overlap_weight * overlap_loss
    )
    components = {
        "spatial_attention_loss": float(total.detach().cpu()),
        "spatial_attention_mil_loss": float(mil_loss.detach().cpu()),
        "spatial_attention_concentration_loss": float(
            concentration_loss.detach().cpu()
        ),
        "spatial_attention_selected_entropy": float(
            selected_entropy.detach().cpu()
        ),
        "spatial_attention_overlap_loss": float(overlap_loss.detach().cpu()),
        "spatial_attention_selected_overlap": float(
            selected_overlap.detach().cpu()
        ),
        "spatial_attention_query_diversity_loss": float(
            diversity_loss.detach().cpu()
        ),
        "spatial_attention_mil_pos_loss": float(
            masked_mean(mil_matrix, positive_mask).detach().cpu()
        ),
        "spatial_attention_mil_neg_loss": float(
            masked_mean(mil_matrix, negative_mask).detach().cpu()
        ),
    }
    gate = outputs.get("spatial_gate")
    if gate is not None:
        mean_gate = gate.detach().float().mean(dim=(0, 1)).reshape(-1)
        for label_index in range(mean_gate.numel()):
            label = (
                LABELS[label_index]
                if label_index < len(LABELS)
                else f"class_{label_index}"
            )
            components[f"spatial_gate_{label}"] = float(
                mean_gate[label_index].cpu()
            )
    if entropy is not None:
        mean_entropy = entropy.detach().float().mean(dim=(0, 1, 3)).reshape(-1)
        for label_index in range(mean_entropy.numel()):
            label = (
                LABELS[label_index]
                if label_index < len(LABELS)
                else f"class_{label_index}"
            )
            components[f"spatial_attention_entropy_{label}"] = float(
                mean_entropy[label_index].cpu()
            )
    return total, components


def spatial_token_pooling_auxiliary_loss(
    outputs: dict[str, Tensor],
    targets: Tensor,
    label_masks: Tensor,
    cfg: Any,
) -> tuple[Tensor, dict[str, float]]:
    slot_logits = outputs.get("spatial_token_slot_logits")
    if slot_logits is None:
        return targets.new_zeros(()), {}
    if slot_logits.ndim != 4:
        raise ValueError(
            "spatial_token_slot_logits must have shape [B,T,K,C]"
        )
    train_cfg = cfg.get("train", ConfigDict())
    slot_logits = slot_logits.to(targets.dtype)
    bsz, frames, queries, labels = slot_logits.shape
    candidates = slot_logits.reshape(bsz, frames * queries, labels)
    topk = min(
        max(int(train_cfg.get("spatial_token_mil_topk", 4)), 1),
        candidates.shape[1],
    )
    selected = torch.topk(candidates, k=topk, dim=1).values
    pooled_logits = selected.logsumexp(dim=1) - math.log(float(topk))
    loss_matrix = F.binary_cross_entropy_with_logits(
        pooled_logits,
        targets.to(pooled_logits.dtype),
        reduction="none",
    )
    masks = label_masks.to(loss_matrix.dtype)
    balance = str(
        train_cfg.get(
            "spatial_token_mil_balance", "per_class_equal_pos_neg"
        )
    ).strip().lower()
    if balance == "per_class_equal_pos_neg":
        mil_loss, positive_loss, negative_loss = per_class_equal_pos_neg_loss(
            loss_matrix, targets, masks
        )
    elif balance == "all":
        positive_mask = masks * (targets > 0.5).to(masks.dtype)
        negative_mask = masks * (targets <= 0.5).to(masks.dtype)
        mil_loss = masked_mean(loss_matrix, masks)
        positive_loss = masked_mean(loss_matrix, positive_mask)
        negative_loss = masked_mean(loss_matrix, negative_mask)
    else:
        raise ValueError(
            "train.spatial_token_mil_balance must be all or "
            "per_class_equal_pos_neg"
        )

    diversity = outputs.get("spatial_token_query_diversity_loss")
    diversity_loss = (
        mil_loss.new_zeros(()) if diversity is None else diversity.mean()
    )
    overlap = outputs.get("spatial_token_attention_overlap")
    overlap_loss = (
        mil_loss.new_zeros(())
        if overlap is None
        else overlap.to(mil_loss.dtype).mean()
    )
    entropy = outputs.get("spatial_token_attention_entropy")
    mean_entropy = (
        mil_loss.new_zeros(())
        if entropy is None
        else entropy.to(mil_loss.dtype).mean()
    )

    mil_weight = float(
        train_cfg.get("spatial_token_mil_loss_weight", 0.0) or 0.0
    )
    diversity_weight = float(
        train_cfg.get("spatial_token_query_diversity_loss_weight", 0.0)
        or 0.0
    )
    overlap_weight = float(
        train_cfg.get("spatial_token_overlap_loss_weight", 0.0) or 0.0
    )
    total = (
        mil_weight * mil_loss
        + diversity_weight * diversity_loss
        + overlap_weight * overlap_loss
    )
    components = {
        "spatial_token_loss": float(total.detach().cpu()),
        "spatial_token_mil_loss": float(mil_loss.detach().cpu()),
        "spatial_token_mil_pos_loss": float(positive_loss.detach().cpu()),
        "spatial_token_mil_neg_loss": float(negative_loss.detach().cpu()),
        "spatial_token_query_diversity_loss": float(
            diversity_loss.detach().cpu()
        ),
        "spatial_token_attention_overlap": float(overlap_loss.detach().cpu()),
        "spatial_token_attention_entropy": float(mean_entropy.detach().cpu()),
    }
    gate = outputs.get("spatial_token_slot_gate")
    if gate is not None:
        for slot_index, value in enumerate(
            gate.detach().float().reshape(-1)
        ):
            components[f"spatial_token_gate_{slot_index}"] = float(
                value.cpu()
            )
    context_scale = outputs.get("spatial_token_context_scale")
    if context_scale is not None:
        components["spatial_token_context_scale"] = float(
            context_scale.detach().float().mean().cpu()
        )
    return total, components



def negative_teacher_guard_loss(
    outputs: dict[str, Tensor],
    targets: Tensor,
    label_masks: Tensor,
    *,
    label_indices: list[int] | tuple[int, ...] | None = None,
    margin: float = 0.0,
    branch: str = "fused",
) -> tuple[Tensor, int]:
    """Prevent trusted-negative logits from rising above a frozen reference."""
    if "global_logits" not in outputs:
        return targets.new_zeros(()), 0
    branch = str(branch).strip().lower()
    candidate_key = {"fused": "logits", "event": "event_logits"}.get(branch)
    if candidate_key is None:
        raise ValueError("negative teacher guard branch must be fused or event")
    if candidate_key not in outputs:
        raise ValueError(
            f"negative teacher guard branch={branch} requires "
            f"outputs[{candidate_key!r}]"
        )
    candidate_logits = outputs[candidate_key]
    reference_logits = outputs.get(
        "retention_reference_logits", outputs["global_logits"]
    ).detach()
    if label_indices is None:
        label_indices = tuple(range(candidate_logits.shape[1]))
    selected_mask = torch.zeros_like(label_masks, dtype=torch.bool)
    for label_index in label_indices:
        if not 0 <= int(label_index) < candidate_logits.shape[1]:
            raise ValueError(
                f"invalid negative teacher guard label index: {label_index}"
            )
        selected_mask[:, int(label_index)] = (
            (label_masks[:, int(label_index)] > 0)
            & (targets[:, int(label_index)] <= 0.5)
        )
    excess = F.relu(candidate_logits - reference_logits - float(margin))
    violating = selected_mask & (excess.detach() > 0)
    return masked_mean(excess.square(), selected_mask), int(violating.sum().item())


def hard_negative_pairwise_rank_loss(
    logits: Tensor,
    targets: Tensor,
    label_masks: Tensor,
    hard_rows: Tensor,
    *,
    label_indices: list[int] | tuple[int, ...] | None = None,
    margin: float = 1.0,
) -> Tensor:
    """Rank labeled positives above mined hard negatives for each class."""
    losses: list[Tensor] = []
    hard_rows = hard_rows.to(device=logits.device, dtype=torch.bool).reshape(-1)
    if label_indices is None:
        label_indices = tuple(range(logits.shape[1]))
    for label_index in label_indices:
        if not 0 <= int(label_index) < logits.shape[1]:
            raise ValueError(f"invalid hard-negative rank label index: {label_index}")
        valid = label_masks[:, label_index] > 0
        positives = valid & (targets[:, label_index] > 0.5)
        hard_negatives = valid & hard_rows & (targets[:, label_index] <= 0.5)
        if not bool(positives.any()) or not bool(hard_negatives.any()):
            continue
        positive_logits = logits[positives, label_index].reshape(-1, 1)
        negative_logits = logits[hard_negatives, label_index].reshape(1, -1)
        losses.append(
            F.softplus(
                logits.new_tensor(float(margin)) - positive_logits + negative_logits
            ).mean()
        )
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def clean_negative_pairwise_rank_loss(
    logits: Tensor,
    targets: Tensor,
    label_masks: Tensor,
    clean_negative_rows: Tensor,
    *,
    label_indices: list[int] | tuple[int, ...] | None = None,
    margin: float = 1.0,
    pairs_per_positive: int = 2,
    temperature: float = 1.0,
) -> tuple[Tensor, dict[str, float]]:
    """Rank positives above uniformly sampled trusted clean negatives."""
    clean_negative_rows = clean_negative_rows.to(
        device=logits.device, dtype=torch.bool
    )
    if clean_negative_rows.ndim == 1:
        if clean_negative_rows.numel() != logits.shape[0]:
            raise ValueError("clean_negative_rows must have one value per batch row")
    elif clean_negative_rows.ndim == 2:
        if tuple(clean_negative_rows.shape) != tuple(logits.shape):
            raise ValueError(
                "per-class clean-negative mask must match logits: "
                f"mask={tuple(clean_negative_rows.shape)} logits={tuple(logits.shape)}"
            )
    else:
        raise ValueError("clean-negative mask must be [B] or [B,C]")
    if label_indices is None:
        label_indices = tuple(range(logits.shape[1]))
    pair_count = max(int(pairs_per_positive), 1)
    tau = max(float(temperature), 1e-6)
    losses: list[Tensor] = []
    detached_gaps: list[Tensor] = []
    detached_violations: list[Tensor] = []
    positive_values: list[Tensor] = []
    negative_values: list[Tensor] = []
    num_pairs = 0
    active_classes = 0
    for label_index in label_indices:
        label_index = int(label_index)
        if not 0 <= label_index < logits.shape[1]:
            raise ValueError(
                f"invalid clean-negative rank label index: {label_index}"
            )
        valid = label_masks[:, label_index] > 0
        positives = valid & (targets[:, label_index] > 0.5)
        class_clean_negatives = (
            clean_negative_rows
            if clean_negative_rows.ndim == 1
            else clean_negative_rows[:, label_index]
        )
        negatives = (
            valid
            & class_clean_negatives
            & (targets[:, label_index] <= 0.5)
        )
        if not bool(positives.any()) or not bool(negatives.any()):
            continue
        pos_logits = logits[positives, label_index]
        neg_logits = logits[negatives, label_index]
        sampled_indices = torch.randint(
            neg_logits.numel(),
            (pos_logits.numel(), pair_count),
            device=logits.device,
        )
        sampled_negatives = neg_logits[sampled_indices]
        gaps = pos_logits.reshape(-1, 1) - sampled_negatives
        losses.append(
            (
                tau
                * F.softplus(
                    (logits.new_tensor(float(margin)) - gaps) / tau
                )
            ).mean()
        )
        detached_gaps.append(gaps.detach().reshape(-1))
        detached_violations.append(
            (gaps.detach() < float(margin)).reshape(-1)
        )
        positive_values.append(pos_logits.detach())
        negative_values.append(sampled_negatives.detach().reshape(-1))
        num_pairs += gaps.numel()
        active_classes += 1
    if not losses:
        zero = logits.sum() * 0.0
        return zero, {
            "clean_negative_rank_pairs": 0.0,
            "clean_negative_rank_active_classes": 0.0,
            "clean_negative_rank_gap": 0.0,
            "clean_negative_rank_violation_fraction": 0.0,
            "clean_negative_rank_positive_logit": 0.0,
            "clean_negative_rank_negative_logit": 0.0,
        }
    gaps = torch.cat(detached_gaps)
    violations = torch.cat(detached_violations).float()
    return torch.stack(losses).mean(), {
        "clean_negative_rank_pairs": float(num_pairs),
        "clean_negative_rank_active_classes": float(active_classes),
        "clean_negative_rank_gap": float(gaps.mean().cpu()),
        "clean_negative_rank_violation_fraction": float(
            violations.mean().cpu()
        ),
        "clean_negative_rank_positive_logit": float(
            torch.cat(positive_values).mean().cpu()
        ),
        "clean_negative_rank_negative_logit": float(
            torch.cat(negative_values).mean().cpu()
        ),
    }


def online_hard_negative_loss(
    logits: Tensor,
    targets: Tensor,
    label_masks: Tensor,
    *,
    label_indices: list[int] | tuple[int, ...] | None = None,
    fraction: float = 0.25,
    min_per_class: int = 1,
) -> tuple[Tensor, int]:
    """Mine the highest-scoring trusted negatives in the current batch."""
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("online hard-negative fraction must be in (0, 1]")
    min_per_class = max(int(min_per_class), 1)
    if label_indices is None:
        label_indices = tuple(range(logits.shape[1]))
    selected_losses: list[Tensor] = []
    selected_count = 0
    for label_index in label_indices:
        if not 0 <= int(label_index) < logits.shape[1]:
            raise ValueError(f"invalid online hard-negative label index: {label_index}")
        negative_mask = (
            (label_masks[:, label_index] > 0)
            & (targets[:, label_index] <= 0.5)
        )
        negative_logits = logits[negative_mask, label_index]
        if negative_logits.numel() == 0:
            continue
        count = int(negative_logits.numel())
        keep = min(
            count,
            max(min_per_class, int(math.ceil(count * float(fraction)))),
        )
        losses = F.softplus(negative_logits)
        selected_losses.append(torch.topk(losses, k=keep, sorted=False).values)
        selected_count += keep
    if not selected_losses:
        return logits.sum() * 0.0, 0
    return torch.cat(selected_losses).mean(), selected_count


def online_hard_negative_rank_loss(
    logits: Tensor,
    targets: Tensor,
    label_masks: Tensor,
    *,
    label_indices: list[int] | tuple[int, ...] | None = None,
    fraction: float = 0.25,
    min_per_class: int = 1,
    margin: float = 1.0,
) -> tuple[Tensor, int]:
    """Rank positives above the highest-scoring trusted batch negatives."""
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("online hard-negative rank fraction must be in (0, 1]")
    min_per_class = max(int(min_per_class), 1)
    if label_indices is None:
        label_indices = tuple(range(logits.shape[1]))
    rank_losses: list[Tensor] = []
    selected_count = 0
    for label_index in label_indices:
        if not 0 <= int(label_index) < logits.shape[1]:
            raise ValueError(
                f"invalid online hard-negative rank label index: {label_index}"
            )
        valid = label_masks[:, label_index] > 0
        positive_logits = logits[
            valid & (targets[:, label_index] > 0.5), label_index
        ]
        negative_logits = logits[
            valid & (targets[:, label_index] <= 0.5), label_index
        ]
        if positive_logits.numel() == 0 or negative_logits.numel() == 0:
            continue
        negative_count = int(negative_logits.numel())
        keep = min(
            negative_count,
            max(
                min_per_class,
                int(math.ceil(negative_count * float(fraction))),
            ),
        )
        hard_indices = torch.topk(
            negative_logits.detach(), k=keep, sorted=False
        ).indices
        hard_negative_logits = negative_logits[hard_indices]
        rank_losses.append(
            F.softplus(
                logits.new_tensor(float(margin))
                - positive_logits.reshape(-1, 1)
                + hard_negative_logits.reshape(1, -1)
            ).mean()
        )
        selected_count += keep
    if not rank_losses:
        return logits.sum() * 0.0, 0
    return torch.stack(rank_losses).mean(), selected_count


def temporal_gate_quality_loss(
    outputs: dict[str, Tensor],
    targets: Tensor,
    label_masks: Tensor,
    *,
    temperature: float = 0.25,
) -> tuple[Tensor, dict[str, float]]:
    required = ("uniform_logits", "event_logits", "temporal_gate")
    if any(key not in outputs for key in required):
        return targets.new_zeros(()), {}
    uniform_error = F.binary_cross_entropy_with_logits(
        outputs["uniform_logits"].detach(), targets, reduction="none"
    )
    event_error = F.binary_cross_entropy_with_logits(
        outputs["event_logits"].detach(), targets, reduction="none"
    )
    advantage = uniform_error - event_error
    temperature = max(float(temperature), 1e-4)
    quality_target = torch.sigmoid(advantage / temperature).detach()
    gate = outputs["temporal_gate"].clamp(1e-5, 1.0 - 1e-5)
    quality_matrix = F.binary_cross_entropy(
        gate, quality_target, reduction="none"
    )
    quality_loss = masked_mean(quality_matrix, label_masks)
    valid_count = label_masks.sum().clamp_min(1.0)
    return quality_loss, {
        "temporal_gate_quality_loss": float(quality_loss.detach().cpu()),
        "temporal_gate_quality_target": float(
            ((quality_target * label_masks).sum() / valid_count).detach().cpu()
        ),
        "temporal_event_advantage": float(
            ((advantage * label_masks).sum() / valid_count).detach().cpu()
        ),
    }


def frame_rank_loss(
    logits: Tensor,
    targets: Tensor,
    masks: Tensor,
    *,
    pos_threshold: float,
    neg_threshold: float,
    margin: float,
) -> Tensor:
    losses: list[Tensor] = []
    for batch_index in range(logits.shape[0]):
        for label_index in range(logits.shape[2]):
            valid = masks[batch_index, :, label_index] > 0
            if not bool(valid.any()):
                continue
            label_targets = targets[batch_index, :, label_index]
            pos_mask = valid & (label_targets >= pos_threshold)
            neg_mask = valid & (label_targets <= neg_threshold)
            if not bool(pos_mask.any()) or not bool(neg_mask.any()):
                continue
            label_logits = logits[batch_index, :, label_index]
            pos_score = label_logits[pos_mask].max()
            neg_score = label_logits[neg_mask].max()
            losses.append(F.softplus(logits.new_tensor(float(margin)) - pos_score + neg_score))
    if not losses:
        return logits.new_zeros(())
    return torch.stack(losses).mean()


def frame_tail_rank_loss(
    logits: Tensor,
    targets: Tensor,
    masks: Tensor,
    label_masks: Tensor,
    *,
    margin: float = 1.0,
    tau: float = 0.5,
    beta: float = 1.0,
    neg_threshold: float = 0.0,
    label_indices: list[int] | tuple[int, ...] | None = None,
) -> tuple[Tensor, dict[str, float]]:
    """Rank the lowest positive frame tail above the highest background tail.

    Inverse of frame_rank_loss: instead of the strongest positive frame, the
    soft-min positive frame must clear the soft-max background frame by
    `margin`. Each (sample, class) contributes an in-window term L_in; a
    batch-level term L_out pools all positive frames against all background
    frames (including negative windows) per class within the local DDP rank.
    Returns the per-class mean of (L_in + L_out) / 2 plus diagnostics.
    """
    if logits.ndim != 3:
        raise ValueError(
            f"frame_tail_rank_loss expects [B,T,C] logits, got {tuple(logits.shape)}"
        )
    if targets.shape != logits.shape or masks.shape != logits.shape:
        raise ValueError(
            f"frame targets/masks {tuple(targets.shape)}/{tuple(masks.shape)} "
            f"must match logits {tuple(logits.shape)}"
        )
    if label_masks.shape != (logits.shape[0], logits.shape[2]):
        raise ValueError(
            f"label_masks {tuple(label_masks.shape)} must be "
            f"[{logits.shape[0]},{logits.shape[2]}] for logits {tuple(logits.shape)}"
        )
    tau = float(tau)
    beta = float(beta)
    if tau <= 0 or beta <= 0:
        raise ValueError("frame_tail_rank tau/beta must be positive")
    if label_indices is None:
        label_indices = tuple(range(logits.shape[2]))
    scores = logits.float()
    targets = targets.float()
    valid_frames = masks > 0
    trusted = label_masks > 0
    class_losses: list[Tensor] = []
    gaps: list[Tensor] = []
    in_term_count = 0
    out_term_count = 0
    for label_index in label_indices:
        idx = int(label_index)
        if not 0 <= idx < logits.shape[2]:
            raise ValueError(f"invalid frame_tail_rank label index: {idx}")
        class_trusted = trusted[:, idx]
        if not bool(class_trusted.any()):
            continue
        class_valid = valid_frames[:, :, idx] & class_trusted.reshape(-1, 1)
        class_targets = targets[:, :, idx]
        class_scores = scores[:, :, idx]
        pos_frames = class_valid & (class_targets > 0.5)
        neg_frames = class_valid & (class_targets <= float(neg_threshold))
        in_terms: list[Tensor] = []
        for batch_index in range(logits.shape[0]):
            pos_mask = pos_frames[batch_index]
            neg_mask = neg_frames[batch_index]
            if not bool(pos_mask.any()) or not bool(neg_mask.any()):
                continue
            pos_tail = -tau * torch.logsumexp(-class_scores[batch_index][pos_mask] / tau, dim=0)
            neg_tail = tau * torch.logsumexp(class_scores[batch_index][neg_mask] / tau, dim=0)
            gap = pos_tail - neg_tail
            in_terms.append(F.softplus((float(margin) - gap) / beta) * beta)
            gaps.append(gap)
        out_terms: list[Tensor] = []
        if bool(pos_frames.any()) and bool(neg_frames.any()):
            pos_tail = -tau * torch.logsumexp(-class_scores[pos_frames] / tau, dim=0)
            neg_tail = tau * torch.logsumexp(class_scores[neg_frames] / tau, dim=0)
            gap = pos_tail - neg_tail
            out_terms.append(F.softplus((float(margin) - gap) / beta) * beta)
            gaps.append(gap)
        in_term_count += len(in_terms)
        out_term_count += len(out_terms)
        parts: list[Tensor] = []
        if in_terms:
            parts.append(torch.stack(in_terms).mean())
        if out_terms:
            parts.append(torch.stack(out_terms).mean())
        if parts:
            class_losses.append(torch.stack(parts).mean())
    diagnostics = {
        "frame_tail_pos_neg_gap": float(
            torch.stack([gap.detach() for gap in gaps]).mean().cpu()
        )
        if gaps
        else 0.0,
        "frame_tail_rank_in_terms": float(in_term_count),
        "frame_tail_rank_out_terms": float(out_term_count),
        "frame_tail_rank_active_classes": float(len(class_losses)),
    }
    if not class_losses:
        return scores.new_zeros(()), diagnostics
    return torch.stack(class_losses).mean(), diagnostics


def masked_temporal_topk_pool(
    logits: Tensor,
    candidate_mask: Tensor,
    topk: int,
) -> Tensor:
    """Pool the strongest valid frame evidence without leaking outside GT windows."""
    if logits.shape != candidate_mask.shape:
        raise ValueError(
            f"temporal candidate mask {tuple(candidate_mask.shape)} must match "
            f"logits {tuple(logits.shape)}"
        )
    frames = logits.shape[1]
    topk = min(max(int(topk), 1), frames)
    valid = candidate_mask.bool()
    has_candidate = valid.any(dim=1, keepdim=True)
    valid = torch.where(has_candidate, valid, torch.ones_like(valid))
    masked_logits = logits.masked_fill(~valid, -1.0e4)
    values = torch.topk(masked_logits, k=topk, dim=1).values
    counts = valid.sum(dim=1).clamp(min=1, max=topk)
    rank = torch.arange(topk, device=logits.device).reshape(1, topk, 1)
    selected = rank < counts.unsqueeze(1)
    values = values.masked_fill(~selected, -1.0e4)
    return values.logsumexp(dim=1) - counts.to(logits.dtype).log()


def spatial_temporal_localization_loss(
    outputs: dict[str, Tensor],
    batch: dict[str, Any],
    cfg: Any,
    device: torch.device,
    *,
    logits_key: str = "spatial_frame_event_logits",
    component_prefix: str = "spatial_temporal",
) -> tuple[Tensor, dict[str, float]]:
    """Anchor ROI frame evidence to annotated event time instead of self top-k."""
    logits = outputs.get(logits_key)
    if (
        logits is None
        or "frame_targets" not in batch
        or "frame_target_masks" not in batch
    ):
        for value in outputs.values():
            if torch.is_tensor(value):
                return value.new_zeros(()), {}
        return torch.zeros((), device=device), {}

    train_cfg = cfg.get("train", ConfigDict())
    frame_targets = batch["frame_targets"].to(
        device=device, dtype=logits.dtype, non_blocking=True
    )
    frame_masks = batch["frame_target_masks"].to(
        device=device, dtype=logits.dtype, non_blocking=True
    )
    clip_targets = batch["targets"].to(
        device=device, dtype=logits.dtype, non_blocking=True
    )
    label_masks = batch["label_masks"].to(
        device=device, dtype=logits.dtype, non_blocking=True
    )
    sample_weights = batch.get("sample_loss_weights")
    if torch.is_tensor(sample_weights):
        label_masks = label_masks * sample_weights.to(
            device=device, dtype=logits.dtype, non_blocking=True
        )
    if frame_targets.shape != logits.shape:
        raise ValueError(
            f"ROI frame targets {tuple(frame_targets.shape)} must match "
            f"logits {tuple(logits.shape)}"
        )

    heatmap_matrix = focal_bce_with_logits(
        logits,
        frame_targets,
        gamma=float(train_cfg.get("spatial_temporal_focal_gamma", 2.0)),
    )
    positive_slots = label_masks * (clip_targets > 0.5).to(logits.dtype)
    negative_slots = label_masks * (clip_targets <= 0.5).to(logits.dtype)
    positive_frame_mask = frame_masks * positive_slots.unsqueeze(1)
    negative_frame_mask = frame_masks * negative_slots.unsqueeze(1)
    heatmap_pos = masked_mean(heatmap_matrix, positive_frame_mask)
    heatmap_neg = masked_mean(heatmap_matrix, negative_frame_mask)
    heatmap_loss = 0.5 * (heatmap_pos + heatmap_neg)

    restricted_candidates = torch.where(
        (clip_targets > 0.5).unsqueeze(1),
        frame_masks > 0,
        torch.ones_like(frame_masks, dtype=torch.bool),
    )
    pooled = masked_temporal_topk_pool(
        logits,
        restricted_candidates,
        int(train_cfg.get("spatial_temporal_topk", 4)),
    )
    mil_matrix = F.binary_cross_entropy_with_logits(
        pooled, clip_targets, reduction="none"
    )
    mil_loss, mil_pos, mil_neg = per_class_equal_pos_neg_loss(
        mil_matrix, clip_targets, label_masks
    )

    near_threshold = float(
        train_cfg.get("spatial_temporal_near_target_threshold", 0.5)
    )
    near = (
        (frame_targets >= near_threshold)
        & (positive_slots.unsqueeze(1) > 0)
    )
    far = (
        (frame_masks <= 0)
        & (positive_slots.unsqueeze(1) > 0)
    )
    near_score = logits.masked_fill(~near, -1.0e4).max(dim=1).values
    far_score = logits.masked_fill(~far, -1.0e4).max(dim=1).values
    rank_valid = (
        near.any(dim=1) & far.any(dim=1)
    ).to(logits.dtype) * positive_slots
    rank_margin = float(train_cfg.get("spatial_temporal_rank_margin", 0.5))
    rank_matrix = F.softplus(rank_margin - near_score + far_score)
    rank_loss = masked_mean(rank_matrix, rank_valid)

    heatmap_weight = float(
        train_cfg.get("spatial_temporal_heatmap_weight", 0.5)
    )
    mil_weight = float(train_cfg.get("spatial_temporal_mil_weight", 0.3))
    rank_weight = float(train_cfg.get("spatial_temporal_rank_weight", 0.2))
    total = (
        heatmap_weight * heatmap_loss
        + mil_weight * mil_loss
        + rank_weight * rank_loss
    )
    return total, {
        f"{component_prefix}_loss": float(total.detach().cpu()),
        f"{component_prefix}_heatmap_loss": float(heatmap_loss.detach().cpu()),
        f"{component_prefix}_heatmap_pos_loss": float(heatmap_pos.detach().cpu()),
        f"{component_prefix}_heatmap_neg_loss": float(heatmap_neg.detach().cpu()),
        f"{component_prefix}_mil_loss": float(mil_loss.detach().cpu()),
        f"{component_prefix}_mil_pos_loss": float(mil_pos.detach().cpu()),
        f"{component_prefix}_mil_neg_loss": float(mil_neg.detach().cpu()),
        f"{component_prefix}_rank_loss": float(rank_loss.detach().cpu()),
        f"{component_prefix}_rank_slots": float(rank_valid.sum().detach().cpu()),
    }


def spatial_counterfactual_causal_loss(
    outputs: dict[str, Tensor],
    targets: Tensor,
    label_masks: Tensor,
    cfg: Any,
) -> tuple[Tensor, dict[str, float]]:
    """Make selected ROI evidence stronger than its differentiable complement."""
    kept = outputs.get("spatial_clip_logits")
    erased = outputs.get("spatial_erased_clip_logits")
    if kept is None or erased is None:
        return targets.new_zeros(()), {}
    positive_mask = (
        label_masks.to(kept.dtype)
        * (targets > 0.5).to(kept.dtype)
    )
    margin = float(
        cfg.get("train", ConfigDict()).get(
            "spatial_counterfactual_margin", 0.5
        )
    )
    gap = kept - erased
    ranking = F.softplus(margin - gap)
    causal_loss = masked_mean(ranking, positive_mask)
    violation = masked_mean((gap < margin).to(kept.dtype), positive_mask)
    return causal_loss, {
        "spatial_counterfactual_loss": float(causal_loss.detach().cpu()),
        "spatial_counterfactual_gap": float(
            masked_mean(gap, positive_mask).detach().cpu()
        ),
        "spatial_counterfactual_violation": float(violation.detach().cpu()),
    }


def class_evidence_counterfactual_causal_loss(
    outputs: dict[str, Tensor],
    targets: Tensor,
    label_masks: Tensor,
    cfg: Any,
) -> tuple[Tensor, dict[str, float]]:
    """Require local evidence to help positives without perturbing negatives.

    Both predictions retain the identical global video.  The counterfactual
    replaces only the learned local evidence with the explicit no-evidence
    token, so the measured gap cannot be explained by scene context alone.
    """
    kept = outputs.get("logits")
    erased = outputs.get("class_evidence_no_evidence_logits")
    if kept is None or erased is None:
        return targets.new_zeros(()), {}
    masks = label_masks.to(kept.dtype)
    positive_mask = masks * (targets > 0.5).to(kept.dtype)
    negative_mask = masks * (targets <= 0.5).to(kept.dtype)
    train_cfg = cfg.get("train", ConfigDict())
    margin = float(
        train_cfg.get("class_evidence_counterfactual_margin", 0.35) or 0.0
    )
    negative_stability_weight = float(
        train_cfg.get(
            "class_evidence_counterfactual_negative_stability_weight", 0.25
        )
        or 0.0
    )
    gap = kept - erased
    positive_ranking = masked_mean(
        F.softplus(margin - gap), positive_mask
    )
    negative_stability = masked_mean(
        F.smooth_l1_loss(kept, erased, reduction="none"), negative_mask
    )
    total = positive_ranking + negative_stability_weight * negative_stability
    violation = masked_mean(
        (gap < margin).to(kept.dtype), positive_mask
    )
    return total, {
        "class_evidence_counterfactual_loss": float(total.detach().cpu()),
        "class_evidence_counterfactual_positive_loss": float(
            positive_ranking.detach().cpu()
        ),
        "class_evidence_counterfactual_negative_stability": float(
            negative_stability.detach().cpu()
        ),
        "class_evidence_counterfactual_gap": float(
            masked_mean(gap, positive_mask).detach().cpu()
        ),
        "class_evidence_counterfactual_violation": float(
            violation.detach().cpu()
        ),
    }


def class_evidence_no_evidence_loss(
    outputs: dict[str, Tensor],
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[Tensor, dict[str, float]]:
    """Teach class-specific routers when spatial evidence is absent.

    The target is complementary to the soft frame target: trusted negative
    frames prefer the no-evidence slot, while event-centred frames prefer real
    patches.  Partial-label masks prevent unreviewed classes from becoming
    accidental negative supervision.
    """
    mass = outputs.get("class_evidence_no_evidence_mass")
    frame_targets = batch.get("frame_targets")
    frame_masks = batch.get("frame_target_masks")
    if mass is None or not torch.is_tensor(frame_targets) or not torch.is_tensor(frame_masks):
        for value in outputs.values():
            if torch.is_tensor(value):
                return value.new_zeros(()), {}
        return torch.zeros((), device=device), {}
    targets = frame_targets.to(device, non_blocking=True).to(mass.dtype)
    masks = frame_masks.to(device, non_blocking=True).to(mass.dtype)
    label_masks = batch.get("label_masks")
    if torch.is_tensor(label_masks):
        masks = masks * label_masks.to(device, non_blocking=True).to(mass.dtype).unsqueeze(1)
    desired = (1.0 - targets).clamp(0.0, 1.0)
    mass_logits = torch.logit(
        mass.float().clamp(1e-5, 1.0 - 1e-5)
    )
    matrix = F.binary_cross_entropy_with_logits(
        mass_logits, desired.float(), reduction="none"
    ).to(mass.dtype)
    loss = masked_mean(matrix, masks)
    valid = masks.sum().clamp_min(1.0)
    mean_mass = (mass * masks).sum() / valid
    event_mask = masks * (targets >= 0.5).to(mass.dtype)
    background_mask = masks * (targets < 0.1).to(mass.dtype)
    diagnostics = {
        "class_evidence_no_evidence_loss": float(loss.detach().cpu()),
        "class_evidence_no_evidence_mass": float(mean_mass.detach().cpu()),
        "class_evidence_no_evidence_event_mass": float(
            masked_mean(mass, event_mask).detach().cpu()
        ),
        "class_evidence_no_evidence_background_mass": float(
            masked_mean(mass, background_mask).detach().cpu()
        ),
    }
    for label_index, label in enumerate(LABELS):
        diagnostics[f"class_evidence_no_evidence_event_mass_{label}"] = float(
            masked_mean(mass[:, :, label_index], event_mask[:, :, label_index]).detach().cpu()
        )
        diagnostics[f"class_evidence_no_evidence_background_mass_{label}"] = float(
            masked_mean(mass[:, :, label_index], background_mask[:, :, label_index]).detach().cpu()
        )
    return loss, diagnostics


def frame_detection_loss(outputs: dict[str, Tensor], batch: dict[str, Any], cfg: Any, device: torch.device) -> tuple[Tensor, dict[str, float]]:
    if "frame_event_logits" not in outputs or "frame_targets" not in batch or "frame_target_masks" not in batch:
        for value in outputs.values():
            if torch.is_tensor(value):
                return value.new_zeros(()), {}
        return torch.zeros((), device=device), {}

    train_cfg = cfg.get("train", ConfigDict())
    targets = batch["frame_targets"].to(device, non_blocking=True)
    masks = batch["frame_target_masks"].to(device, non_blocking=True)
    clip_targets = batch["targets"].to(device, non_blocking=True)
    label_masks = batch["label_masks"].to(device, non_blocking=True)
    sample_loss_weights = batch.get("sample_loss_weights")
    if sample_loss_weights is not None:
        sample_loss_weights = sample_loss_weights.to(device, non_blocking=True)
    hard_negative_rows: Tensor | None = None
    metas = batch.get("meta")
    if isinstance(metas, list):
        hard_negative_rows = torch.tensor(
            [
                str(meta.get("sample_id", "")).startswith("hard_neg_")
                if isinstance(meta, dict)
                else False
                for meta in metas
            ],
            dtype=torch.bool,
            device=device,
        )

    def branch_loss(
        branch_logits: Tensor,
        branch_targets: Tensor,
        branch_masks: Tensor,
        branch_label_masks: Tensor,
        class_weights: Sequence[float] | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        branch_targets = branch_targets.to(branch_logits.dtype)
        branch_masks = branch_masks.to(branch_logits.dtype)
        branch_clip_targets = clip_targets.to(branch_logits.dtype)
        branch_label_masks = branch_label_masks.to(branch_logits.dtype)
        if sample_loss_weights is not None:
            weights = sample_loss_weights.to(branch_logits.dtype)
            branch_masks = branch_masks * weights.reshape(-1, 1, 1)
            branch_label_masks = branch_label_masks * weights.reshape(-1, 1)
        if class_weights is not None:
            class_weight_tensor = branch_logits.new_tensor(class_weights).reshape(1, 1, -1)
            if bool((class_weight_tensor < 0).any()):
                raise ValueError("frame/dense class loss weights must be non-negative")
            branch_masks = branch_masks * class_weight_tensor
            branch_label_masks = branch_label_masks * class_weight_tensor.reshape(1, -1)
        if branch_targets.shape != branch_logits.shape:
            raise ValueError(
                f"frame_targets shape {tuple(branch_targets.shape)} must match frame_event_logits {tuple(branch_logits.shape)}"
            )
        frame_pos_weight = branch_logits.new_tensor(
            per_label_float_tuple(
                train_cfg.get("frame_pos_weight", [1.0] * len(LABELS)),
                default=1.0,
            )
        )
        if bool((frame_pos_weight <= 0).any()):
            raise ValueError("train.frame_pos_weight values must be positive")
        loss_type = str(train_cfg.get("frame_det_loss_type", "focal_bce"))
        if loss_type == "focal_bce":
            heatmap_matrix = focal_bce_with_logits(
                branch_logits,
                branch_targets,
                gamma=float(train_cfg.get("frame_focal_gamma", 2.0)),
            )
            # focal_bce_with_logits does not expose BCE's pos_weight. Weight
            # soft Gaussian targets continuously so background remains
            # unchanged while event-centered frames receive stronger gradients.
            heatmap_matrix = heatmap_matrix * (
                1.0
                + (frame_pos_weight.reshape(1, 1, -1) - 1.0)
                * branch_targets
            )
        elif loss_type == "bce":
            heatmap_matrix = F.binary_cross_entropy_with_logits(
                branch_logits,
                branch_targets,
                reduction="none",
                pos_weight=frame_pos_weight,
            )
        else:
            raise ValueError("train.frame_det_loss_type must be focal_bce or bce")
        heatmap_aggregation = str(
            train_cfg.get("frame_heatmap_aggregation", "all")
        ).strip().lower()
        heatmap_aggregation_diagnostics: dict[str, Tensor] = {}
        if heatmap_aggregation == "all":
            heatmap_loss = masked_mean(heatmap_matrix, branch_masks)
        elif heatmap_aggregation == "per_positive_core_context":
            heatmap_loss, heatmap_aggregation_diagnostics = (
                positive_core_context_heatmap_loss(
                    heatmap_matrix,
                    branch_targets,
                    branch_masks,
                    branch_clip_targets,
                    branch_label_masks,
                    core_min_target=float(
                        train_cfg.get("frame_heatmap_core_min_target", 0.5)
                    ),
                    context_max_target=float(
                        train_cfg.get("frame_heatmap_context_max_target", 0.1)
                    ),
                )
            )
        else:
            raise ValueError(
                "train.frame_heatmap_aggregation must be all or "
                "per_positive_core_context"
            )

        mil_topk = min(max(int(train_cfg.get("frame_mil_topk", 3)), 1), branch_logits.shape[1])
        pooled_logits = torch.topk(branch_logits, k=mil_topk, dim=1).values.logsumexp(dim=1) - math.log(float(mil_topk))
        mil_matrix = F.binary_cross_entropy_with_logits(
            pooled_logits,
            branch_clip_targets,
            reduction="none",
            pos_weight=frame_pos_weight,
        )
        mil_loss = masked_mean(mil_matrix, branch_label_masks)

        positive_clip_slots = branch_label_masks * (branch_clip_targets > 0.5).to(branch_logits.dtype)
        negative_clip_slots = branch_label_masks * (branch_clip_targets <= 0.5).to(branch_logits.dtype)
        heatmap_positive_mask = branch_masks * positive_clip_slots.reshape(
            positive_clip_slots.shape[0], 1, positive_clip_slots.shape[1]
        )
        heatmap_negative_mask = branch_masks * negative_clip_slots.reshape(
            negative_clip_slots.shape[0], 1, negative_clip_slots.shape[1]
        )
        diagnostics: dict[str, Tensor] = {
            "frame_heatmap_pos_clip_loss": masked_mean(heatmap_matrix, heatmap_positive_mask),
            "frame_heatmap_neg_clip_loss": masked_mean(heatmap_matrix, heatmap_negative_mask),
            "frame_mil_pos_clip_loss": masked_mean(mil_matrix, positive_clip_slots),
            "frame_mil_neg_clip_loss": masked_mean(mil_matrix, negative_clip_slots),
            "frame_heatmap_neg_clip_slots": heatmap_negative_mask.sum().detach(),
        }
        if hard_negative_rows is not None and bool(hard_negative_rows.any()):
            hard_slots = (
                branch_label_masks
                * hard_negative_rows.to(branch_logits.dtype).reshape(-1, 1)
                * (branch_clip_targets <= 0.5).to(branch_logits.dtype)
            )
            hard_heatmap_mask = branch_masks * hard_slots.reshape(
                hard_slots.shape[0], 1, hard_slots.shape[1]
            )
            diagnostics.update(
                {
                    "frame_heatmap_hard_neg_loss": masked_mean(
                        heatmap_matrix, hard_heatmap_mask
                    ),
                    "frame_mil_hard_neg_loss": masked_mean(mil_matrix, hard_slots),
                    "frame_heatmap_hard_neg_slots": hard_heatmap_mask.sum().detach(),
                }
            )

        rank = frame_rank_loss(
            branch_logits,
            branch_targets,
            branch_masks,
            pos_threshold=float(train_cfg.get("frame_rank_pos_threshold", 0.5)),
            neg_threshold=float(train_cfg.get("frame_rank_neg_threshold", 0.05)),
            margin=float(train_cfg.get("frame_rank_margin", 1.0)),
        )
        heatmap_weight = float(train_cfg.get("frame_heatmap_loss_weight", 0.5))
        mil_weight = float(train_cfg.get("frame_mil_loss_weight", 0.3))
        rank_weight = float(train_cfg.get("frame_rank_loss_weight", 0.2))
        total = heatmap_weight * heatmap_loss + mil_weight * mil_loss + rank_weight * rank
        return total, {
            "frame_heatmap_loss": heatmap_loss,
            "frame_mil_loss": mil_loss,
            "frame_rank_loss": rank,
            "frame_loss": total,
            **diagnostics,
            **heatmap_aggregation_diagnostics,
        }

    frame_class_weights = per_label_float_tuple(
        train_cfg.get("frame_loss_class_weight", [1.0] * len(LABELS)),
        default=1.0,
    )
    response_curve_class_weights = per_label_float_tuple(
        train_cfg.get("response_curve_frame_loss_class_weight", frame_class_weights),
        default=1.0,
    )

    total, tensors = branch_loss(
        outputs["frame_event_logits"],
        targets,
        masks,
        label_masks,
        class_weights=frame_class_weights,
    )
    components: dict[str, Tensor] = dict(tensors)
    branch_count = 1
    if "local_frame_event_logits" in outputs:
        local_targets = batch.get("local_frame_targets", targets)
        local_targets = local_targets.to(device, non_blocking=True)
        local_masks = batch.get("local_frame_target_masks", masks)
        local_masks = local_masks.to(device, non_blocking=True)
        local_label_masks = label_masks
        roi_frame_valid = batch.get("roi_frame_valid")
        if torch.is_tensor(roi_frame_valid):
            frame_valid = roi_frame_valid.to(device, non_blocking=True)
            local_masks = local_masks * frame_valid.reshape(frame_valid.shape[0], frame_valid.shape[1], 1)
            local_label_masks = local_label_masks * frame_valid.mean(dim=1, keepdim=True)
        else:
            roi_valid = batch.get("roi_valid")
            if torch.is_tensor(roi_valid):
                valid = roi_valid.to(device, non_blocking=True).reshape(-1, 1)
                local_masks = local_masks * valid.reshape(-1, 1, 1)
                local_label_masks = local_label_masks * valid
        local_total, local_tensors = branch_loss(
            outputs["local_frame_event_logits"],
            local_targets,
            local_masks,
            local_label_masks,
            class_weights=frame_class_weights,
        )
        total = total + local_total
        branch_count += 1
        for key, value in local_tensors.items():
            components[f"local_{key}"] = value
    if "class_evidence_frame_event_logits" in outputs:
        # This branch is derived from full-image patch evidence, not detector
        # ROI inputs.  Applying roi_valid here would silently mask every sample
        # when spatial_crop.mode=none.
        evidence_total, evidence_tensors = branch_loss(
            outputs["class_evidence_frame_event_logits"],
            targets,
            masks,
            label_masks,
            class_weights=frame_class_weights,
        )
        total = total + evidence_total
        branch_count += 1
        for key, value in evidence_tensors.items():
            components[f"class_evidence_{key}"] = value
    curve_weight = float(train_cfg.get("response_curve_frame_loss_weight", 0.0))
    if "response_curve_logits" in outputs and curve_weight > 0:
        curve_total, curve_tensors = branch_loss(
            outputs["response_curve_logits"],
            targets,
            masks,
            label_masks,
            class_weights=response_curve_class_weights,
        )
        total = total + curve_weight * curve_total
        branch_count += curve_weight
        for key, value in curve_tensors.items():
            components[f"response_curve_{key}"] = value
    highres_weight = float(
        train_cfg.get("highres_local_frame_loss_weight", 0.35)
    )
    if "highres_local_frame_event_logits" in outputs and highres_weight > 0:
        pool_targets = batch["highres_pool_targets"].to(device, non_blocking=True)
        pool_masks = batch["highres_pool_target_masks"].to(device, non_blocking=True)
        selected = outputs["highres_selected_pool_indices"].long()
        gather = selected.unsqueeze(-1).expand(-1, -1, pool_targets.shape[-1])
        highres_targets = pool_targets.gather(1, gather)
        highres_masks = pool_masks.gather(1, gather)
        highres_total, highres_tensors = branch_loss(
            outputs["highres_local_frame_event_logits"],
            highres_targets,
            highres_masks,
            label_masks,
        )
        total = total + highres_weight * highres_total
        branch_count += highres_weight
        for key, value in highres_tensors.items():
            components[f"highres_{key}"] = value
    total = total / float(branch_count)
    components["frame_loss"] = total
    return total, {key: float(value.detach().cpu()) for key, value in components.items()}


def roi_quality_supervision_loss(
    outputs: dict[str, Tensor],
    batch: dict[str, Any],
    cfg: Any,
    device: torch.device,
) -> tuple[Tensor, dict[str, float]]:
    required = ("roi_quality_logits", "roi_feature_logits", "global_logits")
    if not all(key in outputs for key in required):
        for value in outputs.values():
            if torch.is_tensor(value):
                return value.new_zeros(()), {}
        return torch.zeros((), device=device), {}
    targets = batch["targets"].to(device, non_blocking=True).to(outputs["global_logits"].dtype)
    label_masks = batch["label_masks"].to(device, non_blocking=True).to(targets.dtype)
    roi_valid = batch.get("roi_valid")
    if torch.is_tensor(roi_valid):
        label_masks = label_masks * (roi_valid.to(device, non_blocking=True) > 0).to(targets.dtype).reshape(-1, 1)
    global_error = F.binary_cross_entropy_with_logits(
        outputs["global_logits"].detach(), targets, reduction="none"
    )
    feature_error = F.binary_cross_entropy_with_logits(
        outputs["roi_feature_logits"].detach(), targets, reduction="none"
    )
    advantage = global_error - feature_error
    temperature = max(float(cfg.train.get("roi_quality_temperature", 0.25)), 1e-4)
    quality_target = torch.sigmoid(advantage / temperature).detach()
    quality_matrix = F.binary_cross_entropy_with_logits(
        outputs["roi_quality_logits"], quality_target, reduction="none"
    )
    quality_loss = masked_mean(quality_matrix, label_masks)
    valid_count = label_masks.sum().clamp_min(1.0)
    return quality_loss, {
        "roi_quality_loss": float(quality_loss.detach().cpu()),
        "roi_quality_target": float(((quality_target * label_masks).sum() / valid_count).detach().cpu()),
        "roi_quality_prob": float(
            ((torch.sigmoid(outputs["roi_quality_logits"]) * label_masks).sum() / valid_count).detach().cpu()
        ),
        "roi_feature_advantage": float(((advantage * label_masks).sum() / valid_count).detach().cpu()),
    }


def per_label_float_tuple(
    raw: Any, *, default: float = 0.0
) -> tuple[float, ...]:
    if isinstance(raw, dict):
        return tuple(float(raw.get(label, default)) for label in LABELS)
    if isinstance(raw, str):
        text = raw.strip()
        if "=" in text:
            values: dict[str, float] = {}
            for item in text.split(","):
                item = item.strip()
                if not item:
                    continue
                if "=" not in item:
                    raise ValueError(
                        f"Expected label=value in per-label float string, got {item!r}"
                    )
                label, value = item.split("=", 1)
                values[label.strip()] = float(value)
            return tuple(float(values.get(label, default)) for label in LABELS)
        value = float(text)
        return tuple(value for _ in LABELS)
    if isinstance(raw, (list, tuple)):
        values = [float(value) for value in raw]
        if len(values) < len(LABELS):
            values.extend([float(default)] * (len(LABELS) - len(values)))
        return tuple(values[: len(LABELS)])
    value = float(raw)
    return tuple(value for _ in LABELS)


def per_label_int_tuple(raw: Any, *, default: int = 0) -> tuple[int, ...]:
    if isinstance(raw, dict):
        return tuple(max(int(raw.get(label, default)), 0) for label in LABELS)
    if isinstance(raw, (list, tuple)):
        values = [max(int(value), 0) for value in raw]
        if len(values) < len(LABELS):
            values.extend([max(int(default), 0)] * (len(LABELS) - len(values)))
        return tuple(values[: len(LABELS)])
    value = max(int(raw), 0)
    return tuple(value for _ in LABELS)


def frame_topk_hit_metrics(outputs: dict[str, Tensor], batch: dict[str, Any], topk: int = 4) -> dict[str, Any]:
    if "frame_event_logits" not in outputs or "frame_targets" not in batch:
        return {}
    logits = outputs["frame_event_logits"].detach().float().cpu()
    targets = batch["frame_targets"].detach().float().cpu()
    masks = batch.get("frame_target_masks")
    masks = masks.detach().float().cpu() if torch.is_tensor(masks) else torch.ones_like(targets)
    k = min(max(int(topk), 1), logits.shape[1])
    hits = torch.zeros(logits.shape[2], dtype=torch.float32)
    totals = torch.zeros(logits.shape[2], dtype=torch.float32)
    topk_indices = torch.topk(logits, k=k, dim=1).indices
    for batch_index in range(logits.shape[0]):
        for label_index in range(logits.shape[2]):
            positives = (targets[batch_index, :, label_index] >= 0.5) & (masks[batch_index, :, label_index] > 0)
            if not bool(positives.any()):
                continue
            totals[label_index] += 1.0
            selected = topk_indices[batch_index, :, label_index]
            if bool(positives[selected].any()):
                hits[label_index] += 1.0
    return {
        label: {
            "hit": int(hits[index].item()),
            "total": int(totals[index].item()),
            "hit_rate": float((hits[index] / totals[index].clamp_min(1.0)).item()),
        }
        for index, label in enumerate(LABELS)
    }


def temporal_gate_statistics(
    gates: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label_index, label in enumerate(LABELS):
        valid = masks[:, label_index] > 0
        positive = valid & (targets[:, label_index] > 0)
        negative = valid & ~positive
        values = gates[valid, label_index]
        if values.size == 0:
            continue
        result[label] = {
            "mean": float(values.mean()),
            "positive_mean": (
                float(gates[positive, label_index].mean())
                if positive.any()
                else None
            ),
            "negative_mean": (
                float(gates[negative, label_index].mean())
                if negative.any()
                else None
            ),
            "p10": float(np.quantile(values, 0.10)),
            "p50": float(np.quantile(values, 0.50)),
            "p90": float(np.quantile(values, 0.90)),
            "event_majority_rate": float((values > 0.5).mean()),
            "valid_count": int(values.size),
        }
    return result


def _merge_time_intervals(intervals: Sequence[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted((float(start), float(end)) for start, end in intervals)
    total = 0.0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += max(current_end - current_start, 0.0)
            current_start, current_end = start, end
    return total + max(current_end - current_start, 0.0)


def online_event_metrics(
    probs: np.ndarray,
    candidate_times: np.ndarray,
    metas: Sequence[dict[str, Any]],
    thresholds: np.ndarray,
    masks: np.ndarray | None = None,
    *,
    nms_radius_sec: float,
    tolerance_sec: float,
    capped_clip_sec: float,
    stride_sec: float = 0.0,
    localization_tolerances_sec: Sequence[float] = (2.0,),
) -> dict[str, Any]:
    """One-to-one event spotting metrics over fixed-stride dense windows."""
    if probs.shape != candidate_times.shape:
        raise ValueError(
            f"online probs/times mismatch: {probs.shape} vs {candidate_times.shape}"
        )
    if masks is not None and masks.shape != probs.shape:
        raise ValueError(
            f"online masks/probs mismatch: {masks.shape} vs {probs.shape}"
        )
    videos = sorted(
        {
            (str(meta.get("source", "")), str(meta["video_id"]))
            for meta in metas
        }
    )
    per_class: dict[str, Any] = {}
    all_tp = all_fp = all_fn = 0
    nms_intervals: dict[tuple[str, str], list[tuple[float, float]]] = {video: [] for video in videos}
    raw_intervals: dict[tuple[str, str], list[tuple[float, float]]] = {video: [] for video in videos}
    video_duration: dict[tuple[str, str], float] = {video: 0.0 for video in videos}
    for row_index, meta in enumerate(metas):
        video = (str(meta.get("source", "")), str(meta["video_id"]))
        start = float(meta.get("sampled_clip_start", meta.get("base_clip_start", 0.0)))
        end = float(meta.get("sampled_clip_end", meta.get("base_clip_end", start)))
        video_duration[video] = max(video_duration[video], end)
        if bool((probs[row_index] >= thresholds).any()):
            raw_intervals[video].append((start, end))
    for label_index, label in enumerate(LABELS):
        tp = fp = fn = predictions_after_nms = 0
        support = 0
        partial_known_tp = partial_known_support = 0
        partial_predictions_after_nms = 0
        video_complete = {
            video: all(
                masks is None or float(masks[index, label_index]) > 0.5
                for index, meta in enumerate(metas)
                if (str(meta.get("source", "")), str(meta["video_id"])) == video
            )
            for video in videos
        }
        peak_errors: list[float] = []
        near_duplicate_fp = 0
        # Anchor-position buckets keyed by anchor mod stride: the two covering
        # grid windows place the anchor at r and r+stride, so bucketing by r
        # isolates the [2,3) blind spot of central/edge position sampling.
        bucket_stats: dict[str, list[int]] = {"0_2": [0, 0], "2_3": [0, 0], "3_5": [0, 0]}
        localization: dict[float, list[int]] = {
            float(tol): [0, 0] for tol in localization_tolerances_sec
        }
        for video in videos:
            row_indices = [
                index for index, meta in enumerate(metas)
                if (
                    str(meta.get("source", "")), str(meta["video_id"])
                ) == video
            ]
            gt_values: set[float] = set()
            candidates: list[tuple[float, float]] = []
            for index in row_indices:
                anchors = metas[index].get("online_gt_anchors", ()) or ()
                if len(anchors) == len(LABELS):
                    gt_values.update(round(float(value), 4) for value in anchors[label_index])
                score = float(probs[index, label_index])
                if score >= float(thresholds[label_index]):
                    candidates.append((score, float(candidate_times[index, label_index])))
            kept: list[tuple[float, float]] = []
            for score, timestamp in sorted(candidates, reverse=True):
                if any(abs(timestamp - existing_time) <= nms_radius_sec for _, existing_time in kept):
                    continue
                kept.append((score, timestamp))
            if not video_complete[video]:
                partial_predictions_after_nms += len(kept)
                unmatched_partial = set(gt_values)
                for _score, timestamp in kept:
                    eligible = [
                        value for value in unmatched_partial
                        if abs(timestamp - value) <= tolerance_sec
                    ]
                    if eligible:
                        unmatched_partial.remove(
                            min(eligible, key=lambda value: abs(timestamp - value))
                        )
                    half = capped_clip_sec * 0.5
                    nms_intervals[video].append(
                        (max(timestamp - half, 0.0), min(timestamp + half, video_duration[video]))
                    )
                partial_known_tp += len(gt_values) - len(unmatched_partial)
                partial_known_support += len(gt_values)
                continue
            predictions_after_nms += len(kept)
            support += len(gt_values)
            unmatched = set(gt_values)
            for _score, timestamp in kept:
                eligible = [value for value in unmatched if abs(timestamp - value) <= tolerance_sec]
                if eligible:
                    matched = min(eligible, key=lambda value: abs(timestamp - value))
                    unmatched.remove(matched)
                    tp += 1
                    peak_errors.append(abs(timestamp - matched))
                else:
                    fp += 1
                    if gt_values and min(
                        abs(timestamp - value) for value in gt_values
                    ) <= 2.0 * tolerance_sec:
                        near_duplicate_fp += 1
                half = capped_clip_sec * 0.5
                nms_intervals[video].append(
                    (max(timestamp - half, 0.0), min(timestamp + half, video_duration[video]))
                )
            fn += len(unmatched)
            if stride_sec > 0:
                matched_anchors = gt_values - unmatched
                for value in gt_values:
                    remainder = float(value) % stride_sec
                    bucket = "0_2" if remainder < 2.0 else ("2_3" if remainder < 3.0 else "3_5")
                    bucket_stats[bucket][1] += 1
                    if value in matched_anchors:
                        bucket_stats[bucket][0] += 1
            for loc_tol in localization:
                if loc_tol >= tolerance_sec:
                    continue
                unmatched_loc = set(gt_values)
                for _score, timestamp in kept:
                    eligible = [
                        value for value in unmatched_loc
                        if abs(timestamp - value) <= loc_tol
                    ]
                    if eligible:
                        unmatched_loc.remove(
                            min(eligible, key=lambda value: abs(timestamp - value))
                        )
                localization[loc_tol][0] += len(gt_values) - len(unmatched_loc)
                localization[loc_tol][1] += len(gt_values)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": support,
            "complete_video_count": int(sum(video_complete.values())),
            "partial_video_count": int(len(videos) - sum(video_complete.values())),
            "partial_video_ids": sorted(
                video[1] for video, complete in video_complete.items() if not complete
            ),
            "known_positive_recall_partial": (
                partial_known_tp / partial_known_support
                if partial_known_support else None
            ),
            "known_positive_tp_partial": partial_known_tp,
            "known_positive_support_partial": partial_known_support,
            "predictions_after_nms_partial": partial_predictions_after_nms,
            "predictions_after_nms": predictions_after_nms,
            "threshold": float(thresholds[label_index]),
            "peak_error_median_sec": (
                float(np.median(peak_errors)) if peak_errors else None
            ),
            "peak_error_p90_sec": (
                float(np.quantile(peak_errors, 0.9)) if peak_errors else None
            ),
            "near_duplicate_fp": near_duplicate_fp,
            "anchor_position_buckets": (
                {
                    name: {
                        "recall": hit / max(total, 1),
                        "tp": hit,
                        "support": total,
                    }
                    for name, (hit, total) in bucket_stats.items()
                }
                if stride_sec > 0
                else None
            ),
            "localization_recall": {
                f"tolerance_{tol:g}s": hit / max(total, 1)
                for tol, (hit, total) in localization.items()
            },
        }
        all_tp += tp
        all_fp += fp
        all_fn += fn
    total_video_sec = sum(video_duration.values())
    raw_participation_sec = sum(_merge_time_intervals(value) for value in raw_intervals.values())
    nms_participation_sec = sum(_merge_time_intervals(value) for value in nms_intervals.values())
    micro_precision = all_tp / max(all_tp + all_fp, 1)
    micro_recall = all_tp / max(all_tp + all_fn, 1)
    return {
        "protocol": "fixed_stride_window_peak_pointnms_one_to_one",
        "nms_radius_sec": float(nms_radius_sec),
        "tolerance_sec": float(tolerance_sec),
        "capped_clip_sec": float(capped_clip_sec),
        "per_class": per_class,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": 2.0 * micro_precision * micro_recall / max(micro_precision + micro_recall, 1e-12),
        "tp": all_tp,
        "fp": all_fp,
        "fn": all_fn,
        "video_count": len(videos),
        "total_video_minutes": total_video_sec / 60.0,
        "raw_trigger_window_union_minutes": raw_participation_sec / 60.0,
        "raw_trigger_participation_ratio": raw_participation_sec / max(total_video_sec, 1e-12),
        "nms_capped_union_minutes": nms_participation_sec / 60.0,
        "nms_capped_participation_ratio": nms_participation_sec / max(total_video_sec, 1e-12),
    }


def online_event_selection_metrics(
    online: dict[str, Any],
    window_metrics: dict[str, Any],
    *,
    fbeta_beta: float,
) -> dict[str, Any]:
    """Expose strict event metrics through the existing selection schema."""
    beta = max(float(fbeta_beta), 1e-6)
    beta_sq = beta * beta
    per_class: dict[str, Any] = {}
    for label in LABELS:
        event = dict(online["per_class"][label])
        precision = float(event["precision"])
        recall = float(event["recall"])
        event["fbeta"] = (
            (1.0 + beta_sq) * precision * recall
            / max(beta_sq * precision + recall, 1e-12)
        )
        # Ranking metrics remain window diagnostics and are not used to tune
        # the event threshold or to claim event-level AP.
        event["ap"] = window_metrics["per_class"][label].get("ap")
        event["auroc"] = window_metrics["per_class"][label].get("auroc")
        per_class[label] = event
    macro = lambda key: float(np.mean([float(per_class[label][key]) for label in LABELS]))
    micro_precision = float(online["micro_precision"])
    micro_recall = float(online["micro_recall"])
    micro_fbeta = (
        (1.0 + beta_sq) * micro_precision * micro_recall
        / max(beta_sq * micro_precision + micro_recall, 1e-12)
    )
    return {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": float(online["micro_f1"]),
        "micro_fbeta": micro_fbeta,
        "macro_precision": macro("precision"),
        "macro_recall": macro("recall"),
        "macro_f1": macro("f1"),
        "macro_fbeta": macro("fbeta"),
        "fbeta_beta": beta,
        "mAP": float(window_metrics.get("mAP", 0.0)),
        "mAUROC": float(window_metrics.get("mAUROC", 0.0)),
        "per_class": per_class,
        "metric_scope": "online_event_pointnms_strict_1to1",
    }


def resolve_evaluation_candidate_time_mode(cfg: Any) -> str:
    """Choose timestamp semantics independently of auxiliary batch fields."""

    eval_cfg = cfg.get("eval", ConfigDict())
    requested = str(eval_cfg.get("candidate_time_mode", "auto")).strip().lower()
    if requested not in {"auto", "window_center", "frame_peak"}:
        raise ValueError(
            "eval.candidate_time_mode must be auto, window_center, or frame_peak"
        )
    if requested != "auto":
        return requested
    train_cfg = cfg.get("train", ConfigDict())
    frame_localizer_trained = any(
        float(train_cfg.get(key, 0.0) or 0.0) > 0.0
        for key in (
            "frame_det_loss_weight",
            "frame_tail_rank_loss_weight",
            "spatial_temporal_localization_loss_weight",
            "structured_frame_temporal_localization_loss_weight",
        )
    )
    # Detection/relation auxiliary targets must never silently switch event
    # timestamp semantics.  An untrained frame head is not a localizer.
    return "frame_peak" if frame_localizer_trained else "window_center"


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    cfg: Any,
    device: torch.device,
    *,
    fixed_thresholds: dict[str, float] | None = None,
    online_cache_path: str | Path | None = None,
) -> dict[str, Any]:
    model.eval()
    candidate_time_mode = resolve_evaluation_candidate_time_mode(cfg)
    # Validation may use a larger frozen-backbone frame batch than training.
    # The eval-only override preserves the train-time memory envelope.
    train_backbone_frame_chunk_size = getattr(model, "backbone_frame_chunk_size", None)
    eval_backbone_frame_chunk_size = cfg.get("eval", ConfigDict()).get(
        "backbone_frame_chunk_size", None
    )
    if train_backbone_frame_chunk_size is not None and eval_backbone_frame_chunk_size is not None:
        model.backbone_frame_chunk_size = int(eval_backbone_frame_chunk_size)
        if not distributed_training_active() or dist.get_rank() == 0:
            print(
                "eval_backbone_frame_chunk_size "
                f"train={train_backbone_frame_chunk_size} eval={model.backbone_frame_chunk_size}",
                flush=True,
            )
    all_logits = []
    all_targets = []
    all_masks = []
    all_candidate_times: list[Tensor] = []
    all_frame_peak_probs: list[Tensor] = []
    all_metas: list[dict[str, Any]] = []
    fusion = canonical_temporal_fusion(
        str(cfg.model.get("temporal_fusion", "cls_transformer"))
    )
    collect_uniform_temporal_branches = fusion == "uniform_event_dual_transformer"
    collect_videomae_temporal_branches = fusion == "videomae_residual_transformer"
    collect_temporal_branches = (
        collect_uniform_temporal_branches
        or collect_videomae_temporal_branches
    )
    collect_spatial_branches = bool(
        cfg.model.get("spatial_attention", ConfigDict()).get("enabled", False)
    )
    spatial_branch_logits: dict[str, list[Tensor]] = {
        "global": [],
        "spatial": [],
        "fused": [],
    }
    temporal_branch_logits: dict[str, list[Tensor]] = (
        {"uniform": [], "event": []}
        if collect_uniform_temporal_branches
        else {
            "global_action": [],
            "temporal_only": [],
            "global_plus_temporal": [],
        }
    )
    temporal_gate_batches: list[Tensor] = []
    spatial_erased_batches: list[Tensor] = []
    subtype_logit_batches: list[Tensor] = []
    subtype_target_batches: list[Tensor] = []
    subtype_mask_batches: list[Tensor] = []
    frame_hit_totals = {label: {"hit": 0, "total": 0} for label in LABELS}
    object_validation_totals: dict[str, float] = {}
    object_validation_samples = 0
    for batch in loader:
        targets = batch["targets"].to(device, non_blocking=True)
        label_masks = batch["label_masks"].to(device, non_blocking=True)
        need_aux = (
            "frame_targets" in batch
            or "object_heatmap_targets" in batch
            or collect_temporal_branches
            or candidate_time_mode == "frame_peak"
            or collect_spatial_branches
            or float(cfg.train.get("set_piece_subtype_loss_weight", 0.0) or 0.0) > 0
        )
        with autocast_context(device, bool(cfg.train.amp), str(cfg.train.amp_dtype)):
            outputs = forward_model_batch(model, batch, device, return_aux=need_aux)
        if isinstance(outputs, dict):
            logits = outputs["logits"]
            if (
                "object_heatmap_targets" in batch
                and "object_heatmap_logits" in outputs
            ):
                _, object_components = object_teacher_heatmap_loss(
                    outputs, batch, cfg, device
                )
                object_batch_size = int(targets.shape[0])
                object_validation_samples += object_batch_size
                for key, value in object_components.items():
                    object_validation_totals[key] = (
                        object_validation_totals.get(key, 0.0)
                        + float(value) * object_batch_size
                    )
                if (
                    "ball_goal_relation_targets" in batch
                    and "ball_goal_relation_logits" in outputs
                ):
                    _, relation_components = ball_goal_relation_aux_loss(
                        outputs, batch, cfg, device
                    )
                    for key, value in relation_components.items():
                        object_validation_totals[key] = (
                            object_validation_totals.get(key, 0.0)
                            + float(value) * object_batch_size
                        )
            if "set_piece_subtype_logits" in outputs:
                subtype_logit_batches.append(
                    outputs["set_piece_subtype_logits"].float().cpu()
                )
                subtype_target_batches.append(
                    batch["set_piece_subtype_targets"].float().cpu()
                )
                subtype_mask_batches.append(
                    batch["set_piece_subtype_mask"].float().cpu()
                )
            if collect_spatial_branches:
                spatial_branch_logits["global"].append(
                    outputs["global_logits"].float().cpu()
                )
                spatial_branch_logits["spatial"].append(
                    outputs["spatial_clip_logits"].float().cpu()
                )
                spatial_branch_logits["fused"].append(
                    outputs["spatial_fused_logits"].float().cpu()
                )
                if "spatial_erased_clip_logits" in outputs:
                    spatial_erased_batches.append(
                        outputs["spatial_erased_clip_logits"].float().cpu()
                    )
            if collect_uniform_temporal_branches:
                temporal_branch_logits["uniform"].append(
                    outputs["uniform_logits"].float().cpu()
                )
                temporal_branch_logits["event"].append(
                    outputs["event_logits"].float().cpu()
                )
                temporal_gate_batches.append(
                    outputs["temporal_gate"].float().cpu()
                )
            elif collect_videomae_temporal_branches:
                temporal_branch_logits["global_action"].append(
                    outputs["global_action_logits"].float().cpu()
                )
                temporal_branch_logits["temporal_only"].append(
                    outputs["temporal_delta_logits"].float().cpu()
                )
                temporal_branch_logits["global_plus_temporal"].append(
                    outputs["logits"].float().cpu()
                )
                temporal_gate_batches.append(
                    outputs["temporal_residual_gate"].float().cpu()
                )
            frame_metric_outputs = outputs
            if "spatial_frame_event_logits" in outputs:
                frame_metric_outputs = {
                    **outputs,
                    "frame_event_logits": outputs[
                        "spatial_frame_event_logits"
                    ],
                }
            batch_hits = frame_topk_hit_metrics(
                frame_metric_outputs,
                batch,
                topk=int(cfg.train.get("frame_eval_topk", cfg.model.get("event_topk", 4))),
            )
            for label, item in batch_hits.items():
                frame_hit_totals[label]["hit"] += int(item["hit"])
                frame_hit_totals[label]["total"] += int(item["total"])
        else:
            logits = outputs
        all_logits.append(logits.float().cpu())
        all_targets.append(targets.float().cpu())
        all_masks.append(label_masks.float().cpu())
        frame_logits = None
        if isinstance(outputs, dict):
            frame_logits = outputs.get(
                "spatial_frame_event_logits", outputs.get("frame_event_logits")
            )
        frame_times = batch.get("frame_times")
        frame_peak_probs: Tensor | None = None
        frame_peak_available = (
            torch.is_tensor(frame_logits)
            and torch.is_tensor(frame_times)
            and frame_logits.ndim == 3
            and frame_times.ndim == 2
            and frame_logits.shape[:2] == frame_times.shape
        )
        if candidate_time_mode == "frame_peak":
            if not frame_peak_available:
                raise RuntimeError(
                    "eval candidate_time_mode=frame_peak requires aligned "
                    "frame_event_logits and frame_times"
                )
            assert isinstance(frame_logits, Tensor)
            assert isinstance(frame_times, Tensor)
            peak_indices = frame_logits.float().argmax(dim=1).cpu()
            candidate_times = torch.gather(frame_times.float().cpu(), 1, peak_indices)
            frame_peak_probs = torch.gather(
                torch.sigmoid(frame_logits.float().cpu()),
                1,
                peak_indices.unsqueeze(1),
            ).squeeze(1)
        else:
            centers = torch.tensor(
                [
                    0.5 * (
                        float(meta.get("sampled_clip_start", meta.get("base_clip_start", 0.0)))
                        + float(meta.get("sampled_clip_end", meta.get("base_clip_end", 0.0)))
                    )
                    for meta in batch["meta"]
                ],
                dtype=torch.float32,
            )
            candidate_times = centers[:, None].repeat(1, len(LABELS))
        if frame_peak_probs is None:
            frame_peak_probs = torch.ones(
                candidate_times.shape, dtype=torch.float32
            )
        all_candidate_times.append(candidate_times)
        all_frame_peak_probs.append(frame_peak_probs)
        all_metas.extend(
            {
                "source": str(meta.get("source", "")),
                "video_id": str(meta.get("video_id", "")),
                "sample_id": str(meta.get("sample_id", "")),
                "base_clip_start": float(meta.get("base_clip_start", 0.0)),
                "base_clip_end": float(meta.get("base_clip_end", 0.0)),
                "sampled_clip_start": float(meta.get("sampled_clip_start", 0.0)),
                "sampled_clip_end": float(meta.get("sampled_clip_end", 0.0)),
                "online_gt_anchors": meta.get("online_gt_anchors", ()),
            }
            for meta in batch["meta"]
        )

    # Backbone work is complete; restore the train-time policy before any rank returns.
    if train_backbone_frame_chunk_size is not None:
        model.backbone_frame_chunk_size = train_backbone_frame_chunk_size

    if distributed_training_active():
        local_payload = {
            "logits": torch.cat(all_logits),
            "targets": torch.cat(all_targets),
            "masks": torch.cat(all_masks),
            "candidate_times": torch.cat(all_candidate_times),
            "frame_peak_probs": torch.cat(all_frame_peak_probs),
            "metas": all_metas,
            "spatial_branch_logits": {
                name: (torch.cat(batches) if batches else None)
                for name, batches in spatial_branch_logits.items()
            },
            "temporal_branch_logits": {
                name: (torch.cat(batches) if batches else None)
                for name, batches in temporal_branch_logits.items()
            },
            "temporal_gates": (
                torch.cat(temporal_gate_batches)
                if temporal_gate_batches else None
            ),
            "spatial_erased": (
                torch.cat(spatial_erased_batches)
                if spatial_erased_batches else None
            ),
            "set_piece_subtype_logits": (
                torch.cat(subtype_logit_batches)
                if subtype_logit_batches else None
            ),
            "set_piece_subtype_targets": (
                torch.cat(subtype_target_batches)
                if subtype_target_batches else None
            ),
            "set_piece_subtype_masks": (
                torch.cat(subtype_mask_batches)
                if subtype_mask_batches else None
            ),
            "frame_hit_totals": frame_hit_totals,
            "object_validation_totals": object_validation_totals,
            "object_validation_samples": object_validation_samples,
        }
        rank_payloads = gather_objects_to_main(local_payload)
        if dist.get_rank() != 0:
            return {}
        if rank_payloads is None:
            raise RuntimeError("rank-0 did not receive distributed eval payloads")
        all_logits = [item["logits"] for item in rank_payloads]
        all_targets = [item["targets"] for item in rank_payloads]
        all_masks = [item["masks"] for item in rank_payloads]
        all_candidate_times = [item["candidate_times"] for item in rank_payloads]
        all_frame_peak_probs = [
            item["frame_peak_probs"] for item in rank_payloads
        ]
        all_metas = [
            meta for item in rank_payloads for meta in item["metas"]
        ]
        spatial_branch_logits = {
            name: [
                item["spatial_branch_logits"][name]
                for item in rank_payloads
                if item["spatial_branch_logits"].get(name) is not None
            ]
            for name in spatial_branch_logits
        }
        temporal_branch_logits = {
            name: [
                item["temporal_branch_logits"][name]
                for item in rank_payloads
                if item["temporal_branch_logits"].get(name) is not None
            ]
            for name in temporal_branch_logits
        }
        temporal_gate_batches = [
            item["temporal_gates"]
            for item in rank_payloads
            if item.get("temporal_gates") is not None
        ]
        spatial_erased_batches = [
            item["spatial_erased"]
            for item in rank_payloads
            if item.get("spatial_erased") is not None
        ]
        subtype_logit_batches = [
            item["set_piece_subtype_logits"]
            for item in rank_payloads
            if item.get("set_piece_subtype_logits") is not None
        ]
        subtype_target_batches = [
            item["set_piece_subtype_targets"]
            for item in rank_payloads
            if item.get("set_piece_subtype_targets") is not None
        ]
        subtype_mask_batches = [
            item["set_piece_subtype_masks"]
            for item in rank_payloads
            if item.get("set_piece_subtype_masks") is not None
        ]
        frame_hit_totals = {
            label: {
                "hit": sum(
                    int(item["frame_hit_totals"][label]["hit"])
                    for item in rank_payloads
                ),
                "total": sum(
                    int(item["frame_hit_totals"][label]["total"])
                    for item in rank_payloads
                ),
            }
            for label in LABELS
        }
        object_validation_totals = sum_float_mappings(
            [item["object_validation_totals"] for item in rank_payloads]
        )
        object_validation_samples = sum(
            int(item["object_validation_samples"]) for item in rank_payloads
        )

    logits_np = torch.cat(all_logits).numpy()
    targets_np = torch.cat(all_targets).numpy().astype(np.int32)
    masks_np = torch.cat(all_masks).numpy().astype(np.float32)
    candidate_times_np = torch.cat(all_candidate_times).numpy().astype(np.float32)
    probs_np = 1.0 / (1.0 + np.exp(-logits_np))
    default_threshold = float(cfg.get("eval", ConfigDict()).get("threshold", 0.5))
    if not 0.0 <= default_threshold <= 1.0:
        raise ValueError(f"eval.threshold must be in [0, 1], got {default_threshold}")
    default_thresholds = np.full(len(LABELS), default_threshold, dtype=np.float32)
    eval_cfg = cfg.get("eval", ConfigDict())
    sample_ids = [str(meta.get("sample_id", "")) for meta in all_metas]
    is_online_val = any(value.startswith("online_val_") for value in sample_ids)
    is_online_external = any(value.startswith("online_eval_") for value in sample_ids)
    online_metric_cfg = (
        eval_cfg.get("online_validation", ConfigDict())
        if is_online_val
        else eval_cfg.get("external_audit", ConfigDict()).get(
            "online_mode", ConfigDict()
        )
    )
    # Fused online event score: the detection score, the NMS score and the
    # candidate timestamp all derive from the same event_score(t) peak, i.e.
    # sigmoid(frame_logit at peak) * sigmoid(clip_logit).  "clip" keeps the
    # legacy protocol where score and timestamp come from different heads.
    online_score_fusion = str(
        online_metric_cfg.get("score_fusion", "clip")
    ).strip().lower()
    if online_score_fusion not in {"clip", "clip_x_frame_peak"}:
        raise ValueError(
            "eval online score_fusion must be clip or clip_x_frame_peak, got "
            f"{online_score_fusion}"
        )
    frame_peak_probs_np = (
        torch.cat(all_frame_peak_probs).numpy().astype(np.float32)
    )
    online_probs_np = probs_np
    if (
        (is_online_val or is_online_external)
        and online_score_fusion == "clip_x_frame_peak"
    ):
        online_probs_np = probs_np * frame_peak_probs_np
    policy_cfg = eval_cfg
    if is_online_val and any(
        key in online_metric_cfg
        for key in (
            "tuned_objective",
            "tuned_objective_by_class",
            "tuned_min_recall",
            "tuned_min_recall_by_class",
        )
    ):
        policy_cfg = online_metric_cfg
    tuned_objective = str(policy_cfg.get("tuned_objective", "precision")).strip().lower()
    tuned_objectives, tuned_min_recalls = resolve_threshold_tuning_policy(
        policy_cfg, LABELS
    )
    tuned_fbeta_beta = float(policy_cfg.get("tuned_fbeta_beta", 1.0) or 1.0)
    online_threshold_diagnostics: dict[str, Any] | None = None
    if fixed_thresholds is None:
        window_tuned_thresholds = tune_thresholds(
            targets_np,
            probs_np,
            masks_np,
            min_recalls=tuned_min_recalls,
            objective=tuned_objective,
            objectives=tuned_objectives,
            fbeta_beta=tuned_fbeta_beta,
        )
        window_threshold_source = "tuned_on_this_split_clip_probability"
    else:
        window_tuned_thresholds = np.asarray(
            [float(fixed_thresholds[label]) for label in LABELS],
            dtype=np.float32,
        )
        window_threshold_source = "fixed_external"
    if fixed_thresholds is None:
        if is_online_val and bool(
            online_metric_cfg.get("tune_event_thresholds", True)
        ):
                tuned_thresholds, online_threshold_diagnostics = (
                    tune_online_event_thresholds(
                        online_probs_np,
                        candidate_times_np,
                        all_metas,
                        LABELS,
                        tuned_objectives,
                        tuned_min_recalls,
                        masks=masks_np,
                        nms_radius_sec=float(
                            online_metric_cfg.get("nms_radius_sec", 5.0)
                        ),
                        tolerance_sec=float(
                            online_metric_cfg.get("tolerance_sec", 5.0)
                        ),
                        fbeta_beta=tuned_fbeta_beta,
                        max_candidates=int(
                            online_metric_cfg.get("max_threshold_candidates", 401)
                        ),
                    )
                )
                threshold_source = "tuned_on_online_val_event"
        else:
            tuned_thresholds = tune_thresholds(
                targets_np,
                probs_np,
                masks_np,
                min_recalls=tuned_min_recalls,
                objective=tuned_objective,
                objectives=tuned_objectives,
                fbeta_beta=tuned_fbeta_beta,
            )
            threshold_source = "tuned_on_this_split"
    else:
        tuned_thresholds = np.asarray(
            [float(fixed_thresholds[label]) for label in LABELS],
            dtype=np.float32,
        )
        if bool(((tuned_thresholds < 0.0) | (tuned_thresholds > 1.0)).any()):
            raise ValueError(f"fixed thresholds must be in [0,1], got {fixed_thresholds}")
        threshold_source = "fixed_external"
    result = {
        "default": safe_metrics(targets_np, probs_np, default_thresholds, masks_np, fbeta_beta=tuned_fbeta_beta),
        "tuned": safe_metrics(targets_np, probs_np, tuned_thresholds, masks_np, fbeta_beta=tuned_fbeta_beta),
        "window_tuned": safe_metrics(
            targets_np,
            probs_np,
            window_tuned_thresholds,
            masks_np,
            fbeta_beta=tuned_fbeta_beta,
        ),
        "confidence_separation": confidence_separation_statistics(targets_np, probs_np, masks_np),
        "per_video_default": per_video_metrics(targets_np, probs_np, masks_np, all_metas, default_thresholds),
        "per_video_tuned": per_video_metrics(targets_np, probs_np, masks_np, all_metas, tuned_thresholds),
        "per_video_window_tuned": per_video_metrics(
            targets_np,
            probs_np,
            masks_np,
            all_metas,
            window_tuned_thresholds,
        ),
        "default_threshold": default_threshold,
        "thresholds": {label: float(tuned_thresholds[i]) for i, label in enumerate(LABELS)},
        "clip_thresholds": {
            label: float(window_tuned_thresholds[i])
            for i, label in enumerate(LABELS)
        },
        "threshold_semantics": (
            "online_event_score" if is_online_val else "clip_probability"
        ),
        "clip_threshold_source": window_threshold_source,
        "tuned_objective": tuned_objective,
        "tuned_objective_by_class": {
            label: tuned_objectives[index] for index, label in enumerate(LABELS)
        },
        "tuned_min_recall_by_class": {
            label: tuned_min_recalls[index] for index, label in enumerate(LABELS)
        },
        "tuned_fbeta_beta": tuned_fbeta_beta,
        "threshold_source": threshold_source,
        "candidate_time_mode": candidate_time_mode,
    }
    if object_validation_samples > 0:
        result["object_localization"] = {
            key: value / object_validation_samples
            for key, value in sorted(object_validation_totals.items())
        }
        result["object_localization"]["samples"] = object_validation_samples
        result["object_localization"]["aggregation"] = "sample_weighted"
        if "ball_goal_relation_loss" in result["object_localization"]:
            result["ball_goal_relation"] = dict(
                result["object_localization"]
            )
    if is_online_val or is_online_external:
        online_cfg = online_metric_cfg
        result["online_event"] = online_event_metrics(
            online_probs_np,
            candidate_times_np,
            all_metas,
            tuned_thresholds,
            masks=masks_np,
            nms_radius_sec=float(online_cfg.get("nms_radius_sec", 5.0)),
            tolerance_sec=float(online_cfg.get("tolerance_sec", 5.0)),
            capped_clip_sec=float(online_cfg.get("capped_clip_sec", 10.0)),
            stride_sec=float(online_cfg.get("window_stride_sec", 0.0) or 0.0),
        )
        result["online_event"]["score_fusion"] = online_score_fusion
        if online_threshold_diagnostics is not None:
            result["online_event"]["threshold_tuning"] = (
                online_threshold_diagnostics
            )
        if is_online_val and bool(
            online_cfg.get("use_for_checkpoint_selection", True)
        ):
            result["tuned"] = online_event_selection_metrics(
                result["online_event"],
                result["window_tuned"],
                fbeta_beta=tuned_fbeta_beta,
            )
            result["selection_metric_scope"] = (
                "online_event_pointnms_strict_1to1"
            )
        if online_cache_path:
            cache_path = Path(online_cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                logits=logits_np,
                probs=probs_np,
                frame_peak_probs=frame_peak_probs_np,
                online_probs=online_probs_np,
                targets=targets_np,
                masks=masks_np,
                candidate_times=candidate_times_np,
                video_ids=np.asarray([meta["video_id"] for meta in all_metas]),
                sample_ids=np.asarray([meta["sample_id"] for meta in all_metas]),
                clip_starts=np.asarray([meta["sampled_clip_start"] for meta in all_metas]),
                clip_ends=np.asarray([meta["sampled_clip_end"] for meta in all_metas]),
                thresholds=tuned_thresholds,
                labels=np.asarray(LABELS),
            )
            metadata_path = cache_path.with_suffix(".meta.json")
            metadata_path.write_text(
                json.dumps(all_metas, ensure_ascii=False), encoding="utf-8"
            )
            result["online_event"]["prediction_cache"] = str(cache_path)
            result["online_event"]["prediction_cache_metadata"] = str(
                metadata_path
            )
    audit_common_floor = eval_cfg.get("audit_common_min_recall")
    if audit_common_floor is not None and fixed_thresholds is None:
        common_floors = np.full(
            len(LABELS), float(audit_common_floor), dtype=np.float32
        )
        common_thresholds = tune_thresholds(
            targets_np,
            probs_np,
            masks_np,
            min_recalls=common_floors,
            objective=tuned_objective,
            objectives=tuned_objectives,
            fbeta_beta=tuned_fbeta_beta,
        )
        result["audit_common_recall_floor"] = {
            "floor": float(audit_common_floor),
            "metrics": safe_metrics(
                targets_np,
                probs_np,
                common_thresholds,
                masks_np,
                fbeta_beta=tuned_fbeta_beta,
            ),
            "thresholds": {
                label: float(common_thresholds[index])
                for index, label in enumerate(LABELS)
            },
        }
    if subtype_logit_batches:
        subtype_logits_np = torch.cat(subtype_logit_batches).numpy()
        subtype_targets_np = torch.cat(subtype_target_batches).numpy().astype(np.int32)
        subtype_masks_np = torch.cat(subtype_mask_batches).numpy() > 0
        valid_targets = subtype_targets_np[subtype_masks_np]
        valid_probs = 1.0 / (1.0 + np.exp(-subtype_logits_np[subtype_masks_np]))
        per_subtype: dict[str, Any] = {}
        aps: list[float] = []
        for index, subtype in enumerate(SET_PIECE_SUBTYPES):
            target = valid_targets[:, index] if valid_targets.size else np.asarray([])
            probability = valid_probs[:, index] if valid_probs.size else np.asarray([])
            ap = (
                float(average_precision_score(target, probability))
                if target.size and np.unique(target).size > 1
                else None
            )
            if ap is not None:
                aps.append(ap)
            per_subtype[subtype] = {
                "ap": ap,
                "support": int(target.sum()) if target.size else 0,
            }
        single = valid_targets.sum(axis=1) == 1 if valid_targets.size else np.asarray([], dtype=bool)
        top1 = (
            float(
                (
                    valid_probs[single].argmax(axis=1)
                    == valid_targets[single].argmax(axis=1)
                ).mean()
            )
            if single.any()
            else None
        )
        result["set_piece_subtype"] = {
            "macro_ap": float(np.mean(aps)) if aps else None,
            "top1": top1,
            "valid_rows": int(subtype_masks_np.sum()),
            "single_label_rows": int(single.sum()) if single.size else 0,
            "per_subtype": per_subtype,
        }
    if collect_spatial_branches:
        result["spatial_branches"] = {}
        for branch_name, branch_batches in spatial_branch_logits.items():
            branch_logits_np = torch.cat(branch_batches).numpy()
            branch_probs_np = 1.0 / (1.0 + np.exp(-branch_logits_np))
            branch_thresholds = tune_thresholds(
                targets_np,
                branch_probs_np,
                masks_np,
                min_recalls=tuned_min_recalls,
                objective=tuned_objective,
                objectives=tuned_objectives,
                fbeta_beta=tuned_fbeta_beta,
            )
            result["spatial_branches"][branch_name] = {
                "default": safe_metrics(
                    targets_np, branch_probs_np, default_thresholds, masks_np, fbeta_beta=tuned_fbeta_beta
                ),
                "tuned": safe_metrics(
                    targets_np, branch_probs_np, branch_thresholds, masks_np, fbeta_beta=tuned_fbeta_beta
                ),
                "thresholds": {
                    label: float(branch_thresholds[index])
                    for index, label in enumerate(LABELS)
                },
                "confidence_separation": confidence_separation_statistics(
                    targets_np, branch_probs_np, masks_np
                ),
            }

        if spatial_erased_batches:
            kept_logits_np = torch.cat(
                spatial_branch_logits["spatial"]
            ).numpy()
            erased_logits_np = torch.cat(spatial_erased_batches).numpy()
            causal_gap_np = kept_logits_np - erased_logits_np
            causal_margin = float(
                cfg.train.get("spatial_counterfactual_margin", 0.5)
            )
            result["spatial_counterfactual_gap"] = {}
            for label_index, label in enumerate(LABELS):
                valid = masks_np[:, label_index] > 0
                positive = valid & (targets_np[:, label_index] > 0)
                negative = valid & ~positive
                positive_gaps = causal_gap_np[positive, label_index]
                negative_gaps = causal_gap_np[negative, label_index]
                result["spatial_counterfactual_gap"][label] = {
                    "positive_mean": (
                        float(positive_gaps.mean())
                        if positive_gaps.size
                        else None
                    ),
                    "negative_mean": (
                        float(negative_gaps.mean())
                        if negative_gaps.size
                        else None
                    ),
                    "positive_margin_success_rate": (
                        float((positive_gaps >= causal_margin).mean())
                        if positive_gaps.size
                        else None
                    ),
                }

    if any(item["total"] > 0 for item in frame_hit_totals.values()):
        result["frame_topk_hit_rate"] = {
            label: {
                "hit": item["hit"],
                "total": item["total"],
                "hit_rate": item["hit"] / max(item["total"], 1),
            }
            for label, item in frame_hit_totals.items()
        }
    if collect_temporal_branches:
        result["temporal_branches"] = {}
        gate_np = torch.cat(temporal_gate_batches).numpy()
        result["temporal_gate_stats"] = temporal_gate_statistics(
            gate_np, targets_np, masks_np
        )
        raw_branch_logits: dict[str, Any] = {}
        for branch_name, branch_batches in temporal_branch_logits.items():
            branch_logits_np = torch.cat(branch_batches).numpy()
            branch_probs_np = 1.0 / (1.0 + np.exp(-branch_logits_np))
            branch_thresholds = tune_thresholds(
                targets_np,
                branch_probs_np,
                masks_np,
                min_recalls=tuned_min_recalls,
                objective=tuned_objective,
                objectives=tuned_objectives,
                fbeta_beta=tuned_fbeta_beta,
            )
            result["temporal_branches"][branch_name] = {
                "default": safe_metrics(
                    targets_np,
                    branch_probs_np,
                    default_thresholds,
                    masks_np,
                    fbeta_beta=tuned_fbeta_beta,
                ),
                "tuned": safe_metrics(
                    targets_np,
                    branch_probs_np,
                    branch_thresholds,
                    masks_np,
                    fbeta_beta=tuned_fbeta_beta,
                ),
                "thresholds": {
                    label: float(branch_thresholds[index])
                    for index, label in enumerate(LABELS)
                },
            }
            raw_branch_logits[branch_name] = branch_logits_np.tolist()
        if bool(cfg.get("eval", ConfigDict()).get("save_predictions", False)):
            result["raw_outputs"] = {
                "logits": logits_np.tolist(),
                "targets": targets_np.tolist(),
                "masks": masks_np.tolist(),
                "meta": all_metas,
                "temporal_branch_logits": raw_branch_logits,
                "temporal_gate": gate_np.tolist(),
            }
    return result


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, (nn.DataParallel, DDP)) else model


def format_metric_summary(name: str, values: dict[str, Any]) -> str:
    return (
        f"{name}_macro_p={values['macro_precision']:.4f} "
        f"{name}_macro_r={values['macro_recall']:.4f} "
        f"{name}_macro_f1={values['macro_f1']:.4f} "
        f"{name}_macro_fbeta={values.get('macro_fbeta', values['macro_f1']):.4f} "
        f"{name}_micro_p={values['micro_precision']:.4f} "
        f"{name}_micro_r={values['micro_recall']:.4f} "
        f"{name}_micro_f1={values['micro_f1']:.4f} "
        f"{name}_micro_fbeta={values.get('micro_fbeta', values['micro_f1']):.4f} "
        f"{name}_mAP={values['mAP']:.4f}"
    )


def print_per_class_metrics(prefix: str, values: dict[str, Any]) -> None:
    for label in LABELS:
        item = values["per_class"][label]
        ap = item["ap"]
        ap_text = "nan" if np.isnan(ap) else f"{ap:.4f}"
        print(
            f"{prefix} class={label} "
            f"p={item['precision']:.4f} r={item['recall']:.4f} f1={item['f1']:.4f} "
            f"fbeta={item.get('fbeta', item['f1']):.4f} "
            f"support={item['support']} ap={ap_text} threshold={item['threshold']:.2f}",
            flush=True,
        )


def print_per_video_metrics(prefix: str, rows: Sequence[dict[str, Any]]) -> None:
    for row in rows:
        for label in LABELS:
            item = row["per_class"][label]
            print(
                f"{prefix} source={row['source']} video_id={row['video_id']} class={label} "
                f"p={item['precision']:.4f} r={item['recall']:.4f} f1={item['f1']:.4f} "
                f"tp/fp/fn={item['tp']}/{item['fp']}/{item['fn']} "
                f"support={item['support']} valid={item['valid_count']} threshold={item['threshold']:.2f}",
                flush=True,
            )


def get_resume_cfg(cfg: Any) -> ConfigDict:
    resume = cfg.train.get("resume", ConfigDict())
    if isinstance(resume, str):
        return ConfigDict({"enabled": bool(resume), "checkpoint": resume})
    if isinstance(resume, dict):
        return to_config(resume)
    return ConfigDict({"enabled": bool(resume)})


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)



class TrainableParameterEMA:
    """EMA only for trainable floating parameters.

    The DINOv3-L backbone is mostly frozen in LoRA experiments.  Tracking the
    entire 300M-parameter state would waste memory and make every update slow;
    tracking only trainable tensors is enough to smooth the learned LoRA,
    temporal stem, transformer, and heads.
    """

    def __init__(self, model: nn.Module, decay: float = 0.995) -> None:
        self.decay = min(max(float(decay), 0.0), 0.99999)
        self.num_updates = 0
        self.shadow: dict[str, Tensor] = {}
        self._refresh_trainable_names(model)

    def _refresh_trainable_names(self, model: nn.Module) -> None:
        module = unwrap_model(model)
        self.trainable_names = {
            name
            for name, parameter in module.named_parameters()
            if parameter.requires_grad and parameter.is_floating_point()
        }
        for name, parameter in module.named_parameters():
            if name in self.trainable_names and name not in self.shadow:
                self.shadow[name] = parameter.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        module = unwrap_model(model)
        self.num_updates += 1
        decay = self.decay
        for name, parameter in module.named_parameters():
            if name not in self.trainable_names or not parameter.is_floating_point():
                continue
            value = parameter.detach()
            if name not in self.shadow:
                self.shadow[name] = value.clone()
            else:
                shadow = self.shadow[name]
                if shadow.device != value.device or shadow.dtype != value.dtype:
                    shadow = shadow.to(device=value.device, dtype=value.dtype)
                    self.shadow[name] = shadow
                shadow.mul_(decay).add_(value, alpha=1.0 - decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> dict[str, Tensor]:
        module = unwrap_model(model)
        backup: dict[str, Tensor] = {}
        for name, parameter in module.named_parameters():
            if name not in self.shadow:
                continue
            backup[name] = parameter.detach().clone()
            parameter.copy_(self.shadow[name].to(device=parameter.device, dtype=parameter.dtype))
        return backup

    @torch.no_grad()
    def restore(self, model: nn.Module, backup: dict[str, Tensor]) -> None:
        module = unwrap_model(model)
        for name, parameter in module.named_parameters():
            if name in backup:
                parameter.copy_(backup[name].to(device=parameter.device, dtype=parameter.dtype))

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": {name: tensor.detach().cpu() for name, tensor in self.shadow.items()},
            "trainable_names": sorted(self.trainable_names),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.decay = min(max(float(state.get("decay", self.decay)), 0.0), 0.99999)
        self.num_updates = int(state.get("num_updates", 0) or 0)
        shadow = state.get("shadow", {})
        if isinstance(shadow, dict):
            self.shadow = {
                str(name): tensor.detach().clone()
                for name, tensor in shadow.items()
                if torch.is_tensor(tensor)
            }
        names = state.get("trainable_names")
        if isinstance(names, (list, tuple)):
            self.trainable_names = {str(name) for name in names}


def reset_optimizer_group_lrs_from_config(optimizer: torch.optim.Optimizer, cfg: Any) -> None:
    gpu_ids_cfg = cfg.get("gpu_ids", [])
    fallback_world_size = len(gpu_ids_cfg) if isinstance(gpu_ids_cfg, (list, tuple)) and gpu_ids_cfg else 1
    world_size = int(cfg.get("runtime_topology", {}).get("world_size", fallback_world_size))
    for group in optimizer.param_groups:
        name = str(group.get("name", ""))
        if name == "global_backbone":
            group["lr"] = float(cfg.train.get("global_backbone_lr", cfg.train.backbone_lr))
        elif name == "local_backbone":
            group["lr"] = float(cfg.train.get("local_backbone_lr", cfg.train.backbone_lr))
        elif name == "temporal" and cfg.train.get("temporal_lr_per_gpu") is not None:
            group["lr"] = float(cfg.train.temporal_lr_per_gpu) * world_size
        elif name == "classifier_head" and cfg.train.get("head_lr_per_gpu") is not None:
            group["lr"] = float(cfg.train.head_lr_per_gpu) * world_size
        elif name == "class_evidence_local":
            group["lr"] = float(
                cfg.train.get("class_evidence_local_lr_per_gpu", cfg.train.lr_per_gpu)
            ) * world_size
        elif name == "class_evidence_temporal":
            group["lr"] = float(
                cfg.train.get("class_evidence_temporal_lr_per_gpu", cfg.train.lr_per_gpu)
            ) * world_size
        elif name == "class_evidence_classifier":
            group["lr"] = float(
                cfg.train.get("class_evidence_head_lr_per_gpu", cfg.train.lr_per_gpu)
            ) * world_size
        else:
            group["lr"] = float(cfg.train.lr)


def checkpoint_best_selection_score(checkpoint: dict[str, Any]) -> float:
    if "best_macro_f1" in checkpoint:
        return float(checkpoint["best_macro_f1"])
    metrics = checkpoint.get("metrics")
    if isinstance(metrics, dict):
        return float(metrics.get("tuned", {}).get("macro_f1", -1.0))
    return -1.0


def recall_floor_map_from_config(cfg: Any) -> dict[str, float]:
    raw_recall_floors = cfg.get("eval", ConfigDict()).get(
        "tuned_min_recall", cfg.train.get("checkpoint_min_recall", 0.60)
    )
    if isinstance(raw_recall_floors, dict):
        return {label: float(raw_recall_floors.get(label, 0.0)) for label in LABELS}
    return {label: float(raw_recall_floors) for label in LABELS}


def tuned_mixed_objective_selection_score(
    metrics: dict[str, Any],
    cfg: Any,
) -> tuple[float, dict[str, Any]]:
    objectives, recall_floors = resolve_threshold_tuning_policy(
        cfg.get("eval", ConfigDict()), LABELS
    )
    per_class = metrics["tuned"]["per_class"]
    raw_scores: list[float] = []
    adjusted_scores: list[float] = []
    shortfalls: list[float] = []
    for index, label in enumerate(LABELS):
        objective = objectives[index]
        item = per_class[label]
        raw_score = float(item[objective])
        recall_floor = recall_floors[index]
        shortfall = (
            0.0
            if recall_floor is None
            else max(0.0, recall_floor - float(item["recall"]))
        )
        adjusted_score = (
            raw_score if shortfall <= 1e-12 else raw_score - 1.0 - shortfall
        )
        raw_scores.append(raw_score)
        adjusted_scores.append(adjusted_score)
        shortfalls.append(shortfall)
    score = sum(adjusted_scores) / max(len(adjusted_scores), 1)
    return score, {
        "objectives": {
            label: objectives[index] for index, label in enumerate(LABELS)
        },
        "recall_floors": {
            label: recall_floors[index] for index, label in enumerate(LABELS)
        },
        "raw_scores": {
            label: raw_scores[index] for index, label in enumerate(LABELS)
        },
        "adjusted_scores": {
            label: adjusted_scores[index] for index, label in enumerate(LABELS)
        },
        "recall_shortfalls": {
            label: shortfalls[index] for index, label in enumerate(LABELS)
        },
    }


def online_recall_floor_min_review_selection_score(
    metrics: dict[str, Any],
    cfg: Any,
) -> tuple[float, dict[str, Any]]:
    """Select dense checkpoints by recall feasibility, then review union."""
    online = metrics.get("online_event")
    if not isinstance(online, dict):
        raise ValueError(
            "online_recall_floor_min_review requires metrics.online_event"
        )
    online_cfg = (
        cfg.get("eval", ConfigDict()).get("online_validation", ConfigDict())
    )
    _, recall_floors = resolve_threshold_tuning_policy(online_cfg, LABELS)
    recalls: list[float] = []
    shortfalls: list[float] = []
    precisions: list[float] = []
    for index, label in enumerate(LABELS):
        item = online["per_class"][label]
        recall = float(item["recall"])
        floor = recall_floors[index]
        recalls.append(recall)
        precisions.append(float(item["precision"]))
        shortfalls.append(
            0.0 if floor is None else max(0.0, float(floor) - recall)
        )
    participation = float(online["raw_trigger_participation_ratio"])
    macro_precision = sum(precisions) / max(len(precisions), 1)
    feasible = max(shortfalls, default=0.0) <= 1e-12
    if feasible:
        # Review union is the primary objective; precision only breaks ties.
        score = 1.0 - participation + 1e-3 * macro_precision
    else:
        # Keep infeasible scores below every feasible checkpoint while still
        # ensuring epoch 1 is saveable from the historical -1 initializer.
        normalized_shortfall = min(sum(shortfalls), 1.0)
        score = -0.1 * participation - 0.9 * normalized_shortfall
    return score, {
        "feasible": feasible,
        "recall_floors": {
            label: recall_floors[index] for index, label in enumerate(LABELS)
        },
        "recalls": {label: recalls[index] for index, label in enumerate(LABELS)},
        "recall_shortfalls": {
            label: shortfalls[index] for index, label in enumerate(LABELS)
        },
        "raw_trigger_participation_ratio": participation,
        "raw_trigger_window_union_minutes": float(
            online["raw_trigger_window_union_minutes"]
        ),
        "macro_precision_tiebreak": macro_precision,
        "selection_priority": "recall_feasible_then_min_review_union",
    }


def stable_tuned_macro_precision_selection_score(
    metrics: dict[str, Any],
    cfg: Any,
) -> tuple[float, dict[str, Any]]:
    recall_floors = recall_floor_map_from_config(cfg)
    shortfalls = [
        max(
            0.0,
            recall_floors[label]
            - float(metrics["tuned"]["per_class"][label]["recall"]),
        )
        for label in LABELS
    ]
    precision = float(metrics["tuned"]["macro_precision"])
    base_score = (
        precision
        if max(shortfalls, default=0.0) <= 1e-12
        else precision - 1.0 - sum(shortfalls) / max(len(shortfalls), 1)
    )

    threshold_floors = per_label_float_tuple(
        cfg.train.get("checkpoint_min_threshold", 0.0), default=0.0
    )
    threshold_penalty_weight = float(
        cfg.train.get("checkpoint_threshold_penalty_weight", 0.0) or 0.0
    )
    threshold_shortfalls = []
    for index, label in enumerate(LABELS):
        threshold = float(metrics["thresholds"].get(label, 0.0))
        threshold_shortfalls.append(max(0.0, threshold_floors[index] - threshold))

    positive_p10_floors = per_label_float_tuple(
        cfg.train.get("checkpoint_min_positive_p10", 0.0), default=0.0
    )
    positive_p10_penalty_weight = float(
        cfg.train.get("checkpoint_positive_p10_penalty_weight", 0.0) or 0.0
    )
    positive_p10_shortfalls = []
    for index, label in enumerate(LABELS):
        stats = metrics.get("confidence_separation", {}).get(label, {})
        value = stats.get("positive_p10")
        positive_p10 = 0.0 if value is None else float(value)
        positive_p10_shortfalls.append(max(0.0, positive_p10_floors[index] - positive_p10))

    threshold_penalty = threshold_penalty_weight * sum(threshold_shortfalls) / max(len(threshold_shortfalls), 1)
    positive_p10_penalty = positive_p10_penalty_weight * sum(positive_p10_shortfalls) / max(len(positive_p10_shortfalls), 1)
    score = base_score - threshold_penalty - positive_p10_penalty
    return score, {
        "base_score": base_score,
        "recall_shortfalls": {label: shortfalls[index] for index, label in enumerate(LABELS)},
        "threshold_floors": {label: threshold_floors[index] for index, label in enumerate(LABELS)},
        "threshold_shortfalls": {label: threshold_shortfalls[index] for index, label in enumerate(LABELS)},
        "threshold_penalty": threshold_penalty,
        "positive_p10_floors": {label: positive_p10_floors[index] for index, label in enumerate(LABELS)},
        "positive_p10_shortfalls": {label: positive_p10_shortfalls[index] for index, label in enumerate(LABELS)},
        "positive_p10_penalty": positive_p10_penalty,
    }


def training_checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    metrics: dict[str, Any],
    cfg: Any,
    best_macro_f1: float,
    *,
    ema_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_macro_f1": best_macro_f1,
        "label_schema": LABEL_SCHEMA,
        "labels": LABELS,
        "thresholds": metrics["thresholds"],
        "clip_thresholds": metrics.get("clip_thresholds", metrics["thresholds"]),
        "threshold_semantics": metrics.get(
            "threshold_semantics", "clip_probability"
        ),
        "metrics": metrics,
        "config": to_plain(cfg),
    }
    if ema_state is not None:
        payload["model_ema_trainable"] = ema_state
    return payload


def maybe_resume_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    cfg: Any,
    device: torch.device,
    ema: TrainableParameterEMA | None = None,
) -> tuple[int, float]:
    resume_cfg = get_resume_cfg(cfg)
    if not bool(resume_cfg.get("enabled", False)):
        return 1, -1.0

    checkpoint_path = str(resume_cfg.get("checkpoint", "") or "")
    if not checkpoint_path:
        checkpoint_path = str(Path(cfg.output_dir) / "last.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Unsupported resume checkpoint: {checkpoint_path}")

    strict = bool(resume_cfg.get("strict", True))
    missing, unexpected = unwrap_model(model).load_state_dict(strip_module_prefix(checkpoint["model"]), strict=strict)
    if missing or unexpected:
        print(f"WARN resume model load missing={missing} unexpected={unexpected}", flush=True)

    if bool(resume_cfg.get("load_optimizer", True)) and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        move_optimizer_state_to_device(optimizer, device)
        if bool(resume_cfg.get("reset_optimizer_lr", False)):
            reset_optimizer_group_lrs_from_config(optimizer, cfg)
            print(
                "Reset resumed optimizer LR from current config: "
                f"{optimizer_group_summary(optimizer)}",
                flush=True,
            )
    reset_scheduler = bool(resume_cfg.get("reset_scheduler", False))
    if reset_scheduler:
        print("Reset resumed scheduler from current config; warmup/cosine step count restarts", flush=True)
    elif bool(resume_cfg.get("load_scheduler", True)) and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if bool(resume_cfg.get("load_scaler", True)) and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    if ema is not None and bool(resume_cfg.get("load_ema", True)):
        ema_state = checkpoint.get("model_ema_trainable")
        if isinstance(ema_state, dict):
            ema.load_state_dict(ema_state)
            print(
                f"Loaded trainable EMA state: tensors={len(ema.shadow)} updates={ema.num_updates}",
                flush=True,
            )
        else:
            print("WARN resume requested EMA but checkpoint has no model_ema_trainable; EMA starts from resumed model", flush=True)

    loaded_lrs = [float(group.get("lr", 0.0)) for group in optimizer.param_groups]
    expected_positive_lr = max(
        float(cfg.train.get("lr", 0.0) or 0.0),
        float(cfg.train.get("backbone_lr", 0.0) or 0.0),
    )
    if loaded_lrs and expected_positive_lr > 0.0:
        max_loaded_lr = max(loaded_lrs)
        if (
            max_loaded_lr <= 0.0
            and not bool(resume_cfg.get("allow_zero_lr", False))
        ):
            raise RuntimeError(
                "Resume loaded optimizer param groups with lr=0 while current config "
                f"expects positive lr/backbone_lr. loaded_lrs={loaded_lrs} "
                f"configured_lr={float(cfg.train.get('lr', 0.0) or 0.0)} "
                f"configured_backbone_lr={float(cfg.train.get('backbone_lr', 0.0) or 0.0)}. "
                "This usually happens when resuming from a checkpoint whose scheduler "
                "already reached the end of its cosine schedule. Start a new run with "
                "train.resume.enabled=false, or set train.resume.load_optimizer=false "
                "and train.resume.load_scheduler=false."
            )
        min_lr_fraction = float(
            resume_cfg.get("min_lr_fraction_of_config", 0.0) or 0.0
        )
        if (
            min_lr_fraction > 0.0
            and max_loaded_lr < expected_positive_lr * min_lr_fraction
            and not bool(resume_cfg.get("allow_low_lr", False))
        ):
            raise RuntimeError(
                "Resume loaded a very low optimizer LR; this is likely an end-of-schedule "
                "checkpoint that would silently waste epochs. "
                f"max_loaded_lr={max_loaded_lr:.8g} expected_lr={expected_positive_lr:.8g} "
                f"min_fraction={min_lr_fraction:.4g}. Set "
                "train.resume.reset_optimizer_lr=true and train.resume.reset_scheduler=true "
                "for a fresh fine-tune schedule, or allow_low_lr=true if this is intended."
            )

    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_macro_f1 = checkpoint_best_selection_score(checkpoint)
    print(
        f"Resumed training from {checkpoint_path}: start_epoch={start_epoch} best_macro_f1={best_macro_f1:.4f}",
        flush=True,
    )
    return start_epoch, best_macro_f1


def _curriculum_stage_bounds(stage: Any) -> tuple[int, int]:
    start = int(stage.get("start_epoch", stage.get("start", stage.get("epoch_start", 1))) or 1)
    end = int(stage.get("end_epoch", stage.get("end", stage.get("epoch_end", start))) or start)
    if start <= 0 or end < start:
        raise ValueError(
            f"Invalid train.curriculum_stages bounds: start_epoch={start} end_epoch={end}"
        )
    return start, end


def _curriculum_stage_overrides(stage: Any) -> dict[str, Any]:
    reserved = {"name", "start", "end", "start_epoch", "end_epoch", "epoch_start", "epoch_end", "overrides"}
    overrides: dict[str, Any] = {}
    nested = stage.get("overrides", ConfigDict())
    if isinstance(nested, dict):
        overrides.update(to_plain(nested))
    for key, value in stage.items():
        if key not in reserved:
            overrides[key] = to_plain(value)
    return overrides


def apply_curriculum_stage_for_epoch(
    cfg: Any,
    epoch: int,
    base_train_cfg: dict[str, Any],
) -> dict[str, Any] | None:
    stages = cfg.train.get("curriculum_stages", [])
    if not stages:
        return None
    if not isinstance(stages, list):
        raise ValueError("train.curriculum_stages must be a list of stage dicts")

    override_keys: set[str] = set()
    selected: Any | None = None
    selected_bounds: tuple[int, int] | None = None
    for raw_stage in stages:
        if not isinstance(raw_stage, dict):
            raise ValueError("Each train.curriculum_stages item must be a dict")
        stage = to_config(raw_stage)
        start, end = _curriculum_stage_bounds(stage)
        overrides = _curriculum_stage_overrides(stage)
        override_keys.update(overrides.keys())
        if start <= epoch <= end:
            selected = stage
            selected_bounds = (start, end)

    for key in sorted(override_keys):
        if key in base_train_cfg:
            cfg.train[key] = copy.deepcopy(base_train_cfg[key])
        else:
            cfg.train.pop(key, None)

    if selected is None:
        print(
            f"epoch={epoch} curriculum_stage=base active=false",
            flush=True,
        )
        return None

    overrides = _curriculum_stage_overrides(selected)
    for key, value in overrides.items():
        cfg.train[key] = to_config(copy.deepcopy(value))

    start, end = selected_bounds or _curriculum_stage_bounds(selected)
    stage_info = {
        "name": str(selected.get("name", f"stage_{start}_{end}")),
        "start_epoch": start,
        "end_epoch": end,
        "overrides": overrides,
    }
    print(
        "epoch={epoch} curriculum_stage={name} active=true range={start}-{end} overrides={overrides}".format(
            epoch=epoch,
            name=stage_info["name"],
            start=start,
            end=end,
            overrides=json.dumps(overrides, ensure_ascii=False, sort_keys=True),
        ),
        flush=True,
    )
    return stage_info


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Any,
    device: torch.device,
    train_records: list[Record],
    external_audit_loader: DataLoader | None = None,
) -> None:
    distributed = distributed_training_active()
    rank = dist.get_rank() if distributed else 0
    is_main_process = rank == 0
    output_dir = Path(cfg.output_dir)
    if is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    if distributed:
        dist.barrier()
    pos_weight, pos_weight_mode = resolve_pos_weight(train_records, cfg.train)
    pos_weight = pos_weight.to(device)
    clip_loss_type = str(cfg.train.get("clip_loss_type", "bce")).strip().lower()
    clip_focal_gamma = float(cfg.train.get("clip_focal_gamma", 2.0) or 2.0)
    clip_focal_alpha = cfg.train.get("clip_focal_alpha", None)
    clip_focal_alpha_tensor = None
    if clip_focal_alpha is not None:
        clip_focal_alpha_tensor = torch.tensor(
            per_label_float_tuple(clip_focal_alpha, default=0.5),
            dtype=torch.float32,
            device=device,
        )
        if bool((clip_focal_alpha_tensor <= 0).any() or (clip_focal_alpha_tensor >= 1).any()):
            raise ValueError("train.clip_focal_alpha values must be in (0, 1)")

    def criterion(logits: Tensor, targets: Tensor) -> Tensor:
        if clip_loss_type == "bce":
            return F.binary_cross_entropy_with_logits(
                logits,
                targets,
                pos_weight=pos_weight.to(device=logits.device, dtype=logits.dtype),
                reduction="none",
            )
        if clip_loss_type == "focal_bce":
            alpha = None
            if clip_focal_alpha_tensor is not None:
                alpha = clip_focal_alpha_tensor.to(device=logits.device, dtype=logits.dtype)
            return focal_bce_with_logits(
                logits,
                targets,
                gamma=clip_focal_gamma,
                pos_weight=pos_weight.to(device=logits.device, dtype=logits.dtype),
                alpha=alpha,
            )
        raise ValueError("train.clip_loss_type must be bce or focal_bce")

    optimizer = build_optimizer(model, cfg)
    grad_accum_steps = max(int(cfg.train.grad_accum_steps), 1)
    configured_max_steps = int(cfg.train.get("max_steps_per_epoch", 0) or 0)
    scheduled_micro_steps = (
        min(len(train_loader), configured_max_steps)
        if configured_max_steps > 0 else len(train_loader)
    )
    optimizer_steps_per_epoch = max(
        math.ceil(max(scheduled_micro_steps, 1) / grad_accum_steps), 1
    )
    scheduler = build_scheduler(
        optimizer, cfg, steps_per_epoch=optimizer_steps_per_epoch
    )
    if is_main_process:
        warmup_cfg = cfg.train.get("warmup", ConfigDict())
        warmup_ratio = float(
            warmup_cfg.get("ratio", cfg.train.get("warmup_ratio", 0.0)) or 0.0
        )
        scheduler_epochs = max(
            int(cfg.train.get("scheduler_epochs", cfg.train.epochs)), 1
        )
        total_optimizer_steps = scheduler_epochs * optimizer_steps_per_epoch
        configured_warmup_steps = int(warmup_cfg.get("steps", 0) or 0)
        effective_warmup_steps = (
            configured_warmup_steps
            if configured_warmup_steps > 0
            else int(round(total_optimizer_steps * warmup_ratio))
        )
        print(
            "scheduler_plan "
            f"micro_steps_per_epoch={scheduled_micro_steps} "
            f"grad_accum={grad_accum_steps} "
            f"optimizer_steps_per_epoch={optimizer_steps_per_epoch} "
            f"total_optimizer_steps={total_optimizer_steps} "
            f"warmup_steps={effective_warmup_steps}",
            flush=True,
        )
    amp_enabled = bool(cfg.train.amp) and device.type == "cuda"
    amp_dtype = str(cfg.train.amp_dtype)
    scaler_enabled = amp_enabled and amp_dtype == "fp16"
    scaler = torch.amp.GradScaler(device.type, enabled=scaler_enabled)
    ema_cfg = cfg.train.get("ema", ConfigDict())
    ema: TrainableParameterEMA | None = None
    if bool(ema_cfg.get("enabled", False)):
        # Distributed validation evaluates EMA shards on every rank, so each
        # rank keeps the same trainable-parameter EMA (DDP updates are synced).
        ema = TrainableParameterEMA(
            model, decay=float(ema_cfg.get("decay", 0.995) or 0.995)
        )
        ema_param_count = sum(tensor.numel() for tensor in ema.shadow.values())
        print(
            f"Trainable EMA enabled decay={ema.decay:.6f} "
            f"tensors={len(ema.shadow)} params={ema_param_count}",
            flush=True,
        )
    if is_main_process:
        print(
            f"AMP enabled={amp_enabled} dtype={amp_dtype} grad_scaler={scaler_enabled} "
            f"gradient_checkpointing={bool(cfg.model.get('gradient_checkpointing', False))}",
            flush=True,
        )

    start_epoch, best_macro_f1 = maybe_resume_training(model, optimizer, scheduler, scaler, cfg, device, ema)
    retention_teacher: TrainableParameterEMA | None = None
    retention_teacher_mode = str(
        cfg.train.get("ema_positive_retention_teacher", "ema") or "ema"
    ).strip().lower()
    if retention_teacher_mode not in {"ema", "fixed_init"}:
        raise ValueError(
            "train.ema_positive_retention_teacher must be ema or fixed_init"
        )
    configured_ema_retention_weight = float(
        cfg.train.get("ema_positive_retention_weight", 0.0) or 0.0
    )
    configured_fixed_teacher_guard_weight = float(
        cfg.train.get("fixed_teacher_logit_guard_weight", 0.0) or 0.0
    )
    if configured_fixed_teacher_guard_weight > 0 and retention_teacher_mode != "fixed_init":
        raise ValueError(
            "train.fixed_teacher_logit_guard_weight requires "
            "train.ema_positive_retention_teacher=fixed_init"
        )
    if configured_ema_retention_weight > 0 or configured_fixed_teacher_guard_weight > 0:
        if ema is None:
            raise ValueError(
                "fixed/EMA teacher losses require train.ema.enabled=true"
            )
        if retention_teacher_mode == "fixed_init":
            # Snapshot only trainable tensors after checkpoint loading. This
            # anchors retention without duplicating the frozen DINO backbone.
            retention_teacher = TrainableParameterEMA(model, decay=0.0)
            teacher_param_count = sum(
                tensor.numel() for tensor in retention_teacher.shadow.values()
            )
            print(
                "Positive-retention teacher=fixed_init "
                f"tensors={len(retention_teacher.shadow)} params={teacher_param_count}",
                flush=True,
            )
        else:
            retention_teacher = ema
    early_cfg = cfg.train.get("early_stopping", ConfigDict())
    early_stopping_enabled = bool(early_cfg.get("enabled", False))
    early_stopping_monitor = str(
        early_cfg.get("monitor", "tuned_mAP")
    ).strip()
    if early_stopping_monitor not in {"tuned_mAP", "selection_score"}:
        raise ValueError(
            "train.early_stopping.monitor must be tuned_mAP or selection_score"
        )
    early_stopping_min_epoch = max(int(early_cfg.get("min_epoch", 1)), 1)
    early_stopping_patience = max(int(early_cfg.get("patience", 1)), 1)
    early_stopping_min_delta = max(float(early_cfg.get("min_delta", 0.0)), 0.0)
    early_stopping_best = -float("inf")
    early_stopping_bad_epochs = 0
    hard_negative_loss_weight = 1.0
    if data_mode(cfg) == "long_videos":
        hard_negative_loss_weight = float(
            cfg.data.long_video.get("hard_negative", ConfigDict()).get("loss_weight", 1.0)
        )
    if hard_negative_loss_weight <= 0:
        raise ValueError("data.long_video.hard_negative.loss_weight must be positive")
    if is_main_process:
        print(
            f"optimizer_state_after_resume start_epoch={start_epoch} "
            f"{optimizer_group_summary(optimizer)}",
            flush=True,
        )
        evidence_head = getattr(unwrap_model(model), "class_evidence_head", None)
        gate_weight = getattr(evidence_head, "no_evidence_gate_weight", None)
        gate_bias = getattr(evidence_head, "no_evidence_gate_bias", None)
        if gate_weight is not None and gate_bias is not None:
            gate_parameter_count = gate_weight.numel() + gate_bias.numel()
            print(
                "no_evidence_gate "
                f"type=monotone_saliency params={gate_parameter_count}",
                flush=True,
            )
    if is_main_process:
        save_json(
            output_dir / "label_info.json",
            {
                "label_schema": LABEL_SCHEMA,
                "labels": LABELS,
                "pos_weight": pos_weight.detach().cpu().tolist(),
                "pos_weight_mode": pos_weight_mode,
                "clip_loss_type": clip_loss_type,
                "clip_focal_gamma": clip_focal_gamma,
                "clip_focal_alpha": (
                    None
                    if clip_focal_alpha_tensor is None
                    else clip_focal_alpha_tensor.detach().cpu().tolist()
                ),
            },
        )
        with (output_dir / "config.yaml").open("w") as f:
            yaml.safe_dump(to_plain(cfg), f, sort_keys=False)

    highres_freeze_primary_epochs = max(
        int(cfg.train.get("highres_freeze_primary_epochs", 0)), 0
    )
    primary_model = unwrap_model(model)
    highres_primary_params: list[tuple[nn.Parameter, bool]] = []
    if getattr(primary_model, "highres_glimpse_enabled", False):
        for module in (primary_model.temporal, primary_model.head):
            highres_primary_params.extend(
                (parameter, parameter.requires_grad)
                for parameter in module.parameters()
            )

    base_train_cfg = copy.deepcopy(to_plain(cfg.train))
    online_object_teacher = online_object_teacher_from_config(cfg, device)
    if online_object_teacher is not None:
        print(
            f"online_object_teacher enabled device={device} fill=missing_frames",
            flush=True,
        )
    mechanism_a_cfg = cfg.model.get("mechanism_a", ConfigDict())
    mechanism_a_enabled = bool(mechanism_a_cfg.get("enabled", False))
    mechanism_a_shared_parameters = (
        shared_lora_named_parameters(model) if mechanism_a_enabled else []
    )
    if mechanism_a_enabled:
        shared_parameter_count = sum(
            parameter.numel() for _, parameter in mechanism_a_shared_parameters
        )
        print(
            "mechanism_a shared_route=backbone_lora "
            f"tensors={len(mechanism_a_shared_parameters)} params={shared_parameter_count}",
            flush=True,
        )
    evidence_health_cfg = cfg.train.get(
        "external_evidence_health_assertions", ConfigDict()
    )
    evidence_health_enabled = bool(evidence_health_cfg.get("enabled", False))
    evidence_health_checked = False
    evidence_health_max_gate_grad = 0.0
    evidence_health_max_projection_grad = 0.0
    evidence_health_max_valid_fraction = 0.0
    hard_positive_tail_floors: dict[int, float] = {}
    hard_positive_tail_margins = per_label_float_tuple(
        cfg.train.get(
            "hard_positive_tail_margin",
            cfg.train.get("clean_positive_logit_margin", [0.0] * len(LABELS)),
        ),
        default=0.25,
    )
    for hard_pos_label in cfg.train.get("hard_positive_tail_margin_labels", ["shot", "save"]):
        if hard_pos_label in LABELS:
            hard_pos_index = LABELS.index(hard_pos_label)
            hard_positive_tail_floors[hard_pos_index] = float(
                hard_positive_tail_margins[hard_pos_index]
            )
    for epoch in range(start_epoch, int(cfg.train.epochs) + 1):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        curriculum_stage_info = apply_curriculum_stage_for_epoch(cfg, epoch, base_train_cfg)
        freeze_highres_primary = epoch <= highres_freeze_primary_epochs
        for parameter, originally_trainable in highres_primary_params:
            parameter.requires_grad = originally_trainable and not freeze_highres_primary
        if highres_primary_params:
            print(
                f"epoch={epoch} highres_primary_frozen={freeze_highres_primary}",
                flush=True,
            )
        model.train()
        running_loss = 0.0
        loss_component_totals: dict[str, float] = {}
        recent_loss = 0.0
        recent_loss_component_totals: dict[str, float] = {}
        recent_batches = 0
        recent_data_time_total = 0.0
        recent_step_time_total = 0.0
        hard_negative_loss_total = 0.0
        hard_negative_label_slots = 0.0
        hard_negative_samples = 0
        observed_samples = 0
        num_batches = 0
        data_time_total = 0.0
        step_time_total = 0.0
        mechanism_a_gradient_measurements = 0
        mechanism_a_gradient_records: list[dict[str, float | int]] = []
        # Per-optimizer-step group grad norms, refreshed on optimizer steps and
        # printed with the next log line (E1.3 plan 5.2 monitoring).
        latest_grad_norms: dict[str, float] = {}
        start_time = time.time()
        iter_end = start_time
        optimizer.zero_grad(set_to_none=True)
        max_steps_per_epoch = int(cfg.train.get("max_steps_per_epoch", 0) or 0)
        final_micro_step = (
            min(len(train_loader), max_steps_per_epoch)
            if max_steps_per_epoch > 0 else len(train_loader)
        )
        for step, batch in enumerate(train_loader, start=1):
            # DDP would otherwise all-reduce every micro-batch even though the
            # optimizer only updates every grad_accum_steps.  Synchronize the
            # accumulated gradients only on an update boundary (or the final
            # partial boundary), matching DDP.no_sync semantics.
            if isinstance(model, DDP):
                model.require_backward_grad_sync = (
                    step % grad_accum_steps == 0 or step == final_micro_step
                )
            data_time = time.time() - iter_end
            online_object_teacher_components: dict[str, float] = {}
            if online_object_teacher is not None:
                online_object_teacher_components = (
                    online_object_teacher.fill_missing(batch)
                )
            targets = batch["targets"].to(device, non_blocking=True)
            label_masks = batch["label_masks"].to(device, non_blocking=True)
            targets, label_masks, label_semantics_components = (
                apply_training_label_semantics(
                    targets, label_masks, batch, cfg.train
                )
            )
            online_negative_masks = batch.get("online_negative_masks")
            if online_negative_masks is None:
                online_negative_masks = label_masks
            else:
                online_negative_masks = online_negative_masks.to(
                    device, non_blocking=True
                )
            hard_flags = [
                str(meta.get("sample_id", "")).startswith("hard_neg_")
                for meta in batch["meta"]
            ]
            hard_count = sum(hard_flags)
            hard_rows_bool = torch.tensor(
                hard_flags, dtype=torch.bool, device=device
            )
            sample_loss_weights = torch.ones(
                (targets.shape[0], 1), dtype=label_masks.dtype, device=device
            )
            if hard_count and hard_negative_loss_weight != 1.0:
                sample_loss_weights[hard_rows_bool] = hard_negative_loss_weight
            weighted_label_masks = label_masks * sample_loss_weights
            batch["sample_loss_weights"] = sample_loss_weights
            ema_retention_weight = float(
                cfg.train.get("ema_positive_retention_weight", 0.0) or 0.0
            )
            fixed_teacher_guard_weight = float(
                cfg.train.get("fixed_teacher_logit_guard_weight", 0.0) or 0.0
            )
            teacher_logits = None
            if (ema_retention_weight > 0 or fixed_teacher_guard_weight > 0) and retention_teacher is not None:
                # Build teacher logits before the student graph. Parameter
                # swaps after student forward can invalidate autograd.
                module = unwrap_model(model)
                teacher_backup = retention_teacher.copy_to(module)
                try:
                    with torch.no_grad(), autocast_context(
                        device, bool(cfg.train.amp), str(cfg.train.amp_dtype)
                    ):
                        teacher_outputs = forward_model_batch(
                            model, batch, device, return_aux=False
                        )
                    if isinstance(teacher_outputs, dict):
                        teacher_logits = teacher_outputs["logits"].detach()
                    else:
                        teacher_logits = teacher_outputs.detach()
                finally:
                    retention_teacher.restore(module, teacher_backup)
            with autocast_context(device, bool(cfg.train.amp), str(cfg.train.amp_dtype)):
                outputs = forward_model_batch(
                    model,
                    batch,
                    device,
                    return_aux=training_requires_aux_outputs(batch, cfg),
                )
                if isinstance(outputs, dict):
                    logits = outputs["logits"]
                else:
                    logits = outputs
                loss_matrix = criterion(logits, targets)
                clip_balance = str(
                    cfg.train.get("clip_loss_balance", "all")
                ).strip().lower()
                if clip_balance == "per_class_equal_pos_neg":
                    loss, clip_pos_loss, clip_neg_loss = per_class_equal_pos_neg_loss(
                        loss_matrix, targets, weighted_label_masks
                    )
                    step_components: dict[str, float] = {
                        "clip_loss": float(loss.detach().cpu()),
                        "clip_pos_loss": float(clip_pos_loss.detach().cpu()),
                        "clip_neg_loss": float(clip_neg_loss.detach().cpu()),
                    }
                elif clip_balance == "all":
                    loss = (
                        loss_matrix * weighted_label_masks
                    ).sum() / weighted_label_masks.sum().clamp_min(1.0)
                    step_components = {
                        "clip_loss": float(loss.detach().cpu())
                    }
                else:
                    raise ValueError(
                        "train.clip_loss_balance must be all or "
                        "per_class_equal_pos_neg"
                    )
                step_components.update(label_semantics_components)
                if (
                    isinstance(outputs, dict)
                    and "external_evidence_gate_values" in outputs
                ):
                    gate_values = outputs["external_evidence_gate_values"]
                    for gate_index, gate_label in enumerate(LABELS):
                        step_components[
                            f"external_evidence_gate_{gate_label}"
                        ] = float(gate_values[gate_index].detach().cpu())
                    step_components["external_evidence_gate_abs_mean"] = float(
                        gate_values.detach().abs().mean().cpu()
                    )
                if evidence_health_enabled and "evidence_features" in batch:
                    evidence_tensor = batch["evidence_features"]
                    valid_index = int(
                        evidence_health_cfg.get("valid_feature_index", 0)
                    )
                    if not 0 <= valid_index < evidence_tensor.shape[-1]:
                        raise RuntimeError(
                            "external evidence health valid_feature_index="
                            f"{valid_index} outside feature dim={evidence_tensor.shape[-1]}"
                        )
                    valid_fraction = float(
                        (evidence_tensor[..., valid_index] > 0).float().mean()
                    )
                    evidence_health_max_valid_fraction = max(
                        evidence_health_max_valid_fraction, valid_fraction
                    )
                    step_components["external_evidence_valid_fraction"] = (
                        valid_fraction
                    )
                clip_loss_weight = float(
                    cfg.train.get("clip_loss_weight", 1.0) or 0.0
                )
                raw_clip_loss = loss
                loss = clip_loss_weight * raw_clip_loss
                step_components["weighted_clip_loss"] = float(
                    (clip_loss_weight * raw_clip_loss.detach()).cpu()
                )
                low_tail_bce_weight = float(
                    cfg.train.get("positive_low_tail_bce_mining_loss_weight", 0.0)
                    or 0.0
                )
                if low_tail_bce_weight > 0:
                    low_tail_bce_label_names = cfg.train.get(
                        "positive_low_tail_bce_mining_labels",
                        cfg.train.get("clean_positive_logit_margin_labels", ["shot", "save"]),
                    )
                    if isinstance(low_tail_bce_label_names, str):
                        low_tail_bce_label_names = [
                            value.strip()
                            for value in low_tail_bce_label_names.split(",")
                            if value.strip()
                        ]
                    unknown_low_tail_bce_labels = [
                        label for label in low_tail_bce_label_names if label not in LABELS
                    ]
                    if unknown_low_tail_bce_labels:
                        raise ValueError(
                            "Unknown train.positive_low_tail_bce_mining_labels: "
                            f"{unknown_low_tail_bce_labels}; labels={LABELS}"
                        )
                    low_tail_bce_loss, low_tail_bce_components = (
                        positive_low_tail_bce_mining_loss(
                            loss_matrix,
                            logits,
                            targets,
                            weighted_label_masks,
                            label_indices=[
                                LABELS.index(label) for label in low_tail_bce_label_names
                            ],
                            quantile=float(
                                cfg.train.get(
                                    "positive_low_tail_bce_mining_quantile", 0.25
                                )
                                or 0.25
                            ),
                            min_count=int(
                                cfg.train.get(
                                    "positive_low_tail_bce_mining_min_count", 1
                                )
                                or 1
                            ),
                        )
                    )
                    loss = loss + low_tail_bce_weight * low_tail_bce_loss
                    step_components.update(low_tail_bce_components)
                    step_components["weighted_positive_low_tail_bce_loss"] = float(
                        (low_tail_bce_weight * low_tail_bce_loss.detach()).cpu()
                    )
                highres_local_clip_weight = float(
                    cfg.train.get("highres_local_clip_loss_weight", 0.0) or 0.0
                )
                if highres_local_clip_weight > 0:
                    if not isinstance(outputs, dict) or "highres_local_logits" not in outputs:
                        raise ValueError(
                            "highres_local_clip_loss_weight requires highres local logits"
                        )
                    local_clip_matrix = criterion(
                        outputs["highres_local_logits"], targets
                    )
                    local_clip_balance = str(
                        cfg.train.get(
                            "highres_local_clip_balance",
                            "per_class_equal_pos_neg",
                        )
                    ).strip().lower()
                    if local_clip_balance == "per_class_equal_pos_neg":
                        local_clip_loss, _, _ = per_class_equal_pos_neg_loss(
                            local_clip_matrix, targets, weighted_label_masks
                        )
                    elif local_clip_balance == "all":
                        local_clip_loss = (
                            local_clip_matrix * weighted_label_masks
                        ).sum() / weighted_label_masks.sum().clamp_min(1.0)
                    else:
                        raise ValueError(
                            "train.highres_local_clip_balance must be all or "
                            "per_class_equal_pos_neg"
                        )
                    loss = loss + highres_local_clip_weight * local_clip_loss
                    step_components["highres_local_clip_loss"] = float(
                        local_clip_loss.detach().cpu()
                    )
                highres_entropy_weight = float(
                    cfg.train.get("highres_crop_entropy_loss_weight", 0.0) or 0.0
                )
                if highres_entropy_weight > 0:
                    if not isinstance(outputs, dict) or "highres_crop_attention" not in outputs:
                        raise ValueError(
                            "highres crop entropy loss requires crop attention"
                        )
                    flat_crop_attention = outputs["highres_crop_attention"].flatten(2).clamp_min(1e-8)
                    crop_entropy = -(
                        flat_crop_attention * flat_crop_attention.log()
                    ).sum(dim=-1)
                    crop_entropy = crop_entropy / math.log(
                        max(flat_crop_attention.shape[-1], 2)
                    )
                    entropy_target = float(
                        cfg.train.get("highres_crop_entropy_target", 0.65)
                    )
                    crop_entropy_loss = F.relu(crop_entropy - entropy_target).square().mean()
                    loss = loss + highres_entropy_weight * crop_entropy_loss
                    step_components["highres_crop_entropy_loss"] = float(
                        crop_entropy_loss.detach().cpu()
                    )
                highres_causal_weight = float(
                    cfg.train.get("highres_causal_shuffle_loss_weight", 0.0) or 0.0
                )
                highres_causal_start_epoch = max(
                    int(cfg.train.get("highres_causal_start_epoch", 1)), 1
                )
                if highres_causal_weight > 0 and epoch >= highres_causal_start_epoch:
                    if not isinstance(outputs, dict) or "highres_shuffled_local_logits" not in outputs:
                        raise ValueError(
                            "highres causal supervision requires shuffled-local logits"
                        )
                    positive_mask = weighted_label_masks * (targets > 0.5).to(logits.dtype)
                    causal_margin = float(
                        cfg.train.get("highres_causal_shuffle_margin", 0.25)
                    )
                    shuffled_logits = outputs["highres_shuffled_local_logits"]
                    if shuffled_logits.ndim != 3:
                        raise ValueError(
                            "highres shuffled-local logits must be [B, shifts, C]"
                        )
                    if "highres_causal_valid_mask" not in outputs:
                        raise ValueError(
                            "hard shuffled-local supervision requires donor masks"
                        )
                    donor_negative_mask = outputs["highres_causal_valid_mask"]
                    causal_mask = positive_mask.unsqueeze(1) * donor_negative_mask
                    causal_matrix = causal_margin - logits.unsqueeze(1) + shuffled_logits
                    hardest_causal = causal_matrix.masked_fill(
                        causal_mask <= 0, -1e4
                    ).amax(dim=1)
                    has_valid_donor = (causal_mask.sum(dim=1) > 0).to(logits.dtype)
                    causal_loss = masked_mean(
                        F.relu(hardest_causal), positive_mask * has_valid_donor
                    )
                    loss = loss + highres_causal_weight * causal_loss
                    step_components["highres_causal_shuffle_loss"] = float(
                        causal_loss.detach().cpu()
                    )
                    step_components["highres_causal_valid_fraction"] = float(
                        (
                            (positive_mask * has_valid_donor).sum()
                            / positive_mask.sum().clamp_min(1.0)
                        ).detach().cpu()
                    )
                if isinstance(outputs, dict) and "highres_evidence_gates" in outputs:
                    gates = outputs["highres_evidence_gates"].detach()
                    scales = outputs["highres_crop_scales"].detach()
                    crop_attention = outputs["highres_crop_attention"].detach()
                    flat_attention = crop_attention.flatten(2).clamp_min(1e-8)
                    attention_entropy = -(
                        flat_attention * flat_attention.log()
                    ).sum(dim=-1)
                    attention_entropy = attention_entropy / math.log(
                        max(flat_attention.shape[-1], 2)
                    )
                    step_components["highres_evidence_gate_mean"] = float(
                        gates.mean().cpu()
                    )
                    step_components["highres_crop_scale_w"] = float(
                        scales[..., 0].mean().cpu()
                    )
                    step_components["highres_crop_scale_h"] = float(
                        scales[..., 1].mean().cpu()
                    )
                    step_components["highres_attention_entropy"] = float(
                        attention_entropy.mean().cpu()
                    )
                save_given_shot_weight = float(
                    cfg.train.get("save_given_shot_loss_weight", 0.0) or 0.0
                )
                if save_given_shot_weight > 0:
                    conditional_loss, conditional_components = (
                        save_given_shot_classification_loss(
                            logits, targets, weighted_label_masks
                        )
                    )
                    loss = loss + save_given_shot_weight * conditional_loss
                    step_components.update(conditional_components)
                save_cohort_weight = float(
                    cfg.train.get("save_shot_cohort_loss_weight", 0.0) or 0.0
                )
                if save_cohort_weight > 0:
                    cohort_loss, cohort_components = (
                        save_shot_cohort_classification_loss(
                            logits, batch, device
                        )
                    )
                    loss = loss + save_cohort_weight * cohort_loss
                    step_components.update(cohort_components)
                set_piece_subtype_weight = float(
                    cfg.train.get("set_piece_subtype_loss_weight", 0.0) or 0.0
                )
                if set_piece_subtype_weight > 0:
                    if not isinstance(outputs, dict):
                        raise ValueError(
                            "set-piece subtype supervision requires auxiliary model outputs"
                        )
                    subtype_loss, subtype_components = (
                        set_piece_subtype_classification_loss(
                            outputs, batch, device, cfg
                        )
                    )
                    loss = loss + set_piece_subtype_weight * subtype_loss
                    step_components.update(subtype_components)
                raw_context_span_weight = float(
                    cfg.train.get("raw_context_span_loss_weight", 0.0) or 0.0
                )
                if raw_context_span_weight > 0:
                    if not isinstance(outputs, dict):
                        raise ValueError("raw context-span supervision requires auxiliary outputs")
                    context_loss, context_components = raw_set_piece_context_span_coverage_loss(
                        outputs, batch, device, cfg
                    )
                    loss = loss + raw_context_span_weight * context_loss
                    step_components.update(context_components)
                global_action_aux_weight = float(
                    cfg.train.get("global_action_aux_loss_weight", 0.0) or 0.0
                )
                if (
                    isinstance(outputs, dict)
                    and global_action_aux_weight > 0
                    and "global_action_logits" in outputs
                ):
                    global_action_loss_matrix = criterion(
                        outputs["global_action_logits"], targets
                    )
                    if clip_balance == "per_class_equal_pos_neg":
                        global_action_loss, _, _ = per_class_equal_pos_neg_loss(
                            global_action_loss_matrix,
                            targets,
                            weighted_label_masks,
                        )
                    else:
                        global_action_loss = (
                            global_action_loss_matrix * weighted_label_masks
                        ).sum() / weighted_label_masks.sum().clamp_min(1.0)
                    loss = loss + global_action_aux_weight * global_action_loss
                    step_components["global_action_aux_loss"] = float(
                        global_action_loss.detach().cpu()
                    )
                    step_components["weighted_global_action_aux_loss"] = float(
                        (
                            global_action_aux_weight
                            * global_action_loss.detach()
                        ).cpu()
                    )
                temporal_branch_aux_weight = float(
                    cfg.train.get("temporal_branch_aux_loss_weight", 0.0) or 0.0
                )
                if (
                    isinstance(outputs, dict)
                    and temporal_branch_aux_weight > 0
                    and "temporal_delta_logits" in outputs
                ):
                    temporal_branch_loss_matrix = criterion(
                        outputs["temporal_delta_logits"], targets
                    )
                    if clip_balance == "per_class_equal_pos_neg":
                        temporal_branch_loss, _, _ = per_class_equal_pos_neg_loss(
                            temporal_branch_loss_matrix,
                            targets,
                            weighted_label_masks,
                        )
                    else:
                        temporal_branch_loss = (
                            temporal_branch_loss_matrix * weighted_label_masks
                        ).sum() / weighted_label_masks.sum().clamp_min(1.0)
                    loss = (
                        loss
                        + temporal_branch_aux_weight * temporal_branch_loss
                    )
                    step_components["temporal_branch_aux_loss"] = float(
                        temporal_branch_loss.detach().cpu()
                    )
                    step_components["weighted_temporal_branch_aux_loss"] = float(
                        (
                            temporal_branch_aux_weight
                            * temporal_branch_loss.detach()
                        ).cpu()
                    )
                blocked_online_slots = (
                    (label_masks > 0) & (online_negative_masks <= 0)
                ).sum()
                step_components["online_negative_safety_masked"] = float(
                    blocked_online_slots.detach().cpu()
                )
                if isinstance(outputs, dict) and "temporal_gate" in outputs:
                    mean_gate = outputs["temporal_gate"].detach().float().mean(dim=0)
                    for label_index, label in enumerate(LABELS):
                        step_components[f"temporal_gate_{label}"] = float(
                            mean_gate[label_index].cpu()
                        )
                if (
                    isinstance(outputs, dict)
                    and "temporal_residual_gate" in outputs
                ):
                    mean_gate = outputs[
                        "temporal_residual_gate"
                    ].detach().float().mean(dim=0)
                    for label_index, label in enumerate(LABELS):
                        step_components[
                            f"videomae_temporal_gate_{label}"
                        ] = float(mean_gate[label_index].cpu())
                    step_components["videomae_temporal_delta_abs_mean"] = float(
                        outputs["temporal_delta_logits"]
                        .detach()
                        .float()
                        .abs()
                        .mean()
                        .cpu()
                    )
                if isinstance(outputs, dict) and "roi_gain_weight_a" in outputs:
                    step_components["roi_gain_weight_a"] = float(
                        outputs["roi_gain_weight_a"].detach().float().mean().cpu()
                    )
                    step_components["roi_gain_weight_b"] = float(
                        outputs["roi_gain_weight_b"].detach().float().mean().cpu()
                    )
                if (
                    isinstance(outputs, dict)
                    and "roi_memory_residual_gates" in outputs
                ):
                    step_components["roi_memory_residual_gate"] = float(
                        outputs["roi_memory_residual_gates"]
                        .detach()
                        .float()
                        .abs()
                        .mean()
                        .cpu()
                    )
                    step_components["roi_memory_quality_a"] = float(
                        outputs["roi_memory_quality_a"]
                        .detach()
                        .float()
                        .mean()
                        .cpu()
                    )
                    step_components["roi_memory_quality_b"] = float(
                        outputs["roi_memory_quality_b"]
                        .detach()
                        .float()
                        .mean()
                        .cpu()
                    )
                    step_components["roi_b_valid_fraction"] = float(
                        batch["roi_frame_valid_b"].detach().float().mean().cpu()
                    )
                if isinstance(outputs, dict) and "roi_pair_gate" in outputs:
                    step_components["roi_pair_gate_mean"] = float(
                        outputs["roi_pair_gate"].detach().float().mean().cpu()
                    )
                    step_components["roi_b_valid_fraction"] = float(
                        batch["roi_frame_valid_b"].detach().float().mean().cpu()
                    )
                    if "roi_pair_delta" in outputs:
                        step_components["roi_pair_delta_abs_mean"] = float(
                            outputs["roi_pair_delta"]
                            .detach()
                            .float()
                            .abs()
                            .mean()
                            .cpu()
                        )
                if isinstance(outputs, dict) and "roi_gate" in outputs:
                    mean_roi_gate = outputs["roi_gate"].detach().float().mean(dim=0).reshape(-1)
                    veto_delta = (
                        outputs["global_logits"].detach().float()
                        - outputs["logits"].detach().float()
                    ).mean(dim=0)
                    for label_index, label in enumerate(LABELS):
                        gate_index = label_index if mean_roi_gate.numel() > 1 else 0
                        step_components[f"roi_gate_{label}"] = float(
                            mean_roi_gate[gate_index].cpu()
                        )
                        step_components[f"roi_veto_delta_{label}"] = float(
                            veto_delta[label_index].cpu()
                        )
                if hard_count:
                    hard_rows = hard_rows_bool.to(label_masks.dtype).reshape(-1, 1)
                    hard_mask = label_masks * hard_rows
                    hard_slots = float(hard_mask.sum().detach().cpu())
                    hard_negative_loss_total += float(
                        (loss_matrix * hard_mask).sum().detach().cpu()
                    )
                    hard_negative_label_slots += hard_slots
                    hard_negative_samples += hard_count
                observed_samples += int(targets.shape[0])
                if isinstance(outputs, dict) and "local_logits" in outputs and float(cfg.train.get("local_loss_weight", 0.0)) > 0:
                    roi_valid = batch["roi_valid"].to(device, non_blocking=True).reshape(-1, 1)
                    local_mask = weighted_label_masks * roi_valid
                    local_matrix = criterion(outputs["local_logits"], targets)
                    local_loss = (local_matrix * local_mask).sum() / local_mask.sum().clamp_min(1.0)
                    loss = loss + float(cfg.train.local_loss_weight) * local_loss
                    step_components["local_loss"] = float(local_loss.detach().cpu())
                if isinstance(outputs, dict):
                    for branch_key, config_key, component_key in (
                        (
                            "uniform_logits",
                            "uniform_temporal_loss_weight",
                            "uniform_temporal_loss",
                        ),
                        (
                            "event_logits",
                            "event_temporal_loss_weight",
                            "event_temporal_loss",
                        ),
                    ):
                        branch_weight = float(
                            cfg.train.get(config_key, 0.0) or 0.0
                        )
                        if branch_key not in outputs or branch_weight <= 0:
                            continue
                        branch_matrix = criterion(outputs[branch_key], targets)
                        branch_loss = (
                            branch_matrix * weighted_label_masks
                        ).sum() / weighted_label_masks.sum().clamp_min(1.0)
                        loss = loss + branch_weight * branch_loss
                        step_components[component_key] = float(
                            branch_loss.detach().cpu()
                        )
                clean_rank_weight = float(
                    cfg.train.get("clean_negative_rank_loss_weight", 0.0) or 0.0
                )
                if clean_rank_weight > 0:
                    clean_rank_branch = str(
                        cfg.train.get("clean_negative_rank_branch", "fused")
                    ).strip().lower()
                    if clean_rank_branch == "event":
                        if not isinstance(outputs, dict) or "event_logits" not in outputs:
                            raise ValueError(
                                "train.clean_negative_rank_branch=event requires event_logits"
                            )
                        clean_rank_logits = outputs["event_logits"]
                    elif clean_rank_branch == "fused":
                        clean_rank_logits = logits
                    else:
                        raise ValueError(
                            "train.clean_negative_rank_branch must be event or fused"
                        )
                    explicit_clean_masks = [
                        tuple(meta.get("online_clean_negative_mask", ()) or ())
                        for meta in batch["meta"]
                    ]
                    has_explicit_clean_masks = any(
                        len(mask) == len(LABELS) for mask in explicit_clean_masks
                    )
                    if has_explicit_clean_masks:
                        if not all(
                            len(mask) == len(LABELS)
                            for mask in explicit_clean_masks
                        ):
                            raise ValueError(
                                "online_clean_negative_mask must be present for every "
                                "row in an online-simulation batch"
                            )
                        clean_flags = torch.tensor(
                            explicit_clean_masks,
                            dtype=torch.bool,
                            device=device,
                        )
                        clean_flags = clean_flags & (label_masks > 0)
                    else:
                        clean_flags = torch.tensor(
                            [
                                bool(meta.get("is_negative", False))
                                and str(meta.get("sample_id", "")).startswith("neg_")
                                for meta in batch["meta"]
                            ],
                            dtype=torch.bool,
                            device=device,
                        )
                        if bool(
                            cfg.train.get(
                                "clean_negative_rank_require_all_labels", True
                            )
                        ):
                            clean_flags = clean_flags & (label_masks > 0).all(dim=1)
                    clean_rank_label_names = cfg.train.get(
                        "clean_negative_rank_labels", LABELS
                    )
                    if isinstance(clean_rank_label_names, str):
                        clean_rank_label_names = [
                            value.strip()
                            for value in clean_rank_label_names.split(",")
                            if value.strip()
                        ]
                    unknown_clean_rank_labels = [
                        label
                        for label in clean_rank_label_names
                        if label not in LABELS
                    ]
                    if unknown_clean_rank_labels:
                        raise ValueError(
                            "Unknown train.clean_negative_rank_labels: "
                            f"{unknown_clean_rank_labels}; labels={LABELS}"
                        )
                    clean_rank_loss, clean_rank_components = (
                        clean_negative_pairwise_rank_loss(
                            clean_rank_logits,
                            targets,
                            label_masks,
                            clean_flags,
                            label_indices=[
                                LABELS.index(label)
                                for label in clean_rank_label_names
                            ],
                            margin=float(
                                cfg.train.get("clean_negative_rank_margin", 1.0)
                            ),
                            pairs_per_positive=int(
                                cfg.train.get(
                                    "clean_negative_rank_pairs_per_positive", 2
                                )
                            ),
                            temperature=float(
                                cfg.train.get(
                                    "clean_negative_rank_temperature", 1.0
                                )
                            ),
                        )
                    )
                    loss = loss + clean_rank_weight * clean_rank_loss
                    step_components["clean_negative_rank_loss"] = float(
                        clean_rank_loss.detach().cpu()
                    )
                    step_components.update(clean_rank_components)
                clean_pos_margin_weight = float(
                    cfg.train.get("clean_positive_logit_margin_loss_weight", 0.0)
                    or 0.0
                )
                if clean_pos_margin_weight > 0:
                    clean_pos_label_names = cfg.train.get(
                        "clean_positive_logit_margin_labels", ["shot", "save"]
                    )
                    if isinstance(clean_pos_label_names, str):
                        clean_pos_label_names = [
                            value.strip()
                            for value in clean_pos_label_names.split(",")
                            if value.strip()
                        ]
                    unknown_clean_pos_labels = [
                        label for label in clean_pos_label_names if label not in LABELS
                    ]
                    if unknown_clean_pos_labels:
                        raise ValueError(
                            "Unknown train.clean_positive_logit_margin_labels: "
                            f"{unknown_clean_pos_labels}; labels={LABELS}"
                        )
                    clean_pos_loss, clean_pos_components = (
                        clean_positive_logit_margin_loss(
                            outputs if isinstance(outputs, dict) else {"logits": logits},
                            targets,
                            label_masks,
                            margins=per_label_float_tuple(
                                cfg.train.get(
                                    "clean_positive_logit_margin",
                                    [0.0] * len(LABELS),
                                ),
                                default=0.0,
                            ),
                            label_indices=[
                                LABELS.index(label) for label in clean_pos_label_names
                            ],
                            branch=str(
                                cfg.train.get(
                                    "clean_positive_logit_margin_branch", "fused"
                                )
                            ),
                        )
                    )
                    loss = loss + clean_pos_margin_weight * clean_pos_loss
                    step_components.update(clean_pos_components)
                low_tail_weight = float(
                    cfg.train.get("positive_low_tail_logit_margin_loss_weight", 0.0)
                    or 0.0
                )
                if low_tail_weight > 0:
                    low_tail_label_names = cfg.train.get(
                        "positive_low_tail_logit_margin_labels",
                        cfg.train.get("clean_positive_logit_margin_labels", ["shot", "save"]),
                    )
                    if isinstance(low_tail_label_names, str):
                        low_tail_label_names = [
                            value.strip()
                            for value in low_tail_label_names.split(",")
                            if value.strip()
                        ]
                    unknown_low_tail_labels = [
                        label for label in low_tail_label_names if label not in LABELS
                    ]
                    if unknown_low_tail_labels:
                        raise ValueError(
                            "Unknown train.positive_low_tail_logit_margin_labels: "
                            f"{unknown_low_tail_labels}; labels={LABELS}"
                        )
                    low_tail_loss, low_tail_components = (
                        positive_low_tail_logit_margin_loss(
                            outputs if isinstance(outputs, dict) else {"logits": logits},
                            targets,
                            label_masks,
                            margins=per_label_float_tuple(
                                cfg.train.get(
                                    "positive_low_tail_logit_margin",
                                    cfg.train.get(
                                        "clean_positive_logit_margin",
                                        [0.0] * len(LABELS),
                                    ),
                                ),
                                default=0.0,
                            ),
                            label_indices=[
                                LABELS.index(label) for label in low_tail_label_names
                            ],
                            branch=str(
                                cfg.train.get(
                                    "positive_low_tail_logit_margin_branch",
                                    cfg.train.get(
                                        "clean_positive_logit_margin_branch", "fused"
                                    ),
                                )
                            ),
                            quantile=float(
                                cfg.train.get(
                                    "positive_low_tail_logit_margin_quantile", 0.25
                                )
                                or 0.25
                            ),
                            min_count=int(
                                cfg.train.get(
                                    "positive_low_tail_logit_margin_min_count", 1
                                )
                                or 1
                            ),
                        )
                    )
                    loss = loss + low_tail_weight * low_tail_loss
                    step_components.update(low_tail_components)
                hpm_weight = float(
                    cfg.train.get("hard_positive_tail_margin_loss_weight", 0.0)
                    or 0.0
                )
                if hpm_weight > 0:
                    hpm_label_names = cfg.train.get(
                        "hard_positive_tail_margin_labels",
                        cfg.train.get(
                            "clean_positive_logit_margin_labels", ["shot", "save"]
                        ),
                    )
                    if isinstance(hpm_label_names, str):
                        hpm_label_names = [
                            value.strip()
                            for value in hpm_label_names.split(",")
                            if value.strip()
                        ]
                    unknown_hpm_labels = [
                        label for label in hpm_label_names if label not in LABELS
                    ]
                    if unknown_hpm_labels:
                        raise ValueError(
                            "Unknown train.hard_positive_tail_margin_labels: "
                            f"{unknown_hpm_labels}; labels={LABELS}"
                        )
                    hpm_label_indices = [
                        LABELS.index(label) for label in hpm_label_names
                    ]
                    hpm_margins = per_label_float_tuple(
                        cfg.train.get(
                            "hard_positive_tail_margin",
                            cfg.train.get(
                                "clean_positive_logit_margin", [0.0] * len(LABELS)
                            ),
                        ),
                        default=0.25,
                    )
                    hpm_floors = [
                        hard_positive_tail_floors.get(idx, hpm_margins[idx])
                        for idx in range(len(LABELS))
                    ]
                    hpm_loss, hpm_components = hard_positive_tail_margin_loss(
                        logits,
                        targets,
                        label_masks,
                        floors=hpm_floors,
                        margins=hpm_margins,
                        label_indices=hpm_label_indices,
                    )
                    loss = loss + hpm_weight * hpm_loss
                    step_components.update(hpm_components)
                    hpm_quantile = min(
                        max(
                            float(
                                cfg.train.get("hard_positive_tail_quantile", 0.2)
                                or 0.2
                            ),
                            0.0,
                        ),
                        1.0,
                    )
                    hpm_decay = float(
                        cfg.train.get("hard_positive_tail_ema_decay", 0.95) or 0.95
                    )
                    for hpm_idx in hpm_label_indices:
                        hpm_pos_mask = (label_masks[:, hpm_idx] > 0) & (
                            targets[:, hpm_idx] > 0.5
                        )
                        hpm_vals = logits.detach()[hpm_pos_mask, hpm_idx].float()
                        if distributed_training_active():
                            local_count = torch.tensor(
                                [hpm_vals.numel()],
                                device=logits.device,
                                dtype=torch.long,
                            )
                            gathered_counts = [
                                torch.zeros_like(local_count)
                                for _ in range(dist.get_world_size())
                            ]
                            dist.all_gather(gathered_counts, local_count)
                            counts = [int(value.item()) for value in gathered_counts]
                            max_count = max(counts)
                            if max_count > 0:
                                padded = torch.zeros(
                                    max_count,
                                    device=logits.device,
                                    dtype=torch.float32,
                                )
                                if hpm_vals.numel() > 0:
                                    padded[: hpm_vals.numel()] = hpm_vals
                                gathered_values = [
                                    torch.zeros_like(padded)
                                    for _ in range(dist.get_world_size())
                                ]
                                dist.all_gather(gathered_values, padded)
                                hpm_vals = torch.cat(
                                    [
                                        value[:count]
                                        for value, count in zip(
                                            gathered_values, counts
                                        )
                                    ]
                                )
                        if hpm_vals.numel() == 0:
                            continue
                        hpm_step_q = float(
                            torch.quantile(hpm_vals.float(), hpm_quantile).item()
                        )
                        hpm_old = hard_positive_tail_floors.get(
                            hpm_idx, hpm_margins[hpm_idx]
                        )
                        hard_positive_tail_floors[hpm_idx] = (
                            hpm_decay * hpm_old + (1.0 - hpm_decay) * hpm_step_q
                        )
                    for hpm_idx in hpm_label_indices:
                        step_components[
                            f"hard_positive_tail_floor_{LABELS[hpm_idx]}"
                        ] = hard_positive_tail_floors.get(hpm_idx, 0.0)
                tail_sep_weight = float(
                    cfg.train.get("tail_separation_loss_weight", 0.0) or 0.0
                )
                if tail_sep_weight > 0:
                    tail_sep_label_names = cfg.train.get(
                        "tail_separation_labels", LABELS
                    )
                    unknown_tail_sep_labels = [
                        label
                        for label in tail_sep_label_names
                        if label not in LABELS
                    ]
                    if unknown_tail_sep_labels:
                        raise ValueError(
                            "Unknown train.tail_separation_labels: "
                            f"{unknown_tail_sep_labels}; labels={LABELS}"
                        )
                    tail_sep_label_indices = [
                        LABELS.index(label) for label in tail_sep_label_names
                    ]
                    tail_sep_loss, tail_sep_components = tail_separation_paired_loss(
                        logits,
                        targets,
                        label_masks,
                        batch["meta"],
                        labels=LABELS,
                        margin=float(
                            cfg.train.get("tail_separation_margin", 0.15)
                        ),
                        positive_quantile=float(
                            cfg.train.get(
                                "tail_separation_positive_quantile", 0.25
                            )
                        ),
                        negative_quantile=float(
                            cfg.train.get(
                                "tail_separation_negative_quantile", 0.25
                            )
                        ),
                        min_positive=int(
                            cfg.train.get("tail_separation_min_positive", 2)
                        ),
                        min_negative=int(
                            cfg.train.get("tail_separation_min_negative", 2)
                        ),
                        label_indices=tail_sep_label_indices,
                    )
                    loss = loss + tail_sep_weight * tail_sep_loss
                    step_components.update(tail_sep_components)
                    step_components["weighted_tail_separation_loss"] = float(
                        (tail_sep_weight * tail_sep_loss.detach()).cpu()
                    )
                global_tail_weight = float(
                    cfg.train.get("online_global_tail_rank_loss_weight", 0.0) or 0.0
                )
                if global_tail_weight > 0:
                    global_tail_label_names = cfg.train.get(
                        "online_global_tail_rank_labels", LABELS
                    )
                    if isinstance(global_tail_label_names, str):
                        global_tail_label_names = [
                            value.strip()
                            for value in global_tail_label_names.split(",")
                            if value.strip()
                        ]
                    unknown_global_tail_labels = [
                        label
                        for label in global_tail_label_names
                        if label not in LABELS
                    ]
                    if unknown_global_tail_labels:
                        raise ValueError(
                            "Unknown train.online_global_tail_rank_labels: "
                            f"{unknown_global_tail_labels}; labels={LABELS}"
                        )
                    global_tail_loss, global_tail_components = (
                        online_global_tail_ranking_loss(
                            logits,
                            targets,
                            label_masks,
                            batch["meta"],
                            labels=LABELS,
                            margin=float(
                                cfg.train.get("online_global_tail_rank_margin", 0.5)
                            ),
                            positive_topk=int(
                                cfg.train.get("online_global_tail_rank_positive_topk", 4)
                            ),
                            negative_topk=int(
                                cfg.train.get("online_global_tail_rank_negative_topk", 4)
                            ),
                            label_indices=[
                                LABELS.index(label) for label in global_tail_label_names
                            ],
                        )
                    )
                    loss = loss + global_tail_weight * global_tail_loss
                    step_components.update(global_tail_components)
                    step_components["weighted_online_global_tail_loss"] = float(
                        (global_tail_weight * global_tail_loss.detach()).cpu()
                    )
                hard_rank_weight = float(
                    cfg.train.get("hard_negative_rank_loss_weight", 0.0) or 0.0
                )
                if hard_count and hard_rank_weight > 0:
                    hard_rank_branch = str(
                        cfg.train.get("hard_negative_rank_branch", "event")
                    ).strip().lower()
                    if hard_rank_branch == "event":
                        if not isinstance(outputs, dict) or "event_logits" not in outputs:
                            raise ValueError(
                                "train.hard_negative_rank_branch=event requires event_logits"
                            )
                        hard_rank_logits = outputs["event_logits"]
                    elif hard_rank_branch == "fused":
                        hard_rank_logits = logits
                    else:
                        raise ValueError(
                            "train.hard_negative_rank_branch must be event or fused"
                        )
                    hard_rank_label_names = cfg.train.get(
                        "hard_negative_rank_labels", LABELS
                    )
                    if isinstance(hard_rank_label_names, str):
                        hard_rank_label_names = [
                            value.strip()
                            for value in hard_rank_label_names.split(",")
                            if value.strip()
                        ]
                    unknown_hard_rank_labels = [
                        label for label in hard_rank_label_names if label not in LABELS
                    ]
                    if unknown_hard_rank_labels:
                        raise ValueError(
                            "Unknown train.hard_negative_rank_labels: "
                            f"{unknown_hard_rank_labels}; labels={LABELS}"
                        )
                    hard_rank_loss = hard_negative_pairwise_rank_loss(
                        hard_rank_logits,
                        targets,
                        label_masks,
                        hard_rows_bool,
                        label_indices=[
                            LABELS.index(label) for label in hard_rank_label_names
                        ],
                        margin=float(
                            cfg.train.get("hard_negative_rank_margin", 1.0) or 0.0
                        ),
                    )
                    loss = loss + hard_rank_weight * hard_rank_loss
                    step_components["hard_negative_rank_loss"] = float(
                        hard_rank_loss.detach().cpu()
                    )
                online_hard_weight = float(
                    cfg.train.get("online_hard_negative_loss_weight", 0.0)
                    or 0.0
                )
                if online_hard_weight > 0:
                    online_hard_branch = str(
                        cfg.train.get("online_hard_negative_branch", "event")
                    ).strip().lower()
                    if online_hard_branch == "event":
                        if not isinstance(outputs, dict) or "event_logits" not in outputs:
                            raise ValueError(
                                "train.online_hard_negative_branch=event requires event_logits"
                            )
                        online_hard_logits = outputs["event_logits"]
                    elif online_hard_branch == "fused":
                        online_hard_logits = logits
                    else:
                        raise ValueError(
                            "train.online_hard_negative_branch must be event or fused"
                        )
                    online_label_names = cfg.train.get(
                        "online_hard_negative_labels", ["shot", "save"]
                    )
                    if isinstance(online_label_names, str):
                        online_label_names = [
                            value.strip()
                            for value in online_label_names.split(",")
                            if value.strip()
                        ]
                    unknown_labels = [
                        label for label in online_label_names if label not in LABELS
                    ]
                    if unknown_labels:
                        raise ValueError(
                            "Unknown train.online_hard_negative_labels: "
                            f"{unknown_labels}; labels={LABELS}"
                        )
                    online_label_indices = [
                        LABELS.index(label) for label in online_label_names
                    ]
                    online_hard_loss, online_selected = online_hard_negative_loss(
                        online_hard_logits,
                        targets,
                        online_negative_masks,
                        label_indices=online_label_indices,
                        fraction=float(
                            cfg.train.get(
                                "online_hard_negative_fraction", 0.25
                            )
                            or 0.25
                        ),
                        min_per_class=int(
                            cfg.train.get(
                                "online_hard_negative_min_per_class", 1
                            )
                            or 1
                        ),
                    )
                    loss = loss + online_hard_weight * online_hard_loss
                    step_components["online_hard_negative_loss"] = float(
                        online_hard_loss.detach().cpu()
                    )
                    step_components["online_hard_negative_selected"] = float(
                        online_selected
                    )
                online_rank_weight = float(
                    cfg.train.get(
                        "online_hard_negative_rank_loss_weight", 0.0
                    )
                    or 0.0
                )
                if online_rank_weight > 0:
                    online_rank_branch = str(
                        cfg.train.get(
                            "online_hard_negative_rank_branch", "event"
                        )
                    ).strip().lower()
                    if online_rank_branch == "event":
                        if not isinstance(outputs, dict) or "event_logits" not in outputs:
                            raise ValueError(
                                "train.online_hard_negative_rank_branch=event "
                                "requires event_logits"
                            )
                        online_rank_logits = outputs["event_logits"]
                    elif online_rank_branch == "fused":
                        online_rank_logits = logits
                    else:
                        raise ValueError(
                            "train.online_hard_negative_rank_branch must be "
                            "event or fused"
                        )
                    online_rank_label_names = cfg.train.get(
                        "online_hard_negative_labels", ["shot", "save"]
                    )
                    if isinstance(online_rank_label_names, str):
                        online_rank_label_names = [
                            value.strip()
                            for value in online_rank_label_names.split(",")
                            if value.strip()
                        ]
                    unknown_rank_labels = [
                        label
                        for label in online_rank_label_names
                        if label not in LABELS
                    ]
                    if unknown_rank_labels:
                        raise ValueError(
                            "Unknown train.online_hard_negative_labels: "
                            f"{unknown_rank_labels}; labels={LABELS}"
                        )
                    online_rank_loss, online_rank_selected = (
                        online_hard_negative_rank_loss(
                            online_rank_logits,
                            targets,
                            online_negative_masks,
                            label_indices=[
                                LABELS.index(label)
                                for label in online_rank_label_names
                            ],
                            fraction=float(
                                cfg.train.get(
                                    "online_hard_negative_fraction", 0.25
                                )
                                or 0.25
                            ),
                            min_per_class=int(
                                cfg.train.get(
                                    "online_hard_negative_min_per_class", 1
                                )
                                or 1
                            ),
                            margin=float(
                                cfg.train.get(
                                    "online_hard_negative_rank_margin", 1.0
                                )
                                or 0.0
                            ),
                        )
                    )
                    loss = loss + online_rank_weight * online_rank_loss
                    step_components[
                        "online_hard_negative_rank_loss"
                    ] = float(online_rank_loss.detach().cpu())
                    step_components[
                        "online_hard_negative_rank_selected"
                    ] = float(online_rank_selected)
                correction_weight = float(
                    cfg.train.get(
                        "global_conditioned_correction_loss_weight", 0.0
                    )
                    or 0.0
                )
                if correction_weight > 0:
                    if not isinstance(outputs, dict):
                        raise ValueError(
                            "global conditioned correction requires auxiliary outputs"
                        )
                    correction_thresholds = per_label_float_tuple(
                        cfg.train.get(
                            "global_conditioned_reference_thresholds", 0.5
                        ),
                        default=0.5,
                    )
                    correction_loss, correction_components = (
                        global_conditioned_correction_loss(
                            outputs,
                            targets,
                            label_masks,
                            threshold_probs=correction_thresholds,
                            margin_logit=float(
                                cfg.train.get(
                                    "global_conditioned_margin_logit", 0.25
                                )
                                or 0.0
                            ),
                            fp_weight=float(
                                cfg.train.get("global_conditioned_fp_weight", 1.0)
                                or 0.0
                            ),
                            fn_weight=float(
                                cfg.train.get("global_conditioned_fn_weight", 1.25)
                                or 0.0
                            ),
                            tp_guard_weight=float(
                                cfg.train.get(
                                    "global_conditioned_tp_guard_weight", 0.5
                                )
                                or 0.0
                            ),
                            tn_guard_weight=float(
                                cfg.train.get(
                                    "global_conditioned_tn_guard_weight", 0.25
                                )
                                or 0.0
                            ),
                            stability_weight=float(
                                cfg.train.get(
                                    "global_conditioned_stability_weight", 0.02
                                )
                                or 0.0
                            ),
                            target_max_delta=float(
                                cfg.train.get(
                                    "global_conditioned_target_max_delta", 2.0
                                )
                                or 2.0
                            ),
                            smooth_l1_beta=float(
                                cfg.train.get(
                                    "global_conditioned_smooth_l1_beta", 0.5
                                )
                                or 0.5
                            ),
                        )
                    )
                    loss = loss + correction_weight * correction_loss
                    step_components.update(correction_components)
                class_evidence_diversity_weight = float(
                    cfg.train.get(
                        "class_evidence_query_diversity_loss_weight", 0.0
                    )
                    or 0.0
                )
                if class_evidence_diversity_weight > 0:
                    if not isinstance(outputs, dict) or (
                        "class_evidence_query_diversity_loss" not in outputs
                    ):
                        raise ValueError(
                            "class evidence query diversity supervision requires "
                            "outputs[class_evidence_query_diversity_loss]"
                        )
                    class_evidence_diversity_loss = outputs[
                        "class_evidence_query_diversity_loss"
                    ].mean()
                    loss = (
                        loss
                        + class_evidence_diversity_weight
                        * class_evidence_diversity_loss
                    )
                    step_components[
                        "class_evidence_query_diversity_loss"
                    ] = float(class_evidence_diversity_loss.detach().cpu())
                class_evidence_counterfactual_weight = float(
                    cfg.train.get(
                        "class_evidence_counterfactual_loss_weight", 0.0
                    )
                    or 0.0
                )
                if class_evidence_counterfactual_weight > 0:
                    if not isinstance(outputs, dict):
                        raise ValueError(
                            "class evidence counterfactual supervision requires "
                            "auxiliary model outputs"
                        )
                    (
                        class_evidence_counterfactual_loss,
                        class_evidence_counterfactual_components,
                    ) = class_evidence_counterfactual_causal_loss(
                        outputs,
                        targets,
                        weighted_label_masks,
                        cfg,
                    )
                    if not class_evidence_counterfactual_components:
                        raise ValueError(
                            "class evidence counterfactual supervision requires "
                            "outputs[class_evidence_no_evidence_logits]"
                        )
                    loss = loss + (
                        class_evidence_counterfactual_weight
                        * class_evidence_counterfactual_loss
                    )
                    step_components.update(
                        class_evidence_counterfactual_components
                    )
                no_evidence_weight = float(
                    cfg.train.get("class_evidence_no_evidence_loss_weight", 0.0)
                    or 0.0
                )
                if no_evidence_weight > 0:
                    if not isinstance(outputs, dict):
                        raise ValueError(
                            "no-evidence supervision requires auxiliary model outputs"
                        )
                    no_evidence_balance = str(
                        cfg.train.get(
                            "class_evidence_no_evidence_balance", "all"
                        )
                    ).strip().lower()
                    if no_evidence_balance == "per_class_equal_event_background":
                        no_evidence_loss, no_evidence_components = (
                            balanced_no_evidence_loss(
                                outputs,
                                batch,
                                device,
                                cfg.train,
                                labels=LABELS,
                            )
                        )
                    elif no_evidence_balance == "all":
                        no_evidence_loss, no_evidence_components = (
                            class_evidence_no_evidence_loss(
                                outputs, batch, device
                            )
                        )
                    else:
                        raise ValueError(
                            "class_evidence_no_evidence_balance must be all or "
                            "per_class_equal_event_background"
                        )
                    if not no_evidence_components:
                        raise ValueError(
                            "no-evidence supervision requires frame targets and "
                            "outputs[class_evidence_no_evidence_mass]"
                        )
                    loss = loss + no_evidence_weight * no_evidence_loss
                    step_components.update(no_evidence_components)
                    step_components["weighted_class_evidence_no_evidence_loss"] = float(
                        (no_evidence_weight * no_evidence_loss.detach()).cpu()
                    )
                # E1.3.2 (2026-08-26): bag-level saliency ranking and
                # cross-window saliency consistency replace the frame-level
                # null-mass binary supervision (weight 0). "Not an event
                # frame" != "no local evidence" in football, so the gate is
                # kept as a diagnostic only and the shared patch scorer is
                # taught instead that event windows carry stronger Top-K
                # evidence than clean negatives, consistently across
                # overlapping windows.
                saliency_rank_weight = float(
                    cfg.train.get("class_evidence_saliency_rank_weight", 0.0)
                    or 0.0
                )
                cross_window_weight = float(
                    cfg.train.get("class_evidence_cross_window_weight", 0.0)
                    or 0.0
                )
                if isinstance(outputs, dict) and (
                    saliency_rank_weight > 0 or cross_window_weight > 0
                ):
                    saliency_evidence = outputs.get(
                        "class_evidence_no_evidence_saliency"
                    )
                    if not torch.is_tensor(saliency_evidence):
                        raise ValueError(
                            "saliency rank/consistency supervision requires "
                            "outputs[class_evidence_no_evidence_saliency] "
                            "(model.class_evidence.anti_erasure_no_evidence: "
                            "true)"
                        )
                    if saliency_evidence.ndim != 4:
                        raise ValueError(
                            "class_evidence_no_evidence_saliency must be "
                            f"[B,T,C,3], got {tuple(saliency_evidence.shape)}"
                        )
                    # [B, T, C, 3] -> [B, T, C]: the two score-gap channels
                    # (max_gap, topk_gap) measure evidence concentration; the
                    # std channel depends on absolute score scale and is less
                    # discriminative.
                    saliency_gap = saliency_evidence[..., :2].mean(dim=-1)
                    explicit_clean_masks = [
                        tuple(meta.get("online_clean_negative_mask", ()) or ())
                        for meta in batch["meta"]
                    ]
                    has_explicit_clean_masks = any(
                        len(mask) == len(LABELS)
                        for mask in explicit_clean_masks
                    )
                    if has_explicit_clean_masks:
                        if not all(
                            len(mask) == len(LABELS)
                            for mask in explicit_clean_masks
                        ):
                            raise ValueError(
                                "online_clean_negative_mask must be present "
                                "for every row in an online-simulation batch"
                            )
                        saliency_clean_flags = torch.tensor(
                            explicit_clean_masks,
                            dtype=torch.bool,
                            device=device,
                        ) & (label_masks > 0)
                    else:
                        saliency_clean_flags = torch.tensor(
                            [
                                bool(meta.get("is_negative", False))
                                and str(
                                    meta.get("sample_id", "")
                                ).startswith("neg_")
                                for meta in batch["meta"]
                            ],
                            dtype=torch.bool,
                            device=device,
                        ) & (label_masks > 0).all(dim=1)
                    if saliency_rank_weight > 0:
                        saliency_rank_loss, saliency_rank_components = (
                            bag_saliency_rank_loss(
                                saliency_gap,
                                batch.get("frame_targets"),
                                batch.get("frame_target_masks"),
                                saliency_clean_flags,
                                labels=LABELS,
                                margin=float(
                                    cfg.train.get(
                                        "class_evidence_saliency_rank_margin",
                                        0.10,
                                    )
                                ),
                                topk=int(
                                    cfg.train.get(
                                        "class_evidence_saliency_rank_topk", 3
                                    )
                                ),
                                support_threshold=float(
                                    cfg.train.get(
                                        "class_evidence_saliency_rank_support_threshold",
                                        0.1,
                                    )
                                ),
                                pairs_per_positive=int(
                                    cfg.train.get(
                                        "class_evidence_saliency_rank_pairs_per_positive",
                                        2,
                                    )
                                ),
                            )
                        )
                        loss = loss + saliency_rank_weight * saliency_rank_loss
                        step_components.update(saliency_rank_components)
                        step_components[
                            "weighted_class_evidence_saliency_rank_loss"
                        ] = float(
                            (
                                saliency_rank_weight
                                * saliency_rank_loss.detach()
                            ).cpu()
                        )
                    if cross_window_weight > 0:
                        cross_window_loss, cross_window_components = (
                            cross_window_saliency_consistency_loss(
                                saliency_gap,
                                batch["meta"],
                                clip_duration=float(
                                    cfg.video.get("clip_duration", 10.0)
                                ),
                                num_frames=int(
                                    cfg.video.get("num_frames", 24)
                                ),
                                tolerance_sec=float(
                                    cfg.train.get(
                                        "class_evidence_cross_window_tolerance_sec",
                                        0.6,
                                    )
                                ),
                                labels=LABELS,
                            )
                        )
                        loss = loss + cross_window_weight * cross_window_loss
                        step_components.update(cross_window_components)
                        step_components[
                            "weighted_class_evidence_cross_window_loss"
                        ] = float(
                            (cross_window_weight * cross_window_loss.detach()).cpu()
                        )
                signed_causal_weight = float(
                    cfg.train.get("class_evidence_signed_causal_loss_weight", 0.0)
                    or 0.0
                )
                if signed_causal_weight > 0:
                    if (
                        not isinstance(outputs, dict)
                        or not torch.is_tensor(
                            outputs.get("class_evidence_counterfactual_logits")
                        )
                        or not torch.is_tensor(
                            outputs.get("class_evidence_causal_full_logits")
                        )
                    ):
                        raise ValueError(
                            "signed causal local evidence supervision requires "
                            "outputs[class_evidence_counterfactual_logits]"
                        )
                    explicit_causal_masks = [
                        tuple(meta.get("online_clean_negative_mask", ()) or ())
                        for meta in batch["meta"]
                    ]
                    if not all(
                        len(mask) == len(LABELS)
                        for mask in explicit_causal_masks
                    ):
                        raise ValueError(
                            "signed causal local evidence requires a per-class "
                            "online_clean_negative_mask [B,C] for every row"
                        )
                    causal_clean_mask = torch.tensor(
                        explicit_causal_masks, dtype=torch.bool, device=device
                    ) & (label_masks > 0)
                    signed_causal_loss, signed_causal_components = (
                        signed_causal_local_evidence_loss(
                            outputs["class_evidence_causal_full_logits"],
                            outputs["class_evidence_counterfactual_logits"],
                            targets,
                            weighted_label_masks,
                            causal_clean_mask,
                            labels=LABELS,
                            positive_margin=float(
                                cfg.train.get(
                                    "class_evidence_signed_causal_positive_margin",
                                    0.15,
                                )
                            ),
                            negative_margin=float(
                                cfg.train.get(
                                    "class_evidence_signed_causal_negative_margin",
                                    0.10,
                                )
                            ),
                            negative_weight=float(
                                cfg.train.get(
                                    "class_evidence_signed_causal_negative_weight",
                                    0.5,
                                )
                            ),
                            temperature=float(
                                cfg.train.get(
                                    "class_evidence_signed_causal_temperature",
                                    0.5,
                                )
                            ),
                        )
                    )
                    loss = loss + signed_causal_weight * signed_causal_loss
                    step_components.update(signed_causal_components)
                    step_components[
                        "weighted_class_evidence_signed_causal_loss"
                    ] = float(
                        (signed_causal_weight * signed_causal_loss.detach()).cpu()
                    )
                retention_weight = float(
                    cfg.train.get("positive_retention_loss_weight", 0.0) or 0.0
                )
                if isinstance(outputs, dict) and retention_weight > 0:
                    retention_loss = positive_retention_loss(
                        outputs,
                        targets,
                        label_masks,
                        margin=float(
                            cfg.train.get("positive_retention_margin", 0.0)
                            or 0.0
                        ),
                        branch=str(
                            cfg.train.get(
                                "positive_retention_branch", "fused"
                            )
                        ),
                    )
                    loss = loss + retention_weight * retention_loss
                    step_components["positive_retention_loss"] = float(
                        retention_loss.detach().cpu()
                    )
                teacher_guard_weight = float(
                    cfg.train.get("negative_teacher_guard_loss_weight", 0.0)
                    or 0.0
                )
                if isinstance(outputs, dict) and teacher_guard_weight > 0:
                    guard_label_names = cfg.train.get(
                        "negative_teacher_guard_labels", LABELS
                    )
                    if isinstance(guard_label_names, str):
                        guard_label_names = [
                            value.strip()
                            for value in guard_label_names.split(",")
                            if value.strip()
                        ]
                    unknown_guard_labels = [
                        label for label in guard_label_names if label not in LABELS
                    ]
                    if unknown_guard_labels:
                        raise ValueError(
                            "Unknown train.negative_teacher_guard_labels: "
                            f"{unknown_guard_labels}; labels={LABELS}"
                        )
                    teacher_guard_loss, teacher_guard_violating = (
                        negative_teacher_guard_loss(
                            outputs,
                            targets,
                            online_negative_masks,
                            label_indices=[
                                LABELS.index(label) for label in guard_label_names
                            ],
                            margin=float(
                                cfg.train.get(
                                    "negative_teacher_guard_margin", 0.0
                                )
                                or 0.0
                            ),
                            branch=str(
                                cfg.train.get(
                                    "negative_teacher_guard_branch", "fused"
                                )
                            ),
                        )
                    )
                    loss = loss + teacher_guard_weight * teacher_guard_loss
                    step_components["negative_teacher_guard_loss"] = float(
                        teacher_guard_loss.detach().cpu()
                    )
                    step_components["negative_teacher_guard_violating"] = float(
                        teacher_guard_violating
                    )
                temporal_gate_quality_weight = float(
                    cfg.train.get("temporal_gate_quality_loss_weight", 0.0)
                    or 0.0
                )
                if (
                    isinstance(outputs, dict)
                    and temporal_gate_quality_weight > 0
                ):
                    gate_quality_loss, gate_quality_components = (
                        temporal_gate_quality_loss(
                            outputs,
                            targets,
                            label_masks,
                            temperature=float(
                                cfg.train.get(
                                    "temporal_gate_quality_temperature", 0.25
                                )
                                or 0.25
                            ),
                        )
                    )
                    loss = loss + (
                        temporal_gate_quality_weight * gate_quality_loss
                    )
                    step_components.update(gate_quality_components)
                roi_feature_weight = float(
                    cfg.train.get("roi_feature_loss_weight", 0.0) or 0.0
                )
                if (
                    isinstance(outputs, dict)
                    and "roi_feature_logits" in outputs
                    and roi_feature_weight > 0
                ):
                    roi_feature_matrix = criterion(
                        outputs["roi_feature_logits"], targets
                    )
                    roi_feature_loss = (
                        roi_feature_matrix * weighted_label_masks
                    ).sum() / weighted_label_masks.sum().clamp_min(1.0)
                    loss = loss + roi_feature_weight * roi_feature_loss
                    step_components["roi_feature_loss"] = float(
                        roi_feature_loss.detach().cpu()
                    )
                quality_weight = float(cfg.train.get("roi_quality_loss_weight", 0.0) or 0.0)
                if isinstance(outputs, dict) and quality_weight > 0:
                    quality_loss, quality_components = roi_quality_supervision_loss(
                        outputs, batch, cfg, device
                    )
                    loss = loss + quality_weight * quality_loss
                    step_components.update(quality_components)
                if isinstance(outputs, dict):
                    for output_key, component_key in (
                        ("spatial_context_query_scale", "spatial_context_query_scale"),
                        ("spatial_motion_query_scale", "spatial_motion_query_scale"),
                    ):
                        if output_key in outputs:
                            step_components[component_key] = float(
                                outputs[output_key].detach().float().mean().cpu()
                            )
                    if "spatial_fusion_gate" in outputs:
                        mean_fusion_gate = outputs["spatial_fusion_gate"].detach().float().mean(dim=0)
                        for label_index, label in enumerate(LABELS):
                            step_components[f"spatial_fusion_gate_{label}"] = float(
                                mean_fusion_gate[label_index].cpu()
                            )
                    if "spatial_fusion_delta" in outputs:
                        mean_fusion_delta = (
                            outputs["spatial_fusion_delta"]
                            .detach()
                            .float()
                            .mean(dim=0)
                        )
                        for label_index, label in enumerate(LABELS):
                            step_components[f"spatial_fusion_delta_{label}"] = float(
                                mean_fusion_delta[label_index].cpu()
                            )
                    if "spatial_feature_delta" in outputs:
                        feature_delta = (
                            outputs["spatial_feature_delta"].detach().float()
                        )
                        step_components["spatial_feature_delta_norm"] = float(
                            feature_delta.norm(dim=-1).mean().cpu()
                        )
                        mean_abs_delta = feature_delta.abs().mean(dim=(0, 1, 3))
                        for label_index, label in enumerate(LABELS):
                            step_components[f"spatial_feature_delta_abs_{label}"] = float(
                                mean_abs_delta[label_index].cpu()
                            )
                if isinstance(outputs, dict):
                    for key in (
                        "temporal_difference_gate",
                        "temporal_difference_residual_abs_mean",
                        "temporal_difference_residual_norm",
                    ):
                        if key in outputs:
                            step_components[key] = float(
                                outputs[key].detach().float().mean().cpu()
                            )
                if (
                    isinstance(outputs, dict)
                    and "multi_layer_frame_residual" in outputs
                ):
                    residual_tokens = outputs["multi_layer_frame_residual"]
                    step_components["multi_layer_frame_residual_norm"] = float(
                        residual_tokens.detach().float().norm(dim=-1).mean().cpu()
                    )
                spatial_clip_weight = float(
                    cfg.train.get("spatial_clip_loss_weight", 0.0) or 0.0
                )
                if (
                    isinstance(outputs, dict)
                    and "spatial_clip_logits" in outputs
                    and spatial_clip_weight > 0
                ):
                    spatial_clip_matrix = F.binary_cross_entropy_with_logits(
                        outputs["spatial_clip_logits"],
                        targets.to(outputs["spatial_clip_logits"].dtype),
                        reduction="none",
                    )
                    spatial_clip_loss, spatial_clip_pos, spatial_clip_neg = (
                        per_class_equal_pos_neg_loss(
                            spatial_clip_matrix, targets, weighted_label_masks
                        )
                    )
                    loss = loss + spatial_clip_weight * spatial_clip_loss
                    step_components["spatial_clip_loss"] = float(
                        spatial_clip_loss.detach().cpu()
                    )
                    step_components["spatial_clip_pos_loss"] = float(
                        spatial_clip_pos.detach().cpu()
                    )
                    step_components["spatial_clip_neg_loss"] = float(
                        spatial_clip_neg.detach().cpu()
                    )
                spatial_loss_enabled = any(
                    float(cfg.train.get(key, 0.0) or 0.0) > 0
                    for key in (
                        "spatial_attention_mil_loss_weight",
                        "spatial_attention_query_diversity_loss_weight",
                        "spatial_attention_concentration_loss_weight",
                        "spatial_attention_overlap_loss_weight",
                    )
                )
                if isinstance(outputs, dict) and spatial_loss_enabled:
                    spatial_loss, spatial_components = (
                        spatial_attention_auxiliary_loss(
                            outputs,
                            targets,
                            weighted_label_masks,
                            cfg,
                        )
                    )
                    loss = loss + spatial_loss
                    step_components.update(spatial_components)
                spatial_token_loss_enabled = any(
                    float(cfg.train.get(key, 0.0) or 0.0) > 0
                    for key in (
                        "spatial_token_mil_loss_weight",
                        "spatial_token_query_diversity_loss_weight",
                        "spatial_token_overlap_loss_weight",
                    )
                )
                if isinstance(outputs, dict) and spatial_token_loss_enabled:
                    spatial_token_loss, spatial_token_components = (
                        spatial_token_pooling_auxiliary_loss(
                            outputs,
                            targets,
                            weighted_label_masks,
                            cfg,
                        )
                    )
                    loss = loss + spatial_token_loss
                    step_components.update(spatial_token_components)
                spatial_temporal_weight = float(
                    cfg.train.get(
                        "spatial_temporal_localization_loss_weight", 0.0
                    )
                    or 0.0
                )
                if (
                    isinstance(outputs, dict)
                    and spatial_temporal_weight > 0
                ):
                    temporal_loss, temporal_components = (
                        spatial_temporal_localization_loss(
                            outputs, batch, cfg, device
                        )
                    )
                    loss = loss + spatial_temporal_weight * temporal_loss
                    step_components.update(temporal_components)
                structured_frame_temporal_weight = float(
                    cfg.train.get(
                        "structured_frame_temporal_localization_loss_weight",
                        0.0,
                    )
                    or 0.0
                )
                if (
                    isinstance(outputs, dict)
                    and structured_frame_temporal_weight > 0
                ):
                    if "structured_frame_event_logits" not in outputs:
                        raise ValueError(
                            "structured frame temporal supervision requires "
                            "outputs['structured_frame_event_logits']"
                        )
                    structured_frame_loss, structured_frame_components = (
                        spatial_temporal_localization_loss(
                            outputs,
                            batch,
                            cfg,
                            device,
                            logits_key="structured_frame_event_logits",
                            component_prefix="structured_frame_temporal",
                        )
                    )
                    loss = (
                        loss
                        + structured_frame_temporal_weight
                        * structured_frame_loss
                    )
                    step_components.update(structured_frame_components)
                counterfactual_weight = float(
                    cfg.train.get("spatial_counterfactual_loss_weight", 0.0)
                    or 0.0
                )
                if (
                    isinstance(outputs, dict)
                    and counterfactual_weight > 0
                ):
                    counterfactual_loss, counterfactual_components = (
                        spatial_counterfactual_causal_loss(
                            outputs,
                            targets,
                            weighted_label_masks,
                            cfg,
                        )
                    )
                    loss = loss + counterfactual_weight * counterfactual_loss
                    step_components.update(counterfactual_components)
                frame_weight = float(cfg.train.get("frame_det_loss_weight", 0.0) or 0.0)
                if isinstance(outputs, dict) and frame_weight > 0:
                    frame_loss, frame_components = frame_detection_loss(outputs, batch, cfg, device)
                    loss = loss + frame_weight * frame_loss
                    step_components.update(frame_components)
                    step_components["weighted_frame_loss"] = float(
                        (frame_weight * frame_loss.detach()).cpu()
                    )
                frame_tail_rank_weight = float(
                    cfg.train.get("frame_tail_rank_loss_weight", 0.0) or 0.0
                )
                if isinstance(outputs, dict) and frame_tail_rank_weight > 0:
                    if (
                        "frame_event_logits" not in outputs
                        or "frame_targets" not in batch
                        or "frame_target_masks" not in batch
                    ):
                        raise ValueError(
                            "train.frame_tail_rank_loss_weight>0 requires "
                            "frame_event_logits plus batch frame_targets/"
                            "frame_target_masks (frame supervision must be enabled)"
                        )
                    frame_tail_rank_label_names = cfg.train.get(
                        "frame_tail_rank_labels", LABELS
                    )
                    if isinstance(frame_tail_rank_label_names, str):
                        frame_tail_rank_label_names = [
                            value.strip()
                            for value in frame_tail_rank_label_names.split(",")
                            if value.strip()
                        ]
                    unknown_frame_tail_rank_labels = [
                        label
                        for label in frame_tail_rank_label_names
                        if label not in LABELS
                    ]
                    if unknown_frame_tail_rank_labels:
                        raise ValueError(
                            "Unknown train.frame_tail_rank_labels: "
                            f"{unknown_frame_tail_rank_labels}; labels={LABELS}"
                        )
                    frame_tail_rank_ramp_epochs = float(
                        cfg.train.get("frame_tail_rank_ramp_epochs", 1.0) or 0.0
                    )
                    if frame_tail_rank_ramp_epochs > 0:
                        epoch_progress = (epoch - start_epoch) + (
                            (step - 1) / max(int(final_micro_step), 1)
                        )
                        frame_tail_rank_ramp = min(
                            1.0,
                            max(epoch_progress, 0.0) / frame_tail_rank_ramp_epochs,
                        )
                    else:
                        frame_tail_rank_ramp = 1.0
                    frame_tail_loss, frame_tail_components = frame_tail_rank_loss(
                        outputs["frame_event_logits"],
                        batch["frame_targets"].to(device, non_blocking=True),
                        batch["frame_target_masks"].to(device, non_blocking=True),
                        batch["label_masks"].to(device, non_blocking=True),
                        margin=float(cfg.train.get("frame_tail_rank_margin", 1.0)),
                        tau=float(cfg.train.get("frame_tail_rank_tau", 0.5)),
                        beta=float(cfg.train.get("frame_tail_rank_beta", 1.0)),
                        neg_threshold=float(
                            cfg.train.get("frame_tail_rank_neg_threshold", 0.0)
                        ),
                        label_indices=[
                            LABELS.index(label)
                            for label in frame_tail_rank_label_names
                        ],
                    )
                    ramped_frame_tail_rank_weight = (
                        frame_tail_rank_weight * frame_tail_rank_ramp
                    )
                    if ramped_frame_tail_rank_weight > 0:
                        loss = loss + ramped_frame_tail_rank_weight * frame_tail_loss
                    step_components["frame_tail_rank_loss"] = float(
                        frame_tail_loss.detach().cpu()
                    )
                    step_components["weighted_frame_tail_rank_loss"] = float(
                        (ramped_frame_tail_rank_weight * frame_tail_loss.detach()).cpu()
                    )
                    step_components["frame_tail_rank_ramp"] = float(
                        frame_tail_rank_ramp
                    )
                    step_components.update(frame_tail_components)
                step_components.update(online_object_teacher_components)
                object_loss: Tensor | None = None
                object_heatmap_weight = float(
                    cfg.train.get("object_heatmap_loss_weight", 0.0) or 0.0
                )
                if isinstance(outputs, dict) and object_heatmap_weight > 0:
                    object_loss, object_components = object_teacher_heatmap_loss(
                        outputs, batch, cfg, device
                    )
                    loss = loss + object_heatmap_weight * object_loss
                    step_components.update(object_components)
                    step_components["weighted_object_heatmap_loss"] = float(
                        (object_heatmap_weight * object_loss.detach()).cpu()
                    )
                relation_loss: Tensor | None = None
                relation_weight = float(
                    cfg.train.get("ball_goal_relation_loss_weight", 0.0) or 0.0
                )
                if isinstance(outputs, dict) and relation_weight > 0:
                    relation_loss, relation_components = ball_goal_relation_aux_loss(
                        outputs, batch, cfg, device
                    )
                    loss = loss + relation_weight * relation_loss
                    step_components.update(relation_components)
                    step_components["weighted_ball_goal_relation_loss"] = float(
                        (relation_weight * relation_loss.detach()).cpu()
                    )
                if ema_retention_weight > 0 and teacher_logits is not None:
                    retention_label_names = cfg.train.get(
                        "ema_positive_retention_labels", LABELS
                    )
                    unknown_retention_labels = [
                        label
                        for label in retention_label_names
                        if label not in LABELS
                    ]
                    if unknown_retention_labels:
                        raise ValueError(
                            "Unknown train.ema_positive_retention_labels: "
                            f"{unknown_retention_labels}; labels={LABELS}"
                        )
                    retention_label_indices = [
                        LABELS.index(label) for label in retention_label_names
                    ]
                    retention_loss, retention_components = (
                        ema_positive_retention_loss(
                            logits,
                            teacher_logits,
                            targets,
                            label_masks,
                            labels=LABELS,
                            margin=float(
                                cfg.train.get(
                                    "ema_positive_retention_margin", 0.05
                                )
                            ),
                            temperature=float(
                                cfg.train.get(
                                    "ema_positive_retention_temperature", 0.5
                                )
                            ),
                            label_indices=retention_label_indices,
                        )
                    )
                    loss = loss + ema_retention_weight * retention_loss
                    step_components.update(retention_components)
                    step_components["weighted_ema_retention_loss"] = float(
                        (ema_retention_weight * retention_loss.detach()).cpu()
                    )
                if fixed_teacher_guard_weight > 0 and teacher_logits is not None:
                    guard_label_names = cfg.train.get(
                        "fixed_teacher_logit_guard_labels", LABELS
                    )
                    if isinstance(guard_label_names, str):
                        guard_label_names = [
                            value.strip()
                            for value in guard_label_names.split(",")
                            if value.strip()
                        ]
                    unknown_guard_labels = [
                        label for label in guard_label_names if label not in LABELS
                    ]
                    if unknown_guard_labels:
                        raise ValueError(
                            "Unknown train.fixed_teacher_logit_guard_labels: "
                            f"{unknown_guard_labels}; labels={LABELS}"
                        )
                    teacher_guard_loss, teacher_guard_components = (
                        fixed_teacher_logit_guard_loss(
                            logits,
                            teacher_logits,
                            targets,
                            label_masks,
                            labels=LABELS,
                            positive_margin=float(
                                cfg.train.get(
                                    "fixed_teacher_positive_margin", 0.0
                                )
                            ),
                            negative_margin=float(
                                cfg.train.get(
                                    "fixed_teacher_negative_margin", 0.0
                                )
                            ),
                            label_indices=[
                                LABELS.index(label) for label in guard_label_names
                            ],
                        )
                    )
                    loss = loss + fixed_teacher_guard_weight * teacher_guard_loss
                    step_components.update(teacher_guard_components)
                    step_components["weighted_fixed_teacher_logit_guard_loss"] = float(
                        (fixed_teacher_guard_weight * teacher_guard_loss.detach()).cpu()
                    )
                pair_consistency_cfg = cfg.train.get(
                    "online_pair_consistency", ConfigDict()
                )
                if (
                    isinstance(outputs, dict)
                    and bool(pair_consistency_cfg.get("enabled", False))
                ):
                    pair_consistency_loss, pair_consistency_components = (
                        online_pair_consistency_loss(
                            outputs,
                            batch,
                            pair_consistency_cfg,
                            epoch=epoch,
                            step=step,
                            steps_per_epoch=final_micro_step,
                        )
                    )
                    loss = loss + pair_consistency_loss
                    step_components.update(pair_consistency_components)
                diagnostic_aux_loss = (
                    relation_loss if relation_loss is not None else object_loss
                )
                diagnostic_aux_weight = (
                    relation_weight
                    if relation_loss is not None
                    else object_heatmap_weight
                )
                diagnostic_valid = (
                    step_components.get("goal_teacher_frame_fraction", 0.0)
                    if relation_loss is not None
                    else step_components.get("object_teacher_valid_fraction", 0.0)
                )
                if (
                    diagnostic_aux_loss is not None
                    and float(diagnostic_valid) > 0.0
                    and should_measure_gradient_alignment(
                        mechanism_a_cfg,
                        step=step,
                        measured=mechanism_a_gradient_measurements,
                    )
                ):
                    diagnostics_cfg = mechanism_a_cfg.get(
                        "gradient_diagnostics", ConfigDict()
                    )
                    alignment_metrics = gradient_alignment(
                        raw_clip_loss,
                        diagnostic_aux_loss,
                        mechanism_a_shared_parameters,
                        object_weight=diagnostic_aux_weight,
                        require_nonzero=bool(
                            diagnostics_cfg.get("require_nonzero", True)
                        ),
                    )
                    step_components.update(alignment_metrics)
                    alignment_record: dict[str, float | int] = {
                        "epoch": int(epoch),
                        "step": int(step),
                        **alignment_metrics,
                    }
                    mechanism_a_gradient_records.append(alignment_record)
                    mechanism_a_gradient_measurements += 1
                    if is_main_process:
                        print(
                            "mechanism_a_gradient_alignment "
                            + json.dumps(alignment_record, sort_keys=True),
                            flush=True,
                        )
                    max_gradient_measurements = max(
                        int(
                            diagnostics_cfg.get(
                                "max_measurements_per_epoch", 4
                            )
                        ),
                        1,
                    )
                    if (
                        bool(
                            diagnostics_cfg.get(
                                "exit_after_max_measurements", False
                            )
                        )
                        and mechanism_a_gradient_measurements
                        >= max_gradient_measurements
                    ):
                        if is_main_process:
                            save_json(
                                output_dir / "mechanism_a_gradient_probe.json",
                                {
                                    "status": "complete",
                                    "object_heatmap_loss_weight": object_heatmap_weight,
                                    "ball_goal_relation_loss_weight": relation_weight,
                                    "diagnostic_auxiliary": "relation" if relation_loss is not None else "object_heatmap",
                                    "measurements": mechanism_a_gradient_records,
                                },
                            )
                            print(
                                "mechanism_a_gradient_probe_complete "
                                f"measurements={mechanism_a_gradient_measurements} "
                                f"output={output_dir / 'mechanism_a_gradient_probe.json'}",
                                flush=True,
                            )
                        return
                loss = loss / int(cfg.train.grad_accum_steps)
            if not bool(torch.isfinite(loss.detach()).all()):
                nonfinite_components = [
                    key
                    for key, value in step_components.items()
                    if not math.isfinite(float(value))
                ]
                raise FloatingPointError(
                    f"Non-finite training loss at epoch={epoch} step={step}; "
                    f"components={nonfinite_components}"
                )
            scaler.scale(loss).backward()
            if step % int(cfg.train.grad_accum_steps) == 0:
                scaler.unscale_(optimizer)
                if structured_frame_temporal_weight > 0:
                    frame_delta_grad_sq = 0.0
                    frame_gate_grad_sq = 0.0
                    for name, parameter in model.named_parameters():
                        if parameter.grad is None:
                            continue
                        normalized_name = name.removeprefix("module.")
                        grad_sq = float(parameter.grad.detach().float().square().sum().cpu())
                        if normalized_name.startswith("spatial_feature_structured.frame_delta_head."):
                            frame_delta_grad_sq += grad_sq
                        elif normalized_name.startswith("spatial_feature_structured.frame_gate_head."):
                            frame_gate_grad_sq += grad_sq
                    step_components["structured_frame_delta_grad_norm"] = math.sqrt(frame_delta_grad_sq)
                    step_components["structured_frame_gate_grad_norm"] = math.sqrt(frame_gate_grad_sq)
                latest_grad_norms.clear()
                for group in optimizer.param_groups:
                    group_name = str(group.get("name", "all"))
                    grad_sq = 0.0
                    for parameter in group["params"]:
                        if parameter.grad is None:
                            continue
                        grad_sq += float(
                            parameter.grad.detach().float().square().sum().cpu()
                        )
                    latest_grad_norms[f"grad_norm_{group_name}"] = math.sqrt(grad_sq)
                if evidence_health_enabled:
                    gate_grad_sq = 0.0
                    projection_grad_sq = 0.0
                    for name, parameter in model.named_parameters():
                        if parameter.grad is None:
                            continue
                        normalized_name = name.removeprefix("module.")
                        grad_sq = float(
                            parameter.grad.detach().float().square().sum().cpu()
                        )
                        if normalized_name == "class_evidence_head.evidence_gates":
                            gate_grad_sq += grad_sq
                        elif normalized_name.startswith(
                            "class_evidence_head.evidence_projs."
                        ):
                            projection_grad_sq += grad_sq
                    gate_grad_norm = math.sqrt(gate_grad_sq)
                    projection_grad_norm = math.sqrt(projection_grad_sq)
                    evidence_health_max_gate_grad = max(
                        evidence_health_max_gate_grad, gate_grad_norm
                    )
                    evidence_health_max_projection_grad = max(
                        evidence_health_max_projection_grad,
                        projection_grad_norm,
                    )
                    latest_grad_norms[
                        "grad_norm_external_evidence_gate"
                    ] = gate_grad_norm
                    latest_grad_norms[
                        "grad_norm_external_evidence_projection"
                    ] = projection_grad_norm
                if float(cfg.train.grad_clip_norm) > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip_norm))
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                if (
                    evidence_health_enabled
                    and not evidence_health_checked
                    and epoch == 1
                    and step >= int(evidence_health_cfg.get("check_step", 40))
                ):
                    evidence_head = getattr(
                        unwrap_model(model), "class_evidence_head", None
                    )
                    gate_parameter = getattr(
                        evidence_head, "evidence_gates", None
                    )
                    if gate_parameter is None:
                        raise RuntimeError(
                            "external evidence health check found no evidence_gates"
                        )
                    gate_abs_max = float(
                        torch.tanh(gate_parameter.detach()).abs().max().cpu()
                    )
                    failures = []
                    if evidence_health_max_valid_fraction < float(
                        evidence_health_cfg.get("min_valid_fraction", 1e-3)
                    ):
                        failures.append(
                            "evidence_valid_fraction="
                            f"{evidence_health_max_valid_fraction:.3e}"
                        )
                    if evidence_health_max_gate_grad <= float(
                        evidence_health_cfg.get("min_gate_grad_norm", 1e-10)
                    ):
                        failures.append(
                            "gate_grad_norm="
                            f"{evidence_health_max_gate_grad:.3e}"
                        )
                    if evidence_health_max_projection_grad <= float(
                        evidence_health_cfg.get(
                            "min_projection_grad_norm", 1e-10
                        )
                    ):
                        failures.append(
                            "projection_grad_norm="
                            f"{evidence_health_max_projection_grad:.3e}"
                        )
                    if gate_abs_max < float(
                        evidence_health_cfg.get("min_gate_abs_max", 1e-5)
                    ):
                        failures.append(f"gate_abs_max={gate_abs_max:.3e}")
                    if failures:
                        raise RuntimeError(
                            "external evidence health assertion failed at "
                            f"epoch={epoch} step={step}: "
                            + ", ".join(failures)
                        )
                    evidence_health_checked = True
                    print(
                        "external_evidence_health=PASS "
                        f"step={step} valid_fraction_max="
                        f"{evidence_health_max_valid_fraction:.6f} "
                        f"gate_grad_max={evidence_health_max_gate_grad:.6e} "
                        f"projection_grad_max="
                        f"{evidence_health_max_projection_grad:.6e} "
                        f"gate_abs_max={gate_abs_max:.6e}",
                        flush=True,
                    )
                if ema is not None:
                    ema.update(model)
                optimizer.zero_grad(set_to_none=True)

            step_loss_value = float(loss.detach().cpu()) * int(cfg.train.grad_accum_steps)
            running_loss += step_loss_value
            recent_loss += step_loss_value
            for key, value in step_components.items():
                scalar_value = float(value)
                loss_component_totals[key] = loss_component_totals.get(key, 0.0) + scalar_value
                recent_loss_component_totals[key] = (
                    recent_loss_component_totals.get(key, 0.0) + scalar_value
                )
            num_batches += 1
            recent_batches += 1
            step_time = time.time() - iter_end
            data_time_total += data_time
            step_time_total += step_time
            recent_data_time_total += data_time
            recent_step_time_total += step_time
            iter_end = time.time()
            if step % int(cfg.train.log_interval) == 0:
                log_payload = {
                    "running_loss": running_loss,
                    "num_batches": num_batches,
                    "recent_loss": recent_loss,
                    "recent_batches": recent_batches,
                    "data_time_total": data_time_total,
                    "step_time_total": step_time_total,
                    "recent_data_time_total": recent_data_time_total,
                    "recent_step_time_total": recent_step_time_total,
                    "loss_component_totals": loss_component_totals,
                    "recent_loss_component_totals": recent_loss_component_totals,
                    "hard_negative_samples": hard_negative_samples,
                    "observed_samples": observed_samples,
                    "hard_negative_loss_total": hard_negative_loss_total,
                    "hard_negative_label_slots": hard_negative_label_slots,
                }
                rank_payloads = gather_objects_to_main(log_payload)
                if is_main_process and rank_payloads is not None:
                    global_batches = sum(int(item["num_batches"]) for item in rank_payloads)
                    global_recent_batches = sum(
                        int(item["recent_batches"]) for item in rank_payloads
                    )
                    global_components = sum_float_mappings(
                        item["loss_component_totals"] for item in rank_payloads
                    )
                    global_recent_components = sum_float_mappings(
                        item["recent_loss_component_totals"]
                        for item in rank_payloads
                    )
                    component_values = {
                        key: value / max(global_batches, 1)
                        for key, value in sorted(global_components.items())
                    }
                    add_slot_weighted_no_evidence_metrics(component_values)
                    global_hard_samples = sum(
                        int(item["hard_negative_samples"]) for item in rank_payloads
                    )
                    global_observed_samples = sum(
                        int(item["observed_samples"]) for item in rank_payloads
                    )
                    component_values["hard_negative_sample_fraction"] = (
                        global_hard_samples / max(global_observed_samples, 1)
                    )
                    global_hard_slots = sum(
                        float(item["hard_negative_label_slots"])
                        for item in rank_payloads
                    )
                    if global_hard_slots > 0:
                        component_values["hard_negative_clip_loss"] = (
                            sum(
                                float(item["hard_negative_loss_total"])
                                for item in rank_payloads
                            )
                            / global_hard_slots
                        )
                    recent_unprefixed_components = {
                        key: value / max(global_recent_batches, 1)
                        for key, value in sorted(global_recent_components.items())
                    }
                    add_slot_weighted_no_evidence_metrics(
                        recent_unprefixed_components
                    )
                    recent_component_values = {
                        f"recent_{key}": value
                        for key, value in recent_unprefixed_components.items()
                    }
                    for key, value in sorted(latest_grad_norms.items()):
                        component_values[key] = value
                    component_text = " ".join(
                        f"{key}={value:.5f}"
                        for key, value in sorted(component_values.items())
                    )
                    recent_component_text = " ".join(
                        f"{key}={value:.5f}"
                        for key, value in sorted(recent_component_values.items())
                    )
                    global_running_loss = sum(
                        float(item["running_loss"]) for item in rank_payloads
                    )
                    global_recent_loss = sum(
                        float(item["recent_loss"]) for item in rank_payloads
                    )
                    print(
                        f"epoch={epoch} step={step}/{len(train_loader)} ranks={len(rank_payloads)} "
                        f"loss={global_running_loss / max(global_batches, 1):.5f} "
                        f"recent_loss={global_recent_loss / max(global_recent_batches, 1):.5f} "
                        f"data_time={sum(float(item['data_time_total']) for item in rank_payloads) / max(global_batches, 1):.3f}s "
                        f"recent_data_time={sum(float(item['recent_data_time_total']) for item in rank_payloads) / max(global_recent_batches, 1):.3f}s "
                        f"step_time={sum(float(item['step_time_total']) for item in rank_payloads) / max(global_batches, 1):.3f}s "
                        f"recent_step_time={sum(float(item['recent_step_time_total']) for item in rank_payloads) / max(global_recent_batches, 1):.3f}s "
                        f"{component_text} {recent_component_text}",
                        flush=True,
                    )
                recent_loss = 0.0
                recent_loss_component_totals.clear()
                recent_batches = 0
                recent_data_time_total = 0.0
                recent_step_time_total = 0.0
            if max_steps_per_epoch > 0 and step >= max_steps_per_epoch:
                print(
                    f"max_steps_per_epoch reached epoch={epoch} step={step}/{len(train_loader)}",
                    flush=True,
                )
                break

        if num_batches % int(cfg.train.grad_accum_steps) != 0:
            if float(cfg.train.grad_clip_norm) > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if ema is not None:
                ema.update(model)
            optimizer.zero_grad(set_to_none=True)

        epoch_train_loss = running_loss / max(num_batches, 1)
        if distributed:
            train_stats = torch.tensor(
                [running_loss, float(num_batches)],
                dtype=torch.float64,
                device=device,
            )
            dist.all_reduce(train_stats, op=dist.ReduceOp.SUM)
            epoch_train_loss = float(
                (train_stats[0] / train_stats[1].clamp_min(1.0)).item()
            )
            epoch_payloads = gather_objects_to_main(
                {
                    "loss_component_totals": loss_component_totals,
                    "hard_negative_loss_total": hard_negative_loss_total,
                    "hard_negative_label_slots": hard_negative_label_slots,
                    "hard_negative_samples": hard_negative_samples,
                    "observed_samples": observed_samples,
                }
            )
            if is_main_process and epoch_payloads is not None:
                loss_component_totals = sum_float_mappings(
                    item["loss_component_totals"] for item in epoch_payloads
                )
                hard_negative_loss_total = sum(
                    float(item["hard_negative_loss_total"])
                    for item in epoch_payloads
                )
                hard_negative_label_slots = sum(
                    float(item["hard_negative_label_slots"])
                    for item in epoch_payloads
                )
                hard_negative_samples = sum(
                    int(item["hard_negative_samples"]) for item in epoch_payloads
                )
                observed_samples = sum(
                    int(item["observed_samples"]) for item in epoch_payloads
                )
                num_batches = int(train_stats[1].item())
            dist.barrier()

        if is_main_process and bool(
            cfg.train.get("save_pre_eval_recovery", True)
        ):
            default_threshold = float(cfg.get("eval", ConfigDict()).get("threshold", 0.5))
            recovery_metrics = {
                "thresholds": {label: default_threshold for label in LABELS}
            }
            recovery_payload = training_checkpoint_payload(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                recovery_metrics,
                cfg,
                best_macro_f1,
                ema_state=ema.state_dict() if ema is not None else None,
            )
            recovery_payload["pending_evaluation_epoch"] = epoch
            torch.save(recovery_payload, output_dir / "pre_eval_recovery.pt")
            print(
                f"saved pre-eval recovery checkpoint epoch={epoch} "
                f"path={output_dir / 'pre_eval_recovery.pt'}",
                flush=True,
            )
        ema_metrics: dict[str, Any] | None = None
        ema_selection_source = str(
            ema_cfg.get("selection_source", "ema" if ema is not None else "raw")
        ).strip().lower()
        if ema_selection_source not in {"raw", "ema"}:
            raise ValueError("train.ema.selection_source must be raw or ema")
        ema_eval_enabled = ema is not None and bool(ema_cfg.get("evaluate", True))
        evaluate_raw = (
            bool(ema_cfg.get("evaluate_raw", True))
            or not ema_eval_enabled
            or ema_selection_source == "raw"
        )
        raw_eval_sec = 0.0
        ema_eval_sec = 0.0
        external_eval_sec = 0.0
        external_audit_metrics: dict[str, Any] | None = None
        metrics: dict[str, Any] | None = None
        online_val_cache_enabled = bool(
            cfg.get("eval", ConfigDict())
            .get("online_validation", ConfigDict())
            .get("enabled", False)
        )
        if evaluate_raw:
            raw_eval_started = time.time()
            metrics = evaluate(
                unwrap_model(model), val_loader, cfg, device,
                online_cache_path=(
                    output_dir / f"val15_online_raw_epoch_{epoch:03d}.npz"
                    if online_val_cache_enabled else None
                ),
            )
            raw_eval_sec = time.time() - raw_eval_started
            if external_audit_loader is not None and (
                not ema_eval_enabled or ema_selection_source == "raw"
            ):
                external_started = time.time()
                external_audit_metrics = evaluate(
                    unwrap_model(model), external_audit_loader, cfg, device,
                    fixed_thresholds=(metrics["thresholds"] if is_main_process else None),
                    online_cache_path=(output_dir / f"external18_online_epoch_{epoch:03d}.npz"),
                )
                external_eval_sec = time.time() - external_started
        raw_metrics = metrics
        if ema_eval_enabled:
            ema_backup = ema.copy_to(model)
            ema_eval_started = time.time()
            try:
                ema_metrics = evaluate(
                    unwrap_model(model), val_loader, cfg, device,
                    online_cache_path=(
                        output_dir / f"val15_online_ema_epoch_{epoch:03d}.npz"
                        if online_val_cache_enabled else None
                    ),
                )
                if external_audit_loader is not None and ema_selection_source == "ema":
                    external_started = time.time()
                    external_audit_metrics = evaluate(
                        unwrap_model(model), external_audit_loader, cfg, device,
                        fixed_thresholds=(ema_metrics["thresholds"] if is_main_process else None),
                        online_cache_path=(output_dir / f"external18_online_epoch_{epoch:03d}.npz"),
                    )
                    external_eval_sec = time.time() - external_started
            finally:
                ema.restore(model, ema_backup)
            ema_eval_sec = time.time() - ema_eval_started
            if is_main_process:
                print(
                    f"epoch={epoch} ema_eval "
                    f"{format_metric_summary('tuned', ema_metrics['tuned'])}",
                    flush=True,
                )
                print_per_class_metrics(
                    f"epoch={epoch} ema_tuned", ema_metrics["tuned"]
                )
                online_val_event = ema_metrics.get("online_event")
                if online_val_event is not None:
                    print(
                        f"epoch={epoch} val15_online "
                        f"p={online_val_event['micro_precision']:.4f} "
                        f"r={online_val_event['micro_recall']:.4f} "
                        f"f1={online_val_event['micro_f1']:.4f} "
                        f"nms_capped_min={online_val_event['nms_capped_union_minutes']:.2f} "
                        f"participation={online_val_event['nms_capped_participation_ratio']:.4f}",
                        flush=True,
                    )
        if metrics is None:
            if ema_metrics is None:
                raise RuntimeError("Both raw and EMA validation were disabled")
            # Keep the historical checkpoint/metrics schema stable. When raw
            # evaluation is skipped, the primary metrics field contains EMA.
            metrics = ema_metrics
        if is_main_process:
            print(
                f"epoch={epoch} eval_timing raw_enabled={evaluate_raw} "
                f"raw_sec={raw_eval_sec:.2f} ema_sec={ema_eval_sec:.2f} "
                f"external_fixed_sec={external_eval_sec:.2f}",
                flush=True,
            )
            if external_audit_metrics is not None:
                external_summary = format_metric_summary(
                    "fixed", external_audit_metrics["tuned"]
                )
                print(
                    f"epoch={epoch} external18_fixed_val15 {external_summary}",
                    flush=True,
                )
                print_per_class_metrics(
                    f"epoch={epoch} external18_fixed_val15",
                    external_audit_metrics["tuned"],
                )
                online_event = external_audit_metrics.get("online_event")
                if online_event is not None:
                    print(
                        f"epoch={epoch} external18_online "
                        f"p={online_event['micro_precision']:.4f} "
                        f"r={online_event['micro_recall']:.4f} "
                        f"f1={online_event['micro_f1']:.4f} "
                        f"nms_capped_min={online_event['nms_capped_union_minutes']:.2f} "
                        f"participation={online_event['nms_capped_participation_ratio']:.4f}",
                        flush=True,
                    )
                    for label in LABELS:
                        item = online_event["per_class"][label]
                        print(
                            f"epoch={epoch} external18_online class={label} "
                            f"p={item['precision']:.4f} r={item['recall']:.4f} "
                            f"f1={item['f1']:.4f} tp={item['tp']} fp={item['fp']} "
                            f"fn={item['fn']}",
                            flush=True,
                        )
        if distributed and not is_main_process:
            stop_tensor = torch.zeros(1, dtype=torch.int32, device=device)
            dist.broadcast(stop_tensor, src=0)
            if int(stop_tensor.item()) != 0:
                break
            continue
        selection_metrics = (
            ema_metrics
            if ema_selection_source == "ema" and ema_metrics is not None
            else metrics
        )
        train_component_values = {
            key: value / max(num_batches, 1)
            for key, value in sorted(loss_component_totals.items())
        }
        add_slot_weighted_no_evidence_metrics(train_component_values)
        epoch_summary = {
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "train_loss_components": {
                **train_component_values,
                "hard_negative_sample_fraction": hard_negative_samples
                / max(observed_samples, 1),
                "hard_negative_clip_loss": hard_negative_loss_total
                / max(hard_negative_label_slots, 1.0),
            },
            "elapsed_sec": time.time() - start_time,
            "metrics": metrics,
        }
        if ema_metrics is not None:
            epoch_summary["ema_metrics"] = ema_metrics
        if external_audit_metrics is not None:
            epoch_summary["external18_fixed_val15"] = external_audit_metrics
            epoch_summary["combined33_fixed_val15"] = combine_fixed_threshold_metrics(
                selection_metrics["tuned"], external_audit_metrics["tuned"]
            )
        epoch_summary["validation_runtime"] = {
            "raw_enabled": bool(evaluate_raw),
            "raw_sec": float(raw_eval_sec),
            "ema_sec": float(ema_eval_sec),
            "external_fixed_sec": float(external_eval_sec),
            "primary_metrics_source": "raw" if evaluate_raw else "ema",
        }
        selection_mode = str(cfg.train.get("checkpoint_selection", "tuned_macro_f1"))
        selection_details: dict[str, Any] = {}
        if selection_mode == "default_precision_at_recall_floor":
            precision = float(selection_metrics["default"]["micro_precision"])
            recall = float(selection_metrics["default"]["micro_recall"])
            recall_floor = float(cfg.train.get("checkpoint_min_recall", 0.60))
            selection_score = precision if recall >= recall_floor else precision - 1.0 - (recall_floor - recall)
        elif selection_mode == "tuned_macro_precision_at_recall_floor":
            recall_floors = recall_floor_map_from_config(cfg)
            supported_labels = [
                label for label in LABELS
                if int(selection_metrics["tuned"]["per_class"][label].get("support", 0)) > 0
            ]
            shortfalls = [
                max(0.0, recall_floors[label] - float(selection_metrics["tuned"]["per_class"][label]["recall"]))
                for label in supported_labels
            ]
            precision = (
                sum(
                    float(selection_metrics["tuned"]["per_class"][label]["precision"])
                    for label in supported_labels
                ) / len(supported_labels)
                if supported_labels else 0.0
            )
            selection_score = (
                precision
                if max(shortfalls, default=0.0) <= 1e-12
                else precision - 1.0 - sum(shortfalls) / max(len(shortfalls), 1)
            )
            selection_details = {
                "supported_labels": supported_labels,
                "unsupported_labels": [label for label in LABELS if label not in supported_labels],
                "recall_shortfalls": {
                    label: shortfalls[index] for index, label in enumerate(supported_labels)
                },
            }
        elif selection_mode == "online_recall_floor_min_review":
            selection_score, selection_details = (
                online_recall_floor_min_review_selection_score(selection_metrics, cfg)
            )
        elif selection_mode == "stable_tuned_macro_precision_at_recall_floor":
            selection_score, selection_details = stable_tuned_macro_precision_selection_score(
                selection_metrics, cfg
            )
        elif selection_mode == "tuned_mixed_objective":
            selection_score, selection_details = tuned_mixed_objective_selection_score(
                selection_metrics, cfg
            )
        elif selection_mode == "tuned_macro_fbeta_at_recall_floor":
            recall_floors = recall_floor_map_from_config(cfg)
            shortfalls = [
                max(0.0, recall_floors[label] - float(selection_metrics["tuned"]["per_class"][label]["recall"]))
                for label in LABELS
            ]
            fbeta_score = float(selection_metrics["tuned"].get("macro_fbeta", selection_metrics["tuned"]["macro_f1"]))
            selection_score = (
                fbeta_score
                if max(shortfalls, default=0.0) <= 1e-12
                else fbeta_score - 1.0 - sum(shortfalls) / max(len(shortfalls), 1)
            )
            selection_details = {
                "fbeta_beta": float(selection_metrics["tuned"].get("fbeta_beta", 1.0)),
                "recall_shortfalls": {label: shortfalls[index] for index, label in enumerate(LABELS)},
            }
        elif selection_mode in ("mAP", "tuned_mAP"):
            selection_score = float(selection_metrics["tuned"]["mAP"])
        elif selection_mode == "default_micro_f1":
            selection_score = float(selection_metrics["default"]["micro_f1"])
        elif selection_mode == "default_micro_precision":
            selection_score = float(selection_metrics["default"]["micro_precision"])
        else:
            selection_score = float(selection_metrics["tuned"]["macro_f1"])
        epoch_summary["checkpoint_selection"] = {
            "mode": selection_mode,
            "score": selection_score,
            "source": ema_selection_source if selection_metrics is ema_metrics else "raw",
            "min_recall": float(cfg.train.get("checkpoint_min_recall", 0.0)),
            "details": selection_details,
        }
        early_stopping_score = (
            float(metrics["tuned"]["mAP"])
            if early_stopping_monitor == "tuned_mAP"
            else float(selection_score)
        )
        early_stopping_improved = (
            early_stopping_score
            > early_stopping_best + early_stopping_min_delta
        )
        if early_stopping_improved:
            early_stopping_best = early_stopping_score
            early_stopping_bad_epochs = 0
        elif epoch >= early_stopping_min_epoch:
            early_stopping_bad_epochs += 1
        epoch_summary["early_stopping"] = {
            "enabled": early_stopping_enabled,
            "monitor": early_stopping_monitor,
            "score": early_stopping_score,
            "best": early_stopping_best,
            "improved": early_stopping_improved,
            "bad_epochs": early_stopping_bad_epochs,
            "min_epoch": early_stopping_min_epoch,
            "patience": early_stopping_patience,
            "min_delta": early_stopping_min_delta,
        }
        save_json(output_dir / f"metrics_epoch_{epoch:03d}.json", epoch_summary)
        print(
            f"epoch={epoch} train_loss={epoch_summary['train_loss']:.5f} "
            f"{format_metric_summary('default', metrics['default'])} "
            f"{format_metric_summary('tuned', metrics['tuned'])} "
            f"selection={selection_mode}:{selection_score:.4f}",
            flush=True,
        )
        print_per_class_metrics(f"epoch={epoch} default", metrics["default"])
        print_per_class_metrics(f"epoch={epoch} tuned", metrics["tuned"])
        if "audit_common_recall_floor" in metrics:
            audit = metrics["audit_common_recall_floor"]
            print(
                f"epoch={epoch} audit_common_recall_floor={audit['floor']:.2f} "
                f"{format_metric_summary('tuned', audit['metrics'])}",
                flush=True,
            )
            print_per_class_metrics(
                f"epoch={epoch} audit_common_r{int(round(audit['floor'] * 100))}",
                audit["metrics"],
            )
        if "spatial_branches" in metrics:
            for branch_name, branch_metrics in metrics["spatial_branches"].items():
                print(
                    f"epoch={epoch} spatial_branch={branch_name} "
                    f"{format_metric_summary('tuned', branch_metrics['tuned'])}",
                    flush=True,
                )
                print_per_class_metrics(
                    f"epoch={epoch} spatial_branch={branch_name} tuned",
                    branch_metrics["tuned"],
                )
        if "temporal_branches" in metrics:
            for branch_name, branch_metrics in metrics["temporal_branches"].items():
                print(
                    f"epoch={epoch} temporal_branch={branch_name} "
                    f"{format_metric_summary('tuned', branch_metrics['tuned'])}",
                    flush=True,
                )
                print_per_class_metrics(
                    f"epoch={epoch} temporal_branch={branch_name} tuned",
                    branch_metrics["tuned"],
                )
        for label, item in metrics["confidence_separation"].items():
            if item["mean_gap"] is None:
                continue
            print(
                f"epoch={epoch} confidence_gap class={label} "
                f"positive_mean={item['positive_mean']:.4f} negative_mean={item['negative_mean']:.4f} "
                f"mean_gap={item['mean_gap']:.4f} positive_p10={item['positive_p10']:.4f} "
                f"negative_p90={item['negative_p90']:.4f} tail_gap={item['tail_gap']:.4f}",
                flush=True,
            )
        if "frame_topk_hit_rate" in metrics:
            for label, item in metrics["frame_topk_hit_rate"].items():
                print(
                    f"epoch={epoch} frame_topk class={label} "
                    f"hit_rate={item['hit_rate']:.4f} hit/total={item['hit']}/{item['total']}",
                    flush=True,
                )
        if "set_piece_subtype" in metrics:
            subtype_metrics = metrics["set_piece_subtype"]
            macro_ap = subtype_metrics.get("macro_ap")
            top1 = subtype_metrics.get("top1")
            print(
                f"epoch={epoch} set_piece_subtype "
                f"macro_ap={'nan' if macro_ap is None else f'{macro_ap:.4f}'} "
                f"top1={'nan' if top1 is None else f'{top1:.4f}'} "
                f"valid_rows={subtype_metrics['valid_rows']} "
                f"single_label_rows={subtype_metrics['single_label_rows']}",
                flush=True,
            )
            for subtype, item in subtype_metrics["per_subtype"].items():
                ap = item.get("ap")
                print(
                    f"epoch={epoch} set_piece_subtype class={subtype} "
                    f"ap={'nan' if ap is None else f'{ap:.4f}'} "
                    f"support={item['support']}",
                    flush=True,
                )
        if "temporal_gate_stats" in metrics:
            for label, item in metrics["temporal_gate_stats"].items():
                positive_text = (
                    "nan"
                    if item["positive_mean"] is None
                    else f"{item['positive_mean']:.4f}"
                )
                negative_text = (
                    "nan"
                    if item["negative_mean"] is None
                    else f"{item['negative_mean']:.4f}"
                )
                print(
                    f"epoch={epoch} temporal_gate class={label} "
                    f"mean={item['mean']:.4f} "
                    f"positive_mean={positive_text} "
                    f"negative_mean={negative_text} "
                    f"p10/p50/p90={item['p10']:.4f}/{item['p50']:.4f}/{item['p90']:.4f} "
                    f"event_majority_rate={item['event_majority_rate']:.4f}",
                    flush=True,
                )
        # Per-video rows remain in metrics/checkpoints for audit, but printing hundreds
        # of lines obscures the epoch-level signal in DDP training logs.
        if bool(cfg.train.get("print_per_video_metrics", False)):
            print_per_video_metrics(f"epoch={epoch} val_tuned", metrics["per_video_tuned"])

        is_best = selection_score > best_macro_f1
        if is_best:
            best_macro_f1 = selection_score

        checkpoint_payload = training_checkpoint_payload(
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            metrics,
            cfg,
            best_macro_f1,
            ema_state=ema.state_dict() if ema is not None else None,
        )
        if ema_metrics is not None:
            checkpoint_payload["ema_metrics"] = ema_metrics
        if external_audit_metrics is not None:
            checkpoint_payload["external18_fixed_val15"] = external_audit_metrics
            checkpoint_payload["combined33_fixed_val15"] = epoch_summary[
                "combined33_fixed_val15"
            ]
        checkpoint_payload["metrics_source"] = "raw" if raw_metrics is not None else "ema"
        torch.save(checkpoint_payload, output_dir / "last.pt")
        if bool(cfg.train.get("save_epoch_checkpoints", False)):
            torch.save(checkpoint_payload, output_dir / f"epoch_{epoch}.pt")

        if is_best:
            best_payload = checkpoint_payload
            if (
                ema is not None
                and ema_metrics is not None
                and ema_selection_source == "ema"
                and bool(ema_cfg.get("save_best_as_ema", True))
            ):
                ema_backup = ema.copy_to(model)
                try:
                    best_payload = training_checkpoint_payload(
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch,
                        ema_metrics,
                        cfg,
                        best_macro_f1,
                        ema_state=ema.state_dict(),
                    )
                    if raw_metrics is not None:
                        best_payload["raw_metrics"] = raw_metrics
                    else:
                        best_payload["raw_metrics_skipped"] = True
                    best_payload["checkpoint_selection_source"] = "ema"
                    if external_audit_metrics is not None:
                        best_payload["external18_fixed_val15"] = external_audit_metrics
                        best_payload["combined33_fixed_val15"] = epoch_summary[
                            "combined33_fixed_val15"
                        ]
                finally:
                    ema.restore(model, ema_backup)
            torch.save(best_payload, output_dir / "best.pt")
            save_json(output_dir / "best_metrics.json", epoch_summary)

        should_stop = (
            early_stopping_enabled
            and epoch >= early_stopping_min_epoch
            and early_stopping_bad_epochs >= early_stopping_patience
        )
        if should_stop:
            print(
                f"early_stopping epoch={epoch} monitor={early_stopping_monitor} "
                f"score={early_stopping_score:.6f} "
                f"best={early_stopping_best:.6f} "
                f"bad_epochs={early_stopping_bad_epochs} "
                f"patience={early_stopping_patience}",
                flush=True,
            )
        if distributed:
            stop_tensor = torch.tensor(
                [1 if should_stop else 0], dtype=torch.int32, device=device
            )
            dist.broadcast(stop_tensor, src=0)
        if should_stop:
            break


@torch.no_grad()
def build_feature_cache(cfg: Any, records_by_split: dict[str, list[ClipRecord]], device: torch.device) -> None:
    if data_mode(cfg) != "clips":
        raise ValueError("Feature cache is only supported for data.mode=clips because long_videos samples dynamic windows")
    cache_dir = Path(cfg.cache.dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    backbone = build_backbone(cfg).to(device)
    configure_backbone_trainability(backbone, freeze=True, finetune_last_blocks=0)
    backbone.eval()
    image_size = parse_image_size(cfg.video.image_size)

    for split, records in records_by_split.items():
        dataset = FootballClipDataset(
            records,
            num_frames=int(cfg.video.num_frames),
            image_size=image_size,
            is_train=False,
            hflip_prob=0.0,
        )
        loader = make_loader(dataset, cfg, is_train=False, batch_size=int(cfg.cache.batch_size))
        written = 0
        skipped = 0
        for batch in loader:
            inputs = batch["inputs"].to(device, non_blocking=True)
            bsz, frames, channels, height, width = inputs.shape
            missing_indices = []
            cache_paths = []
            for i, key in enumerate(batch["cache_key"]):
                path = cache_dir / key
                cache_paths.append(path)
                if path.exists() and not bool(cfg.cache.rebuild):
                    skipped += 1
                else:
                    missing_indices.append(i)
            if not missing_indices:
                continue

            selected = inputs[missing_indices].reshape(len(missing_indices) * frames, channels, height, width)
            with autocast_context(device, bool(cfg.train.amp), str(cfg.train.amp_dtype)):
                features = extract_dino_frame_features(backbone, selected).float()
            features = features.reshape(len(missing_indices), frames, -1).cpu()
            for out_i, batch_i in enumerate(missing_indices):
                path = cache_paths[batch_i]
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "features": features[out_i],
                        "label_schema": LABEL_SCHEMA,
                        "labels": LABELS,
                        "target": batch["targets"][batch_i],
                        "label_mask": batch["label_masks"][batch_i],
                        "meta": batch["meta"][batch_i],
                    },
                    path,
                )
                written += 1
        print(f"cache split={split} written={written} skipped={skipped} dir={cache_dir}", flush=True)


def build_online_validation_records(
    records: Sequence[LongVideoRecord],
    events_by_video: dict[tuple[str, str], list[FootballEvent]],
    *,
    clip_duration: float,
    stride_sec: float,
) -> list[LongVideoRecord]:
    """Build a deterministic full-video grid for strict online validation.

    Merely enabling eval.online_validation previously left the ordinary
    event-centered validation records untouched, so online_event was silently
    null and checkpoint selection fell back to clip metrics.  Grid records
    carry the complete per-video GT anchor set and are recognizable by the
    online_val_ prefix consumed by evaluate().
    """
    if clip_duration <= 0.0 or stride_sec <= 0.0:
        raise ValueError(
            "online validation clip_duration and stride_sec must be positive"
        )
    templates: dict[tuple[str, str], LongVideoRecord] = {}
    for record in records:
        templates.setdefault((record.source, record.video_id), record)
    dense_records: list[LongVideoRecord] = []
    for key in sorted(templates):
        template = templates[key]
        events = [event for event in events_by_video.get(key, ()) if not event.is_ignored]
        anchors = tuple(
            tuple(sorted({float(event.anchor_time) for event in events if event.labels[index] > 0.5}))
            for index in range(len(LABELS))
        )
        duration = max(float(template.video_duration), 0.0)
        max_start = max(duration - clip_duration, 0.0)
        starts = [
            min(index * stride_sec, max_start)
            for index in range(int(math.floor(max_start / stride_sec)) + 1)
        ]
        if not starts:
            starts = [0.0]
        if max_start - starts[-1] > 1e-6:
            starts.append(max_start)
        for window_index, start in enumerate(starts):
            end = min(start + clip_duration, duration)
            labels = labels_for_window(events, start, end)
            dense_records.append(
                LongVideoRecord(
                    source=template.source,
                    split=template.split,
                    video_id=template.video_id,
                    sample_id=f"online_val_{window_index:06d}",
                    video_path=template.video_path,
                    annotation_path=template.annotation_path,
                    anchor_time=start + 0.5 * clip_duration,
                    base_clip_start=start,
                    base_clip_end=start + clip_duration,
                    video_duration=duration,
                    is_negative=not any(value > 0.5 for value in labels),
                    labels=labels,
                    label_mask=tuple(1.0 for _ in LABELS),
                    focus_labels=tuple(0.0 for _ in LABELS),
                    online_chunk_id=f"{template.source}:{template.video_id}",
                    online_chunk_mode="validation_grid",
                    online_window_index=window_index,
                    online_gt_anchors=anchors,
                )
            )
    return dense_records


def prepare_datasets(cfg: Any, *, use_cache: bool) -> tuple[Dataset, Dataset, list[Record], list[Record]]:
    mode = data_mode(cfg)
    image_size = parse_image_size(cfg.video.image_size)
    normalize_on_cpu = not normalize_on_device_enabled(cfg)
    decode_strategy = str(cfg.data.get("video_decode_strategy", "multi_seek"))
    video_reader_cache_size = int(cfg.data.get("video_reader_cache_size", 0))
    dataset_num_frames = effective_num_frames(cfg)
    use_frame_supervision = frame_supervision_enabled(cfg)
    sigma_sec = frame_label_sigma_seconds(cfg)
    ignore_radius_sec = frame_label_ignore_radius_seconds(cfg)
    frame_time_jitter_sec = frame_label_time_jitter_seconds(cfg)
    if mode == "clips":
        train_records = load_clip_records(cfg.data.roots, "train")
        val_records = load_clip_records(cfg.data.roots, "val")
        if use_cache:
            train_dataset: Dataset = CachedFootballFeatureDataset(train_records, cfg.cache.dir)
            val_dataset: Dataset = CachedFootballFeatureDataset(val_records, cfg.cache.dir)
        else:
            train_dataset = FootballClipDataset(
                train_records,
                num_frames=dataset_num_frames,
                image_size=image_size,
                is_train=True,
                hflip_prob=float(cfg.video.hflip_prob),
                normalize_on_cpu=normalize_on_cpu,
                decode_strategy=decode_strategy,
                video_reader_cache_size=video_reader_cache_size,
            )
            val_dataset = FootballClipDataset(
                val_records,
                num_frames=dataset_num_frames,
                image_size=image_size,
                is_train=False,
                hflip_prob=0.0,
                normalize_on_cpu=normalize_on_cpu,
                decode_strategy=decode_strategy,
                video_reader_cache_size=video_reader_cache_size,
            )
        return train_dataset, val_dataset, train_records, val_records

    if mode != "long_videos":
        raise ValueError("data.mode must be 'clips' or 'long_videos'")
    if use_cache:
        raise ValueError("cache.enabled must be false for data.mode=long_videos")
    train_records, train_events = load_long_video_records(cfg, "train")
    val_records, val_events = load_long_video_records(cfg, "val")
    online_val_cfg = cfg.get("eval", ConfigDict()).get(
        "online_validation", ConfigDict()
    )
    if bool(online_val_cfg.get("enabled", False)):
        val_records = build_online_validation_records(
            val_records,
            val_events,
            clip_duration=float(cfg.video.get("clip_duration", 9.0)),
            stride_sec=float(online_val_cfg.get("window_stride_sec", 5.0)),
        )
        print(
            "online_validation_grid "
            f"videos={len({(record.source, record.video_id) for record in val_records})} "
            f"windows={len(val_records)} "
            f"clip_sec={float(cfg.video.get('clip_duration', 9.0)):.3f} "
            f"stride_sec={float(online_val_cfg.get('window_stride_sec', 5.0)):.3f}",
            flush=True,
        )
    robust_cropper = RobustClipCropper.from_config(cfg)
    detection_hint_renderer = DetectionHintRenderer.from_config(cfg)
    cropper: DetectorAwareCropper | RobustClipCropper | TopBandCropProvider | None = (
        robust_cropper
        or DetectorAwareCropper.from_config(cfg)
        or TopBandCropProvider.from_config(cfg)
    )
    spatial_cfg = cfg.get("spatial_crop", ConfigDict())
    highres_cfg = cfg.model.get("highres_glimpse", ConfigDict())
    highres_pool_frames = (
        int(highres_cfg.get("pool_frames", 32))
        if bool(highres_cfg.get("enabled", False))
        else 0
    )
    highres_pool_size = highres_cfg.get("pool_image_size", [720, 1280])
    detector_view_mode = str(spatial_cfg.get("output_view", "roi_only"))
    global_image_size = spatial_cfg.get("global_image_size", image_size)
    if detection_hint_renderer is not None:
        split_video_ids = {
            record.video_id for record in train_records + val_records
        }
        missing_hint_ids = sorted(
            video_id
            for video_id in split_video_ids
            if not detection_hint_renderer.has_video(video_id)
        )
        if missing_hint_ids:
            raise FileNotFoundError(
                f"Missing detection hint indices for {len(missing_hint_ids)} split videos; "
                f"first={missing_hint_ids[:10]} index_root={detection_hint_renderer.index_root}"
            )
    if robust_cropper is not None and bool(spatial_cfg.get("require_index", True)):
        missing_by_split = {
            "train": sorted({
                record.video_id for record in train_records if not robust_cropper.has_video(record.video_id)
            }),
            "val": sorted({
                record.video_id for record in val_records if not robust_cropper.has_video(record.video_id)
            }),
        }
        missing_ids = sorted(set(missing_by_split["train"] + missing_by_split["val"]))
        if missing_ids and bool(spatial_cfg.get("drop_missing_index_videos", False)):
            train_missing = set(missing_by_split["train"])
            val_missing = set(missing_by_split["val"])
            train_records = [record for record in train_records if record.video_id not in train_missing]
            val_records = [record for record in val_records if record.video_id not in val_missing]
            train_keys = {(record.source, record.video_id) for record in train_records}
            val_keys = {(record.source, record.video_id) for record in val_records}
            train_events = {key: events for key, events in train_events.items() if key in train_keys}
            val_events = {key: events for key, events in val_events.items() if key in val_keys}
            save_json(
                Path(cfg.output_dir) / "dropped_missing_roi_videos.json",
                {
                    "index_root": str(robust_cropper.index_root),
                    "train": missing_by_split["train"],
                    "val": missing_by_split["val"],
                },
            )
            for split_name, split_missing, split_records in (
                ("train", missing_by_split["train"], train_records),
                ("val", missing_by_split["val"], val_records),
            ):
                if split_missing:
                    print(
                        f"Dropped videos without ROI index split={split_name} count={len(split_missing)} "
                        f"ids={split_missing} remaining_clips={len(split_records)}",
                        flush=True,
                    )
            if not train_records or not val_records:
                raise RuntimeError(
                    "Dropping videos without ROI indices left an empty train or val dataset"
                )
        elif missing_ids:
            raise FileNotFoundError(
                f"Missing robust ROI indices for {len(missing_ids)} split videos; first={missing_ids[:10]} "
                f"index_root={robust_cropper.index_root}"
            )
    train_negative_margin, train_negative_label_margins = (
        resolve_negative_safety_margins(cfg.data.long_video, "train")
    )
    val_negative_margin, val_negative_label_margins = (
        resolve_negative_safety_margins(cfg.data.long_video, "val")
    )
    raw_supervision_cfg = cfg.get("raw_set_piece_supervision", ConfigDict())
    train_evidence_provider = EvidenceProvider.from_config(cfg, is_train=True)
    val_evidence_provider = EvidenceProvider.from_config(cfg, is_train=False)
    train_object_teacher_provider = object_teacher_provider_from_config(cfg)
    val_object_teacher_provider = object_teacher_provider_from_config(cfg)
    train_dataset = FootballLongVideoDataset(
        train_records,
        train_events,
        num_frames=dataset_num_frames,
        image_size=image_size,
        clip_duration=float(cfg.video.get("clip_duration", 9.0)),
        sampling_duration=float(
            cfg.video.get("sampling_duration", cfg.video.get("clip_duration", 9.0))
        ),
        sampling_temporal_jitter_sec=float(
            cfg.video.get("sampling_temporal_jitter_sec", 0.0)
        ),
        event_margin=float(cfg.video.get("event_margin", 0.5)),
        temporal_jitter_sec=float(cfg.video.get("temporal_jitter_sec", 1.0)),
        hard_negative_temporal_jitter_sec=float(
            cfg.data.long_video.get("hard_negative", ConfigDict()).get(
                "temporal_jitter_sec", cfg.video.get("temporal_jitter_sec", 1.0)
            )
        ),
        is_train=True,
        hflip_prob=float(cfg.video.hflip_prob),
        crop_provider=cropper,
        detector_view_mode=detector_view_mode,
        global_image_size=global_image_size,
        dual_sampling=str(cfg.video.get("dual_sampling", "aligned")),
        roi_overlap_frames=int(cfg.video.get("roi_overlap_frames", 8)),
        num_rois=int(spatial_cfg.get("num_rois", 1)),
        roi_noise_cfg=spatial_cfg.get("noise_augmentation", ConfigDict()),
        normalize_on_cpu=normalize_on_cpu,
        decode_strategy=decode_strategy,
        video_reader_cache_size=video_reader_cache_size,
        frame_supervision=use_frame_supervision,
        frame_label_sigma_sec=sigma_sec,
        frame_label_ignore_radius_sec=ignore_radius_sec,
        frame_label_time_jitter_sec=frame_time_jitter_sec,
        online_hard_negative_safety_margin_sec=float(
            cfg.train.get("online_hard_negative_safety_margin_sec", 0.0) or 0.0
        ),
        negative_safety_margin_sec=train_negative_margin,
        negative_label_safety_margin_sec=train_negative_label_margins,
        detection_hint_renderer=detection_hint_renderer,
        highres_pool_frames=highres_pool_frames,
        highres_pool_size=highres_pool_size,
        positive_window_strategy=str(cfg.video.get("positive_window_strategy", "legacy_random")),
        positive_anchor_min_sec=float(
            cfg.video.get("positive_anchor_min_sec", cfg.video.get("dense_positive_min_anchor_sec", cfg.video.get("event_margin", 0.5)))
        ),
        positive_anchor_max_sec=float(
            cfg.video.get(
                "positive_anchor_max_sec",
                cfg.video.get("dense_positive_max_anchor_sec", float(cfg.video.get("clip_duration", 9.0)) - float(cfg.video.get("event_margin", 0.5))),
            )
        ),
        decode_max_attempts_train=int(
            cfg.data.get("decode_max_attempts_train", 4)
        ),
        decode_same_record_retries=int(
            cfg.data.get("decode_same_record_retries", 1)
        ),
        decode_attempt_timeout_sec=float(
            cfg.data.get("decode_attempt_timeout_sec", 60.0)
        ),
        raw_rejected_ignore_margin_sec=(
            float(raw_supervision_cfg.get("rejected_ignore_margin_sec", 0.0) or 0.0)
            if bool(raw_supervision_cfg.get("rejected_ignore_enabled", False))
            else 0.0
        ),
        raw_context_span_min_duration_sec=float(
            raw_supervision_cfg.get("context_span_min_duration_sec", 0.25) or 0.25
        ),
        evidence_provider=train_evidence_provider,
        object_teacher_provider=train_object_teacher_provider,
    )
    val_dataset = FootballLongVideoDataset(
        val_records,
        val_events,
        num_frames=dataset_num_frames,
        image_size=image_size,
        clip_duration=float(cfg.video.get("clip_duration", 9.0)),
        sampling_duration=float(
            cfg.video.get("sampling_duration", cfg.video.get("clip_duration", 9.0))
        ),
        sampling_temporal_jitter_sec=0.0,
        event_margin=float(cfg.video.get("event_margin", 0.5)),
        temporal_jitter_sec=0.0,
        hard_negative_temporal_jitter_sec=0.0,
        is_train=False,
        hflip_prob=0.0,
        crop_provider=cropper,
        detector_view_mode=detector_view_mode,
        global_image_size=global_image_size,
        dual_sampling=str(cfg.video.get("dual_sampling", "aligned")),
        roi_overlap_frames=int(cfg.video.get("roi_overlap_frames", 8)),
        num_rois=int(spatial_cfg.get("num_rois", 1)),
        roi_noise_cfg=ConfigDict(),
        normalize_on_cpu=normalize_on_cpu,
        decode_strategy=decode_strategy,
        video_reader_cache_size=video_reader_cache_size,
        frame_supervision=use_frame_supervision,
        frame_label_sigma_sec=sigma_sec,
        frame_label_ignore_radius_sec=ignore_radius_sec,
        negative_safety_margin_sec=val_negative_margin,
        negative_label_safety_margin_sec=val_negative_label_margins,
        detection_hint_renderer=detection_hint_renderer,
        highres_pool_frames=highres_pool_frames,
        highres_pool_size=highres_pool_size,
        positive_window_strategy="centered",
        positive_anchor_min_sec=float(cfg.video.get("event_margin", 0.5)),
        positive_anchor_max_sec=float(float(cfg.video.get("clip_duration", 9.0)) - float(cfg.video.get("event_margin", 0.5))),
        raw_rejected_ignore_margin_sec=0.0,
        decode_attempt_timeout_sec=float(
            cfg.data.get("decode_attempt_timeout_sec", 60.0)
        ),
        raw_context_span_min_duration_sec=float(
            raw_supervision_cfg.get("context_span_min_duration_sec", 0.25) or 0.25
        ),
        evidence_provider=val_evidence_provider,
        object_teacher_provider=val_object_teacher_provider,
    )
    return train_dataset, val_dataset, train_records, val_records


def prepare_external_audit_dataset(
    cfg: Any,
) -> tuple[Dataset | None, list[Record]]:
    """Build an untouched audit split without allowing it to tune thresholds."""
    audit_cfg = cfg.get("eval", ConfigDict()).get(
        "external_audit", ConfigDict()
    )
    if not bool(audit_cfg.get("enabled", False)):
        return None, []
    if data_mode(cfg) != "long_videos":
        raise ValueError("eval.external_audit currently requires data.mode=long_videos")
    split = str(audit_cfg.get("split", "external"))
    records, events = load_long_video_records(cfg, split)
    image_size = parse_image_size(cfg.video.image_size)
    normalize_on_cpu = not normalize_on_device_enabled(cfg)
    spatial_cfg = cfg.get("spatial_crop", ConfigDict())
    robust_cropper = RobustClipCropper.from_config(cfg)
    cropper: DetectorAwareCropper | RobustClipCropper | TopBandCropProvider | None = (
        robust_cropper
        or DetectorAwareCropper.from_config(cfg)
        or TopBandCropProvider.from_config(cfg)
    )
    detection_hint_renderer = DetectionHintRenderer.from_config(cfg)
    highres_cfg = cfg.model.get("highres_glimpse", ConfigDict())
    negative_margin, negative_label_margins = resolve_negative_safety_margins(
        cfg.data.long_video, split
    )
    raw_supervision_cfg = cfg.get("raw_set_piece_supervision", ConfigDict())
    dataset = FootballLongVideoDataset(
        records,
        events,
        num_frames=effective_num_frames(cfg),
        image_size=image_size,
        clip_duration=float(cfg.video.get("clip_duration", 9.0)),
        sampling_duration=float(
            cfg.video.get("sampling_duration", cfg.video.get("clip_duration", 9.0))
        ),
        sampling_temporal_jitter_sec=0.0,
        event_margin=float(cfg.video.get("event_margin", 0.5)),
        temporal_jitter_sec=0.0,
        hard_negative_temporal_jitter_sec=0.0,
        is_train=False,
        hflip_prob=0.0,
        crop_provider=cropper,
        detector_view_mode=str(spatial_cfg.get("output_view", "roi_only")),
        global_image_size=spatial_cfg.get("global_image_size", image_size),
        dual_sampling=str(cfg.video.get("dual_sampling", "aligned")),
        roi_overlap_frames=int(cfg.video.get("roi_overlap_frames", 8)),
        num_rois=int(spatial_cfg.get("num_rois", 1)),
        roi_noise_cfg=ConfigDict(),
        normalize_on_cpu=normalize_on_cpu,
        decode_strategy=str(cfg.data.get("video_decode_strategy", "multi_seek")),
        video_reader_cache_size=int(cfg.data.get("video_reader_cache_size", 0)),
        frame_supervision=frame_supervision_enabled(cfg),
        frame_label_sigma_sec=frame_label_sigma_seconds(cfg),
        frame_label_ignore_radius_sec=frame_label_ignore_radius_seconds(cfg),
        negative_safety_margin_sec=negative_margin,
        negative_label_safety_margin_sec=negative_label_margins,
        detection_hint_renderer=detection_hint_renderer,
        highres_pool_frames=(
            int(highres_cfg.get("pool_frames", 32))
            if bool(highres_cfg.get("enabled", False)) else 0
        ),
        highres_pool_size=highres_cfg.get("pool_image_size", [720, 1280]),
        positive_window_strategy="centered",
        positive_anchor_min_sec=float(cfg.video.get("event_margin", 0.5)),
        positive_anchor_max_sec=float(
            float(cfg.video.get("clip_duration", 9.0))
            - float(cfg.video.get("event_margin", 0.5))
        ),
        raw_rejected_ignore_margin_sec=0.0,
        raw_context_span_min_duration_sec=float(
            raw_supervision_cfg.get("context_span_min_duration_sec", 0.25) or 0.25
        ),
        evidence_provider=EvidenceProvider.from_config(cfg, is_train=False),
        object_teacher_provider=object_teacher_provider_from_config(cfg),
    )
    return dataset, records


def run_dry_run(cfg: Any) -> None:
    image_size = parse_image_size(cfg.video.image_size)
    dataset_num_frames = effective_num_frames(cfg)
    dry_num_frames = min(dataset_num_frames, 4)
    if data_mode(cfg) == "clips":
        train_records = load_clip_records(cfg.data.roots, "train")
        val_records = load_clip_records(cfg.data.roots, "val")
        dataset: Dataset = FootballClipDataset(
            train_records[: min(4, len(train_records))],
            num_frames=dry_num_frames,
            image_size=image_size,
            is_train=True,
            hflip_prob=0.0,
        )
    elif data_mode(cfg) == "long_videos":
        train_records, train_events = load_long_video_records(cfg, "train")
        val_records, _ = load_long_video_records(cfg, "val")
        dataset = FootballLongVideoDataset(
            train_records[: min(4, len(train_records))],
            train_events,
            num_frames=dry_num_frames,
            image_size=image_size,
            clip_duration=float(cfg.video.get("clip_duration", 9.0)),
            sampling_duration=float(
                cfg.video.get("sampling_duration", cfg.video.get("clip_duration", 9.0))
            ),
            sampling_temporal_jitter_sec=float(
                cfg.video.get("sampling_temporal_jitter_sec", 0.0)
            ),
            event_margin=float(cfg.video.get("event_margin", 0.5)),
            temporal_jitter_sec=float(cfg.video.get("temporal_jitter_sec", 1.0)),
            is_train=True,
            hflip_prob=0.0,
            hard_negative_temporal_jitter_sec=float(
                cfg.data.long_video.get("hard_negative", ConfigDict()).get(
                    "temporal_jitter_sec", cfg.video.get("temporal_jitter_sec", 1.0)
                )
            ),
            crop_provider=(
                RobustClipCropper.from_config(cfg)
                or DetectorAwareCropper.from_config(cfg)
                or TopBandCropProvider.from_config(cfg)
            ),
            detection_hint_renderer=DetectionHintRenderer.from_config(cfg),
            detector_view_mode=str(cfg.get("spatial_crop", ConfigDict()).get("output_view", "roi_only")),
            global_image_size=cfg.get("spatial_crop", ConfigDict()).get("global_image_size", image_size),
            dual_sampling=str(cfg.video.get("dual_sampling", "aligned")),
            roi_overlap_frames=int(cfg.video.get("roi_overlap_frames", 8)),
            num_rois=int(
                cfg.get("spatial_crop", ConfigDict()).get("num_rois", 1)
            ),
            roi_noise_cfg=ConfigDict(),
            frame_supervision=frame_supervision_enabled(cfg),
            frame_label_sigma_sec=frame_label_sigma_seconds(cfg),
            frame_label_ignore_radius_sec=frame_label_ignore_radius_seconds(cfg),
            negative_safety_margin_sec=resolve_negative_safety_margins(
                cfg.data.long_video, "train"
            )[0],
            negative_label_safety_margin_sec=resolve_negative_safety_margins(
                cfg.data.long_video, "train"
            )[1],
            positive_window_strategy=str(
                cfg.video.get("positive_window_strategy", "centered")
            ),
            positive_anchor_min_sec=float(
                cfg.video.get("dense_positive_min_anchor_sec", cfg.video.get("event_margin", 0.5))
            ),
            positive_anchor_max_sec=float(
                cfg.video.get(
                    "dense_positive_max_anchor_sec",
                    float(cfg.video.get("clip_duration", 9.0)) - float(cfg.video.get("event_margin", 0.5)),
                )
            ),
        )
    else:
        raise ValueError("data.mode must be 'clips' or 'long_videos'")
    print("train", json.dumps(summarize_records(train_records), indent=2, sort_keys=True, ensure_ascii=False))
    print("val", json.dumps(summarize_records(val_records), indent=2, sort_keys=True, ensure_ascii=False))
    item = dataset[0]
    print(
        json.dumps(
            {
                "sample_input_shape": list(item["inputs"].shape),
                "sample_roi_input_shape": (
                    list(item["roi_inputs"].shape)
                    if "roi_inputs" in item
                    else None
                ),
                "sample_roi_b_input_shape": (
                    list(item["roi_inputs_b"].shape)
                    if "roi_inputs_b" in item
                    else None
                ),
                "sample_frame_times": (
                    item["frame_times"].tolist()
                    if "frame_times" in item
                    else None
                ),
                "sample_local_frame_times": (
                    item["local_frame_times"].tolist()
                    if "local_frame_times" in item
                    else None
                ),
                "sample_local_frame_times_b": (
                    item["local_frame_times_b"].tolist()
                    if "local_frame_times_b" in item
                    else None
                ),
                "sample_frame_target_shape": (
                    list(item["frame_targets"].shape)
                    if "frame_targets" in item
                    else None
                ),
                "sample_local_frame_target_shape": (
                    list(item["local_frame_targets"].shape)
                    if "local_frame_targets" in item
                    else None
                ),
                "sample_target": item["targets"].tolist(),
                "sample_meta": item["meta"],
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DINOv3 football multi-label event classifier")
    parser.add_argument("--config", required=True, help="Path to a YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Validate data loading without loading DINO weights")
    parser.add_argument("--extract-features", action="store_true", help="Only build the frozen-DINO feature cache")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate the initialized checkpoint without training")
    parser.add_argument("--eval-output", default="", help="Optional JSON path for --eval-only metrics")
    parser.add_argument("overrides", nargs="*", help="dotlist overrides, e.g. train.epochs=1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    validate_mechanism_a_config(cfg)
    env_world_size = int(os.environ.get("WORLD_SIZE", "1") or 1)
    local_rank = int(os.environ.get("LOCAL_RANK", "0") or 0)
    if env_world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("torchrun DDP requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        cfg["device"] = f"cuda:{local_rank}"
        cfg["gpu_ids"] = list(range(env_world_size))
    rank = dist.get_rank() if distributed_training_active() else 0
    if (
        distributed_training_active()
        and rank != 0
        and bool(cfg.train.get("ddp_suppress_non_main_stdout", True))
    ):
        # Rank-specific tracebacks still go to stderr; ordinary duplicated
        # status output is suppressed so train_console.log remains readable.
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
        warnings.filterwarnings("ignore")
    configure_runtime_threads(cfg)
    configure_label_schema(cfg)
    seed_everything(
        int(cfg.get("seed", 42)) + rank,
        bool(cfg.get("deterministic", True)),
    )
    print(
        f"label_schema={LABEL_SCHEMA} labels={LABELS} "
        f"distributed={distributed_training_active()} rank={rank}/{env_world_size} "
        f"local_rank={local_rank}",
        flush=True,
    )
    fusion = canonical_temporal_fusion(str(cfg.model.get("temporal_fusion", "cls_transformer")))
    decoded_frames = effective_num_frames(cfg)
    candidate_frames = int(cfg.video.get("candidate_num_frames", cfg.video.num_frames))
    adaptive_sampling = fusion in {
        "event_topk_transformer",
        "event_anchor_transformer",
        "uniform_event_dual_transformer",
    }
    event_frames = int(cfg.model.get("event_topk", 0)) if adaptive_sampling else 0
    context_frames = int(cfg.model.get("context_frames", 0)) if adaptive_sampling else 0
    if fusion == "event_anchor_transformer":
        temporal_input_frames = min(decoded_frames, int(cfg.model.get("event_anchor_max_frames", cfg.video.num_frames)))
        anchor_counts = per_label_int_tuple(
            cfg.model.get("event_anchor_topk_per_class", {"shot": 2, "save": 2, "set_piece": 1}),
            default=0,
        )
        anchor_detail = (
            f" event_anchor_topk_per_class={dict(zip(LABELS, anchor_counts))} "
            f"event_anchor_offsets={list(cfg.model.get('event_anchor_offsets', [-2, -1, 0, 1, 2]))} "
            f"event_anchor_nms_radius={int(cfg.model.get('event_anchor_nms_radius', 2))}"
        )
    elif fusion == "uniform_event_dual_transformer":
        temporal_input_frames = min(
            decoded_frames, max(event_frames + context_frames, 1)
        )
        anchor_detail = (
            f" uniform_frames={int(cfg.model.get('uniform_frames', cfg.video.num_frames))}"
            f" gate_init={dict(zip(LABELS, per_label_float_tuple(cfg.model.get('uniform_event_gate_init', 0.25), default=0.25)))}"
        )
    else:
        temporal_input_frames = (
            min(decoded_frames, max(event_frames + context_frames, 1))
            if fusion == "event_topk_transformer"
            else decoded_frames
        )
        anchor_detail = ""
    print(
        "frame_sampling "
        f"fusion={fusion} configured_num_frames={int(cfg.video.num_frames)} "
        f"candidate_num_frames={candidate_frames} "
        f"decoded_frames={decoded_frames} event_topk={event_frames} "
        f"context_frames={context_frames} temporal_input_frames={temporal_input_frames}"
        f"{anchor_detail}",
        flush=True,
    )
    if args.dry_run:
        run_dry_run(cfg)
        return

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    topology = resolve_runtime_topology(cfg, device)
    if distributed_training_active():
        cfg.data.num_workers = int(topology["num_workers_per_gpu"])
    use_cache = (
        data_mode(cfg) == "clips"
        and bool(cfg.cache.enabled)
        and bool(cfg.model.freeze_backbone)
        and int(cfg.model.finetune_last_blocks) == 0
        and not bool(cfg.model.get("lora", ConfigDict()).get("enabled", False))
    )

    if data_mode(cfg) == "clips":
        train_records = load_clip_records(cfg.data.roots, "train")
        val_records = load_clip_records(cfg.data.roots, "val")
        print("train", summarize_records(train_records), flush=True)
        print("val", summarize_records(val_records), flush=True)
        if bool(cfg.cache.enabled) or args.extract_features:
            build_feature_cache(cfg, {"train": train_records, "val": val_records}, device)
            if args.extract_features:
                return
    else:
        if args.extract_features:
            raise ValueError("--extract-features is only supported for data.mode=clips")
        if bool(cfg.cache.enabled):
            raise ValueError("cache.enabled must be false for data.mode=long_videos")

    train_dataset, val_dataset, train_records, val_records = prepare_datasets(cfg, use_cache=use_cache)
    external_audit_dataset, external_audit_records = prepare_external_audit_dataset(cfg)
    print("train", summarize_records(train_records), flush=True)
    print("val", summarize_records(val_records), flush=True)
    if external_audit_records:
        print("external_audit", summarize_records(external_audit_records), flush=True)
    if distributed_training_active():
        train_batch_size = int(topology["train_per_gpu_batch_size"])
        eval_batch_size = int(topology["eval_per_gpu_batch_size"])
    else:
        train_batch_size = int(cfg.train.batch_size)
        eval_batch_size = int(cfg.eval.batch_size)
    train_loader = make_loader(
        train_dataset,
        cfg,
        is_train=True,
        batch_size=train_batch_size,
        distributed=distributed_training_active(),
    )
    val_loader = make_loader(
        val_dataset,
        cfg,
        is_train=False,
        batch_size=eval_batch_size,
        distributed=distributed_training_active(),
    )
    external_audit_loader = (
        make_loader(
            external_audit_dataset,
            cfg,
            is_train=False,
            batch_size=eval_batch_size,
            distributed=distributed_training_active(),
        )
        if external_audit_dataset is not None else None
    )
    model = make_model(cfg, use_cached_features=use_cache, device=device)
    gpu_ids = [int(gpu_id) for gpu_id in cfg.get("gpu_ids", [])]
    if distributed_training_active():
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=bool(
                cfg.train.get("ddp_find_unused_parameters", True)
            ),
        )
        print(
            f"Using DDP rank={rank}/{dist.get_world_size()} local_rank={local_rank}; "
            f"per_gpu_batch_size={train_batch_size} "
            f"effective_batch_size={topology["effective_batch_size"]}",
            flush=True,
        )
    elif len(gpu_ids) > 1:
        if device.type != "cuda":
            print(f"WARN: gpu_ids={gpu_ids} ignored because device={device}", flush=True)
        elif torch.cuda.device_count() < len(gpu_ids):
            raise RuntimeError(f"Requested gpu_ids={gpu_ids}, but only {torch.cuda.device_count()} CUDA device(s) are visible")
        else:
            model = nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])
            print(
                f"Using DataParallel on gpu_ids={gpu_ids}; "
                f"per_gpu_batch_size={cfg.runtime_topology.train_per_gpu_batch_size} "
                f"global_batch_size={cfg.train.batch_size}",
                flush=True,
            )
    if args.eval_only:
        # Every rank must enter evaluate(): the loader is distributed and the
        # function gathers rank-local payloads.  Only serialization is rank-0.
        eval_output_path = (
            Path(args.eval_output)
            if args.eval_output
            else Path(cfg.output_dir) / "eval_only_metrics.json"
        )
        metrics = evaluate(
            unwrap_model(model), val_loader, cfg, device,
            online_cache_path=eval_output_path.with_suffix(".predictions.npz"),
        )
        if rank == 0:
            output_path = eval_output_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(metrics, indent=2, sort_keys=True, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"eval_only_metrics={output_path}", flush=True)
        if distributed_training_active():
            dist.barrier()
            dist.destroy_process_group()
        return
    train(
        model, train_loader, val_loader, cfg, device, train_records,
        external_audit_loader=external_audit_loader,
    )
    if distributed_training_active():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
