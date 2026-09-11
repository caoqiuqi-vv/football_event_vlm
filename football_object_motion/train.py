#!/usr/bin/env python
"""Isolated training entry for Object Motion Evidence Adapter.

The legacy trainer remains untouched.  This entry installs narrow runtime
hooks for dataset decoding, model fusion, Teacher generation and auxiliary
losses, then delegates optimization/evaluation/checkpointing to the mature
football trainer.
"""

from __future__ import annotations

import json
import math
import os
import types
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Sampler

import train_football_events as base
import train_football_events_online_simulation_e16 as online_e16
from football_object_motion.ball_backbone import (
    ball_lora_enabled,
    ball_lora_modules,
    extract_intermediate_patch_layers,
    inject_ball_lora,
    validate_trainable_allowlist,
)
from football_object_motion.losses import object_motion_auxiliary_loss, reset_pair_residual_queues
from football_object_motion.sampling import (
    DensePairBatchSampler, annotate_stream_item, build_dense_training_records,
    resolve_stream_row,
)
from football_object_motion.offline_teacher import OfflineTrackedBallTeacher
from football_object_motion.model import (
    OBJECT_NAMES,
    ObjectMotionEvidenceAdapter,
    interpolate_motion_residual,
)
from football_object_motion.teacher import OnlineObjectMotionTeacher


_ORIGINAL_MAKE_MODEL = base.make_model
_ORIGINAL_PREPARE_DATASETS = base.prepare_datasets
_ORIGINAL_COLLATE = base.football_collate
_ORIGINAL_FORWARD_BATCH = base.forward_model_batch
_ORIGINAL_MAKE_LOADER = base.make_loader
_ORIGINAL_BUILD_OPTIMIZER = base.build_optimizer


def _motion_cfg(cfg: Any) -> Any:
    return cfg.model.get("object_motion", base.ConfigDict())


def full_window_segment_bounds(clip_start: float, clip_end: float) -> tuple[tuple[float, float], ...]:
    """Three overlapping four-second views spanning a ten-second window."""
    if float(clip_end) - float(clip_start) < 9.999:
        return ((float(clip_start), float(clip_end)),)
    return tuple((float(clip_start) + offset, float(clip_start) + offset + 4.0) for offset in (0.0, 3.0, 6.0))


def full_window_coverage(times: Tensor, clip_start: float, clip_end: float) -> float:
    """Fraction of the clip timeline bracketed by sampled absolute times."""
    duration = max(float(clip_end) - float(clip_start), 1e-6)
    return min(max((float(times.max()) - float(times.min())) / duration, 0.0), 1.0)


class SameVideoPairBatchSampler(Sampler[list[int]]):
    """Deterministic equal-length DDP shards of complete adjacent pairs."""

    def __init__(self, pairs: list[tuple[int, int]], *, rank: int = 0, world_size: int = 1, seed: int = 42) -> None:
        if not pairs:
            raise ValueError("relation pairing enabled but pairs=0")
        if world_size < 1 or rank < 0 or rank >= world_size:
            raise ValueError("invalid pair sampler rank/world_size")
        usable = (len(pairs) // world_size) * world_size
        if usable == 0:
            raise ValueError(f"pairs={len(pairs)} cannot provide one complete pair per rank for world_size={world_size}")
        self.pairs = list(pairs[:usable])
        self.rank, self.world_size, self.seed, self.epoch = rank, world_size, seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.pairs), generator=generator).tolist()
        local = order[self.rank::self.world_size]
        if len(local) != len(self):
            raise RuntimeError("pair sampler produced unequal DDP shard")
        for pair_index in local:
            positive, negative = self.pairs[pair_index]
            yield [int(positive), int(negative)]

    def __len__(self) -> int:
        return len(self.pairs) // self.world_size


class SameVideoPairQueueBatchSampler(Sampler[list[int]]):
    """Batch-one sampler that keeps each pair adjacent and flips order by epoch."""

    def __init__(self, pairs: list[tuple[int, int]], *, rank: int = 0, world_size: int = 1, seed: int = 42) -> None:
        if not pairs:
            raise ValueError("relation pairing enabled but pairs=0")
        usable = (len(pairs) // int(world_size)) * int(world_size)
        if usable == 0:
            raise ValueError("pair queue cannot provide a complete pair per rank")
        self.pairs = list(pairs[:usable])
        self.rank, self.world_size, self.seed, self.epoch = int(rank), int(world_size), int(seed), 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.pairs), generator=generator).tolist()
        for pair_index in order[self.rank :: self.world_size]:
            positive, negative = self.pairs[pair_index]
            current = (positive, negative) if self.epoch % 2 == 0 else (negative, positive)
            yield [int(current[0])]
            yield [int(current[1])]

    def __len__(self) -> int:
        return 2 * (len(self.pairs) // self.world_size)


def dense_protocol_prediction(protocol: str, clip_score: Tensor, frame_scores: Tensor, frame_times: Tensor, window_center: Tensor) -> tuple[Tensor, Tensor]:
    """Return semantically matched dense score and event time."""
    if protocol == "clip_center":
        return clip_score, window_center
    if protocol == "clip_x_frame_peak":
        peak_score, peak_index = frame_scores.max(dim=1)
        peak_time = frame_times.gather(1, peak_index.unsqueeze(1)).squeeze(1)
        return clip_score * peak_score, peak_time
    raise ValueError(f"unsupported dense protocol: {protocol}")


def _initialize_motion_adapter(
    adapter: ObjectMotionEvidenceAdapter, cfg: Any
) -> None:
    """Load a prior adapter, or bootstrap ball/goal localization safely.

    The base trainer loads the checkpoint before this isolated adapter exists.
    We therefore perform a second, weights-only pass over only the small tensors
    relevant to this branch.  New motion checkpoints resume exactly; the old
    two-object spatial head transfers LayerNorm plus ball/goal linear rows while
    leaving the new person channel and relation model freshly initialized.
    """
    checkpoint_path = str(cfg.model.get("init_checkpoint", "") or "")
    if not checkpoint_path:
        return
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    raw_state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not isinstance(raw_state, dict):
        raise ValueError(f"checkpoint has no tensor state: {checkpoint_path}")
    state = base.strip_module_prefix(raw_state)
    prefix = "object_motion_adapter."
    motion_state = {
        key[len(prefix) :]: value
        for key, value in state.items()
        if key.startswith(prefix)
    }
    if motion_state:
        missing, unexpected = adapter.load_experiment_state_dict(motion_state)
        print(
            f"Loaded object-motion adapter from {checkpoint_path}: "
            f"tensors={len(motion_state)} missing={len(missing)} "
            f"unexpected={len(unexpected)}",
            flush=True,
        )
        return

    source_prefix = "object_spatial_aux.heatmap_head."
    target_state = adapter.heatmap_head.state_dict()
    transferred: list[str] = []
    for suffix in ("0.weight", "0.bias"):
        source = state.get(source_prefix + suffix)
        if torch.is_tensor(source) and source.shape == target_state[suffix].shape:
            target_state[suffix].copy_(source)
            transferred.append(suffix)
    for suffix in ("1.weight", "1.bias"):
        source = state.get(source_prefix + suffix)
        target = target_state[suffix]
        if torch.is_tensor(source) and source.ndim == target.ndim:
            rows = min(int(source.shape[0]), 2, int(target.shape[0]))
            if rows > 0 and source.shape[1:] == target.shape[1:]:
                target[:rows].copy_(source[:rows])
                transferred.append(f"{suffix}[:{rows}]")
    adapter.heatmap_head.load_state_dict(target_state)
    legacy_norm_weight = state.get(source_prefix + "0.weight")
    legacy_norm_bias = state.get(source_prefix + "0.bias")
    legacy_linear_weight = state.get(source_prefix + "1.weight")
    legacy_linear_bias = state.get(source_prefix + "1.bias")
    if torch.is_tensor(legacy_linear_weight) and legacy_linear_weight.shape[0] >= 1:
        with torch.no_grad():
            for norm, head in zip(adapter.ball_layer_norms, adapter.ball_layer_heads):
                if torch.is_tensor(legacy_norm_weight) and legacy_norm_weight.shape == norm.weight.shape:
                    norm.weight.copy_(legacy_norm_weight)
                if torch.is_tensor(legacy_norm_bias) and legacy_norm_bias.shape == norm.bias.shape:
                    norm.bias.copy_(legacy_norm_bias)
                if legacy_linear_weight[0].shape == head.weight[0].shape:
                    head.weight[0].copy_(legacy_linear_weight[0])
                    if torch.is_tensor(legacy_linear_bias):
                        head.bias[0].copy_(legacy_linear_bias[0])
        transferred.append("legacy ball row -> four BallLoRA readouts")
    print(
        "Bootstrapped object-motion ball/goal heatmap from legacy spatial head: "
        + (", ".join(transferred) if transferred else "no compatible tensors"),
        flush=True,
    )


def _restore_ball_backbone_state(model: nn.Module, cfg: Any) -> None:
    """Restore a BallLoRA checkpoint after the overlay has been injected.

    The base model loader runs before ``inject_ball_lora`` and therefore cannot
    match wrapped keys such as ``attn.qkv.base.weight`` or the BallLoRA deltas.
    First-time runs use a legacy anchor and have no BallLoRA keys, so they are
    deliberately left untouched.
    """
    checkpoint_path = str(cfg.model.get("init_checkpoint", "") or "")
    if not checkpoint_path:
        return
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    raw_state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not isinstance(raw_state, dict):
        raise ValueError(f"checkpoint has no tensor state: {checkpoint_path}")
    state = base.strip_module_prefix(raw_state)
    ball_keys = [
        key
        for key in state
        if key.startswith("backbone.") and "ball_lora_" in key
    ]
    if not ball_keys:
        return
    target = model.state_dict()
    matched = {
        key: value
        for key, value in state.items()
        if key.startswith("backbone.")
        and key in target
        and torch.is_tensor(value)
        and value.shape == target[key].shape
    }
    missing_ball = [key for key in ball_keys if key not in matched]
    if missing_ball:
        raise RuntimeError(
            "BallLoRA checkpoint is incompatible with the injected backbone: "
            f"missing={missing_ball[:8]}"
        )
    model.load_state_dict(matched, strict=False)
    print(
        f"Restored BallLoRA backbone from {checkpoint_path}: "
        f"tensors={len(matched)} ball_tensors={len(ball_keys)}",
        flush=True,
    )


def _load_reviewed_negative_entries(paths: list[str]) -> dict[tuple[str, str, int, float], dict[str, Any]]:
    reviewed: dict[tuple[str, str, int, float], dict[str, Any]] = {}
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"reviewed negative manifest missing: {path}")
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict) or not isinstance(payload.get("hard_negatives"), list):
            raise ValueError(f"invalid reviewed negative manifest schema: {path}")
        for item in payload["hard_negatives"]:
            source, video_id = str(item.get("source", "")), str(item.get("video_id", ""))
            center = float(item.get("center_sec", 0.5 * (float(item.get("start_sec", 0.0)) + float(item.get("end_sec", 0.0)))))
            for label in item.get("labels", ()):
                if str(label) in base.LABEL_TO_INDEX:
                    key = (source, video_id, base.LABEL_TO_INDEX[str(label)], round(center, 3))
                    reviewed[key] = {"manifest": str(path), "item": item}
    return reviewed


def _usable_relation_negative(dataset: Dataset, record: Any, class_index: int) -> bool:
    """Match relation-pair filtering to runtime dynamic label/mask semantics."""
    events = getattr(dataset, "events_by_video", {}).get(
        (record.source, record.video_id), ()
    )
    start = float(record.base_clip_start)
    end = float(record.base_clip_end)
    labels = base.labels_for_window(events, start, end)
    masks = base.label_mask_for_sample(record.label_mask, labels)
    masks = base.rejected_set_piece_label_mask(
        events,
        start,
        end,
        labels,
        masks,
        float(getattr(dataset, "raw_rejected_ignore_margin_sec", 0.0)),
    )
    weights = tuple(record.online_clip_loss_weights or ())
    online_weight = float(weights[class_index]) if weights else 1.0
    return (
        float(labels[class_index]) <= 0.5
        and float(masks[class_index]) > 0.0
        and online_weight > 0.0
    )


def build_relation_pair_rows(dataset: Dataset, cfg: Any, *, allow_empty: bool = False) -> list[dict[str, Any]]:
    motion_cfg = _motion_cfg(cfg)
    paths = [str(value) for value in motion_cfg.get("reviewed_negative_manifests", ())]
    if not paths:
        raise ValueError("require_true_pairs needs reviewed_negative_manifests")
    reviewed = _load_reviewed_negative_entries(paths)
    records = list(getattr(dataset, "records", ()))
    events_by_video = getattr(dataset, "events_by_video", {})
    positives: dict[tuple[str, str, int], list[int]] = {}
    for index, record in enumerate(records):
        if record.is_negative:
            continue
        for class_index, value in enumerate(record.labels):
            if float(value) > 0.5 and float(record.label_mask[class_index]) > 0:
                positives.setdefault((record.source, record.video_id, class_index), []).append(index)
    rows: list[dict[str, Any]] = []
    pair_count = 0
    matched_reviewed_keys: set[tuple[str, str, int, float]] = set()
    for negative_index, record in enumerate(records):
        if not record.is_negative:
            continue
        center = 0.5 * (float(record.base_clip_start) + float(record.base_clip_end))
        for class_index, mask_value in enumerate(record.label_mask):
            key = (record.source, record.video_id, class_index, round(center, 3))
            audit = reviewed.get(key)
            candidates = positives.get(key[:3], ())
            if audit is None or not candidates or float(mask_value) <= 0:
                continue
            gt_times = [float(event.anchor_time) for event in events_by_video.get(key[:2], ()) if not event.is_ignored and float(event.labels[class_index]) > 0.5]
            nearest_gap = min((abs(center - value) for value in gt_times), default=float("inf"))
            if nearest_gap < float(motion_cfg.get("pair_min_gap_sec", 5.0)):
                continue
            if not _usable_relation_negative(dataset, record, class_index):
                continue
            positive_index = candidates[pair_count % len(candidates)]
            pair_id = f"pair-{pair_count:08d}"
            common = {"pair_id": pair_id, "pair_label": base.LABELS[class_index], "relation_class_index": class_index}
            rows.append({"base_index": positive_index, "pair_role": "positive", "nearest_same_class_gt_gap": 0.0, **common})
            rows.append({"base_index": negative_index, "pair_role": "negative", "reviewed_negative": True, "full_clean_window": True, "nearest_same_class_gt_gap": nearest_gap, "review_manifest": audit["manifest"], **common})
            matched_reviewed_keys.add(key)
            pair_count += 1

    # E1.6 rebuilds its epoch records and can omit the base hard-negative rows.
    # Materialize only explicitly reviewed windows from a same-video record so
    # the production pair queue never silently falls back to random negatives.
    representatives: dict[tuple[str, str], int] = {}
    for index, record in enumerate(records):
        representatives.setdefault((record.source, record.video_id), index)
    mutable_records = getattr(dataset, "records", None)
    if not isinstance(mutable_records, list):
        raise TypeError("reviewed pair synthesis requires a mutable dataset.records list")
    for key, audit in reviewed.items():
        if key in matched_reviewed_keys:
            continue
        source, video_id, class_index, center = key
        candidates = positives.get((source, video_id, class_index), ())
        representative_index = representatives.get((source, video_id))
        if not candidates or representative_index is None:
            continue
        gt_times = [
            float(event.anchor_time)
            for event in events_by_video.get((source, video_id), ())
            if not event.is_ignored and float(event.labels[class_index]) > 0.5
        ]
        nearest_gap = min((abs(float(center) - value) for value in gt_times), default=float("inf"))
        if nearest_gap < float(motion_cfg.get("pair_min_gap_sec", 5.0)):
            continue
        item = audit["item"]
        representative = records[representative_index]
        clip_start = max(float(item.get("start_sec", float(center) - 5.0)), 0.0)
        clip_end = min(float(item.get("end_sec", clip_start + 10.0)), float(representative.video_duration))
        if clip_end - clip_start < 9.999:
            clip_start = max(min(float(center) - 5.0, float(representative.video_duration) - 10.0), 0.0)
            clip_end = min(clip_start + 10.0, float(representative.video_duration))
        class_mask = tuple(1.0 if index == class_index else 0.0 for index in range(len(base.LABELS)))
        synthetic = replace(
            representative,
            sample_id=f"reviewed_pair_{pair_count:08d}_{class_index}",
            anchor_time=float(center),
            base_clip_start=clip_start,
            base_clip_end=clip_end,
            is_negative=True,
            labels=tuple(0.0 for _ in base.LABELS),
            label_mask=class_mask,
            online_negative_kind="reviewed_hard_negative",
            online_clean_negative_mask=class_mask,
            online_clip_loss_weights=class_mask,
            online_frame_loss_weights=class_mask,
            online_label_loss_weights=class_mask,
            online_pair_id="",
            online_pair_role="",
            online_pair_class_mask=(),
        )
        if not _usable_relation_negative(dataset, synthetic, class_index):
            continue
        negative_index = len(mutable_records)
        mutable_records.append(synthetic)
        records.append(synthetic)
        positive_index = candidates[pair_count % len(candidates)]
        pair_id = f"pair-{pair_count:08d}"
        common = {"pair_id": pair_id, "pair_label": base.LABELS[class_index], "relation_class_index": class_index}
        rows.append({"base_index": positive_index, "pair_role": "positive", "nearest_same_class_gt_gap": 0.0, **common})
        rows.append({"base_index": negative_index, "pair_role": "negative", "reviewed_negative": True, "full_clean_window": True, "nearest_same_class_gt_gap": nearest_gap, "review_manifest": audit["manifest"], **common})
        pair_count += 1
    if not rows and not allow_empty:
        raise ValueError("relation pairing enabled but reviewed same-video pairs=0")
    return rows


class ObjectMotionDataset(Dataset):
    """Add an independently decoded short high-frame-rate full-image view."""

    def __init__(self, dataset: Dataset, cfg: Any) -> None:
        self.dataset = dataset
        self.records = getattr(dataset, "records", ())
        self.events_by_video = getattr(dataset, "events_by_video", {})
        motion_cfg = _motion_cfg(cfg)
        self.frames_per_segment = int(motion_cfg.get("frames_per_segment", 17))
        self.frames = self.frames_per_segment * 3
        self.duration = float(motion_cfg.get("duration_sec", 4.0))
        self.image_size = base.parse_image_size(
            motion_cfg.get("image_size", [512, 896])
        )
        self.teacher_image_size = base.parse_image_size(
            motion_cfg.get("teacher_image_size", self.image_size)
        )
        self.patch_size = int(
            motion_cfg.get(
                "heatmap_patch_size",
                motion_cfg.get("patch_size", 16),
            )
        )
        if self.image_size[0] % self.patch_size or self.image_size[1] % self.patch_size:
            raise ValueError(
                f"object motion image_size={self.image_size} must be divisible "
                f"by patch_size={self.patch_size}"
            )
        self.patch_count = (
            self.image_size[0] // self.patch_size
        ) * (self.image_size[1] // self.patch_size)
        self.is_train = bool(getattr(dataset, "is_train", False))
        self.sigma = base.frame_label_sigma_seconds(cfg)
        self.ignore_radius = base.frame_label_ignore_radius_seconds(cfg)
        mode = str(motion_cfg.get("sampling_mode", "legacy_pairs"))
        if mode not in {"legacy_pairs", "dense_mixed"}:
            raise ValueError("sampling_mode must be legacy_pairs or dense_mixed")
        self.mixed_sampling = self.is_train and mode == "dense_mixed"
        # Capture BEFORE pair synthesis appends aliases' source records.
        self.natural_count = len(dataset) if self.mixed_sampling else 0
        if self.mixed_sampling and not all(
            row.sample_id.startswith("motion_dense_train_") for row in self.records
        ):
            raise ValueError("dense_mixed requires the long-video dense training load hook")
        self.pair_rows = (
            build_relation_pair_rows(dataset, cfg, allow_empty=self.mixed_sampling)
            if self.is_train and bool(motion_cfg.get("require_true_pairs", False)) else []
        )
        offset = self.natural_count
        self.relation_pair_indices = [(offset + index, offset + index + 1) for index in range(0, len(self.pair_rows), 2)]
        if self.mixed_sampling:
            self.records = list(dataset.records[:self.natural_count]) + [
                dataset.records[row["base_index"]] for row in self.pair_rows
            ]

    def __len__(self) -> int:
        if self.mixed_sampling:
            return self.natural_count + len(self.pair_rows)
        return len(self.pair_rows) if self.pair_rows else len(self.dataset)

    def _empty_teacher(self) -> dict[str, Tensor]:
        return {
            key: value[0]
            for key, value in OnlineObjectMotionTeacher.empty(
                1, self.frames, self.patch_count
            ).items()
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.mixed_sampling or self.pair_rows:
            base_index, pair_row = resolve_stream_row(
                index, self.natural_count, self.pair_rows, mixed=self.mixed_sampling
            )
        else:
            base_index, pair_row = int(index), None
        item = self.dataset[base_index]
        if self.is_train:
            item = annotate_stream_item(
                item, self.dataset.records[base_index], pair_row,
                stream="pair" if pair_row is not None else "natural",
            )
        meta = item["meta"]
        if bool(meta.get("decode_failed", False)):
            item["object_motion_inputs"] = torch.zeros(
                self.frames, 3, *self.image_size, dtype=torch.uint8
            )
            if self.teacher_image_size != self.image_size:
                item["object_motion_teacher_inputs"] = torch.zeros(
                    self.frames, 3, *self.teacher_image_size, dtype=torch.uint8
                )
            item["object_motion_times"] = torch.linspace(
                float(meta.get("sampled_clip_start", 0.0)),
                float(meta.get("sampled_clip_end", self.duration)),
                self.frames,
            )
            item.update(self._empty_teacher())
            item["object_motion_frame_targets"] = torch.zeros(
                self.frames, len(base.LABELS), dtype=torch.float32
            )
            item["object_motion_frame_target_masks"] = torch.zeros_like(
                item["object_motion_frame_targets"]
            )
            return item

        clip_start = float(meta["sampled_clip_start"])
        clip_end = float(meta["sampled_clip_end"])
        video_duration = max(float(meta.get("video_duration", clip_end)), clip_end)
        segments = full_window_segment_bounds(clip_start, clip_end)
        decoded = [
            base.read_video_segment(
                str(meta["video_path"]), self.frames_per_segment, self.teacher_image_size,
                self.is_train, 0.0, start_sec=start, end_sec=end, normalize=False,
                cap_cache=getattr(self.dataset, "cap_cache", None),
                decode_strategy=str(getattr(self.dataset, "decode_strategy", "single_seek")),
                return_frame_times=True, hflip_override=False,
            )
            for start, end in segments
        ]
        teacher_frames = torch.cat([value[0] for value in decoded], dim=0)
        motion_times = torch.cat([value[1] for value in decoded], dim=0)
        order = torch.argsort(motion_times, stable=True)
        teacher_frames, motion_times = teacher_frames[order], motion_times[order]
        if self.teacher_image_size == self.image_size:
            motion_frames = teacher_frames
        else:
            resized = []
            for chunk in teacher_frames.split(4, dim=0):
                resized.append(
                    F.interpolate(
                        chunk.float(), size=self.image_size, mode="bilinear", align_corners=False
                    ).round().clamp_(0, 255).to(torch.uint8)
                )
            motion_frames = torch.cat(resized, dim=0)
        motion_start, motion_end = clip_start, clip_end
        meta["object_motion_coverage"] = full_window_coverage(motion_times, clip_start, clip_end)
        item["object_motion_inputs"] = motion_frames
        if self.teacher_image_size != self.image_size:
            item["object_motion_teacher_inputs"] = teacher_frames
        item["object_motion_times"] = motion_times
        item.update(self._empty_teacher())
        events_by_video = getattr(self.dataset, "events_by_video", {})
        events = events_by_video.get(
            (str(meta["source"]), str(meta["video_id"])), ()
        )
        frame_targets, frame_masks = base.gaussian_frame_targets(
            events,
            motion_start,
            motion_end,
            motion_times.tolist(),
            item["label_masks"].tolist(),
            self.sigma,
            self.ignore_radius,
        )
        item["object_motion_frame_targets"] = frame_targets
        item["object_motion_frame_target_masks"] = frame_masks
        return item


def motion_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    result = _ORIGINAL_COLLATE(batch)
    for key in (
        "object_motion_inputs",
        "object_motion_times",
        "object_motion_heatmap_targets",
        "object_motion_heatmap_masks",
        "object_motion_presence_targets",
        "object_motion_presence_masks",
        "object_motion_coordinate_targets",
        "object_motion_coordinate_masks",
        "object_motion_teacher_confidences",
        "object_motion_motion_quality",
        "object_motion_frame_targets",
        "object_motion_frame_target_masks",
    ):
        result[key] = torch.stack([item[key] for item in batch], dim=0)
    teacher_key = "object_motion_teacher_inputs"
    if all(teacher_key in item for item in batch):
        result[teacher_key] = torch.stack([item[teacher_key] for item in batch], dim=0)
    elif any(teacher_key in item for item in batch):
        raise ValueError("mixed object-motion Teacher resolutions in one batch")
    # Preserve E1.6's class-wise boundary/ambiguity safety weights without its
    # batch-of-five sampler (the 33-frame ViT-L branch requires batch size 1).
    raw_clip = [
        tuple(item["meta"].get("online_clip_loss_weights", ()) or ())
        for item in batch
    ]
    raw_frame = [
        tuple(item["meta"].get("online_frame_loss_weights", ()) or ())
        for item in batch
    ]
    if any(raw_clip) or any(raw_frame):
        expected = len(base.LABELS)
        if not all(len(weights) == expected for weights in raw_clip):
            raise ValueError("online clip weights must have one value per class")
        if not all(len(weights) == expected for weights in raw_frame):
            raise ValueError("online frame weights must have one value per class")
        clip_weights = torch.tensor(raw_clip, dtype=result["label_masks"].dtype)
        frame_weights = torch.tensor(raw_frame, dtype=result["label_masks"].dtype)
        result["label_masks"] = result["label_masks"] * clip_weights
        for key in (
            "frame_target_masks",
            "local_frame_target_masks",
            "highres_pool_target_masks",
            "object_motion_frame_target_masks",
        ):
            value = result.get(key)
            if torch.is_tensor(value):
                result[key] = value * frame_weights.unsqueeze(1).to(value.dtype)
    return result


def load_motion_records(cfg: Any, split: str):
    if split != "train" or str(_motion_cfg(cfg).get("sampling_mode", "legacy_pairs")) != "dense_mixed":
        return online_e16.online_load(cfg, split)
    # Expand the original TRAIN source inventory, before E1.6's event-biased
    # subsampling can omit a background-only video. Evaluation remains intact.
    records, events = online_e16._original_load(cfg, split)
    records = build_dense_training_records(
        records, events, cfg,
        grid_builder=online_e16.e13.build_online_eval_records,
        record_builder=online_e16._e16_record,
    )
    print("object_motion_dense_pool " + json.dumps({
        "windows": len(records),
        "videos": len({(row.source, row.video_id) for row in records}),
        "positive_windows": [sum(row.labels[c] > 0.5 for row in records) for c in range(len(base.LABELS))],
        "valid_negative_windows": [sum(row.labels[c] <= 0.5 and row.label_mask[c] > 0 and row.online_clip_loss_weights[c] > 0 for row in records) for c in range(len(base.LABELS))],
        "masked_windows": [sum(row.label_mask[c] <= 0 or row.online_clip_loss_weights[c] <= 0 for row in records) for c in range(len(base.LABELS))],
    }, sort_keys=True), flush=True)
    return records, events


def prepare_motion_datasets(
    cfg: Any, *, use_cache: bool
) -> tuple[Dataset, Dataset, list[Any], list[Any]]:
    train_dataset, val_dataset, train_records, val_records = (
        _ORIGINAL_PREPARE_DATASETS(cfg, use_cache=use_cache)
    )
    return (
        ObjectMotionDataset(train_dataset, cfg),
        ObjectMotionDataset(val_dataset, cfg),
        train_records,
        val_records,
    )


class _PairLoaderView:
    """Delegate DataLoader APIs while exposing batch_sampler as sampler."""

    def __init__(self, loader: DataLoader, sampler: SameVideoPairBatchSampler) -> None:
        self._loader, self.sampler = loader, sampler

    def __iter__(self):
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self._loader)

    def __getattr__(self, name: str):
        return getattr(self._loader, name)


def make_motion_loader(dataset: Dataset, cfg: Any, *, is_train: bool, batch_size: int | None = None, distributed: bool = False) -> DataLoader:
    motion_cfg = _motion_cfg(cfg)
    mixed = bool(getattr(dataset, "mixed_sampling", False))
    budget = int(motion_cfg.get("sampling_batches_per_rank", 0))
    if not is_train or (not mixed and not bool(motion_cfg.get("require_true_pairs", False))):
        return _ORIGINAL_MAKE_LOADER(dataset, cfg, is_train=is_train, batch_size=batch_size, distributed=distributed)
    local_batch_size = int(batch_size or cfg.train.batch_size)
    if local_batch_size not in (1, 2):
        raise ValueError(f"true relation pairing requires per-rank batch_size 1 or 2, got {local_batch_size}")
    pairs = list(getattr(dataset, "relation_pair_indices", ()))
    if distributed and not base.distributed_training_active():
        raise RuntimeError("distributed motion loader requires an initialized process group")
    rank = base.dist.get_rank() if distributed else 0
    world_size = base.dist.get_world_size() if distributed else 1
    if mixed or budget > 0:
        accumulation = int(cfg.train.get("grad_accum_steps", 1))
        if accumulation <= 0 or budget % accumulation:
            raise ValueError("sampling batch budget must be divisible by grad_accum_steps")
        if pairs and local_batch_size == 1 and accumulation % 2:
            raise ValueError("batch-one reviewed pairs require even grad_accum_steps")
        batch_sampler = DensePairBatchSampler(
            int(getattr(dataset, "natural_count", 0)), pairs,
            batches_per_rank=budget, batch_size=local_batch_size,
            natural_fraction=float(motion_cfg.get("natural_window_fraction", 0.5)) if mixed else 0.0,
            rank=rank, world_size=world_size, seed=int(cfg.get("seed", 42)),
            on_epoch=reset_pair_residual_queues, report=(rank == 0),
        )
    else:
        sampler_type = SameVideoPairQueueBatchSampler if local_batch_size == 1 else SameVideoPairBatchSampler
        batch_sampler = sampler_type(pairs, rank=rank, world_size=world_size, seed=int(cfg.get("seed", 42)))
    generator = torch.Generator().manual_seed(int(cfg.get("seed", 42)) + rank * 100003)
    workers = int(cfg.data.num_workers)
    kwargs: dict[str, Any] = {"batch_sampler": batch_sampler, "num_workers": workers, "pin_memory": bool(cfg.data.pin_memory), "collate_fn": motion_collate, "generator": generator}
    if workers > 0:
        kwargs.update({"worker_init_fn": base.football_worker_init, "persistent_workers": bool(cfg.data.get("persistent_workers", True)), "prefetch_factor": int(cfg.data.get("prefetch_factor", 4))})
    if rank == 0:
        print(f"object_motion_pair_loader pairs={len(pairs)} batches_per_rank={len(batch_sampler)} batch_size={local_batch_size} world_size={world_size}", flush=True)
    return _PairLoaderView(DataLoader(dataset, **kwargs), batch_sampler)


def _motion_forward(
    self: nn.Module,
    inputs: Tensor,
    *args: Any,
    return_aux: bool = False,
    **kwargs: Any,
) -> Tensor | dict[str, Tensor]:
    original_forward = self.__dict__["_object_motion_original_forward"]
    runtime_cfg = self.__dict__["_object_motion_runtime_cfg"]
    motion_cfg = runtime_cfg.model.object_motion
    shared_anchor_ball_lora = bool(
        motion_cfg.get("shared_anchor_ball_lora", False)
    )
    # The trusted source path remains an immutable reference.  In the new
    # shared mode a second anchor pass exposes BallLoRA to event gradients.
    self.backbone.eval()
    with ball_lora_enabled(self.backbone, False), torch.no_grad():
        reference_outputs = original_forward(
            inputs, *args, return_aux=True, **kwargs
        )
    if shared_anchor_ball_lora:
        with ball_lora_enabled(self.backbone, True):
            anchor_outputs = original_forward(
                inputs, *args, return_aux=True, **kwargs
            )
    else:
        anchor_outputs = reference_outputs
    if not isinstance(anchor_outputs, dict):
        raise RuntimeError("object motion anchor must return auxiliary outputs")
    motion_inputs = self.__dict__.get("_object_motion_runtime_inputs")
    motion_times = self.__dict__.get("_object_motion_runtime_times")
    global_times = self.__dict__.get("_object_motion_runtime_global_times")
    if not torch.is_tensor(motion_inputs) or not torch.is_tensor(motion_times):
        raise RuntimeError("object motion runtime batch was not installed")
    ball_layers = tuple(int(value) for value in motion_cfg.get("ball_feature_layers", [11, 17, 20, 23]))
    chunk_size = int(motion_cfg.get("backbone_frame_chunk_size", 1))
    with ball_lora_enabled(self.backbone, True):
        ball_patch_layers = extract_intermediate_patch_layers(
            self,
            motion_inputs,
            layers=ball_layers,
            chunk_size=chunk_size,
            checkpoint_trainable_blocks=bool(
                motion_cfg.get("checkpoint_trainable_blocks", True)
            ),
        )
    patch_tokens = ball_patch_layers[:, :, -1]
    preserve_indices = torch.tensor(
        sorted({0, int(motion_inputs.shape[1]) // 2, int(motion_inputs.shape[1]) - 1}),
        device=motion_inputs.device,
    )
    with ball_lora_enabled(self.backbone, False), torch.no_grad():
        preserve_reference = extract_intermediate_patch_layers(
            self,
            motion_inputs.index_select(1, preserve_indices),
            layers=(ball_layers[-1],),
            chunk_size=chunk_size,
        )[:, :, 0]
    patch_size = int(self.__dict__["_object_motion_backbone_patch_size"])
    grid_h = int(motion_inputs.shape[-2]) // patch_size
    grid_w = int(motion_inputs.shape[-1]) // patch_size
    motion = self.object_motion_adapter(
        patch_tokens,
        motion_times,
        grid_h=grid_h,
        grid_w=grid_w,
        ball_patch_layers=ball_patch_layers,
    )
    anchor_logits = anchor_outputs["logits"]
    anchor_frame_logits = anchor_outputs["frame_event_logits"]
    if not torch.is_tensor(global_times):
        global_times = torch.stack(
            [
                torch.linspace(
                    float(times[0]), float(times[-1]), anchor_frame_logits.shape[1],
                    device=times.device, dtype=times.dtype,
                )
                for times in motion_times
            ],
            dim=0,
        )
    # The dense high-resolution view covers only the center four seconds of a
    # ten-second online window.  Let it alter a class only when the frozen
    # anchor's own frame response places evidence inside that observed span.
    # This gate is label-free and is therefore identical in train/eval/inference.
    local_support = (
        (global_times >= motion_times[:, :1])
        & (global_times <= motion_times[:, -1:])
    ).to(anchor_frame_logits.dtype)
    anchor_time_attention = torch.softmax(
        anchor_frame_logits.detach().float(), dim=1
    ).to(anchor_frame_logits.dtype)
    anchor_local_coverage = (
        anchor_time_attention * local_support.unsqueeze(-1)
    ).sum(dim=1).clamp(0.0, 1.0)
    event_residual_scale = 1.0
    if runtime_cfg is not None:
        event_residual_scale = float(
            runtime_cfg.train.get("object_motion_event_residual_scale", 1.0)
            or 0.0
        )
    event_residual_scale = min(max(event_residual_scale, 0.0), 1.0)
    adapter_clip_residual = motion["clip_residual"]
    adapter_frame_residual = motion["frame_residual"]
    fusion_delta = float(self.object_motion_adapter.fusion_delta)
    object_fusion: dict[str, Tensor] | None = None
    if bool(motion_cfg.get("object_cross_attention_enabled", False)):
        global_event_token = anchor_outputs.get("global_temporal")
        if not torch.is_tensor(global_event_token):
            raise RuntimeError(
                "object-token fusion requires anchor global_temporal"
            )
        object_fusion = self.object_motion_adapter.fuse_global_event(
            global_event_token, motion
        )
        adapter_clip_residual = object_fusion["correction"]
        gated_clip_residual = (
            adapter_clip_residual * event_residual_scale
        )
    else:
        gated_clip_residual = (
            adapter_clip_residual
            * anchor_local_coverage
            * fusion_delta
            * event_residual_scale
        )
    applied_frame_residual = (
        adapter_frame_residual * fusion_delta * event_residual_scale
        if bool(motion_cfg.get("legacy_frame_residual_enabled", True))
        else torch.zeros_like(adapter_frame_residual)
    )
    if object_fusion is not None and "frame_correction" in object_fusion:
        # One frame correction path, never add the legacy residual a second time.
        adapter_frame_residual = object_fusion["frame_correction"]
        applied_frame_residual = adapter_frame_residual * event_residual_scale
    final_logits = anchor_logits + gated_clip_residual
    anchor_on_motion = interpolate_motion_residual(
        anchor_frame_logits, global_times, motion_times
    )
    motion_frame_logits = anchor_on_motion + applied_frame_residual
    frame_residual_on_anchor = interpolate_motion_residual(
        applied_frame_residual, motion_times, global_times
    )
    final_frame_logits = anchor_frame_logits + frame_residual_on_anchor
    result = dict(anchor_outputs)
    result.update(
        {
            "logits": final_logits,
            "frame_event_logits": final_frame_logits,
            "retention_reference_logits": reference_outputs["logits"],
            "object_motion_shared_anchor_logits": anchor_logits,
            "object_motion_shared_anchor_delta": (
                anchor_logits - reference_outputs["logits"]
            ),
            "object_motion_heatmap_logits": motion["heatmap_logits"],
            "object_motion_ball_lora_logits": motion["ball_lora_logits"],
            "object_motion_ball_student_features": motion["ball_student_features"],
            "object_motion_ball_student_center": motion["ball_student_center"],
            "object_motion_ball_student_presence": motion["ball_student_presence"],
            "object_motion_ball_student_entropy": motion["ball_student_entropy"],
            "object_motion_ball_layer_weights": motion["ball_layer_weights"],
            "object_motion_ball_preserve_features": patch_tokens.index_select(1, preserve_indices),
            "object_motion_ball_preserve_reference": preserve_reference,
            "object_motion_ball_preserve_indices": preserve_indices,
            "object_motion_presence_logits": motion["presence_logits"],
            "object_motion_centers": motion["centers"],
            "object_motion_spreads": motion["spreads"],
            "object_motion_velocity": motion["velocity"],
            "object_motion_acceleration": motion["acceleration"],
            "object_motion_class_attention": motion["class_attention"],
            "object_motion_frame_evidence_gate": motion["frame_evidence_gate"],
            "object_motion_clip_evidence_gate": motion["clip_evidence_gate"],
            "object_motion_learned_frame_gate": motion["learned_frame_gate"],
            "object_motion_learned_clip_gate": motion["learned_clip_gate"],
            "object_motion_frame_gate": motion["frame_gate"],
            "object_motion_clip_gate": motion["clip_gate"],
            "object_motion_raw_frame_residual": (
                object_fusion["raw_frame_delta"]
                if object_fusion is not None and "raw_frame_delta" in object_fusion
                else motion["raw_frame_residual"]
            ),
            "object_motion_raw_clip_residual_logits": (
                object_fusion["raw_delta"]
                if object_fusion is not None
                else motion["raw_clip_residual"]
            ),
            "object_motion_adapter_frame_residual": adapter_frame_residual,
            "object_motion_adapter_clip_residual": adapter_clip_residual,
            "object_motion_frame_residual": applied_frame_residual,
            "object_motion_raw_clip_residual": adapter_clip_residual,
            "object_motion_clip_residual": gated_clip_residual,
            "object_motion_event_residual_scale": gated_clip_residual.new_tensor(
                event_residual_scale
            ),
            "object_motion_anchor_local_coverage": anchor_local_coverage,
            "object_motion_frame_logits": motion_frame_logits,
            "object_motion_times": motion_times,
        }
    )
    if object_fusion is not None:
        result.update(
            {
                "object_motion_object_token_attention": object_fusion[
                    "attention"
                ],
                "object_motion_object_token_gate": object_fusion["gate"],
                "object_motion_object_token_attended": object_fusion[
                    "attended_tokens"
                ],
                "object_motion_learned_clip_gate": object_fusion["gate"],
                "object_motion_clip_gate": object_fusion["gate"],
            }
        )
        if "frame_correction" in object_fusion:
            frame_gate = object_fusion["gate"].unsqueeze(1).expand_as(adapter_frame_residual)
            result["object_motion_learned_frame_gate"] = frame_gate
            result["object_motion_frame_gate"] = frame_gate
    return result if return_aux else final_logits


def make_motion_model(
    cfg: Any, *, use_cached_features: bool, device: torch.device
) -> nn.Module:
    model = _ORIGINAL_MAKE_MODEL(
        cfg, use_cached_features=use_cached_features, device=device
    )
    motion_cfg = _motion_cfg(cfg)
    if not bool(motion_cfg.get("enabled", False)):
        return model
    if model.backbone is None or bool(getattr(model.backbone, "is_video_backbone", False)):
        raise ValueError("object motion requires an uncached DINO image backbone")
    for parameter in model.parameters():
        parameter.requires_grad = False
    # VideoEventClassifier.train() honors this flag by keeping the complete
    # full-image reference path in eval mode while newly attached modules stay
    # trainable.  This makes the anchor deterministic despite temporal dropout.
    model.freeze_global_branch = True
    adapter = ObjectMotionEvidenceAdapter(
        patch_dim=int(model.backbone.num_features),
        hidden_dim=int(motion_cfg.get("hidden_dim", cfg.model.hidden_dim)),
        num_labels=len(base.LABELS),
        num_heads=int(motion_cfg.get("num_heads", 8)),
        temporal_layers=int(motion_cfg.get("temporal_layers", 2)),
        dropout=float(motion_cfg.get("dropout", cfg.model.dropout)),
        topk_ratios=tuple(motion_cfg.get("topk_ratios", [0.01, 0.05, 0.08])),
        attention_temperature=float(motion_cfg.get("temperature", 0.5)),
        residual_max_delta=float(motion_cfg.get("residual_max_delta", 1.0)),
        frame_residual_max_delta=float(
            motion_cfg.get(
                "frame_residual_max_delta",
                motion_cfg.get("residual_max_delta", 1.0),
            )
        ),
        clip_residual_max_delta=float(
            motion_cfg.get(
                "clip_residual_max_delta",
                motion_cfg.get("residual_max_delta", 1.0),
            )
        ),
        gate_init=float(motion_cfg.get("gate_init", 0.10)),
        gate_max=float(motion_cfg.get("gate_max", 1.0)),
        gate_floor=float(motion_cfg.get("gate_floor", 0.25)),
        relation_delta=float(motion_cfg.get("relation_delta", 1.0)),
        fusion_delta=float(motion_cfg.get("fusion_delta", 0.5)),
        evidence_gate_floor=float(motion_cfg.get("evidence_gate_floor", 0.0)),
        class_evidence_floors=tuple(
            motion_cfg.get(
                "class_evidence_floors",
                [float(motion_cfg.get("evidence_gate_floor", 0.0))]
                * len(base.LABELS),
            )
        ),
        detach_detector_for_event=bool(
            motion_cfg.get("detach_detector_for_event", True)
        ),
        ball_layer_weights=tuple(motion_cfg.get("ball_layer_weights", [0.15, 0.25, 0.25, 0.35])),
        ball_topk=int(motion_cfg.get("ball_topk", 4)),
        ball_temperature=float(motion_cfg.get("ball_temperature", 0.25)),
        heatmap_upsample_factor=int(
            motion_cfg.get("heatmap_upsample_factor", 1)
        ),
        ball_fpn_dim=int(motion_cfg.get("ball_fpn_dim", 64)),
        object_cross_attention_enabled=bool(
            motion_cfg.get("object_cross_attention_enabled", False)
        ),
        object_cross_attention_max_delta=float(
            motion_cfg.get("object_cross_attention_max_delta", 0.35)
        ),
        object_cross_attention_gate_init=float(
            motion_cfg.get("object_cross_attention_gate_init", 0.25)
        ),
        event_relation_grad_enabled=bool(motion_cfg.get("event_relation_grad_enabled", False)),
        event_context_enabled=bool(motion_cfg.get("event_context_enabled", False)),
        event_context_feature_grad=bool(motion_cfg.get("event_context_feature_grad", False)),
        event_context_uniform=bool(motion_cfg.get("event_context_uniform", False)),
        event_frame_fusion_enabled=bool(motion_cfg.get("event_frame_fusion_enabled", False)),
    )
    injected = inject_ball_lora(
        model.backbone,
        rank=int(motion_cfg.get("ball_lora_rank", 4)),
        alpha=float(motion_cfg.get("ball_lora_alpha", 4.0)),
        last_blocks=int(motion_cfg.get("ball_lora_last_blocks", 8)),
    )
    if injected != int(motion_cfg.get("ball_lora_last_blocks", 8)) * 2:
        raise RuntimeError(f"unexpected ball LoRA injection count: {injected}")
    _initialize_motion_adapter(adapter, cfg)
    adapter.to(device)
    model.add_module("object_motion_adapter", adapter)
    _restore_ball_backbone_state(model, cfg)
    # Activation-checkpoint recomputation happens after the temporary forward
    # context has exited. Keep shared-anchor LoRA enabled as the model's
    # persistent state so recomputation follows the same graph; immutable
    # reference passes still disable it explicitly in _motion_forward.
    shared_anchor_ball_lora = bool(
        motion_cfg.get("shared_anchor_ball_lora", False)
    )
    for module in ball_lora_modules(model.backbone):
        module.enabled = shared_anchor_ball_lora
    model.backbone_has_trainable_params = True
    ball_names, adapter_names = validate_trainable_allowlist(model)
    model.__dict__["_object_motion_ball_trainable_names"] = tuple(ball_names)
    model.__dict__["_object_motion_adapter_trainable_names"] = tuple(adapter_names)
    model.__dict__["_object_motion_backbone_patch_size"] = int(
        motion_cfg.get("patch_size", 16)
    )
    model.__dict__["_object_motion_heatmap_patch_size"] = int(
        motion_cfg.get(
            "heatmap_patch_size",
            motion_cfg.get("patch_size", 16),
        )
    )
    # Curriculum stages mutate cfg.train at each epoch; retaining the config
    # reference lets forward apply a gradual residual scale without rebuilding
    # or unfreezing the trusted anchor.
    model.__dict__["_object_motion_runtime_cfg"] = cfg
    model.__dict__["_object_motion_original_forward"] = model.forward
    model.forward = types.MethodType(_motion_forward, model)
    trainable = sum(
        parameter.numel() for parameter in adapter.parameters() if parameter.requires_grad
    )
    print(
        f"object_motion_adapter enabled trainable_params={trainable} "
        f"frames={int(motion_cfg.get('frames_per_segment', 11)) * 3} "
        f"duration={float(motion_cfg.get('duration_sec', 4.0)):.2f}s "
        f"image_size={list(motion_cfg.get('image_size', [512, 896]))}",
        flush=True,
    )
    return model


def forward_motion_batch(
    model: nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    *,
    return_aux: bool = False,
) -> Tensor | dict[str, Tensor]:
    module = base.unwrap_model(model)
    module.__dict__["_object_motion_runtime_inputs"] = batch[
        "object_motion_inputs"
    ].to(device, non_blocking=True)
    module.__dict__["_object_motion_runtime_times"] = batch[
        "object_motion_times"
    ].to(device, non_blocking=True)
    frame_times = batch.get("frame_times")
    module.__dict__["_object_motion_runtime_global_times"] = (
        frame_times.to(device, non_blocking=True)
        if torch.is_tensor(frame_times)
        else None
    )
    try:
        return _ORIGINAL_FORWARD_BATCH(
            model, batch, device, return_aux=return_aux
        )
    finally:
        for key in (
            "_object_motion_runtime_inputs",
            "_object_motion_runtime_times",
            "_object_motion_runtime_global_times",
        ):
            module.__dict__.pop(key, None)


class _TeacherHook:
    def __init__(self, teachers: list[Any]) -> None:
        if not teachers:
            raise ValueError("object motion teacher hook requires at least one teacher")
        self.teachers = teachers
        # Compatibility for heatmap audit utilities that inspect the online
        # teacher directly.
        self.teacher = teachers[-1]

    def fill_missing(self, batch: dict[str, Any]) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for teacher in self.teachers:
            metrics.update(teacher.fill(batch))
        return metrics


def object_motion_teacher_from_config(
    cfg: Any, device: torch.device
) -> _TeacherHook | None:
    motion_cfg = _motion_cfg(cfg)
    if not bool(motion_cfg.get("enabled", False)):
        return None
    teachers: list[Any] = []
    offline_cfg = motion_cfg.get("offline_ball_teacher", base.ConfigDict())
    if bool(offline_cfg.get("enabled", False)):
        teachers.append(
            OfflineTrackedBallTeacher(
                index_root=str(offline_cfg.get("index_root", "")),
                patch_size=int(
                    motion_cfg.get(
                        "heatmap_patch_size",
                        motion_cfg.get("patch_size", 16),
                    )
                ),
                max_time_delta_sec=float(
                    offline_cfg.get("max_time_delta_sec", 0.10)
                ),
                ball_sigma_patches=float(
                    offline_cfg.get("ball_sigma_patches", 1.25)
                ),
                cache_videos=int(offline_cfg.get("cache_videos", 4)),
            )
        )
    teacher_cfg = motion_cfg.get("online_teacher", base.ConfigDict())
    if bool(teacher_cfg.get("enabled", False)):
        teacher_device = torch.device(device)
        if teacher_device.type == "cuda" and torch.cuda.is_available():
            # cfg.device/current_device are cuda:0 in every process in this
            # trainer; torchrun LOCAL_RANK is the authoritative binding.
            local_rank = int(os.environ.get("LOCAL_RANK", torch.cuda.current_device()))
            if local_rank != 0:
                # Checkpoint restoration temporarily materializes tensors on
                # logical cuda:0 in every rank. Release that non-local cache
                # before the per-rank YOLO teacher is lazily constructed.
                with torch.cuda.device(0):
                    torch.cuda.empty_cache()
            teacher_device = torch.device("cuda", local_rank)
        teachers.append(
            OnlineObjectMotionTeacher(
                device=teacher_device,
                ball_checkpoint=str(teacher_cfg.ball_checkpoint),
                scene_checkpoint=str(teacher_cfg.scene_checkpoint),
                patch_size=int(
                    motion_cfg.get(
                        "heatmap_patch_size",
                        motion_cfg.get("patch_size", 16),
                    )
                ),
                ball_confidence=float(teacher_cfg.get("ball_confidence", 0.10)),
                goal_confidence=float(teacher_cfg.get("goal_confidence", 0.25)),
                person_confidence=float(teacher_cfg.get("person_confidence", 0.25)),
                ball_sigma_patches=float(teacher_cfg.get("ball_sigma_patches", 1.25)),
                box_dilation_patches=float(teacher_cfg.get("box_dilation_patches", 0.5)),
                input_size=tuple(teacher_cfg.get("input_size", [1088, 1920])),
                batch_size=int(teacher_cfg.get("batch_size", 16)),
                half=bool(teacher_cfg.get("half", True)),
                iou=float(teacher_cfg.get("iou", 0.70)),
                absence_presence_weights=tuple(
                    teacher_cfg.get(
                        "absence_presence_weights", [0.05, 0.02, 0.02]
                    )
                ),
                enabled_objects=tuple(
                    teacher_cfg.get("enabled_objects", OBJECT_NAMES)
                ),
            )
        )
    if not teachers:
        return None
    return _TeacherHook(teachers)


def object_motion_loss_hook(
    outputs: dict[str, Tensor],
    batch: dict[str, Any],
    cfg: Any,
    device: torch.device,
) -> tuple[Tensor, dict[str, float]]:
    return object_motion_auxiliary_loss(
        outputs,
        batch,
        cfg,
        device,
        clip_targets=batch["targets"].to(device, non_blocking=True),
        clip_label_masks=batch["label_masks"].to(device, non_blocking=True),
    )



def build_motion_optimizer(model: nn.Module, cfg: Any) -> torch.optim.Optimizer:
    """Two-group optimizer with an extra per-group clipping safety hook."""
    ball_parameters: list[nn.Parameter] = []
    adapter_parameters: list[nn.Parameter] = []
    illegal: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        normalized = name.removeprefix("module.")
        if normalized.startswith("backbone.") and normalized.endswith(("ball_lora_a", "ball_lora_b")):
            ball_parameters.append(parameter)
        elif normalized.startswith("object_motion_adapter."):
            adapter_parameters.append(parameter)
        else:
            illegal.append(normalized)
    if illegal or not ball_parameters or not adapter_parameters:
        raise RuntimeError(
            f"invalid optimizer allowlist ball={len(ball_parameters)} "
            f"adapter={len(adapter_parameters)} illegal={illegal[:8]}"
        )
    optimizer = torch.optim.AdamW(
        [
            {
                "name": "ball_lora",
                "params": ball_parameters,
                "lr": float(cfg.train.get("ball_lora_lr", 1e-6)),
                "weight_decay": float(cfg.train.get("ball_lora_weight_decay", 0.0)),
            },
            {
                "name": "head",
                "params": adapter_parameters,
                "lr": float(cfg.train.get("object_motion_head_lr", 2e-5)),
                "weight_decay": float(cfg.train.get("object_motion_head_weight_decay", 0.05)),
            },
        ]
    )
    ball_clip = float(cfg.train.get("ball_lora_grad_clip", 0.1))
    head_clip = float(cfg.train.get("object_motion_head_grad_clip", 1.0))

    def clip_groups(current: torch.optim.Optimizer, _args: tuple[Any, ...], _kwargs: dict[str, Any]) -> None:
        nn.utils.clip_grad_norm_(current.param_groups[0]["params"], ball_clip)
        nn.utils.clip_grad_norm_(current.param_groups[1]["params"], head_clip)

    optimizer.register_step_pre_hook(clip_groups)
    return optimizer

def install_hooks() -> None:
    # Preserve E1.6 in legacy mode; dense_mixed expands the training grid and
    # uses a fixed-budget sampler compatible with high-resolution batch one.
    base.load_long_video_records = load_motion_records
    base.FootballLongVideoDataset._sample_window = online_e16.e13.exact_online_window
    base.make_model = make_motion_model
    base.prepare_datasets = prepare_motion_datasets
    base.football_collate = motion_collate
    base.forward_model_batch = forward_motion_batch
    base.make_loader = make_motion_loader
    base.build_optimizer = build_motion_optimizer
    base.online_object_teacher_from_config = object_motion_teacher_from_config
    # Reuse the mature trainer's generic weighted auxiliary-loss call site.
    base.object_teacher_heatmap_loss = object_motion_loss_hook


def main() -> None:
    install_hooks()
    base.main()


if __name__ == "__main__":
    main()
