from __future__ import annotations

import random
from functools import lru_cache
import math
from pathlib import Path
from typing import Mapping

import torch
from torch.utils.data import Dataset, Sampler

from .annotations import (
    Event,
    build_family_supervision_mask,
    build_family_targets,
    build_label_supervision_mask,
    build_label_targets,
    load_event_annotations,
)
from .feature_store import AlignedTimeline, load_aligned_npz


class VideoCohortBatchSampler(Sampler[list[int]]):
    """Keep batches video-diverse while routing repeated blocks to the same workers."""

    def __init__(
        self, video_count: int, blocks_per_video: int, *, batch_size: int,
        worker_count: int, seed: int = 42,
    ) -> None:
        if video_count <= 0 or blocks_per_video <= 0 or batch_size <= 0:
            raise ValueError("video_count, blocks_per_video and batch_size must be positive")
        self.video_count = int(video_count)
        self.blocks_per_video = int(blocks_per_video)
        self.batch_size = int(batch_size)
        self.worker_count = max(int(worker_count), 1)
        self.cohort_size = self.batch_size * self.worker_count
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        video_order = list(range(self.video_count))
        rng.shuffle(video_order)
        for cohort_start in range(0, self.video_count, self.cohort_size):
            cohort = video_order[cohort_start:cohort_start + self.cohort_size]
            for block_index in range(self.blocks_per_video):
                for batch_start in range(0, len(cohort), self.batch_size):
                    videos = cohort[batch_start:batch_start + self.batch_size]
                    yield [block_index * self.video_count + video_index for video_index in videos]

    def __len__(self) -> int:
        full, remainder = divmod(self.video_count, self.cohort_size)
        batches_per_round = full * self.worker_count
        if remainder:
            batches_per_round += math.ceil(remainder / self.batch_size)
        return batches_per_round * self.blocks_per_video


class VideoBalancedTimelineDataset(Dataset[dict[str, torch.Tensor]]):
    """Samples one block per video before repeating any video in an epoch cycle."""

    def __init__(
        self,
        video_ids: tuple[str, ...],
        *,
        feature_store: str | Path,
        annotations: str | Path,
        families: tuple[str, ...],
        timeline_hz: float,
        sequence_seconds: float,
        labels: tuple[str, ...] | None = None,
        blocks_per_video: int = 8,
        positive_probability: float = 0.75,
        jitter_seconds: float = 15.0,
        seed: int = 42,
        feature_split: str | None = None,
        cache_id_by_video_id: Mapping[str, str] | None = None,
        annotation_id_by_video_id: Mapping[str, str] | None = None,
        sigma_seconds: dict[str, float] | None = None,
        rejected_ignore_radius_seconds: dict[str, float] | None = None,
        class_sigma_seconds: dict[str, float] | None = None,
        rejected_class_ignore_radius_seconds: dict[str, float] | None = None,
        class_state_sigma_seconds: dict[str, float] | None = None,
        require_reviewed_annotations: bool = False,
        require_all_videos: bool = True,
    ) -> None:
        base_feature_store = Path(feature_store)
        self.feature_split = feature_split
        self.feature_store = (
            base_feature_store / feature_split if feature_split is not None else base_feature_store
        )
        self.annotations = Path(annotations)
        self.families = families
        self.labels = tuple(labels or ())
        self.timeline_hz = float(timeline_hz)
        self.block_steps = int(round(timeline_hz * sequence_seconds))
        self.blocks_per_video = int(blocks_per_video)
        self.positive_probability = float(positive_probability)
        self.jitter_seconds = float(jitter_seconds)
        self.seed = int(seed)
        self.sigma_seconds = sigma_seconds
        self.rejected_ignore_radius_seconds = rejected_ignore_radius_seconds
        self.class_sigma_seconds = class_sigma_seconds
        self.rejected_class_ignore_radius_seconds = rejected_class_ignore_radius_seconds
        self.class_state_sigma_seconds = class_state_sigma_seconds or {
            "shot": 3.0, "save": 3.0, "corner": 8.0, "penalty": 10.0,
            "freekick": 8.0, "kickoff": 8.0,
        }
        self.epoch = 0
        self.requested_video_ids = tuple(video_ids)
        supplied_cache_ids = dict(cache_id_by_video_id or {})
        unknown_aliases = sorted(set(supplied_cache_ids) - set(self.requested_video_ids))
        if unknown_aliases:
            raise ValueError(f"cache aliases not requested by this dataset: {unknown_aliases[:5]}")
        self.cache_id_by_video_id = {
            video_id: supplied_cache_ids.get(video_id, video_id)
            for video_id in self.requested_video_ids
        }
        supplied_annotation_ids = dict(annotation_id_by_video_id or {})
        unknown_annotation_overrides = sorted(set(supplied_annotation_ids) - set(self.requested_video_ids))
        if unknown_annotation_overrides:
            raise ValueError(f"annotation overrides not requested by this dataset: {unknown_annotation_overrides[:5]}")
        self.annotation_id_by_video_id = {
            video_id: supplied_annotation_ids.get(video_id, video_id)
            for video_id in self.requested_video_ids
        }
        self.video_ids = tuple(
            video_id
            for video_id in self.requested_video_ids
            if (self.feature_store / self.cache_id_by_video_id[video_id] / "timeline.npz").is_file()
            and (self.annotations / f"{self.annotation_id_by_video_id[video_id]}.json").is_file()
        )
        if not self.video_ids:
            raise RuntimeError(
                f"no ready timeline caches under {self.feature_store}; run feature extraction first"
            )
        self.missing_video_ids = tuple(
            video_id for video_id in self.requested_video_ids if video_id not in self.video_ids
        )
        if require_all_videos and self.missing_video_ids:
            raise RuntimeError(
                f"missing {len(self.missing_video_ids)} requested timeline/annotation pairs: "
                f"{self.missing_video_ids[:8]}"
            )
        supervision = {
            video_id: load_event_annotations(
                self.annotations / f"{self.annotation_id_by_video_id[video_id]}.json",
                require_reviewed=require_reviewed_annotations,
            )
            for video_id in self.video_ids
        }
        self.events = {video_id: item.accepted for video_id, item in supervision.items()}
        self.rejected_events = {video_id: item.rejected for video_id, item in supervision.items()}

    @lru_cache(maxsize=4)
    def _load_timeline(self, cache_id: str) -> AlignedTimeline:
        return load_aligned_npz(self.feature_store / cache_id / "timeline.npz")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.video_ids) * self.blocks_per_video

    def _choose_start(self, timeline: AlignedTimeline, events: tuple[Event, ...], rng: random.Random) -> int:
        maximum = max(timeline.timestamps.numel() - self.block_steps, 0)
        if events and rng.random() < self.positive_probability:
            available_labels = sorted({
                event.label for event in events if not self.labels or event.label in self.labels
            })
            label = rng.choice(available_labels)
            anchor = rng.choice([event.timestamp for event in events if event.label == label])
            center = anchor + rng.uniform(-self.jitter_seconds, self.jitter_seconds)
            start_time = max(center - 0.5 * self.block_steps / self.timeline_hz, 0.0)
            start = int(torch.searchsorted(timeline.timestamps, torch.tensor(start_time)).item())
            return min(start, maximum)
        return rng.randint(0, maximum) if maximum else 0

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        # Consecutive indices map to distinct videos. Keep DataLoader shuffle off.
        video_index = index % len(self.video_ids)
        video_id = self.video_ids[video_index]
        timeline = self._load_timeline(self.cache_id_by_video_id[video_id])
        rng = random.Random(self.seed + self.epoch * len(self) + index)
        start = self._choose_start(timeline, self.events[video_id], rng)
        stop = min(start + self.block_steps, timeline.timestamps.numel())

        def pad(value: torch.Tensor, fill: float = 0.0) -> torch.Tensor:
            value = value[start:stop]
            missing = self.block_steps - value.shape[0]
            if missing <= 0:
                return value
            shape = (missing, *value.shape[1:])
            return torch.cat([value, value.new_full(shape, fill)], dim=0)

        timestamps = pad(timeline.timestamps)
        if stop - start < self.block_steps and stop > start:
            step = 1.0 / self.timeline_hz
            timestamps[stop - start :] = timestamps[stop - start - 1] + step * torch.arange(
                1, self.block_steps - (stop - start) + 1, dtype=timestamps.dtype
            )
        targets, offsets = build_family_targets(
            timestamps, self.events[video_id], self.families,
            sigma_seconds=self.sigma_seconds,
        )
        feature_valid = pad(timeline.context_valid).bool() & pad(timeline.motion_valid).bool()
        supervision_valid = build_family_supervision_mask(
            timestamps,
            self.rejected_events[video_id],
            self.families,
            ignore_radius_seconds=self.rejected_ignore_radius_seconds,
        )
        valid = feature_valid.unsqueeze(-1) & supervision_valid
        class_targets = build_label_targets(
            timestamps, self.events[video_id], self.labels,
            sigma_seconds=self.class_sigma_seconds,
        )
        class_state_targets = build_label_targets(
            timestamps, self.events[video_id], self.labels,
            sigma_seconds=self.class_state_sigma_seconds,
        )
        class_supervision_valid = build_label_supervision_mask(
            timestamps, self.rejected_events[video_id], self.labels,
            ignore_radius_seconds=self.rejected_class_ignore_radius_seconds,
        )
        class_valid = feature_valid.unsqueeze(-1) & class_supervision_valid
        return {
            "timestamps": timestamps,
            "context": pad(timeline.context),
            "motion": pad(timeline.motion),
            "context_valid": pad(timeline.context_valid).bool(),
            "motion_valid": pad(timeline.motion_valid).bool(),
            "targets": targets,
            "offset_targets": offsets,
            "valid": valid,
            "class_targets": class_targets,
            "class_state_targets": class_state_targets,
            "class_valid": class_valid,
        }
