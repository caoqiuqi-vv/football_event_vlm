from __future__ import annotations

"""Memory-mapped datasets for stitched, non-overlapping VideoMAE chunks."""

import json
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .annotations import (
    Event,
    build_family_supervision_mask,
    build_label_supervision_mask,
    build_label_targets,
    load_event_annotations,
)


CHUNK_STORE_SCHEMA = "football_longform_v2.videomae_chunk_store.v1"


@dataclass(frozen=True)
class ChunkTimeline:
    features: np.ndarray
    timestamps: np.ndarray


def build_label_offsets(
    timestamps: Tensor,
    events: tuple[Event, ...],
    labels: tuple[str, ...],
    *,
    max_offset_seconds: float,
) -> Tensor:
    """Nearest class-specific event offsets, clipped to the model's range."""
    offsets = timestamps.new_zeros((timestamps.numel(), len(labels)))
    limit = max(float(max_offset_seconds), 0.0)
    for label_index, label in enumerate(labels):
        event_times = [event.timestamp for event in events if event.label == label]
        if not event_times:
            continue
        anchors = timestamps.new_tensor(event_times)
        delta = anchors[:, None] - timestamps[None, :]
        nearest = delta.abs().argmin(dim=0)
        offsets[:, label_index] = delta[
            nearest, torch.arange(timestamps.numel(), device=timestamps.device)
        ].clamp(-limit, limit)
    return offsets


def build_family_state_targets(
    timestamps: Tensor,
    events: tuple[Event, ...],
    families: tuple[str, ...],
    *,
    sigma_seconds: dict[str, float] | None = None,
) -> Tensor:
    """Broad family-state targets retaining both shot and save phases."""
    sigma_seconds = sigma_seconds or {"shot_chain": 3.0, "restart": 4.0}
    targets = timestamps.new_zeros((timestamps.numel(), len(families)))
    for family_index, family in enumerate(families):
        anchors = [event.timestamp for event in events if event.family == family]
        if not anchors:
            continue
        event_times = timestamps.new_tensor(anchors)
        nearest = (event_times[:, None] - timestamps[None, :]).abs().amin(dim=0)
        sigma = max(float(sigma_seconds.get(family, 1.0)), 1e-3)
        targets[:, family_index] = torch.exp(-0.5 * (nearest / sigma).square())
    return targets


class SequentialChunkDataset(Dataset[dict[str, Tensor]]):
    """Video-balanced long blocks over a once-computed VideoMAE timeline.

    Raw video chunks never overlap.  Random blocks here only index the cached
    tubelet sequence, so sampling diversity adds no repeated backbone compute.
    """

    def __init__(
        self,
        video_ids: tuple[str, ...],
        *,
        store_root: str | Path,
        split: str,
        annotations: str | Path,
        annotation_id_by_video_id: Mapping[str, str] | None = None,
        labels: tuple[str, ...] = ("shot", "save", "corner", "freekick", "penalty"),
        families: tuple[str, ...] = ("shot_chain", "restart"),
        block_seconds: float = 192.0,
        blocks_per_video: int = 12,
        positive_probability: float = 0.75,
        jitter_seconds: float = 24.0,
        seed: int = 42,
        max_offset_seconds: float = 2.0,
        class_sigma_seconds: dict[str, float] | None = None,
        family_sigma_seconds: dict[str, float] | None = None,
        class_ignore_radius_seconds: dict[str, float] | None = None,
        family_ignore_radius_seconds: dict[str, float] | None = None,
        require_all_videos: bool = True,
    ) -> None:
        if not video_ids:
            raise ValueError("video_ids cannot be empty")
        if block_seconds <= 0 or blocks_per_video <= 0:
            raise ValueError("block_seconds and blocks_per_video must be positive")
        if not 0.0 <= positive_probability <= 1.0:
            raise ValueError("positive_probability must be in [0,1]")
        self.store_split = Path(store_root) / split
        self.annotations = Path(annotations)
        self.labels = tuple(labels)
        self.families = tuple(families)
        self.blocks_per_video = int(blocks_per_video)
        self.positive_probability = float(positive_probability)
        self.jitter_seconds = float(jitter_seconds)
        self.seed = int(seed)
        self.max_offset_seconds = float(max_offset_seconds)
        self.class_sigma_seconds = class_sigma_seconds
        self.family_sigma_seconds = family_sigma_seconds
        self.class_ignore_radius_seconds = class_ignore_radius_seconds
        self.family_ignore_radius_seconds = family_ignore_radius_seconds
        self.epoch = 0

        requested = tuple(video_ids)
        supplied = dict(annotation_id_by_video_id or {})
        unknown = sorted(set(supplied) - set(requested))
        if unknown:
            raise ValueError(f"annotation overrides not requested: {unknown[:5]}")
        self.annotation_id_by_video_id = {
            video_id: supplied.get(video_id, video_id) for video_id in requested
        }
        ready: list[str] = []
        metadata: dict[str, dict] = {}
        for video_id in requested:
            base = self.store_split / video_id
            metadata_path = base / "metadata.json"
            annotation_path = self.annotations / f"{self.annotation_id_by_video_id[video_id]}.json"
            if not (
                metadata_path.is_file()
                and (base / "features.npy").is_file()
                and (base / "timestamps.npy").is_file()
                and annotation_path.is_file()
            ):
                continue
            item = json.loads(metadata_path.read_text(encoding="utf-8"))
            if item.get("schema") != CHUNK_STORE_SCHEMA:
                raise ValueError(f"unsupported chunk store schema for {video_id}")
            if str(item.get("video_id")) != video_id:
                raise ValueError(f"chunk metadata video ID mismatch for {video_id}")
            ready.append(video_id)
            metadata[video_id] = item
        self.requested_video_ids = requested
        self.video_ids = tuple(ready)
        self.missing_video_ids = tuple(item for item in requested if item not in ready)
        if not self.video_ids:
            raise RuntimeError(f"no ready chunk timelines under {self.store_split}")
        if require_all_videos and self.missing_video_ids:
            raise RuntimeError(
                f"missing {len(self.missing_video_ids)} chunk/annotation pairs: "
                f"{self.missing_video_ids[:8]}"
            )

        feature_dims = {int(metadata[item]["feature_dim"]) for item in self.video_ids}
        tubelet_seconds = {
            round(float(metadata[item]["tubelet_seconds"]), 8) for item in self.video_ids
        }
        checkpoint_hashes = {
            str(metadata[item]["checkpoint_sha256"]) for item in self.video_ids
        }
        chunks_seconds = {
            round(float(metadata[item]["chunk_seconds"]), 8) for item in self.video_ids
        }
        if (
            len(feature_dims) != 1 or len(tubelet_seconds) != 1
            or len(checkpoint_hashes) != 1 or len(chunks_seconds) != 1
        ):
            raise RuntimeError("chunk store mixes incompatible feature provenance")
        self.feature_dim = feature_dims.pop()
        self.step_seconds = tubelet_seconds.pop()
        self.chunk_seconds = chunks_seconds.pop()
        self.tubelets_per_chunk = int(round(self.chunk_seconds / self.step_seconds))
        if self.tubelets_per_chunk <= 0:
            raise RuntimeError("invalid chunk/tubelet timing contract")
        self.checkpoint_sha256 = checkpoint_hashes.pop()
        self.block_steps = max(int(round(block_seconds / self.step_seconds)), 1)
        supervision = {
            video_id: load_event_annotations(
                self.annotations / f"{self.annotation_id_by_video_id[video_id]}.json"
            )
            for video_id in self.video_ids
        }
        # An excluded task label must not leak back through the shared family
        # state target (for example a rare penalty supervising restart).  Keep
        # only explicitly configured classes for every form of supervision.
        self.events = {
            video_id: tuple(event for event in item.accepted if event.label in self.labels)
            for video_id, item in supervision.items()
        }
        self.rejected_events = {
            video_id: tuple(event for event in item.rejected if event.label in self.labels)
            for video_id, item in supervision.items()
        }

    @lru_cache(maxsize=4)
    def _load_timeline(self, video_id: str) -> ChunkTimeline:
        base = self.store_split / video_id
        features = np.load(base / "features.npy", mmap_mode="r", allow_pickle=False)
        timestamps = np.load(base / "timestamps.npy", mmap_mode="r", allow_pickle=False)
        if features.ndim != 2 or features.shape != (timestamps.size, self.feature_dim):
            raise RuntimeError(f"invalid chunk array shape for {video_id}")
        if timestamps.ndim != 1 or timestamps.size == 0:
            raise RuntimeError(f"empty or invalid timestamps for {video_id}")
        return ChunkTimeline(features=features, timestamps=timestamps)

    def full_timeline(self, video_id: str) -> tuple[Tensor, Tensor]:
        timeline = self._load_timeline(video_id)
        # Copy to writable memory before torch conversion; numpy mmap is read-only.
        features = torch.from_numpy(np.array(timeline.features, dtype=np.float32, copy=True))
        timestamps = torch.from_numpy(np.array(timeline.timestamps, dtype=np.float32, copy=True))
        return features, timestamps

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.video_ids) * self.blocks_per_video

    def _choose_start(
        self, video_id: str, timestamps: np.ndarray, rng: random.Random
    ) -> int:
        maximum = max(int(timestamps.size) - self.block_steps, 0)
        events = tuple(event for event in self.events[video_id] if event.label in self.labels)
        if events and rng.random() < self.positive_probability:
            # Choose label first so frequent shots do not erase rare restart modes.
            available_labels = sorted({event.label for event in events})
            label = rng.choice(available_labels)
            anchor = rng.choice([event.timestamp for event in events if event.label == label])
            center = anchor + rng.uniform(-self.jitter_seconds, self.jitter_seconds)
            start_time = max(center - 0.5 * self.block_steps * self.step_seconds, 0.0)
            start = int(np.searchsorted(timestamps, start_time, side="left"))
            return min(start, maximum)
        return rng.randint(0, maximum) if maximum else 0

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        video_index = int(index) % len(self.video_ids)
        video_id = self.video_ids[video_index]
        timeline = self._load_timeline(video_id)
        rng = random.Random(self.seed + self.epoch * len(self) + int(index))
        start = self._choose_start(video_id, timeline.timestamps, rng)
        stop = min(start + self.block_steps, timeline.timestamps.size)
        count = stop - start

        features = torch.zeros(self.block_steps, self.feature_dim, dtype=torch.float32)
        timestamps = torch.empty(self.block_steps, dtype=torch.float32)
        valid = torch.zeros(self.block_steps, dtype=torch.bool)
        if count:
            features[:count] = torch.from_numpy(
                np.array(timeline.features[start:stop], dtype=np.float32, copy=True)
            )
            timestamps[:count] = torch.from_numpy(
                np.array(timeline.timestamps[start:stop], dtype=np.float32, copy=True)
            )
            valid[:count] = True
            if count < self.block_steps:
                timestamps[count:] = timestamps[count - 1] + self.step_seconds * torch.arange(
                    1, self.block_steps - count + 1, dtype=torch.float32
                )
        else:
            timestamps[:] = self.step_seconds * torch.arange(self.block_steps)

        events = self.events[video_id]
        rejected = self.rejected_events[video_id]
        class_targets = build_label_targets(
            timestamps, events, self.labels, sigma_seconds=self.class_sigma_seconds
        )
        family_targets = build_family_state_targets(
            timestamps, events, self.families, sigma_seconds=self.family_sigma_seconds
        )
        class_valid = valid[:, None] & build_label_supervision_mask(
            timestamps,
            rejected,
            self.labels,
            ignore_radius_seconds=self.class_ignore_radius_seconds,
        )
        family_valid = valid[:, None] & build_family_supervision_mask(
            timestamps,
            rejected,
            self.families,
            ignore_radius_seconds=self.family_ignore_radius_seconds,
        )
        offsets = build_label_offsets(
            timestamps,
            events,
            self.labels,
            max_offset_seconds=self.max_offset_seconds,
        )
        return {
            "features": features,
            "timestamps": timestamps,
            "valid": valid,
            "chunk_phase": torch.arange(
                start, start + self.block_steps, dtype=torch.long
            ).remainder(self.tubelets_per_chunk),
            "class_targets": class_targets,
            "family_targets": family_targets,
            "class_valid": class_valid,
            "family_valid": family_valid,
            "offset_targets": offsets,
            "video_index": torch.tensor(video_index, dtype=torch.int64),
        }
