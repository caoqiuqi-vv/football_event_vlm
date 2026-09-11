from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import cv2
import lmdb
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import gaussian_blur

from .annotations import (
    FAMILIES, LABELS, Event, gaussian_targets, load_annotations,
    nearest_offsets, supervision_mask,
)


IMAGENET_MEAN = torch.tensor((0.485, 0.456, 0.406)).reshape(1, 3, 1, 1)
IMAGENET_STD = torch.tensor((0.229, 0.224, 0.225)).reshape(1, 3, 1, 1)


class PixelAudioSequenceDataset(Dataset):
    """Video-balanced dense sequences; each selected event is used at most once/epoch."""

    def __init__(
        self,
        video_ids: tuple[str, ...],
        *,
        store_root: str | Path,
        split: str,
        annotations: str | Path,
        sequence_frames: int,
        sequences_per_video: int,
        positive_probability: float,
        temporal_jitter_seconds: float,
        seed: int,
        class_sigma_seconds: dict[str, float],
        family_sigma_seconds: dict[str, float],
        ignore_radius_seconds: dict[str, float],
        max_offset_seconds: float,
        augment: dict[str, Any] | None = None,
    ) -> None:
        if not video_ids or len(video_ids) != len(set(video_ids)):
            raise ValueError("video IDs must be non-empty and unique")
        self.video_ids = tuple(video_ids)
        self.store_root = Path(store_root).expanduser().resolve()
        self.split = str(split)
        self.annotations = Path(annotations).expanduser().resolve()
        self.sequence_frames = max(int(sequence_frames), 1)
        self.sequences_per_video = max(int(sequences_per_video), 1)
        self.positive_probability = min(max(float(positive_probability), 0.0), 1.0)
        self.temporal_jitter_seconds = max(float(temporal_jitter_seconds), 0.0)
        self.seed = int(seed)
        self.class_sigma_seconds = dict(class_sigma_seconds)
        self.family_sigma_seconds = dict(family_sigma_seconds)
        self.ignore_radius_seconds = dict(ignore_radius_seconds)
        self.max_offset_seconds = float(max_offset_seconds)
        self.augment = dict(augment or {})
        self._environments: dict[str, lmdb.Environment] = {}
        self.metadata = {}
        self.audio = {}
        self.events: dict[str, tuple[Event, ...]] = {}
        self.rejected: dict[str, tuple[Event, ...]] = {}
        contracts = set()
        for video_id in self.video_ids:
            base = self.store_root / self.split / video_id
            metadata = json.loads((base / "metadata.json").read_text(encoding="utf-8"))
            if metadata.get("schema") != "football_e2e_spotter.pixel_audio_store.v1":
                raise ValueError(f"unsupported store metadata: {base}")
            contracts.add((
                float(metadata["sample_fps"]), tuple(metadata["image_size"]),
                int(metadata["audio_mel_bins"]), str(metadata["config_sha256"]),
            ))
            self.metadata[video_id] = metadata
            self.audio[video_id] = np.load(
                base / "audio_logmel.npy", mmap_mode="r", allow_pickle=False
            )
            supervision = load_annotations(self.annotations / f"{metadata['annotation_id']}.json")
            self.events[video_id] = supervision.accepted
            self.rejected[video_id] = supervision.rejected
        if len(contracts) != 1:
            raise RuntimeError("pixel/audio store mixes incompatible preprocessing contracts")
        self.sample_fps, self.image_size, self.audio_mel_bins, self.preprocessing_sha256 = contracts.pop()
        self.epoch = 0
        self._schedule: dict[tuple[int, int], Event | None] = {}
        self.set_epoch(0)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_environments"] = {}
        return state

    def __len__(self) -> int:
        return len(self.video_ids) * self.sequences_per_video

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._schedule.clear()
        positive_slots = min(
            int(round(self.sequences_per_video * self.positive_probability)),
            self.sequences_per_video,
        )
        for video_index, video_id in enumerate(self.video_ids):
            rng = random.Random(self.seed + 1_000_003 * self.epoch + 9_973 * video_index)
            by_label = {
                label: [event for event in self.events[video_id] if event.label == label]
                for label in LABELS
            }
            by_label = {label: values for label, values in by_label.items() if values}
            for values in by_label.values():
                rng.shuffle(values)
            chosen: list[Event] = []
            labels = list(by_label)
            rng.shuffle(labels)
            cursor = {label: 0 for label in labels}
            while len(chosen) < positive_slots and labels:
                next_labels = []
                for label in labels:
                    index = cursor[label]
                    if index < len(by_label[label]) and len(chosen) < positive_slots:
                        chosen.append(by_label[label][index])
                        cursor[label] += 1
                    if cursor[label] < len(by_label[label]):
                        next_labels.append(label)
                labels = next_labels
            rng.shuffle(chosen)
            for slot in range(self.sequences_per_video):
                self._schedule[(video_index, slot)] = chosen[slot] if slot < len(chosen) else None

    def _environment(self, video_id: str) -> lmdb.Environment:
        environment = self._environments.get(video_id)
        if environment is None:
            environment = lmdb.open(
                str(self.store_root / self.split / video_id / "frames.lmdb"),
                readonly=True, lock=False, readahead=False, max_readers=256,
            )
            self._environments[video_id] = environment
        return environment

    def read_frame_indices(self, video_id: str, indices: list[int]) -> Tensor:
        environment = self._environment(video_id)
        frames = []
        with environment.begin(buffers=True) as transaction:
            for index in indices:
                value = transaction.get(f"f/{index:08d}".encode())
                if value is None:
                    raise KeyError(f"missing frame {video_id}/{index}")
                frame = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise RuntimeError(f"corrupt JPEG {video_id}/{index}")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(torch.from_numpy(frame).permute(2, 0, 1))
        return torch.stack(frames)

    def _choose_start(self, video_index: int, slot: int, rng: random.Random) -> int:
        video_id = self.video_ids[video_index]
        frame_count = int(self.metadata[video_id]["frame_count"])
        maximum = max(frame_count - self.sequence_frames, 0)
        event = self._schedule[(video_index, slot)]
        if event is not None:
            center = int(round(event.timestamp * self.sample_fps))
            jitter = int(round(rng.uniform(
                -self.temporal_jitter_seconds, self.temporal_jitter_seconds
            ) * self.sample_fps))
            return min(max(center - self.sequence_frames // 2 + jitter, 0), maximum)
        # Background is sampled by temporal coverage, not by model score. Avoid
        # placing a known event at the core of a nominal background sequence.
        margin = max(self.ignore_radius_seconds.values(), default=0.0)
        for _ in range(32):
            start = rng.randint(0, maximum) if maximum else 0
            left = start / self.sample_fps - margin
            right = (start + self.sequence_frames) / self.sample_fps + margin
            if not any(left <= event.timestamp <= right for event in self.events[video_id]):
                return start
        return rng.randint(0, maximum) if maximum else 0

    def _augment(self, frames: Tensor, rng: random.Random) -> Tensor:
        frames = frames.float().div_(255.0)
        if rng.random() < float(self.augment.get("horizontal_flip_probability", 0.0)):
            frames = torch.flip(frames, dims=(-1,))
        if rng.random() < float(self.augment.get("color_jitter_probability", 0.0)):
            strength = float(self.augment.get("color_jitter_strength", 0.2))
            brightness = rng.uniform(1.0 - strength, 1.0 + strength)
            contrast = rng.uniform(1.0 - strength, 1.0 + strength)
            saturation = rng.uniform(1.0 - strength, 1.0 + strength)
            frames = frames * brightness
            mean = frames.mean(dim=(-1, -2), keepdim=True)
            frames = (frames - mean) * contrast + mean
            gray = (
                0.299 * frames[:, 0:1] + 0.587 * frames[:, 1:2] + 0.114 * frames[:, 2:3]
            )
            frames = gray + saturation * (frames - gray)
            frames = frames.clamp_(0.0, 1.0)
        if rng.random() < float(self.augment.get("gaussian_blur_probability", 0.0)):
            frames = gaussian_blur(frames, kernel_size=[5, 5], sigma=[0.4, 1.2])
        return (frames - IMAGENET_MEAN) / IMAGENET_STD

    @staticmethod
    def normalize_eval_frames(frames: Tensor) -> Tensor:
        frames = frames.float().div_(255.0)
        return (frames - IMAGENET_MEAN) / IMAGENET_STD

    def __getitem__(self, item: int) -> dict[str, Tensor]:
        video_index = int(item) % len(self.video_ids)
        slot = int(item) // len(self.video_ids)
        video_id = self.video_ids[video_index]
        rng = random.Random(self.seed + 10_000_019 * self.epoch + 101 * int(item))
        start = self._choose_start(video_index, slot, rng)
        frame_count = int(self.metadata[video_id]["frame_count"])
        raw_indices = list(range(start, start + self.sequence_frames))
        valid = torch.tensor([index < frame_count for index in raw_indices], dtype=torch.bool)
        indices = [min(index, frame_count - 1) for index in raw_indices]
        frames = self._augment(self.read_frame_indices(video_id, indices), rng)
        audio_array = self.audio[video_id]
        audio = torch.from_numpy(np.array(audio_array[indices], dtype=np.float32, copy=True))
        timestamps = (torch.arange(self.sequence_frames, dtype=torch.float32) + start + 0.5) / self.sample_fps
        events = self.events[video_id]
        rejected = self.rejected[video_id]
        class_targets = gaussian_targets(
            timestamps, events, LABELS, self.class_sigma_seconds
        )
        family_targets = gaussian_targets(
            timestamps, events, FAMILIES, self.family_sigma_seconds, by_family=True
        )
        class_valid = valid[:, None] & supervision_mask(
            timestamps, rejected, LABELS, self.ignore_radius_seconds
        )
        family_radius = {
            "shot_chain": max(self.ignore_radius_seconds["shot"], self.ignore_radius_seconds["save"]),
            "restart": max(self.ignore_radius_seconds["corner"], self.ignore_radius_seconds["freekick"]),
        }
        family_valid = valid[:, None] & supervision_mask(
            timestamps, rejected, FAMILIES, family_radius, by_family=True
        )
        offsets = nearest_offsets(
            timestamps, events, LABELS, self.max_offset_seconds
        )
        return {
            "frames": frames,
            "audio": audio,
            "timestamps": timestamps,
            "valid": valid,
            "class_targets": class_targets,
            "family_targets": family_targets,
            "class_valid": class_valid,
            "family_valid": family_valid,
            "offset_targets": offsets,
            "video_index": torch.tensor(video_index),
            "start_index": torch.tensor(start),
        }
