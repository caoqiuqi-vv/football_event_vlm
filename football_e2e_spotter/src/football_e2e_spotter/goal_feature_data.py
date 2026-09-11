"""Feature-bank dataset for 180 s context / 60 s core retrieval training."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .goal_annotations import LABELS, GoalEvent, load_goal_annotations
from .goal_retriever import RetrieverGeometry


LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}


@dataclass(frozen=True)
class ScheduledWindow:
    video_id: str
    focus_event: GoalEvent | None
    ordinal: int


def _load_manifest(path: Path, split: str) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(item["media_id"]): item for item in payload[split]}


class GoalFeatureWindowDataset(Dataset):
    """Every accepted GT is a focus event once per epoch plus 1:1 background."""

    def __init__(
        self,
        *,
        manifest: str | Path,
        feature_root: str | Path,
        split: str,
        geometry: RetrieverGeometry = RetrieverGeometry(),
        seed: int = 3407,
        background_ratio: float = 1.0,
    ) -> None:
        self.manifest_path = Path(manifest).expanduser().resolve()
        self.feature_root = Path(feature_root).expanduser().resolve()
        self.split = str(split)
        self.geometry = geometry
        self.seed = int(seed)
        self.background_ratio = float(background_ratio)
        self.items = _load_manifest(self.manifest_path, self.split)
        self.events: dict[str, tuple[GoalEvent, ...]] = {}
        self.metadata: dict[str, dict] = {}
        missing = []
        for video_id, item in self.items.items():
            base = self.feature_root / self.split / video_id
            metadata_path = base / "metadata.json"
            timeline_path = base / "timeline.npz"
            if metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            elif timeline_path.is_file():
                with np.load(timeline_path, allow_pickle=False) as timeline:
                    metadata = {
                        "steps": int(timeline["timestamps"].shape[0]),
                        "appearance_dim": int(timeline["appearance"].shape[1]),
                        "motion_dim": int(timeline["motion"].shape[1]),
                        "audio_dim": int(timeline["audio"].shape[1]),
                    }
            else:
                missing.append(video_id)
                continue
            self.metadata[video_id] = metadata
            self.events[video_id] = load_goal_annotations(item["annotation_path"])
        self.items = {video_id: self.items[video_id] for video_id in self.metadata}
        if not self.items:
            raise RuntimeError(f"no feature-bank videos ready for {split}; missing={missing[:5]}")
        dimensions = {
            (int(row["appearance_dim"]), int(row["motion_dim"]), int(row["audio_dim"]))
            for row in self.metadata.values()
        }
        if len(dimensions) != 1:
            raise RuntimeError(f"incompatible feature dimensions: {dimensions}")
        self.appearance_dim, self.motion_dim, self.audio_dim = dimensions.pop()
        self.epoch = 0
        self.schedule: list[ScheduledWindow] = []
        self.set_epoch(0)

    def __getstate__(self):
        state = self.__dict__.copy()
        # lru_cache is process-local and should never be serialized.
        self._timeline.cache_clear()
        return state

    @lru_cache(maxsize=3)
    def _timeline(self, video_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        base = self.feature_root / self.split / video_id
        paths = [base / name for name in ("timestamps.npy", "appearance.npy", "motion.npy", "audio.npy")]
        if all(path.is_file() for path in paths):
            return tuple(np.load(path, mmap_mode="r", allow_pickle=False) for path in paths)  # type: ignore[return-value]
        timeline = np.load(base / "timeline.npz", allow_pickle=False)
        # Keep the NpzFile alive through array ownership; this fallback is only
        # for early feature-bank shards before mmap conversion.
        return tuple(np.asarray(timeline[name]) for name in ("timestamps", "appearance", "motion", "audio"))  # type: ignore[return-value]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        rng = random.Random(self.seed + 1_000_003 * self.epoch)
        positives = [
            ScheduledWindow(video_id, event, ordinal)
            for video_id, events in self.events.items()
            for ordinal, event in enumerate(events)
            if event.accepted
        ]
        rng.shuffle(positives)
        backgrounds = []
        count = int(round(len(positives) * self.background_ratio))
        video_ids = sorted(self.items)
        for ordinal in range(count):
            backgrounds.append(ScheduledWindow(video_ids[ordinal % len(video_ids)], None, ordinal))
        rng.shuffle(backgrounds)
        self.schedule = positives + backgrounds
        rng.shuffle(self.schedule)

    def __len__(self) -> int:
        return len(self.schedule)

    def _duration(self, video_id: str) -> float:
        timestamps, _appearance, _motion, _audio = self._timeline(video_id)
        return float(timestamps[-1] + 0.5) if len(timestamps) else 0.0

    def _core_start(self, window: ScheduledWindow, index: int) -> float:
        duration = self._duration(window.video_id)
        maximum = max(duration - self.geometry.core_seconds, 0.0)
        rng = random.Random(self.seed + 10_000_019 * self.epoch + 97_003 * index)
        if window.focus_event is not None:
            position = rng.uniform(8.0, self.geometry.core_seconds - 8.0)
            return min(max(window.focus_event.timestamp - position, 0.0), maximum)
        accepted = [event.timestamp for event in self.events[window.video_id] if event.accepted]
        for _ in range(64):
            start = rng.uniform(0.0, maximum) if maximum else 0.0
            if all(not start - 3.0 <= timestamp < start + self.geometry.core_seconds + 3.0 for timestamp in accepted):
                return start
        return rng.uniform(0.0, maximum) if maximum else 0.0

    def __getitem__(self, index: int) -> dict:
        window = self.schedule[index]
        core_start = self._core_start(window, index)
        context_start = core_start - self.geometry.context_left_seconds
        timestamps, appearance, motion, audio = self._timeline(window.video_id)
        source_indices = np.floor(context_start + np.arange(self.geometry.context_steps)).astype(np.int64)
        valid = (source_indices >= 0) & (source_indices < len(timestamps))
        clipped = np.clip(source_indices, 0, max(len(timestamps) - 1, 0))
        appearance_window = np.asarray(appearance[clipped], dtype=np.float32)
        motion_window = np.asarray(motion[clipped], dtype=np.float32)
        audio_window = np.asarray(audio[clipped], dtype=np.float32)
        appearance_window[~valid] = 0.0
        motion_window[~valid] = 0.0
        audio_window[~valid] = 0.0
        core_events = [
            event for event in self.events[window.video_id]
            if event.accepted and core_start <= event.timestamp < core_start + self.geometry.core_seconds
        ]
        return {
            "appearance": torch.from_numpy(appearance_window.copy()),
            "motion": torch.from_numpy(motion_window.copy()),
            "audio": torch.from_numpy(audio_window.copy()),
            "valid": torch.from_numpy(valid),
            "target": {
                "labels": torch.as_tensor([LABEL_TO_INDEX[event.label] for event in core_events], dtype=torch.long),
                "times": torch.as_tensor([event.timestamp - core_start for event in core_events], dtype=torch.float32),
                "event_ids": [event.event_id for event in core_events],
            },
            "video_id": window.video_id,
            "core_start": core_start,
        }


def collate_goal_windows(batch: list[dict]) -> dict:
    return {
        "appearance": torch.stack([item["appearance"] for item in batch]),
        "motion": torch.stack([item["motion"] for item in batch]),
        "audio": torch.stack([item["audio"] for item in batch]),
        "valid": torch.stack([item["valid"] for item in batch]),
        "targets": [item["target"] for item in batch],
        "video_ids": [item["video_id"] for item in batch],
        "core_starts": torch.as_tensor([item["core_start"] for item in batch]),
    }

