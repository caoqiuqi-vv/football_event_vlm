"""Five-class, core-aligned dataset for TemporalSetSpotter."""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch import Tensor

from .data import PixelAudioSequenceDataset
from .set_spotting import EventInstance, SET_LABELS


LABEL_MAP = {
    "射门": "shot", "其他射门类型": "shot", "扑救": "save", "任意球": "freekick",
    "角球": "corner", "中圈开球": "kickoff", "开球": "kickoff",
}


def _time(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        try:
            parts = [float(item) for item in str(value).split(":")]
            return parts[-1] + 60 * parts[-2] + (3600 * parts[-3] if len(parts) == 3 else 0)
        except (ValueError, IndexError):
            return None


def load_set_events(path: Path) -> tuple[tuple[int, float], ...]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    records = raw.get("data", raw.get("events", raw.get("annotations", []))) if isinstance(raw, dict) else raw
    events = []
    for item in records:
        if not isinstance(item, dict) or item.get("label_correct") is False:
            continue
        label = LABEL_MAP.get(str(item.get("label", "")))
        timestamp = _time(item.get("startTime", item.get("timestamp")))
        if label is not None and timestamp is not None and timestamp >= 0:
            events.append((SET_LABELS.index(label), timestamp))
    return tuple(sorted(events, key=lambda item: item[1]))


class TemporalSetCoreDataset(PixelAudioSequenceDataset):
    """Each item has a unique 30-second core, plus fixed left/right context."""

    def __init__(self, *args, core_seconds: float = 30, context_seconds: float = 15, background_ratio: float = 1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.core_seconds, self.context_seconds = float(core_seconds), float(context_seconds)
        self.core_frames, self.context_frames = int(round(self.core_seconds * self.sample_fps)), int(round(self.context_seconds * self.sample_fps))
        self.sequence_frames = self.core_frames + 2 * self.context_frames
        self.background_ratio = float(background_ratio)
        self.set_events = {video_id: load_set_events(self.annotations / f"{self.metadata[video_id]['annotation_id']}.json") for video_id in self.video_ids}
        self._set_schedule: list[tuple[int, tuple[int, float] | None]] = []
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        # super().__init__ calls this before set_events exists.
        if not hasattr(self, "set_events"):
            self.epoch = int(epoch)
            return
        self.epoch = int(epoch)
        schedule = []
        rng = random.Random(self.seed + 17_171 * self.epoch)
        for video_index, video_id in enumerate(self.video_ids):
            values = list(self.set_events[video_id])
            rng.shuffle(values)
            schedule.extend((video_index, event) for event in values)
            schedule.extend((video_index, None) for _ in range(max(1, round(len(values) * self.background_ratio))))
        rng.shuffle(schedule)
        self._set_schedule = schedule

    def __len__(self) -> int:
        return len(self._set_schedule)

    def _core_start(self, video_index: int, event: tuple[int, float] | None, rng: random.Random) -> int:
        video_id = self.video_ids[video_index]
        frames = int(self.metadata[video_id]["frame_count"])
        maximum = max(frames - self.core_frames, 0)
        if event is not None:
            center = int(round(event[1] * self.sample_fps))
            offset = rng.randint(0, max(self.core_frames - 1, 0))
            return min(max(center - offset, 0), maximum)
        for _ in range(64):
            start = rng.randint(0, maximum) if maximum else 0
            left, right = start / self.sample_fps, (start + self.core_frames) / self.sample_fps
            if not any(left <= timestamp < right for _, timestamp in self.set_events[video_id]):
                return start
        return rng.randint(0, maximum) if maximum else 0

    def __getitem__(self, item: int) -> dict[str, Tensor | list[EventInstance] | str | float]:
        video_index, anchor = self._set_schedule[item]
        video_id = self.video_ids[video_index]
        rng = random.Random(self.seed + 2_000_003 * self.epoch + item)
        core_start = self._core_start(video_index, anchor, rng)
        read_start = core_start - self.context_frames
        frame_count = int(self.metadata[video_id]["frame_count"])
        raw = list(range(read_start, read_start + self.sequence_frames))
        valid = torch.tensor([0 <= index < frame_count for index in raw], dtype=torch.bool)
        indices = [min(max(index, 0), frame_count - 1) for index in raw]
        frames = self._augment(self.read_frame_indices(video_id, indices), rng)
        audio = torch.from_numpy(self.audio[video_id][indices].astype("float32", copy=True))
        left, right = core_start / self.sample_fps, (core_start + self.core_frames) / self.sample_fps
        targets = [EventInstance(label, (timestamp - left) / self.core_seconds) for label, timestamp in self.set_events[video_id] if left <= timestamp < right]
        return {"frames": frames, "audio": audio, "valid": valid, "targets": targets, "video_id": video_id, "core_start_seconds": left}


def set_collate(batch: list[dict]) -> dict:
    return {
        "frames": torch.stack([item["frames"] for item in batch]),
        "audio": torch.stack([item["audio"] for item in batch]),
        "valid": torch.stack([item["valid"] for item in batch]),
        "targets": [item["targets"] for item in batch],
        "video_id": [item["video_id"] for item in batch],
        "core_start_seconds": torch.tensor([item["core_start_seconds"] for item in batch]),
    }
