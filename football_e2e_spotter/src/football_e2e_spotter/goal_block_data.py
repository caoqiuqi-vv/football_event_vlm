"""Dense aligned-core dataset for KPI-aligned block retrieval."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .goal_annotations import LABELS, load_goal_annotations
from .goal_block_retriever import BlockGeometry
from .goal_retriever_eval import load_timeline


LABEL_INDEX = {label: index for index, label in enumerate(LABELS)}


class GoalBlockDataset(Dataset):
    def __init__(self, *, manifest: str | Path, feature_root: str | Path, split: str, geometry: BlockGeometry = BlockGeometry()) -> None:
        self.manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))[split]
        self.feature_root = Path(feature_root) / split
        self.geometry = geometry
        self.items = []
        self.events = {}
        self.timelines = {}
        for item in self.manifest:
            video_id = str(item["media_id"]); base = self.feature_root / video_id
            if not ((base / "metadata.json").is_file() or (base / "timeline.npz").is_file()):
                continue
            timeline = load_timeline(base); self.timelines[video_id] = timeline
            self.events[video_id] = tuple(event for event in load_goal_annotations(item["annotation_path"]) if event.accepted)
            duration = len(timeline[1])
            for core_start in np.arange(0, duration, geometry.core_seconds):
                self.items.append((video_id, float(core_start)))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        video_id, core_start = self.items[index]
        _timestamps, appearance, motion, audio = self.timelines[video_id]
        context_start = core_start - self.geometry.context_left
        source = np.floor(context_start + np.arange(self.geometry.context_steps)).astype(np.int64)
        valid = (source >= 0) & (source < len(appearance)); clipped = np.clip(source, 0, max(len(appearance) - 1, 0))
        streams = []
        for values in (appearance, motion, audio):
            value = np.asarray(values[clipped], dtype=np.float32); value[~valid] = 0.0; streams.append(torch.from_numpy(value.copy()))
        block_targets = torch.zeros(self.geometry.blocks, len(LABELS))
        dense_targets = torch.zeros(self.geometry.core_seconds, len(LABELS))
        for event in self.events[video_id]:
            relative = event.timestamp - core_start
            if not 0 <= relative < self.geometry.core_seconds:
                continue
            label = LABEL_INDEX[event.label]; block = min(int(relative // self.geometry.block_seconds), self.geometry.blocks - 1)
            block_targets[block, label] = 1.0
            seconds = torch.arange(self.geometry.core_seconds, dtype=torch.float32) + 0.5
            dense_targets[:, label] = torch.maximum(dense_targets[:, label], torch.exp(-0.5 * ((seconds - relative) / 1.5).square()))
        return {
            "appearance": streams[0], "motion": streams[1], "audio": streams[2],
            "valid": torch.from_numpy(valid), "block_targets": block_targets,
            "dense_targets": dense_targets, "video_id": video_id, "core_start": core_start,
        }

