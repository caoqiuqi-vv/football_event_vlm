"""Training-only patch targets from the v9 repaired ball-track index."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor


OBJECT_CHANNELS = ("ball", "goal")


class TrackedBallTargetProvider:
    """Create a ball heatmap while leaving unavailable targets unknown.

    The offline index stores normalized boxes, timestamp alignment, source
    flags, and a repaired-track quality weight. Goal stays fully masked in the
    first Mechanism-A experiment so teacher sources are not mixed.
    """

    def __init__(
        self,
        index_root: str | Path,
        *,
        image_size: Sequence[int],
        patch_size: int = 16,
        max_frame_gap_sec: float = 0.10,
        ball_sigma_patches: float = 1.25,
        min_confidence: float = 0.0,
        min_quality: float = 0.0,
        teacher_time_offset_sec: float = 0.0,
        max_cached_videos: int = 4,
    ) -> None:
        self.index_root = Path(index_root).expanduser()
        if not self.index_root.is_dir():
            raise FileNotFoundError(
                f"tracked ball teacher index not found: {self.index_root}"
            )
        self.image_height, self.image_width = (int(value) for value in image_size)
        self.patch_size = int(patch_size)
        if self.image_height % self.patch_size or self.image_width % self.patch_size:
            raise ValueError(
                "tracked ball teacher requires image dimensions divisible by "
                f"patch_size: image={image_size} patch={patch_size}"
            )
        self.grid_h = self.image_height // self.patch_size
        self.grid_w = self.image_width // self.patch_size
        self.patch_count = self.grid_h * self.grid_w
        self.max_frame_gap_sec = max(float(max_frame_gap_sec), 0.0)
        self.ball_sigma_patches = max(float(ball_sigma_patches), 0.25)
        self.min_confidence = max(float(min_confidence), 0.0)
        self.min_quality = max(float(min_quality), 0.0)
        self.teacher_time_offset_sec = float(teacher_time_offset_sec)
        self.max_cached_videos = max(int(max_cached_videos), 1)
        self._cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        grid_y, grid_x = torch.meshgrid(
            torch.arange(self.grid_h, dtype=torch.float32) + 0.5,
            torch.arange(self.grid_w, dtype=torch.float32) + 0.5,
            indexing="ij",
        )
        self._grid_x = grid_x
        self._grid_y = grid_y

    def has_video(self, video_id: str) -> bool:
        return (self.index_root / f"{video_id}.npz").is_file()

    def _load(self, video_id: str) -> dict[str, np.ndarray] | None:
        key = str(video_id)
        cached = self._cache.pop(key, None)
        if cached is not None:
            self._cache[key] = cached
            return cached
        path = self.index_root / f"{key}.npz"
        if not path.is_file():
            return None
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "timestamp_sec",
                "bbox_xyxy_norm",
                "confidence",
                "quality_weight",
                "flags",
            }
            missing = sorted(required - set(archive.files))
            if missing:
                raise ValueError(
                    f"tracked ball index {path} is missing fields: {missing}"
                )
            payload = {name: archive[name] for name in archive.files}
        rows = len(payload["timestamp_sec"])
        for name in required - {"timestamp_sec"}:
            if len(payload[name]) != rows:
                raise ValueError(
                    f"tracked ball index {path} has inconsistent rows for {name}"
                )
        self._cache[key] = payload
        while len(self._cache) > self.max_cached_videos:
            self._cache.popitem(last=False)
        return payload

    def empty(self, frames: int) -> tuple[Tensor, Tensor]:
        shape = (int(frames), self.patch_count, len(OBJECT_CHANNELS))
        return (
            torch.zeros(shape, dtype=torch.float32),
            torch.zeros(shape, dtype=torch.float32),
        )

    @staticmethod
    def _nearest_indices(
        source: np.ndarray, query: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        right = np.searchsorted(source, query, side="left").clip(
            0, len(source) - 1
        )
        left = (right - 1).clip(0, len(source) - 1)
        choose_left = np.abs(source[left] - query) <= np.abs(source[right] - query)
        selected = np.where(choose_left, left, right)
        return selected, np.abs(source[selected] - query)

    def _ball_target(self, center_x: float, center_y: float) -> Tensor:
        center_x_patch = float(center_x) * self.grid_w
        center_y_patch = float(center_y) * self.grid_h
        distance2 = (self._grid_x - center_x_patch).square() + (
            self._grid_y - center_y_patch
        ).square()
        return torch.exp(
            -0.5 * distance2 / (self.ball_sigma_patches**2)
        )

    def targets(
        self, video_id: str, absolute_times: Sequence[float]
    ) -> tuple[Tensor, Tensor]:
        targets, masks, _metadata = self.targets_with_metadata(
            video_id, absolute_times
        )
        return targets, masks

    def targets_with_metadata(
        self, video_id: str, absolute_times: Sequence[float]
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        payload = self._load(str(video_id))
        targets, masks = self.empty(len(absolute_times))
        metadata = {
            "bbox_xyxy_norm": torch.zeros(
                len(absolute_times), 4, dtype=torch.float32
            ),
            "confidence": torch.zeros(len(absolute_times), dtype=torch.float32),
            "quality": torch.zeros(len(absolute_times), dtype=torch.float32),
            "flags": torch.zeros(len(absolute_times), dtype=torch.uint8),
            "valid": torch.zeros(len(absolute_times), dtype=torch.float32),
        }
        if payload is None or not len(payload["timestamp_sec"]):
            return targets, masks, metadata
        source_times = payload["timestamp_sec"].astype(np.float64, copy=False)
        query_times = (
            np.asarray(absolute_times, dtype=np.float64)
            + self.teacher_time_offset_sec
        )
        selected, deltas = self._nearest_indices(source_times, query_times)
        valid = deltas <= self.max_frame_gap_sec
        for frame_index in np.flatnonzero(valid):
            row = int(selected[frame_index])
            confidence = float(payload["confidence"][row])
            quality = float(payload["quality_weight"][row])
            flags = int(payload["flags"][row])
            if (
                not bool(flags & 1)
                or confidence < self.min_confidence
                or quality < self.min_quality
            ):
                continue
            x1, y1, x2, y2 = (
                float(value) for value in payload["bbox_xyxy_norm"][row]
            )
            center_x = 0.5 * (x1 + x2)
            center_y = 0.5 * (y1 + y2)
            if not (0.0 <= center_x <= 1.0 and 0.0 <= center_y <= 1.0):
                continue
            metadata["bbox_xyxy_norm"][frame_index] = torch.tensor(
                [x1, y1, x2, y2], dtype=torch.float32
            )
            metadata["confidence"][frame_index] = confidence
            metadata["quality"][frame_index] = quality
            metadata["flags"][frame_index] = flags
            metadata["valid"][frame_index] = 1.0
            targets[frame_index, :, 0] = self._ball_target(
                center_x, center_y
            ).flatten()
            # Confidence remains provenance. Repaired-track quality controls
            # loss strength without weakening the positive heatmap target.
            masks[frame_index, :, 0] = max(min(quality, 1.0), 0.0)
        return targets, masks, metadata
