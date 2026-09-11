"""Offline tracked-ball pseudo-label teacher.

The index is deliberately partitioned by video.  A training rank only loads
the videos it samples, avoiding a many-million-row Python dictionary in every
DDP process. Missing detections remain unknown rather than becoming negatives.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor


class OfflineTrackedBallTeacher:
    """Fill the ball channel from temporally repaired detection tracks."""

    def __init__(
        self,
        *,
        index_root: str,
        patch_size: int,
        max_time_delta_sec: float = 0.10,
        ball_sigma_patches: float = 1.25,
        cache_videos: int = 4,
    ) -> None:
        self.index_root = Path(index_root).expanduser()
        if not self.index_root.is_dir():
            raise FileNotFoundError(f"offline ball index not found: {self.index_root}")
        self.patch_size = int(patch_size)
        self.max_time_delta_sec = max(float(max_time_delta_sec), 0.0)
        self.ball_sigma_patches = max(float(ball_sigma_patches), 0.25)
        self.cache_videos = max(int(cache_videos), 1)
        self._cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

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
            value = {name: archive[name] for name in archive.files}
        self._cache[key] = value
        while len(self._cache) > self.cache_videos:
            self._cache.popitem(last=False)
        return value

    @staticmethod
    def _nearest_indices(source: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        right = np.searchsorted(source, query, side="left").clip(0, len(source) - 1)
        left = (right - 1).clip(0, len(source) - 1)
        choose_left = np.abs(source[left] - query) <= np.abs(source[right] - query)
        selected = np.where(choose_left, left, right)
        return selected, np.abs(source[selected] - query)

    def _heatmap(
        self,
        center_x: float,
        center_y: float,
        *,
        grid_h: int,
        grid_w: int,
        grid_y: Tensor,
        grid_x: Tensor,
    ) -> Tensor:
        cx = float(center_x) * grid_w
        cy = float(center_y) * grid_h
        distance2 = (grid_x - cx).square() + (grid_y - cy).square()
        return torch.exp(-0.5 * distance2 / (self.ball_sigma_patches**2))

    def fill(self, batch: dict[str, Any]) -> dict[str, float]:
        started = time.perf_counter()
        inputs = batch.get("object_motion_inputs")
        times = batch.get("object_motion_times")
        targets = batch.get("object_motion_heatmap_targets")
        masks = batch.get("object_motion_heatmap_masks")
        metas = batch.get("meta")
        if not torch.is_tensor(inputs) or inputs.ndim != 5:
            raise ValueError("offline ball teacher requires inputs [B,T,C,H,W]")
        if not torch.is_tensor(times) or times.ndim != 2:
            raise ValueError("offline ball teacher requires times [B,T]")
        if not torch.is_tensor(targets) or not torch.is_tensor(masks):
            raise ValueError("offline ball teacher target placeholders are missing")
        if not isinstance(metas, list) or len(metas) != inputs.shape[0]:
            raise ValueError("offline ball teacher requires one meta mapping per sample")

        batch_size, frames, patch_count, _ = targets.shape
        grid_h = int(inputs.shape[-2]) // self.patch_size
        grid_w = int(inputs.shape[-1]) // self.patch_size
        if grid_h * grid_w != patch_count:
            raise ValueError(
                f"offline ball patch grid mismatch: {grid_h}x{grid_w}!={patch_count}"
            )
        grid_y, grid_x = torch.meshgrid(
            torch.arange(grid_h, dtype=torch.float32) + 0.5,
            torch.arange(grid_w, dtype=torch.float32) + 0.5,
            indexing="ij",
        )
        presence_targets = batch["object_motion_presence_targets"]
        presence_masks = batch["object_motion_presence_masks"]
        coordinate_targets = batch["object_motion_coordinate_targets"]
        coordinate_masks = batch["object_motion_coordinate_masks"]
        confidences = batch["object_motion_teacher_confidences"]
        motion_quality = batch["object_motion_motion_quality"]
        filled = batch.get("object_motion_teacher_filled")
        if not torch.is_tensor(filled) or filled.shape != (batch_size, frames):
            filled = torch.zeros(batch_size, frames, dtype=torch.float32)

        matched_count = 0
        heatmap_count = 0
        motion_count = 0
        missing_video_count = 0
        query_times = times.detach().cpu().numpy()
        for sample_index, meta in enumerate(metas):
            video_id = str(meta.get("video_id", ""))
            index = self._load(video_id)
            if index is None or not len(index.get("timestamp_sec", ())):
                missing_video_count += 1
                continue
            selected, deltas = self._nearest_indices(
                index["timestamp_sec"].astype(np.float64, copy=False),
                query_times[sample_index].astype(np.float64, copy=False),
            )
            valid = deltas <= self.max_time_delta_sec
            for frame_index in np.flatnonzero(valid):
                row = int(selected[frame_index])
                box = index["bbox_xyxy_norm"][row].astype(np.float32, copy=False)
                confidence = float(index["confidence"][row])
                quality = float(index["quality_weight"][row])
                flags = int(index["flags"][row])
                heatmap_usable = bool(flags & 1)
                motion_usable = bool(flags & 2)
                x1, y1, x2, y2 = (float(value) for value in box)
                cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
                matched_count += 1
                filled[sample_index, frame_index] = 1.0
                presence_targets[sample_index, frame_index, 0] = 1.0
                confidences[sample_index, frame_index, 0] = confidence
                # Interpolation supports visibility and motion, but never acts
                # as a strong localization target.
                presence_masks[sample_index, frame_index, 0] = (
                    quality if heatmap_usable else 0.25 * quality
                )
                if heatmap_usable:
                    target = self._heatmap(
                        cx,
                        cy,
                        grid_h=grid_h,
                        grid_w=grid_w,
                        grid_y=grid_y,
                        grid_x=grid_x,
                    ).flatten()
                    targets[sample_index, frame_index, :, 0] = target
                    masks[sample_index, frame_index, :, 0] = quality
                    coordinate_targets[sample_index, frame_index, 0] = torch.tensor(
                        [2.0 * cx - 1.0, 2.0 * cy - 1.0, 2.0 * (x2 - x1), 2.0 * (y2 - y1)],
                        dtype=torch.float32,
                    )
                    coordinate_masks[sample_index, frame_index, 0] = quality
                    heatmap_count += 1
                if motion_usable:
                    motion_quality[sample_index, frame_index, 0] = quality
                    motion_count += 1

        batch["object_motion_teacher_filled"] = filled
        batch["object_motion_heatmap_targets"] = targets
        batch["object_motion_heatmap_masks"] = masks
        elapsed = time.perf_counter() - started
        total = max(batch_size * frames, 1)
        return {
            "offline_ball_teacher_frames": float(total),
            "offline_ball_teacher_matched": float(matched_count),
            "offline_ball_teacher_match_fraction": float(matched_count / total),
            "offline_ball_teacher_heatmap_fraction": float(heatmap_count / total),
            "offline_ball_teacher_motion_fraction": float(motion_count / total),
            "offline_ball_teacher_missing_videos": float(missing_video_count),
            "offline_ball_teacher_seconds": float(elapsed),
        }
