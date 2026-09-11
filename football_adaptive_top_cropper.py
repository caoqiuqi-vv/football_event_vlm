from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


class AdaptiveTopWindowCropper:
    """One detector-guided, aspect-preserving crop shared by a whole clip."""

    def __init__(self, index_root: str, target_aspect: float, min_ratio: float = 0.05,
                 max_ratio: float = 0.25, fallback_ratio: float = 0.10,
                 person_conf: float = 0.35):
        self.index_root = Path(index_root).expanduser()
        self.target_aspect = max(float(target_aspect), 1e-6)
        self.min_ratio = float(min_ratio)
        self.max_ratio = float(max_ratio)
        self.fallback_ratio = float(fallback_ratio)
        self.person_conf = float(person_conf)
        if not 0 <= self.min_ratio <= self.max_ratio < 0.5:
            raise ValueError("adaptive ratios must satisfy 0 <= min <= max < 0.5")
        if not self.min_ratio <= self.fallback_ratio <= self.max_ratio:
            raise ValueError("fallback ratio must lie within [min_ratio, max_ratio]")
        self._payloads: dict[str, dict[str, Any]] = {}
        self._geometries: dict[str, tuple[int, int, float, int]] = {}

    def _load(self, video_id: str) -> dict[str, Any]:
        if video_id not in self._payloads:
            path = self.index_root / f"{video_id}.pt"
            if not path.exists():
                raise FileNotFoundError(f"Missing detector index: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(payload, dict) or int(payload.get("version", 0)) != 2:
                raise ValueError(f"Unsupported detector index: {path}")
            self._payloads[video_id] = payload
        return self._payloads[video_id]

    def _geometry(self, video_path: str) -> tuple[int, int, float, int]:
        if video_path not in self._geometries:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                cap.release()
                raise FileNotFoundError(f"Could not open video: {video_path}")
            value = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
                     float(cap.get(cv2.CAP_PROP_FPS) or 0),
                     int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
            cap.release()
            if min(value) <= 0:
                raise RuntimeError(f"Invalid video geometry: {video_path}: {value}")
            self._geometries[video_path] = value
        return self._geometries[video_path]

    def get_window_roi(self, video_path: str, video_id: str, start_sec: float,
                       end_sec: float) -> tuple[tuple[int, int, int, int], dict[str, Any]]:
        width, height, fps, frame_count = self._geometry(video_path)
        payload = self._load(video_id)
        start_frame = max(0, min(round(start_sec * fps), frame_count - 1))
        end_frame = max(start_frame, min(round(end_sec * fps), frame_count - 1))
        frame_ids = payload["frame_ids"].numpy()
        offsets = payload["frame_offsets"].numpy()
        lo = int(np.searchsorted(frame_ids, start_frame, side="left"))
        hi = int(np.searchsorted(frame_ids, end_frame, side="right"))
        object_lo = int(offsets[min(lo, len(offsets) - 1)])
        object_hi = int(offsets[min(hi, len(offsets) - 1)])
        boxes = payload["boxes"][object_lo:object_hi].float().numpy()
        classes = payload["classes"][object_lo:object_hi].numpy()
        confidences = payload["confidences"][object_lo:object_hi].float().numpy()

        det_size = payload.get("image_size", {}) or {}
        if len(boxes):
            boxes = boxes.astype(np.float64, copy=True)
            boxes[:, (0, 2)] *= width / max(float(det_size.get("width", width)), 1)
            boxes[:, (1, 3)] *= height / max(float(det_size.get("height", height)), 1)
        frame_area = float(width * height)
        areas = (np.maximum(boxes[:, 2] - boxes[:, 0], 0)
                 * np.maximum(boxes[:, 3] - boxes[:, 1], 0)) if len(boxes) else np.empty(0)
        mask = ((classes == 0) & (confidences >= self.person_conf)
                & (areas >= 0.0005 * frame_area) & (areas <= 0.15 * frame_area)) \
            if len(boxes) else np.zeros(0, dtype=bool)
        people = boxes[mask]
        if len(people) >= 3:
            content_top = float(np.quantile(people[:, 1], 0.01)) - 0.05 * height
            top_ratio = float(np.clip(content_top / height, self.min_ratio, self.max_ratio))
            center_x = float(np.median((people[:, 0] + people[:, 2]) * 0.5))
            reason = "people_quantile"
        else:
            top_ratio = self.fallback_ratio
            center_x = width * 0.5
            reason = "fallback_insufficient_people"

        top = int(np.clip(round(top_ratio * height), 0, height - 1))
        crop_height = height - top
        crop_width = min(float(width), crop_height * self.target_aspect)
        left = float(np.clip(center_x - crop_width * 0.5, 0, max(width - crop_width, 0)))
        if len(people) >= 3:
            need_left = float(np.quantile(people[:, 0], 0.05))
            need_right = float(np.quantile(people[:, 2], 0.95))
            if need_right - need_left <= crop_width:
                left = min(left, need_left)
                left = max(left, need_right - crop_width)
                left = float(np.clip(left, 0, max(width - crop_width, 0)))

        x1 = int(np.clip(round(left), 0, width - 1))
        x2 = int(np.clip(round(left + crop_width), x1 + 1, width))
        roi = (x1, top, x2, height)
        return roi, {
            "crop_mode": "adaptive_top_fixed",
            "crop_roi": list(roi),
            "crop_area_ratio": (x2 - x1) * (height - top) / frame_area,
            "crop_target_aspect": self.target_aspect,
            "crop_reason": reason,
            "crop_person_count": int(len(people)),
            "adaptive_top_ratio": top / height,
            "adaptive_person_count": int(len(people)),
        }
