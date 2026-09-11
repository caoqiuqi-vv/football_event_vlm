from __future__ import annotations

import math
import random
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import cv2

from football_roi_scoring import (
    rank_ball_candidates,
    rank_object_candidates,
    select_ball_candidate,
    select_goal_candidate,
    stable_nearby_people,
)


ROI_MODES = ("goal_ball", "goal_players", "ball_players", "center_circle_players")
ROI_FALLBACK_GROUPS = ("ok", "train_augmented", "low_confidence", "area_invalid", "missing_or_no_anchor")
ROI_META_DIM = 7 + len(ROI_MODES) + len(ROI_FALLBACK_GROUPS)


def _normalized_payload_boxes(
    payload: dict[str, Any],
    key: str,
    lo: int | None = None,
    hi: int | None = None,
) -> np.ndarray:
    """Read normalized xyxy boxes, with a legacy absolute-coordinate fallback."""
    absolute_value = payload[key]
    absolute = (
        absolute_value.float().numpy()
        if isinstance(absolute_value, torch.Tensor)
        else np.asarray(absolute_value, dtype=np.float32)
    ).reshape(-1, 4)
    normalized_value = payload.get(f"{key}_normalized")
    if normalized_value is not None:
        normalized = (
            normalized_value.float().numpy()
            if isinstance(normalized_value, torch.Tensor)
            else np.asarray(normalized_value, dtype=np.float32)
        ).reshape(-1, 4)
        if normalized.shape != absolute.shape:
            raise ValueError(
                f"{key}_normalized shape {normalized.shape} does not match {key} shape {absolute.shape}"
            )
        if not np.isfinite(normalized).all():
            raise ValueError(f"{key}_normalized contains non-finite values")
        if len(normalized) and (float(normalized.min()) < -1e-4 or float(normalized.max()) > 1.0001):
            raise ValueError(f"{key}_normalized must be within [0, 1]")
        normalized = np.clip(normalized, 0.0, 1.0)
    else:
        image_size = payload.get("image_size", {})
        reference_width = max(float(image_size.get("width", 0.0)), 1.0)
        reference_height = max(float(image_size.get("height", 0.0)), 1.0)
        normalized = absolute.astype(np.float64, copy=True)
        normalized[:, (0, 2)] /= reference_width
        normalized[:, (1, 3)] /= reference_height
    return normalized[slice(lo, hi)]


def _payload_boxes_for_frame(
    payload: dict[str, Any],
    key: str,
    width: int,
    height: int,
    lo: int | None = None,
    hi: int | None = None,
) -> np.ndarray:
    boxes = _normalized_payload_boxes(payload, key, lo, hi).astype(np.float64, copy=True)
    boxes[:, (0, 2)] *= float(width)
    boxes[:, (1, 3)] *= float(height)
    return boxes


class DetectionHintRenderer:
    """Render confidence-weighted detector hints without replacing RGB pixels."""

    def __init__(self, cfg: Any):
        root = str(cfg.get("index_root", "")).strip()
        if not root:
            raise ValueError("detection_hint.index_root is required")
        self.index_root = Path(root).expanduser()
        self.topk_ball = max(int(cfg.get("topk_ball", 3)), 0)
        self.topk_goal = max(int(cfg.get("topk_goal", 1)), 0)
        self.ball_conf = float(cfg.get("ball_conf", 0.10))
        self.goal_conf = float(cfg.get("goal_conf", 0.40))
        self.ball_marker_scale = max(float(cfg.get("ball_marker_scale", 3.0)), 1.0)
        self.ball_min_radius_ratio = max(float(cfg.get("ball_min_radius_ratio", 0.006)), 0.0)
        self.ball_max_radius_ratio = max(float(cfg.get("ball_max_radius_ratio", 0.025)), 0.0)
        self.ball_alpha = float(np.clip(float(cfg.get("ball_alpha", 0.70)), 0.0, 1.0))
        self.goal_alpha = float(np.clip(float(cfg.get("goal_alpha", 0.45)), 0.0, 1.0))
        self.line_width_ratio = max(float(cfg.get("line_width_ratio", 0.002)), 0.0005)
        self.max_frame_gap_sec = max(float(cfg.get("max_frame_gap_sec", 0.35)), 0.0)
        self.raw_rgb_prob = float(np.clip(float(cfg.get("raw_rgb_prob", 0.20)), 0.0, 1.0))
        self.ball_dropout_prob = float(np.clip(float(cfg.get("ball_dropout_prob", 0.20)), 0.0, 1.0))
        self.goal_dropout_prob = float(np.clip(float(cfg.get("goal_dropout_prob", 0.20)), 0.0, 1.0))
        self.center_jitter_ratio = max(float(cfg.get("center_jitter_ratio", 0.08)), 0.0)
        self.radius_jitter_ratio = max(float(cfg.get("radius_jitter_ratio", 0.12)), 0.0)
        self.max_cached_videos = max(int(cfg.get("max_cached_videos", 2)), 1)
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

    @classmethod
    def from_config(cls, cfg: Any) -> "DetectionHintRenderer | None":
        hint_cfg = cfg.get("detection_hint", {})
        return cls(hint_cfg) if bool(hint_cfg.get("enabled", False)) else None

    def has_video(self, video_id: str) -> bool:
        return (self.index_root / f"{video_id}.pt").exists()

    def _load(self, video_id: str) -> dict[str, Any]:
        cached = self._cache.pop(video_id, None)
        if cached is not None:
            self._cache[video_id] = cached
            return cached
        path = self.index_root / f"{video_id}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing detection hint index for video_id={video_id}: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or int(payload.get("version", 0)) != 2:
            raise ValueError(f"Unsupported detection hint index: {path}")
        if str(payload.get("frame_id_semantics", "")) != "original_source_frame_id":
            raise ValueError(f"Detection hint index is not source-frame aligned: {path}")
        self._cache[video_id] = payload
        while len(self._cache) > self.max_cached_videos:
            self._cache.popitem(last=False)
        return payload

    def _objects_for_frame(
        self, payload: dict[str, Any], frame_id: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        frame_ids = payload["frame_ids"].numpy()
        empty = (
            np.empty(0, dtype=np.int8),
            np.empty(0, dtype=np.float32),
            np.empty((0, 4), dtype=np.float32),
        )
        if len(frame_ids) == 0:
            return empty
        position = int(np.searchsorted(frame_ids, int(frame_id), side="left"))
        candidates = [i for i in (position - 1, position) if 0 <= i < len(frame_ids)]
        nearest = min(candidates, key=lambda i: abs(int(frame_ids[i]) - int(frame_id)))
        fps = max(float(payload.get("fps", 0.0)), 1e-6)
        if abs(int(frame_ids[nearest]) - int(frame_id)) / fps > self.max_frame_gap_sec:
            return empty
        offsets = payload["frame_offsets"].numpy()
        lo, hi = int(offsets[nearest]), int(offsets[nearest + 1])
        return (
            payload["classes"][lo:hi].numpy(),
            payload["confidences"][lo:hi].float().numpy(),
            _normalized_payload_boxes(payload, "boxes", lo, hi),
        )

    @staticmethod
    def _scaled_box(box: Sequence[float], width: int, height: int) -> np.ndarray:
        result = np.asarray(box, dtype=np.float64).copy()
        result[[0, 2]] *= float(width)
        result[[1, 3]] *= float(height)
        return result

    @staticmethod
    def _blend(frame: np.ndarray, overlay: np.ndarray, alpha: float) -> np.ndarray:
        return cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0)

    def render(
        self, frame: np.ndarray, video_id: str, frame_id: int, *, is_train: bool
    ) -> np.ndarray:
        if is_train and random.random() < self.raw_rgb_prob:
            return frame
        payload = self._load(video_id)
        classes, confidences, boxes = self._objects_for_frame(payload, frame_id)
        if len(classes) == 0:
            return frame
        height, width = frame.shape[:2]
        result = frame.copy()
        line_width = max(int(round(min(width, height) * self.line_width_ratio)), 1)

        ball_indices = np.flatnonzero((classes == 1) & (confidences >= self.ball_conf))
        if len(ball_indices):
            ball_indices = ball_indices[np.argsort(-confidences[ball_indices])[: self.topk_ball]]
        for index in ball_indices:
            if is_train and random.random() < self.ball_dropout_prob:
                continue
            box = self._scaled_box(boxes[index], width, height)
            center_x = float(box[0] + box[2]) * 0.5
            center_y = float(box[1] + box[3]) * 0.5
            radius = 0.5 * max(float(box[2] - box[0]), float(box[3] - box[1]), 1.0)
            radius *= self.ball_marker_scale
            min_radius = min(width, height) * self.ball_min_radius_ratio
            max_radius = max(min(width, height) * self.ball_max_radius_ratio, min_radius)
            radius = float(np.clip(radius, min_radius, max_radius))
            if is_train:
                center_x += random.uniform(-1.0, 1.0) * radius * self.center_jitter_ratio
                center_y += random.uniform(-1.0, 1.0) * radius * self.center_jitter_ratio
                radius *= 1.0 + random.uniform(-1.0, 1.0) * self.radius_jitter_ratio
            center = (
                int(np.clip(round(center_x), 0, width - 1)),
                int(np.clip(round(center_y), 0, height - 1)),
            )
            overlay = result.copy()
            cv2.circle(overlay, center, max(int(round(radius)), 2), (0, 215, 255), line_width * 2)
            cv2.circle(overlay, center, line_width, (0, 215, 255), -1)
            alpha = self.ball_alpha * float(np.clip(confidences[index], 0.25, 1.0))
            result = self._blend(result, overlay, alpha)

        goal_indices = np.flatnonzero((classes == 2) & (confidences >= self.goal_conf))
        if len(goal_indices):
            goal_indices = goal_indices[np.argsort(-confidences[goal_indices])[: self.topk_goal]]
        for index in goal_indices:
            if is_train and random.random() < self.goal_dropout_prob:
                continue
            box = self._scaled_box(boxes[index], width, height)
            x1 = int(np.clip(round(box[0]), 0, width - 1))
            y1 = int(np.clip(round(box[1]), 0, height - 1))
            x2 = int(np.clip(round(box[2]), x1 + 1, width))
            y2 = int(np.clip(round(box[3]), y1 + 1, height))
            overlay = result.copy()
            cv2.rectangle(overlay, (x1, y1), (x2 - 1, y2 - 1), (255, 200, 0), line_width)
            alpha = self.goal_alpha * float(np.clip(confidences[index], 0.25, 1.0))
            result = self._blend(result, overlay, alpha)
        return result


@dataclass(frozen=True)
class ROIProposal:
    bbox: tuple[int, int, int, int] | None
    valid: bool
    mode: str = "invalid"
    roi_confidence: float = 0.0
    goal_score: float = 0.0
    ball_score: float = 0.0
    center_circle_score: float = 0.0
    person_support: float = 0.0
    area_ratio: float = 0.0
    fallback_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["bbox"] = list(self.bbox) if self.bbox is not None else None
        return result

    def meta_vector(self) -> torch.Tensor:
        mode_values = [1.0 if self.mode == mode else 0.0 for mode in ROI_MODES]
        reason = self.fallback_reason
        if reason in ("", "ok"):
            fallback_group = "ok"
        elif reason.startswith("train_"):
            fallback_group = "train_augmented"
        elif "confidence" in reason:
            fallback_group = "low_confidence"
        elif "area" in reason or "aspect" in reason or "large" in reason:
            fallback_group = "area_invalid"
        else:
            fallback_group = "missing_or_no_anchor"
        fallback_values = [1.0 if fallback_group == group else 0.0 for group in ROI_FALLBACK_GROUPS]
        return torch.tensor(
            [
                float(self.valid),
                float(self.roi_confidence),
                float(self.area_ratio),
                float(self.goal_score),
                float(self.ball_score),
                float(self.center_circle_score),
                float(self.person_support),
                *mode_values,
                *fallback_values,
            ],
            dtype=torch.float32,
        )


def invalid_roi(reason: str, **scores: float) -> ROIProposal:
    return ROIProposal(
        bbox=None,
        valid=False,
        fallback_reason=reason,
        goal_score=float(scores.get("goal_score", 0.0)),
        ball_score=float(scores.get("ball_score", 0.0)),
        center_circle_score=float(scores.get("center_circle_score", 0.0)),
        person_support=float(scores.get("person_support", 0.0)),
    )


def _box_area(box: Sequence[float]) -> float:
    return max(float(box[2]) - float(box[0]), 0.0) * max(float(box[3]) - float(box[1]), 0.0)


def _box_center(box: Sequence[float]) -> tuple[float, float]:
    return (float(box[0]) + float(box[2])) * 0.5, (float(box[1]) + float(box[3])) * 0.5


def _box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(right - left, 0.0) * max(bottom - top, 0.0)
    union = _box_area(first) + _box_area(second) - intersection
    return intersection / max(union, 1e-9)


def _center_inside(box: Sequence[float], region: Sequence[float]) -> bool:
    center_x, center_y = _box_center(box)
    return (
        float(region[0]) <= center_x <= float(region[2])
        and float(region[1]) <= center_y <= float(region[3])
    )


def _robust_box(boxes: np.ndarray, low: float = 0.1, high: float = 0.9) -> list[float]:
    if len(boxes) == 1:
        return boxes[0].astype(np.float64).tolist()
    return [
        float(np.quantile(boxes[:, 0], low)),
        float(np.quantile(boxes[:, 1], low)),
        float(np.quantile(boxes[:, 2], high)),
        float(np.quantile(boxes[:, 3], high)),
    ]


def _union(boxes: Sequence[Sequence[float]]) -> list[float]:
    return [
        min(float(box[0]) for box in boxes),
        min(float(box[1]) for box in boxes),
        max(float(box[2]) for box in boxes),
        max(float(box[3]) for box in boxes),
    ]


def _distance_to_box(point: tuple[float, float], box: Sequence[float]) -> float:
    x, y = point
    dx = max(float(box[0]) - x, 0.0, x - float(box[2]))
    dy = max(float(box[1]) - y, 0.0, y - float(box[3]))
    return math.hypot(dx, dy)


class _Kalman1D:
    def __init__(self, process_noise: float, measurement_noise: float):
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.estimate: float | None = None
        self.error = 1.0

    def reset(self, value: float) -> float:
        self.estimate = float(value)
        self.error = 1.0
        return self.estimate

    def update(self, value: float, confidence: float = 1.0) -> float:
        if self.estimate is None:
            return self.reset(value)
        self.error += self.process_noise
        measurement_noise = self.measurement_noise / max(float(confidence), 0.05)
        gain = self.error / (self.error + measurement_noise)
        self.estimate += gain * (float(value) - self.estimate)
        self.error *= 1.0 - gain
        return self.estimate


class RobustClipCropper:
    """Build static clip ROIs or smoothed frame-aligned dynamic ROIs.

    The class consumes compact ``.pt`` indices produced by
    ``scripts/build_football_roi_indices.py``. Keeping the compact index separate
    avoids loading 400-900 MB per-frame JSON files in every DataLoader worker.
    """

    def __init__(self, cfg: Any):
        self.index_root = Path(str(cfg.get("index_root", ""))).expanduser()
        if not self.index_root:
            raise ValueError("spatial_crop.index_root is required for robust_detector_aware")
        self.padding = float(cfg.get("padding", 0.15))
        self.min_crop_area_ratio = float(cfg.get("min_crop_area_ratio", 0.0))
        self.max_crop_area_ratio = float(cfg.get("max_crop_area_ratio", 0.35))
        self.min_roi_confidence = float(cfg.get("min_roi_confidence", 0.40))
        self.goal_conf = float(cfg.get("goal_conf", 0.40))
        self.person_conf = float(cfg.get("person_conf", 0.35))
        self.center_circle_conf = float(cfg.get("center_circle_conf", 0.35))
        self.min_goal_frames = int(cfg.get("min_goal_frames", 3))
        self.min_ball_points = int(cfg.get("min_ball_points", 5))
        self.min_person_area_ratio = float(cfg.get("min_person_area_ratio", 0.0005))
        self.max_person_area_ratio = float(cfg.get("max_person_area_ratio", 0.15))
        self.min_person_frames = int(cfg.get("min_person_frames", 5))
        self.max_people = int(cfg.get("max_people", 10))
        self.max_cached_videos = max(int(cfg.get("max_cached_videos", 2)), 1)
        self.raw_ball_conf = float(cfg.get("raw_ball_conf", 0.10))
        self.ball_min_area_ratio = float(cfg.get("ball_min_area_ratio", 1e-6))
        self.ball_max_area_ratio = float(cfg.get("ball_max_area_ratio", 2e-3))
        self.goal_cc_min_angle_deg = float(cfg.get("goal_cc_min_angle_deg", 60.0))
        self.complementary_max_iou = float(cfg.get("complementary_max_iou", 0.45))
        self.complementary_min_people = max(
            int(cfg.get("complementary_min_people", 2)), 0
        )
        self.complementary_shift_ratios = tuple(
            float(value)
            for value in cfg.get(
                "complementary_shift_ratios", [-0.85, -0.60, 0.60, 0.85]
            )
        )
        self.temporal_mode = str(cfg.get("temporal_mode", "clip"))
        if self.temporal_mode not in ("clip", "dynamic"):
            raise ValueError("spatial_crop.temporal_mode must be clip or dynamic")
        self.dynamic_context_sec = max(float(cfg.get("dynamic_context_sec", 3.0)), 0.25)
        self.temporal_smoothing_window_sec = max(
            float(cfg.get("temporal_smoothing_window_sec", 1.5)), 0.0
        )
        self.temporal_max_hold_sec = max(float(cfg.get("temporal_max_hold_sec", 1.5)), 0.0)
        self.temporal_confidence_decay_sec = max(
            float(cfg.get("temporal_confidence_decay_sec", 1.0)), 1e-6
        )
        self.temporal_process_noise = max(float(cfg.get("temporal_process_noise", 1e-2)), 1e-9)
        self.temporal_measurement_noise = max(
            float(cfg.get("temporal_measurement_noise", 1e-1)), 1e-9
        )
        self.temporal_max_center_jump_ratio = max(
            float(cfg.get("temporal_max_center_jump_ratio", 0.30)), 0.0
        )
        self.temporal_causal = bool(cfg.get("temporal_causal", False))
        self.dynamic_fallback_to_clip = bool(cfg.get("dynamic_fallback_to_clip", True))
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

    @classmethod
    def from_config(cls, cfg: Any) -> "RobustClipCropper | None":
        spatial_cfg = cfg.get("spatial_crop", {})
        if str(spatial_cfg.get("mode", "none")) != "robust_detector_aware":
            return None
        return cls(spatial_cfg)

    def has_video(self, video_id: str) -> bool:
        return (self.index_root / f"{video_id}.pt").exists()

    def _load(self, video_id: str) -> dict[str, Any]:
        cached = self._cache.pop(video_id, None)
        if cached is not None:
            self._cache[video_id] = cached
            return cached
        path = self.index_root / f"{video_id}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing robust ROI index for video_id={video_id}: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or int(payload.get("version", 0)) != 2:
            raise ValueError(f"Unsupported robust ROI index: {path}")
        self._cache[video_id] = payload
        while len(self._cache) > self.max_cached_videos:
            self._cache.popitem(last=False)
        return payload

    @staticmethod
    def _scaled_boxes(boxes: np.ndarray, det_width: float, det_height: float, width: int, height: int) -> np.ndarray:
        result = boxes.astype(np.float64, copy=True)
        result[:, (0, 2)] *= float(width) / max(det_width, 1.0)
        result[:, (1, 3)] *= float(height) / max(det_height, 1.0)
        return result

    @staticmethod
    def _synthetic_track_ids(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
        centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
        widths = np.maximum(boxes[:, 2] - boxes[:, 0], 1.0)
        heights = np.maximum(boxes[:, 3] - boxes[:, 1], 1.0)
        x_bin = np.floor(centers[:, 0] / max(width * 0.08, 1.0)).astype(np.int64)
        y_bin = np.floor(centers[:, 1] / max(height * 0.08, 1.0)).astype(np.int64)
        size_bin = np.floor(np.log2(np.maximum(widths * heights, 1.0))).astype(np.int64)
        return -(1 + x_bin + 100 * y_bin + 10000 * size_bin)

    def _best_object_track(
        self,
        boxes: np.ndarray,
        confidences: np.ndarray,
        track_ids: np.ndarray,
        frame_ids: np.ndarray,
        *,
        width: int,
        height: int,
        min_frames: int,
    ) -> tuple[list[float] | None, float, int]:
        if len(boxes) == 0:
            return None, 0.0, 0
        track_ids = track_ids.astype(np.int64, copy=True)
        missing = track_ids < 0
        if missing.any():
            track_ids[missing] = self._synthetic_track_ids(boxes[missing], width, height)
        best_box: list[float] | None = None
        best_score = 0.0
        best_count = 0
        diagonal = max(math.hypot(width, height), 1.0)
        for track_id in np.unique(track_ids):
            selected = track_ids == track_id
            unique_frames = len(np.unique(frame_ids[selected]))
            if unique_frames < min_frames:
                continue
            candidate_boxes = boxes[selected]
            centers = (candidate_boxes[:, :2] + candidate_boxes[:, 2:]) * 0.5
            center_std = float(np.linalg.norm(np.std(centers, axis=0))) / diagonal
            persistence = min(unique_frames / max(float(min_frames * 2), 1.0), 1.0)
            median_conf = float(np.median(confidences[selected]))
            stability = math.exp(-center_std / 0.06)
            score = 0.40 * persistence + 0.35 * median_conf + 0.25 * stability
            if score > best_score:
                best_box = _robust_box(candidate_boxes)
                best_score = score
                best_count = unique_frames
        return best_box, min(best_score, 1.0), best_count

    def _best_ball_track(
        self,
        payload: dict[str, Any],
        start_frame: int,
        end_frame: int,
        *,
        width: int,
        height: int,
        people_boxes: np.ndarray,
    ) -> tuple[list[float] | None, float, int]:
        offsets = payload["ball_track_offsets"].numpy()
        frame_ids = payload["ball_frame_ids"].numpy()
        raw_boxes = _payload_boxes_for_frame(
            payload, "ball_boxes", width, height
        )
        point_touched = payload["ball_point_touched"].numpy().astype(bool)
        track_touched = payload["ball_track_touched"].numpy().astype(bool)
        fps = float(payload["fps"])
        diagonal = max(math.hypot(width, height), 1.0)
        people_centers = (people_boxes[:, :2] + people_boxes[:, 2:]) * 0.5 if len(people_boxes) else np.empty((0, 2))

        best_box: list[float] | None = None
        best_score = 0.0
        best_points = 0
        for track_index in range(len(offsets) - 1):
            lo, hi = int(offsets[track_index]), int(offsets[track_index + 1])
            ids = frame_ids[lo:hi]
            selected = (ids >= start_frame) & (ids <= end_frame)
            if int(selected.sum()) < self.min_ball_points:
                continue
            ids = ids[selected]
            boxes = raw_boxes[lo:hi][selected]
            widths = np.maximum(boxes[:, 2] - boxes[:, 0], 1.0)
            heights = np.maximum(boxes[:, 3] - boxes[:, 1], 1.0)
            median_area_ratio = float(np.median(widths * heights)) / max(float(width * height), 1.0)
            median_aspect = float(np.median(widths / heights))
            plausible = float(
                self.ball_min_area_ratio <= median_area_ratio <= self.ball_max_area_ratio
                and 0.35 <= median_aspect <= 2.85
            )
            duration = max(float(ids[-1] - ids[0]) / max(fps, 1e-6), 0.0)
            persistence = min(duration / 1.0, 1.0)
            centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
            if len(centers) >= 3:
                steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
                step_median = float(np.median(steps))
                step_mad = float(np.median(np.abs(steps - step_median)))
                smoothness = 1.0 / (1.0 + step_mad / max(step_median, 1.0))
            else:
                smoothness = 0.5
            if len(people_centers):
                distances = np.linalg.norm(centers[:, None, :] - people_centers[None, :, :], axis=-1)
                proximity = math.exp(-float(np.min(distances)) / (0.08 * diagonal))
            else:
                proximity = 0.0
            touched = bool(track_touched[track_index]) or bool(point_touched[lo:hi][selected].any())
            score = 0.30 * persistence + 0.20 * plausible + 0.15 * smoothness + 0.25 * proximity + 0.10 * float(touched)
            if plausible <= 0 and not touched:
                score *= 0.25
            if score > best_score:
                best_box = _robust_box(boxes)
                best_score = score
                best_points = len(ids)
        return best_box, min(best_score, 1.0), best_points

    def _nearby_people(
        self,
        boxes: np.ndarray,
        track_ids: np.ndarray,
        anchor_boxes: Sequence[Sequence[float]],
        *,
        width: int,
        height: int,
    ) -> tuple[list[list[float]], float]:
        if len(boxes) == 0 or not anchor_boxes:
            return [], 0.0
        anchor = _union(anchor_boxes)
        anchor_center = _box_center(anchor)
        track_ids = track_ids.astype(np.int64, copy=True)
        missing = track_ids < 0
        if missing.any():
            track_ids[missing] = self._synthetic_track_ids(boxes[missing], width, height)
        candidates: list[tuple[float, list[float]]] = []
        for track_id in np.unique(track_ids):
            person_box = _robust_box(boxes[track_ids == track_id], 0.2, 0.8)
            distance = _distance_to_box(anchor_center, person_box)
            candidates.append((distance, person_box))
        candidates.sort(key=lambda item: item[0])
        selected = [box for _, box in candidates[: self.max_people]]
        if not selected:
            return [], 0.0
        diagonal = max(math.hypot(width, height), 1.0)
        mean_distance = float(np.mean([distance for distance, _ in candidates[: self.max_people]]))
        support = min(len(selected) / max(float(self.max_people), 1.0), 1.0) * math.exp(-mean_distance / (0.15 * diagonal))
        return selected, min(support, 1.0)

    def _fit_roi(
        self,
        boxes: Sequence[Sequence[float]],
        width: int,
        height: int,
        target_size: tuple[int, int],
        focus_boxes: Sequence[Sequence[float]] | None = None,
    ) -> tuple[tuple[int, int, int, int] | None, float, str]:
        if not boxes:
            return None, 0.0, "no_boxes"
        x1, y1, x2, y2 = _union(boxes)
        pad_x = max(x2 - x1, 1.0) * self.padding
        pad_y = max(y2 - y1, 1.0) * self.padding
        x1, y1 = max(x1 - pad_x, 0.0), max(y1 - pad_y, 0.0)
        x2, y2 = min(x2 + pad_x, float(width)), min(y2 + pad_y, float(height))
        box_w, box_h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        output_h, output_w = map(int, target_size)
        if output_h <= 0 or output_w <= 0:
            raise ValueError(f"Invalid ROI target_size={target_size}")
        min_area = self.min_crop_area_ratio * width * height
        required_scale = max(
            1,
            math.ceil(box_w / output_w - 1e-9),
            math.ceil(box_h / output_h - 1e-9),
            math.ceil(math.sqrt(min_area / (output_w * output_h)) - 1e-9),
        )
        target_w = required_scale * output_w
        target_h = required_scale * output_h
        required_area_ratio = target_w * target_h / max(float(width * height), 1.0)
        if required_area_ratio > self.max_crop_area_ratio + 1e-9:
            return None, required_area_ratio, "required_area_too_large"
        if target_w > width or target_h > height:
            return None, required_area_ratio, "integer_scale_does_not_fit"
        if focus_boxes:
            fx1, fy1, fx2, fy2 = _union(focus_boxes)
            cx, cy = (fx1 + fx2) * 0.5, (fy1 + fy2) * 0.5
        else:
            cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5

        # Keep required boxes inside the crop while placing the crop center near
        # the action focus. Goals are often edge anchors, so ball/people should
        # decide the view when the geometry allows it.
        min_left = max(0.0, x2 - target_w)
        max_left = min(x1, float(width) - target_w)
        min_top = max(0.0, y2 - target_h)
        max_top = min(y1, float(height) - target_h)
        if min_left > max_left + 1e-6 or min_top > max_top + 1e-6:
            return None, required_area_ratio, "required_boxes_do_not_fit"
        desired_left = cx - target_w * 0.5
        desired_top = cy - target_h * 0.5
        left = int(round(min(max(desired_left, min_left), max_left)))
        top = int(round(min(max(desired_top, min_top), max_top)))
        roi = (
            left,
            top,
            left + target_w,
            top + target_h,
        )
        area_ratio = _box_area(roi) / max(float(width * height), 1.0)
        return roi, area_ratio, "ok"

    def _fit_ranked_people_roi(
        self,
        anchor_boxes: Sequence[Sequence[float]],
        ranked_people: Sequence[Sequence[float]],
        width: int,
        height: int,
        target_size: tuple[int, int],
        focus_boxes: Sequence[Sequence[float]] | None = None,
    ) -> tuple[tuple[int, int, int, int] | None, float, str, int]:
        """Keep semantic anchors and remove the lowest-priority people only as needed."""
        last_area, last_reason = 0.0, "no_boxes"
        for keep_count in range(len(ranked_people), -1, -1):
            kept_people = list(ranked_people[:keep_count])
            focus = list(focus_boxes) if focus_boxes else []
            focus.extend(kept_people)
            roi, area_ratio, reason = self._fit_roi(
                list(anchor_boxes) + kept_people,
                width,
                height,
                target_size,
                focus_boxes=focus or None,
            )
            if roi is not None:
                return roi, area_ratio, reason, keep_count
            last_area, last_reason = area_ratio, reason
        return None, last_area, last_reason, 0

    def get_window_roi(
        self,
        video_id: str,
        start_sec: float,
        end_sec: float,
        width: int,
        height: int,
        target_size: tuple[int, int],
    ) -> ROIProposal:
        if not self.has_video(video_id):
            return invalid_roi("missing_index")
        payload = self._load(video_id)
        fps = float(payload["fps"])
        start_frame = max(int(round(start_sec * fps)), 0)
        end_frame = max(int(round(end_sec * fps)) - 1, start_frame)
        sampled_frame_ids = payload["frame_ids"].numpy()
        offsets = payload["frame_offsets"].numpy()
        frame_lo = int(np.searchsorted(sampled_frame_ids, start_frame, side="left"))
        frame_hi = int(np.searchsorted(sampled_frame_ids, end_frame, side="right"))
        if frame_lo >= frame_hi:
            return invalid_roi("no_sampled_detection_frames")
        object_lo, object_hi = int(offsets[frame_lo]), int(offsets[frame_hi])
        classes = payload["classes"][object_lo:object_hi].numpy()
        confidences = payload["confidences"][object_lo:object_hi].float().numpy()
        track_ids = payload["track_ids"][object_lo:object_hi].numpy()
        counts = np.diff(offsets[frame_lo : frame_hi + 1])
        object_frame_ids = np.repeat(sampled_frame_ids[frame_lo:frame_hi], counts)
        boxes = _payload_boxes_for_frame(
            payload, "boxes", width, height, object_lo, object_hi
        )

        people_mask = (classes == 0) & (confidences >= self.person_conf)
        raw_ball_mask = (classes == 1) & (confidences >= self.raw_ball_conf)
        goal_mask = (classes == 2) & (confidences >= self.goal_conf)
        cc_mask = (classes == 3) & (confidences >= self.center_circle_conf)
        people_boxes = boxes[people_mask]
        people_confidences = confidences[people_mask]
        people_track_ids = track_ids[people_mask]
        people_frame_ids = object_frame_ids[people_mask]

        # Filter person boxes by pixel area to remove noisy tiny/large detections
        if len(people_boxes) > 0:
            people_areas = (people_boxes[:, 2] - people_boxes[:, 0]) * (people_boxes[:, 3] - people_boxes[:, 1])
            frame_area = float(width * height)
            area_valid = (
                (people_areas >= self.min_person_area_ratio * frame_area)
                & (people_areas <= self.max_person_area_ratio * frame_area)
            )
            people_boxes = people_boxes[area_valid]
            people_confidences = people_confidences[area_valid]
            people_track_ids = people_track_ids[area_valid]
            people_frame_ids = people_frame_ids[area_valid]

        goal_candidates = rank_object_candidates(
            boxes[goal_mask],
            confidences[goal_mask],
            track_ids[goal_mask],
            object_frame_ids[goal_mask],
            width=width,
            height=height,
            min_frames=self.min_goal_frames,
        )
        cc_candidates = rank_object_candidates(
            boxes[cc_mask],
            confidences[cc_mask],
            track_ids[cc_mask],
            object_frame_ids[cc_mask],
            width=width,
            height=height,
            min_frames=self.min_goal_frames,
        )
        cc_candidate = cc_candidates[0] if cc_candidates else None
        cc_box = list(cc_candidate.box) if cc_candidate is not None else None
        cc_score = float(cc_candidate.score) if cc_candidate is not None else 0.0
        ball_candidates = rank_ball_candidates(
            payload,
            start_frame,
            end_frame,
            width=width,
            height=height,
            people_boxes=people_boxes,
            raw_ball_frames=object_frame_ids[raw_ball_mask],
            raw_ball_boxes=boxes[raw_ball_mask],
            min_points=self.min_ball_points,
            min_area_ratio=self.ball_min_area_ratio,
            max_area_ratio=self.ball_max_area_ratio,
        )
        initial_ball = ball_candidates[0] if ball_candidates else None
        goal_candidate, goal_score = select_goal_candidate(
            goal_candidates,
            initial_ball,
            cc_candidate,
            width=width,
            height=height,
        )
        ball_candidate = select_ball_candidate(
            ball_candidates,
            goal_candidate,
            cc_candidate,
            width=width,
            height=height,
        )
        if ball_candidate is not initial_ball:
            goal_candidate, goal_score = select_goal_candidate(
                goal_candidates,
                ball_candidate,
                cc_candidate,
                width=width,
                height=height,
            )
        goal_box = list(goal_candidate.box) if goal_candidate is not None else None
        ball_box = list(ball_candidate.box) if ball_candidate is not None else None
        ball_score = float(ball_candidate.score) if ball_candidate is not None else 0.0
        if goal_box is not None and ball_box is not None:
            normalized_distance = math.dist(_box_center(goal_box), _box_center(ball_box)) / max(
                math.hypot(width, height), 1.0
            )
            if normalized_distance > 0.80:
                if goal_score >= ball_score:
                    ball_score *= 0.5
                else:
                    goal_score *= 0.5

        mode = "invalid"
        anchor_boxes: list[list[float]] = []
        if goal_box is not None and goal_score >= 0.45 and ball_box is not None and ball_score >= 0.35:
            mode = "goal_ball"
            anchor_boxes = [goal_box, ball_box]
        elif goal_box is not None and goal_score >= 0.50:
            mode = "goal_players"
            anchor_boxes = [goal_box]
        elif ball_box is not None and ball_score >= 0.40:
            mode = "ball_players"
            anchor_boxes = [ball_box]
        elif cc_box is not None and cc_score >= 0.55:
            mode = "center_circle_players"
            anchor_boxes = [cc_box]
        else:
            return invalid_roi(
                "no_reliable_anchor",
                goal_score=goal_score,
                ball_score=ball_score,
                center_circle_score=cc_score,
            )

        nearby_people, person_support = stable_nearby_people(
            people_boxes,
            people_confidences,
            people_track_ids,
            people_frame_ids,
            anchor_boxes,
            width=width,
            height=height,
            max_people=self.max_people,
            min_frames=self.min_person_frames,
        )
        roi, area_ratio, fit_reason, kept_people = self._fit_ranked_people_roi(
            anchor_boxes,
            nearby_people,
            width,
            height,
            target_size,
            focus_boxes=[box for box in (ball_box, cc_box) if box is not None],
        )
        if nearby_people:
            person_support *= kept_people / len(nearby_people)
        if roi is None and mode == "goal_ball":
            # When the combined goal+ball crop is too large, keep ball evidence
            # before falling back to a goal-only crop. Ball position is the more
            # direct anchor for football event timing, while goal evidence remains
            # a strong secondary fallback for shots/saves.
            fallback_modes: list[tuple[str, list[list[float]]]] = []
            if ball_box is not None and ball_score >= 0.40:
                fallback_modes.append(("ball_players", [ball_box]))
            if goal_box is not None and goal_score >= 0.60:
                fallback_modes.append(("goal_players", [goal_box]))
            if goal_box is not None and goal_score >= 0.50 and not any(item[0] == "goal_players" for item in fallback_modes):
                fallback_modes.append(("goal_players", [goal_box]))
            for fallback_mode, fallback_anchors in fallback_modes:
                fallback_people, fallback_support = stable_nearby_people(
                    people_boxes,
                    people_confidences,
                    people_track_ids,
                    people_frame_ids,
                    fallback_anchors,
                    width=width,
                    height=height,
                    max_people=self.max_people,
                    min_frames=self.min_person_frames,
                )
                fallback_focus = [box for box in fallback_anchors if ball_box is not None and box == ball_box]
                fallback_roi, fallback_area, fallback_reason, fallback_kept = self._fit_ranked_people_roi(
                    fallback_anchors,
                    fallback_people,
                    width,
                    height,
                    target_size,
                    focus_boxes=fallback_focus or None,
                )
                if fallback_roi is None:
                    continue
                mode, anchor_boxes = fallback_mode, fallback_anchors
                nearby_people = fallback_people[:fallback_kept]
                person_support = fallback_support * (
                    fallback_kept / len(fallback_people) if fallback_people else 1.0
                )
                roi, area_ratio, fit_reason = fallback_roi, fallback_area, fallback_reason
                break
        if roi is None:
            return invalid_roi(
                fit_reason,
                goal_score=goal_score,
                ball_score=ball_score,
                center_circle_score=cc_score,
                person_support=person_support,
            )

        if mode == "goal_ball":
            confidence = 0.45 * goal_score + 0.45 * ball_score + 0.10 * person_support
        elif mode == "goal_players":
            confidence = min(0.75 * goal_score + 0.25 * person_support, 0.80)
        elif mode == "ball_players":
            confidence = min(0.75 * ball_score + 0.25 * person_support, 0.80)
        else:
            confidence = min(0.75 * cc_score + 0.25 * person_support, 0.65)
        if confidence < self.min_roi_confidence:
            return invalid_roi(
                "low_roi_confidence",
                goal_score=goal_score,
                ball_score=ball_score,
                center_circle_score=cc_score,
                person_support=person_support,
            )
        return ROIProposal(
            bbox=roi,
            valid=True,
            mode=mode,
            roi_confidence=float(confidence),
            goal_score=float(goal_score),
            ball_score=float(ball_score),
            center_circle_score=float(cc_score),
            person_support=float(person_support),
            area_ratio=float(area_ratio),
            fallback_reason="ok",
        )

    def get_complementary_window_roi(
        self,
        video_id: str,
        start_sec: float,
        end_sec: float,
        width: int,
        height: int,
        target_size: tuple[int, int],
        primary: ROIProposal,
    ) -> ROIProposal:
        """Select a different crop that adds trusted object coverage."""
        if not primary.valid or primary.bbox is None:
            return invalid_roi("missing_primary_roi")
        if not self.has_video(video_id):
            return invalid_roi("missing_index")

        payload = self._load(video_id)
        fps = float(payload["fps"])
        start_frame = max(int(round(start_sec * fps)), 0)
        end_frame = max(int(round(end_sec * fps)) - 1, start_frame)
        sampled_frame_ids = payload["frame_ids"].numpy()
        offsets = payload["frame_offsets"].numpy()
        frame_lo = int(np.searchsorted(sampled_frame_ids, start_frame, side="left"))
        frame_hi = int(np.searchsorted(sampled_frame_ids, end_frame, side="right"))
        if frame_lo >= frame_hi:
            return invalid_roi("no_sampled_detection_frames")

        object_lo, object_hi = int(offsets[frame_lo]), int(offsets[frame_hi])
        classes = payload["classes"][object_lo:object_hi].numpy()
        confidences = payload["confidences"][object_lo:object_hi].float().numpy()
        boxes = _payload_boxes_for_frame(
            payload, "boxes", width, height, object_lo, object_hi
        )
        trusted = (
            ((classes == 0) & (confidences >= self.person_conf))
            | ((classes == 1) & (confidences >= self.raw_ball_conf))
            | ((classes == 2) & (confidences >= self.goal_conf))
            | ((classes == 3) & (confidences >= self.center_circle_conf))
        )
        classes = classes[trusted]
        confidences = confidences[trusted]
        boxes = boxes[trusted]
        if len(boxes) == 0:
            return invalid_roi("no_complementary_objects")

        primary_box = tuple(int(value) for value in primary.bbox)
        crop_width = primary_box[2] - primary_box[0]
        crop_height = primary_box[3] - primary_box[1]
        output_h, output_w = map(int, target_size)
        if crop_width <= 0 or crop_height <= 0:
            return invalid_roi("invalid_primary_roi")
        if crop_width % output_w != 0 or crop_height % output_h != 0:
            return invalid_roi("primary_roi_aspect_mismatch")

        primary_center_x, primary_center_y = _box_center(primary_box)

        def place(center_x: float, center_y: float) -> tuple[int, int, int, int]:
            left = int(round(min(max(center_x - crop_width * 0.5, 0.0), width - crop_width)))
            top = int(round(min(max(center_y - crop_height * 0.5, 0.0), height - crop_height)))
            return left, top, left + crop_width, top + crop_height

        candidates: set[tuple[int, int, int, int]] = set()
        for shift_ratio in self.complementary_shift_ratios:
            candidates.add(
                place(
                    primary_center_x + shift_ratio * crop_width,
                    primary_center_y,
                )
            )
        for box in boxes:
            if not _center_inside(box, primary_box):
                candidates.add(place(*_box_center(box)))

        class_weight = {0: 1.0, 1: 4.0, 2: 2.5, 3: 0.75}
        primary_inside = np.asarray(
            [_center_inside(box, primary_box) for box in boxes], dtype=bool
        )
        best: tuple[float, tuple[int, int, int, int], np.ndarray] | None = None
        for candidate in candidates:
            overlap = _box_iou(primary_box, candidate)
            if overlap > self.complementary_max_iou + 1e-9:
                continue
            inside = np.asarray(
                [_center_inside(box, candidate) for box in boxes], dtype=bool
            )
            if not bool(inside.any()):
                continue
            new_inside = inside & ~primary_inside
            people_count = int(np.count_nonzero(inside & (classes == 0)))
            has_anchor = bool(np.any(inside & np.isin(classes, (1, 2))))
            if people_count < self.complementary_min_people and not has_anchor:
                continue
            weighted = np.asarray(
                [class_weight.get(int(class_id), 0.5) for class_id in classes],
                dtype=np.float64,
            ) * confidences.astype(np.float64)
            added_support = float(weighted[new_inside].sum())
            total_support = float(weighted[inside].sum())
            score = 1.75 * added_support + 0.25 * total_support + 0.50 * (1.0 - overlap)
            if best is None or score > best[0]:
                best = score, candidate, inside

        if best is None:
            return invalid_roi("no_complementary_candidate")
        score, candidate, inside = best
        selected_classes = classes[inside]
        selected_confidences = confidences[inside]

        def class_score(class_id: int) -> float:
            values = selected_confidences[selected_classes == class_id]
            return float(values.max()) if len(values) else 0.0

        people_count = int(np.count_nonzero(selected_classes == 0))
        ball_score = class_score(1)
        goal_score = class_score(2)
        center_circle_score = class_score(3)
        if ball_score > 0 and goal_score > 0:
            mode = "goal_ball"
        elif ball_score > 0:
            mode = "ball_players"
        elif goal_score > 0:
            mode = "goal_players"
        else:
            mode = "center_circle_players"
        person_support = min(people_count / max(self.max_people, 1), 1.0)
        confidence = min(0.35 + 0.08 * score, 0.85)
        return ROIProposal(
            bbox=candidate,
            valid=True,
            mode=mode,
            roi_confidence=float(confidence),
            goal_score=goal_score,
            ball_score=ball_score,
            center_circle_score=center_circle_score,
            person_support=person_support,
            area_ratio=_box_area(candidate) / max(float(width * height), 1.0),
            fallback_reason="ok",
        )

    @staticmethod
    def aggregate_temporal_proposals(proposals: Sequence[ROIProposal]) -> ROIProposal:
        valid = [proposal for proposal in proposals if proposal.valid and proposal.bbox is not None]
        if not valid:
            reasons = [proposal.fallback_reason for proposal in proposals if proposal.fallback_reason]
            reason = max(set(reasons), key=reasons.count) if reasons else "no_valid_temporal_roi"
            return invalid_roi(reason)
        weights = np.asarray([max(proposal.roi_confidence, 0.05) for proposal in valid], dtype=np.float64)
        weights /= max(float(weights.sum()), 1e-9)
        mode_scores: dict[str, float] = {}
        for proposal, weight in zip(valid, weights.tolist()):
            mode_scores[proposal.mode] = mode_scores.get(proposal.mode, 0.0) + weight
        mode = max(mode_scores, key=mode_scores.get)
        middle = valid[len(valid) // 2]
        valid_fraction = len(valid) / max(len(proposals), 1)

        def weighted(name: str) -> float:
            return float(sum(float(getattr(proposal, name)) * weight for proposal, weight in zip(valid, weights.tolist())))

        return ROIProposal(
            bbox=middle.bbox,
            valid=True,
            mode=mode,
            roi_confidence=weighted("roi_confidence") * valid_fraction,
            goal_score=weighted("goal_score"),
            ball_score=weighted("ball_score"),
            center_circle_score=weighted("center_circle_score"),
            person_support=weighted("person_support"),
            area_ratio=weighted("area_ratio"),
            fallback_reason="ok",
        )

    def _dynamic_proposal_window(
        self,
        time_sec: float,
        clip_start: float,
        clip_end: float,
    ) -> tuple[float, float]:
        clip_duration = max(float(clip_end) - float(clip_start), 1e-3)
        duration = min(self.dynamic_context_sec, clip_duration)
        start = float(time_sec) - duration * 0.5
        start = min(max(start, float(clip_start)), float(clip_end) - duration)
        return start, start + duration

    def smooth_temporal_proposals(
        self,
        proposals: Sequence[ROIProposal],
        frame_times: Sequence[float],
        width: int,
        height: int,
        target_size: tuple[int, int],
    ) -> list[ROIProposal]:
        if len(proposals) != len(frame_times):
            raise ValueError("proposals and frame_times must have the same length")
        if not proposals:
            return []
        times = np.asarray(frame_times, dtype=np.float64)
        valid_indices = np.asarray(
            [index for index, proposal in enumerate(proposals) if proposal.valid and proposal.bbox is not None],
            dtype=np.int64,
        )
        if len(valid_indices) == 0:
            return list(proposals)

        output_h, output_w = map(int, target_size)
        max_scale = min(width // output_w, height // output_h)
        if max_scale < 1:
            return [invalid_roi("dynamic_target_does_not_fit") for _ in proposals]
        diagonal = max(math.hypot(width, height), 1.0)
        radius = self.temporal_smoothing_window_sec * (1.0 if self.temporal_causal else 0.5)
        kf_cx = _Kalman1D(self.temporal_process_noise, self.temporal_measurement_noise)
        kf_cy = _Kalman1D(self.temporal_process_noise, self.temporal_measurement_noise)
        smoothed: list[ROIProposal] = []
        previous_measurement: tuple[float, float] | None = None

        for index, raw in enumerate(proposals):
            time_sec = float(times[index])
            if self.temporal_causal:
                candidates = [
                    int(candidate)
                    for candidate in valid_indices
                    if 0.0 <= time_sec - float(times[candidate]) <= radius + 1e-9
                ]
            else:
                candidates = [
                    int(candidate)
                    for candidate in valid_indices
                    if abs(float(times[candidate]) - time_sec) <= radius + 1e-9
                ]
            nearest_index = int(valid_indices[np.argmin(np.abs(times[valid_indices] - time_sec))])
            nearest_gap = abs(float(times[nearest_index]) - time_sec)
            if not candidates and nearest_gap <= self.temporal_max_hold_sec + 1e-9:
                candidates = [nearest_index]
            if not candidates:
                smoothed.append(invalid_roi("temporal_gap_exceeded"))
                continue

            candidate_proposals = [proposals[candidate] for candidate in candidates]
            candidate_times = times[np.asarray(candidates, dtype=np.int64)]
            temporal_scale = max(self.temporal_smoothing_window_sec * 0.5, 1e-3)
            weights = np.asarray(
                [max(proposal.roi_confidence, 0.05) for proposal in candidate_proposals],
                dtype=np.float64,
            )
            weights *= np.exp(-0.5 * ((candidate_times - time_sec) / temporal_scale) ** 2)
            weights /= max(float(weights.sum()), 1e-9)
            boxes = np.asarray([proposal.bbox for proposal in candidate_proposals], dtype=np.float64)
            centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
            measured_cx, measured_cy = np.sum(centers * weights[:, None], axis=0).tolist()
            scales = np.maximum(
                np.maximum((boxes[:, 2] - boxes[:, 0]) / output_w, (boxes[:, 3] - boxes[:, 1]) / output_h),
                1.0,
            )
            scale = int(np.clip(round(float(np.sum(scales * weights))), 1, max_scale))
            crop_w, crop_h = scale * output_w, scale * output_h
            confidence = float(
                sum(proposal.roi_confidence * weight for proposal, weight in zip(candidate_proposals, weights.tolist()))
            )
            if not raw.valid:
                confidence *= math.exp(-nearest_gap / self.temporal_confidence_decay_sec)

            if previous_measurement is not None:
                jump = math.dist(previous_measurement, (measured_cx, measured_cy)) / diagonal
                if jump > self.temporal_max_center_jump_ratio > 0:
                    kf_cx.reset(measured_cx)
                    kf_cy.reset(measured_cy)
            cx = kf_cx.update(measured_cx, confidence)
            cy = kf_cy.update(measured_cy, confidence)
            previous_measurement = (measured_cx, measured_cy)
            left = int(round(min(max(cx - crop_w * 0.5, 0.0), width - crop_w)))
            top = int(round(min(max(cy - crop_h * 0.5, 0.0), height - crop_h)))
            bbox = (left, top, left + crop_w, top + crop_h)

            mode_scores: dict[str, float] = {}
            for proposal, weight in zip(candidate_proposals, weights.tolist()):
                mode_scores[proposal.mode] = mode_scores.get(proposal.mode, 0.0) + weight
            mode = max(mode_scores, key=mode_scores.get)

            def weighted(name: str) -> float:
                return float(
                    sum(float(getattr(proposal, name)) * weight for proposal, weight in zip(candidate_proposals, weights.tolist()))
                )

            smoothed.append(
                ROIProposal(
                    bbox=bbox,
                    valid=True,
                    mode=mode,
                    roi_confidence=min(max(confidence, 0.0), 1.0),
                    goal_score=weighted("goal_score"),
                    ball_score=weighted("ball_score"),
                    center_circle_score=weighted("center_circle_score"),
                    person_support=weighted("person_support"),
                    area_ratio=_box_area(bbox) / max(float(width * height), 1.0),
                    fallback_reason="ok" if raw.valid else "temporal_hold",
                )
            )
        return smoothed

    def get_clip_rois(
        self,
        video_id: str,
        start_sec: float,
        end_sec: float,
        frame_times: Sequence[float],
        width: int,
        height: int,
        target_size: tuple[int, int],
        fixed_fallback: ROIProposal | None = None,
    ) -> tuple[list[ROIProposal], ROIProposal]:
        if self.temporal_mode == "clip":
            proposal = self.get_window_roi(video_id, start_sec, end_sec, width, height, target_size)
            return [proposal for _ in frame_times], proposal
        if fixed_fallback is None:
            fixed_proposal = self.get_window_roi(
                video_id, start_sec, end_sec, width, height, target_size
            )
        else:
            fixed_proposal = fixed_fallback
        raw: list[ROIProposal] = []
        for frame_time in frame_times:
            local_start, local_end = self._dynamic_proposal_window(frame_time, start_sec, end_sec)
            raw.append(
                self.get_window_roi(video_id, local_start, local_end, width, height, target_size)
            )
        proposals = self.smooth_temporal_proposals(raw, frame_times, width, height, target_size)
        fallback_count = 0
        if self.dynamic_fallback_to_clip:
            proposals, fallback_count = self.apply_fixed_fallback(
                proposals,
                fixed_proposal,
                reason="dynamic_fixed_fallback",
            )
        aggregate = self.aggregate_temporal_proposals(proposals)
        if fallback_count > 0 and aggregate.valid:
            reason = (
                "dynamic_fixed_fallback"
                if fallback_count == len(proposals)
                else "partial_dynamic_fixed_fallback"
            )
            aggregate = replace(aggregate, fallback_reason=reason)
        return proposals, aggregate

    @staticmethod
    def apply_fixed_fallback(
        proposals: Sequence[ROIProposal],
        fixed_proposal: ROIProposal | None,
        *,
        reason: str,
    ) -> tuple[list[ROIProposal], int]:
        proposals = list(proposals)
        if (
            fixed_proposal is None
            or not fixed_proposal.valid
            or fixed_proposal.bbox is None
        ):
            return proposals, 0
        output: list[ROIProposal] = []
        fallback_count = 0
        for proposal in proposals:
            if proposal.valid and proposal.bbox is not None:
                output.append(proposal)
                continue
            output.append(replace(fixed_proposal, fallback_reason=reason))
            fallback_count += 1
        return output, fallback_count

    def augment_sequence(
        self,
        proposals: Sequence[ROIProposal],
        width: int,
        height: int,
        cfg: Any,
        target_size: tuple[int, int],
        fixed_fallback: ROIProposal | None = None,
    ) -> tuple[list[ROIProposal], ROIProposal]:
        proposals = list(proposals)
        if not proposals:
            aggregate = invalid_roi("empty_temporal_roi")
            return proposals, aggregate
        if random.random() < float(cfg.get("invalid_prob", 0.0)):
            invalid = [invalid_roi("train_forced_invalid") for _ in proposals]
            invalid, fallback_count = self.apply_fixed_fallback(
                invalid,
                fixed_fallback,
                reason="train_dynamic_fixed_fallback",
            )
            aggregate = self.aggregate_temporal_proposals(invalid)
            if fallback_count > 0 and aggregate.valid:
                aggregate = replace(
                    aggregate, fallback_reason="train_dynamic_fixed_fallback"
                )
            return invalid, aggregate

        drop_ball = random.random() < float(cfg.get("ball_drop_prob", 0.0))
        drop_goal = random.random() < float(cfg.get("goal_drop_prob", 0.0))
        augmented: list[ROIProposal] = []
        for proposal in proposals:
            if not proposal.valid or proposal.bbox is None:
                augmented.append(proposal)
                continue
            if proposal.mode == "goal_ball":
                if drop_ball and drop_goal:
                    augmented.append(invalid_roi("train_anchor_dropout"))
                    continue
                if drop_ball:
                    proposal = replace(
                        proposal,
                        mode="goal_players",
                        ball_score=0.0,
                        roi_confidence=min(0.75 * proposal.goal_score + 0.25 * proposal.person_support, 0.80),
                        fallback_reason="train_ball_dropout",
                    )
                elif drop_goal:
                    proposal = replace(
                        proposal,
                        mode="ball_players",
                        goal_score=0.0,
                        roi_confidence=min(0.75 * proposal.ball_score + 0.25 * proposal.person_support, 0.80),
                        fallback_reason="train_goal_dropout",
                    )
            elif (proposal.mode == "goal_players" and drop_goal) or (
                proposal.mode == "ball_players" and drop_ball
            ):
                augmented.append(invalid_roi("train_anchor_dropout"))
                continue
            augmented.append(proposal)

        augmented, fallback_count = self.apply_fixed_fallback(
            augmented,
            fixed_fallback,
            reason="train_dynamic_fixed_fallback",
        )
        valid = [proposal for proposal in augmented if proposal.valid and proposal.bbox is not None]
        if not valid:
            return augmented, self.aggregate_temporal_proposals(augmented)
        isolated = random.random() < float(cfg.get("isolated_false_positive_prob", 0.0))
        center_jitter = max(float(cfg.get("center_jitter", 0.0)), 0.0)
        scale_min = max(float(cfg.get("scale_min", 1.0)), 1e-3)
        scale_max = max(float(cfg.get("scale_max", 1.0)), 1e-3)
        if scale_max < scale_min:
            scale_min, scale_max = scale_max, scale_min
        sequence_scale = random.uniform(scale_min, scale_max)
        output_h, output_w = map(int, target_size)
        max_fit_scale = min(width // output_w, height // output_h)
        dx = random.uniform(-center_jitter, center_jitter) * width
        dy = random.uniform(-center_jitter, center_jitter) * height
        if isolated:
            centers = np.asarray([_box_center(proposal.bbox) for proposal in valid], dtype=np.float64)
            median_cx, median_cy = np.median(centers, axis=0).tolist()
            first_w = float(valid[0].bbox[2] - valid[0].bbox[0])
            first_h = float(valid[0].bbox[3] - valid[0].bbox[1])
            target_cx = random.uniform(first_w * 0.5, max(width - first_w * 0.5, first_w * 0.5))
            target_cy = random.uniform(first_h * 0.5, max(height - first_h * 0.5, first_h * 0.5))
            dx, dy = target_cx - median_cx, target_cy - median_cy

        moved: list[ROIProposal] = []
        for proposal in augmented:
            if not proposal.valid or proposal.bbox is None:
                moved.append(proposal)
                continue
            x1, y1, x2, y2 = proposal.bbox
            current_scale = max(
                int(round((x2 - x1) / max(output_w, 1))),
                int(round((y2 - y1) / max(output_h, 1))),
                1,
            )
            quantized_scale = int(
                np.clip(round(current_scale * sequence_scale), 1, max(max_fit_scale, 1))
            )
            box_w, box_h = quantized_scale * output_w, quantized_scale * output_h
            cx = (x1 + x2) * 0.5 + dx
            cy = (y1 + y2) * 0.5 + dy
            left = int(round(min(max(cx - box_w * 0.5, 0.0), width - box_w)))
            top = int(round(min(max(cy - box_h * 0.5, 0.0), height - box_h)))
            bbox = (left, top, left + box_w, top + box_h)
            if isolated:
                moved.append(
                    replace(
                        proposal,
                        bbox=bbox,
                        area_ratio=_box_area(bbox) / max(float(width * height), 1.0),
                        roi_confidence=proposal.roi_confidence * 0.5,
                        goal_score=0.0,
                        ball_score=0.0,
                        center_circle_score=0.0,
                        person_support=0.0,
                        fallback_reason="train_isolated_false_positive",
                    )
                )
            else:
                moved.append(
                    replace(
                        proposal,
                        bbox=bbox,
                        area_ratio=_box_area(bbox) / max(float(width * height), 1.0),
                        fallback_reason=(
                            proposal.fallback_reason
                            if proposal.fallback_reason == "train_dynamic_fixed_fallback"
                            else "train_augmented"
                        ),
                    )
                )
        aggregate = self.aggregate_temporal_proposals(moved)
        if fallback_count > 0 and aggregate.valid:
            aggregate = replace(
                aggregate, fallback_reason="train_dynamic_fixed_fallback"
            )
        return moved, aggregate

    def augment(
        self,
        proposal: ROIProposal,
        width: int,
        height: int,
        cfg: Any,
        target_size: tuple[int, int],
    ) -> ROIProposal:
        if not proposal.valid or proposal.bbox is None:
            return proposal
        if random.random() < float(cfg.get("invalid_prob", 0.0)):
            return replace(proposal, bbox=None, valid=False, mode="invalid", area_ratio=0.0, fallback_reason="train_forced_invalid")

        drop_ball = random.random() < float(cfg.get("ball_drop_prob", 0.0))
        drop_goal = random.random() < float(cfg.get("goal_drop_prob", 0.0))
        if proposal.mode == "goal_ball":
            if drop_ball and drop_goal:
                return replace(proposal, bbox=None, valid=False, mode="invalid", area_ratio=0.0, fallback_reason="train_anchor_dropout")
            if drop_ball:
                confidence = min(0.75 * proposal.goal_score + 0.25 * proposal.person_support, 0.80)
                proposal = replace(
                    proposal,
                    mode="goal_players",
                    ball_score=0.0,
                    roi_confidence=confidence,
                    fallback_reason="train_ball_dropout",
                )
            elif drop_goal:
                confidence = min(0.75 * proposal.ball_score + 0.25 * proposal.person_support, 0.80)
                proposal = replace(
                    proposal,
                    mode="ball_players",
                    goal_score=0.0,
                    roi_confidence=confidence,
                    fallback_reason="train_goal_dropout",
                )
        elif (proposal.mode == "goal_players" and drop_goal) or (proposal.mode == "ball_players" and drop_ball):
            return replace(proposal, bbox=None, valid=False, mode="invalid", area_ratio=0.0, fallback_reason="train_anchor_dropout")

        if random.random() < float(cfg.get("isolated_false_positive_prob", 0.0)):
            x1, y1, x2, y2 = map(float, proposal.bbox)
            box_w, box_h = x2 - x1, y2 - y1
            left = random.uniform(0.0, max(float(width) - box_w, 0.0))
            top = random.uniform(0.0, max(float(height) - box_h, 0.0))
            bbox = (int(round(left)), int(round(top)), int(round(left + box_w)), int(round(top + box_h)))
            return replace(
                proposal,
                bbox=bbox,
                roi_confidence=proposal.roi_confidence * 0.5,
                goal_score=0.0,
                ball_score=0.0,
                center_circle_score=0.0,
                person_support=0.0,
                fallback_reason="train_isolated_false_positive",
            )

        center_jitter = max(float(cfg.get("center_jitter", 0.0)), 0.0)
        scale_min = float(cfg.get("scale_min", 1.0))
        scale_max = float(cfg.get("scale_max", 1.0))
        x1, y1, x2, y2 = map(float, proposal.bbox)
        box_w, box_h = int(round(x2 - x1)), int(round(y2 - y1))
        output_h, output_w = map(int, target_size)
        if box_w % output_w != 0 or box_h % output_h != 0 or box_w // output_w != box_h // output_h:
            raise RuntimeError(f"ROI bbox is not an integer multiple of target_size={target_size}: {proposal.bbox}")
        current_scale = box_w // output_w
        min_scale = max(1, int(math.ceil(current_scale * scale_min - 1e-9)))
        max_scale = max(1, int(math.floor(current_scale * scale_max + 1e-9)))
        max_fit_scale = min(width // output_w, height // output_h)
        allowed = [value for value in range(min_scale, max_scale + 1) if value <= max_fit_scale]
        quantized_scale = random.choice(allowed) if allowed else current_scale
        target_w, target_h = quantized_scale * output_w, quantized_scale * output_h
        cx = (x1 + x2) * 0.5 + random.uniform(-center_jitter, center_jitter) * box_w
        cy = (y1 + y2) * 0.5 + random.uniform(-center_jitter, center_jitter) * box_h
        left = min(max(cx - target_w * 0.5, 0.0), width - target_w)
        top = min(max(cy - target_h * 0.5, 0.0), height - target_h)
        left, top = int(round(left)), int(round(top))
        bbox = (left, top, left + target_w, top + target_h)
        area_ratio = _box_area(bbox) / max(float(width * height), 1.0)
        reason = proposal.fallback_reason if proposal.fallback_reason.startswith("train_") else "train_augmented"
        return replace(proposal, bbox=bbox, area_ratio=area_ratio, fallback_reason=reason)
