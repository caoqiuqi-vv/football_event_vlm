from __future__ import annotations

import math
import random
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

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


class RobustClipCropper:
    """Build one detection-aware ROI for an entire temporal clip.

    The class consumes compact ``.pt`` indices produced by
    ``scripts/build_football_roi_indices.py``. Keeping the compact index separate
    avoids loading 400-900 MB per-frame JSON files in every DataLoader worker.
    """

    def __init__(self, cfg: Any):
        self.index_root = Path(str(cfg.get("index_root", ""))).expanduser()
        if not self.index_root:
            raise ValueError("spatial_crop.index_root is required for robust_detector_aware")
        self.padding = float(cfg.get("padding", 0.15))
        self.min_crop_area_ratio = float(cfg.get("min_crop_area_ratio", 0.15))
        self.max_crop_area_ratio = float(cfg.get("max_crop_area_ratio", 0.60))
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
        raw_boxes = payload["ball_boxes"].numpy()
        point_touched = payload["ball_point_touched"].numpy().astype(bool)
        track_touched = payload["ball_track_touched"].numpy().astype(bool)
        det_width = float(payload["image_size"]["width"])
        det_height = float(payload["image_size"]["height"])
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
            boxes = self._scaled_boxes(raw_boxes[lo:hi][selected], det_width, det_height, width, height)
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
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        left = int(round(min(max(cx - target_w * 0.5, 0.0), width - target_w)))
        top = int(round(min(max(cy - target_h * 0.5, 0.0), height - target_h)))
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
    ) -> tuple[tuple[int, int, int, int] | None, float, str, int]:
        """Keep semantic anchors and remove the lowest-priority people only as needed."""
        last_area, last_reason = 0.0, "no_boxes"
        for keep_count in range(len(ranked_people), -1, -1):
            roi, area_ratio, reason = self._fit_roi(
                list(anchor_boxes) + list(ranked_people[:keep_count]),
                width,
                height,
                target_size,
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
        det_width = float(payload["image_size"]["width"])
        det_height = float(payload["image_size"]["height"])

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
        boxes = self._scaled_boxes(
            payload["boxes"][object_lo:object_hi].float().numpy(), det_width, det_height, width, height
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
            anchor_boxes, nearby_people, width, height, target_size
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
                fallback_roi, fallback_area, fallback_reason, fallback_kept = self._fit_ranked_people_roi(
                    fallback_anchors, fallback_people, width, height, target_size
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
