from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch

from football_roi_scoring import rank_object_candidates


def _area(box: Sequence[float]) -> float:
    return max(float(box[2]) - float(box[0]), 0.0) * max(float(box[3]) - float(box[1]), 0.0)


def _union(boxes: Sequence[Sequence[float]]) -> list[float]:
    return [
        min(float(box[0]) for box in boxes),
        min(float(box[1]) for box in boxes),
        max(float(box[2]) for box in boxes),
        max(float(box[3]) for box in boxes),
    ]


def _expand(box: Sequence[float], ratio: float, width: int, height: int) -> list[float]:
    pad_x = max(float(box[2]) - float(box[0]), 1.0) * ratio
    pad_y = max(float(box[3]) - float(box[1]), 1.0) * ratio
    return [
        max(float(box[0]) - pad_x, 0.0),
        max(float(box[1]) - pad_y, 0.0),
        min(float(box[2]) + pad_x, float(width)),
        min(float(box[3]) + pad_y, float(height)),
    ]


def _intersects(first: Sequence[float], second: Sequence[float]) -> bool:
    return min(float(first[2]), float(second[2])) > max(float(first[0]), float(second[0])) and min(
        float(first[3]), float(second[3])
    ) > max(float(first[1]), float(second[1]))


class LegacyIndexedWindowCropper:
    """Reproduce the pre-robust max-goal/all-ball hard crop from compact v2 indices."""

    def __init__(
        self,
        index_root: str,
        *,
        target_aspect: float,
        roi_samples: int = 8,
        ball_conf: float = 0.5,
        goal_conf: float = 0.5,
        person_conf: float = 0.4,
        padding: float = 0.12,
        min_crop_area_ratio: float = 0.15,
        max_crop_area_ratio: float = 0.85,
        max_cached_videos: int = 2,
    ):
        self.index_root = Path(index_root)
        self.target_aspect = float(target_aspect)
        self.roi_samples = max(int(roi_samples), 1)
        self.ball_conf = float(ball_conf)
        self.goal_conf = float(goal_conf)
        self.person_conf = float(person_conf)
        self.padding = float(padding)
        self.min_crop_area_ratio = float(min_crop_area_ratio)
        self.max_crop_area_ratio = float(max_crop_area_ratio)
        self.max_cached_videos = max(int(max_cached_videos), 1)
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._geometry_cache: dict[str, tuple[int, int]] = {}

    def _load(self, video_id: str) -> dict[str, Any]:
        payload = self._cache.pop(video_id, None)
        if payload is None:
            path = self.index_root / f"{video_id}.pt"
            payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            if int(payload.get("version", 0)) != 2:
                raise ValueError(f"Legacy indexed baseline requires v2 index: {path}")
        self._cache[video_id] = payload
        while len(self._cache) > self.max_cached_videos:
            self._cache.popitem(last=False)
        return payload

    def _geometry(self, video_path: str) -> tuple[int, int]:
        if video_path in self._geometry_cache:
            return self._geometry_cache[video_path]
        capture = cv2.VideoCapture(video_path)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        capture.release()
        if width <= 0 or height <= 0:
            raise RuntimeError(f"Invalid video geometry: {video_path}")
        self._geometry_cache[video_path] = (width, height)
        return width, height

    def _fit(self, boxes: list[list[float]], width: int, height: int) -> tuple[tuple[int, int, int, int], float, bool]:
        roi = _expand(_union(boxes), self.padding, width, height)
        box_w = max(roi[2] - roi[0], 1.0)
        box_h = max(roi[3] - roi[1], 1.0)
        target_w = max(box_w, box_h * self.target_aspect)
        target_h = target_w / self.target_aspect
        min_area = self.min_crop_area_ratio * width * height
        max_area = self.max_crop_area_ratio * width * height
        if target_w * target_h < min_area:
            target_h = math.sqrt(min_area / self.target_aspect)
            target_w = target_h * self.target_aspect
        capped = target_w * target_h > max_area
        if capped:
            target_h = math.sqrt(max_area / self.target_aspect)
            target_w = target_h * self.target_aspect
        if target_w > width:
            target_w = float(width)
            target_h = target_w / self.target_aspect
            capped = True
        if target_h > height:
            target_h = float(height)
            target_w = target_h * self.target_aspect
            capped = True
        cx, cy = (roi[0] + roi[2]) * 0.5, (roi[1] + roi[3]) * 0.5
        left = min(max(cx - target_w * 0.5, 0.0), width - target_w)
        top = min(max(cy - target_h * 0.5, 0.0), height - target_h)
        result = (
            int(round(left)),
            int(round(top)),
            int(round(left + target_w)),
            int(round(top + target_h)),
        )
        return result, _area(result) / max(float(width * height), 1.0), capped

    def get_window_roi(
        self,
        video_path: str,
        video_id: str,
        start_sec: float,
        end_sec: float,
    ) -> tuple[tuple[int, int, int, int] | None, dict[str, Any]]:
        width, height = self._geometry(video_path)
        payload = self._load(video_id)
        fps = float(payload["fps"])
        start_frame = max(int(round(start_sec * fps)), 0)
        end_frame = max(int(round(end_sec * fps)) - 1, start_frame)
        sampled_frame_ids = payload["frame_ids"].numpy()
        offsets = payload["frame_offsets"].numpy()
        frame_lo = int(np.searchsorted(sampled_frame_ids, start_frame, side="left"))
        frame_hi = int(np.searchsorted(sampled_frame_ids, end_frame, side="right"))
        stats: dict[str, Any] = {
            "crop_mode": "legacy_indexed",
            "crop_roi": None,
            "crop_area_ratio": 0.0,
            "crop_target_aspect": self.target_aspect,
            "crop_reason": "no_detection_frames",
            "crop_goal_count": 0,
            "crop_ball_count": 0,
            "crop_person_count": 0,
            "roi_valid": 0.0,
            "roi_confidence": 0.0,
            "roi_proposal_mode": "legacy_invalid",
        }
        if frame_lo >= frame_hi:
            return None, stats
        object_lo, object_hi = int(offsets[frame_lo]), int(offsets[frame_hi])
        classes = payload["classes"][object_lo:object_hi].numpy()
        confidences = payload["confidences"][object_lo:object_hi].float().numpy()
        track_ids = payload["track_ids"][object_lo:object_hi].numpy()
        counts = np.diff(offsets[frame_lo : frame_hi + 1])
        object_frame_ids = np.repeat(sampled_frame_ids[frame_lo:frame_hi], counts)
        boxes = payload["boxes"][object_lo:object_hi].float().numpy().astype(np.float64)
        det_width = float(payload["image_size"]["width"])
        det_height = float(payload["image_size"]["height"])
        boxes[:, (0, 2)] *= float(width) / max(det_width, 1.0)
        boxes[:, (1, 3)] *= float(height) / max(det_height, 1.0)

        anchor_frames = np.unique(object_frame_ids[np.isin(classes, [0, 2])])
        if len(anchor_frames) > self.roi_samples:
            indices = np.linspace(0, len(anchor_frames) - 1, self.roi_samples).round().astype(np.int64)
            anchor_frames = anchor_frames[indices]
        selected = np.isin(object_frame_ids, anchor_frames)
        classes = classes[selected]
        confidences = confidences[selected]
        boxes = boxes[selected]
        selected_track_ids = track_ids[selected]
        selected_frame_ids = object_frame_ids[selected]

        goal_mask = (classes == 2) & (confidences >= self.goal_conf)
        goals_arr = boxes[goal_mask]
        goal_confs = confidences[goal_mask]
        goal_tids = selected_track_ids[goal_mask]
        goal_fids = selected_frame_ids[goal_mask]

        ball_mask = (classes == 1) & (confidences >= self.ball_conf)
        balls = boxes[ball_mask].tolist()

        person_mask = (classes == 0) & (confidences >= self.person_conf)
        persons_arr = boxes[person_mask]
        person_confs = confidences[person_mask]
        person_tids = selected_track_ids[person_mask]
        person_fids = selected_frame_ids[person_mask]

        stats.update(
            {
                "crop_goal_count": len(goals_arr),
                "crop_ball_count": len(balls),
                "crop_person_count": len(persons_arr),
            }
        )

        # Track-aware goal selection — rank by persistence/stability
        goal: list[float] | None = None
        if len(goals_arr) > 0:
            goal_candidates = rank_object_candidates(
                goals_arr.astype(np.float32),
                goal_confs.astype(np.float32),
                goal_tids.astype(np.int32),
                goal_fids.astype(np.int32),
                width=width,
                height=height,
                min_frames=3,
            )
            goal = list(goal_candidates[0].box) if goal_candidates else None

        # Filter persons by box area and trajectory persistence
        persons: list[list[float]] = []
        if len(persons_arr) > 0:
            frame_area = float(width * height)
            areas = (persons_arr[:, 2] - persons_arr[:, 0]) * (persons_arr[:, 3] - persons_arr[:, 1])
            area_valid = (areas >= 0.0005 * frame_area) & (areas <= 0.15 * frame_area)
            if area_valid.any():
                per_arr = persons_arr[area_valid].astype(np.float32)
                per_confs_arr = person_confs[area_valid].astype(np.float32)
                per_tids_arr = person_tids[area_valid].astype(np.int32)
                per_fids_arr = person_fids[area_valid].astype(np.int32)
                person_candidates = rank_object_candidates(
                    per_arr,
                    per_confs_arr,
                    per_tids_arr,
                    per_fids_arr,
                    width=width,
                    height=height,
                    min_frames=2,
                )
                persons = [list(candidate.box) for candidate in person_candidates[:10]]

        def nearby(reference: Sequence[float]) -> list[list[float]]:
            result = []
            for person in persons:
                cx = (person[0] + person[2]) * 0.5
                cy = (person[1] + person[3]) * 0.5
                if _intersects(person, reference) or (reference[0] <= cx <= reference[2] and reference[1] <= cy <= reference[3]):
                    result.append(person)
            return result

        mode = "legacy_invalid"
        reason = "no_goal_ball_or_person"
        chosen: list[list[float]] = []
        if goal is not None:
            base = [goal] + balls
            reference = _expand(_union(base), 0.35, width, height)
            chosen = base + nearby(reference)
            mode, reason = "legacy_goal", "max_goal_all_balls"
        elif balls:
            ball_union = _union(balls)
            chosen = [ball_union] + nearby(_expand(ball_union, 2.5, width, height))
            mode, reason = "legacy_ball", "all_balls_fallback"
        elif persons:
            chosen = [_union(persons)]
            mode, reason = "legacy_people", "people_only_fallback"
        if not chosen:
            return None, stats
        roi, area_ratio, capped = self._fit(chosen, width, height)
        stats.update(
            {
                "crop_roi": list(roi),
                "crop_area_ratio": area_ratio,
                "crop_reason": f"{reason}_area_capped" if capped else reason,
                "roi_valid": 1.0,
                "roi_confidence": 1.0,
                "roi_proposal_mode": mode,
            }
        )
        return roi, stats
