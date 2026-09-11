#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset

from football_adaptive_top_cropper import AdaptiveTopWindowCropper
from football_detection_aware import RobustClipCropper
from football_legacy_indexed_cropper import LegacyIndexedWindowCropper
from football_roi_scoring import rank_object_candidates
from football_eval_semantics import SCORE_SEMANTICS_VERSION
import train_football_events as train_mod
from train_football_events import (
    ConfigDict,
    DetectorAwareCropper,
    VideoCaptureCache,
    autocast_context,
    clamp,
    configure_label_schema,
    load_annotation_events,
    make_model,
    parse_image_size,
    read_video_segment,
    read_video_segment_views,
    sample_frame_indices,
    to_config,
)

TARGET_LABELS = ("shot", "save", "set_piece")
DEFAULT_REPAIRED_GT_DIR = Path("~/code/football_events_human_repair").expanduser()


@dataclass(frozen=True)
class WindowRecord:
    index: int
    start_sec: float
    end_sec: float


class ConstantCropProvider:
    def __init__(self, roi: tuple[int, int, int, int] | None):
        self.roi = roi

    def get_roi(self, video_id: str, frame_id: int, width: int, height: int) -> tuple[int, int, int, int] | None:
        return self.roi


class WindowFixedDetectorCropper:
    """Build one stable detector-aware ROI per temporal window."""

    def __init__(self, cropper: DetectorAwareCropper, roi_samples: int, target_aspect: float):
        self.cropper = cropper
        self.roi_samples = max(1, int(roi_samples))
        self.target_aspect = max(float(target_aspect), 1e-6)

    @classmethod
    def from_args(cls, args: argparse.Namespace, image_size: tuple[int, int]) -> "WindowFixedDetectorCropper | None":
        if args.spatial_crop_mode != "detector_aware":
            return None
        if not args.detector_manifest_root:
            raise ValueError("--detector-manifest-root is required when --spatial-crop-mode detector_aware")
        cfg = ConfigDict(
            {
                "manifest_root": args.detector_manifest_root,
                "ball_conf": args.detector_ball_conf,
                "goal_conf": args.detector_goal_conf,
                "person_conf": args.detector_person_conf,
                "padding": args.detector_padding,
                "min_crop_area_ratio": args.detector_min_crop_area_ratio,
                "max_crop_area_ratio": args.detector_max_crop_area_ratio,
                "max_frame_gap": args.detector_max_frame_gap,
                "use_detection_goals": args.detector_use_detection_goals,
            }
        )
        out_h, out_w = image_size
        return cls(DetectorAwareCropper(cfg), args.detector_window_roi_samples, float(out_w) / max(float(out_h), 1.0))

    def _window_frame_indices(self, video_path: str, video_id: str, start_sec: float, end_sec: float) -> tuple[list[int], int, int]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        if total_frames <= 0 or fps <= 0 or width <= 0 or height <= 0:
            return [], width, height
        start_frame = int(clamp(round(start_sec * fps), 0, total_frames - 1))
        end_frame = int(clamp(round(end_sec * fps) - 1, start_frame, total_frames - 1))
        detector_frames = sorted(frame_id for frame_id in self.cropper._load_video(video_id) if start_frame <= frame_id <= end_frame)
        if detector_frames:
            num_samples = min(self.roi_samples, len(detector_frames))
            indices = sample_frame_indices(len(detector_frames), num_samples, False)
            return [detector_frames[index] for index in indices], width, height
        segment_count = max(end_frame - start_frame + 1, 1)
        num_samples = min(self.roi_samples, segment_count)
        return [start_frame + idx for idx in sample_frame_indices(segment_count, num_samples, False)], width, height

    def _fit_roi_to_target_aspect(
        self,
        roi: list[float],
        width: int,
        height: int,
        min_area: float,
        max_area: float,
        priority_box: list[float] | None = None,
    ) -> tuple[list[float], bool]:
        x1, y1, x2, y2 = map(float, roi)
        box_w = max(x2 - x1, 1.0)
        box_h = max(y2 - y1, 1.0)
        target_w = max(box_w, box_h * self.target_aspect)
        target_h = target_w / self.target_aspect
        if target_h < box_h:
            target_h = box_h
            target_w = target_h * self.target_aspect
        if min_area > 0 and target_w * target_h < min_area:
            area_h = (min_area / self.target_aspect) ** 0.5
            area_w = area_h * self.target_aspect
            target_w = max(target_w, area_w)
            target_h = max(target_h, area_h)

        capped = False
        if max_area > 0 and target_w * target_h > max_area:
            capped = True
            target_h = (max_area / self.target_aspect) ** 0.5
            target_w = target_h * self.target_aspect

        # Keep the model input aspect ratio. If the requested ROI cannot fit in the frame,
        # use the largest in-frame crop with that aspect ratio rather than stretching content.
        if target_w > width:
            capped = True
            target_w = float(width)
            target_h = target_w / self.target_aspect
        if target_h > height:
            capped = True
            target_h = float(height)
            target_w = target_h * self.target_aspect
        target_w = min(float(width), max(1.0, target_w))
        target_h = min(float(height), max(1.0, target_h))

        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        left = clamp(cx - target_w * 0.5, 0.0, max(float(width) - target_w, 0.0))
        top = clamp(cy - target_h * 0.5, 0.0, max(float(height) - target_h, 0.0))

        if priority_box is not None:
            px1, py1, px2, py2 = map(float, priority_box)
            if px2 > left + target_w:
                left = min(px2 - target_w, float(width) - target_w)
            if px1 < left:
                left = max(px1, 0.0)
            if py2 > top + target_h:
                top = min(py2 - target_h, float(height) - target_h)
            if py1 < top:
                top = max(py1, 0.0)
            left = clamp(left, 0.0, max(float(width) - target_w, 0.0))
            top = clamp(top, 0.0, max(float(height) - target_h, 0.0))
        return [left, top, left + target_w, top + target_h], capped

    def _candidate_roi(self, boxes: list[list[float]], width: int, height: int) -> tuple[tuple[int, int, int, int] | None, float, bool]:
        if not boxes:
            return None, 0.0, False
        roi = self.cropper._expand(self.cropper._union(boxes), self.cropper.padding, width, height)
        min_area = self.cropper.min_crop_area_ratio * width * height
        max_area = self.cropper.max_crop_area_ratio * width * height
        roi, capped = self._fit_roi_to_target_aspect(roi, width, height, min_area, max_area, priority_box=boxes[0])
        area_ratio = self.cropper._area(roi) / max(float(width * height), 1.0)
        x1 = int(clamp(round(roi[0]), 0, width - 1))
        y1 = int(clamp(round(roi[1]), 0, height - 1))
        x2 = int(clamp(round(roi[2]), x1 + 1, width))
        y2 = int(clamp(round(roi[3]), y1 + 1, height))
        return (x1, y1, x2, y2), area_ratio, capped

    def get_window_roi(self, video_path: str, video_id: str, start_sec: float, end_sec: float) -> tuple[tuple[int, int, int, int] | None, dict[str, Any]]:
        frame_indices, width, height = self._window_frame_indices(video_path, video_id, start_sec, end_sec)
        stats: dict[str, Any] = {
            "crop_mode": "detector_aware_window",
            "crop_roi": None,
            "crop_area_ratio": 0.0,
            "crop_target_aspect": self.target_aspect,
            "crop_sampled_frames": len(frame_indices),
            "crop_goal_count": 0,
            "crop_ball_count": 0,
            "crop_person_count": 0,
            "crop_reason": "no_frames",
        }
        if not frame_indices or width <= 0 or height <= 0:
            return None, stats

        goals_arr: list[list[float]] = []
        goal_confs: list[float] = []
        goal_track_ids: list[int] = []
        goal_frame_ids: list[int] = []
        balls: list[list[float]] = []
        persons_arr: list[list[float]] = []
        person_confs: list[float] = []
        person_track_ids: list[int] = []
        person_frame_ids: list[int] = []
        for frame_id in frame_indices:
            for obj in self.cropper._objects_for_frame(video_id, frame_id):
                bbox = obj.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                bbox = self.cropper._scale_bbox(video_id, bbox, width, height)
                cls_id = int(obj.get("cls", obj.get("class", -1)))
                conf = float(obj.get("conf", obj.get("score", 0.0)))
                track_id = int(obj.get("trackID", obj.get("track_id", -1)) or -1)
                if cls_id == 0 and conf >= self.cropper.person_conf:
                    persons_arr.append(bbox)
                    person_confs.append(conf)
                    person_track_ids.append(track_id)
                    person_frame_ids.append(frame_id)
                elif cls_id == 1 and conf >= self.cropper.ball_conf:
                    balls.append(bbox)
                elif cls_id == 2 and conf >= self.cropper.goal_conf:
                    goals_arr.append(bbox)
                    goal_confs.append(conf)
                    goal_track_ids.append(track_id)
                    goal_frame_ids.append(frame_id)

        stats.update({"crop_goal_count": len(goals_arr), "crop_ball_count": len(balls), "crop_person_count": len(persons_arr)})

        # Track-aware goal selection — rank by persistence/stability, not just largest area
        goal: list[float] | None = None
        if goals_arr:
            goal_candidates = rank_object_candidates(
                np.asarray(goals_arr, dtype=np.float32),
                np.asarray(goal_confs, dtype=np.float32),
                np.asarray(goal_track_ids, dtype=np.int32),
                np.asarray(goal_frame_ids, dtype=np.int32),
                width=width,
                height=height,
                min_frames=3,
            )
            goal = list(goal_candidates[0].box) if goal_candidates else None

        # Track-aware person selection — filter by area and trajectory persistence
        persons: list[list[float]] = []
        if persons_arr:
            per_arr = np.asarray(persons_arr, dtype=np.float32)
            frame_area = float(width * height)
            areas = (per_arr[:, 2] - per_arr[:, 0]) * (per_arr[:, 3] - per_arr[:, 1])
            area_valid = (areas >= 0.0005 * frame_area) & (areas <= 0.15 * frame_area)
            if area_valid.any():
                per_arr = per_arr[area_valid]
                per_confs_arr = np.asarray(person_confs, dtype=np.float32)[area_valid]
                per_track_arr = np.asarray(person_track_ids, dtype=np.int32)[area_valid]
                per_frame_arr = np.asarray(person_frame_ids, dtype=np.int32)[area_valid]
                person_candidates = rank_object_candidates(
                    per_arr,
                    per_confs_arr,
                    per_track_arr,
                    per_frame_arr,
                    width=width,
                    height=height,
                    min_frames=2,
                )
                persons = [list(candidate.box) for candidate in person_candidates[:10]]

        def nearby_people(reference: list[float]) -> list[list[float]]:
            selected: list[list[float]] = []
            for person in persons:
                cx = (float(person[0]) + float(person[2])) * 0.5
                cy = (float(person[1]) + float(person[3])) * 0.5
                if self.cropper._intersects(person, reference) or (reference[0] <= cx <= reference[2] and reference[1] <= cy <= reference[3]):
                    selected.append(person)
            return selected

        if goal is not None:
            base_boxes = [goal] + balls
            reference = self.cropper._expand(self.cropper._union(base_boxes), 0.35, width, height)
            nearby_persons = nearby_people(reference)
            roi, area_ratio, capped = self._candidate_roi(base_boxes + nearby_persons, width, height)
            reason = "area_capped" if capped else "ok"
            if roi is None and balls:
                roi, area_ratio, capped = self._candidate_roi([goal] + nearby_persons, width, height)
                reason = "area_capped_without_balls" if capped else "ok_without_balls"
            elif roi is None:
                reason = "too_large"
        elif balls:
            ball_union = self.cropper._union(balls)
            reference = self.cropper._expand(ball_union, 2.5, width, height)
            nearby_persons = nearby_people(reference)
            roi, area_ratio, capped = self._candidate_roi([ball_union] + nearby_persons, width, height)
            reason = "ball_people_fallback_area_capped" if capped else "ball_people_fallback"
            if roi is None:
                roi, area_ratio, capped = self._candidate_roi([ball_union], width, height)
                reason = "ball_only_fallback_area_capped" if capped else "ball_only_fallback"
        elif persons:
            person_union = self.cropper._union(persons)
            roi, area_ratio, capped = self._candidate_roi([person_union], width, height)
            reason = "people_fallback_area_capped" if capped else "people_fallback"
        else:
            roi, area_ratio, reason = None, 0.0, "no_goal_ball_or_person"

        if roi is not None:
            stats.update({"crop_roi": list(roi), "crop_area_ratio": area_ratio, "crop_reason": reason})
        else:
            stats.update({"crop_area_ratio": area_ratio, "crop_reason": reason})
        return roi, stats


class RobustWindowCropper:
    def __init__(self, cropper: RobustClipCropper, target_size: tuple[int, int]):
        self.cropper = cropper
        self.target_size = target_size
        self.target_aspect = float(target_size[1]) / max(float(target_size[0]), 1.0)
        self._geometry_cache: dict[str, tuple[int, int, float, int]] = {}

    @classmethod
    def from_args(
        cls,
        args: argparse.Namespace,
        image_size: tuple[int, int],
        checkpoint_cfg: Any,
    ) -> "RobustWindowCropper | None":
        if args.spatial_crop_mode != "robust_detector_aware":
            return None
        spatial_cfg = dict(checkpoint_cfg.get("spatial_crop", {}) or {})
        if args.detector_index_root:
            spatial_cfg["index_root"] = args.detector_index_root
        temporal_overrides = {
            "roi_temporal_mode": "temporal_mode",
            "roi_dynamic_context_sec": "dynamic_context_sec",
            "roi_temporal_smoothing_window_sec": "temporal_smoothing_window_sec",
            "roi_temporal_max_hold_sec": "temporal_max_hold_sec",
            "roi_temporal_confidence_decay_sec": "temporal_confidence_decay_sec",
        }
        for arg_name, config_name in temporal_overrides.items():
            value = getattr(args, arg_name, None)
            if value not in (None, "", "checkpoint"):
                spatial_cfg[config_name] = value
        if not spatial_cfg.get("index_root"):
            raise ValueError(
                "--detector-index-root is required when the checkpoint config has no spatial_crop.index_root"
            )
        return cls(RobustClipCropper(ConfigDict(spatial_cfg)), image_size)

    def _geometry(self, video_path: str) -> tuple[int, int, float, int]:
        if video_path in self._geometry_cache:
            return self._geometry_cache[video_path]
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            raise FileNotFoundError(f"Could not open video: {video_path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0:
            raise RuntimeError(
                f"Invalid video geometry for {video_path}: {width}x{height} fps={fps} frames={frame_count}"
            )
        self._geometry_cache[video_path] = (width, height, fps, frame_count)
        return width, height, fps, frame_count

    def get_window_roi(
        self,
        video_path: str,
        video_id: str,
        start_sec: float,
        end_sec: float,
    ) -> tuple[tuple[int, int, int, int] | None, dict[str, Any]]:
        width, height, _, _ = self._geometry(video_path)
        proposal = self.cropper.get_window_roi(
            video_id,
            start_sec,
            end_sec,
            width,
            height,
            self.target_size,
        )
        stats = {
            "crop_mode": "robust_detector_aware",
            "crop_roi": list(proposal.bbox) if proposal.bbox is not None else None,
            "crop_area_ratio": proposal.area_ratio,
            "crop_target_aspect": self.target_aspect,
            "crop_reason": proposal.fallback_reason,
            "crop_goal_count": 0,
            "crop_ball_count": 0,
            "crop_person_count": 0,
            "roi_valid": float(proposal.valid),
            "roi_confidence": proposal.roi_confidence,
            "roi_proposal_mode": proposal.mode,
            "roi_goal_score": proposal.goal_score,
            "roi_ball_score": proposal.ball_score,
            "roi_center_circle_score": proposal.center_circle_score,
            "roi_person_support": proposal.person_support,
            "_roi_meta": proposal.meta_vector(),
        }
        return proposal.bbox if proposal.valid else None, stats

    def get_window_rois(
        self,
        video_path: str,
        video_id: str,
        start_sec: float,
        end_sec: float,
        num_frames: int,
        frame_indices: Sequence[int] | None = None,
    ) -> tuple[list[tuple[int, int, int, int] | None], dict[str, Any]]:
        width, height, fps, frame_count = self._geometry(video_path)
        if frame_indices is None:
            frame_indices = train_mod.segment_frame_indices(
                frame_count,
                fps,
                num_frames,
                False,
                start_sec=start_sec,
                end_sec=end_sec,
            )
        frame_times = train_mod.frame_times_from_indices(frame_indices, fps)
        proposals, aggregate = self.cropper.get_clip_rois(
            video_id,
            start_sec,
            end_sec,
            frame_times,
            width,
            height,
            self.target_size,
        )
        valid_fraction = sum(float(proposal.valid) for proposal in proposals) / max(len(proposals), 1)
        stats = {
            "crop_mode": "robust_detector_aware",
            "crop_roi": list(aggregate.bbox) if aggregate.bbox is not None else None,
            "crop_frame_rois": [list(proposal.bbox) if proposal.bbox is not None else None for proposal in proposals],
            "crop_area_ratio": aggregate.area_ratio,
            "crop_target_aspect": self.target_aspect,
            "crop_reason": aggregate.fallback_reason,
            "crop_goal_count": 0,
            "crop_ball_count": 0,
            "crop_person_count": 0,
            "roi_valid": valid_fraction,
            "roi_confidence": aggregate.roi_confidence,
            "roi_proposal_mode": aggregate.mode,
            "roi_goal_score": aggregate.goal_score,
            "roi_ball_score": aggregate.ball_score,
            "roi_center_circle_score": aggregate.center_circle_score,
            "roi_person_support": aggregate.person_support,
            "_roi_meta": aggregate.meta_vector(),
            "_roi_frame_meta": torch.stack([proposal.meta_vector() for proposal in proposals]),
            "_roi_frame_valid": torch.tensor([float(proposal.valid) for proposal in proposals], dtype=torch.float32),
            "_frame_indices": frame_indices,
        }
        return [proposal.bbox if proposal.valid else None for proposal in proposals], stats

    def get_window_multi_rois(
        self,
        video_path: str,
        video_id: str,
        start_sec: float,
        end_sec: float,
        roi_a_frame_indices: Sequence[int],
        roi_b_frame_indices: Sequence[int],
    ) -> tuple[
        list[tuple[int, int, int, int] | None],
        list[tuple[int, int, int, int] | None],
        dict[str, Any],
    ]:
        width, height, fps, _ = self._geometry(video_path)
        times_a = train_mod.frame_times_from_indices(roi_a_frame_indices, fps)
        proposals_a, aggregate_a = self.cropper.get_clip_rois(
            video_id,
            start_sec,
            end_sec,
            times_a,
            width,
            height,
            self.target_size,
        )
        aggregate_b = self.cropper.get_complementary_window_roi(
            video_id,
            start_sec,
            end_sec,
            width,
            height,
            self.target_size,
            aggregate_a,
        )
        proposals_b = [aggregate_b for _ in roi_b_frame_indices]
        valid_fraction_a = sum(float(item.valid) for item in proposals_a) / max(
            len(proposals_a), 1
        )
        valid_fraction_b = sum(float(item.valid) for item in proposals_b) / max(
            len(proposals_b), 1
        )
        stats = {
            "crop_mode": "robust_detector_aware_multi_roi",
            "roi_valid": valid_fraction_a,
            "roi_confidence": aggregate_a.roi_confidence,
            "_roi_meta": aggregate_a.meta_vector(),
            "_roi_frame_meta": torch.stack(
                [item.meta_vector() for item in proposals_a]
            ),
            "_roi_frame_valid": torch.tensor(
                [float(item.valid) for item in proposals_a],
                dtype=torch.float32,
            ),
            "roi_valid_b": valid_fraction_b,
            "roi_confidence_b": aggregate_b.roi_confidence,
            "_roi_meta_b": aggregate_b.meta_vector(),
            "_roi_frame_meta_b": torch.stack(
                [item.meta_vector() for item in proposals_b]
            ),
            "_roi_frame_valid_b": torch.tensor(
                [float(item.valid) for item in proposals_b],
                dtype=torch.float32,
            ),
        }
        return (
            [item.bbox if item.valid else None for item in proposals_a],
            [item.bbox if item.valid else None for item in proposals_b],
            stats,
        )


class SlidingWindowVideoDataset(Dataset):
    def __init__(
        self,
        *,
        video_path: str,
        video_id: str,
        windows: Sequence[WindowRecord],
        num_frames: int,
        image_size: tuple[int, int],
        normalize_on_cpu: bool,
        decode_strategy: str,
        video_reader_cache_size: int,
        window_cropper: WindowFixedDetectorCropper | RobustWindowCropper | None = None,
        frame_crop_provider: train_mod.TopBandCropProvider | None = None,
        view_mode: str = "single",
        global_image_size: tuple[int, int] | None = None,
        dual_sampling: str = "aligned",
        roi_overlap_frames: int = 8,
        num_rois: int = 1,
        highres_pool_frames: int = 0,
        highres_pool_size: tuple[int, int] | None = None,
        object_motion_frames_per_segment: int = 0,
        object_motion_image_size: tuple[int, int] | None = None,
    ):
        self.video_path = video_path
        self.video_id = video_id
        self.windows = list(windows)
        self.num_frames = num_frames
        self.image_size = image_size
        self.normalize_on_cpu = normalize_on_cpu
        self.decode_strategy = decode_strategy
        self.window_cropper = window_cropper
        self.frame_crop_provider = frame_crop_provider
        self.view_mode = view_mode
        self.global_image_size = global_image_size or image_size
        self.dual_sampling = str(dual_sampling).strip().lower()
        if self.dual_sampling not in ("aligned", "staggered", "multi_staggered"):
            raise ValueError(
                "dual_sampling must be aligned, staggered, or multi_staggered"
            )
        self.roi_overlap_frames = min(max(int(roi_overlap_frames), 0), self.num_frames)
        self.num_rois = int(num_rois)
        if self.num_rois not in (1, 2):
            raise ValueError("num_rois must be 1 or 2")
        if self.num_rois == 2 and self.view_mode == "single":
            raise ValueError("two ROI evaluation requires a dual view model")
        self.highres_pool_frames = max(int(highres_pool_frames), 0)
        self.highres_pool_size = highres_pool_size or self.global_image_size
        self.object_motion_frames_per_segment = max(
            int(object_motion_frames_per_segment), 0
        )
        self.object_motion_image_size = object_motion_image_size
        if self.object_motion_frames_per_segment and self.object_motion_image_size is None:
            raise ValueError("object-motion evaluation requires object_motion_image_size")
        if self.highres_pool_frames and self.view_mode != "single":
            raise ValueError("high-resolution glimpse dense evaluation currently requires a single-view anchor")

        self.cap_cache = VideoCaptureCache(video_reader_cache_size) if video_reader_cache_size > 0 else None

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = self.windows[index]
        roi: tuple[int, int, int, int] | list[tuple[int, int, int, int] | None] | None = None
        frame_indices: list[int] | None = None
        local_frame_indices: list[int] | None = None
        local_frame_indices_b: list[int] | None = None
        roi_b: list[tuple[int, int, int, int] | None] | None = None
        crop_meta: dict[str, Any] = {"crop_mode": "none"}
        if self.frame_crop_provider is not None:
            crop_meta = {
                "crop_mode": "top_fixed",
                "top_crop_ratio": self.frame_crop_provider.top_crop_ratio,
                "crop_area_ratio": 1.0 - self.frame_crop_provider.top_crop_ratio,
            }
        if isinstance(self.window_cropper, RobustWindowCropper):
            if self.dual_sampling == "multi_staggered":
                _, _, fps, frame_count = self.window_cropper._geometry(
                    self.video_path
                )
                frame_indices = train_mod.segment_frame_indices(
                    frame_count,
                    fps,
                    self.num_frames,
                    False,
                    start_sec=window.start_sec,
                    end_sec=window.end_sec,
                )
                segment_start_frame = int(
                    train_mod.clamp(
                        round(window.start_sec * fps), 0, frame_count - 1
                    )
                )
                segment_end_frame = int(
                    train_mod.clamp(
                        round(window.end_sec * fps) - 1,
                        segment_start_frame,
                        frame_count - 1,
                    )
                )
                local_frame_indices, local_frame_indices_b = (
                    train_mod.staggered_multi_roi_indices(
                        frame_indices,
                        segment_start_frame,
                        segment_end_frame,
                    )
                )
                roi, roi_b, crop_meta = self.window_cropper.get_window_multi_rois(
                    self.video_path,
                    self.video_id,
                    window.start_sec,
                    window.end_sec,
                    local_frame_indices,
                    local_frame_indices_b,
                )
            elif self.dual_sampling == "staggered" and self.view_mode != "single":
                _, _, fps, frame_count = self.window_cropper._geometry(
                    self.video_path
                )
                frame_indices = train_mod.segment_frame_indices(
                    frame_count,
                    fps,
                    self.num_frames,
                    False,
                    start_sec=window.start_sec,
                    end_sec=window.end_sec,
                )
                local_frame_indices = train_mod.staggered_local_indices(
                    frame_indices, self.roi_overlap_frames
                )
                roi, crop_meta = self.window_cropper.get_window_rois(
                    self.video_path,
                    self.video_id,
                    window.start_sec,
                    window.end_sec,
                    self.num_frames,
                    frame_indices=local_frame_indices,
                )
                crop_meta.pop("_frame_indices")
            else:
                roi, crop_meta = self.window_cropper.get_window_rois(
                    self.video_path,
                    self.video_id,
                    window.start_sec,
                    window.end_sec,
                    self.num_frames,
                )
                frame_indices = crop_meta.pop("_frame_indices")
                local_frame_indices = frame_indices
        elif self.window_cropper is not None:
            roi, crop_meta = self.window_cropper.get_window_roi(
                self.video_path,
                self.video_id,
                window.start_sec,
                window.end_sec,
            )
        roi_meta = crop_meta.pop("_roi_meta", torch.zeros(train_mod.ROI_META_DIM, dtype=torch.float32))
        roi_frame_meta = crop_meta.pop(
            "_roi_frame_meta",
            roi_meta.reshape(1, -1).expand(self.num_frames, -1).clone(),
        )
        roi_valid = torch.tensor(float(crop_meta.get("roi_valid", roi is not None)), dtype=torch.float32)
        roi_frame_valid = crop_meta.pop(
            "_roi_frame_valid",
            torch.full((self.num_frames,), float(roi_valid), dtype=torch.float32),
        )
        roi_meta_b = crop_meta.pop(
            "_roi_meta_b", torch.zeros(train_mod.ROI_META_DIM, dtype=torch.float32)
        )
        roi_valid_b = torch.tensor(
            float(crop_meta.get("roi_valid_b", roi_b is not None)),
            dtype=torch.float32,
        )
        roi_frame_meta_b = crop_meta.pop(
            "_roi_frame_meta_b",
            roi_meta_b.reshape(1, -1).expand(self.num_frames, -1).clone(),
        )
        roi_frame_valid_b = crop_meta.pop(
            "_roi_frame_valid_b",
            torch.full(
                (self.num_frames,), float(roi_valid_b), dtype=torch.float32
            ),
        )
        roi_inputs: torch.Tensor | None = None
        roi_inputs_b: torch.Tensor | None = None
        local_frame_times: torch.Tensor | None = None
        local_frame_times_b: torch.Tensor | None = None
        highres_pool_inputs: torch.Tensor | None = None
        highres_pool_times: torch.Tensor | None = None
        if self.view_mode != "single":
            if self.dual_sampling == "multi_staggered":
                if (
                    frame_indices is None
                    or local_frame_indices is None
                    or local_frame_indices_b is None
                    or not isinstance(roi, list)
                    or roi_b is None
                ):
                    raise RuntimeError(
                        "multi-staggered evaluation requires robust two-ROI inputs"
                    )
                (
                    frames,
                    roi_inputs,
                    roi_inputs_b,
                    frame_times,
                    local_frame_times,
                    local_frame_times_b,
                ) = train_mod.read_video_segment_multi_roi_views(
                    self.video_path,
                    self.num_frames,
                    self.global_image_size,
                    self.image_size,
                    roi,
                    roi_b,
                    False,
                    0.0,
                    start_sec=window.start_sec,
                    end_sec=window.end_sec,
                    frame_indices=frame_indices,
                    roi_a_frame_indices=local_frame_indices,
                    roi_b_frame_indices=local_frame_indices_b,
                    normalize=self.normalize_on_cpu,
                    cap_cache=self.cap_cache,
                    decode_strategy=self.decode_strategy,
                    return_frame_times=True,
                )
            elif self.dual_sampling == "staggered":
                if (
                    frame_indices is None
                    or local_frame_indices is None
                    or not isinstance(roi, list)
                ):
                    raise RuntimeError(
                        "staggered D7 evaluation requires robust per-frame ROI indices"
                    )
                (
                    frames,
                    roi_inputs,
                    frame_times,
                    local_frame_times,
                ) = train_mod.read_video_staggered_views(
                    self.video_path,
                    frame_indices,
                    local_frame_indices,
                    self.global_image_size,
                    self.image_size,
                    roi,
                    False,
                    0.0,
                    video_id=self.video_id,
                    normalize=self.normalize_on_cpu,
                    cap_cache=self.cap_cache,
                    decode_strategy=self.decode_strategy,
                )
            else:
                frames, roi_inputs, frame_times = read_video_segment_views(
                    self.video_path,
                    self.num_frames,
                    self.global_image_size,
                    self.image_size,
                    roi,
                    False,
                    0.0,
                    start_sec=window.start_sec,
                    end_sec=window.end_sec,
                    frame_indices=frame_indices,
                    normalize=self.normalize_on_cpu,
                    cap_cache=self.cap_cache,
                    decode_strategy=self.decode_strategy,
                    return_frame_times=True,
                )
                local_frame_times = frame_times
        else:
            if isinstance(roi, list):
                crop_provider = train_mod.FrameCropProvider(frame_indices or [], roi)
            else:
                crop_provider = ConstantCropProvider(roi) if roi is not None else None
            if crop_provider is None:
                crop_provider = self.frame_crop_provider
            if self.highres_pool_frames:
                if crop_provider is not None or frame_indices is not None:
                    raise ValueError(
                        "high-resolution glimpse dense evaluation requires uncropped full-image windows"
                    )
                (
                    frames,
                    frame_times,
                    highres_pool_inputs,
                    highres_pool_times,
                ) = train_mod.read_video_global_highres_pool(
                    self.video_path,
                    self.num_frames,
                    self.highres_pool_frames,
                    self.image_size,
                    self.highres_pool_size,
                    False,
                    0.0,
                    start_sec=window.start_sec,
                    end_sec=window.end_sec,
                    video_id=self.video_id,
                    normalize_global=self.normalize_on_cpu,
                    cap_cache=self.cap_cache,
                    decode_strategy=self.decode_strategy,
                )
            else:
                frames, frame_times = read_video_segment(
                    self.video_path,
                    self.num_frames,
                    self.image_size,
                    False,
                    0.0,
                    start_sec=window.start_sec,
                    end_sec=window.end_sec,
                    frame_indices=frame_indices,
                    video_id=self.video_id,
                    crop_provider=crop_provider,
                    normalize=self.normalize_on_cpu,
                    cap_cache=self.cap_cache,
                    decode_strategy=self.decode_strategy,
                    return_frame_times=True,
                )
        meta = {
            "index": window.index,
            "start_sec": window.start_sec,
            "end_sec": window.end_sec,
        }
        meta.update(crop_meta)
        result = {
            "inputs": frames,
            "meta": meta,
            "roi_valid": roi_valid,
            "roi_meta": roi_meta,
            "roi_frame_valid": roi_frame_valid,
            "roi_frame_meta": roi_frame_meta,
            "roi_valid_b": roi_valid_b,
            "roi_meta_b": roi_meta_b,
            "roi_frame_valid_b": roi_frame_valid_b,
            "roi_frame_meta_b": roi_frame_meta_b,
            "frame_times": frame_times,
        }
        if self.object_motion_frames_per_segment:
            from football_object_motion.train import full_window_segment_bounds

            decoded = [
                read_video_segment(
                    self.video_path,
                    self.object_motion_frames_per_segment,
                    self.object_motion_image_size,
                    False,
                    0.0,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    video_id=self.video_id,
                    normalize=False,
                    cap_cache=self.cap_cache,
                    decode_strategy=self.decode_strategy,
                    return_frame_times=True,
                    hflip_override=False,
                )
                for start_sec, end_sec in full_window_segment_bounds(
                    window.start_sec, window.end_sec
                )
            ]
            motion_frames = torch.cat([value[0] for value in decoded], dim=0)
            motion_times = torch.cat([value[1] for value in decoded], dim=0)
            order = torch.argsort(motion_times, stable=True)
            result["object_motion_inputs"] = motion_frames[order]
            result["object_motion_times"] = motion_times[order]
        if roi_inputs is not None:
            result["roi_inputs"] = roi_inputs
        if roi_inputs_b is not None:
            result["roi_inputs_b"] = roi_inputs_b
        if local_frame_times is not None:
            result["local_frame_times"] = local_frame_times
        if local_frame_times_b is not None:
            result["local_frame_times_b"] = local_frame_times_b
        if highres_pool_inputs is not None and highres_pool_times is not None:
            result["highres_pool_inputs"] = highres_pool_inputs
            result["highres_pool_times"] = highres_pool_times
        return result


def collate_windows(batch: list[dict[str, Any]]) -> dict[str, Any]:
    result = {
        "inputs": torch.stack([item["inputs"] for item in batch], dim=0),
        "meta": [item["meta"] for item in batch],
        "roi_valid": torch.stack([item["roi_valid"] for item in batch], dim=0),
        "roi_meta": torch.stack([item["roi_meta"] for item in batch], dim=0),
        "roi_frame_valid": torch.stack([item["roi_frame_valid"] for item in batch], dim=0),
        "roi_frame_meta": torch.stack([item["roi_frame_meta"] for item in batch], dim=0),
        "frame_times": torch.stack([item["frame_times"] for item in batch], dim=0),
    }
    if "roi_inputs" in batch[0]:
        result["roi_inputs"] = torch.stack(
            [item["roi_inputs"] for item in batch], dim=0
        )
    if "roi_inputs_b" in batch[0]:
        result["roi_inputs_b"] = torch.stack(
            [item["roi_inputs_b"] for item in batch], dim=0
        )
    for key in (
        "roi_valid_b",
        "roi_meta_b",
        "roi_frame_valid_b",
        "roi_frame_meta_b",
    ):
        if key in batch[0]:
            result[key] = torch.stack([item[key] for item in batch], dim=0)
    if "local_frame_times" in batch[0]:
        result["local_frame_times"] = torch.stack(
            [item["local_frame_times"] for item in batch], dim=0
        )
    if "local_frame_times_b" in batch[0]:
        result["local_frame_times_b"] = torch.stack(
            [item["local_frame_times_b"] for item in batch], dim=0
        )
    if "highres_pool_inputs" in batch[0]:
        result["highres_pool_inputs"] = torch.stack(
            [item["highres_pool_inputs"] for item in batch], dim=0
        )
        result["highres_pool_times"] = torch.stack(
            [item["highres_pool_times"] for item in batch], dim=0
        )
    if "object_motion_inputs" in batch[0]:
        result["object_motion_inputs"] = torch.stack(
            [item["object_motion_inputs"] for item in batch], dim=0
        )
        result["object_motion_times"] = torch.stack(
            [item["object_motion_times"] for item in batch], dim=0
        )
    return result


def get_video_duration(path: str) -> float:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    cap.release()
    if fps <= 0 or frames <= 0:
        raise RuntimeError(f"Could not read duration from video: {path}")
    return frames / fps


def build_windows(duration: float, clip_sec: float, stride_sec: float, include_tail: bool) -> list[WindowRecord]:
    windows: list[WindowRecord] = []
    start = 0.0
    index = 0
    max_start = max(duration - clip_sec, 0.0)
    while start <= max_start + 1e-6:
        windows.append(WindowRecord(index=index, start_sec=start, end_sec=min(start + clip_sec, duration)))
        index += 1
        start += stride_sec
    if include_tail and windows and windows[-1].end_sec < duration - 1e-6:
        tail_start = max_start
        if tail_start > windows[-1].start_sec + 1e-6:
            windows.append(WindowRecord(index=index, start_sec=tail_start, end_sec=duration))
    if not windows:
        windows.append(WindowRecord(index=0, start_sec=0.0, end_sec=duration))
    return windows


def load_proposal_windows(proposal_dir: str, duration: float, clip_sec: float, labels: Sequence[str], dedupe_sec: float) -> tuple[list[WindowRecord], list[dict[str, Any]]]:
    label_to_file = {
        "shot": "shot_proposals.json",
        "save": "save_proposals.json",
        "set_piece": "set_piece_proposals.json",
    }
    proposals: list[dict[str, Any]] = []
    root = Path(proposal_dir)
    for label in labels:
        path = root / label_to_file[label]
        if not path.exists():
            continue
        with path.open() as f:
            items = json.load(f)
        for item in items:
            if not isinstance(item, dict) or "time_sec" not in item:
                continue
            proposals.append({
                "proposal_label": label,
                "time_sec": float(item["time_sec"]),
                "confidence": float(item.get("confidence", 0.0) or 0.0),
                "source": str(item.get("source", "")),
                "raw_event_type": str(item.get("event_type", label)),
            })
    proposals.sort(key=lambda item: (item["time_sec"], item["proposal_label"]))

    kept: list[dict[str, Any]] = []
    for proposal in proposals:
        if dedupe_sec > 0 and any(abs(float(proposal["time_sec"]) - float(old["time_sec"])) <= dedupe_sec for old in kept):
            continue
        kept.append(proposal)

    max_start = max(duration - clip_sec, 0.0)
    windows = []
    for index, proposal in enumerate(kept):
        start = min(max(float(proposal["time_sec"]) - clip_sec * 0.5, 0.0), max_start)
        windows.append(WindowRecord(index=index, start_sec=start, end_sec=min(start + clip_sec, duration)))
    return windows, kept


def load_checkpoint_model(checkpoint_path: str, device: torch.device, gpu_ids: list[int]) -> tuple[nn.Module, ConfigDict, list[str], dict[str, float]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint or "config" not in checkpoint:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    cfg = to_config(checkpoint["config"])
    configure_label_schema(cfg)
    labels = list(checkpoint.get("labels", train_mod.LABELS))
    thresholds = checkpoint.get("thresholds") or {label: 0.5 for label in labels}
    thresholds = {str(label): float(value) for label, value in thresholds.items()}

    # A full football checkpoint already contains the complete model state. Its
    # saved training config may reference an initialization checkpoint that is
    # unavailable on the evaluation machine, and loading it here would be both
    # redundant and incorrect.
    init_checkpoint = cfg.model.get("init_checkpoint", "")
    cfg.model["init_checkpoint"] = ""
    object_motion_enabled = bool(
        cfg.model.get("object_motion", ConfigDict()).get("enabled", False)
    )
    try:
        if object_motion_enabled:
            from football_object_motion.train import make_motion_model

            model = make_motion_model(
                cfg, use_cached_features=False, device=device
            )
        else:
            model = make_model(cfg, use_cached_features=False, device=device)
    finally:
        cfg.model["init_checkpoint"] = init_checkpoint
    checkpoint_keys = {
        str(key).removeprefix("module.") for key in checkpoint["model"]
    }
    object_residual_weights_present = any(
        key.startswith("object_spatial_aux.residual_head.")
        for key in checkpoint_keys
    )
    highres_residual_weights_present = any(
        key.startswith("highres_residual_head.")
        or key == "highres_residual_gate_logits"
        for key in checkpoint_keys
    )
    if "spatial_attention.roi_query_identity" in checkpoint_keys:
        from football_sparsemax_roi_residual import upgrade_roi_only_residual

        if model.spatial_attention is None:
            raise RuntimeError(
                "Sparse ROI checkpoint requires model.spatial_attention"
            )
        model.spatial_attention = upgrade_roi_only_residual(
            model.spatial_attention, sparsemax_temperature=4.0
        ).to(device)
        print(
            "Restored sparsemax ROI-only residual architecture from checkpoint",
            flush=True,
        )
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if object_motion_enabled:
        bad_missing = [
            key
            for key in missing
            if key.startswith("object_motion_adapter.") or "ball_lora_" in key
        ]
        bad_unexpected = [
            key
            for key in unexpected
            if key.startswith("object_motion_adapter.") or "ball_lora_" in key
        ]
        if bad_missing or bad_unexpected:
            raise RuntimeError(
                "Object Motion checkpoint did not load strictly: "
                f"missing={bad_missing[:8]} unexpected={bad_unexpected[:8]}"
            )
    if missing or unexpected:
        print(f"WARN load_state_dict missing={missing} unexpected={unexpected}", flush=True)
    if object_motion_enabled and len(gpu_ids) > 1:
        raise ValueError(
            "Object Motion dense evaluation must use one GPU per process; "
            "shard videos across processes instead of nn.DataParallel"
        )
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])
    # Keep the public return signature stable for downstream scripts while
    # exposing enough checkpoint provenance for the dense-inference guard.
    setattr(
        model,
        "_checkpoint_object_residual_weights_present",
        object_residual_weights_present,
    )
    setattr(
        model,
        "_checkpoint_highres_residual_weights_present",
        highres_residual_weights_present,
    )
    setattr(model, "_checkpoint_object_motion_enabled", object_motion_enabled)
    setattr(model, "_checkpoint_clip_thresholds", checkpoint.get("clip_thresholds"))
    setattr(
        model,
        "_checkpoint_threshold_semantics",
        checkpoint.get("threshold_semantics", "legacy_unspecified"),
    )
    model.eval()
    return model, cfg, labels, thresholds


def map_eval_label(label: str) -> str | None:
    if label == "shot":
        return "shot"
    if label == "save":
        return "save"
    if label in {"corner", "freekick", "penalty", "set_piece"}:
        return "set_piece"
    return None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_gt_events(annotation_path: str, source: str, video_id: str) -> list[dict[str, Any]]:
    events = load_annotation_events(Path(annotation_path).expanduser(), source, video_id)
    gt: list[dict[str, Any]] = []
    for event in events:
        labels = list(event.labels)
        for i, value in enumerate(labels):
            if value <= 0:
                continue
            label = map_eval_label(train_mod.LABELS[i])
            if label is None:
                continue
            gt.append({
                "label": label,
                "time_sec": float(event.anchor_time),
                "start_sec": float(event.start_time),
                "end_sec": float(event.end_time),
                "raw_label": event.raw_label,
                "event_type": event.event_type,
                "event_id": event.event_id,
            })
    gt.sort(key=lambda item: (item["label"], item["time_sec"]))
    return gt


def merge_predictions(rows: list[dict[str, Any]], labels: Sequence[str], thresholds: dict[str, float], merge_gap_sec: float) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for label in labels:
        positive = [row for row in rows if float(row[f"prob_{label}"]) >= thresholds[label]]
        positive.sort(key=lambda row: row["start_sec"])
        current: dict[str, Any] | None = None
        for row in positive:
            score = float(row[f"prob_{label}"])
            if current is None:
                current = {
                    "label": label,
                    "start_sec": float(row["start_sec"]),
                    "end_sec": float(row["end_sec"]),
                    "score": score,
                    "window_indices": [int(row["index"])],
                    "num_windows": 1,
                }
                continue
            if float(row["start_sec"]) <= float(current["end_sec"]) + merge_gap_sec:
                current["end_sec"] = max(float(current["end_sec"]), float(row["end_sec"]))
                current["score"] = max(float(current["score"]), score)
                current["window_indices"].append(int(row["index"]))
                current["num_windows"] += 1
            else:
                merged.append(current)
                current = {
                    "label": label,
                    "start_sec": float(row["start_sec"]),
                    "end_sec": float(row["end_sec"]),
                    "score": score,
                    "window_indices": [int(row["index"])],
                    "num_windows": 1,
                }
        if current is not None:
            merged.append(current)
    merged.sort(key=lambda item: (item["start_sec"], item["label"]))
    return merged


def point_nms_predictions(
    rows: list[dict[str, Any]],
    labels: Sequence[str],
    thresholds: dict[str, float],
    nms_radius_sec: float,
) -> list[dict[str, Any]]:
    """Convert positive clip windows into point events using score-ordered temporal NMS."""
    predictions: list[dict[str, Any]] = []
    for label in labels:
        candidates: list[dict[str, Any]] = []
        for row in rows:
            score = float(row[f"prob_{label}"])
            if score < thresholds[label]:
                continue
            support_start = float(row["start_sec"])
            support_end = float(row["end_sec"])
            event_time = (support_start + support_end) * 0.5
            candidates.append(
                {
                    "label": label,
                    "time_sec": event_time,
                    "start_sec": event_time,
                    "end_sec": event_time,
                    "support_start_sec": support_start,
                    "support_end_sec": support_end,
                    "score": score,
                    "window_indices": [int(row["index"])],
                    "num_windows": 1,
                }
            )

        candidates.sort(key=lambda item: (-float(item["score"]), float(item["time_sec"])))
        kept: list[dict[str, Any]] = []
        for candidate in candidates:
            nearby = next(
                (
                    old
                    for old in kept
                    if abs(float(candidate["time_sec"]) - float(old["time_sec"])) <= nms_radius_sec
                ),
                None,
            )
            if nearby is not None:
                nearby["window_indices"].extend(candidate["window_indices"])
                nearby["num_windows"] += 1
                nearby["support_start_sec"] = min(
                    float(nearby["support_start_sec"]), float(candidate["support_start_sec"])
                )
                nearby["support_end_sec"] = max(
                    float(nearby["support_end_sec"]), float(candidate["support_end_sec"])
                )
                continue
            kept.append(candidate)
        predictions.extend(kept)
    predictions.sort(key=lambda item: (float(item["time_sec"]), item["label"]))
    return predictions


def window_overlap_predictions(
    rows: list[dict[str, Any]],
    labels: Sequence[str],
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    """Keep every positive sliding window as an independent prediction."""
    predictions: list[dict[str, Any]] = []
    for row in rows:
        support_start = float(row["start_sec"])
        support_end = float(row["end_sec"])
        event_time = (support_start + support_end) * 0.5
        for label in labels:
            score = float(row[f"prob_{label}"])
            if score < thresholds[label]:
                continue
            predictions.append(
                {
                    "label": label,
                    "time_sec": event_time,
                    "start_sec": support_start,
                    "end_sec": support_end,
                    "support_start_sec": support_start,
                    "support_end_sec": support_end,
                    "score": score,
                    "window_indices": [int(row["index"])],
                    "num_windows": 1,
                }
            )
    predictions.sort(key=lambda item: (float(item["time_sec"]), item["label"], -float(item["score"])))
    return predictions


def merge_gt_events(gt_events: list[dict[str, Any]], merge_gap_sec: float) -> list[dict[str, Any]]:
    if merge_gap_sec <= 0:
        return gt_events
    merged: list[dict[str, Any]] = []
    for label in TARGET_LABELS:
        items = sorted([event for event in gt_events if event["label"] == label], key=lambda event: event["time_sec"])
        current: dict[str, Any] | None = None
        for event in items:
            start = float(event["time_sec"])
            end = float(event["time_sec"])
            if current is None:
                current = {**event, "start_sec": start, "end_sec": end, "event_ids": [event["event_id"]], "num_events": 1}
                continue
            if start <= float(current["end_sec"]) + merge_gap_sec:
                current["end_sec"] = end
                current["time_sec"] = (float(current["start_sec"]) + float(current["end_sec"])) * 0.5
                current["event_ids"].append(event["event_id"])
                current["num_events"] += 1
            else:
                merged.append(current)
                current = {**event, "start_sec": start, "end_sec": end, "event_ids": [event["event_id"]], "num_events": 1}
        if current is not None:
            merged.append(current)
    merged.sort(key=lambda item: (item["label"], item["time_sec"]))
    return merged


def prediction_time(pred: dict[str, Any]) -> float:
    if "time_sec" in pred:
        return float(pred["time_sec"])
    return (float(pred["start_sec"]) + float(pred["end_sec"])) * 0.5


def pred_matches_gt(
    pred: dict[str, Any],
    gt: dict[str, Any],
    tolerance_sec: float,
    *,
    matching_mode: str = "point",
) -> bool:
    if pred["label"] != gt["label"]:
        return False
    if matching_mode == "point":
        return abs(prediction_time(pred) - float(gt["time_sec"])) <= tolerance_sec
    if matching_mode == "interval":
        gt_start = float(gt.get("start_sec", gt["time_sec"]))
        gt_end = float(gt.get("end_sec", gt["time_sec"]))
        return float(pred["end_sec"]) + tolerance_sec >= gt_start and float(pred["start_sec"]) - tolerance_sec <= gt_end
    if matching_mode == "window":
        return float(pred["start_sec"]) - tolerance_sec <= float(gt["time_sec"]) <= float(pred["end_sec"]) + tolerance_sec
    raise ValueError(f"Unsupported matching_mode={matching_mode}")


def compute_event_metrics(
    predictions: list[dict[str, Any]],
    gt_events: list[dict[str, Any]],
    tolerance_sec: float,
    *,
    matching_mode: str = "point",
    allow_many_predictions_per_gt: bool = False,
) -> dict[str, Any]:
    per_class: dict[str, Any] = {}
    total_tp = total_fp = total_fn = 0
    matches: list[dict[str, Any]] = []
    for label in TARGET_LABELS:
        preds = [pred for pred in predictions if pred["label"] == label]
        gts = [gt for gt in gt_events if gt["label"] == label]
        used_gt: set[int] = set()
        matched_gt: set[int] = set()
        tp = 0
        fp = 0
        for pred in sorted(preds, key=lambda item: (-float(item["score"]), prediction_time(item))):
            pred_time = prediction_time(pred)
            matching_gt_indices = [
                (gt_idx, abs(pred_time - float(gt["time_sec"])))
                for gt_idx, gt in enumerate(gts)
                if (allow_many_predictions_per_gt or gt_idx not in used_gt)
                and pred_matches_gt(pred, gt, tolerance_sec, matching_mode=matching_mode)
            ]
            if matching_gt_indices:
                gt_idx, distance = min(matching_gt_indices, key=lambda item: item[1])
                if allow_many_predictions_per_gt:
                    matched_gt.update(item[0] for item in matching_gt_indices)
                else:
                    used_gt.add(gt_idx)
                    matched_gt.add(gt_idx)
                tp += 1
                matches.append(
                    {
                        "label": label,
                        "pred_time_sec": pred_time,
                        "pred_start_sec": pred["start_sec"],
                        "pred_end_sec": pred["end_sec"],
                        "pred_score": pred["score"],
                        "gt_time_sec": gts[gt_idx]["time_sec"],
                        "gt_event_id": gts[gt_idx].get("event_id", ""),
                        "distance_sec": distance,
                    }
                )
            else:
                fp += 1
        if allow_many_predictions_per_gt:
            fn = len(gts) - len(matched_gt)
            recall = len(matched_gt) / len(gts) if gts else 0.0
        else:
            fn = len(gts) - len(used_gt)
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        per_class[label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "num_pred": len(preds),
            "num_gt": len(gts),
            "num_matched_gt": len(matched_gt if allow_many_predictions_per_gt else used_gt),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        total_tp += tp
        total_fp += fp
        total_fn += fn
    micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
    if allow_many_predictions_per_gt:
        total_gt = sum(int(item["num_gt"]) for item in per_class.values())
        total_matched_gt = sum(int(item["num_matched_gt"]) for item in per_class.values())
        micro_recall = total_matched_gt / total_gt if total_gt > 0 else 0.0
    else:
        micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall > 0
        else 0.0
    )
    return {
        "matching_mode": matching_mode,
        "allow_many_predictions_per_gt": allow_many_predictions_per_gt,
        "tolerance_sec": tolerance_sec,
        "per_class": per_class,
        "micro": {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1,
        },
        "matches": sorted(matches, key=lambda item: (item["label"], item["gt_time_sec"])),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def parse_thresholds(raw: str | None, labels: Sequence[str], checkpoint_thresholds: dict[str, float]) -> dict[str, float]:
    if not raw or raw == "checkpoint":
        return {label: float(checkpoint_thresholds.get(label, 0.5)) for label in labels}
    if "=" not in raw:
        threshold = float(raw)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Scalar threshold must be in [0, 1]")
        return {label: threshold for label in labels}
    result = {label: float(checkpoint_thresholds.get(label, 0.5)) for label in labels}
    for item in raw.split(","):
        if not item.strip():
            continue
        key, value = item.split("=", 1)
        result[key.strip()] = float(value)
    return result


def random_topk_expected_min_offset(distances: Sequence[float], topk: int) -> float:
    ordered = sorted(float(value) for value in distances)
    count = len(ordered)
    if count == 0:
        return 0.0
    k = min(max(int(topk), 1), count)
    denominator = math.comb(count, k)
    expected = 0.0
    for rank in range(0, count - k + 1):
        probability = math.comb(count - rank - 1, k - 1) / denominator
        expected += ordered[rank] * probability
    return expected


def parse_label_float_map(raw: str, labels: Sequence[str], default: float) -> dict[str, float]:
    result = {label: float(default) for label in labels}
    if not raw.strip():
        return result
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            value = float(item)
            result = {label: value for label in labels}
            continue
        label, value = item.split("=", 1)
        label = label.strip()
        if label not in result:
            raise ValueError(f"Unknown label in map '{raw}': {label}; labels={list(labels)}")
        result[label] = float(value)
    return result


def calibrated_logit_fusion(
    clip_logits: Tensor,
    response_logits: Tensor,
    labels: Sequence[str],
    alpha_by_label: dict[str, float],
    clip_temp_by_label: dict[str, float],
    response_temp_by_label: dict[str, float],
) -> Tensor:
    if clip_logits.shape != response_logits.shape:
        raise ValueError(
            f"clip logits shape {tuple(clip_logits.shape)} != response logits {tuple(response_logits.shape)}"
        )
    fused = clip_logits.clone()
    for index, label in enumerate(labels):
        alpha = clamp(float(alpha_by_label.get(label, 0.5)), 0.0, 1.0)
        clip_temp = max(float(clip_temp_by_label.get(label, 1.0)), 1e-6)
        response_temp = max(float(response_temp_by_label.get(label, 1.0)), 1e-6)
        fused[:, index] = (
            alpha * clip_logits[:, index] / clip_temp
            + (1.0 - alpha) * response_logits[:, index] / response_temp
        )
    return fused


def select_window_score_logits(
    outputs: dict[str, Tensor],
    score_source: str,
    labels: Sequence[str],
    alpha_by_label: dict[str, float],
    clip_temp_by_label: dict[str, float],
    response_temp_by_label: dict[str, float],
) -> Tensor:
    """Select inference logits while keeping the model's final clip path intact.

    ``outputs['logits']`` is the public, fully fused model prediction. In
    particular, it includes optional spatial/object residuals that are absent
    from the diagnostic ``temporal_logits`` branch.
    """
    final_clip_logits = outputs["logits"]
    response_logits = outputs.get("response_clip_logits")
    if score_source in {"response", "fusion"} and response_logits is None:
        raise RuntimeError(
            f"--score-source={score_source} requires model "
            "outputs['response_clip_logits']; ensure response_curve_primary "
            "config exists and return_aux=True"
        )
    if score_source == "clip":
        return final_clip_logits
    if score_source == "response":
        assert response_logits is not None
        return response_logits
    if score_source == "fusion":
        assert response_logits is not None
        return calibrated_logit_fusion(
            final_clip_logits,
            response_logits,
            labels,
            alpha_by_label,
            clip_temp_by_label,
            response_temp_by_label,
        )
    raise ValueError(f"Unsupported score source: {score_source}")


def object_residual_inactive_reason(
    *,
    enabled: bool,
    weights_present: bool,
    observed: bool,
    max_abs: float,
) -> str | None:
    if not enabled or not weights_present:
        return None
    if not observed:
        return (
            "the checkpoint contains object residual weights and the branch "
            "is enabled, but inference returned no object residual"
        )
    if float(max_abs) == 0.0:
        return (
            "the enabled checkpoint-backed object residual was exactly zero "
            "for every evaluated window"
        )
    return None


def enforce_object_residual_activity(
    inactive_reason: str | None,
    *,
    fail: bool,
) -> None:
    if inactive_reason is None:
        return
    message = f"OBJECT RESIDUAL INACTIVE: {inactive_reason}."
    if fail:
        raise RuntimeError(message)
    warnings.warn(message, RuntimeWarning, stacklevel=1)


def gaussian_ranking_ndcg(scores: Sequence[float], distances: Sequence[float], sigma_sec: float) -> float:
    if not scores:
        return 0.0
    sigma = max(float(sigma_sec), 1e-6)
    relevance = [math.exp(-0.5 * (float(distance) / sigma) ** 2) for distance in distances]

    def dcg(order: Sequence[int]) -> float:
        return sum(
            (2.0 ** relevance[index] - 1.0) / math.log2(rank + 2.0)
            for rank, index in enumerate(order)
        )

    predicted = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))
    ideal = sorted(range(len(scores)), key=lambda index: (-relevance[index], index))
    ideal_dcg = dcg(ideal)
    return dcg(predicted) / ideal_dcg if ideal_dcg > 0 else 0.0


def binary_auc(scores: Sequence[float], targets: Sequence[int]) -> float | None:
    positives = sum(int(target) for target in targets)
    negatives = len(targets) - positives
    if positives == 0 or negatives == 0:
        return None
    ordered = sorted(enumerate(scores), key=lambda item: float(item[1]))
    rank_sum = 0.0
    position = 0
    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and float(ordered[end][1]) == float(ordered[position][1]):
            end += 1
        average_rank = (position + 1 + end) * 0.5
        rank_sum += average_rank * sum(int(targets[index]) for index, _ in ordered[position:end])
        position = end
    return (rank_sum - positives * (positives + 1) * 0.5) / (positives * negatives)


def mean_ci95(values: Sequence[float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    mean = sum(float(value) for value in values) / len(values)
    if len(values) == 1:
        return mean, mean, mean
    variance = sum((float(value) - mean) ** 2 for value in values) / (len(values) - 1)
    margin = 1.96 * math.sqrt(variance / len(values))
    return mean, mean - margin, mean + margin



def summarize_frame_localization_samples(
    samples: Sequence[dict[str, Any]],
    labels: Sequence[str],
) -> dict[str, Any]:
    metric_names = (
        "top1_abs_offset_sec",
        "topk_min_abs_offset_sec",
        "topk_mean_abs_offset_sec",
        "topk_weighted_abs_offset_sec",
        "oracle_min_abs_offset_sec",
        "random_topk_expected_min_abs_offset_sec",
        "random_topk_expected_mean_abs_offset_sec",
        "gaussian_ndcg",
        "top1_hit_1s",
        "top1_hit_2s",
        "topk_hit_1s",
        "topk_hit_2s",
    )
    result: dict[str, Any] = {}
    cohorts = sorted({str(row["cohort"]) for row in samples})
    branches = sorted({str(row["branch"]) for row in samples})
    for cohort in cohorts:
        result[cohort] = {}
        for branch in branches:
            branch_rows = [
                row for row in samples
                if str(row["cohort"]) == cohort and str(row["branch"]) == branch
            ]
            if not branch_rows:
                continue
            result[cohort][branch] = {}
            for label in [*labels, "all"]:
                selected = branch_rows if label == "all" else [
                    row for row in branch_rows if str(row["label"]) == label
                ]
                if not selected:
                    continue
                summary = {"num_samples": len(selected)}
                for name in metric_names:
                    summary[name] = sum(float(row[name]) for row in selected) / len(selected)
                gain_values = {
                    "top1_gain_vs_random_sec": [
                        float(row["random_topk_expected_mean_abs_offset_sec"])
                        - float(row["top1_abs_offset_sec"])
                        for row in selected
                    ],
                    "topk_min_gain_vs_random_sec": [
                        float(row["random_topk_expected_min_abs_offset_sec"])
                        - float(row["topk_min_abs_offset_sec"])
                        for row in selected
                    ],
                    "topk_mean_gain_vs_random_sec": [
                        float(row["random_topk_expected_mean_abs_offset_sec"])
                        - float(row["topk_mean_abs_offset_sec"])
                        for row in selected
                    ],
                }
                for name, values in gain_values.items():
                    mean, low, high = mean_ci95(values)
                    summary[name] = mean
                    summary[f"{name}_ci95_low"] = low
                    summary[f"{name}_ci95_high"] = high
                    summary[f"{name}_positive_rate"] = (
                        sum(value > 0.0 for value in values) / len(values)
                    )
                    if selected and all("video_id" in row for row in selected):
                        clusters: dict[tuple[str, str, str], list[float]] = {}
                        for row, value in zip(selected, values):
                            event_key = str(row.get("gt_event_id") or row.get("gt_time_sec"))
                            key = (str(row["video_id"]), str(row["label"]), event_key)
                            clusters.setdefault(key, []).append(value)
                        cluster_means = [
                            sum(cluster) / len(cluster) for cluster in clusters.values()
                        ]
                        cluster_mean, cluster_low, cluster_high = mean_ci95(cluster_means)
                        summary["num_unique_gt_events"] = len(clusters)
                        summary[f"{name}_event_macro"] = cluster_mean
                        summary[f"{name}_event_macro_ci95_low"] = cluster_low
                        summary[f"{name}_event_macro_ci95_high"] = cluster_high
                        summary[f"{name}_event_macro_positive_rate"] = (
                            sum(value > 0.0 for value in cluster_means) / len(cluster_means)
                        )
                random_room = (
                    summary["random_topk_expected_min_abs_offset_sec"]
                    - summary["oracle_min_abs_offset_sec"]
                )
                summary["topk_min_normalized_gain_vs_random"] = (
                    summary["topk_min_gain_vs_random_sec"] / random_room
                    if random_room > 1e-9 else 0.0
                )
                result[cohort][branch][label] = summary
    return result


def summarize_frame_window_scores(
    rows: Sequence[dict[str, Any]],
    labels: Sequence[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    branches = sorted({str(row["branch"]) for row in rows})
    for branch in branches:
        result[branch] = {}
        for label in labels:
            selected = [
                row for row in rows
                if str(row["branch"]) == branch and str(row["label"]) == label
            ]
            positives = [float(row["max_frame_prob"]) for row in selected if int(row["target"]) == 1]
            negatives = [float(row["max_frame_prob"]) for row in selected if int(row["target"]) == 0]
            targets = [int(row["target"]) for row in selected]
            scores = [float(row["max_frame_prob"]) for row in selected]
            result[branch][label] = {
                "num_positive_windows": len(positives),
                "num_negative_windows": len(negatives),
                "mean_max_prob_positive": sum(positives) / len(positives) if positives else 0.0,
                "mean_max_prob_negative": sum(negatives) / len(negatives) if negatives else 0.0,
                "mean_max_prob_gap": (
                    sum(positives) / len(positives) - sum(negatives) / len(negatives)
                    if positives and negatives else 0.0
                ),
                "auroc": binary_auc(scores, targets),
            }
    return result


def analyze_frame_event_outputs(
    frame_rows: Sequence[dict[str, Any]],
    gt_events: Sequence[dict[str, Any]],
    matches: Sequence[dict[str, Any]],
    labels: Sequence[str],
    *,
    topk: int,
    sigma_by_label: dict[str, float],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in frame_rows:
        grouped.setdefault(int(row["window_index"]), []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["frame_index"]))

    branch_prefixes = {"global": ""}
    if frame_rows and f"local_frame_logit_{labels[0]}" in frame_rows[0]:
        branch_prefixes["local"] = "local_"
        branch_prefixes["roi_fused"] = "roi_fused_"

    covering_pairs: set[tuple[int, str, str, float]] = set()
    for window_index, window_rows in grouped.items():
        start_sec = float(window_rows[0]["start_sec"])
        end_sec = float(window_rows[0]["end_sec"])
        for event in gt_events:
            label = str(event["label"])
            event_time = float(event["time_sec"])
            if label in labels and start_sec <= event_time <= end_sec:
                covering_pairs.add((
                    window_index,
                    label,
                    str(event.get("event_id", "")),
                    event_time,
                ))

    matched_pairs: set[tuple[int, str, str, float]] = set()
    tolerance_only_matches = 0
    unmatched_window_references = 0
    for match in matches:
        label = str(match["label"])
        if label not in labels:
            continue
        candidate_indices = [
            window_index
            for window_index, window_rows in grouped.items()
            if abs(float(window_rows[0]["start_sec"]) - float(match["pred_start_sec"])) < 1e-6
            and abs(float(window_rows[0]["end_sec"]) - float(match["pred_end_sec"])) < 1e-6
        ]
        if not candidate_indices:
            unmatched_window_references += 1
            continue
        event_time = float(match["gt_time_sec"])
        for window_index in candidate_indices:
            window_rows = grouped[window_index]
            if float(window_rows[0]["start_sec"]) <= event_time <= float(window_rows[0]["end_sec"]):
                matched_pairs.add((
                    window_index,
                    label,
                    str(match.get("gt_event_id", "")),
                    event_time,
                ))
            else:
                tolerance_only_matches += 1

    samples: list[dict[str, Any]] = []
    for cohort, pairs in (
        ("matched_windows", matched_pairs),
        ("gt_covering_windows", covering_pairs),
    ):
        for window_index, label, event_id, event_time in sorted(pairs):
            window_rows = grouped[window_index]
            frame_times = [float(row["frame_time_sec"]) for row in window_rows]
            distances = [abs(frame_time - event_time) for frame_time in frame_times]
            k = min(max(int(topk), 1), len(window_rows))
            for branch, prefix in branch_prefixes.items():
                scores = [float(row[f"{prefix}frame_logit_{label}"]) for row in window_rows]
                ranked = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
                selected = ranked[:k]
                selected_distances = [distances[index] for index in selected]
                selected_scores = [scores[index] for index in selected]
                max_score = max(selected_scores)
                weights = [math.exp(score - max_score) for score in selected_scores]
                weight_sum = sum(weights)
                weighted_offset = (
                    sum(weight * distance for weight, distance in zip(weights, selected_distances))
                    / max(weight_sum, 1e-12)
                )
                top1_offset = distances[ranked[0]]
                topk_min_offset = min(selected_distances)
                sample = {
                    "cohort": cohort,
                    "branch": branch,
                    "window_index": window_index,
                    "start_sec": float(window_rows[0]["start_sec"]),
                    "end_sec": float(window_rows[0]["end_sec"]),
                    "label": label,
                    "gt_event_id": event_id,
                    "gt_time_sec": event_time,
                    "topk": k,
                    "top1_frame_time_sec": frame_times[ranked[0]],
                    "top1_abs_offset_sec": top1_offset,
                    "topk_min_abs_offset_sec": topk_min_offset,
                    "topk_mean_abs_offset_sec": sum(selected_distances) / k,
                    "topk_weighted_abs_offset_sec": weighted_offset,
                    "oracle_min_abs_offset_sec": min(distances),
                    "random_topk_expected_min_abs_offset_sec": random_topk_expected_min_offset(distances, k),
                    "random_topk_expected_mean_abs_offset_sec": sum(distances) / len(distances),
                    "gaussian_ndcg": gaussian_ranking_ndcg(
                        scores,
                        distances,
                        sigma_by_label.get(label, 1.0),
                    ),
                    "top1_hit_1s": int(top1_offset <= 1.0),
                    "top1_hit_2s": int(top1_offset <= 2.0),
                    "topk_hit_1s": int(topk_min_offset <= 1.0),
                    "topk_hit_2s": int(topk_min_offset <= 2.0),
                    "topk_frame_indices": json.dumps(selected),
                    "topk_frame_times_sec": json.dumps([frame_times[index] for index in selected]),
                    "topk_abs_offsets_sec": json.dumps(selected_distances),
                    "topk_logits": json.dumps(selected_scores),
                }
                samples.append(sample)

    window_scores: list[dict[str, Any]] = []
    for window_index, window_rows in sorted(grouped.items()):
        start_sec = float(window_rows[0]["start_sec"])
        end_sec = float(window_rows[0]["end_sec"])
        for label in labels:
            target = int(any(
                str(event["label"]) == label
                and start_sec <= float(event["time_sec"]) <= end_sec
                for event in gt_events
            ))
            for branch, prefix in branch_prefixes.items():
                window_scores.append({
                    "window_index": window_index,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "branch": branch,
                    "label": label,
                    "target": target,
                    "max_frame_prob": max(
                        float(row[f"{prefix}frame_prob_{label}"]) for row in window_rows
                    ),
                })

    analysis = {
        "topk": int(topk),
        "frame_score_branches": list(branch_prefixes),
        "matched_window_definition": (
            "A clip-level true-positive window whose matched GT timestamp is inside the actual sampled window. "
            "Matches produced only by the evaluation tolerance outside the clip are excluded."
        ),
        "gt_covering_window_definition": (
            "Every window containing a same-class GT timestamp, independent of the clip classifier result."
        ),
        "num_tolerance_only_matches_excluded": tolerance_only_matches,
        "num_matches_without_exact_window": unmatched_window_references,
        "localization": summarize_frame_localization_samples(samples, labels),
        "window_discrimination": summarize_frame_window_scores(window_scores, labels),
    }
    return analysis, samples, window_scores



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a football checkpoint on one long video with sliding clips")
    parser.add_argument("--checkpoint", default="/mnt/data_16t/football/qiuqi/checkpoints/epoch_8.pt")
    parser.add_argument("--video-id", default="2027564888428580866")
    parser.add_argument("--video-path", default="/mnt/data/Datasets/Datasets/Football/xbotgo_football_data_0608/videos/2027564888428580866.mp4")
    parser.add_argument(
        "--annotation-path",
        default=str(DEFAULT_REPAIRED_GT_DIR / "2027564888428580866.json"),
        help="Repaired GT JSON. Defaults to ~/code/football_events_human_repair/{default_video_id}.json.",
    )
    parser.add_argument("--source", default="xbotgo_0608")
    parser.add_argument("--output-dir", default="outputs/long_video_eval/2027564888428580866_epoch_8")
    parser.add_argument("--clip-sec", type=float, default=10.0)
    parser.add_argument("--stride-sec", type=float, default=5.0)
    parser.add_argument("--image-size", default="", help="Optional inference image size override as H,W or HxW. Defaults to checkpoint config.")
    parser.add_argument("--proposal-dir", default="", help="If set, evaluate 10s windows centered on detector proposal times instead of dense sliding windows.")
    parser.add_argument("--proposal-dedupe-sec", type=float, default=0.0, help="Deduplicate proposal centers within this many seconds before model inference.")
    parser.add_argument("--include-tail", action="store_true", default=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu-ids", default="0", help="Comma-separated CUDA ids for DataParallel, e.g. 0,1. Empty disables DataParallel.")
    parser.add_argument(
        "--thresholds",
        default="0.5",
        help="Default: 0.5. Also accepts checkpoint or a comma list like shot=0.4,save=0.5,set_piece=0.6",
    )
    parser.add_argument("--eval-labels", default="", help="Comma-separated labels to evaluate, e.g. shot or shot,save. Defaults to all checkpoint target labels.")
    parser.add_argument(
        "--score-source",
        default="clip",
        choices=["clip", "response", "fusion"],
        help="Window score source: clip uses main clip logits, response uses dense-response pooled logits, fusion uses calibrated per-class logit fusion.",
    )
    parser.add_argument(
        "--object-motion-mode",
        default="checkpoint",
        choices=["checkpoint", "anchor-only"],
        help="Use the checkpoint residual, or force its scale to zero for an exact frozen-anchor ablation.",
    )
    parser.add_argument(
        "--fusion-alpha",
        default="shot=0.4,save=0.4,set_piece=0.8",
        help="Per-class clip-logit weight for --score-source=fusion. Dense weight is 1-alpha.",
    )
    parser.add_argument("--clip-temperature", default="1.0", help="Scalar or per-class temperature for clip logits in fusion.")
    parser.add_argument("--response-temperature", default="1.0", help="Scalar or per-class temperature for response logits in fusion.")
    parser.add_argument(
        "--prediction-postprocess",
        default="window_overlap",
        choices=["window_overlap", "point_nms", "interval_merge"],
        help=(
            "window_overlap keeps every positive sliding window and allows one GT event to match multiple windows; "
            "point_nms performs strict event spotting; interval_merge reproduces the legacy interval-coverage metric."
        ),
    )
    parser.add_argument("--nms-radius-sec", type=float, default=5.0)
    parser.add_argument("--merge-gap-sec", type=float, default=2.0, help="Legacy interval_merge only.")
    parser.add_argument("--match-tolerance-sec", type=float, default=2.0)
    parser.add_argument("--gt-merge-gap-sec", type=float, default=0.0)
    parser.add_argument("--decode-strategy", default="single_seek", choices=["single_seek", "multi_seek"])
    parser.add_argument("--video-reader-cache-size", type=int, default=2)
    parser.add_argument("--normalize-on-device", action="store_true", default=True)
    parser.add_argument("--spatial-crop-mode", default="none", choices=["none", "top_fixed", "adaptive_top_fixed", "detector_aware", "legacy_indexed", "robust_detector_aware"], help="Optional fixed-window spatial crop.")
    parser.add_argument("--top-crop-ratio", type=float, default=None, help="Top fraction removed in top_fixed mode; defaults to checkpoint spatial_crop.top_crop_ratio.")
    parser.add_argument("--adaptive-top-min-ratio", type=float, default=0.05)
    parser.add_argument("--adaptive-top-max-ratio", type=float, default=0.25)
    parser.add_argument("--adaptive-top-fallback-ratio", type=float, default=0.10)
    parser.add_argument("--adaptive-top-person-conf", type=float, default=0.35)
    parser.add_argument("--detector-index-root", default="", help="Compact .pt ROI indices for robust_detector_aware. Defaults to the checkpoint config.")
    parser.add_argument("--roi-temporal-mode", default="checkpoint", choices=["checkpoint", "clip", "dynamic"], help="Override robust ROI temporal mode from checkpoint config.")
    parser.add_argument("--roi-dynamic-context-sec", type=float, default=None)
    parser.add_argument("--roi-temporal-smoothing-window-sec", type=float, default=None)
    parser.add_argument("--roi-temporal-max-hold-sec", type=float, default=None)
    parser.add_argument("--roi-temporal-confidence-decay-sec", type=float, default=None)
    parser.add_argument("--detector-manifest-root", default="", help="Root containing per-video detection_tracking outputs; tracked_objects.json supplies people/ball and detections.json supplies goals.")
    parser.add_argument("--detector-ball-conf", type=float, default=0.5)
    parser.add_argument("--detector-goal-conf", type=float, default=0.5)
    parser.add_argument("--detector-person-conf", type=float, default=0.4)
    parser.add_argument("--detector-padding", type=float, default=0.12)
    parser.add_argument("--detector-min-crop-area-ratio", type=float, default=0.15)
    parser.add_argument("--detector-max-crop-area-ratio", type=float, default=0.85)
    parser.add_argument("--detector-max-frame-gap", type=int, default=5)
    parser.add_argument("--detector-window-roi-samples", type=int, default=8, help="Number of frames sampled from each temporal window to build one fixed ROI.")
    parser.add_argument("--detector-use-detection-goals", action="store_true", help="Also merge cls=2 goal boxes from detections.json. Default uses tracked_objects.json only.")
    parser.add_argument("--save-frame-event-logits", action="store_true", help="Save per-frame event logits and evaluate their temporal localization quality.")
    parser.add_argument("--frame-event-topk", type=int, default=8, help="Top-k frames used by frame event localization metrics.")
    parser.add_argument(
        "--fail-on-zero-object-residual",
        action="store_true",
        help=(
            "Fail instead of emitting a RuntimeWarning when an enabled, "
            "checkpoint-backed object residual remains exactly zero."
        ),
    )
    parser.add_argument("--max-windows", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotation_path = Path(args.annotation_path).expanduser().resolve()
    if not annotation_path.is_file():
        raise FileNotFoundError(f"GT annotation does not exist: {annotation_path}")
    args.annotation_path = str(annotation_path)
    if args.nms_radius_sec < 0:
        raise ValueError("--nms-radius-sec must be >= 0")
    if args.match_tolerance_sec < 0:
        raise ValueError("--match-tolerance-sec must be >= 0")
    if args.frame_event_topk <= 0:
        raise ValueError("--frame-event-topk must be > 0")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    gpu_ids = [int(item) for item in args.gpu_ids.split(",") if item.strip()] if device.type == "cuda" else []
    if len(gpu_ids) > 1:
        device = torch.device(f"cuda:{gpu_ids[0]}")

    print(f"loading checkpoint={args.checkpoint} device={device} gpu_ids={gpu_ids}", flush=True)
    model, cfg, labels, checkpoint_thresholds = load_checkpoint_model(args.checkpoint, device, gpu_ids)
    object_motion_enabled = bool(
        cfg.model.get("object_motion", ConfigDict()).get("enabled", False)
    )
    checkpoint_event_residual_scale = float(
        cfg.train.get("object_motion_event_residual_scale", 1.0) or 0.0
    )
    if args.object_motion_mode == "anchor-only":
        if not object_motion_enabled:
            raise ValueError("--object-motion-mode=anchor-only requires an Object Motion checkpoint")
        cfg.train["object_motion_event_residual_scale"] = 0.0
    if args.eval_labels.strip():
        requested_labels = [item.strip() for item in args.eval_labels.split(",") if item.strip()]
        invalid_labels = [label for label in requested_labels if label not in TARGET_LABELS or label not in labels]
        if invalid_labels:
            raise ValueError(f"Unsupported eval labels {invalid_labels}; checkpoint labels={labels}, supported={TARGET_LABELS}")
        eval_labels = requested_labels
    else:
        eval_labels = [label for label in TARGET_LABELS if label in labels]
    if args.thresholds.strip().lower() == "checkpoint" and args.score_source == "clip":
        saved_clip_thresholds = getattr(model, "_checkpoint_clip_thresholds", None)
        if isinstance(saved_clip_thresholds, dict):
            checkpoint_thresholds = {
                str(label): float(value)
                for label, value in saved_clip_thresholds.items()
            }
        online_cfg = cfg.get("eval", ConfigDict()).get(
            "online_validation", ConfigDict()
        )
        if (
            not isinstance(saved_clip_thresholds, dict)
            and
            bool(online_cfg.get("enabled", False))
            and str(online_cfg.get("score_fusion", "clip")) != "clip"
        ):
            raise ValueError(
                "checkpoint thresholds were tuned for online-event fused scores, not raw clip probabilities; "
                "pass --thresholds=0.5 or explicit clip-calibrated thresholds"
            )
    thresholds = parse_thresholds(args.thresholds, eval_labels, checkpoint_thresholds)
    fusion_alpha = parse_label_float_map(args.fusion_alpha, labels, 0.5)
    clip_temperature = parse_label_float_map(args.clip_temperature, labels, 1.0)
    response_temperature = parse_label_float_map(args.response_temperature, labels, 1.0)
    image_size = parse_image_size(args.image_size) if args.image_size else parse_image_size(cfg.video.image_size)
    num_frames = train_mod.effective_num_frames(cfg)

    duration = get_video_duration(args.video_path)
    proposal_rows: list[dict[str, Any]] = []
    if args.proposal_dir:
        windows, proposal_rows = load_proposal_windows(args.proposal_dir, duration, args.clip_sec, eval_labels, args.proposal_dedupe_sec)
        window_source = f"proposals:{args.proposal_dir}"
    else:
        windows = build_windows(duration, args.clip_sec, args.stride_sec, args.include_tail)
        window_source = "dense_sliding"
    if args.max_windows > 0:
        windows = windows[: args.max_windows]
        proposal_rows = proposal_rows[: args.max_windows]
    robust_cropper = RobustWindowCropper.from_args(args, image_size, cfg)
    legacy_indexed_cropper = None
    if args.spatial_crop_mode == "legacy_indexed":
        if not args.detector_index_root:
            raise ValueError("--detector-index-root is required for legacy_indexed")
        legacy_indexed_cropper = LegacyIndexedWindowCropper(
            args.detector_index_root,
            target_aspect=float(image_size[1]) / max(float(image_size[0]), 1.0),
            roi_samples=args.detector_window_roi_samples,
            ball_conf=args.detector_ball_conf,
            goal_conf=args.detector_goal_conf,
            person_conf=args.detector_person_conf,
            padding=args.detector_padding,
            min_crop_area_ratio=args.detector_min_crop_area_ratio,
            max_crop_area_ratio=args.detector_max_crop_area_ratio,
        )
    adaptive_top_cropper = None
    if args.spatial_crop_mode == "adaptive_top_fixed":
        if not args.detector_index_root:
            raise ValueError("--detector-index-root is required for adaptive_top_fixed")
        adaptive_top_cropper = AdaptiveTopWindowCropper(
            args.detector_index_root,
            target_aspect=float(image_size[1]) / max(float(image_size[0]), 1.0),
            min_ratio=args.adaptive_top_min_ratio,
            max_ratio=args.adaptive_top_max_ratio,
            fallback_ratio=args.adaptive_top_fallback_ratio,
            person_conf=args.adaptive_top_person_conf,
        )
    window_cropper = (
        adaptive_top_cropper
        or robust_cropper
        or legacy_indexed_cropper
        or WindowFixedDetectorCropper.from_args(args, image_size)
    )
    top_crop_provider = None
    if args.spatial_crop_mode == "top_fixed":
        configured_ratio = cfg.get("spatial_crop", ConfigDict()).get(
            "top_crop_ratio", 0.0
        )
        top_crop_provider = train_mod.TopBandCropProvider(
            configured_ratio if args.top_crop_ratio is None else args.top_crop_ratio
        )
    view_mode = str(cfg.model.get("view_fusion", "single"))
    global_image_size = parse_image_size(cfg.get("spatial_crop", ConfigDict()).get("global_image_size", image_size))
    highres_cfg = cfg.model.get("highres_glimpse", ConfigDict())
    highres_enabled = bool(highres_cfg.get("enabled", False))
    highres_pool_frames = int(highres_cfg.get("pool_frames", 0)) if highres_enabled else 0
    highres_pool_size = parse_image_size(
        highres_cfg.get("pool_image_size", image_size)
    )
    motion_cfg = cfg.model.get("object_motion", ConfigDict())
    motion_frames_per_segment = (
        int(motion_cfg.get("frames_per_segment", 0))
        if object_motion_enabled
        else 0
    )
    motion_image_size = (
        parse_image_size(motion_cfg.get("image_size", image_size))
        if object_motion_enabled
        else None
    )
    if highres_enabled and highres_pool_frames <= 0:
        raise RuntimeError("enabled high-resolution glimpse requires model.highres_glimpse.pool_frames > 0")
    if highres_enabled and args.spatial_crop_mode != "none":
        raise ValueError("high-resolution glimpse must receive uncropped full-image dense windows")
    print(
        f"video_id={args.video_id} duration={duration:.3f}s windows={len(windows)} source={window_source} "
        f"clip={args.clip_sec}s stride={args.stride_sec}s labels={eval_labels} thresholds={thresholds} "
        f"spatial_crop={args.spatial_crop_mode} highres_glimpse={highres_enabled} "
        f"highres_pool={highres_pool_frames}x{highres_pool_size}",
        flush=True,
    )

    dataset = SlidingWindowVideoDataset(
        video_path=args.video_path,
        video_id=args.video_id,
        windows=windows,
        num_frames=num_frames,
        image_size=image_size,
        normalize_on_cpu=not args.normalize_on_device,
        decode_strategy=args.decode_strategy,
        video_reader_cache_size=args.video_reader_cache_size,
        window_cropper=window_cropper,
        frame_crop_provider=top_crop_provider,
        view_mode=view_mode,
        global_image_size=global_image_size,
        dual_sampling=str(cfg.video.get("dual_sampling", "aligned")),
        roi_overlap_frames=int(cfg.video.get("roi_overlap_frames", 8)),
        num_rois=int(
            cfg.get("spatial_crop", ConfigDict()).get("num_rois", 1)
        ),
        highres_pool_frames=highres_pool_frames,
        highres_pool_size=highres_pool_size,
        object_motion_frames_per_segment=motion_frames_per_segment,
        object_motion_image_size=motion_image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
        collate_fn=collate_windows,
    )

    rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    object_residual_enabled = bool(
        cfg.model.get("object_spatial_aux", ConfigDict()).get("enabled", False)
    )
    object_motion_residual_observed = False
    object_motion_residual_max_abs = 0.0
    object_residual_weights_present = bool(
        getattr(model, "_checkpoint_object_residual_weights_present", False)
    )
    object_residual_observed = False
    object_residual_max_abs = 0.0
    highres_residual_weights_present = bool(
        getattr(model, "_checkpoint_highres_residual_weights_present", False)
    )
    highres_residual_observed = False
    highres_residual_max_abs = 0.0
    start_time = time.time()
    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            with autocast_context(device, bool(cfg.train.amp), str(cfg.train.amp_dtype)):
                if object_motion_enabled:
                    from football_object_motion.train import forward_motion_batch

                    outputs = forward_motion_batch(
                        model, batch, device, return_aux=True
                    )
                else:
                    outputs = train_mod.forward_model_batch(
                        model,
                        batch,
                        device,
                        return_aux=True,
                    )
            frame_logits = frame_probs = None
            local_frame_logits = local_frame_probs = None
            fused_frame_logits = fused_frame_probs = None
            mixed_frame_times = mixed_frame_view_ids = None
            mixed_frame_attention = mixed_frame_event_probs = None

            roi_quality_probs = roi_frame_quality_values = None
            spatial_probs = None
            if isinstance(outputs, dict):
                final_clip_logits = outputs["logits"]
                temporal_tensor = outputs.get("temporal_logits")
                response_tensor = outputs.get("response_clip_logits")
                logits = select_window_score_logits(
                    outputs,
                    args.score_source,
                    labels,
                    fusion_alpha,
                    clip_temperature,
                    response_temperature,
                )
                clip_probs = (
                    torch.sigmoid(final_clip_logits).float().cpu().numpy()
                )
                anchor_tensor = outputs.get("retention_reference_logits")
                anchor_probs = (
                    torch.sigmoid(anchor_tensor).float().cpu().numpy()
                    if anchor_tensor is not None
                    else None
                )
                object_motion_residual_tensor = outputs.get(
                    "object_motion_clip_residual"
                )
                object_motion_residual_values = (
                    object_motion_residual_tensor.float().cpu().numpy()
                    if object_motion_residual_tensor is not None
                    else None
                )
                if object_motion_residual_tensor is not None:
                    object_motion_residual_observed = True
                    object_motion_residual_max_abs = max(
                        object_motion_residual_max_abs,
                        float(
                            object_motion_residual_tensor.detach().abs().max().cpu()
                        ),
                    )
                temporal_probs = (
                    torch.sigmoid(temporal_tensor).float().cpu().numpy()
                    if temporal_tensor is not None
                    else None
                )
                response_probs = (
                    torch.sigmoid(response_tensor).float().cpu().numpy()
                    if response_tensor is not None
                    else None
                )
                global_tensor = outputs.get("global_logits")
                global_probs = (
                    torch.sigmoid(global_tensor).float().cpu().numpy()
                    if global_tensor is not None
                    else None
                )
                spatial_tensor = outputs.get("spatial_clip_logits")
                spatial_probs = (
                    torch.sigmoid(spatial_tensor).float().cpu().numpy()
                    if spatial_tensor is not None
                    else None
                )
                object_residual_tensor = outputs.get("object_spatial_residual")
                object_residual_values = (
                    object_residual_tensor.float().cpu().numpy()
                    if object_residual_tensor is not None
                    else None
                )
                if object_residual_tensor is not None:
                    object_residual_observed = True
                    object_residual_max_abs = max(
                        object_residual_max_abs,
                        float(object_residual_tensor.detach().abs().max().cpu()),
                    )
                highres_residual_tensor = outputs.get("highres_residual_delta")
                highres_residual_values = (
                    highres_residual_tensor.float().cpu().numpy()
                    if highres_residual_tensor is not None
                    else None
                )
                if highres_residual_tensor is not None:
                    highres_residual_observed = True
                    highres_residual_max_abs = max(
                        highres_residual_max_abs,
                        float(highres_residual_tensor.detach().abs().max().cpu()),
                    )
                if view_mode != "single":
                    local_probs = torch.sigmoid(outputs["local_logits"]).float().cpu().numpy()
                    roi_gates = outputs["roi_gate"].float().cpu().numpy()
                    quality_tensor = outputs.get("roi_quality_logits")
                    if quality_tensor is not None:
                        roi_quality_probs = torch.sigmoid(quality_tensor).float().cpu().numpy()
                    frame_quality_tensor = outputs.get("roi_frame_quality")
                    if frame_quality_tensor is not None:
                        roi_frame_quality_values = frame_quality_tensor.float().cpu().numpy()
                else:
                    local_probs = roi_gates = None
                if args.save_frame_event_logits:
                    frame_tensor = outputs.get("frame_event_logits")
                    if frame_tensor is None:
                        raise RuntimeError("Model did not return frame_event_logits")
                    frame_logits = frame_tensor.float().cpu().numpy()
                    frame_probs = torch.sigmoid(frame_tensor).float().cpu().numpy()
                    local_tensor = outputs.get("local_frame_event_logits")
                    if local_tensor is not None:
                        local_frame_logits = local_tensor.float().cpu().numpy()
                        local_frame_probs = torch.sigmoid(local_tensor).float().cpu().numpy()
                        fused_tensor = outputs.get("fused_frame_event_logits")
                        if fused_tensor is None and view_mode == "dual_gate":
                            gate_tensor = outputs["roi_gate"].to(local_tensor.dtype).unsqueeze(1)
                            fused_tensor = frame_tensor + gate_tensor * (local_tensor - frame_tensor)
                        if fused_tensor is not None:
                            fused_frame_logits = fused_tensor.float().cpu().numpy()
                            fused_frame_probs = torch.sigmoid(fused_tensor).float().cpu().numpy()
                            fused_time_tensor = outputs.get("fused_frame_times")
                            fused_view_tensor = outputs.get("fused_frame_view_ids")
                            fused_attention_tensor = outputs.get("fused_frame_attention")
                            if (
                                fused_time_tensor is not None
                                and fused_view_tensor is not None
                                and fused_attention_tensor is not None
                                and fused_tensor.shape[1] != frame_tensor.shape[1]
                            ):
                                mixed_frame_times = fused_time_tensor.float().cpu().numpy()
                                mixed_frame_view_ids = fused_view_tensor.long().cpu().numpy()
                                mixed_frame_attention = fused_attention_tensor.float().cpu().numpy()
                                attention_scale = mixed_frame_attention.max(axis=1, keepdims=True)
                                attention_scale = np.maximum(attention_scale, 1e-8)
                                mixed_frame_event_probs = fused_frame_probs
                                frame_logits = fused_frame_logits
                                frame_probs = fused_frame_probs * (mixed_frame_attention / attention_scale)
                                local_frame_logits = local_frame_probs = None
                                fused_frame_logits = fused_frame_probs = None
            else:
                if args.score_source != "clip":
                    raise RuntimeError("Non-dict model outputs only support --score-source=clip")
                logits = outputs
                clip_probs = torch.sigmoid(logits).float().cpu().numpy()
                response_probs = None
                temporal_probs = None
                object_residual_values = None
                object_motion_residual_values = None
                anchor_probs = None
                highres_residual_values = None
                global_probs = local_probs = roi_gates = None
                if args.save_frame_event_logits:
                    raise RuntimeError("Frame event logging requires model auxiliary outputs")
            probs = torch.sigmoid(logits).float().cpu().numpy()
            frame_times = batch["frame_times"].float().cpu().numpy()
            if mixed_frame_times is not None:
                frame_times = mixed_frame_times
            roi_frame_meta_values = batch["roi_frame_meta"].float().cpu().numpy()
            roi_frame_valid_values = batch["roi_frame_valid"].float().cpu().numpy()
            for i, meta in enumerate(batch["meta"]):
                row: dict[str, Any] = {
                    "index": int(meta["index"]),
                    "start_sec": float(meta["start_sec"]),
                    "end_sec": float(meta["end_sec"]),
                    "crop_mode": meta.get("crop_mode", "none"),
                    "crop_roi": json.dumps(meta.get("crop_roi"), ensure_ascii=False) if meta.get("crop_roi") is not None else "",
                    "crop_frame_rois": json.dumps(meta.get("crop_frame_rois") or [], ensure_ascii=False),
                    "roi_frame_valids": json.dumps(roi_frame_valid_values[i].tolist()),
                    "roi_frame_confidences": json.dumps(roi_frame_meta_values[i, :, 1].tolist()),
                    "crop_area_ratio": float(meta.get("crop_area_ratio", 0.0) or 0.0),
                    "adaptive_top_ratio": float(meta.get("adaptive_top_ratio", 0.0) or 0.0),
                    "crop_target_aspect": float(meta.get("crop_target_aspect", 0.0) or 0.0),
                    "crop_reason": meta.get("crop_reason", ""),
                    "crop_goal_count": int(meta.get("crop_goal_count", 0) or 0),
                    "crop_ball_count": int(meta.get("crop_ball_count", 0) or 0),
                    "crop_person_count": int(meta.get("crop_person_count", 0) or 0),
                    "roi_valid": float(meta.get("roi_valid", 0.0) or 0.0),
                    "roi_confidence": float(meta.get("roi_confidence", 0.0) or 0.0),
                    "roi_proposal_mode": meta.get("roi_proposal_mode", ""),
                    "roi_goal_score": float(meta.get("roi_goal_score", 0.0) or 0.0),
                    "roi_ball_score": float(meta.get("roi_ball_score", 0.0) or 0.0),
                    "roi_center_circle_score": float(meta.get("roi_center_circle_score", 0.0) or 0.0),
                    "roi_person_support": float(meta.get("roi_person_support", 0.0) or 0.0),
                    "roi_frame_quality_mean": (
                        float(roi_frame_quality_values[i].mean())
                        if roi_frame_quality_values is not None
                        else 0.0
                    ),
                }
                for label in eval_labels:
                    label_idx = labels.index(label)
                    row[f"prob_{label}"] = float(probs[i, label_idx])
                    row[f"pred_{label}"] = int(row[f"prob_{label}"] >= thresholds[label])
                    row[f"clip_prob_{label}"] = float(clip_probs[i, label_idx])
                    if anchor_probs is not None:
                        row[f"anchor_prob_{label}"] = float(
                            anchor_probs[i, label_idx]
                        )
                    if object_motion_residual_values is not None:
                        row[f"object_motion_residual_logit_{label}"] = float(
                            object_motion_residual_values[i, label_idx]
                        )
                    if temporal_probs is not None:
                        row[f"temporal_prob_{label}"] = float(
                            temporal_probs[i, label_idx]
                        )
                    if object_residual_values is not None:
                        row[f"object_residual_logit_{label}"] = float(
                            object_residual_values[i, label_idx]
                        )
                    if highres_residual_values is not None:
                        row[f"highres_residual_logit_{label}"] = float(
                            highres_residual_values[i, label_idx]
                        )
                    if response_probs is not None:
                        row[f"response_prob_{label}"] = float(response_probs[i, label_idx])
                    if global_probs is not None:
                        row[f"global_prob_{label}"] = float(global_probs[i, label_idx])
                    if spatial_probs is not None:
                        row[f"spatial_prob_{label}"] = float(spatial_probs[i, label_idx])
                    if global_probs is not None and local_probs is not None and roi_gates is not None:
                        row[f"global_prob_{label}"] = float(global_probs[i, label_idx])
                        row[f"local_prob_{label}"] = float(local_probs[i, label_idx])
                        gate_idx = label_idx if roi_gates.shape[1] > 1 else 0
                        row[f"roi_gate_{label}"] = float(roi_gates[i, gate_idx])
                        if roi_quality_probs is not None:
                            row[f"roi_quality_prob_{label}"] = float(roi_quality_probs[i, label_idx])
                    if frame_probs is not None:
                        frame_values = frame_probs[i, :, label_idx]
                        row[f"frame_max_prob_{label}"] = float(np.max(frame_values))
                        frame_topk = min(max(int(args.frame_event_topk), 1), int(frame_values.shape[0]))
                        if frame_topk > 0:
                            topk_values = np.partition(frame_values, -frame_topk)[-frame_topk:]
                            row[f"frame_topk_mean_prob_{label}"] = float(np.mean(topk_values))
                if args.save_frame_event_logits:
                    assert frame_logits is not None and frame_probs is not None
                    for frame_index, frame_time in enumerate(frame_times[i]):
                        view_id = -1
                        local_index = frame_index
                        if mixed_frame_view_ids is not None:
                            view_id = int(mixed_frame_view_ids[i, frame_index])
                            local_index = (
                                int(mixed_frame_view_ids[i, : frame_index + 1].sum()) - 1
                                if view_id == 1 else -1
                            )
                        crop_rois = meta.get("crop_frame_rois") or []
                        if 0 <= local_index < len(roi_frame_valid_values[i]):
                            crop_roi = crop_rois[local_index] if local_index < len(crop_rois) else None
                            frame_roi_valid = float(roi_frame_valid_values[i, local_index])
                            frame_roi_confidence = float(roi_frame_meta_values[i, local_index, 1])
                            frame_roi_quality = (
                                float(roi_frame_quality_values[i, local_index])
                                if roi_frame_quality_values is not None else 0.0
                            )
                        else:
                            crop_roi = None
                            frame_roi_valid = frame_roi_confidence = frame_roi_quality = 0.0
                        frame_row: dict[str, Any] = {
                            "window_index": int(meta["index"]),
                            "start_sec": float(meta["start_sec"]),
                            "end_sec": float(meta["end_sec"]),
                            "frame_index": int(frame_index),
                            "frame_time_sec": float(frame_time),
                            "relative_time_sec": float(frame_time) - float(meta["start_sec"]),
                            "view": "roi" if view_id == 1 else ("global" if view_id == 0 else "aligned"),
                            "crop_roi": json.dumps(crop_roi, ensure_ascii=False),
                            "roi_valid": frame_roi_valid,
                            "roi_confidence": frame_roi_confidence,
                            "roi_frame_quality": frame_roi_quality,
                        }
                        for label in eval_labels:
                            label_idx = labels.index(label)
                            frame_row[f"frame_logit_{label}"] = float(frame_logits[i, frame_index, label_idx])
                            frame_row[f"frame_prob_{label}"] = float(frame_probs[i, frame_index, label_idx])
                            if (
                                mixed_frame_event_probs is not None
                                and mixed_frame_attention is not None
                            ):
                                frame_row[f"frame_event_prob_{label}"] = float(mixed_frame_event_probs[i, frame_index, label_idx])
                                frame_row[f"class_attention_{label}"] = float(mixed_frame_attention[i, frame_index, label_idx])
                            if (
                                local_frame_logits is not None
                                and local_frame_probs is not None
                                and fused_frame_logits is not None
                                and fused_frame_probs is not None
                            ):
                                frame_row[f"local_frame_logit_{label}"] = float(
                                    local_frame_logits[i, frame_index, label_idx]
                                )
                                frame_row[f"local_frame_prob_{label}"] = float(
                                    local_frame_probs[i, frame_index, label_idx]
                                )
                                frame_row[f"roi_fused_frame_logit_{label}"] = float(
                                    fused_frame_logits[i, frame_index, label_idx]
                                )
                                frame_row[f"roi_fused_frame_prob_{label}"] = float(
                                    fused_frame_probs[i, frame_index, label_idx]
                                )
                        frame_rows.append(frame_row)
                rows.append(row)
            if step == 1 or step % 10 == 0 or step == len(loader):
                elapsed = time.time() - start_time
                print(f"step={step}/{len(loader)} windows={min(step * args.batch_size, len(windows))}/{len(windows)} elapsed={elapsed:.1f}s", flush=True)

    object_residual_all_zero = bool(
        object_residual_enabled
        and object_residual_weights_present
        and object_residual_observed
        and object_residual_max_abs == 0.0
    )
    inactive_reason = object_residual_inactive_reason(
        enabled=object_residual_enabled,
        weights_present=object_residual_weights_present,
        observed=object_residual_observed,
        max_abs=object_residual_max_abs,
    )
    enforce_object_residual_activity(
        inactive_reason, fail=args.fail_on_zero_object_residual
    )
    if object_motion_enabled and args.object_motion_mode == "checkpoint":
        if not object_motion_residual_observed:
            raise RuntimeError(
                "OBJECT MOTION INACTIVE: model returned no object_motion_clip_residual"
            )
        if checkpoint_event_residual_scale > 0.0 and object_motion_residual_max_abs == 0.0:
            raise RuntimeError(
                "OBJECT MOTION INACTIVE: enabled nonzero residual was exactly zero for every window"
            )
    if highres_enabled:
        if not highres_residual_weights_present:
            raise RuntimeError(
                "HIGHRES ROI INACTIVE: enabled checkpoint has no highres residual weights"
            )
        if not highres_residual_observed:
            raise RuntimeError(
                "HIGHRES ROI INACTIVE: model returned no highres_residual_delta"
            )
        if highres_residual_max_abs == 0.0:
            raise RuntimeError(
                "HIGHRES ROI INACTIVE: highres residual was exactly zero for every window"
            )

    gt_events = load_gt_events(args.annotation_path, args.source, args.video_id)
    gt_events = [event for event in gt_events if event["label"] in eval_labels]
    eval_gt_events = merge_gt_events(gt_events, args.gt_merge_gap_sec)
    allow_many_predictions_per_gt = False
    if args.prediction_postprocess == "window_overlap":
        predictions = window_overlap_predictions(rows, eval_labels, thresholds)
        matching_mode = "window"
        allow_many_predictions_per_gt = True
    elif args.prediction_postprocess == "point_nms":
        predictions = point_nms_predictions(rows, eval_labels, thresholds, args.nms_radius_sec)
        matching_mode = "point"
    else:
        predictions = merge_predictions(rows, eval_labels, thresholds, args.merge_gap_sec)
        matching_mode = "interval"
    metrics = compute_event_metrics(
        predictions,
        eval_gt_events,
        args.match_tolerance_sec,
        matching_mode=matching_mode,
        allow_many_predictions_per_gt=allow_many_predictions_per_gt,
    )
    anchor_predictions: list[dict[str, Any]] = []
    anchor_metrics: dict[str, Any] | None = None
    if (
        object_motion_enabled
        and args.score_source == "clip"
        and rows
        and all(f"anchor_prob_{label}" in rows[0] for label in eval_labels)
    ):
        anchor_rows = []
        for row in rows:
            anchor_row = dict(row)
            for label in eval_labels:
                anchor_row[f"prob_{label}"] = row[f"anchor_prob_{label}"]
                anchor_row[f"pred_{label}"] = int(
                    anchor_row[f"prob_{label}"] >= thresholds[label]
                )
            anchor_rows.append(anchor_row)
        if args.prediction_postprocess == "window_overlap":
            anchor_predictions = window_overlap_predictions(
                anchor_rows, eval_labels, thresholds
            )
        elif args.prediction_postprocess == "point_nms":
            anchor_predictions = point_nms_predictions(
                anchor_rows, eval_labels, thresholds, args.nms_radius_sec
            )
        else:
            anchor_predictions = merge_predictions(
                anchor_rows, eval_labels, thresholds, args.merge_gap_sec
            )
        anchor_metrics = compute_event_metrics(
            anchor_predictions,
            eval_gt_events,
            args.match_tolerance_sec,
            matching_mode=matching_mode,
            allow_many_predictions_per_gt=allow_many_predictions_per_gt,
        )
        metrics["anchor_only"] = {
            key: value for key, value in anchor_metrics.items() if key != "matches"
        }

    frame_analysis: dict[str, Any] | None = None
    frame_samples: list[dict[str, Any]] = []
    frame_window_scores: list[dict[str, Any]] = []
    if args.save_frame_event_logits:
        sigma_cfg = cfg.train.get("frame_label_sigma_sec", {})
        sigma_by_label = {
            "shot": float(sigma_cfg.get("shot", 0.8)),
            "save": float(sigma_cfg.get("save", 0.8)),
            "set_piece": float(sigma_cfg.get("set_piece", 1.5)),
        }
        frame_analysis, frame_samples, frame_window_scores = analyze_frame_event_outputs(
            frame_rows,
            eval_gt_events,
            metrics["matches"],
            eval_labels,
            topk=args.frame_event_topk,
            sigma_by_label=sigma_by_label,
        )


    manifest = {
        "checkpoint": args.checkpoint,
        "video_id": args.video_id,
        "video_path": args.video_path,
        "annotation_path": args.annotation_path,
        "annotation_sha256": file_sha256(annotation_path),
        "duration_sec": duration,
        "clip_sec": args.clip_sec,
        "stride_sec": args.stride_sec,
        "image_size": list(image_size),
        "global_image_size": list(global_image_size),
        "view_fusion": view_mode,
        "checkpoint_image_size": list(parse_image_size(cfg.video.image_size)),
        "num_windows": len(windows),
        "window_source": window_source,
        "proposal_dir": args.proposal_dir,
        "proposal_dedupe_sec": args.proposal_dedupe_sec,
        "num_proposals": len(proposal_rows),
        "labels": eval_labels,
        "thresholds": thresholds,
        "score_source": args.score_source,
        "object_motion": {
            "enabled": object_motion_enabled,
            "mode": args.object_motion_mode,
            "checkpoint_event_residual_scale": checkpoint_event_residual_scale,
            "effective_event_residual_scale": float(
                cfg.train.get("object_motion_event_residual_scale", 1.0) or 0.0
            ),
            "frames_per_segment": motion_frames_per_segment,
            "image_size": list(motion_image_size) if motion_image_size else None,
            "residual_observed": object_motion_residual_observed,
            "max_abs_logit_delta": object_motion_residual_max_abs,
            "anchor_only_metrics": (
                {key: value for key, value in anchor_metrics.items() if key != "matches"}
                if anchor_metrics is not None
                else None
            ),
        },
        "score_semantics_version": SCORE_SEMANTICS_VERSION,
        "fail_on_zero_object_residual": args.fail_on_zero_object_residual,
        "score_diagnostics": {
            "clip_field": "outputs.logits (final fully fused clip logits)",
            "temporal_field": "outputs.temporal_logits (diagnostic only)",
            "global_field": "outputs.global_logits (diagnostic only)",
            "object_residual": {
                "enabled": object_residual_enabled,
                "checkpoint_weights_present": object_residual_weights_present,
                "observed": object_residual_observed,
                "max_abs_logit_delta": object_residual_max_abs,
                "all_zero": object_residual_all_zero,
            },
            "highres_glimpse": {
                "enabled": highres_enabled,
                "checkpoint_weights_present": highres_residual_weights_present,
                "pool_frames": highres_pool_frames,
                "pool_image_size": list(highres_pool_size),
                "observed": highres_residual_observed,
                "max_abs_logit_delta": highres_residual_max_abs,
                "active": bool(
                    highres_enabled
                    and highres_residual_weights_present
                    and highres_residual_observed
                    and highres_residual_max_abs > 0.0
                ),
            },
        },
        "fusion": {
            "alpha": {label: fusion_alpha[label] for label in eval_labels},
            "clip_temperature": {label: clip_temperature[label] for label in eval_labels},
            "response_temperature": {label: response_temperature[label] for label in eval_labels},
        },
        "prediction_postprocess": args.prediction_postprocess,
        "nms_radius_sec": args.nms_radius_sec,
        "merge_gap_sec": args.merge_gap_sec,
        "match_tolerance_sec": args.match_tolerance_sec,
        "gt_merge_gap_sec": args.gt_merge_gap_sec,
        "frame_event": {"enabled": args.save_frame_event_logits, "topk": args.frame_event_topk},
        "spatial_crop": {
            "mode": args.spatial_crop_mode,
            "adaptive_top_min_ratio": args.adaptive_top_min_ratio,
            "adaptive_top_max_ratio": args.adaptive_top_max_ratio,
            "adaptive_top_fallback_ratio": args.adaptive_top_fallback_ratio,
            "adaptive_top_person_conf": args.adaptive_top_person_conf,
            "detector_index_root": args.detector_index_root,
            "roi_temporal_mode": (window_cropper.cropper.temporal_mode if isinstance(window_cropper, RobustWindowCropper) else None),
            "roi_dynamic_context_sec": (window_cropper.cropper.dynamic_context_sec if isinstance(window_cropper, RobustWindowCropper) else None),
            "roi_temporal_smoothing_window_sec": (window_cropper.cropper.temporal_smoothing_window_sec if isinstance(window_cropper, RobustWindowCropper) else None),
            "roi_temporal_max_hold_sec": (window_cropper.cropper.temporal_max_hold_sec if isinstance(window_cropper, RobustWindowCropper) else None),
            "roi_temporal_confidence_decay_sec": (window_cropper.cropper.temporal_confidence_decay_sec if isinstance(window_cropper, RobustWindowCropper) else None),
            "detector_manifest_root": args.detector_manifest_root,
            "detector_ball_conf": args.detector_ball_conf,
            "detector_goal_conf": args.detector_goal_conf,
            "detector_person_conf": args.detector_person_conf,
            "detector_padding": args.detector_padding,
            "detector_min_crop_area_ratio": args.detector_min_crop_area_ratio,
            "detector_max_crop_area_ratio": args.detector_max_crop_area_ratio,
            "detector_max_frame_gap": args.detector_max_frame_gap,
            "detector_window_roi_samples": args.detector_window_roi_samples,
            "detector_use_detection_goals": args.detector_use_detection_goals,
            "target_aspect": window_cropper.target_aspect if window_cropper is not None else None,
        },
        "crop_stats": {
            "num_cropped_windows": sum(1 for row in rows if row.get("crop_roi")),
            "num_uncropped_windows": sum(1 for row in rows if not row.get("crop_roi")),
            "mean_roi_confidence": (
                sum(float(row.get("roi_confidence", 0.0)) for row in rows) / max(len(rows), 1)
            ),
            "reasons": {reason: sum(1 for row in rows if row.get("crop_reason", "") == reason) for reason in sorted({str(row.get("crop_reason", "")) for row in rows})},
        },
        "metrics": {key: value for key, value in metrics.items() if key != "matches"},
        "elapsed_sec": time.time() - start_time,
    }
    (output_dir / "summary.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    (output_dir / "predicted_events.json").write_text(json.dumps(predictions, ensure_ascii=False, indent=2))
    if anchor_metrics is not None:
        (output_dir / "anchor_only_predicted_events.json").write_text(
            json.dumps(anchor_predictions, ensure_ascii=False, indent=2)
        )
    (output_dir / "gt_events.json").write_text(json.dumps(eval_gt_events, ensure_ascii=False, indent=2))
    if frame_analysis is not None:
        (output_dir / "frame_event_analysis.json").write_text(
            json.dumps(frame_analysis, ensure_ascii=False, indent=2)
        )
        if frame_rows:
            write_csv(output_dir / "frame_event_logits.csv", frame_rows, list(frame_rows[0]))
        if frame_samples:
            write_csv(output_dir / "frame_event_samples.csv", frame_samples, list(frame_samples[0]))
        if frame_window_scores:
            write_csv(
                output_dir / "frame_event_window_scores.csv",
                frame_window_scores,
                list(frame_window_scores[0]),
            )
    if proposal_rows:
        (output_dir / "input_proposals.json").write_text(json.dumps(proposal_rows, ensure_ascii=False, indent=2))

    window_fields = [
        "index",
        "start_sec",
        "end_sec",
        "crop_mode",
        "crop_roi",
        "crop_frame_rois",
        "roi_frame_valids",
        "roi_frame_confidences",
        "crop_area_ratio",
        "adaptive_top_ratio",
        "crop_target_aspect",
        "crop_reason",
        "crop_goal_count",
        "crop_ball_count",
        "crop_person_count",
        "roi_valid",
        "roi_confidence",
        "roi_proposal_mode",
        "roi_goal_score",
        "roi_ball_score",
        "roi_center_circle_score",
        "roi_person_support",
        "roi_frame_quality_mean",
    ]
    for label in eval_labels:
        window_fields.extend([f"prob_{label}", f"pred_{label}", f"clip_prob_{label}"])
        if rows and f"anchor_prob_{label}" in rows[0]:
            window_fields.append(f"anchor_prob_{label}")
        if rows and f"object_motion_residual_logit_{label}" in rows[0]:
            window_fields.append(f"object_motion_residual_logit_{label}")
        if rows and f"temporal_prob_{label}" in rows[0]:
            window_fields.append(f"temporal_prob_{label}")
        if rows and f"object_residual_logit_{label}" in rows[0]:
            window_fields.append(f"object_residual_logit_{label}")
        if rows and f"highres_residual_logit_{label}" in rows[0]:
            window_fields.append(f"highres_residual_logit_{label}")
        if rows and f"response_prob_{label}" in rows[0]:
            window_fields.append(f"response_prob_{label}")
        if rows and f"frame_max_prob_{label}" in rows[0]:
            window_fields.append(f"frame_max_prob_{label}")
        if rows and f"frame_topk_mean_prob_{label}" in rows[0]:
            window_fields.append(f"frame_topk_mean_prob_{label}")
        if rows and f"global_prob_{label}" in rows[0]:
            window_fields.append(f"global_prob_{label}")
        if rows and f"spatial_prob_{label}" in rows[0]:
            window_fields.append(f"spatial_prob_{label}")
        if view_mode != "single":
            window_fields.extend(
                [
                    f"local_prob_{label}",
                    f"roi_gate_{label}",
                    f"roi_quality_prob_{label}",
                ]
            )
    write_csv(output_dir / "window_predictions.csv", rows, window_fields)
    write_csv(
        output_dir / "predicted_events.csv",
        predictions,
        [
            "label",
            "time_sec",
            "start_sec",
            "end_sec",
            "support_start_sec",
            "support_end_sec",
            "score",
            "num_windows",
            "window_indices",
        ],
    )
    write_csv(output_dir / "gt_events.csv", eval_gt_events, ["label", "time_sec", "start_sec", "end_sec", "raw_label", "event_type", "event_id", "num_events", "event_ids"])
    write_csv(
        output_dir / "matches.csv",
        metrics["matches"],
        [
            "label",
            "pred_time_sec",
            "pred_start_sec",
            "pred_end_sec",
            "pred_score",
            "gt_time_sec",
            "gt_event_id",
            "distance_sec",
        ],
    )
    if proposal_rows:
        write_csv(output_dir / "input_proposals.csv", proposal_rows, ["proposal_label", "time_sec", "confidence", "source", "raw_event_type"])

    print(json.dumps(manifest["metrics"], ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
