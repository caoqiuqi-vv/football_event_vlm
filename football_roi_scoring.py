from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ObjectCandidate:
    box: tuple[float, float, float, float]
    score: float
    frame_count: int
    median_confidence: float
    center_stability: float
    size_stability: float
    iou_stability: float
    track_consistency: float


@dataclass(frozen=True)
class BallCandidate:
    box: tuple[float, float, float, float]
    score: float
    point_count: int
    detection_support: float
    max_gap_sec: float
    motion_continuity: float
    scale_stability: float
    touched: bool


def box_area(box: Sequence[float]) -> float:
    return max(float(box[2]) - float(box[0]), 0.0) * max(float(box[3]) - float(box[1]), 0.0)


def box_center(box: Sequence[float]) -> tuple[float, float]:
    return (float(box[0]) + float(box[2])) * 0.5, (float(box[1]) + float(box[3])) * 0.5


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(right - left, 0.0) * max(bottom - top, 0.0)
    union = box_area(first) + box_area(second) - intersection
    return intersection / union if union > 0 else 0.0


def robust_box(boxes: np.ndarray, low: float = 0.1, high: float = 0.9) -> tuple[float, float, float, float]:
    if len(boxes) == 1:
        return tuple(float(value) for value in boxes[0])
    return (
        float(np.quantile(boxes[:, 0], low)),
        float(np.quantile(boxes[:, 1], low)),
        float(np.quantile(boxes[:, 2], high)),
        float(np.quantile(boxes[:, 3], high)),
    )


def _geometry_distance(box: Sequence[float], reference: Sequence[float], width: int, height: int) -> tuple[float, float]:
    diagonal = max(math.hypot(width, height), 1.0)
    cx, cy = box_center(box)
    rx, ry = box_center(reference)
    center_distance = math.hypot(cx - rx, cy - ry) / diagonal
    box_w = max(float(box[2]) - float(box[0]), 1.0)
    box_h = max(float(box[3]) - float(box[1]), 1.0)
    ref_w = max(float(reference[2]) - float(reference[0]), 1.0)
    ref_h = max(float(reference[3]) - float(reference[1]), 1.0)
    size_distance = abs(math.log(box_w / ref_w)) + abs(math.log(box_h / ref_h))
    return center_distance, size_distance


def rank_object_candidates(
    boxes: np.ndarray,
    confidences: np.ndarray,
    track_ids: np.ndarray,
    frame_ids: np.ndarray,
    *,
    width: int,
    height: int,
    min_frames: int,
) -> list[ObjectCandidate]:
    """Cluster goal/center-circle detections using geometry and track evidence."""
    if len(boxes) == 0:
        return []
    order = np.argsort(frame_ids, kind="stable")
    clusters: list[dict[str, list]] = []
    for item_index in order.tolist():
        box = boxes[item_index].astype(np.float64)
        frame_id = int(frame_ids[item_index])
        track_id = int(track_ids[item_index])
        best_cluster = -1
        best_cost = float("inf")
        for cluster_index, cluster in enumerate(clusters):
            if frame_id in cluster["frame_ids"]:
                continue
            reference = cluster["reference"]
            center_distance, size_distance = _geometry_distance(box, reference, width, height)
            overlap = box_iou(box, reference)
            positive_tracks = [value for value in cluster["track_ids"] if value >= 0]
            same_track = track_id >= 0 and track_id in positive_tracks
            track_consistent = same_track and center_distance <= 0.12 and size_distance <= 1.50
            geometrically_close = overlap >= 0.12 or (center_distance <= 0.055 and size_distance <= 0.85)
            if not track_consistent and not geometrically_close:
                continue
            cost = center_distance + 0.05 * size_distance - 0.04 * overlap - 0.05 * float(track_consistent)
            if cost < best_cost:
                best_cluster = cluster_index
                best_cost = cost
        if best_cluster < 0:
            clusters.append({
                "boxes": [box],
                "frame_ids": [frame_id],
                "track_ids": [track_id],
                "confidences": [float(confidences[item_index])],
                "reference": box.copy(),
            })
        else:
            cluster = clusters[best_cluster]
            cluster["boxes"].append(box)
            cluster["frame_ids"].append(frame_id)
            cluster["track_ids"].append(track_id)
            cluster["confidences"].append(float(confidences[item_index]))
            cluster["reference"] = 0.8 * cluster["reference"] + 0.2 * box

    diagonal = max(math.hypot(width, height), 1.0)
    candidates: list[ObjectCandidate] = []
    for cluster in clusters:
        unique_frames = len(set(cluster["frame_ids"]))
        if unique_frames < min_frames:
            continue
        cluster_boxes = np.asarray(cluster["boxes"], dtype=np.float64)
        representative = robust_box(cluster_boxes)
        centers = (cluster_boxes[:, :2] + cluster_boxes[:, 2:]) * 0.5
        center_std = float(np.linalg.norm(np.std(centers, axis=0))) / diagonal
        center_stability = math.exp(-center_std / 0.045)
        areas = np.maximum((cluster_boxes[:, 2] - cluster_boxes[:, 0]) * (cluster_boxes[:, 3] - cluster_boxes[:, 1]), 1.0)
        size_stability = math.exp(-float(np.std(np.log(areas))) / 0.45)
        iou_stability = float(np.median([box_iou(box, representative) for box in cluster_boxes]))
        positive_tracks = [value for value in cluster["track_ids"] if value >= 0]
        if positive_tracks:
            _, counts = np.unique(np.asarray(positive_tracks, dtype=np.int64), return_counts=True)
            track_consistency = float(counts.max()) / len(positive_tracks)
        else:
            track_consistency = 0.5
        persistence = min(unique_frames / max(float(min_frames * 2), 1.0), 1.0)
        median_confidence = float(np.median(cluster["confidences"]))
        score = (
            0.25 * persistence
            + 0.22 * median_confidence
            + 0.18 * center_stability
            + 0.15 * size_stability
            + 0.12 * iou_stability
            + 0.08 * track_consistency
        )
        candidates.append(
            ObjectCandidate(
                box=representative,
                score=min(score, 1.0),
                frame_count=unique_frames,
                median_confidence=median_confidence,
                center_stability=center_stability,
                size_stability=size_stability,
                iou_stability=iou_stability,
                track_consistency=track_consistency,
            )
        )
    return sorted(candidates, key=lambda item: (item.score, item.frame_count), reverse=True)


def _raw_detection_support(
    point_frames: np.ndarray,
    point_boxes: np.ndarray,
    raw_frames: np.ndarray,
    raw_boxes: np.ndarray,
    *,
    fps: float,
    width: int,
    height: int,
) -> np.ndarray:
    support = np.zeros(len(point_frames), dtype=bool)
    if len(raw_frames) == 0:
        return support
    diagonal = max(math.hypot(width, height), 1.0)
    frame_tolerance = max(int(round(0.10 * fps)), 1)
    raw_centers = (raw_boxes[:, :2] + raw_boxes[:, 2:]) * 0.5
    for index, (frame_id, box) in enumerate(zip(point_frames, point_boxes)):
        nearby = np.abs(raw_frames - frame_id) <= frame_tolerance
        if not nearby.any():
            continue
        center = np.asarray(box_center(box))
        distances = np.linalg.norm(raw_centers[nearby] - center, axis=1) / diagonal
        overlaps = np.asarray([box_iou(box, raw_box) for raw_box in raw_boxes[nearby]])
        support[index] = bool((distances <= 0.025).any() or (overlaps >= 0.05).any())
    return support


def _best_short_gap_segment(frame_ids: np.ndarray, trusted: np.ndarray, fps: float) -> np.ndarray:
    if len(frame_ids) == 0:
        return np.zeros(0, dtype=bool)
    max_gap_frames = max(int(round(0.5 * fps)), 1)
    split_points = np.where(np.diff(frame_ids) > max_gap_frames)[0] + 1
    segments = np.split(np.arange(len(frame_ids)), split_points)
    best = max(segments, key=lambda values: (int(trusted[values].sum()), len(values)))
    selected = np.zeros(len(frame_ids), dtype=bool)
    selected[best] = True
    if trusted.any():
        distance_to_trusted = np.min(np.abs(frame_ids[:, None] - frame_ids[trusted][None, :]), axis=1)
        selected &= distance_to_trusted <= max_gap_frames
    return selected


def rank_ball_candidates(
    payload: dict,
    start_frame: int,
    end_frame: int,
    *,
    width: int,
    height: int,
    people_boxes: np.ndarray,
    raw_ball_frames: np.ndarray,
    raw_ball_boxes: np.ndarray,
    min_points: int,
    min_area_ratio: float,
    max_area_ratio: float,
) -> list[BallCandidate]:
    offsets = payload["ball_track_offsets"].numpy()
    all_frame_ids = payload["ball_frame_ids"].numpy()
    all_boxes = payload["ball_boxes"].numpy()
    all_point_touched = payload["ball_point_touched"].numpy().astype(bool)
    all_track_touched = payload["ball_track_touched"].numpy().astype(bool)
    det_width = float(payload["image_size"]["width"])
    det_height = float(payload["image_size"]["height"])
    fps = float(payload["fps"])
    diagonal = max(math.hypot(width, height), 1.0)
    people_centers = (people_boxes[:, :2] + people_boxes[:, 2:]) * 0.5 if len(people_boxes) else np.empty((0, 2))
    candidates: list[BallCandidate] = []

    for track_index in range(len(offsets) - 1):
        lo, hi = int(offsets[track_index]), int(offsets[track_index + 1])
        selected = (all_frame_ids[lo:hi] >= start_frame) & (all_frame_ids[lo:hi] <= end_frame)
        if int(selected.sum()) < min_points:
            continue
        frame_ids = all_frame_ids[lo:hi][selected]
        order = np.argsort(frame_ids)
        frame_ids = frame_ids[order]
        boxes = all_boxes[lo:hi][selected][order].astype(np.float64, copy=True)
        boxes[:, (0, 2)] *= float(width) / max(det_width, 1.0)
        boxes[:, (1, 3)] *= float(height) / max(det_height, 1.0)
        touched_points = all_point_touched[lo:hi][selected][order]
        detection_support = _raw_detection_support(
            frame_ids,
            boxes,
            raw_ball_frames,
            raw_ball_boxes,
            fps=fps,
            width=width,
            height=height,
        )
        track_touched = bool(all_track_touched[track_index])
        trusted = detection_support | touched_points
        if track_touched and not trusted.any():
            trusted[:] = True
        segment = _best_short_gap_segment(frame_ids, trusted, fps)
        if int(segment.sum()) < min_points:
            continue
        frame_ids = frame_ids[segment]
        boxes = boxes[segment]
        touched_points = touched_points[segment]
        detection_support = detection_support[segment]

        widths = np.maximum(boxes[:, 2] - boxes[:, 0], 1.0)
        heights = np.maximum(boxes[:, 3] - boxes[:, 1], 1.0)
        areas = widths * heights
        median_area_ratio = float(np.median(areas)) / max(float(width * height), 1.0)
        median_aspect = float(np.median(widths / heights))
        plausible = float(min_area_ratio <= median_area_ratio <= max_area_ratio and 0.35 <= median_aspect <= 2.85)
        duration = max(float(frame_ids[-1] - frame_ids[0]) / max(fps, 1e-6), 0.0)
        persistence = min(duration / 1.0, 1.0)
        gaps = np.diff(frame_ids).astype(np.float64) / max(fps, 1e-6)
        max_gap_sec = float(gaps.max()) if len(gaps) else 0.0
        gap_score = math.exp(-max_gap_sec / 0.5)
        centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
        if len(centers) >= 3:
            dt = np.maximum(np.diff(frame_ids).astype(np.float64) / max(fps, 1e-6), 1e-3)
            velocities = np.diff(centers, axis=0) / dt[:, None]
            speeds = np.linalg.norm(velocities, axis=1)
            if len(velocities) >= 2:
                norms = np.maximum(speeds, 1.0)
                direction_cosine = np.sum(velocities[:-1] * velocities[1:], axis=1) / (norms[:-1] * norms[1:])
                direction_score = float(np.clip((np.median(direction_cosine) + 1.0) * 0.5, 0.0, 1.0))
                acceleration = np.linalg.norm(np.diff(velocities, axis=0), axis=1)
                acceleration_score = 1.0 / (1.0 + float(np.median(acceleration)) / max(float(np.median(speeds)), 1.0))
            else:
                direction_score = acceleration_score = 0.5
            motion_continuity = 0.55 * direction_score + 0.45 * acceleration_score
        else:
            motion_continuity = 0.4
        scale_stability = math.exp(-float(np.std(np.log(np.maximum(areas, 1.0)))) / 0.55)
        if len(people_centers):
            distances = np.linalg.norm(centers[:, None, :] - people_centers[None, :, :], axis=-1)
            proximity = math.exp(-float(np.min(distances)) / (0.08 * diagonal))
        else:
            proximity = 0.0
        support_ratio = float(detection_support.mean()) if len(detection_support) else 0.0
        touched = track_touched or bool(touched_points.any())
        score = (
            0.18 * persistence
            + 0.14 * gap_score
            + 0.16 * plausible
            + 0.12 * support_ratio
            + 0.14 * motion_continuity
            + 0.08 * scale_stability
            + 0.12 * proximity
            + 0.06 * float(touched)
        )
        if plausible <= 0 and not touched:
            score *= 0.25
        if support_ratio <= 0 and not touched:
            score *= 0.65
        candidates.append(
            BallCandidate(
                box=robust_box(boxes),
                score=min(score, 1.0),
                point_count=len(frame_ids),
                detection_support=support_ratio,
                max_gap_sec=max_gap_sec,
                motion_continuity=motion_continuity,
                scale_stability=scale_stability,
                touched=touched,
            )
        )
    return sorted(candidates, key=lambda item: (item.score, item.detection_support, item.point_count), reverse=True)[:2]


def select_goal_candidate(
    candidates: Sequence[ObjectCandidate],
    ball: BallCandidate | None,
    center_circle: ObjectCandidate | None,
    *,
    width: int,
    height: int,
) -> tuple[ObjectCandidate | None, float]:
    if not candidates:
        return None, 0.0
    diagonal = max(math.hypot(width, height), 1.0)
    ranked: list[tuple[float, ObjectCandidate]] = []
    for candidate in candidates:
        adjusted = candidate.score
        if ball is not None and len(candidates) > 1:
            distance = math.dist(box_center(candidate.box), box_center(ball.box)) / diagonal
            adjusted = 0.85 * adjusted + 0.15 * math.exp(-distance / 0.65)
        if center_circle is not None and center_circle.score >= 0.55:
            gcx, gcy = box_center(candidate.box)
            ccx, ccy = box_center(center_circle.box)
            dx, dy = abs(gcx - ccx), abs(gcy - ccy)
            angle = math.degrees(math.atan2(dx, max(dy, 1e-6)))
            cc_width = max(center_circle.box[2] - center_circle.box[0], 1.0)
            goal_width = max(candidate.box[2] - candidate.box[0], 1.0)
            if angle < 60.0 or dx < max(cc_width * 0.5, goal_width * 0.6, width * 0.02):
                adjusted *= 0.15
        ranked.append((adjusted, candidate))
    adjusted, candidate = max(ranked, key=lambda item: item[0])
    return candidate, min(float(adjusted), 1.0)


def select_ball_candidate(
    candidates: Sequence[BallCandidate],
    goal: ObjectCandidate | None,
    center_circle: ObjectCandidate | None,
    *,
    width: int,
    height: int,
) -> BallCandidate | None:
    if not candidates:
        return None
    if len(candidates) == 1 or (goal is None and center_circle is None):
        return candidates[0]
    anchor = goal.box if goal is not None else center_circle.box
    diagonal = max(math.hypot(width, height), 1.0)
    return max(
        candidates,
        key=lambda candidate: 0.85 * candidate.score
        + 0.15 * math.exp(-math.dist(box_center(candidate.box), box_center(anchor)) / (0.75 * diagonal)),
    )


def stable_nearby_people(
    boxes: np.ndarray,
    confidences: np.ndarray,
    track_ids: np.ndarray,
    frame_ids: np.ndarray,
    anchor_boxes: Sequence[Sequence[float]],
    *,
    width: int,
    height: int,
    max_people: int,
    min_frames: int = 5,
) -> tuple[list[list[float]], float]:
    if len(boxes) == 0 or not anchor_boxes:
        return [], 0.0
    candidates = rank_object_candidates(
        boxes,
        confidences,
        track_ids,
        frame_ids,
        width=width,
        height=height,
        min_frames=min_frames,
    )
    if not candidates:
        return [], 0.0
    anchor_center = box_center((
        min(float(box[0]) for box in anchor_boxes),
        min(float(box[1]) for box in anchor_boxes),
        max(float(box[2]) for box in anchor_boxes),
        max(float(box[3]) for box in anchor_boxes),
    ))
    diagonal = max(math.hypot(width, height), 1.0)
    centers = np.asarray([box_center(candidate.box) for candidate in candidates], dtype=np.float64)
    anchor_distances = np.linalg.norm(centers - np.asarray(anchor_center, dtype=np.float64), axis=1)
    if len(candidates) > 1:
        pairwise = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
        density = (np.exp(-pairwise / (0.08 * diagonal)).sum(axis=1) - 1.0) / (len(candidates) - 1)
    else:
        density = np.ones(1, dtype=np.float64)
    attention_scores = []
    for index, candidate in enumerate(candidates):
        anchor_proximity = math.exp(-float(anchor_distances[index]) / (0.22 * diagonal))
        persistence = min(candidate.frame_count / 4.0, 1.0)
        attention_scores.append(
            0.40 * anchor_proximity
            + 0.35 * float(density[index])
            + 0.15 * persistence
            + 0.10 * candidate.score
        )
    ranked_indices = sorted(range(len(candidates)), key=lambda index: attention_scores[index], reverse=True)[:max_people]
    ranked = [candidates[index] for index in ranked_indices]
    mean_distance = float(np.mean([math.dist(anchor_center, box_center(candidate.box)) for candidate in ranked]))
    persistence = float(np.mean([min(candidate.frame_count / 4.0, 1.0) for candidate in ranked]))
    selected_density = float(np.mean(density[ranked_indices]))
    spatial_support = 0.65 * math.exp(-mean_distance / (0.18 * diagonal)) + 0.35 * selected_density
    support = min(len(ranked) / max(float(max_people), 1.0), 1.0) * persistence * spatial_support
    return [list(candidate.box) for candidate in ranked], min(support, 1.0)
