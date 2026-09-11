#!/usr/bin/env python
"""Build camera-compensated external trajectory evidence for experiment B.

The index deliberately contains no event labels and no DINO-derived ball
predictions.  Training videos use the offline YOLO+track pseudo labels; the
held-out test18 videos use the existing RF-DETR tracking states.  Legacy
person/goal tracks are used only to estimate robust global image translation
and to encode the two goal sides symmetrically.

Output NPZ contract:
  frame_ids: int32 [N]
  fps: float32 scalar
  feats: float32 [N, 34]
  feature_names: unicode [34]
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import numpy as np

SCHEMA = "football.external-trajectory-evidence.v2"
FEATURE_NAMES = (
    "evidence_valid",
    "ball_tracked",
    "ball_observed",
    "ball_x",
    "ball_y",
    "ball_confidence",
    "ball_quality",
    "ball_uncertainty",
    "seconds_since_observed_norm",
    "ball_width",
    "ball_height",
    "ball_raw_vx",
    "ball_raw_vy",
    "ball_comp_vx",
    "ball_comp_vy",
    "ball_comp_speed",
    "ball_comp_accel",
    "camera_vx",
    "camera_vy",
    "camera_reliability",
    "camera_speed",
    "left_goal_visible",
    "left_goal_dx",
    "left_goal_dy",
    "left_goal_dist",
    "right_goal_visible",
    "right_goal_dx",
    "right_goal_dy",
    "right_goal_dist",
    "nearest_goal_dist",
    "both_goals_visible",
    "any_goal_visible",
    "ball_source_observed",
    "ball_source_interpolated",
)
FEATURE_DIM = len(FEATURE_NAMES)
IDX = {name: index for index, name in enumerate(FEATURE_NAMES)}
PERSON_CLASS = 0
GOAL_CLASS = 2


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                yield json.loads(raw)


def load_legacy(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def robust_camera_motion(
    frames: list[dict[str, Any]], fps: float, width: float, height: float
) -> tuple[np.ndarray, np.ndarray]:
    """Median tracked-person image velocity, a robust pan/tilt proxy."""
    count = len(frames)
    velocity = np.zeros((count, 2), dtype=np.float32)
    reliability = np.zeros(count, dtype=np.float32)
    previous: dict[int, tuple[float, float]] = {}
    previous_frame: int | None = None
    for index, frame in enumerate(frames):
        current: dict[int, tuple[float, float]] = {}
        for obj in frame.get("o", []):
            if len(obj) < 7 or int(obj[0]) != PERSON_CLASS:
                continue
            track_id = int(obj[6])
            if track_id < 0:
                continue
            current[track_id] = (
                0.5 * (float(obj[2]) + float(obj[4])) / width,
                0.5 * (float(obj[3]) + float(obj[5])) / height,
            )
        if previous_frame is not None:
            dt = (int(frame["f"]) - previous_frame) / max(fps, 1e-6)
            shared = sorted(set(previous).intersection(current))
            if dt > 0 and len(shared) >= 4:
                delta = np.asarray(
                    [
                        (
                            (current[key][0] - previous[key][0]) / dt,
                            (current[key][1] - previous[key][1]) / dt,
                        )
                        for key in shared
                    ],
                    dtype=np.float32,
                )
                median = np.median(delta, axis=0)
                mad = float(np.median(np.linalg.norm(delta - median, axis=1)))
                velocity[index] = median
                reliability[index] = min(len(shared) / 12.0, 1.0) * max(
                    0.0, 1.0 - mad / 0.08
                )
        previous = current
        previous_frame = int(frame["f"])

    # A short robust smoother removes single-frame identity switches without
    # suppressing real broadcast pans.
    result = velocity.copy()
    for index in range(count):
        lo, hi = max(0, index - 2), min(count, index + 3)
        valid = reliability[lo:hi] > 0
        if valid.any():
            result[index] = np.median(velocity[lo:hi][valid], axis=0)
    return np.clip(result, -2.0, 2.0), reliability


def goal_slots(
    objects: list[list[Any]], width: float, height: float
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    goals = []
    for obj in objects:
        if len(obj) < 6 or int(obj[0]) != GOAL_CLASS:
            continue
        goals.append(
            (
                0.5 * (float(obj[2]) + float(obj[4])) / width,
                0.5 * (float(obj[3]) + float(obj[5])) / height,
            )
        )
    goals.sort(key=lambda item: item[0])
    if not goals:
        return None, None
    if len(goals) == 1:
        return (goals[0], None) if goals[0][0] < 0.5 else (None, goals[0])
    return goals[0], goals[-1]


def yolo_rows(path: Path) -> dict[int, dict[str, Any]]:
    result = {}
    for row in read_jsonl(path):
        frame_id = int(row["source_frame_id"])
        old = result.get(frame_id)
        priority = (
            str(row.get("source", "")) == "track_observed",
            float(row.get("quality_weight", 0.0)),
            float(row.get("confidence", 0.0)),
        )
        old_priority = old.get("_priority", (-1, -1.0, -1.0)) if old else None
        if old is None or priority > old_priority:
            row["_priority"] = priority
            result[frame_id] = row
    return result


def rfdetr_rows(video_dir: Path, fps: float) -> dict[int, dict[str, Any]]:
    """Select one reliable tracked state per source frame across all segments."""
    detections: dict[int, list[dict[str, Any]]] = {}
    for path in video_dir.glob("part_*/football_predictions.jsonl"):
        for row in read_jsonl(path):
            frame_id = int(
                row.get(
                    "frame_index",
                    round(float(row["timestamp_seconds"]) * fps),
                )
            )
            x, y, width, height = (float(value) for value in row["xywh"])
            detections.setdefault(frame_id, []).append(
                {
                    "track_id": row.get("track_id"),
                    "center": (x, y),
                    "size": (width, height),
                    "score": float(row.get("score", 0.0)),
                }
            )

    def matched_confidence(frame_id: int, state: dict[str, Any]) -> float:
        candidates = detections.get(frame_id, [])
        if not candidates:
            return 0.0
        track_id = int(state.get("track_id", -1))
        exact = [
            row
            for row in candidates
            if row["track_id"] is not None
            and int(row["track_id"]) == track_id
        ]
        if exact:
            return max(float(row["score"]) for row in exact)
        center = tuple(float(value) for value in state["center"])
        nearest = min(
            candidates,
            key=lambda row: math.hypot(
                row["center"][0] - center[0],
                row["center"][1] - center[1],
            ),
        )
        distance = math.hypot(
            nearest["center"][0] - center[0],
            nearest["center"][1] - center[1],
        )
        tolerance = max(12.0, 0.75 * max(nearest["size"]))
        return float(nearest["score"]) if distance <= tolerance else 0.0

    selected: dict[int, tuple[tuple[float, ...], dict[str, Any], float]] = {}
    for path in video_dir.glob("part_*/track_states.jsonl"):
        for row in read_jsonl(path):
            timestamp = float(row.get("timestamp_seconds", 0.0))
            frame_id = int(
                row.get("frame_index", round(timestamp * fps))
            )
            status = str(row.get("status", ""))
            observation = str(row.get("observation", ""))
            seconds_missing = float(row.get("seconds_since_observed", 0.0))
            if observation != "observed" and seconds_missing > 0.5:
                continue
            hits = int(row.get("hits", 0))
            covariance = row.get("covariance_diagonal", [1e6, 1e6])
            position_variance = float(covariance[0]) + float(covariance[1])
            confidence = matched_confidence(frame_id, row)
            # Track confirmation is the first reliability condition.  Within
            # confirmed candidates prefer direct observations and their own
            # matched detector score, not a frame-global maximum.
            priority = (
                float(status == "confirmed"),
                float(observation == "observed"),
                confidence,
                min(hits / 20.0, 1.0),
                -position_variance,
            )
            old = selected.get(frame_id)
            if old is None or priority > old[0]:
                selected[frame_id] = (priority, row, confidence)

    result = {}
    for frame_id, (_, row, confidence) in selected.items():
        x, y = (float(value) for value in row["center"])
        bx, by, bw, bh = (float(value) for value in row["bbox"])
        covariance = row.get("covariance_diagonal", [64.0, 64.0])
        covariance_quality = 1.0 / (
            1.0
            + math.sqrt(
                max(float(covariance[0]) + float(covariance[1]), 0.0)
            )
            / 16.0
        )
        observed = str(row.get("observation", "")) == "observed"
        quality = math.sqrt(max(confidence, 0.0) * covariance_quality)
        if not observed:
            quality *= 0.5
        result[frame_id] = {
            "center_px": (x, y),
            "bbox_px": (bx, by, bx + bw, by + bh),
            "confidence": confidence,
            "quality_weight": quality,
            "source": (
                "track_observed" if observed else "track_interpolation"
            ),
            "seconds_since_observed": float(
                row.get("seconds_since_observed", 0.0)
            ),
            "track_id": int(row.get("track_id", -1)),
        }
    return result


def fill_ball(
    feats: np.ndarray,
    index: int,
    row: dict[str, Any],
    *,
    width: float,
    height: float,
    source: str,
) -> None:
    if "center_norm" in row:
        x, y = (float(value) for value in row["center_norm"])
        x1, y1, x2, y2 = (float(value) for value in row["bbox_xyxy_norm"])
    else:
        x, y = row["center_px"]
        x, y = x / width, y / height
        x1, y1, x2, y2 = row["bbox_px"]
        x1, x2 = x1 / width, x2 / width
        y1, y2 = y1 / height, y2 / height
    source_name = str(row.get("source", "unknown"))
    observed = source_name == "track_observed"
    confidence = float(row.get("confidence", 0.0))
    quality = float(row.get("quality_weight", 0.0))
    gap = float(row.get("seconds_since_observed", 0.0))
    if not observed and gap <= 0:
        gap = 0.25
    uncertainty = min(
        1.0,
        0.45 * (1.0 - max(0.0, min(confidence, 1.0)))
        + 0.35 * (1.0 - max(0.0, min(quality, 1.0)))
        + (0.25 if not observed else 0.0)
        + min(gap / 2.0, 0.25),
    )
    feats[index, IDX["ball_tracked"]] = 1.0
    feats[index, IDX["ball_observed"]] = float(observed)
    feats[index, IDX["ball_x"]] = np.clip(x, 0.0, 1.0)
    feats[index, IDX["ball_y"]] = np.clip(y, 0.0, 1.0)
    feats[index, IDX["ball_confidence"]] = np.clip(confidence, 0.0, 1.0)
    feats[index, IDX["ball_quality"]] = np.clip(quality, 0.0, 1.0)
    feats[index, IDX["ball_uncertainty"]] = uncertainty
    feats[index, IDX["seconds_since_observed_norm"]] = min(gap / 2.0, 1.0)
    feats[index, IDX["ball_width"]] = np.clip(x2 - x1, 0.0, 1.0)
    feats[index, IDX["ball_height"]] = np.clip(y2 - y1, 0.0, 1.0)
    feats[index, IDX["ball_source_observed"]] = float(observed)
    feats[index, IDX["ball_source_interpolated"]] = float(not observed)


def fill_kinematics(
    feats: np.ndarray, frame_ids: np.ndarray, fps: float, camera: np.ndarray
) -> None:
    tracked = feats[:, IDX["ball_tracked"]] > 0
    raw_velocity = np.zeros((len(feats), 2), dtype=np.float32)
    velocity_valid = np.zeros(len(feats), dtype=bool)
    previous: int | None = None
    for index in np.flatnonzero(tracked):
        if previous is not None:
            dt = (frame_ids[index] - frame_ids[previous]) / max(fps, 1e-6)
            if 0 < dt <= 0.35:
                raw_velocity[index] = (
                    feats[index, [IDX["ball_x"], IDX["ball_y"]]]
                    - feats[previous, [IDX["ball_x"], IDX["ball_y"]]]
                ) / dt
                velocity_valid[index] = True
        previous = int(index)
    compensated = raw_velocity - camera
    compensated[~velocity_valid] = 0.0
    raw_velocity[~velocity_valid] = 0.0
    acceleration = np.zeros(len(feats), dtype=np.float32)
    valid_indices = np.flatnonzero(velocity_valid)
    for current, old in zip(valid_indices[1:], valid_indices[:-1]):
        dt = (frame_ids[current] - frame_ids[old]) / max(fps, 1e-6)
        if 0 < dt <= 0.35:
            acceleration[current] = np.linalg.norm(
                compensated[current] - compensated[old]
            ) / dt
    feats[:, IDX["ball_raw_vx"]:IDX["ball_raw_vy"] + 1] = np.clip(
        raw_velocity, -2.0, 2.0
    )
    feats[:, IDX["ball_comp_vx"]:IDX["ball_comp_vy"] + 1] = np.clip(
        compensated, -2.0, 2.0
    )
    feats[:, IDX["ball_comp_speed"]] = np.clip(
        np.linalg.norm(compensated, axis=1), 0.0, 2.0
    )
    feats[:, IDX["ball_comp_accel"]] = np.clip(acceleration, 0.0, 4.0)


def build_video(
    video_id: str,
    *,
    yolo_root: Path,
    rfdetr_root: Path,
    legacy_root: Path,
    output_dir: Path,
    overwrite: bool,
) -> dict[str, Any]:
    destination = output_dir / f"{video_id}.npz"
    if destination.is_file() and not overwrite:
        with np.load(destination, allow_pickle=False) as payload:
            cached_schema = (
                str(payload["schema"].item())
                if "schema" in payload.files else ""
            )
            cached_names = (
                tuple(str(value) for value in payload["feature_names"])
                if "feature_names" in payload.files else ()
            )
            cached_features = payload["feats"]
            cache_valid = (
                cached_schema == SCHEMA
                and cached_features.ndim == 2
                and cached_features.shape[1] == FEATURE_DIM
                and cached_names == FEATURE_NAMES
            )
            if cache_valid:
                return {
                    "video_id": video_id,
                    "status": "cached",
                    "frames": int(len(payload["frame_ids"])),
                    "ball_fraction": float(
                        (
                            cached_features[:, IDX["ball_tracked"]] > 0
                        ).mean()
                    ),
                }

    legacy_path = legacy_root / video_id / "compact_detection_tracks.json.gz"
    legacy = load_legacy(legacy_path)
    if legacy is not None:
        meta = legacy["metadata"]
        fps = float(meta["sampling"]["source_fps"])
        width = float(meta["image_size"]["width"])
        height = float(meta["image_size"]["height"])
        frames = list(legacy.get("frames", []))
    else:
        # Some newly added training videos and four test18 videos have no
        # legacy person/goal pass.  Preserve ball evidence on a metadata-only
        # timeline; camera/goal masks stay explicitly unavailable.
        yolo_metadata = yolo_root / video_id / "metadata.json"
        if yolo_metadata.is_file():
            meta = json.loads(yolo_metadata.read_text(encoding="utf-8"))
            fps = float(meta["fps"])
            width = float(meta["canonical_width"])
            height = float(meta["canonical_height"])
            frame_count = int(meta["source_frames"])
            stride = int(meta["sample_stride"])
        else:
            metadata_paths = sorted(
                (rfdetr_root / video_id).glob(
                    "part_*/canonical_v2/*/metadata.json"
                )
            )
            if not metadata_paths:
                return {"video_id": video_id, "status": "missing_timeline_metadata"}
            meta = json.loads(metadata_paths[0].read_text(encoding="utf-8"))
            video_meta = meta["video"]
            fps = float(video_meta["fps_float"])
            width = float(video_meta["decoded_width"])
            height = float(video_meta["decoded_height"])
            frame_count = int(video_meta["frame_count"])
            stride = 2
        frames = [
            {"f": frame_id, "o": []}
            for frame_id in range(0, frame_count, max(stride, 1))
        ]
    frame_ids = np.asarray([int(frame["f"]) for frame in frames], dtype=np.int32)
    feats = np.zeros((len(frames), FEATURE_DIM), dtype=np.float32)
    feats[:, IDX["evidence_valid"]] = 1.0
    frame_to_index = {int(value): index for index, value in enumerate(frame_ids)}

    camera, camera_reliability = robust_camera_motion(
        frames, fps, width, height
    )
    feats[:, IDX["camera_vx"]:IDX["camera_vy"] + 1] = camera
    feats[:, IDX["camera_reliability"]] = camera_reliability
    feats[:, IDX["camera_speed"]] = np.linalg.norm(camera, axis=1)

    yolo_path = yolo_root / video_id / "ball_pseudolabels.jsonl"
    source = ""
    rows: dict[int, dict[str, Any]] = {}
    if yolo_path.is_file():
        rows = yolo_rows(yolo_path)
        source = "yolo"
    elif (rfdetr_root / video_id).is_dir():
        rows = rfdetr_rows(rfdetr_root / video_id, fps)
        source = "rfdetr"
    else:
        return {"video_id": video_id, "status": "missing_ball_source"}

    for frame_id, row in rows.items():
        nearest = frame_to_index.get(int(frame_id))
        if nearest is None:
            position = int(np.searchsorted(frame_ids, frame_id))
            candidates = [p for p in (position - 1, position) if 0 <= p < len(frame_ids)]
            if not candidates:
                continue
            nearest = min(candidates, key=lambda p: abs(int(frame_ids[p]) - frame_id))
            if abs(int(frame_ids[nearest]) - frame_id) / fps > 0.06:
                continue
        fill_ball(
            feats, nearest, row, width=width, height=height, source=source
        )

    for index, frame in enumerate(frames):
        left, right = goal_slots(frame.get("o", []), width, height)
        ball_visible = feats[index, IDX["ball_tracked"]] > 0
        ball = feats[index, [IDX["ball_x"], IDX["ball_y"]]]
        distances = []
        for side, goal, start in (
            ("left", left, IDX["left_goal_visible"]),
            ("right", right, IDX["right_goal_visible"]),
        ):
            if goal is None:
                continue
            feats[index, start] = 1.0
            if ball_visible:
                dx, dy = goal[0] - ball[0], goal[1] - ball[1]
                feats[index, IDX[f"{side}_goal_dx"]] = dx
                feats[index, IDX[f"{side}_goal_dy"]] = dy
                distance = math.hypot(dx, dy)
                feats[index, IDX[f"{side}_goal_dist"]] = distance
                distances.append(distance)
        feats[index, IDX["both_goals_visible"]] = float(
            left is not None and right is not None
        )
        feats[index, IDX["any_goal_visible"]] = float(
            left is not None or right is not None
        )
        if distances:
            feats[index, IDX["nearest_goal_dist"]] = min(distances)

    fill_kinematics(feats, frame_ids, fps, camera)
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema=np.asarray(SCHEMA),
            frame_ids=frame_ids,
            fps=np.float32(fps),
            feats=feats,
            feature_names=np.asarray(FEATURE_NAMES),
            source=np.asarray(source),
        )
    os.replace(temporary, destination)
    return {
        "video_id": video_id,
        "status": "built",
        "source": source,
        "frames": len(frames),
        "ball_fraction": float((feats[:, IDX["ball_tracked"]] > 0).mean()),
        "observed_fraction": float((feats[:, IDX["ball_observed"]] > 0).mean()),
        "camera_reliable_fraction": float((camera_reliability > 0).mean()),
        "goal_fraction": float((feats[:, IDX["any_goal_visible"]] > 0).mean()),
    }


def discover_ids(yolo_root: Path, rfdetr_root: Path) -> list[str]:
    values = {
        path.parent.name
        for path in yolo_root.glob("*/ball_pseudolabels.jsonl")
    }
    values.update(
        path.name
        for path in rfdetr_root.iterdir()
        if path.is_dir() and path.name.isdigit()
    )
    return sorted(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolo-root", type=Path, required=True)
    parser.add_argument("--rfdetr-root", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video-ids", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.video_ids:
        video_ids = [
            line.strip()
            for line in args.video_ids.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    else:
        video_ids = discover_ids(args.yolo_root, args.rfdetr_root)
    if args.limit > 0:
        video_ids = video_ids[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    common = {
        "yolo_root": args.yolo_root,
        "rfdetr_root": args.rfdetr_root,
        "legacy_root": args.legacy_root,
        "output_dir": args.output_dir,
        "overwrite": args.overwrite,
    }
    results = []
    with ProcessPoolExecutor(max_workers=max(args.workers, 1)) as pool:
        futures = {
            pool.submit(build_video, video_id, **common): video_id
            for video_id in video_ids
        }
        for position, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                f"[{position}/{len(futures)}] {result['video_id']} "
                f"{result['status']}",
                flush=True,
            )

    results.sort(key=lambda row: row["video_id"])
    manifest = {
        "schema": SCHEMA,
        "feature_dim": FEATURE_DIM,
        "feature_names": FEATURE_NAMES,
        "sources": {
            "yolo_root": str(args.yolo_root.resolve()),
            "rfdetr_root": str(args.rfdetr_root.resolve()),
            "legacy_root": str(args.legacy_root.resolve()),
        },
        "videos": results,
        "summary": {
            status: sum(row["status"] == status for row in results)
            for status in sorted({row["status"] for row in results})
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    temporary = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, manifest_path)
    print(json.dumps(manifest["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
