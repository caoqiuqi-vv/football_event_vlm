#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import cv2


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from football_detection_aware import ROI_MODES, RobustClipCropper
from train_football_events import (
    configure_label_schema,
    get_video_duration,
    labels_for_window,
    load_annotation_events,
    to_config,
)


MANIFEST_VERSION = "robust_window_roi_v3"
VIDEO_EXTENSIONS = (".mp4", ".MP4", ".mov", ".MOV", ".mkv", ".MKV")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a versioned robust-ROI manifest and visual audit set.")
    parser.add_argument("--index-root", default="outputs/football_roi_indices/robust_v2")
    parser.add_argument(
        "--video-root",
        default="/mnt/data_16t/football/raw_video_720P",
    )
    parser.add_argument(
        "--annotation-dir",
        default="/home/new_users/qiuqi/code/football_events_human_repair",
    )
    parser.add_argument(
        "--video-id-file",
        default="configs/football/splits/detector_aware_seed42_58videos/val_video_ids.txt",
    )
    parser.add_argument("--video-ids", default="", help="Optional comma-separated override for smoke tests.")
    parser.add_argument("--output-dir", default="outputs/football_roi_audit/robust_v3_val11")
    parser.add_argument("--clip-sec", type=float, default=10.0)
    parser.add_argument("--stride-sec", type=float, default=5.0)
    parser.add_argument("--input-size", default="384,640", help="Local classifier input height,width.")
    parser.add_argument("--max-windows-per-video", type=int, default=0)
    parser.add_argument("--visualizations-per-bucket", type=int, default=2)
    parser.add_argument("--padding", type=float, default=0.0)
    parser.add_argument("--min-area-ratio", type=float, default=0.0)
    parser.add_argument("--max-area-ratio", type=float, default=0.35)
    parser.add_argument("--min-roi-confidence", type=float, default=0.40)
    return parser.parse_args()


def read_video_ids(path: str) -> list[str]:
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip() and not line.startswith("#")]


def parse_input_size(value: str) -> tuple[int, int]:
    parts = [int(item.strip()) for item in value.split(",")]
    if len(parts) != 2 or min(parts) <= 0:
        raise ValueError(f"--input-size must be positive height,width, got {value}")
    return parts[0], parts[1]


def find_video(root: Path, video_id: str) -> Path:
    for extension in VIDEO_EXTENSIONS:
        path = root / f"{video_id}{extension}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing video_id={video_id} under {root}")


def video_geometry(path: Path) -> tuple[int, int, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise FileNotFoundError(f"Could not open {path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    capture.release()
    if width <= 0 or height <= 0 or fps <= 0:
        raise RuntimeError(f"Invalid video metadata for {path}: {width}x{height} fps={fps}")
    return width, height, fps


def build_windows(duration: float, clip_sec: float, stride_sec: float) -> list[tuple[float, float]]:
    max_start = max(duration - clip_sec, 0.0)
    starts: list[float] = []
    start = 0.0
    while start <= max_start + 1e-6:
        starts.append(start)
        start += stride_sec
    if not starts:
        starts = [0.0]
    if starts[-1] < max_start - 1e-6:
        starts.append(max_start)
    return [(start, min(start + clip_sec, duration)) for start in starts]


def bbox_iou(first: Sequence[float] | None, second: Sequence[float] | None) -> float:
    if first is None or second is None:
        return 0.0
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(right - left, 0.0) * max(bottom - top, 0.0)
    first_area = max(float(first[2]) - float(first[0]), 0.0) * max(float(first[3]) - float(first[1]), 0.0)
    second_area = max(float(second[2]) - float(second[0]), 0.0) * max(float(second[3]) - float(second[1]), 0.0)
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def row_buckets(row: dict[str, Any]) -> set[str]:
    buckets = {f"mode:{row['mode']}", "sample:background" if row["is_background"] else "sample:positive"}
    confidence = float(row["roi_confidence"])
    if not row["valid"]:
        buckets.add("confidence:invalid")
    elif confidence >= 0.65:
        buckets.add("confidence:high")
    else:
        buckets.add("confidence:mid")
    return buckets


def select_audit_rows(rows: list[dict[str, Any]], per_bucket: int) -> list[dict[str, Any]]:
    requested = [*(f"mode:{mode}" for mode in ROI_MODES), "confidence:high", "confidence:mid", "confidence:invalid", "sample:positive", "sample:background"]
    selected: dict[str, dict[str, Any]] = {}
    ranked = sorted(rows, key=lambda row: (-float(row["roi_confidence"]), row["video_id"], row["start_sec"]))
    for bucket in requested:
        matches = [row for row in ranked if bucket in row_buckets(row)]
        if bucket == "confidence:invalid":
            matches = sorted(matches, key=lambda row: (row["video_id"], row["start_sec"]))
        for row in matches[: max(per_bucket, 0)]:
            selected[row["row_id"]] = row
    return sorted(selected.values(), key=lambda row: (row["video_id"], row["start_sec"]))


def load_existing_reviews(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        return {row["row_id"]: row for row in csv.DictReader(handle)}


def write_visualization(row: dict[str, Any], video_path: Path, output_path: Path) -> None:
    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_MSEC, (float(row["start_sec"]) + float(row["end_sec"])) * 500.0)
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        return
    bbox = row.get("bbox")
    color = (0, 200, 0) if row["valid"] else (0, 0, 220)
    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
    labels = ",".join(row["positive_labels"]) or "background"
    text = f"{row['mode']} conf={row['roi_confidence']:.2f} area={row['area_ratio']:.2f} {labels}"
    cv2.rectangle(frame, (0, 0), (min(frame.shape[1], 1200), 48), (0, 0, 0), -1)
    cv2.putText(frame, text, (12, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame)


def manual_review_summary(selected: list[dict[str, Any]], existing: dict[str, dict[str, str]]) -> dict[str, Any]:
    high_ids = {row["row_id"] for row in selected if row["valid"] and float(row["roi_confidence"]) >= 0.65}
    reviewed = []
    for row_id in high_ids:
        value = str(existing.get(row_id, {}).get("reviewer_correct", "")).strip().lower()
        if value in {"1", "true", "yes", "y", "0", "false", "no", "n"}:
            reviewed.append(value in {"1", "true", "yes", "y"})
    accuracy = sum(reviewed) / len(reviewed) if reviewed else None
    return {
        "high_conf_selected": len(high_ids),
        "high_conf_reviewed": len(reviewed),
        "high_conf_correct": int(sum(reviewed)),
        "high_conf_accuracy": accuracy,
        "passes_0p90": accuracy >= 0.90 if accuracy is not None else None,
    }


def main() -> None:
    args = parse_args()
    configure_label_schema(to_config({"task": {"label_schema": "set_piece"}}))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_root = Path(args.video_root)
    annotation_dir = Path(args.annotation_dir)
    input_size = parse_input_size(args.input_size)
    cropper = RobustClipCropper(
        {
            "index_root": args.index_root,
            "padding": args.padding,
            "min_crop_area_ratio": args.min_area_ratio,
            "max_crop_area_ratio": args.max_area_ratio,
            "min_roi_confidence": args.min_roi_confidence,
        }
    )

    rows: list[dict[str, Any]] = []
    video_paths: dict[str, Path] = {}
    video_ids = (
        [item.strip() for item in args.video_ids.split(",") if item.strip()]
        if args.video_ids
        else read_video_ids(args.video_id_file)
    )
    for video_id in video_ids:
        video_path = find_video(video_root, video_id)
        annotation_path = annotation_dir / f"{video_id}.json"
        if not annotation_path.exists():
            raise FileNotFoundError(annotation_path)
        video_paths[video_id] = video_path
        duration = get_video_duration(str(video_path))
        width, height, _ = video_geometry(video_path)
        events = load_annotation_events(annotation_path, "xbotgo_0608", video_id)
        windows = build_windows(duration, args.clip_sec, args.stride_sec)
        if args.max_windows_per_video > 0:
            windows = windows[: args.max_windows_per_video]
        for window_index, (start_sec, end_sec) in enumerate(windows):
            proposal = cropper.get_window_roi(video_id, start_sec, end_sec, width, height, input_size)
            labels = labels_for_window(events, start_sec, end_sec)
            positive_labels = [label for label, value in zip(("shot", "save", "set_piece"), labels) if value > 0]
            crop_width = int(proposal.bbox[2] - proposal.bbox[0]) if proposal.bbox is not None else 0
            crop_height = int(proposal.bbox[3] - proposal.bbox[1]) if proposal.bbox is not None else 0
            integer_scale = crop_width // input_size[1] if proposal.bbox is not None else 0
            row = {
                "manifest_version": MANIFEST_VERSION,
                "index_version": 2,
                "row_id": f"{video_id}_{window_index:05d}",
                "video_id": video_id,
                "window_index": window_index,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "bbox": list(proposal.bbox) if proposal.bbox is not None else None,
                "valid": bool(proposal.valid),
                "mode": proposal.mode,
                "roi_confidence": float(proposal.roi_confidence),
                "goal_score": float(proposal.goal_score),
                "ball_score": float(proposal.ball_score),
                "center_circle_score": float(proposal.center_circle_score),
                "person_support": float(proposal.person_support),
                "area_ratio": float(proposal.area_ratio),
                "crop_width": crop_width,
                "crop_height": crop_height,
                "input_scale": integer_scale,
                "fallback_reason": proposal.fallback_reason,
                "positive_labels": positive_labels,
                "is_background": not positive_labels,
            }
            rows.append(row)
        print(f"video_id={video_id} windows={len(windows)}", flush=True)

    manifest_path = output_dir / "roi_manifest.jsonl"
    with manifest_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    adjacent_ious: list[float] = []
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_video[row["video_id"]].append(row)
    for video_rows in by_video.values():
        video_rows.sort(key=lambda row: row["start_sec"])
        for previous, current in zip(video_rows, video_rows[1:]):
            if previous["valid"] and current["valid"]:
                adjacent_ious.append(bbox_iou(previous["bbox"], current["bbox"]))

    valid_rows = [row for row in rows if row["valid"]]
    selected = select_audit_rows(rows, args.visualizations_per_bucket)
    review_path = output_dir / "manual_audit.csv"
    existing_reviews = load_existing_reviews(review_path)
    visualization_dir = output_dir / "visualizations"
    review_rows = []
    for row in selected:
        filename = f"{row['row_id']}_{row['mode']}.jpg"
        write_visualization(row, video_paths[row["video_id"]], visualization_dir / filename)
        old = existing_reviews.get(row["row_id"], {})
        review_rows.append(
            {
                "row_id": row["row_id"],
                "video_id": row["video_id"],
                "start_sec": row["start_sec"],
                "mode": row["mode"],
                "roi_confidence": row["roi_confidence"],
                "area_ratio": row["area_ratio"],
                "positive_labels": ",".join(row["positive_labels"]),
                "buckets": ",".join(sorted(row_buckets(row))),
                "visualization": str(Path("visualizations") / filename),
                "reviewer_correct": old.get("reviewer_correct", ""),
                "comment": old.get("comment", ""),
            }
        )
    with review_path.open("w", newline="") as handle:
        fieldnames = list(review_rows[0]) if review_rows else ["row_id", "reviewer_correct", "comment"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(review_rows)

    median_area = statistics.median([row["area_ratio"] for row in valid_rows]) if valid_rows else None
    median_adjacent_iou = statistics.median(adjacent_ious) if adjacent_ious else None
    manual = manual_review_summary(selected, existing_reviews)
    summary = {
        "manifest_version": MANIFEST_VERSION,
        "num_videos": len(by_video),
        "num_windows": len(rows),
        "num_valid": len(valid_rows),
        "valid_rate": len(valid_rows) / len(rows) if rows else 0.0,
        "mode_counts": dict(Counter(row["mode"] for row in rows)),
        "fallback_counts": dict(Counter(row["fallback_reason"] for row in rows if not row["valid"])),
        "confidence_counts": dict(Counter(next(bucket for bucket in row_buckets(row) if bucket.startswith("confidence:")) for row in rows)),
        "positive_windows": sum(not row["is_background"] for row in rows),
        "background_windows": sum(row["is_background"] for row in rows),
        "median_valid_area_ratio": median_area,
        "median_adjacent_valid_iou": median_adjacent_iou,
        "manual_review": manual,
        "acceptance": {
            "median_area_le_0p60": median_area <= 0.60 if median_area is not None else False,
            "median_adjacent_iou_ge_0p60": median_adjacent_iou >= 0.60 if median_adjacent_iou is not None else False,
            "high_conf_accuracy_ge_0p90": manual["passes_0p90"],
        },
        "config": {
            "index_root": args.index_root,
            "video_id_file": args.video_id_file,
            "clip_sec": args.clip_sec,
            "stride_sec": args.stride_sec,
            "input_size": list(input_size),
            "padding": args.padding,
            "min_area_ratio": args.min_area_ratio,
            "max_area_ratio": args.max_area_ratio,
            "min_roi_confidence": args.min_roi_confidence,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (output_dir / "manifest_meta.json").write_text(
        json.dumps({"manifest_version": MANIFEST_VERSION, "num_rows": len(rows), "fields": list(rows[0]) if rows else []}, ensure_ascii=False, indent=2)
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {manifest_path} and {review_path}", flush=True)


if __name__ == "__main__":
    main()
