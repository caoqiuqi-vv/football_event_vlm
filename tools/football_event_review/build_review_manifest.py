#!/usr/bin/env python3
"""Convert dense DINO football evaluation outputs into review-tool input."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_LABELS = ("shot", "save", "set_piece")
DEFAULT_REVIEW_LABELS = ("shot", "save", "free_kick", "penalty", "corner", "shot_on_target")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".m4v")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_json_list(value: str | None) -> list[int]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
        return [int(item) for item in parsed]
    except (json.JSONDecodeError, TypeError, ValueError):
        return []


def find_video(video_root: Path, video_id: str) -> Path | None:
    for extension in VIDEO_EXTENSIONS:
        candidate = video_root / f"{video_id}{extension}"
        if candidate.exists():
            return candidate.resolve()
    direct = sorted(
        path for path in video_root.glob(f"{video_id}*") if path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if direct:
        return direct[0].resolve()
    recursive = sorted(
        path for path in video_root.rglob(f"{video_id}*") if path.suffix.lower() in VIDEO_EXTENSIONS
    )
    return recursive[0].resolve() if recursive else None


def get_video_ids(run_dir: Path, requested: Iterable[str] | None) -> list[str]:
    if requested:
        return list(dict.fromkeys(str(item) for item in requested))
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        configured = [str(item) for item in config.get("video_ids", [])]
        existing = [item for item in configured if (run_dir / item / "predicted_events.csv").exists()]
        if existing:
            return existing
    return sorted(
        path.name
        for path in run_dir.iterdir()
        if path.is_dir() and (path / "predicted_events.csv").exists()
    )


def stable_event_id(video_id: str, label: str, members: list[dict[str, Any]]) -> str:
    signature = ";".join(
        f"{row['time_sec']:.3f}:{row['score']:.6f}" for row in members
    )
    digest = hashlib.sha1(f"{video_id}|{label}|{signature}".encode()).hexdigest()[:14]
    return f"{video_id}_{label}_{digest}"


def merge_candidates(
    rows: list[dict[str, Any]], merge_sec: float, labels: tuple[str, ...]
) -> list[list[dict[str, Any]]]:
    if merge_sec <= 0:
        return [[row] for row in rows]
    groups: list[list[dict[str, Any]]] = []
    for label in labels:
        label_rows = sorted(
            (row for row in rows if row["label"] == label), key=lambda row: row["time_sec"]
        )
        current: list[dict[str, Any]] = []
        for row in label_rows:
            if current and (
                row["time_sec"] - current[-1]["time_sec"] > merge_sec + 1e-6
                or row["evaluation_status"] != current[-1]["evaluation_status"]
            ):
                groups.append(current)
                current = []
            current.append(row)
        if current:
            groups.append(current)
    return sorted(groups, key=lambda group: (group[0]["time_sec"], group[0]["label"]))


def build_video_entry(
    video_id: str,
    eval_dir: Path,
    video_path: Path,
    labels: tuple[str, ...],
    merge_sec: float,
    match_tolerance_sec: float = 2.0,
    exclusions: frozenset[tuple[str, str]] = frozenset(),
) -> dict[str, Any]:
    window_rows = read_csv(eval_dir / "window_predictions.csv")
    windows_by_index = {int(row["index"]): row for row in window_rows}

    frame_rows = read_csv(eval_dir / "frame_event_window_scores.csv")
    frame_by_window_label: dict[tuple[int, str], float] = {}
    for row in frame_rows:
        if row.get("branch", "global") != "global":
            continue
        key = (int(row["window_index"]), row["label"])
        frame_by_window_label[key] = max(
            frame_by_window_label.get(key, 0.0), as_float(row.get("max_frame_prob"))
        )

    timeline: list[dict[str, Any]] = []
    for row in window_rows:
        window_index = int(row["index"])
        timeline.append(
            {
                "index": window_index,
                "start_sec": as_float(row.get("start_sec")),
                "end_sec": as_float(row.get("end_sec")),
                "dino": {label: as_float(row.get(f"prob_{label}")) for label in labels},
                "frame_detection": {
                    label: frame_by_window_label.get((window_index, label), 0.0)
                    for label in labels
                },
                "roi": {
                    "valid": as_float(row.get("roi_valid")) > 0.5,
                    "confidence": as_float(row.get("roi_confidence")),
                    "mode": row.get("roi_proposal_mode", ""),
                },
            }
        )

    gt_by_label: dict[str, list[float]] = {label: [] for label in labels}
    for row in read_csv(eval_dir / "gt_events.csv"):
        if row.get("label") in gt_by_label:
            gt_by_label[row["label"]].append(as_float(row.get("time_sec")))

    raw_events: list[dict[str, Any]] = []
    for row in read_csv(eval_dir / "predicted_events.csv"):
        label = row.get("label", "")
        if label not in labels:
            continue
        time_sec = as_float(row.get("time_sec"))
        start_sec = as_float(row.get("start_sec"), time_sec - 5)
        end_sec = as_float(row.get("end_sec"), time_sec + 5)
        matching_gt_times = [
            gt_time
            for gt_time in gt_by_label[label]
            if start_sec - match_tolerance_sec <= gt_time <= end_sec + match_tolerance_sec
        ]
        if (video_id, label) in exclusions:
            evaluation_status = "unlabeled"
        else:
            evaluation_status = "matched" if matching_gt_times else "fp"
        raw_events.append(
            {
                "label": label,
                "time_sec": time_sec,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "support_start_sec": as_float(row.get("support_start_sec"), as_float(row.get("start_sec"))),
                "support_end_sec": as_float(row.get("support_end_sec"), as_float(row.get("end_sec"))),
                "score": as_float(row.get("score")),
                "window_indices": parse_json_list(row.get("window_indices")),
                "evaluation_status": evaluation_status,
                "matching_gt_times": matching_gt_times,
            }
        )

    events: list[dict[str, Any]] = []
    for group in merge_candidates(raw_events, merge_sec, labels):
        label = group[0]["label"]
        representative = max(group, key=lambda row: (row["score"], -row["time_sec"]))
        window_indices = sorted(
            set(index for row in group for index in row["window_indices"])
        )
        if not window_indices:
            window_indices = [
                index
                for index, window in windows_by_index.items()
                if as_float(window.get("start_sec")) <= representative["time_sec"] <= as_float(window.get("end_sec"))
            ]
        dino_by_label = {
            item: max(
                [as_float(windows_by_index[index].get(f"prob_{item}")) for index in window_indices if index in windows_by_index]
                or [0.0]
            )
            for item in labels
        }
        frame_by_label = {
            item: max(
                [frame_by_window_label.get((index, item), 0.0) for index in window_indices]
                or [0.0]
            )
            for item in labels
        }
        events.append(
            {
                "id": stable_event_id(video_id, label, group),
                "video_id": video_id,
                "label": label,
                "time_sec": representative["time_sec"],
                "start_sec": min(row["start_sec"] for row in group),
                "end_sec": max(row["end_sec"] for row in group),
                "support_start_sec": min(row["support_start_sec"] for row in group),
                "support_end_sec": max(row["support_end_sec"] for row in group),
                "score": representative["score"],
                "dino_scores": dino_by_label,
                "frame_detection_scores": frame_by_label,
                "window_indices": window_indices,
                "merged_predictions": len(group),
                "evaluation_status": group[0]["evaluation_status"],
                "matching_gt_times": sorted(
                    set(gt_time for row in group for gt_time in row["matching_gt_times"])
                ),
                "match_tolerance_sec": match_tolerance_sec,
            }
        )

    duration = max((row["end_sec"] for row in timeline), default=0.0)
    return {
        "video_id": video_id,
        "video_path": str(video_path),
        "duration_sec": duration,
        "events": events,
        "timeline": timeline,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-ids", nargs="*")
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--review-labels", default=",".join(DEFAULT_REVIEW_LABELS))
    parser.add_argument(
        "--candidate-merge-sec",
        type=float,
        default=5.0,
        help="Merge adjacent same-class predictions to reduce duplicate human review; use 0 to disable.",
    )
    parser.add_argument("--allow-missing-video", action="store_true")
    parser.add_argument("--match-tolerance-sec", type=float)
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="VIDEO_ID:LABEL",
        help="Mark an unlabeled video/class pair so its predictions are not treated as FP.",
    )
    parser.add_argument(
        "--evaluation-filter",
        choices=("all", "fp", "matched", "unlabeled"),
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    video_root = args.video_root.resolve()
    labels = tuple(item.strip() for item in args.labels.split(",") if item.strip())
    review_labels = tuple(item.strip() for item in args.review_labels.split(",") if item.strip())
    exclusions = frozenset(
        tuple(item.rsplit(":", 1)) for item in args.exclude
    )
    run_config_path = run_dir / "run_config.json"
    run_config = json.loads(run_config_path.read_text(encoding="utf-8")) if run_config_path.exists() else {}
    match_tolerance_sec = (
        args.match_tolerance_sec
        if args.match_tolerance_sec is not None
        else float(run_config.get("match_tolerance_sec", 2.0))
    )
    video_ids = get_video_ids(run_dir, args.video_ids)

    videos: list[dict[str, Any]] = []
    missing_videos: list[str] = []
    for video_id in video_ids:
        eval_dir = run_dir / video_id
        if not (eval_dir / "predicted_events.csv").exists():
            continue
        video_path = find_video(video_root, video_id)
        if video_path is None:
            missing_videos.append(video_id)
            if args.allow_missing_video:
                continue
            raise FileNotFoundError(f"No video found for {video_id} under {video_root}")
        video_entry = build_video_entry(
            video_id,
            eval_dir,
            video_path,
            labels,
            args.candidate_merge_sec,
            match_tolerance_sec,
            exclusions,
        )
        if args.evaluation_filter != "all":
            video_entry["events"] = [
                event
                for event in video_entry["events"]
                if event["evaluation_status"] == args.evaluation_filter
            ]
        if video_entry["events"]:
            videos.append(video_entry)

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "labels": labels,
        "review_labels": review_labels,
        "source": {
            "run_dir": str(run_dir),
            "video_root": str(video_root),
            "candidate_merge_sec": args.candidate_merge_sec,
            "match_tolerance_sec": match_tolerance_sec,
            "exclusions": sorted(f"{video_id}:{label}" for video_id, label in exclusions),
            "evaluation_filter": args.evaluation_filter,
        },
        "videos": videos,
        "summary": {
            "num_videos": len(videos),
            "num_events": sum(len(video["events"]) for video in videos),
            "missing_videos": missing_videos,
            "events_by_evaluation": {
                status: sum(
                    event.get("evaluation_status") == status
                    for video in videos
                    for event in video["events"]
                )
                for status in ("fp", "matched", "unlabeled")
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
