#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("/mnt/data_16t/football/xbotgo_0807_datas")
DEFAULT_OUTPUT = Path("outputs/football_positive_only/xbotgo_0807")

KNOWN_BAD_OR_DUPLICATE_VIDEO_IDS = {
    # OpenCV cannot decode these files reliably in the current dataset snapshot.
    "2074009560226439169",
    "2079371983183650818",
    "2083135796219453441",
    # Exact duplicate annotation/video sequence with 2077959767652347906.
    "2074060516790677506",
}

BAD_EVENT_WINDOWS = {
    # Keep the video, but drop only positive anchors that reproducibly hit
    # corrupted H264 NAL units during event-centered decoding.
    "2074019202637733889": [
        (1416.0, 2.0),
        (1793.0, 2.0),
    ],
}

EVENT_LABELS = {
    "shot": "shot",
    "goal": "shot",
    "save": "save",
    "corner": "corner",
    "freekick": "freekick",
    "freeKick": "freekick",
    "penalty": "penalty",
    "penaltyKick": "penalty",
    "kickoff": "kickoff",
    "kickOff": "kickoff",
}

CHINESE_LABELS = {
    "shot": "射门",
    "save": "扑救",
    "corner": "角球",
    "freekick": "任意球",
    "penalty": "点球",
    "kickoff": "中圈开球",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert xbotgo_0807 partial team annotations into positive-only football-event annotations."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--include-known-problem-videos", action="store_true")
    parser.add_argument("--no-video-validate", action="store_true")
    return parser.parse_args()


def parse_time_to_sec(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    if ":" not in text:
        try:
            return float(text)
        except ValueError:
            return None
    parts = text.split(":")
    try:
        values = [float(part) for part in parts]
    except ValueError:
        return None
    total = 0.0
    for value in values:
        total = total * 60.0 + value
    return total


def sec_to_timestamp(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    whole = int(round(seconds))
    h = whole // 3600
    m = (whole % 3600) // 60
    s = whole % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def video_duration_sec(path: Path) -> float:
    try:
        import cv2
    except Exception:
        return 0.0
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return 0.0
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        if fps <= 0.0 or frames <= 0.0:
            return 0.0
        return frames / fps
    finally:
        cap.release()


def iter_team_events(raw: dict[str, Any]):
    teams = raw.get("teams") or {}
    if not isinstance(teams, dict):
        return
    for team_key, team in teams.items():
        if not isinstance(team, dict):
            continue
        events = team.get("events") or []
        if not isinstance(events, list):
            continue
        for event in events:
            if isinstance(event, dict):
                yield str(team_key), event


def normalize_event(video_id: str, team_key: str, event: dict[str, Any]) -> dict[str, Any] | None:
    raw_type = str(event.get("eventType", "")).strip()
    base_label = EVENT_LABELS.get(raw_type) or EVENT_LABELS.get(raw_type.lower())
    if base_label is None:
        return None
    timestamp = parse_time_to_sec(event.get("eventTimestamp"))
    if timestamp is None:
        return None
    start_time = parse_time_to_sec(event.get("eventStartTime"))
    end_time = parse_time_to_sec(event.get("eventEndTime"))
    if start_time is None:
        start_time = max(timestamp - 0.5, 0.0)
    if end_time is None or end_time < start_time:
        end_time = max(timestamp, start_time)
    event_id = str(event.get("eventId", "")).strip() or f"{team_key}_{raw_type}_{timestamp:.3f}"
    return {
        "id": f"{video_id}_{team_key}_{event_id}_{base_label}_{timestamp:.3f}",
        "label": CHINESE_LABELS[base_label],
        "eventType": raw_type,
        "event_type": raw_type,
        "timestamp": float(timestamp),
        "startTime": float(start_time),
        "endTime": float(end_time),
        "sourceTeam": team_key,
        "sourceEventId": event_id,
        "positive_only": True,
    }
def bad_event_reason(video_id: str, item: dict[str, Any]) -> str | None:
    timestamp = float(item.get("timestamp", -1.0))
    for center, radius in BAD_EVENT_WINDOWS.get(video_id, []):
        if abs(timestamp - float(center)) <= float(radius):
            return f"h264_bad_anchor_{center:.1f}_pm{radius:.1f}s"
    return None


def filter_bad_event_windows(video_id: str, items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for item in items:
        reason = bad_event_reason(video_id, item)
        if reason is None:
            kept.append(item)
        else:
            dropped.append({**item, "drop_reason": reason})
    return kept, dropped


def convert_annotation(path: Path, video_id: str) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        return []
    converted: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for team_key, event in iter_team_events(raw):
        item = normalize_event(video_id, team_key, event)
        if item is None:
            continue
        key = (str(item["label"]), int(round(float(item["timestamp"]) * 10.0)), str(item["sourceTeam"]))
        if key in seen:
            continue
        seen.add(key)
        converted.append(item)
    converted.sort(key=lambda item: (float(item["timestamp"]), str(item["label"]), str(item["sourceTeam"])))
    return converted


def main() -> None:
    args = parse_args()
    root = args.root.expanduser()
    videos_dir = root / "videos"
    annotations_dir = root / "annotations"
    output_dir = args.output
    output_annotations = output_dir / "annotations_flat"
    output_annotations.mkdir(parents=True, exist_ok=True)

    if not videos_dir.is_dir() or not annotations_dir.is_dir():
        raise FileNotFoundError(f"Expected videos/ and annotations/ under {root}")

    skipped: dict[str, str] = {}
    label_counts: Counter[str] = Counter()
    team_counts: Counter[str] = Counter()
    video_counts: dict[str, Counter[str]] = {}
    durations: dict[str, float] = {}
    written_video_ids: list[str] = []
    dropped_events: dict[str, list[dict[str, Any]]] = {}

    for annotation_path in sorted(annotations_dir.glob("*.json")):
        video_id = annotation_path.stem
        if not args.include_known_problem_videos and video_id in KNOWN_BAD_OR_DUPLICATE_VIDEO_IDS:
            skipped[video_id] = "known_bad_or_duplicate"
            continue
        video_path = videos_dir / f"{video_id}.mp4"
        if not video_path.exists():
            skipped[video_id] = "missing_video"
            continue
        duration = 0.0 if args.no_video_validate else video_duration_sec(video_path)
        if not args.no_video_validate and duration <= 0.0:
            skipped[video_id] = "decode_failed"
            continue
        items = convert_annotation(annotation_path, video_id)
        if not items:
            skipped[video_id] = "no_target_events"
            continue
        if duration > 0.0:
            items = [item for item in items if 0.0 <= float(item["timestamp"]) <= duration + 1.0]
        items, dropped = filter_bad_event_windows(video_id, items)
        dropped_events[video_id] = dropped
        if not items:
            skipped[video_id] = "no_events_after_bad_window_filter_or_duration"
            continue
        (output_annotations / f"{video_id}.json").write_text(
            json.dumps(items, ensure_ascii=False, indent=2) + "\n"
        )
        written_video_ids.append(video_id)
        durations[video_id] = duration
        counts = Counter(str(item["label"]) for item in items)
        video_counts[video_id] = counts
        label_counts.update(counts)
        team_counts.update(str(item["sourceTeam"]) for item in items)

    manifest = {
        "source_root": str(root),
        "videos_dir": str(videos_dir),
        "annotations_dir": str(annotations_dir),
        "output_annotations_dir": str(output_annotations),
        "num_written_videos": len(written_video_ids),
        "written_video_ids": written_video_ids,
        "skipped": skipped,
        "label_counts_chinese": dict(label_counts),
        "team_counts": dict(team_counts),
        "durations_sec": durations,
        "per_video_label_counts_chinese": {
            key: dict(value) for key, value in video_counts.items()
        },
        "dropped_events": dropped_events,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
