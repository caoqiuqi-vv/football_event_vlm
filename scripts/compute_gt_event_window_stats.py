#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2

ROOTS = [
    {
        "source": "may_xbotgo",
        "videos_dir": "/mnt/data/Datasets/Datasets/Football/may/Xbotgo/videos",
        "annotations_dir": "/mnt/data/Datasets/Datasets/Football/may/Xbotgo/annotations",
    },
    {
        "source": "xbotgo_0608",
        "videos_dir": "/mnt/data/Datasets/Datasets/Football/xbotgo_football_data_0608/videos",
        "annotations_dir": "/mnt/data/Datasets/Datasets/Football/xbotgo_football_data_0608/annotations",
    },
]
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".m4v", ".MP4", ".MOV", ".MKV", ".AVI", ".M4V")
WHISTLE_LABELS = {"裁判鸣哨"}
WHISTLE_TYPES = {"S1005"}
URL_FILES = ["football_human_verion2_0608.json", "metabase_info.json"]


def as_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def canonical_label(item: dict[str, Any]) -> str:
    raw = str(item.get("label", "")).strip()
    etype = str(item.get("eventType", "")).strip()
    if raw in WHISTLE_LABELS or etype in WHISTLE_TYPES:
        return "裁判鸣哨"
    if "射门" in raw or etype in {"S0199", "B0199"}:
        return "射门"
    if "扑救" in raw or etype in {"S0401", "B0401"}:
        return "扑救"
    if "角球" in raw or etype in {"S0201", "B0201"}:
        return "角球"
    if "任意球" in raw or etype in {"S0202", "B0202"}:
        return "任意球"
    if "点球" in raw or etype in {"S0101", "B0101"}:
        return "点球"
    if "中圈开球" in raw or etype == "S1004":
        return "中圈开球"
    if "犯规" in raw or etype in {"S0399", "B0399"}:
        return "犯规"
    if "传球" in raw or etype in {"S0299", "B0299"}:
        return "传球"
    if "拦截" in raw or etype in {"S0402", "B0402"}:
        return "拦截"
    if "抢断" in raw or etype in {"S0403", "B0403"}:
        return "抢断"
    if "解围" in raw or etype in {"S0404", "B0404"}:
        return "解围"
    if "对抗成功" in raw or etype == "S0405":
        return "对抗成功"
    if "盘带" in raw or etype == "S0901":
        return "盘带"
    if "助攻" in raw or etype == "S0203":
        return "助攻"
    if "过人" in raw:
        return "过人"
    if "越位" in raw or etype == "S0301":
        return "越位"
    if "黄牌" in raw or etype == "S1003":
        return "黄牌"
    if "红牌" in raw or etype == "S1002":
        return "红牌"
    if "乌龙" in raw or etype == "S1001":
        return "乌龙球"
    return raw or etype or "未知"


def find_video(videos_dir: Path, video_id: str) -> Path | None:
    for ext in VIDEO_EXTS:
        p = videos_dir / f"{video_id}{ext}"
        if p.exists():
            return p
    matches = sorted(p for p in videos_dir.glob(f"{video_id}*") if p.suffix in VIDEO_EXTS)
    return matches[0] if matches else None


def video_duration(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0.0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    cap.release()
    if fps <= 0 or frames <= 0:
        return 0.0
    return frames / fps




def load_metadata_durations(base_dir: Path) -> dict[str, float]:
    durations: dict[str, float] = {}
    for rel in URL_FILES:
        path = base_dir / rel
        if not path.exists():
            continue
        with path.open() as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            continue
        for row in payload:
            if not isinstance(row, dict):
                continue
            video_id = str(row.get("ID", "")).strip()
            if not video_id or video_id in durations:
                continue
            extra_raw = row.get("EXTRA") or "{}"
            try:
                extra = json.loads(extra_raw) if isinstance(extra_raw, str) else extra_raw
            except Exception:
                extra = {}
            duration = extra.get("duration") if isinstance(extra, dict) else None
            try:
                duration = float(duration)
            except Exception:
                continue
            # Metadata duration is stored in milliseconds in both source metadata files.
            if duration > 10000:
                duration = duration / 1000.0
            if duration > 0:
                durations[video_id] = duration
    return durations


def load_events(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        payload = json.load(f)
    items = payload.get("data", payload) if isinstance(payload, dict) else payload
    return [x for x in items if isinstance(x, dict)]


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))
    return merged


def interval_sum(intervals: list[tuple[float, float]]) -> float:
    return sum(max(0.0, e - s) for s, e in intervals)


def add_stat(stats: dict[str, Any], source: str, video_id: str, label: str, interval: tuple[float, float]) -> None:
    key = (source, video_id, label)
    stats["intervals_by_video_label"][key].append(interval)
    stats["naive_by_label"][label] += interval[1] - interval[0]
    stats["count_by_label"][label] += 1
    skey = (source, label)
    stats["source_naive_by_label"][skey] += interval[1] - interval[0]
    stats["source_count_by_label"][skey] += 1
    stats["source_intervals_by_video_label"][(source, video_id, label)].append(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute GT event 10s-window duration ratios excluding whistles.")
    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gt_event_window_stats"))
    parser.add_argument("--base-dir", type=Path, default=Path("."), help="Directory containing football_human_verion2_0608.json and metabase_info.json for duration fallback.")
    args = parser.parse_args()

    half = args.window_sec / 2.0
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    metadata_durations = load_metadata_durations(args.base_dir)

    stats: dict[str, Any] = {
        "intervals_by_video_label": defaultdict(list),
        "source_intervals_by_video_label": defaultdict(list),
        "naive_by_label": defaultdict(float),
        "count_by_label": defaultdict(int),
        "source_naive_by_label": defaultdict(float),
        "source_count_by_label": defaultdict(int),
    }
    all_non_whistle_by_video: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    all_non_whistle_by_source_video: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    source_total_duration: dict[str, float] = defaultdict(float)
    video_rows = []
    total_duration = 0.0
    total_raw_events = 0
    total_non_whistle_events = 0
    skipped_videos = []

    for root in ROOTS:
        source = root["source"]
        videos_dir = Path(root["videos_dir"])
        annotations_dir = Path(root["annotations_dir"])
        for ann in sorted(annotations_dir.glob("*.json")):
            if ann.name.startswith(".") or ".swp" in ann.name:
                continue
            video_id = ann.stem
            video = find_video(videos_dir, video_id)
            duration_source = "video_file"
            if video is None:
                duration = metadata_durations.get(video_id, 0.0)
                duration_source = "metadata_fallback"
            else:
                duration = video_duration(video)
                if duration <= 0:
                    duration = metadata_durations.get(video_id, 0.0)
                    duration_source = "metadata_fallback"
            if duration <= 0:
                skipped_videos.append({"source": source, "video_id": video_id, "reason": "missing_video_and_metadata_duration"})
                continue
            total_duration += duration
            source_total_duration[source] += duration
            raw_events = load_events(ann)
            total_raw_events += len(raw_events)
            kept_count = 0
            for item in raw_events:
                label = canonical_label(item)
                if label == "裁判鸣哨":
                    continue
                start = as_float(item.get("startTime"), 0.0)
                end = as_float(item.get("endTime"), start)
                if end < start:
                    end = start
                anchor = (start + end) * 0.5 if end > start else start
                win_start = max(0.0, anchor - half)
                win_end = min(duration, anchor + half)
                if win_end <= win_start:
                    continue
                interval = (win_start, win_end)
                add_stat(stats, source, video_id, label, interval)
                all_non_whistle_by_video[(source, video_id)].append(interval)
                all_non_whistle_by_source_video[(source, video_id)].append(interval)
                kept_count += 1
                total_non_whistle_events += 1
            video_rows.append({
                "source": source,
                "video_id": video_id,
                "duration_sec": duration,
                "duration_source": duration_source,
                "raw_events": len(raw_events),
                "non_whistle_events": kept_count,
            })

    # Overall label union duration: merge per-video, then sum across videos.
    label_union: dict[str, float] = defaultdict(float)
    source_label_union: dict[tuple[str, str], float] = defaultdict(float)
    for (source, video_id, label), intervals in stats["intervals_by_video_label"].items():
        union_sec = interval_sum(merge_intervals(intervals))
        label_union[label] += union_sec
        source_label_union[(source, label)] += union_sec

    all_non_whistle_union = sum(interval_sum(merge_intervals(v)) for v in all_non_whistle_by_video.values())
    source_all_union: dict[str, float] = defaultdict(float)
    for (source, video_id), intervals in all_non_whistle_by_source_video.items():
        source_all_union[source] += interval_sum(merge_intervals(intervals))

    duration_by_video = {(row["source"], row["video_id"]): float(row["duration_sec"]) for row in video_rows}
    label_present_video_duration: dict[str, float] = defaultdict(float)
    source_label_present_video_duration: dict[tuple[str, str], float] = defaultdict(float)
    for source, video_id, label in stats["intervals_by_video_label"]:
        duration = duration_by_video.get((source, video_id), 0.0)
        label_present_video_duration[label] += duration
        source_label_present_video_duration[(source, label)] += duration

    label_rows = []
    for label in sorted(stats["count_by_label"], key=lambda x: (-label_union[x], x)):
        union_sec = label_union[label]
        naive_sec = stats["naive_by_label"][label]
        present_duration = label_present_video_duration[label]
        label_rows.append({
            "label": label,
            "event_count": stats["count_by_label"][label],
            "present_video_count": sum(1 for source, video_id, cur_label in stats["intervals_by_video_label"] if cur_label == label),
            "label_present_video_duration_sec": round(present_duration, 3),
            "union_window_sec": round(union_sec, 3),
            "union_percent_of_present_video_duration": round(union_sec / present_duration * 100.0, 6) if present_duration else 0.0,
            "union_percent_of_total_video": round(union_sec / total_duration * 100.0, 6),
            "naive_window_sec": round(naive_sec, 3),
            "naive_percent_of_present_video_duration": round(naive_sec / present_duration * 100.0, 6) if present_duration else 0.0,
            "naive_percent_of_total_video": round(naive_sec / total_duration * 100.0, 6),
        })

    source_rows = []
    for source in sorted(source_total_duration):
        source_rows.append({
            "source": source,
            "total_video_duration_sec": round(source_total_duration[source], 3),
            "all_non_whistle_union_sec": round(source_all_union[source], 3),
            "all_non_whistle_union_percent": round(source_all_union[source] / source_total_duration[source] * 100.0, 6),
        })

    source_label_rows = []
    for (source, label), union_sec in sorted(source_label_union.items(), key=lambda x: (x[0][0], -x[1], x[0][1])):
        duration = source_total_duration[source]
        present_duration = source_label_present_video_duration[(source, label)]
        naive_sec = stats["source_naive_by_label"][(source, label)]
        source_label_rows.append({
            "source": source,
            "label": label,
            "event_count": stats["source_count_by_label"][(source, label)],
            "present_video_count": sum(1 for cur_source, video_id, cur_label in stats["intervals_by_video_label"] if cur_source == source and cur_label == label),
            "label_present_video_duration_sec": round(present_duration, 3),
            "union_window_sec": round(union_sec, 3),
            "union_percent_of_present_video_duration": round(union_sec / present_duration * 100.0, 6) if present_duration else 0.0,
            "union_percent_of_source_video": round(union_sec / duration * 100.0, 6),
            "naive_window_sec": round(naive_sec, 3),
            "naive_percent_of_present_video_duration": round(naive_sec / present_duration * 100.0, 6) if present_duration else 0.0,
            "naive_percent_of_source_video": round(naive_sec / duration * 100.0, 6),
        })

    def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            path.write_text("")
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(out / "event_label_window_stats.csv", label_rows)
    write_csv(out / "source_window_stats.csv", source_rows)
    write_csv(out / "source_event_label_window_stats.csv", source_label_rows)
    write_csv(out / "video_event_counts.csv", video_rows)

    summary = {
        "window_sec": args.window_sec,
        "window_definition": "centered at event anchor: midpoint(startTime,endTime), clipped to video boundary",
        "label_percent_definition": "per-label main percent uses only videos where that label appears as denominator; global total-video percent is retained as a secondary field",
        "whistle_excluded": {"labels": sorted(WHISTLE_LABELS), "event_types": sorted(WHISTLE_TYPES)},
        "annotation_files": len(video_rows),
        "total_video_duration_sec": round(total_duration, 3),
        "total_raw_events": total_raw_events,
        "total_non_whistle_events": total_non_whistle_events,
        "all_non_whistle_union_window_sec": round(all_non_whistle_union, 3),
        "all_non_whistle_union_percent_of_total_video": round(all_non_whistle_union / total_duration * 100.0, 6) if total_duration else 0.0,
        "event_label_stats": label_rows,
        "source_stats": source_rows,
        "metadata_duration_fallback_count": sum(1 for row in video_rows if row.get("duration_source") == "metadata_fallback"),
        "skipped_videos": skipped_videos,
        "outputs": {
            "event_label_window_stats_csv": str(out / "event_label_window_stats.csv"),
            "source_window_stats_csv": str(out / "source_window_stats.csv"),
            "source_event_label_window_stats_csv": str(out / "source_event_label_window_stats.csv"),
            "video_event_counts_csv": str(out / "video_event_counts.csv"),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
