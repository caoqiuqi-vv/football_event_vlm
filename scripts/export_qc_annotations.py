#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ANNOTATION_ROOTS = [
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
URL_FILES = [
    "football_human_verion2_0608.json",
    "metabase_info.json",
]
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".m4v", ".MP4", ".MOV", ".MKV", ".AVI", ".M4V")
KEEP_LABELS = {"射门", "扑救", "点球", "角球", "任意球", "中圈开球"}


@dataclass
class VideoUrlInfo:
    video_id: str
    url: str
    file_path: str
    name: str
    width: str
    height: str
    metadata_source: str


def as_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    return float(value)


def fmt_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000.0))
    ms = total_ms % 1000
    total_sec = total_ms // 1000
    s = total_sec % 60
    total_min = total_sec // 60
    m = total_min % 60
    h = total_min // 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def compact_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total_ms = int(round(seconds * 1000.0))
    ms = total_ms % 1000
    total_sec = total_ms // 1000
    return f"{total_sec:06d}.{ms:03d}"


def canonical_label(item: dict[str, Any]) -> str:
    raw_label = str(item.get("label", "")).strip()
    event_type = str(item.get("eventType", "")).strip()
    text = f"{raw_label} {event_type}"

    # Key-event canonicalization. Old and new annotation schemas use both S/B event codes
    # and several Chinese label variants. Keep all events, but normalize obvious synonyms.
    if "射门" in raw_label or event_type in {"S0199", "B0199"}:
        return "射门"
    if "扑救" in raw_label or event_type in {"S0401", "B0401"}:
        return "扑救"
    if "角球" in raw_label or event_type in {"S0201", "B0201"}:
        return "角球"
    if "任意球" in raw_label or event_type in {"S0202", "B0202"}:
        return "任意球"
    if "点球" in raw_label or event_type in {"S0101", "B0101"}:
        return "点球"

    # Non-target labels are normalized only when the synonym is explicit.
    if "犯规" in raw_label or event_type in {"S0399", "B0399"}:
        return "犯规"
    if "传球" in raw_label or event_type in {"S0299", "B0299"}:
        return "传球"
    if "裁判鸣哨" in raw_label or event_type == "S1005":
        return "裁判鸣哨"
    if "黄牌" in raw_label or event_type == "S1003":
        return "黄牌"
    if "红牌" in raw_label or event_type == "S1002":
        return "红牌"
    if "中圈开球" in raw_label or event_type == "S1004":
        return "中圈开球"
    if "乌龙" in raw_label or event_type == "S1001":
        return "乌龙球"
    if "越位" in raw_label or event_type == "S0301":
        return "越位"
    if "助攻" in raw_label or event_type == "S0203":
        return "助攻"
    if "盘带" in raw_label or event_type == "S0901":
        return "盘带"
    if "拦截" in raw_label or event_type in {"S0402", "B0402"}:
        return "拦截"
    if "抢断" in raw_label or event_type in {"S0403", "B0403"}:
        return "抢断"
    if "解围" in raw_label or event_type in {"S0404", "B0404"}:
        return "解围"
    if "对抗成功" in raw_label or event_type == "S0405":
        return "对抗成功"
    if "过人" in raw_label or event_type == "S0601":
        return "过人"
    return raw_label or event_type or "未知"


def event_confidence(item: dict[str, Any]) -> float | None:
    extra = item.get("extra") or {}
    for container in (item, extra):
        for key in ("confidence", "conf", "score"):
            if key in container and container[key] not in (None, ""):
                try:
                    return round(float(container[key]), 4)
                except Exception:
                    pass
    return None


def load_json_items(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        payload = json.load(f)
    items = payload.get("data", payload) if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise ValueError(f"Unsupported annotation JSON structure: {path}")
    return [item for item in items if isinstance(item, dict)]


def find_video_path(videos_dir: Path, video_id: str) -> str:
    for ext in VIDEO_EXTS:
        path = videos_dir / f"{video_id}{ext}"
        if path.exists():
            return str(path)
    matches = sorted(p for p in videos_dir.glob(f"{video_id}*") if p.suffix in VIDEO_EXTS)
    return str(matches[0]) if matches else ""


def convert_item(source: str, video_id: str, output_index: int, item: dict[str, Any], video_url: str, local_video_path: str) -> dict[str, Any]:
    start = as_float(item.get("startTime"), 0.0)
    end = as_float(item.get("endTime"), start)
    if end < start:
        end = start
    confidence = event_confidence(item)
    return {
        "id": f"{video_id}_{output_index:05d}_{compact_timestamp(start)}",
        "video_id": video_id,
        "timestamp": fmt_timestamp(start),
        "label": canonical_label(item),
        "foul_type": [],
        "confidence": confidence if confidence is not None else 1.0,
    }


def load_video_urls(base_dir: Path, url_files: list[str]) -> dict[str, VideoUrlInfo]:
    mapping: dict[str, VideoUrlInfo] = {}
    for rel in url_files:
        path = base_dir / rel
        with path.open() as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            raise ValueError(f"Video URL metadata must be a list: {path}")
        for row in payload:
            if not isinstance(row, dict):
                continue
            video_id = str(row.get("ID", "")).strip()
            if not video_id:
                continue
            file_path = str(row.get("FILE_PATH", "") or "")
            video_url = str(row.get("video_url", "") or "")
            url = video_url or file_path
            # Keep the first URL for duplicate IDs, but prefer rows with a non-empty URL.
            if video_id in mapping and mapping[video_id].url:
                continue
            mapping[video_id] = VideoUrlInfo(
                video_id=video_id,
                url=url,
                file_path=file_path,
                name=str(row.get("NAME", "") or ""),
                width=str(row.get("WIDTH", "") or ""),
                height=str(row.get("HEIGHT", "") or ""),
                metadata_source=path.name,
            )
    return mapping


def kept_original_items(original: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in original if canonical_label(item) in KEEP_LABELS]


def validate_conversion(original_kept: list[dict[str, Any]], converted: list[dict[str, Any]], path: Path) -> list[str]:
    errors: list[str] = []
    if len(original_kept) != len(converted):
        errors.append(f"{path}: kept count mismatch original_kept={len(original_kept)} converted={len(converted)}")
        return errors
    for idx, (src, dst) in enumerate(zip(original_kept, converted)):
        start = as_float(src.get("startTime"), 0.0)
        expected_id = f"{Path(path).stem}_{idx:05d}_{compact_timestamp(start)}"
        if dst.get("id") != expected_id:
            errors.append(f"{path}: idx={idx} id mismatch expected={expected_id} dst={dst.get('id')}")
        expected_label = canonical_label(src)
        if dst.get("label") != expected_label:
            errors.append(f"{path}: idx={idx} label mismatch expected={expected_label} dst={dst.get('label')}")
        if dst.get("timestamp") != fmt_timestamp(start):
            errors.append(f"{path}: idx={idx} timestamp mismatch src={fmt_timestamp(start)} dst={dst.get('timestamp')}")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export football annotations to a QC-friendly one-event-per-row JSON format.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/human_qc_annotations"))
    parser.add_argument("--base-dir", type=Path, default=Path("."))
    parser.add_argument("--fail-on-missing-url", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    annotations_out = output_dir / "annotations"
    annotations_out.mkdir(parents=True, exist_ok=True)

    url_map = load_video_urls(args.base_dir, URL_FILES)
    source_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    all_errors: list[str] = []
    total_events = 0
    total_files = 0
    missing_urls: list[str] = []

    for root_cfg in ANNOTATION_ROOTS:
        source = root_cfg["source"]
        videos_dir = Path(root_cfg["videos_dir"])
        annotations_dir = Path(root_cfg["annotations_dir"])
        source_out = annotations_out / source
        source_out.mkdir(parents=True, exist_ok=True)
        for annotation_path in sorted(annotations_dir.glob("*.json")):
            if annotation_path.name.startswith(".") or ".swp" in annotation_path.name:
                continue
            video_id = annotation_path.stem
            original_items = load_json_items(annotation_path)
            info = url_map.get(video_id)
            video_url = info.url if info else ""
            if not video_url:
                missing_urls.append(video_id)
            local_video_path = find_video_path(videos_dir, video_id)
            original_kept = kept_original_items(original_items)
            converted = [convert_item(source, video_id, idx, item, video_url, local_video_path) for idx, item in enumerate(original_kept)]
            errors = validate_conversion(original_kept, converted, annotation_path)
            all_errors.extend(errors)

            out_path = source_out / f"{video_id}.json"
            out_path.write_text(json.dumps(converted, ensure_ascii=False, indent=2), encoding="utf-8")

            total_files += 1
            total_events += len(converted)
            source_rows.append({
                "video_id": video_id,
                "source": source,
                "video_url": video_url,
                "local_video_path": local_video_path,
                "metadata_source": info.metadata_source if info else "",
                "name": info.name if info else "",
                "width": info.width if info else "",
                "height": info.height if info else "",
                "annotation_json": str(out_path),
                "original_annotation_json": str(annotation_path),
                "event_count": len(converted),
                "original_event_count": len(original_items),
            })
            summary_rows.append({
                "source": source,
                "video_id": video_id,
                "original_count": len(original_items),
                "kept_count": len(original_kept),
                "converted_count": len(converted),
                "dropped_count": len(original_items) - len(original_kept),
                "validation_errors": len(errors),
                "first_timestamp": converted[0]["timestamp"] if converted else "",
                "last_timestamp": converted[-1]["timestamp"] if converted else "",
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    video_csv = output_dir / "video_urls.csv"
    with video_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(source_rows[0].keys()))
        writer.writeheader()
        writer.writerows(source_rows)

    summary_csv = output_dir / "conversion_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    manifest = {
        "output_dir": str(output_dir),
        "annotation_files": total_files,
        "events": total_events,
        "kept_labels": sorted(KEEP_LABELS),
        "video_url_rows": len(source_rows),
        "missing_url_count": len(missing_urls),
        "missing_url_video_ids": sorted(missing_urls),
        "validation_error_count": len(all_errors),
        "validation_errors": all_errors[:200],
        "annotation_roots": ANNOTATION_ROOTS,
        "url_files": URL_FILES,
        "label_rule": "Only KEEP_LABELS events are exported; output event JSON contains only id/video_id/timestamp/label/foul_type/confidence; label is canonicalized by eventType/Chinese keyword before filtering.",
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({k: manifest[k] for k in ["annotation_files", "events", "video_url_rows", "missing_url_count", "validation_error_count"]}, ensure_ascii=False, indent=2))
    print(f"annotations: {annotations_out}")
    print(f"video_urls: {video_csv}")
    print(f"summary: {summary_csv}")
    print(f"manifest: {manifest_path}")
    if all_errors:
        raise SystemExit(2)
    if args.fail_on_missing_url and missing_urls:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
