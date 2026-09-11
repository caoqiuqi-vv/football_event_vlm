#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_RAW = Path("/home/new_users/qiuqi/code/football_events_raw")
DEFAULT_REPAIR = Path("/home/new_users/qiuqi/code/football_events_human_repair")
DEFAULT_OUTPUT = Path("outputs/football_positive_only/raw_missing_shot_save")

TARGET_LABELS = {"shot", "save"}
LABEL_BY_TEXT = {
    "射门": "shot",
    "其他射门类型": "shot",
    "扑救": "save",
}
LABEL_BY_TYPE = {
    "S0199": "shot",
    "B0199": "shot",
    "S0401": "save",
    "B0401": "save",
}
CHINESE_LABEL = {
    "shot": "射门",
    "save": "扑救",
}
EVENT_TYPE = {
    "shot": "raw_missing_shot",
    "save": "raw_missing_save",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build train-only weak positive annotations for raw shot/save events "
            "that are absent from human_repair annotations."
        )
    )
    parser.add_argument("--raw-annotations", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--repair-annotations", type=Path, default=DEFAULT_REPAIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--train-video-ids",
        type=Path,
        required=True,
        help="Only these video ids are emitted to avoid train/val leakage.",
    )
    parser.add_argument(
        "--match-tolerance-sec",
        type=float,
        default=2.0,
        help="Raw event is considered already repaired if same label exists within this tolerance.",
    )
    parser.add_argument(
        "--dedupe-tolerance-sec",
        type=float,
        default=1.0,
        help="Merge duplicate raw events of the same label within this tolerance.",
    )
    return parser.parse_args()


def parse_time(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    if ":" in text:
        parts = text.split(":")
        try:
            values = [float(part) for part in parts]
        except ValueError:
            return None
        if len(values) == 3:
            return values[0] * 3600.0 + values[1] * 60.0 + values[2]
        if len(values) == 2:
            return values[0] * 60.0 + values[1]
        return None
    try:
        return float(text)
    except ValueError:
        return None


def base_label(item: dict[str, Any]) -> str | None:
    label = LABEL_BY_TEXT.get(str(item.get("label", "")))
    if label is not None:
        return label
    return LABEL_BY_TYPE.get(str(item.get("eventType", item.get("event_type", ""))))


def load_items(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    items = raw.get("data", raw) if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        return []
    output: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("label_correct") is False:
            continue
        label = base_label(item)
        if label not in TARGET_LABELS:
            continue
        start = parse_time(item.get("startTime", item.get("timestamp")))
        end = parse_time(item.get("endTime"))
        if start is None:
            continue
        if end is None or end < start:
            end = start
        anchor = 0.5 * (start + end) if end > start else start
        output.append(
            {
                "label": label,
                "start": float(start),
                "end": float(end),
                "anchor": float(anchor),
                "raw": item,
            }
        )
    output.sort(key=lambda item: (float(item["anchor"]), str(item["label"])))
    return output


def dedupe(items: list[dict[str, Any]], tolerance: float) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for item in items:
        label = str(item["label"])
        anchor = float(item["anchor"])
        if any(
            str(existing["label"]) == label
            and abs(float(existing["anchor"]) - anchor) <= tolerance
            for existing in kept
        ):
            continue
        kept.append(item)
    return kept


def is_matched(
    item: dict[str, Any],
    repaired: list[dict[str, Any]],
    tolerance: float,
) -> bool:
    label = str(item["label"])
    anchor = float(item["anchor"])
    return any(
        str(other["label"]) == label and abs(float(other["anchor"]) - anchor) <= tolerance
        for other in repaired
    )


def read_video_ids(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def convert_item(video_id: str, index: int, item: dict[str, Any]) -> dict[str, Any]:
    label = str(item["label"])
    raw = item["raw"]
    anchor = float(item["anchor"])
    start = float(item["start"])
    end = float(item["end"])
    raw_id = str(raw.get("id", f"raw_{index:04d}_{anchor:.3f}"))
    return {
        "id": f"{video_id}_raw_missing_{index:04d}_{label}_{anchor:.3f}",
        "video_id": video_id,
        "timestamp": anchor,
        "startTime": start,
        "endTime": end,
        "label": CHINESE_LABEL[label],
        "eventType": EVENT_TYPE[label],
        "event_type": EVENT_TYPE[label],
        "sourceEventId": raw_id,
        "positive_only": True,
        "weak_label": True,
        "weak_label_source": "raw_missing_from_human_repair",
    }


def main() -> None:
    args = parse_args()
    train_ids = read_video_ids(args.train_video_ids)
    output_annotations = args.output / "annotations_flat"
    output_annotations.mkdir(parents=True, exist_ok=True)
    for old_file in output_annotations.glob("*.json"):
        old_file.unlink()

    label_counts: Counter[str] = Counter()
    per_video_counts: dict[str, Counter[str]] = {}
    written_video_ids: list[str] = []

    for video_id in sorted(train_ids):
        raw_path = args.raw_annotations / f"{video_id}.json"
        repair_path = args.repair_annotations / f"{video_id}.json"
        raw_items = dedupe(load_items(raw_path), args.dedupe_tolerance_sec)
        repaired_items = load_items(repair_path)
        missing = [
            item
            for item in raw_items
            if not is_matched(item, repaired_items, args.match_tolerance_sec)
        ]
        if not missing:
            continue
        converted = [convert_item(video_id, index, item) for index, item in enumerate(missing)]
        (output_annotations / f"{video_id}.json").write_text(
            json.dumps(converted, ensure_ascii=False, indent=2) + "\n"
        )
        written_video_ids.append(video_id)
        counts = Counter(str(item["label"]) for item in converted)
        per_video_counts[video_id] = counts
        label_counts.update(counts)

    manifest = {
        "raw_annotations": str(args.raw_annotations),
        "repair_annotations": str(args.repair_annotations),
        "train_video_ids": str(args.train_video_ids),
        "output_annotations_dir": str(output_annotations),
        "match_tolerance_sec": float(args.match_tolerance_sec),
        "dedupe_tolerance_sec": float(args.dedupe_tolerance_sec),
        "num_written_videos": len(written_video_ids),
        "written_video_ids": written_video_ids,
        "label_counts_chinese": dict(label_counts),
        "per_video_label_counts_chinese": {
            key: dict(value) for key, value in per_video_counts.items()
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
