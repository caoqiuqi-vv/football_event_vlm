#!/usr/bin/env python3
"""Create chronologically sorted per-video labels with explicit event provenance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def event_key(event: dict[str, Any]) -> tuple[float, str, str]:
    return (
        float(event["time_sec"]),
        str(event.get("semantic_label") or event.get("label") or ""),
        str(event.get("source_id") or event.get("id") or ""),
    )


def is_manual_addition(event: dict[str, Any]) -> bool:
    source_id = str(event.get("source_id") or event.get("id") or "")
    return bool(
        event.get("human_added") is True
        or "human_added" in source_id
        or event.get("origin") == "final_manual"
        or source_id.startswith("anchor_")
    )


def classify_event(
    event: dict[str, Any], original_gt: dict[str, dict[str, Any]]
) -> tuple[str, str, str]:
    lineage = [str(item) for item in event.get("lineage_gt_ids") or [] if item]
    if lineage:
        label = str(event.get("semantic_label") or event.get("label") or "")
        time_sec = float(event["time_sec"])
        unchanged = any(
            gt_id in original_gt
            and str(original_gt[gt_id].get("semantic_label") or "") == label
            and abs(float(original_gt[gt_id]["time_sec"]) - time_sec) <= 0.001
            for gt_id in lineage
        )
        detail = "original_gt_retained" if unchanged else "original_gt_revised"
        return "original_gt", "原 GT" if unchanged else "原 GT · 已修正", detail
    if is_manual_addition(event):
        return "human_added", "人工新增", "human_added"
    return "model_added", "模型新增", "model_added"


def format_time(seconds: float) -> str:
    milliseconds = int(round(max(0.0, seconds) * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{millis:03d}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--final-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    source_manifest = load_json(args.input_dir / "manifest.json")
    final_labels = load_json(args.final_labels)
    if final_labels.get("errors") or final_labels.get("warnings"):
        raise ValueError("final-label source contains errors or warnings")
    original_gt = {
        str(gt_id): value["original"]
        for gt_id, value in final_labels.get("gt_dispositions", {}).items()
        if isinstance(value, dict) and isinstance(value.get("original"), dict)
    }

    temporary = args.output_dir.with_name(args.output_dir.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    manifest_rows: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    total_origins: Counter[str] = Counter()
    total_details: Counter[str] = Counter()

    try:
        for source_row in sorted(source_manifest["files"], key=lambda row: str(row["video_id"])):
            video_id = str(source_row["video_id"])
            source_path = args.input_dir / str(source_row["file"])
            payload = load_json(source_path)
            events = sorted(list(payload.get("events", [])), key=event_key)
            origins: Counter[str] = Counter()
            details: Counter[str] = Counter()
            enriched: list[dict[str, Any]] = []
            for index, source_event in enumerate(events, start=1):
                event = dict(source_event)
                origin, origin_zh, detail = classify_event(event, original_gt)
                event.update(
                    {
                        "timeline_index": index,
                        "event_origin": origin,
                        "event_origin_zh": origin_zh,
                        "origin_detail": detail,
                        "is_original_gt": origin == "original_gt",
                        "is_model_added": origin == "model_added",
                    }
                )
                enriched.append(event)
                origins[origin] += 1
                details[detail] += 1
                csv_rows.append(
                    {
                        "video_id": video_id,
                        "timeline_index": index,
                        "time_sec": f"{float(event['time_sec']):.3f}",
                        "time_hms": format_time(float(event["time_sec"])),
                        "semantic_label": event.get("semantic_label") or event.get("label") or "",
                        "event_origin": origin,
                        "event_origin_zh": origin_zh,
                        "origin_detail": detail,
                        "lineage_gt_ids": "|".join(str(item) for item in event.get("lineage_gt_ids") or []),
                        "source_id": event.get("source_id") or event.get("id") or "",
                    }
                )

            if [event_key(item) for item in enriched] != sorted(event_key(item) for item in enriched):
                raise RuntimeError(f"{video_id}: chronological sort verification failed")
            total_origins.update(origins)
            total_details.update(details)
            output_payload = {
                **{key: value for key, value in payload.items() if key not in {"schema_version", "summary", "events"}},
                "schema_version": "football_final_video_labels_v2_provenance",
                "source_per_video_file": str(source_path.resolve()),
                "source_per_video_sha256": sha256(source_path),
                "sort_order": ["time_sec", "semantic_label", "source_id"],
                "provenance_policy": {
                    "original_gt": "lineage_gt_ids 非空；保留或修正自原始 GT",
                    "model_added": "无 GT lineage，来自模型 repair 候选",
                    "human_added": "无 GT lineage，由人工新增或最终人工编辑",
                },
                "summary": {
                    "events": len(enriched),
                    "events_by_label": dict(sorted(Counter(str(item.get("semantic_label") or item.get("label") or "unknown") for item in enriched).items())),
                    "events_by_origin": dict(sorted(origins.items())),
                    "events_by_origin_detail": dict(sorted(details.items())),
                },
                "events": enriched,
            }
            output_path = temporary / f"{video_id}.json"
            write_json(output_path, output_payload)
            manifest_rows.append(
                {
                    "video_id": video_id,
                    "file": output_path.name,
                    "events": len(enriched),
                    "events_by_origin": dict(sorted(origins.items())),
                    "sha256": sha256(output_path),
                }
            )

        csv_path = temporary / "events_timeline.csv"
        with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
        manifest = {
            "schema_version": "football_final_per_video_manifest_v2_provenance",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_dir": str(args.input_dir.resolve()),
            "source_manifest_sha256": sha256(args.input_dir / "manifest.json"),
            "source_final_labels": str(args.final_labels.resolve()),
            "source_final_labels_sha256": sha256(args.final_labels),
            "videos": len(manifest_rows),
            "events": len(csv_rows),
            "events_by_origin": dict(sorted(total_origins.items())),
            "events_by_origin_detail": dict(sorted(total_details.items())),
            "sort_order": ["time_sec", "semantic_label", "source_id"],
            "timeline_csv": {"file": csv_path.name, "sha256": sha256(csv_path)},
            "files": manifest_rows,
        }
        write_json(temporary / "manifest.json", manifest)
        temporary.replace(args.output_dir)
        if args.seal:
            for path in args.output_dir.iterdir():
                os.chmod(path, 0o444)
            os.chmod(args.output_dir, 0o555)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "videos": len(manifest_rows),
        "events": len(csv_rows),
        "events_by_origin": dict(sorted(total_origins.items())),
        "events_by_origin_detail": dict(sorted(total_details.items())),
        "sealed": args.seal,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
