#!/usr/bin/env python3
"""Add explicit before/after audit details to provenance-enriched final labels."""

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


LABEL_ZH = {
    "shot": "射门", "save": "扑救", "set_piece": "定位球",
    "free_kick": "任意球", "corner": "角球", "kickoff": "开球",
    "back_pass": "回传", "throw_in": "界外球",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def format_time(seconds: float) -> str:
    milliseconds = int(round(max(0.0, seconds) * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{millis:03d}"


def event_key(event: dict[str, Any]) -> tuple[float, str, str]:
    return (
        float(event["time_sec"]),
        str(event.get("semantic_label") or event.get("label") or ""),
        str(event.get("source_id") or event.get("id") or ""),
    )


def build_gt_audit(
    event: dict[str, Any], original_gt: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    lineage = [str(item) for item in event.get("lineage_gt_ids") or [] if item]
    if not lineage:
        return None
    missing = sorted(set(lineage) - set(original_gt))
    if missing:
        raise ValueError(f"missing original GT records: {missing}")
    after_label = str(event.get("semantic_label") or event.get("label") or "")
    after_time = float(event["time_sec"])
    comparisons = []
    for gt_id in lineage:
        original = original_gt[gt_id]
        before_label = str(original.get("semantic_label") or original.get("parent_label") or "")
        before_time = float(original["time_sec"])
        delta = after_time - before_time
        label_changed = before_label != after_label
        time_changed = abs(delta) > 0.001
        changed_fields = []
        if label_changed:
            changed_fields.append("semantic_label")
        if time_changed:
            changed_fields.append("time_sec")
        summary = []
        if label_changed:
            summary.append(
                f"类别 {original.get('raw_label') or LABEL_ZH.get(before_label, before_label)}"
                f" → {LABEL_ZH.get(after_label, after_label)}"
            )
        if time_changed:
            summary.append(
                f"时间 {format_time(before_time)} → {format_time(after_time)}（{delta:+.3f} 秒）"
            )
        comparisons.append(
            {
                "gt_id": gt_id,
                "changed": bool(changed_fields),
                "changed_fields": changed_fields,
                "before": {
                    "semantic_label": before_label,
                    "semantic_label_zh": str(original.get("raw_label") or LABEL_ZH.get(before_label, before_label)),
                    "time_sec": before_time,
                    "time_hms": format_time(before_time),
                },
                "after": {
                    "semantic_label": after_label,
                    "semantic_label_zh": LABEL_ZH.get(after_label, after_label),
                    "time_sec": after_time,
                    "time_hms": format_time(after_time),
                },
                "time_delta_sec": round(delta, 6),
                "summary_zh": "；".join(summary) if summary else "类别与时间未发生实质变化",
            }
        )
    return {
        "changed": any(item["changed"] for item in comparisons),
        "comparisons": comparisons,
        "summary_zh": "；".join(item["summary_zh"] for item in comparisons),
    }


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
    audit_counts: Counter[str] = Counter()

    try:
        for source_row in sorted(source_manifest["files"], key=lambda row: str(row["video_id"])):
            video_id = str(source_row["video_id"])
            source_path = args.input_dir / str(source_row["file"])
            payload = load_json(source_path)
            events = sorted(list(payload["events"]), key=event_key)
            for index, event in enumerate(events, start=1):
                if int(event.get("timeline_index", index)) != index:
                    raise ValueError(f"{video_id}: invalid source timeline_index")
                audit = build_gt_audit(event, original_gt)
                event["timeline_index"] = index
                event["original_gt_audit"] = audit
                if event.get("event_origin") == "original_gt":
                    if audit is None:
                        raise ValueError(f"{video_id}: original_gt event has no lineage")
                    expected = "original_gt_revised" if audit["changed"] else "original_gt_retained"
                    if event.get("origin_detail") != expected:
                        raise ValueError(f"{video_id}: inconsistent GT revision classification")
                    audit_counts[expected] += 1
                csv_rows.append(
                    {
                        "video_id": video_id,
                        "timeline_index": index,
                        "time_sec": f"{float(event['time_sec']):.3f}",
                        "time_hms": format_time(float(event["time_sec"])),
                        "semantic_label": event.get("semantic_label") or event.get("label") or "",
                        "event_origin": event.get("event_origin") or "",
                        "event_origin_zh": event.get("event_origin_zh") or "",
                        "origin_detail": event.get("origin_detail") or "",
                        "revision_summary_zh": audit["summary_zh"] if audit else "",
                        "lineage_gt_ids": "|".join(str(item) for item in event.get("lineage_gt_ids") or []),
                        "source_id": event.get("source_id") or event.get("id") or "",
                    }
                )
            output_payload = {
                **{key: value for key, value in payload.items() if key not in {"schema_version", "events"}},
                "schema_version": "football_final_video_labels_v3_provenance_audit",
                "source_provenance_file": str(source_path.resolve()),
                "source_provenance_sha256": sha256(source_path),
                "gt_revision_audit_policy": {
                    "label_change": "最终 semantic_label 与原 GT semantic_label 不同",
                    "time_change": "最终 time_sec 与原 GT time_sec 的绝对差大于 0.001 秒",
                    "retained": "类别一致且时间差不超过 0.001 秒",
                },
                "events": events,
            }
            output_path = temporary / f"{video_id}.json"
            write_json(output_path, output_payload)
            manifest_rows.append(
                {
                    **{key: value for key, value in source_row.items() if key != "sha256"},
                    "sha256": sha256(output_path),
                }
            )

        csv_path = temporary / "events_timeline.csv"
        with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
        manifest = {
            **{key: value for key, value in source_manifest.items() if key not in {"schema_version", "created_at", "files", "timeline_csv"}},
            "schema_version": "football_final_per_video_manifest_v3_provenance_audit",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_dir": str(args.input_dir.resolve()),
            "source_manifest_sha256": sha256(args.input_dir / "manifest.json"),
            "original_gt_audit_counts": dict(sorted(audit_counts.items())),
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
        "original_gt_audit_counts": dict(sorted(audit_counts.items())),
        "sealed": args.seal,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
