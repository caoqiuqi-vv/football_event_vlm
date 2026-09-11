#!/usr/bin/env python3
"""Audit grid-aligned, lineage-free repaired shots against original shot GT."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def one_to_one(rows: list[dict], tolerance: float) -> set[tuple[int, str]]:
    candidates = []
    for index, row in enumerate(rows):
        for gt in row["all_original_gt"]:
            delta = abs(row["time_sec"] - gt["time_sec"])
            if delta <= tolerance:
                candidates.append((delta, index, gt["id"]))
    used_rows: set[int] = set()
    used_gt: set[str] = set()
    matched: set[tuple[int, str]] = set()
    for _delta, index, gt_id in sorted(candidates):
        if index in used_rows or gt_id in used_gt:
            continue
        used_rows.add(index)
        used_gt.add(gt_id)
        matched.add((index, gt_id))
    return matched


def bucket(delta: float) -> str:
    if delta <= 3:
        return "<=3s"
    if delta <= 5:
        return "(3,5]s"
    if delta <= 10:
        return "(5,10]s"
    if delta <= 15:
        return "(10,15]s"
    if delta <= 30:
        return "(15,30]s"
    if delta <= 60:
        return "(30,60]s"
    return ">60s"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    manifest = json.loads((args.input_dir / "manifest.json").read_text())
    final = json.loads(Path(manifest["source_final_labels"]).read_text())
    disposition_by_gt = final["gt_dispositions"]
    original_by_video: dict[str, list[dict]] = defaultdict(list)
    lineage_final_by_video: dict[str, list[dict]] = defaultdict(list)
    for gt_id, disposition in disposition_by_gt.items():
        original = disposition["original"]
        if original["semantic_label"] != "shot":
            continue
        original_by_video[original["video_id"]].append({
            "id": gt_id,
            "time_sec": float(original["time_sec"]),
            "disposition": disposition["disposition"],
            "case_id": disposition["case_id"],
            "final_times": [float(x["time_sec"]) for x in disposition.get("final_events", [])],
        })

    payloads = []
    for path in sorted(args.input_dir.glob("*.json")):
        if path.name == "manifest.json":
            continue
        payload = json.loads(path.read_text())
        payloads.append(payload)
        video_id = payload["video_id"]
        for event in payload["events"]:
            label = str(event.get("semantic_label") or event.get("label") or "")
            if label == "shot" and event.get("lineage_gt_ids"):
                lineage_final_by_video[video_id].append(event)

    rows = []
    for payload in payloads:
        video_id = payload["video_id"]
        for event in payload["events"]:
            label = str(event.get("semantic_label") or event.get("label") or "")
            time_sec = float(event["time_sec"])
            if label != "shot" or event.get("lineage_gt_ids"):
                continue
            if abs(time_sec / 5.0 - round(time_sec / 5.0)) > 1e-9:
                continue
            originals = sorted(original_by_video[video_id], key=lambda gt: abs(time_sec - gt["time_sec"]))
            nearest = originals[0] if originals else None
            lineage_events = sorted(
                lineage_final_by_video[video_id],
                key=lambda item: abs(time_sec - float(item["time_sec"])),
            )
            nearest_final = lineage_events[0] if lineage_events else None
            rows.append({
                "video_id": video_id,
                "time_sec": time_sec,
                "source_id": event.get("source_id") or event.get("id"),
                "case_id": event.get("case_id"),
                "origin": event.get("origin") or event.get("source") or "unknown",
                "nearest_original_gt_id": nearest["id"] if nearest else "",
                "nearest_original_gt_time_sec": nearest["time_sec"] if nearest else None,
                "signed_delta_to_original_sec": time_sec - nearest["time_sec"] if nearest else None,
                "abs_delta_to_original_sec": abs(time_sec - nearest["time_sec"]) if nearest else None,
                "nearest_original_disposition": nearest["disposition"] if nearest else "",
                "nearest_original_same_case": bool(nearest and nearest["case_id"] == event.get("case_id")),
                "nearest_original_final_times": nearest["final_times"] if nearest else [],
                "nearest_lineage_final_time_sec": float(nearest_final["time_sec"]) if nearest_final else None,
                "abs_delta_to_lineage_final_sec": abs(time_sec - float(nearest_final["time_sec"])) if nearest_final else None,
                "all_original_gt": original_by_video[video_id],
            })

    assert len(rows) == 130, f"expected 130 rows, found {len(rows)}"
    matches = {tolerance: one_to_one(rows, tolerance) for tolerance in (3.0, 5.0, 10.0, 15.0)}
    for index, row in enumerate(rows):
        for tolerance, pairs in matches.items():
            matched_gt = next((gt_id for row_index, gt_id in pairs if row_index == index), "")
            row[f"one_to_one_match_{int(tolerance)}s_gt_id"] = matched_gt
        row["nearest_delta_bucket"] = bucket(float(row["abs_delta_to_original_sec"]))
        row.pop("all_original_gt")

    fields = list(rows[0])
    with (args.output_dir / "grid_shot_vs_original_gt.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "grid_shot_vs_original_gt.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
    )

    nearest_buckets = Counter(row["nearest_delta_bucket"] for row in rows)
    disposition_buckets = Counter(
        (row["nearest_delta_bucket"], row["nearest_original_disposition"]) for row in rows
    )
    summary = {
        "definition": "shot, empty lineage_gt_ids, time_sec exactly divisible by 5",
        "events": len(rows),
        "nearest_original_gt_delta_buckets": dict(sorted(nearest_buckets.items())),
        "nearest_gt_bucket_by_disposition": {
            f"{interval}|{disposition}": count
            for (interval, disposition), count in sorted(disposition_buckets.items())
        },
        "one_to_one_reassociations": {
            f"within_{int(tolerance)}s": len(pairs) for tolerance, pairs in matches.items()
        },
        "same_case_as_nearest_original_gt": sum(row["nearest_original_same_case"] for row in rows),
        "nearest_original_deleted": sum(
            row["nearest_original_disposition"] == "deleted" for row in rows
        ),
        "nearest_lineage_final_within_5s": sum(
            row["abs_delta_to_lineage_final_sec"] is not None
            and row["abs_delta_to_lineage_final_sec"] <= 5
            for row in rows
        ),
    }
    (args.output_dir / "SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
