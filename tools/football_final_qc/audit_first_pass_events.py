#!/usr/bin/env python3
"""Audit completed first-pass annotations without counting timestamp jitter as misses."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


IGNORED = {"throw_in"}
RAW_LABELS = {"射门": "shot", "扑救": "save", "角球": "corner", "任意球": "free_kick", "中圈开球": "kickoff", "点球": "penalty"}


def timestamp_seconds(value: str) -> float:
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def unique_events(cases: list[dict], key: str) -> dict[str, dict]:
    return {str(event["id"]): event for case in cases for event in case.get(key, [])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--historical-invalid-tolerance-sec", type=float, default=15.0)
    args = parser.parse_args()
    data = json.loads(args.cases.read_text())
    tolerance = float(data["source"].get("match_tolerance_sec", 5.0))
    completed = {vid for vid, row in data["videos"].items() if row.get("first_pass_complete")}
    all_cases = [
        case for case in data.get("cases", []) + data.get("auto_resolved_cases", [])
        if case["video_id"] in completed
    ]
    gt_root = Path(data["source"]["gt_run_dir"])
    gt_by_video: dict[str, list[dict]] = {}
    gt_by_id: dict[str, dict] = {}
    for video_id in completed:
        rows = json.loads((gt_root / video_id / "gt_events.json").read_text())
        normalized = []
        for row in rows:
            event = {
                "id": str(row.get("event_id") or row["id"]),
                "video_id": video_id,
                "semantic_label": str(row.get("semantic_label") or row["label"]),
                "parent_label": str(row["label"]),
                "time_sec": float(row["time_sec"]),
            }
            # The builder has already resolved set-piece subtypes; recover that
            # authoritative representation from the case snapshot where possible.
            normalized.append(event)
            gt_by_id[event["id"]] = event
        gt_by_video[video_id] = normalized
    for case in all_cases:
        for event in case.get("original_gt", []):
            if event["id"] in gt_by_id:
                gt_by_id[event["id"]]["semantic_label"] = event["semantic_label"]

    recommended = unique_events(all_cases, "recommended_events")
    first_pass = unique_events(all_cases, "first_pass_events")
    recommended = {eid: e for eid, e in recommended.items() if e["semantic_label"] not in IGNORED}
    # Exact same-label/same-time outputs from adjacent review segments cannot
    # represent two distinct annotations. Collapse them before quality counts.
    exact_groups: dict[tuple[str, str, float], list[dict]] = defaultdict(list)
    for event in recommended.values():
        exact_groups[(event["video_id"], event["semantic_label"], round(float(event["time_sec"]), 3))].append(event)
    exact_duplicate_events = []
    deduplicated_recommended = {}
    for events in exact_groups.values():
        ranked = sorted(events, key=lambda e: (not bool(e.get("lineage_gt_ids")), bool(e.get("human_added")), str(e["id"])))
        deduplicated_recommended[ranked[0]["id"]] = ranked[0]
        exact_duplicate_events.extend(ranked[1:])
    recommended = deduplicated_recommended

    associations: dict[str, list[dict]] = defaultdict(list)
    missed = []
    for event in recommended.values():
        linked = [
            gt_by_id[gid] for gid in event.get("lineage_gt_ids", [])
            if gid in gt_by_id
        ]
        inferred = event.get("inferred_gt_id")
        if not linked and inferred in gt_by_id:
            linked = [gt_by_id[inferred]]
        if not linked:
            nearby = [
                gt for gt in gt_by_video[event["video_id"]]
                if gt["semantic_label"] == event["semantic_label"]
                and abs(gt["time_sec"] - float(event["time_sec"])) <= tolerance
            ]
            if nearby:
                linked = [min(nearby, key=lambda gt: abs(gt["time_sec"] - float(event["time_sec"])))]
        if linked:
            for gt in linked:
                associations[gt["id"]].append(event)
        else:
            missed.append(event)

    incorrect = []
    for gt in gt_by_id.values():
        events = associations.get(gt["id"], [])
        if not events:
            subtype = "gt_deleted_no_human_event"
        elif any(
            e["semantic_label"] == gt["semantic_label"]
            and abs(float(e["time_sec"]) - gt["time_sec"]) <= tolerance
            for e in events
        ):
            continue
        elif any(e["semantic_label"] == gt["semantic_label"] for e in events):
            subtype = f"wrong_time_over_{tolerance:g}s"
        else:
            subtype = "wrong_label"
        incorrect.append({"subtype": subtype, **gt})

    offset_candidates = []
    recommended_ids = set(recommended)
    for event in first_pass.values():
        if event["semantic_label"] in IGNORED:
            continue
        nearby = [
            gt for gt in gt_by_video[event["video_id"]]
            if gt["semantic_label"] == event["semantic_label"]
            and 3.0 < abs(gt["time_sec"] - float(event["time_sec"])) <= tolerance
        ]
        if nearby:
            gt = min(nearby, key=lambda row: abs(row["time_sec"] - float(event["time_sec"])))
            offset_candidates.append({
                "event_id": event["id"], "video_id": event["video_id"],
                "label": event["semantic_label"], "human_time_sec": event["time_sec"],
                "gt_id": gt["id"], "gt_time_sec": gt["time_sec"],
                "abs_delta_sec": abs(gt["time_sec"] - float(event["time_sec"])),
                "retained_in_recommended": event["id"] in recommended_ids,
            })

    # Some upstream annotations were retained in the source JSON but excluded
    # from gt_events.json because label_correct=false. When a human-approved
    # event of the same class lands nearby, this is a repaired timestamp/state,
    # not a completely missing annotation. Match these one-to-one separately.
    rejected_gt = []
    run_root = Path(data["source"]["gt_run_dir"])
    for video_id in completed:
        summary = json.loads((run_root / video_id / "summary.json").read_text())
        annotation_path = Path(summary["annotation_path"])
        payload = json.loads(annotation_path.read_text())
        rows = payload if isinstance(payload, list) else payload.get("data", [])
        for row in rows:
            if row.get("label_correct") is not False or row.get("label") not in RAW_LABELS:
                continue
            rejected_gt.append({"id": str(row["id"]), "video_id": video_id, "semantic_label": RAW_LABELS[row["label"]], "time_sec": timestamp_seconds(row["timestamp"]), "raw_label": row["label"]})
    historical_candidates = []
    for mi, event in enumerate(missed):
        for gi, gt in enumerate(rejected_gt):
            delta = abs(float(event["time_sec"]) - gt["time_sec"])
            if event["video_id"] == gt["video_id"] and event["semantic_label"] == gt["semantic_label"] and delta <= args.historical_invalid_tolerance_sec:
                historical_candidates.append((delta, mi, gi))
    used_missed, used_rejected, historical_repairs = set(), set(), []
    for delta, mi, gi in sorted(historical_candidates):
        if mi in used_missed or gi in used_rejected:
            continue
        used_missed.add(mi); used_rejected.add(gi)
        historical_repairs.append({"human_event": missed[mi], "historical_gt": rejected_gt[gi], "abs_delta_sec": delta})
    true_missed = [event for index, event in enumerate(missed) if index not in used_missed]

    missed_counts = Counter(e["semantic_label"] for e in true_missed)
    wrong_counts = Counter(e["subtype"] for e in incorrect)
    offset_counts = Counter(e["label"] for e in offset_candidates)
    result = {
        "schema_version": 1,
        "source_cases": str(args.cases.resolve()),
        "source_created_at": data.get("created_at"),
        "scope": "completed first-pass annotations; final adjudication may still change counts",
        "completed_videos": len(completed),
        "completed_video_ids": sorted(completed),
        "qc_tolerance_sec": tolerance,
        "formal_evaluation_tolerance_sec": float(data["source"].get("formal_evaluation_tolerance_sec", 3.0)),
        "original_gt": {"total": len(gt_by_id), "by_label": dict(sorted(Counter(g["semantic_label"] for g in gt_by_id.values()).items()))},
        "missed_annotations": {
            "total": len(true_missed),
            "core_total_excluding_back_pass": sum(v for k, v in missed_counts.items() if k != "back_pass"),
            "by_label": dict(sorted(missed_counts.items())),
            "definition": "deduplicated recommended human event with no valid GT lineage/same-class GT within +/-5s and no same-class label_correct=false historical GT within +/-15s",
            "pre_historical_reconciliation_total": len(missed),
            "historical_rejected_gt_reclassified_as_repairs": {"total": len(historical_repairs), "details": historical_repairs},
            "exact_same_label_time_duplicates_removed": {"total": len(exact_duplicate_events), "events": exact_duplicate_events},
        },
        "incorrect_gt": {"total": len(incorrect), "by_type": dict(sorted(wrong_counts.items()))},
        "timestamp_offset_candidates_excluded_from_missed": {
            "total": len(offset_candidates), "interval_sec": "(3, 5]",
            "by_label": dict(sorted(offset_counts.items())), "details": offset_candidates,
        },
        "details": {"missed_annotations": true_missed, "incorrect_gt": incorrect},
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("completed_videos", "original_gt", "missed_annotations", "incorrect_gt", "timestamp_offset_candidates_excluded_from_missed")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
