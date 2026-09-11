#!/usr/bin/env python3
"""Build a read-only snapshot of original-GT versus committed human decisions."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


GT_LABELS = {"射门": "shot", "扑救": "save", "角球": "corner",
             "任意球": "free_kick", "中圈开球": "kickoff"}
SET_PIECE_TYPES = {"corner", "free_kick", "kickoff"}


def seconds(value: str) -> float:
    parts = [float(x) for x in value.split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]


def semantic(label: str, secondary: list[str]) -> str:
    if label != "set_piece":
        return label
    return next((item for item in secondary if item in SET_PIECE_TYPES), "set_piece")


def match_same(old: list[dict], new: list[dict], tolerance: float) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    candidates = []
    for i, left in enumerate(old):
        for j, right in enumerate(new):
            if left["semantic_label"] == right["semantic_label"]:
                corrected_delta = abs(left["time_sec"] - right["time_sec"])
                source_delta = abs(left["time_sec"] - right.get("source_time_sec", right["time_sec"]))
                lineage = any(abs(left["time_sec"] - float(t)) < 0.01 for t in right.get("matching_gt_times", []))
                if lineage or source_delta <= tolerance or corrected_delta <= tolerance:
                    # Explicit GT lineage wins, then the pre-edit source time;
                    # corrected time is intentionally allowed to move far away.
                    priority = 0 if lineage else (1 if source_delta <= tolerance else 2)
                    candidates.append((priority, min(source_delta, corrected_delta), i, j))
    used_old: set[int] = set()
    used_new: set[int] = set()
    pairs = []
    for _, _, i, j in sorted(candidates):
        if i not in used_old and j not in used_new:
            used_old.add(i); used_new.add(j); pairs.append((i, j))
    return pairs, used_old, used_new


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--match-tolerance-sec", type=float, default=3.0)
    # Keep this aligned with the official event-localization tolerance: small
    # human/AI timestamp offsets inside +/-3s are not annotation errors.
    parser.add_argument("--time-change-sec", type=float, default=3.0)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    video = next(item for item in manifest["videos"] if str(item["video_id"]) == args.video_id)
    original = []
    for index, item in enumerate(json.loads(args.gt.read_text())):
        label = GT_LABELS.get(str(item["label"]), str(item["label"]))
        original.append({"id": str(item.get("id", f"gt-{index}")), "semantic_label": label,
                         "label": label, "time_sec": seconds(str(item["timestamp"])), "raw": item})

    connection = sqlite3.connect(args.db)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """SELECT e.id,e.source_label,e.source_time_sec,e.payload_json,
                  r.status,r.corrected_label,r.corrected_time_sec,r.note,r.reviewer,
                  r.secondary_labels_json,r.event_team,r.goal_side,r.updated_at
           FROM events e JOIN reviews r ON e.id=r.event_id
           WHERE e.video_id=? ORDER BY e.source_time_sec,e.source_label""", (args.video_id,)
    ).fetchall()
    connection.close()

    final = []
    unreviewed_by_segment: dict[str, list[dict]] = {}
    reviewed_segments: set[str] = set()
    all_segments: set[str] = set()
    for row in rows:
        payload = json.loads(row["payload_json"])
        segment_id = str(payload.get("segment_id") or row["id"])
        all_segments.add(segment_id)
        if row["status"] == "unreviewed":
            unreviewed_by_segment.setdefault(segment_id, []).append(payload)
            continue
        reviewed_segments.add(segment_id)
        if row["status"] not in {"accepted", "modified"}:
            continue
        secondary = json.loads(row["secondary_labels_json"] or "[]")
        label = str(row["corrected_label"] or row["source_label"])
        final.append({
            "id": row["id"], "semantic_label": semantic(label, secondary), "label": label,
            "secondary_labels": secondary,
            "time_sec": float(row["corrected_time_sec"] if row["corrected_time_sec"] is not None else row["source_time_sec"]),
            "source_time_sec": float(row["source_time_sec"]), "segment_id": segment_id,
            "human_added": bool(payload.get("human_added")), "note": row["note"],
            "reviewer": row["reviewer"], "updated_at": row["updated_at"],
            "event_team": row["event_team"], "goal_side": row["goal_side"],
            "review_sources": payload.get("review_sources", []),
            "matching_gt_times": payload.get("matching_gt_times", []),
            "evidence_anchors": payload.get("evidence_anchors", []),
        })

    target_semantics = {
        "shot", "save", "corner", "free_kick", "kickoff", "set_piece"
    }
    comparison_final = [
        item for item in final if item["semantic_label"] in target_semantics
    ]
    pairs, used_old, used_new = match_same(
        original, comparison_final, args.match_tolerance_sec
    )
    differences = []
    for i, j in pairs:
        delta = comparison_final[j]["time_sec"] - original[i]["time_sec"]
        if abs(delta) > args.time_change_sec:
            differences.append({"kind": "time_changed", "time_sec": comparison_final[j]["time_sec"],
                                "original": original[i], "human": comparison_final[j], "delta_sec": delta})

    # Pair still-unmatched nearby events of different semantics as explicit mislabels.
    cross = []
    for i, left in enumerate(original):
        if i in used_old: continue
        for j, right in enumerate(comparison_final):
            if j in used_new: continue
            delta = min(abs(left["time_sec"] - right["time_sec"]),
                        abs(left["time_sec"] - right.get("source_time_sec", right["time_sec"])))
            if delta <= args.match_tolerance_sec:
                cross.append((delta, i, j))
    for _, i, j in sorted(cross):
        if i in used_old or j in used_new: continue
        used_old.add(i); used_new.add(j)
        differences.append({"kind": "label_changed", "time_sec": comparison_final[j]["time_sec"],
                            "original": original[i], "human": comparison_final[j],
                            "delta_sec": comparison_final[j]["time_sec"] - original[i]["time_sec"]})

    for i, item in enumerate(original):
        if i not in used_old:
            differences.append({"kind": "gt_removed", "time_sec": item["time_sec"],
                                "original": item, "human": None})
    for j, item in enumerate(comparison_final):
        if j not in used_new:
            differences.append({"kind": "gt_missing", "time_sec": item["time_sec"],
                                "original": None, "human": item})

    for segment_id, events in unreviewed_by_segment.items():
        start = min(float(e.get("start_sec", e.get("time_sec", 0))) for e in events)
        end = max(float(e.get("end_sec", e.get("time_sec", 0))) for e in events)
        differences.append({"kind": "review_gap", "time_sec": (start + end) / 2,
                            "original": None, "human": None, "segment_id": segment_id,
                            "start_sec": start, "end_sec": end,
                            "candidate_labels": sorted({str(e.get("label")) for e in events})})

    for index, item in enumerate(sorted(differences, key=lambda x: (x["time_sec"], x["kind"]))):
        item["id"] = f"diff-{index:04d}"
        item.setdefault("start_sec", max(0.0, item["time_sec"] - 7.0))
        item.setdefault("end_sec", min(float(video["duration_sec"]), item["time_sec"] + 7.0))
        item["nearby_original"] = [
            {"id": event["id"], "semantic_label": event["semantic_label"],
             "time_sec": event["time_sec"]}
            for event in original
            if item["start_sec"] <= event["time_sec"] <= item["end_sec"]
        ]
        item["nearby_human"] = [
            {"id": event["id"], "semantic_label": event["semantic_label"],
             "time_sec": event["time_sec"]}
            for event in comparison_final
            if item["start_sec"] <= event["time_sec"] <= item["end_sec"]
        ]

    counts = Counter(item["kind"] for item in differences)
    snapshot = {
        "schema_version": "gt_human_discrepancy_v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "video_id": args.video_id, "video_path": video["video_path"], "duration_sec": video["duration_sec"],
        "source_manifest": str(args.manifest.resolve()), "source_db": str(args.db.resolve()),
        "source_gt": str(args.gt.resolve()), "match_tolerance_sec": args.match_tolerance_sec,
        "summary": {"original_gt_events": len(original), "human_final_events": len(comparison_final),
                    "all_review_segments": len(all_segments), "committed_review_segments": len(reviewed_segments),
                    "unreviewed_segments": len(unreviewed_by_segment), "differences": len(differences),
                    "by_kind": dict(sorted(counts.items()))},
        "differences": sorted(differences, key=lambda x: (x["time_sec"], x["kind"])),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(snapshot["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
