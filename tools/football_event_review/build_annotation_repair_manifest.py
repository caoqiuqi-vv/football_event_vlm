#!/usr/bin/env python3
"""Build an instance-safe streaming UI manifest from an annotation-repair queue."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PRIMARY_LABELS = ("shot", "save", "set_piece")
MATCH_TOLERANCE_SEC = 3.0
FAMILY_LABELS = {
    "shot_save": {"shot", "save"},
    "set_piece": {"set_piece"},
}
SUBTYPE_MAP = {
    "任意球": "free_kick",
    "点球": "penalty",
    "角球": "corner",
    "中圈开球": "kickoff",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def find_video(root: Path, video_id: str) -> Path | None:
    for suffix in (".mp4", ".mov", ".mkv", ".avi", ".webm"):
        path = root / f"{video_id}{suffix}"
        if path.exists():
            return path.resolve()
    matches = sorted(root.glob(f"**/{video_id}.*"))
    return matches[0].resolve() if matches else None


def stable_id(video_id: str, segment_index: int, label: str, time_sec: float) -> str:
    raw = f"{video_id}|{segment_index}|{label}|{time_sec:.3f}"
    digest = hashlib.sha1(raw.encode()).hexdigest()[:14]
    return f"{video_id}_repair{segment_index:05d}_{label}_{digest}"


def anchor_key(anchor: dict[str, Any]) -> str:
    if anchor.get("source") == "gt":
        return f"gt:{anchor.get('label', '')}"
    return f"candidate:{anchor.get('family', '')}"


def anchor_family(anchor: dict[str, Any]) -> str:
    if anchor.get("source") == "candidate":
        return str(anchor.get("family", ""))
    label = str(anchor.get("label", ""))
    return "shot_save" if label in {"shot", "save"} else label


def source_compatible(anchor: dict[str, Any], existing: list[dict[str, Any]]) -> bool:
    """A model anchor may accompany GT only with direct same-family evidence."""
    source = str(anchor.get("source", "candidate"))
    opposite = [item for item in existing if str(item.get("source", "candidate")) != source]
    family = anchor_family(anchor)
    if not opposite:
        # Keep different model families as separate review tasks. Otherwise an
        # unrelated shot response could hitchhike on a set-piece GT task.
        if source == "candidate":
            return all(anchor_family(item) == family for item in existing)
        return True
    time_sec = float(anchor["time_sec"])
    return any(
        anchor_family(item) == family
        and abs(time_sec - float(item["time_sec"])) <= MATCH_TOLERANCE_SEC
        for item in opposite
    )


def split_instances(row: dict[str, Any], duration_sec: float) -> list[dict[str, Any]]:
    """Group compatible co-temporal evidence while preserving repeated instances."""
    anchors = sorted(row.get("anchors", []), key=lambda item: float(item["time_sec"]))
    if not anchors:
        anchors = [{
            "source": "candidate",
            "family": "unknown",
            "time_sec": (float(row["start_sec"]) + float(row["end_sec"])) / 2.0,
        }]
    groups: list[dict[str, Any]] = []
    for anchor in anchors:
        time_sec = float(anchor["time_sec"])
        key = anchor_key(anchor)
        compatible = [
            group
            for group in groups
            if key not in group["keys"]
            # Overlapping playback context does not imply one event instance.
            # Keep decisions at least as strict as the evaluation tolerance.
            and abs(time_sec - group["center_sec"]) <= MATCH_TOLERANCE_SEC
            and source_compatible(anchor, group["anchors"])
        ]
        if compatible:
            group = min(compatible, key=lambda item: abs(time_sec - item["center_sec"]))
            group["anchors"].append(anchor)
            group["keys"].add(key)
            group["center_sec"] = sum(
                float(item["time_sec"]) for item in group["anchors"]
            ) / len(group["anchors"])
        else:
            groups.append({
                "anchors": [anchor],
                "keys": {key},
                "center_sec": time_sec,
            })

    output = []
    row_labels = set(row.get("labels", []))
    for group in groups:
        labels: set[str] = set()
        times = [float(anchor["time_sec"]) for anchor in group["anchors"]]
        for anchor in group["anchors"]:
            if anchor.get("source") == "gt":
                labels.add(str(anchor["label"]))
            else:
                family = str(anchor.get("family", ""))
                labels.update(row_labels & FAMILY_LABELS.get(family, row_labels))
        labels &= set(PRIMARY_LABELS)
        if not labels:
            labels = row_labels & set(PRIMARY_LABELS)
        support_starts = [
            float(anchor.get("support_start_sec", float(anchor["time_sec"]) - 5.0))
            for anchor in group["anchors"]
        ]
        support_ends = [
            float(anchor.get("support_end_sec", float(anchor["time_sec"]) + 5.0))
            for anchor in group["anchors"]
        ]
        output.append({
            "start_sec": max(0.0, min(support_starts)),
            "end_sec": min(duration_sec, max(support_ends)),
            "labels": labels,
            "anchors": group["anchors"],
            # Source ownership follows this atomic group, not the wider viewing
            # interval. This prevents a nearby candidate from turning a GT task
            # into a candidate task (or vice versa).
            "sources": {str(anchor.get("source", "candidate")) for anchor in group["anchors"]},
            "has_any_gt": any(anchor.get("source") == "gt" for anchor in group["anchors"]),
            "has_same_label_gt": any(anchor.get("source") == "gt" for anchor in group["anchors"]),
        })
    return output


def merge_candidate_proposals_into_gt(
    proposals: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Attach a same-family candidate within 3s to its closest GT task.

    The candidate remains visible as evidence but no longer creates a second
    human decision. Distinct GT proposals are never merged here.
    """
    gt_indexes = [
        index for index, proposal in enumerate(proposals)
        if any(anchor.get("source") == "gt" for anchor in proposal["anchors"])
    ]
    absorbed: set[int] = set()
    for index, proposal in enumerate(proposals):
        if index in gt_indexes or any(anchor.get("source") == "gt" for anchor in proposal["anchors"]):
            continue
        candidates = [anchor for anchor in proposal["anchors"] if anchor.get("source") == "candidate"]
        options: list[tuple[float, int]] = []
        for gt_index in gt_indexes:
            target = proposals[gt_index]
            gt_anchors = [anchor for anchor in target["anchors"] if anchor.get("source") == "gt"]
            for candidate in candidates:
                for gt in gt_anchors:
                    if str(candidate.get("family", "")) != anchor_family(gt):
                        continue
                    delta = abs(float(candidate["time_sec"]) - float(gt["time_sec"]))
                    if delta <= MATCH_TOLERANCE_SEC:
                        options.append((delta, gt_index))
        if not options:
            continue
        _, gt_index = min(options)
        target = proposals[gt_index]
        existing = {
            (str(anchor.get("source", "")), str(anchor.get("family") or anchor.get("label") or ""), round(float(anchor["time_sec"]), 3))
            for anchor in target["anchors"]
        }
        target["anchors"].extend(
            anchor for anchor in proposal["anchors"]
            if (str(anchor.get("source", "")), str(anchor.get("family") or anchor.get("label") or ""), round(float(anchor["time_sec"]), 3)) not in existing
        )
        target["labels"] = set(target["labels"]) | set(proposal["labels"])
        target["sources"] = set(target["sources"]) | set(proposal["sources"])
        target["start_sec"] = min(float(target["start_sec"]), float(proposal["start_sec"]))
        target["end_sec"] = max(float(target["end_sec"]), float(proposal["end_sec"]))
        target.setdefault("absorbed_candidate_proposals", 0)
        target["absorbed_candidate_proposals"] += 1
        absorbed.add(index)
    return [proposal for index, proposal in enumerate(proposals) if index not in absorbed], len(absorbed)

def timeline_rows(eval_dir: Path) -> list[dict[str, Any]]:
    result = []
    for row in read_csv(eval_dir / "window_predictions.csv"):
        result.append({
            "index": int(row["index"]),
            "start_sec": float(row["start_sec"]),
            "end_sec": float(row["end_sec"]),
            "dino": {
                label: float(row.get(f"prob_{label}") or 0.0)
                for label in PRIMARY_LABELS
            },
            "frame_detection": {label: 0.0 for label in PRIMARY_LABELS},
            "roi": {
                "valid": float(row.get("roi_valid") or 0.0) > 0.5,
                "confidence": float(row.get("roi_confidence") or 0.0),
                "mode": row.get("roi_proposal_mode", ""),
            },
        })
    return result


def suggested_details(
    label: str,
    time_sec: float,
    gt_events: list[dict[str, Any]],
) -> list[str]:
    if label != "set_piece":
        return []
    matches = [
        event
        for event in gt_events
        if event.get("label") == label
        and abs(float(event["time_sec"]) - time_sec) <= 0.05
    ]
    if not matches:
        return []
    raw = str(matches[0].get("raw_label") or matches[0].get("event_type") or "")
    subtype = SUBTYPE_MAP.get(raw)
    return [subtype] if subtype else []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-jsonl", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, default=None)
    args = parser.parse_args()

    split_by_video = {}
    path_by_video: dict[str, Path] = {}
    if args.inventory is not None:
        inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
        split_by_video = {
            str(row["video_id"]): str(row["split"])
            for row in inventory.get("videos", [])
        }
        path_by_video = {
            str(row["video_id"]): Path(str(row["video_path"])).resolve()
            for row in inventory.get("videos", [])
            if row.get("video_path")
        }

    queue = read_jsonl(args.queue_jsonl)
    rows_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in queue:
        rows_by_video[str(row["video_id"])].append(row)

    videos = []
    missing_videos = []
    source_segments = 0
    atomic_segments = 0
    gt_suggested = 0
    absorbed_candidate_segments = 0
    for video_id, rows in sorted(rows_by_video.items()):
        inventory_path = path_by_video.get(video_id)
        video_path = (
            inventory_path
            if inventory_path is not None and inventory_path.is_file()
            else find_video(args.video_root, video_id)
        )
        if video_path is None:
            missing_videos.append(video_id)
            continue
        eval_dir = args.run_dir / video_id
        summary = json.loads((eval_dir / "summary.json").read_text(encoding="utf-8"))
        duration_sec = float(summary["duration_sec"])
        timeline = timeline_rows(eval_dir)
        gt_events = json.loads((eval_dir / "gt_events.json").read_text(encoding="utf-8"))

        proposals: list[dict[str, Any]] = []
        for row in rows:
            source_segments += 1
            proposals.extend(split_instances(row, duration_sec))
        proposals, absorbed = merge_candidate_proposals_into_gt(proposals)
        absorbed_candidate_segments += absorbed
        proposals.sort(key=lambda item: (item["start_sec"], item["end_sec"]))
        atomic_segments += len(proposals)

        events = []
        for segment_index, proposal in enumerate(proposals):
            segment_id = f"{video_id}_repair_segment{segment_index:05d}"
            overlapping = [
                item
                for item in timeline
                if item["end_sec"] > proposal["start_sec"]
                and item["start_sec"] < proposal["end_sec"]
            ]
            if not overlapping and timeline:
                center = (proposal["start_sec"] + proposal["end_sec"]) / 2.0
                overlapping = [
                    min(
                        timeline,
                        key=lambda item: abs(
                            (item["start_sec"] + item["end_sec"]) / 2.0 - center
                        ),
                    )
                ]
            scores = {
                label: max((item["dino"][label] for item in overlapping), default=0.0)
                for label in PRIMARY_LABELS
            }
            window_indices = [item["index"] for item in overlapping]
            gt_task_anchors = [
                item for item in proposal["anchors"] if item.get("source") == "gt"
            ]
            candidate_task_anchors = [
                item for item in proposal["anchors"] if item.get("source") == "candidate"
            ]
            gt_labels = sorted({str(item["label"]) for item in gt_task_anchors})
            candidate_labels = sorted(set().union(*(
                set(proposal["labels"]) & FAMILY_LABELS.get(str(item.get("family", "")), set(proposal["labels"]))
                for item in candidate_task_anchors
            ))) if candidate_task_anchors else []
            task_source = "gt" if gt_task_anchors else "candidate"
            for label in sorted(proposal["labels"]):
                gt_anchors = [
                    anchor
                    for anchor in proposal["anchors"]
                    if anchor.get("source") == "gt" and anchor.get("label") == label
                ]
                family = "shot_save" if label in {"shot", "save"} else "set_piece"
                candidate_anchors = [
                    anchor
                    for anchor in proposal["anchors"]
                    if anchor.get("source") == "candidate"
                    and anchor.get("family") == family
                ]
                chosen = (
                    gt_anchors[0]
                    if gt_anchors
                    else candidate_anchors[0]
                    if candidate_anchors
                    else {"time_sec": (proposal["start_sec"] + proposal["end_sec"]) / 2.0}
                )
                time_sec = float(chosen["time_sec"])
                details = suggested_details(label, time_sec, gt_events)
                gt_suggested += int(bool(details))
                matching_gt_times = [
                    float(anchor["time_sec"])
                    for anchor in gt_anchors
                ]
                events.append({
                    "id": stable_id(video_id, segment_index, label, time_sec),
                    "segment_id": segment_id,
                    "segment_index": segment_index,
                    "segment_labels": sorted(proposal["labels"]),
                    "video_id": video_id,
                    "label": label,
                    "time_sec": time_sec,
                    "start_sec": proposal["start_sec"],
                    "end_sec": proposal["end_sec"],
                    "support_start_sec": proposal["start_sec"],
                    "support_end_sec": proposal["end_sec"],
                    "score": scores[label],
                    "dino_scores": scores,
                    "frame_detection_scores": {
                        item: max(
                            (window["frame_detection"][item] for window in overlapping),
                            default=0.0,
                        )
                        for item in PRIMARY_LABELS
                    },
                    "window_indices": window_indices,
                    "merged_predictions": len(window_indices),
                    "evaluation_status": "matched" if matching_gt_times else "fp",
                    "matching_gt_times": matching_gt_times,
                    "match_tolerance_sec": MATCH_TOLERANCE_SEC,
                    "review_protocol": "annotation_repair_gt_anchor_safe_v2",
                    "review_sources": sorted(proposal["sources"]),
                    "task_source": task_source,
                    "gt_labels": gt_labels,
                    "gt_times": sorted(float(item["time_sec"]) for item in gt_task_anchors),
                    "candidate_labels": candidate_labels,
                    "candidate_times": sorted(float(item["time_sec"]) for item in candidate_task_anchors),
                    "suggested_secondary_labels": details,
                    "evidence_anchors": proposal["anchors"],
                })
        videos.append({
            "video_id": video_id,
            "split": split_by_video.get(video_id, ""),
            "video_path": str(video_path),
            "duration_sec": duration_sec,
            "events": events,
            "timeline": timeline,
        })

    manifest = {
        "schema_version": 3,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "labels": list(PRIMARY_LABELS),
        "review_labels": ["shot", "save", "set_piece", "back_pass", "throw_in"],
        "source": {
            "queue_jsonl": str(args.queue_jsonl.resolve()),
            "run_dir": str(args.run_dir.resolve()),
            "video_root": str(args.video_root.resolve()),
            "protocol": "annotation_repair_gt_anchor_safe_v2",
            "definition": "every GT keeps its exact label/time anchor; model evidence never replaces GT; only compatible evidence within 3s may share playback",
        },
        "videos": videos,
        "summary": {
            "num_videos": len(videos),
            "source_review_segments": source_segments,
            "atomic_review_segments": atomic_segments,
            "segment_label_decisions": sum(len(video["events"]) for video in videos),
            "gt_set_piece_subtype_suggestions": gt_suggested,
            "candidate_segments_absorbed_into_gt": absorbed_candidate_segments,
            "missing_videos": missing_videos,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
