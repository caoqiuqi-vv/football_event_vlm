#!/usr/bin/env python
"""Build and measure a recall-first annotation-repair review queue.

This tool deliberately separates event evaluation from human review routing.  It
never removes model predictions.  Dense, overlapping responses are only folded
into review episodes so that an annotator watches a piece of video once and may
create multiple events inside it.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LABELS = ("shot", "save", "set_piece")
FAMILY = {"shot": "shot_save", "save": "shot_save", "set_piece": "set_piece"}


def load_json_relaxed(path: Path) -> Any:
    text = path.read_text(encoding="utf-8").strip()
    # A few historical reports ended with the two literal characters ``\n``.
    if text.endswith("\\n"):
        text = text[:-2].rstrip()
    return json.loads(text)


def logit(value: float) -> float:
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


def adaptive_parameters(report: dict[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = {}
    for label in LABELS:
        folds = report["per_class"][label]["adaptive_median_logit_loov"]["folds"]
        result[label] = {
            str(fold["held_out_video_id"]): {
                "alpha": float(fold["alpha"]),
                "threshold": float(fold["adaptive_threshold"]),
            }
            for fold in folds
        }
    return result


def merge_intervals(rows: list[dict[str, Any]], gap_sec: float = 0.0) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: (item["start_sec"], item["end_sec"])):
        if not merged or row["start_sec"] > merged[-1]["end_sec"] + gap_sec:
            merged.append(
                {
                    "start_sec": float(row["start_sec"]),
                    "end_sec": float(row["end_sec"]),
                    "labels": set(row.get("labels", [])),
                    "sources": set(row.get("sources", [])),
                    "anchors": list(row.get("anchors", [])),
                }
            )
            continue
        merged[-1]["end_sec"] = max(merged[-1]["end_sec"], float(row["end_sec"]))
        merged[-1]["labels"].update(row.get("labels", []))
        merged[-1]["sources"].update(row.get("sources", []))
        merged[-1]["anchors"].extend(row.get("anchors", []))
    return merged


def local_peak_anchors(
    rows: list[dict[str, Any]], *, min_peak_distance_sec: float, rescue_distance_sec: float
) -> list[dict[str, Any]]:
    """Route dense responses to sparse review anchors without deleting events."""
    if not rows:
        return []
    rows = sorted(rows, key=lambda row: row["center_sec"])
    bursts: list[list[dict[str, Any]]] = []
    for row in rows:
        if not bursts or row["index"] > bursts[-1][-1]["index"] + 1:
            bursts.append([row])
        else:
            bursts[-1].append(row)

    selected: list[dict[str, Any]] = []
    for burst in bursts:
        candidates: list[dict[str, Any]] = []
        for index, row in enumerate(burst):
            left = burst[index - 1]["margin"] if index else -math.inf
            right = burst[index + 1]["margin"] if index + 1 < len(burst) else -math.inf
            if row["margin"] >= left and row["margin"] >= right:
                candidates.append(row)
        if not candidates:
            candidates = [max(burst, key=lambda row: row["margin"])]

        kept: list[dict[str, Any]] = []
        for row in sorted(candidates, key=lambda item: item["margin"], reverse=True):
            if all(
                abs(row["center_sec"] - previous["center_sec"]) >= min_peak_distance_sec
                for previous in kept
            ):
                kept.append(row)
        if not kept:
            kept.append(max(burst, key=lambda row: row["margin"]))

        # Very long responses are not assumed to contain only one event.  Add a
        # representative anchor whenever an active window is too far from all
        # selected peaks.  The UI may still add multiple events per episode.
        while True:
            uncovered = [
                row
                for row in burst
                if min(abs(row["center_sec"] - peak["center_sec"]) for peak in kept)
                > rescue_distance_sec
            ]
            if not uncovered:
                break
            rescue = max(uncovered, key=lambda row: row["margin"])
            kept.append(rescue)
        selected.extend(kept)
    return sorted(selected, key=lambda row: row["center_sec"])


def review_segments_for_video(
    windows: list[dict[str, str]],
    video_id: str,
    params: dict[str, dict[str, dict[str, float]]],
    *,
    clip_sec: float,
    min_peak_distance_sec: float,
    rescue_distance_sec: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    locations = {
        label: statistics.median(logit(float(row[f"prob_{label}"])) for row in windows)
        for label in LABELS
    }
    family_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_intervals: list[dict[str, Any]] = []
    raw_by_label: Counter[str] = Counter()
    for row in windows:
        active: dict[str, float] = {}
        for label in LABELS:
            cfg = params[label][video_id]
            margin = logit(float(row[f"prob_{label}"])) - cfg["alpha"] * locations[label] - cfg["threshold"]
            if margin >= 0.0:
                active[label] = margin
                raw_by_label[label] += 1
        if not active:
            continue
        raw_intervals.append(
            {
                "start_sec": float(row["start_sec"]),
                "end_sec": float(row["end_sec"]),
                "labels": set(active),
                "sources": {"candidate"},
                "anchors": [],
            }
        )
        for family in sorted({FAMILY[label] for label in active}):
            family_labels = {label for label in active if FAMILY[label] == family}
            family_rows[family].append(
                {
                    "index": int(row["index"]),
                    "center_sec": (float(row["start_sec"]) + float(row["end_sec"])) / 2.0,
                    "margin": max(active[label] for label in family_labels),
                    "labels": family_labels,
                    "family": family,
                }
            )

    half = clip_sec / 2.0
    routed: list[dict[str, Any]] = []
    for rows in family_rows.values():
        for anchor in local_peak_anchors(
            rows,
            min_peak_distance_sec=min_peak_distance_sec,
            rescue_distance_sec=rescue_distance_sec,
        ):
            routed.append(
                {
                    "start_sec": max(0.0, anchor["center_sec"] - half),
                    "end_sec": anchor["center_sec"] + half,
                    "labels": set(anchor["labels"]),
                    "sources": {"candidate"},
                    "anchors": [
                        {
                            "source": "candidate",
                            "family": anchor["family"],
                            "time_sec": anchor["center_sec"],
                            "margin": anchor["margin"],
                        }
                    ],
                }
            )
    return merge_intervals(routed), merge_intervals(raw_intervals), raw_by_label


def segment_has_gt(segment: dict[str, Any], gt: list[dict[str, Any]], tolerance: float, same_label: bool) -> bool:
    for event in gt:
        if same_label and event["label"] not in segment["labels"]:
            continue
        if segment["start_sec"] - tolerance <= float(event["time_sec"]) <= segment["end_sec"] + tolerance:
            return True
    return False


def gt_coverage(segments: list[dict[str, Any]], gt: list[dict[str, Any]], same_label: bool) -> dict[str, Any]:
    counts = Counter(event["label"] for event in gt)
    covered: Counter[str] = Counter()
    for event in gt:
        time_sec = float(event["time_sec"])
        if any(
            segment["start_sec"] <= time_sec <= segment["end_sec"]
            and (not same_label or event["label"] in segment["labels"])
            for segment in segments
        ):
            covered[event["label"]] += 1
    return {
        label: {
            "gt": counts[label],
            "covered": covered[label],
            "recall": covered[label] / counts[label] if counts[label] else None,
        }
        for label in LABELS
    }


def summarize_segments(segments: list[dict[str, Any]], total_duration: float) -> dict[str, Any]:
    duration = sum(segment["end_sec"] - segment["start_sec"] for segment in segments)
    return {
        "segments": len(segments),
        "duration_sec": duration,
        "duration_ratio": duration / total_duration if total_duration else 0.0,
        "segments_per_video_hour": len(segments) / (total_duration / 3600.0) if total_duration else 0.0,
    }


def review_database_summary(path: Path) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """SELECT e.source_label, e.payload_json, r.status, r.secondary_labels_json
           FROM events e JOIN reviews r ON r.event_id=e.id"""
    ).fetchall()
    connection.close()
    original = [row for row in rows if not json.loads(row["payload_json"]).get("human_added")]
    added = [row for row in rows if json.loads(row["payload_json"]).get("human_added")]
    by_status = Counter(row["status"] for row in original)
    by_label_status: dict[str, Counter[str]] = defaultdict(Counter)
    accepted_subtypes: Counter[str] = Counter()
    for row in original:
        by_label_status[row["source_label"]][row["status"]] += 1
        if row["status"] in {"accepted", "modified"}:
            accepted_subtypes.update(json.loads(row["secondary_labels_json"] or "[]"))
    reviewed = sum(value for key, value in by_status.items() if key != "unreviewed")
    retained = by_status["accepted"] + by_status["modified"]
    return {
        "original_candidates": len(original),
        "original_status": dict(by_status),
        "reviewed_original_candidates": reviewed,
        "retained_original_candidates": retained,
        "old_gt_fp_that_is_real_event_ratio": retained / reviewed if reviewed else None,
        "by_source_label_and_status": {key: dict(value) for key, value in by_label_status.items()},
        "accepted_secondary_labels": dict(accepted_subtypes),
        "human_added_events": len(added),
        "human_added_labels": dict(Counter(row["source_label"] for row in added)),
    }


def subtype_gap_statistics(all_gt: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    gaps: dict[str, list[float]] = defaultdict(list)
    counts: Counter[str] = Counter()
    for events in all_gt.values():
        by_subtype: dict[str, list[float]] = defaultdict(list)
        for event in events:
            if event["label"] != "set_piece":
                continue
            subtype = str(event.get("raw_label") or event.get("event_type") or "set_piece")
            counts[subtype] += 1
            by_subtype[subtype].append(float(event["time_sec"]))
        for subtype, times in by_subtype.items():
            times.sort()
            gaps[subtype].extend(right - left for left, right in zip(times, times[1:]))
    result: dict[str, Any] = {}
    for subtype in sorted(counts):
        values = sorted(gaps[subtype])
        result[subtype] = {
            "events": counts[subtype],
            "adjacent_pairs": len(values),
            "gap_lt_30_sec": sum(value < 30 for value in values),
            "gap_lt_60_sec": sum(value < 60 for value in values),
            "gap_lt_120_sec": sum(value < 120 for value in values),
            "min_gap_sec": values[0] if values else None,
            "median_gap_sec": statistics.median(values) if values else None,
        }
    return result


def serializable(segment: dict[str, Any], video_id: str, gt: list[dict[str, Any]], tolerance: float) -> dict[str, Any]:
    return {
        "video_id": video_id,
        "start_sec": segment["start_sec"],
        "end_sec": segment["end_sec"],
        "duration_sec": segment["end_sec"] - segment["start_sec"],
        "labels": sorted(segment["labels"]),
        "sources": sorted(segment["sources"]),
        "has_any_gt": segment_has_gt(segment, gt, tolerance, same_label=False),
        "has_same_label_gt": segment_has_gt(segment, gt, tolerance, same_label=True),
        "anchors": segment["anchors"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--adaptive-report", type=Path, required=True)
    parser.add_argument("--review-db", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clip-sec", type=float, default=16.0)
    parser.add_argument("--min-peak-distance-sec", type=float, default=12.0)
    parser.add_argument("--long-response-rescue-sec", type=float, default=30.0)
    parser.add_argument("--match-tolerance-sec", type=float, default=3.0)
    args = parser.parse_args()

    config = load_json_relaxed(args.run_dir / "run_config.json")
    video_ids = [str(value) for value in config["video_ids"]]
    params = adaptive_parameters(load_json_relaxed(args.adaptive_report))
    all_gt: dict[str, list[dict[str, Any]]] = {}
    total_duration = 0.0
    candidate_all: list[dict[str, Any]] = []
    candidate_fp_only: list[dict[str, Any]] = []
    candidate_same_label_fp_only: list[dict[str, Any]] = []
    candidate_plus_gt: list[dict[str, Any]] = []
    gt_audit_plus_external_candidates: list[dict[str, Any]] = []
    gt_audit_only: list[dict[str, Any]] = []
    close_set_piece_gt_audit: list[dict[str, Any]] = []
    legacy_union: list[dict[str, Any]] = []
    raw_by_label: Counter[str] = Counter()
    per_video: list[dict[str, Any]] = []
    queue_rows_by_protocol: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for video_id in video_ids:
        video_dir = args.run_dir / video_id
        with (video_dir / "window_predictions.csv").open(newline="") as handle:
            windows = list(csv.DictReader(handle))
        gt = load_json_relaxed(video_dir / "gt_events.json")
        all_gt[video_id] = gt
        duration = float(load_json_relaxed(video_dir / "summary.json")["duration_sec"])
        total_duration += duration
        routed, raw_union, raw_counts = review_segments_for_video(
            windows,
            video_id,
            params,
            clip_sec=args.clip_sec,
            min_peak_distance_sec=args.min_peak_distance_sec,
            rescue_distance_sec=args.long_response_rescue_sec,
        )
        raw_by_label.update(raw_counts)
        fp_only = [
            segment for segment in routed
            if not segment_has_gt(segment, gt, args.match_tolerance_sec, same_label=False)
        ]
        same_label_fp = [
            segment for segment in routed
            if not segment_has_gt(segment, gt, args.match_tolerance_sec, same_label=True)
        ]
        half = args.clip_sec / 2.0
        gt_segments = [
            {
                "start_sec": max(0.0, float(event["time_sec"]) - half),
                "end_sec": min(duration, float(event["time_sec"]) + half),
                "labels": {event["label"]},
                "sources": {"gt"},
                "anchors": [{"source": "gt", "time_sec": float(event["time_sec"]), "label": event["label"]}],
            }
            for event in gt
        ]
        gt_only_merged = merge_intervals(gt_segments)

        close_set_piece_ids: set[str] = set()
        set_piece_by_subtype: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in gt:
            if event["label"] == "set_piece":
                subtype = str(
                    event.get("raw_label") or event.get("event_type") or "set_piece"
                )
                set_piece_by_subtype[subtype].append(event)
        for events in set_piece_by_subtype.values():
            events.sort(key=lambda event: float(event["time_sec"]))
            for left, right in zip(events, events[1:]):
                if float(right["time_sec"]) - float(left["time_sec"]) < 120.0:
                    close_set_piece_ids.add(str(left.get("event_id", id(left))))
                    close_set_piece_ids.add(str(right.get("event_id", id(right))))
        close_gt_segments = merge_intervals(
            [
                segment
                for segment, event in zip(gt_segments, gt)
                if str(event.get("event_id", id(event))) in close_set_piece_ids
            ]
        )
        combined = merge_intervals(routed + gt_segments)
        repair_combined = merge_intervals(fp_only + gt_segments)

        candidate_all.extend({"video_id": video_id, **segment} for segment in routed)
        candidate_fp_only.extend({"video_id": video_id, **segment} for segment in fp_only)
        candidate_same_label_fp_only.extend({"video_id": video_id, **segment} for segment in same_label_fp)
        candidate_plus_gt.extend({"video_id": video_id, **segment} for segment in combined)
        gt_audit_plus_external_candidates.extend(
            {"video_id": video_id, **segment} for segment in repair_combined
        )
        gt_audit_only.extend(
            {"video_id": video_id, **segment} for segment in gt_only_merged
        )
        close_set_piece_gt_audit.extend(
            {"video_id": video_id, **segment} for segment in close_gt_segments
        )
        legacy_union.extend({"video_id": video_id, **segment} for segment in raw_union)
        for protocol, protocol_segments in {
            "candidate_fp_only": fp_only,
            "candidate_online": routed,
            "candidate_plus_all_gt": combined,
            "gt_audit_plus_gt_external_candidates": repair_combined,
            "gt_audit_only": gt_only_merged,
            "close_set_piece_gt_under_120s": close_gt_segments,
        }.items():
            queue_rows_by_protocol[protocol].extend(
                serializable(segment, video_id, gt, args.match_tolerance_sec)
                for segment in protocol_segments
            )
        per_video.append(
            {
                "video_id": video_id,
                "duration_sec": duration,
                "candidate_online": summarize_segments(routed, duration),
                "candidate_fp_only": summarize_segments(fp_only, duration),
                "candidate_plus_all_gt": summarize_segments(combined, duration),
                "gt_audit_only": summarize_segments(gt_only_merged, duration),
                "close_set_piece_gt_under_120s": summarize_segments(
                    close_gt_segments, duration
                ),
                "gt_audit_plus_gt_external_candidates": summarize_segments(
                    repair_combined, duration
                ),
                "same_label_gt_coverage": gt_coverage(routed, gt, same_label=True),
                "any_candidate_clip_gt_coverage": gt_coverage(routed, gt, same_label=False),
            }
        )

    def summarize_global(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return summarize_segments(rows, total_duration)

    flattened_gt = [event for events in all_gt.values() for event in events]
    report = {
        "protocol": "adaptive_loov_response_peak_review_routing_v1",
        "semantics": {
            "candidate_fp_only": "model-routed clips with no old-GT event of any class; old GT is not assumed complete",
            "candidate_same_label_fp_only": "model-routed clips without an old-GT event of the predicted class",
            "candidate_online": "all model-routed clips, including clips that hit old GT; deployable online workload",
            "candidate_plus_all_gt": "union of all model-routed clips and one audit clip around every old GT; annotation-repair workload",
            "gt_audit_only": "one audit clip around every old GT, merged only when clips overlap",
            "close_set_piece_gt_under_120s": "audit clips for both endpoints of adjacent same-subtype set-piece GT pairs less than 120 seconds apart; a risk queue, not evidence that both labels are correct",
            "gt_audit_plus_gt_external_candidates": "one audit clip around every old GT plus only model clips with no old-GT event of any class; GT-near model responses are reviewed once at the GT anchor",
            "important": "routing merges viewing intervals only; it does not NMS or delete event predictions",
        },
        "configuration": {
            "clip_sec": args.clip_sec,
            "min_peak_distance_sec": args.min_peak_distance_sec,
            "long_response_rescue_sec": args.long_response_rescue_sec,
            "match_tolerance_sec": args.match_tolerance_sec,
            "thresholds": "deployable leave-one-video-out adaptive median-logit",
        },
        "review_database": review_database_summary(args.review_db),
        "dataset": {
            "videos": len(video_ids),
            "total_duration_sec": total_duration,
            "gt_events": len(flattened_gt),
            "gt_by_label": dict(Counter(event["label"] for event in flattened_gt)),
            "raw_active_windows_by_label": dict(raw_by_label),
        },
        "workload": {
            "legacy_full_positive_window_union": summarize_global(legacy_union),
            "candidate_fp_only": summarize_global(candidate_fp_only),
            "candidate_same_label_fp_only": summarize_global(candidate_same_label_fp_only),
            "candidate_online": summarize_global(candidate_all),
            "candidate_plus_all_gt": summarize_global(candidate_plus_gt),
            "gt_audit_only": summarize_global(gt_audit_only),
            "close_set_piece_gt_under_120s": summarize_global(
                close_set_piece_gt_audit
            ),
            "gt_audit_plus_gt_external_candidates": summarize_global(
                gt_audit_plus_external_candidates
            ),
        },
        "set_piece_subtype_gap_statistics": subtype_gap_statistics(all_gt),
        "per_video": per_video,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for protocol, rows in queue_rows_by_protocol.items():
        with (args.output_dir / f"review_queue_{protocol}.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.output_dir / "per_video.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "video_id", "duration_sec", "online_segments", "online_duration_sec", "online_ratio",
                "fp_only_segments", "fp_only_duration_sec", "fp_only_ratio",
                "plus_gt_segments", "plus_gt_duration_sec", "plus_gt_ratio",
                "repair_segments", "repair_duration_sec", "repair_ratio",
            ],
        )
        writer.writeheader()
        for row in per_video:
            writer.writerow(
                {
                    "video_id": row["video_id"],
                    "duration_sec": row["duration_sec"],
                    "online_segments": row["candidate_online"]["segments"],
                    "online_duration_sec": row["candidate_online"]["duration_sec"],
                    "online_ratio": row["candidate_online"]["duration_ratio"],
                    "fp_only_segments": row["candidate_fp_only"]["segments"],
                    "fp_only_duration_sec": row["candidate_fp_only"]["duration_sec"],
                    "fp_only_ratio": row["candidate_fp_only"]["duration_ratio"],
                    "plus_gt_segments": row["candidate_plus_all_gt"]["segments"],
                    "plus_gt_duration_sec": row["candidate_plus_all_gt"]["duration_sec"],
                    "plus_gt_ratio": row["candidate_plus_all_gt"]["duration_ratio"],
                    "repair_segments": row["gt_audit_plus_gt_external_candidates"]["segments"],
                    "repair_duration_sec": row["gt_audit_plus_gt_external_candidates"]["duration_sec"],
                    "repair_ratio": row["gt_audit_plus_gt_external_candidates"]["duration_ratio"],
                }
            )
    print(json.dumps({"workload": report["workload"], "review_database": report["review_database"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
