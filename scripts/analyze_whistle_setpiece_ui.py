#!/usr/bin/env python3
"""Estimate set-piece recall/workload when whistle proposals are added to review UI."""
from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_VIDEOS = (
    "2027564888428580866",
    "2027572095412195330",
    "2027572406738604033",
    "2042125172076392450",
    "2042520971893485569",
    "2042525152494694401",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def within(time_sec: float, start_sec: float, end_sec: float, tolerance_sec: float) -> bool:
    return start_sec - tolerance_sec <= time_sec <= end_sec + tolerance_sec


def temporal_nms(rows: list[dict[str, str]], min_distance_sec: float) -> list[dict[str, float]]:
    """Score-ordered point NMS, then restore chronological order."""
    candidates = [
        {"time_sec": float(row["peak_time_sec"]), "score": float(row["peak_score"])}
        for row in rows
    ]
    kept: list[dict[str, float]] = []
    for item in sorted(candidates, key=lambda x: (-x["score"], x["time_sec"])):
        if all(abs(item["time_sec"] - other["time_sec"]) > min_distance_sec for other in kept):
            kept.append(item)
    return sorted(kept, key=lambda x: x["time_sec"])


def safe_list(raw: str) -> list[str]:
    value = ast.literal_eval(raw)
    return [str(item) for item in value]


def evaluate_setting(
    videos: list[str],
    gt: dict[str, list[dict[str, str]]],
    segments: dict[str, list[dict[str, object]]],
    whistle_rows: dict[str, list[dict[str, str]]],
    pre_sec: float,
    post_sec: float,
    nms_sec: float,
    tolerance_sec: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    num_gt = 0
    model_labeled_hits = 0
    existing_visible_hits = 0
    whistle_hits = 0
    combined_hits = 0
    whistle_candidates = 0
    candidate_tp = 0
    standalone_candidates = 0
    flagged_existing_segments: set[tuple[str, int]] = set()
    matched_by_raw = Counter()
    total_by_raw = Counter()
    detail: list[dict[str, object]] = []

    for video_id in videos:
        video_gt = gt[video_id]
        video_segments = segments.get(video_id, [])
        whistles = temporal_nms(whistle_rows[video_id], nms_sec)
        proposals = [
            {
                "video_id": video_id,
                "time_sec": item["time_sec"],
                "score": item["score"],
                "start_sec": max(0.0, item["time_sec"] - pre_sec),
                "end_sec": item["time_sec"] + post_sec,
            }
            for item in whistles
        ]
        whistle_candidates += len(proposals)

        for proposal in proposals:
            hits = [
                event for event in video_gt
                if within(float(event["time_sec"]), float(proposal["start_sec"]), float(proposal["end_sec"]), tolerance_sec)
            ]
            proposal["matched_set_piece"] = bool(hits)
            proposal["matched_raw_labels"] = sorted({event["raw_label"] for event in hits})
            if hits:
                candidate_tp += 1
            overlap_indices = [
                index for index, segment in enumerate(video_segments)
                if float(proposal["end_sec"]) > float(segment["start_sec"])
                and float(proposal["start_sec"]) < float(segment["end_sec"])
            ]
            if overlap_indices:
                # Multiple whistle points in the same DINO segment require only one extra set-piece decision.
                best = max(
                    overlap_indices,
                    key=lambda index: min(float(proposal["end_sec"]), float(video_segments[index]["end_sec"]))
                    - max(float(proposal["start_sec"]), float(video_segments[index]["start_sec"])),
                )
                flagged_existing_segments.add((video_id, best))
                proposal["ui_action"] = "flag_existing_segment"
                proposal["existing_segment_index"] = int(video_segments[best]["segment_index"])
            else:
                standalone_candidates += 1
                proposal["ui_action"] = "new_review_segment"
                proposal["existing_segment_index"] = ""
            detail.append(proposal)

        for event in video_gt:
            time_sec = float(event["time_sec"])
            total_by_raw[event["raw_label"]] += 1
            label_hit = any(
                "set_piece" in segment["labels"]
                and within(time_sec, float(segment["start_sec"]), float(segment["end_sec"]), tolerance_sec)
                for segment in video_segments
            )
            visible_hit = any(
                within(time_sec, float(segment["start_sec"]), float(segment["end_sec"]), tolerance_sec)
                for segment in video_segments
            )
            whistle_hit = any(
                within(time_sec, float(proposal["start_sec"]), float(proposal["end_sec"]), tolerance_sec)
                for proposal in proposals
            )
            num_gt += 1
            model_labeled_hits += int(label_hit)
            existing_visible_hits += int(visible_hit)
            whistle_hits += int(whistle_hit)
            combined_hits += int(visible_hit or whistle_hit)
            if whistle_hit:
                matched_by_raw[event["raw_label"]] += 1

    precision = candidate_tp / whistle_candidates if whistle_candidates else 0.0
    result: dict[str, object] = {
        "whistle_view": {"pre_sec": pre_sec, "post_sec": post_sec, "duration_sec": pre_sec + post_sec},
        "whistle_point_nms_sec": nms_sec,
        "num_set_piece_gt": num_gt,
        "model_set_piece_labeled_recall": model_labeled_hits / num_gt if num_gt else 0.0,
        "existing_ui_any_label_candidate_recall": existing_visible_hits / num_gt if num_gt else 0.0,
        "whistle_candidate_recall": whistle_hits / num_gt if num_gt else 0.0,
        "combined_ui_candidate_recall": combined_hits / num_gt if num_gt else 0.0,
        "additional_gt_recovered_over_existing_ui": combined_hits - existing_visible_hits,
        "whistle_candidates": whistle_candidates,
        "whistle_candidates_containing_set_piece": candidate_tp,
        "whistle_candidate_yield": precision,
        "new_standalone_review_segments": standalone_candidates,
        "existing_review_segments_flagged": len(flagged_existing_segments),
        "additional_set_piece_decisions": standalone_candidates + len(flagged_existing_segments),
        "additional_playback_seconds_upper_bound": standalone_candidates * (pre_sec + post_sec),
        "per_raw_label_whistle_recall": {
            label: {
                "matched": matched_by_raw[label],
                "gt": total,
                "recall": matched_by_raw[label] / total if total else 0.0,
            }
            for label, total in sorted(total_by_raw.items())
        },
    }
    return result, detail


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--review-segments", type=Path, required=True)
    parser.add_argument("--whistle-dir", type=Path, required=True)
    parser.add_argument("--videos", default=",".join(DEFAULT_VIDEOS))
    parser.add_argument("--exclude", default="2027572406738604033:set_piece")
    parser.add_argument("--tolerance-sec", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    videos = [value.strip() for value in args.videos.split(",") if value.strip()]
    excluded = {tuple(item.split(":", 1)) for item in args.exclude.split(",") if item.strip()}
    gt: dict[str, list[dict[str, str]]] = {}
    whistle_rows: dict[str, list[dict[str, str]]] = {}
    for video_id in videos:
        gt[video_id] = [
            row for row in read_rows(args.run_dir / video_id / "gt_events.csv")
            if row["label"] == "set_piece" and (video_id, "set_piece") not in excluded
        ]
        whistle_rows[video_id] = read_rows(args.whistle_dir / f"{video_id}_whistles.csv")

    segments: dict[str, list[dict[str, object]]] = defaultdict(list)
    all_segments = read_rows(args.review_segments)
    for row in all_segments:
        parsed: dict[str, object] = {
            "video_id": str(row["video_id"]),
            "segment_index": int(row["segment_index"]),
            "start_sec": float(row["start_sec"]),
            "end_sec": float(row["end_sec"]),
            "labels": safe_list(row["labels"]),
        }
        segments[str(row["video_id"])].append(parsed)

    # Exact duration statistics for incorrect (segment, label) decisions.
    wrong_lengths: list[float] = []
    wrong_by_label: dict[str, list[float]] = defaultdict(list)
    for video_id in videos:
        all_gt_rows = [
            row for row in read_rows(args.run_dir / video_id / "gt_events.csv")
            if (video_id, row["label"]) not in excluded
        ]
        for segment in segments.get(video_id, []):
            duration = float(segment["end_sec"]) - float(segment["start_sec"])
            for label in segment["labels"]:
                correct = any(
                    row["label"] == label
                    and within(float(row["time_sec"]), float(segment["start_sec"]), float(segment["end_sec"]), args.tolerance_sec)
                    for row in all_gt_rows
                )
                if not correct:
                    wrong_lengths.append(duration)
                    wrong_by_label[str(label)].append(duration)

    def duration_stats(values: list[float]) -> dict[str, object]:
        values = sorted(values)
        percentile = lambda q: values[min(len(values) - 1, int(round(q * (len(values) - 1))))]
        return {
            "count": len(values), "mean_sec": sum(values) / len(values), "median_sec": percentile(0.5),
            "p90_sec": percentile(0.9), "max_sec": max(values),
            "counts_by_duration_sec": dict(sorted(Counter(round(value, 3) for value in values).items())),
        }

    settings = []
    details_by_name: dict[str, list[dict[str, object]]] = {}
    for pre_sec, post_sec, nms_sec in ((5.0, 5.0, 5.0), (5.0, 10.0, 5.0), (5.0, 15.0, 5.0), (5.0, 10.0, 10.0)):
        result, detail = evaluate_setting(
            videos, gt, segments, whistle_rows, pre_sec, post_sec, nms_sec, args.tolerance_sec
        )
        name = f"pre{pre_sec:g}_post{post_sec:g}_nms{nms_sec:g}"
        result["name"] = name
        settings.append(result)
        details_by_name[name] = detail

    # Default UI option: 5 seconds before and 10 seconds after, 5-second point NMS.
    selected_name = "pre5_post10_nms5"
    output = {
        "definitions": {
            "whistle_candidate_yield": "fraction of whistle UI candidates whose displayed interval contains >=1 set_piece GT; this is pre-review yield, not automatic classifier precision",
            "combined_ui_candidate_recall": "set_piece GT visible in either an existing DINO review segment or a whistle review interval",
            "additional_set_piece_decisions": "one decision per newly created whistle segment plus one per existing DINO segment receiving >=1 whistle flag",
            "manual_final_precision": "not estimated: with perfect human confirmation it is 1.0; actual value depends on reviewer error",
        },
        "tolerance_sec": args.tolerance_sec,
        "excluded_video_label_pairs": sorted([list(item) for item in excluded]),
        "wrong_segment_label_duration": {
            "all": duration_stats(wrong_lengths),
            "by_label": {label: duration_stats(values) for label, values in sorted(wrong_by_label.items())},
        },
        "settings": settings,
        "selected_setting": selected_name,
        "selected_result": next(item for item in settings if item["name"] == selected_name),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    fields = sorted({key for row in details_by_name[selected_name] for key in row})
    with (args.output_dir / "whistle_candidates_selected.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(details_by_name[selected_name])
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
