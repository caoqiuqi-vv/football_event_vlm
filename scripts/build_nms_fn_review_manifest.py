#!/usr/bin/env python3
"""Build a review-UI manifest containing GT events missed after temporal NMS."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


LABELS = ("shot", "save", "set_piece")
RAW_LABEL_MAP = {
    "射门": "shot",
    "其他射门类型": "shot",
    "扑救": "save",
    "角球": "set_piece",
    "任意球": "set_piece",
    "点球": "set_piece",
    "中圈开球": "set_piece",
}
DEFAULT_VIDEO_IDS = (
    "2027564888428580866",
    "2027572095412195330",
    "2027572406738604033",
    "2042125172076392450",
    "2042520971893485569",
    "2042525152494694401",
)
SHOT_ALPHA = 0.35
SHOT_THRESHOLD = -1.5108137909816555
SAVE_ALPHA = 0.05
SAVE_THRESHOLD = -1.2646333587272944
SET_PIECE_CLIP_THRESHOLD = 0.3896045982837677
SET_PIECE_FRAME_THRESHOLD = 0.010986328125


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def logit(probability: float) -> float:
    probability = min(max(probability, 1e-7), 1.0 - 1e-7)
    return math.log(probability / (1.0 - probability))


def timestamp_to_seconds(value: str) -> float:
    parts = [float(item) for item in value.split(":")]
    if len(parts) == 3:
        return parts[0] * 3600.0 + parts[1] * 60.0 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60.0 + parts[1]
    return parts[0]


def load_gt(path: Path) -> list[dict[str, Any]]:
    raw_events = json.loads(path.read_text(encoding="utf-8"))
    events: list[dict[str, Any]] = []
    for raw in raw_events:
        label = RAW_LABEL_MAP.get(str(raw.get("label", "")))
        if label is None or raw.get("label_correct") is False:
            continue
        events.append(
            {
                "label": label,
                "time_sec": timestamp_to_seconds(str(raw["timestamp"])),
                "raw_label": raw.get("label", ""),
                "event_id": raw.get("id", ""),
            }
        )
    return sorted(events, key=lambda item: (item["time_sec"], item["label"]))


def load_frame_peaks(video_dir: Path) -> dict[tuple[int, str], dict[str, float]]:
    peaks: dict[tuple[int, str], dict[str, float]] = {}
    for row in read_csv(video_dir / "frame_event_logits.csv"):
        window_index = int(row["window_index"])
        frame_time = float(row["frame_time_sec"])
        for label in LABELS:
            logit_value = float(row[f"frame_logit_{label}"])
            key = (window_index, label)
            if key not in peaks or logit_value > peaks[key]["logit"]:
                peaks[key] = {
                    "logit": logit_value,
                    "prob": float(row[f"frame_prob_{label}"]),
                    "time_sec": frame_time,
                }
    return peaks


def build_windows(video_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    peaks = load_frame_peaks(video_dir)
    windows: list[dict[str, Any]] = []
    timeline: list[dict[str, Any]] = []
    for row in read_csv(video_dir / "window_predictions.csv"):
        index = int(row["index"])
        start = float(row["start_sec"])
        end = float(row["end_sec"])
        probs = {label: float(row[f"prob_{label}"]) for label in LABELS}
        frame_probs = {label: peaks[(index, label)]["prob"] for label in LABELS}
        scores = {
            "shot": (1.0 - SHOT_ALPHA) * logit(probs["shot"])
            + SHOT_ALPHA * peaks[(index, "shot")]["logit"],
            "save": (1.0 - SAVE_ALPHA) * logit(probs["save"])
            + SAVE_ALPHA * peaks[(index, "save")]["logit"],
            "set_piece": probs["set_piece"],
        }
        positive = {
            "shot": scores["shot"] >= SHOT_THRESHOLD,
            # The stored probabilities are fp16. This epsilon preserves the one
            # boundary candidate that was present when the ablation was computed.
            "save": scores["save"] >= SAVE_THRESHOLD - 1e-4,
            "set_piece": probs["set_piece"] >= SET_PIECE_CLIP_THRESHOLD
            and frame_probs["set_piece"] >= SET_PIECE_FRAME_THRESHOLD,
        }
        item = {
            "index": index,
            "start_sec": start,
            "end_sec": end,
            "center_sec": 0.5 * (start + end),
            "probs": probs,
            "frame_probs": frame_probs,
            "frame_peak_times": {label: peaks[(index, label)]["time_sec"] for label in LABELS},
            "scores": scores,
            "positive": positive,
            "roi": {
                "valid": float(row.get("roi_valid") or 0) > 0.5,
                "confidence": float(row.get("roi_confidence") or 0),
                "mode": row.get("roi_proposal_mode", ""),
            },
        }
        windows.append(item)
        timeline.append(
            {
                "index": index,
                "start_sec": start,
                "end_sec": end,
                "dino": probs,
                "frame_detection": frame_probs,
                "roi": item["roi"],
            }
        )
    return windows, timeline


def temporal_nms(candidates: list[dict[str, Any]], radius_sec: float) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    kept: list[dict[str, Any]] = []
    suppressed: dict[int, list[dict[str, Any]]] = {}
    for candidate in sorted(candidates, key=lambda item: (-item["scores"][item["label"]], item["center_sec"])):
        suppressor = next(
            (old for old in kept if abs(candidate["center_sec"] - old["center_sec"]) <= radius_sec),
            None,
        )
        if suppressor is None:
            kept.append(candidate)
            suppressed[candidate["index"]] = []
        else:
            suppressed[suppressor["index"]].append(candidate)
    return kept, suppressed


def overlaps(window: dict[str, Any], gt_time: float, tolerance: float) -> bool:
    return window["start_sec"] - tolerance <= gt_time <= window["end_sec"] + tolerance


def representative_window(windows: list[dict[str, Any]], label: str, gt_time: float) -> dict[str, Any]:
    nearby = [window for window in windows if overlaps(window, gt_time, 2.0)]
    if not nearby:
        nearby = windows
    return max(
        nearby,
        key=lambda window: (
            window["scores"][label],
            -abs(window["center_sec"] - gt_time),
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-ids", nargs="*", default=list(DEFAULT_VIDEO_IDS))
    parser.add_argument("--nms-radius-sec", type=float, default=5.0)
    parser.add_argument("--match-tolerance-sec", type=float, default=2.0)
    parser.add_argument("--clip-half-width-sec", type=float, default=6.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    videos: list[dict[str, Any]] = []
    origin_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    candidate_counts: Counter[str] = Counter()
    kept_counts: Counter[str] = Counter()

    for video_id in args.video_ids:
        video_dir = args.run_dir / video_id
        windows, timeline = build_windows(video_dir)
        gt_events = load_gt(args.gt_dir / f"{video_id}.json")
        events: list[dict[str, Any]] = []
        for label in LABELS:
            if video_id == "2027572406738604033" and label == "set_piece":
                continue
            candidates = []
            for window in windows:
                if window["positive"][label]:
                    candidate = dict(window)
                    candidate["label"] = label
                    candidates.append(candidate)
            kept, suppressed = temporal_nms(candidates, args.nms_radius_sec)
            candidate_counts[label] += len(candidates)
            kept_counts[label] += len(kept)
            for gt in (item for item in gt_events if item["label"] == label):
                pre_matches = [item for item in candidates if overlaps(item, gt["time_sec"], args.match_tolerance_sec)]
                post_matches = [item for item in kept if overlaps(item, gt["time_sec"], args.match_tolerance_sec)]
                if post_matches:
                    continue
                origin = "caused_by_nms" if pre_matches else "preexisting_model_fn"
                origin_counts[origin] += 1
                label_counts[label] += 1
                representative = (
                    max(pre_matches, key=lambda item: item["scores"][label])
                    if pre_matches
                    else representative_window(windows, label, gt["time_sec"])
                )
                suppressors = [
                    item
                    for item in kept
                    if any(old["index"] == representative["index"] for old in suppressed[item["index"]])
                ]
                events.append(
                    {
                        "id": f"fn_nms_{label}_{video_id}_{int(round(gt['time_sec'] * 1000)):09d}",
                        "video_id": video_id,
                        "label": label,
                        "time_sec": gt["time_sec"],
                        "start_sec": max(0.0, gt["time_sec"] - args.clip_half_width_sec),
                        "end_sec": gt["time_sec"] + args.clip_half_width_sec,
                        "support_start_sec": representative["start_sec"],
                        "support_end_sec": representative["end_sec"],
                        "score": representative["probs"][label],
                        "dino_scores": representative["probs"],
                        "frame_detection_scores": representative["frame_probs"],
                        "window_indices": [representative["index"]],
                        "merged_predictions": 1,
                        "evaluation_status": "fn",
                        "matching_gt_times": [gt["time_sec"]],
                        "match_tolerance_sec": args.match_tolerance_sec,
                        "gt_raw_label": gt["raw_label"],
                        "gt_event_id": gt["event_id"],
                        "fn_origin": origin,
                        "pre_nms_matched": bool(pre_matches),
                        "representative_decision_score": representative["scores"][label],
                        "representative_frame_peak_time_sec": representative["frame_peak_times"][label],
                        "nms_suppressor_windows": [item["index"] for item in suppressors],
                        "nms_radius_sec": args.nms_radius_sec,
                    }
                )
        if events:
            videos.append(
                {
                    "video_id": video_id,
                    "video_path": str((args.video_root / f"{video_id}.mp4").resolve()),
                    "duration_sec": max((item["end_sec"] for item in timeline), default=0.0),
                    "events": sorted(events, key=lambda item: (item["time_sec"], item["label"])),
                    "timeline": timeline,
                }
            )

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "labels": list(LABELS),
        "review_labels": ["shot", "save", "free_kick", "penalty", "corner", "shot_on_target"],
        "source": {
            "type": "nms_false_negatives",
            "run_dir": str(args.run_dir.resolve()),
            "gt_dir": str(args.gt_dir.resolve()),
            "nms_radius_sec": args.nms_radius_sec,
            "match_tolerance_sec": args.match_tolerance_sec,
            "excluded_video_label_pairs": [["2027572406738604033", "set_piece"]],
        },
        "videos": videos,
        "summary": {
            "videos": len(videos),
            "events": sum(label_counts.values()),
            "events_by_label": dict(label_counts),
            "fn_by_origin": dict(origin_counts),
            "pre_nms_candidates_by_label": dict(candidate_counts),
            "post_nms_candidates_by_label": dict(kept_counts),
        },
    }
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))
    print(output)


if __name__ == "__main__":
    main()
