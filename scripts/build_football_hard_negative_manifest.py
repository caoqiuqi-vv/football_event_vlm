#!/usr/bin/env python
from __future__ import annotations

import argparse
import bisect
import csv
import json
from pathlib import Path
from typing import Any


LABELS = ["shot", "save", "set_piece"]


def parse_thresholds(raw: str) -> dict[str, float]:
    thresholds = {label: float("inf") for label in LABELS}
    if not raw.strip():
        return thresholds
    for item in raw.split(","):
        if not item.strip():
            continue
        label, value = item.split("=", 1)
        label = label.strip()
        if label not in thresholds:
            raise ValueError(f"Unsupported label '{label}', expected one of {LABELS}")
        thresholds[label] = float(value)
    return thresholds


def load_gt_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            label = row.get("label", "")
            if label not in LABELS:
                continue
            time_sec = float(row["time_sec"])
            events.append(
                {
                    "label": label,
                    "time_sec": time_sec,
                    "event_id": row.get("event_id", ""),
                }
            )
    return events


def index_gt_times(gt_events: list[dict[str, Any]]) -> dict[str, list[float]]:
    times_by_label = {label: [] for label in LABELS}
    times_by_label["__any__"] = []
    for event in gt_events:
        time_sec = float(event["time_sec"])
        times_by_label.setdefault(event["label"], []).append(time_sec)
        times_by_label["__any__"].append(time_sec)
    for times in times_by_label.values():
        times.sort()
    return times_by_label


def overlaps_gt(
    *,
    label: str,
    start_sec: float,
    end_sec: float,
    gt_times: dict[str, list[float]],
    tolerance_sec: float,
    any_label: bool,
) -> bool:
    safe_start = start_sec - tolerance_sec
    safe_end = end_sec + tolerance_sec
    times = gt_times["__any__"] if any_label else gt_times.get(label, [])
    index = bisect.bisect_left(times, safe_start)
    return index < len(times) and times[index] <= safe_end


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.eval_run_dir)
    thresholds = parse_thresholds(args.min_probs)
    hard_negatives: list[dict[str, Any]] = []
    per_video_counts: dict[str, int] = {}
    per_label_counts = {label: 0 for label in LABELS}

    video_dirs = sorted(
        path for path in run_dir.iterdir() if path.is_dir() and (path / "window_predictions.csv").exists()
    )
    for video_dir in video_dirs:
        gt_path = video_dir / "gt_events.csv"
        if not gt_path.exists():
            continue
        video_id = video_dir.name
        gt_events = load_gt_events(gt_path)
        gt_times = index_gt_times(gt_events)
        candidates: list[dict[str, Any]] = []
        with (video_dir / "window_predictions.csv").open(newline="") as f:
            for row in csv.DictReader(f):
                start_sec = float(row["start_sec"])
                end_sec = float(row["end_sec"])
                for label, min_prob in thresholds.items():
                    score = float(row.get(f"prob_{label}", 0.0) or 0.0)
                    if score < min_prob:
                        continue
                    if overlaps_gt(
                        label=label,
                        start_sec=start_sec,
                        end_sec=end_sec,
                        gt_times=gt_times,
                        tolerance_sec=args.tolerance_sec,
                        any_label=args.reject_any_label_gt,
                    ):
                        continue
                    candidates.append(
                        {
                            "source": args.source,
                            "video_id": video_id,
                            "label": label,
                            "score": score,
                            "start_sec": start_sec,
                            "end_sec": end_sec,
                            "center_sec": (start_sec + end_sec) * 0.5,
                            "window_index": int(row["index"]),
                            "mining_run": run_dir.name,
                        }
                    )
        candidates.sort(key=lambda item: float(item["score"]), reverse=True)
        if args.max_per_video_per_label > 0:
            limited: list[dict[str, Any]] = []
            for label in LABELS:
                label_candidates = [item for item in candidates if item["label"] == label]
                limited.extend(label_candidates[: args.max_per_video_per_label])
            candidates = sorted(limited, key=lambda item: float(item["score"]), reverse=True)
        elif args.max_per_video > 0:
            candidates = candidates[: args.max_per_video]
        hard_negatives.extend(candidates)
        per_video_counts[video_id] = len(candidates)
        for item in candidates:
            per_label_counts[item["label"]] += 1

    return {
        "schema": "football_hard_negative_manifest_v1",
        "eval_run_dir": str(run_dir),
        "source": args.source,
        "min_probs": thresholds,
        "tolerance_sec": args.tolerance_sec,
        "reject_any_label_gt": args.reject_any_label_gt,
        "max_per_video": args.max_per_video,
        "max_per_video_per_label": args.max_per_video_per_label,
        "num_videos": len(video_dirs),
        "num_hard_negatives": len(hard_negatives),
        "per_label_counts": per_label_counts,
        "per_video_counts": per_video_counts,
        "hard_negatives": hard_negatives,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build hard-negative manifest from saved dense eval window outputs.")
    parser.add_argument("--eval-run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", default="xbotgo_0608")
    parser.add_argument("--min-probs", default="shot=0.35,save=0.45")
    parser.add_argument("--tolerance-sec", type=float, default=5.0)
    parser.add_argument("--max-per-video", type=int, default=0)
    parser.add_argument("--max-per-video-per-label", type=int, default=40)
    parser.add_argument(
        "--reject-any-label-gt",
        action="store_true",
        help="Reject windows near any GT label, not only the mined label. Recommended for multi-label negative training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(
        f"wrote hard_negatives={manifest['num_hard_negatives']} "
        f"videos={manifest['num_videos']} output={output}"
    )
    print(f"per_label_counts={manifest['per_label_counts']}")


if __name__ == "__main__":
    main()
