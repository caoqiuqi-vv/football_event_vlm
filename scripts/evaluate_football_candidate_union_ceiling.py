#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


DEFAULT_LABELS = ("shot", "save", "set_piece")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def parse_label_floats(raw: str, labels: Sequence[str], default: float) -> dict[str, float]:
    values = {label: float(default) for label in labels}
    if not raw:
        return values
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Expected label=value item, got {item!r}")
        label, value = item.split("=", 1)
        label = label.strip()
        if label not in values:
            raise ValueError(f"Unknown label {label!r}; expected one of {labels}")
        values[label] = float(value)
    return values


def load_video_ids(run_dirs: Sequence[Path], video_id_file: str) -> list[str]:
    if video_id_file:
        return [
            line.strip()
            for line in Path(video_id_file).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    ids: set[str] = set()
    for run_dir in run_dirs:
        ids.update(
            path.name
            for path in run_dir.iterdir()
            if path.is_dir() and (path / "window_predictions.csv").exists()
        )
    return sorted(ids)


def load_gt(video_dir: Path, labels: Sequence[str]) -> dict[str, list[float]]:
    result = {label: [] for label in labels}
    for row in read_csv(video_dir / "gt_events.csv"):
        label = row.get("label", "")
        if label in result:
            result[label].append(float(row["time_sec"]))
    for times in result.values():
        times.sort()
    return result


def merge_segments(segments: list[tuple[float, float]], merge_gap_sec: float) -> list[tuple[float, float]]:
    if not segments:
        return []
    ordered = sorted(segments)
    merged: list[list[float]] = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        if start <= merged[-1][1] + merge_gap_sec:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(float(start), float(end)) for start, end in merged]


def selected_segments_for_video(
    run_dirs: Sequence[Path],
    video_id: str,
    labels: Sequence[str],
    thresholds: dict[str, float],
    *,
    pad_sec: float,
    score_mode: str,
) -> dict[str, list[tuple[float, float]]]:
    by_label: dict[str, list[tuple[float, float]]] = {label: [] for label in labels}
    for run_dir in run_dirs:
        window_path = run_dir / video_id / "window_predictions.csv"
        if not window_path.exists():
            continue
        for row in read_csv(window_path):
            start = float(row["start_sec"]) - pad_sec
            end = float(row["end_sec"]) + pad_sec
            for label in labels:
                columns = [f"prob_{label}"]
                if score_mode == "clip_or_frame":
                    columns.extend((f"frame_prob_{label}", f"global_frame_prob_{label}"))
                score = max(
                    (float(row[col]) for col in columns if row.get(col) not in (None, "")),
                    default=0.0,
                )
                if score >= thresholds[label]:
                    by_label[label].append((start, end))
    return by_label


def evaluate(
    run_dirs: Sequence[Path],
    video_ids: Sequence[str],
    labels: Sequence[str],
    thresholds: dict[str, float],
    *,
    merge_gap_sec: float,
    pad_sec: float,
    score_mode: str,
) -> dict[str, Any]:
    per_class: dict[str, dict[str, Any]] = {
        label: {
            "matched": 0,
            "gt": 0,
            "segments": 0,
            "coverage_seconds": 0.0,
        }
        for label in labels
    }
    per_video: list[dict[str, Any]] = []
    used_video_ids: list[str] = []
    for video_id in video_ids:
        gt_dir = next((run_dir / video_id for run_dir in run_dirs if (run_dir / video_id / "gt_events.csv").exists()), None)
        if gt_dir is None:
            continue
        used_video_ids.append(video_id)
        gt_by_label = load_gt(gt_dir, labels)
        raw_segments = selected_segments_for_video(
            run_dirs,
            video_id,
            labels,
            thresholds,
            pad_sec=pad_sec,
            score_mode=score_mode,
        )
        for label in labels:
            segments = merge_segments(raw_segments[label], merge_gap_sec)
            matched = 0
            for gt_time in gt_by_label[label]:
                if any(start <= gt_time <= end for start, end in segments):
                    matched += 1
            coverage = sum(max(0.0, end - start) for start, end in segments)
            per_class[label]["matched"] += matched
            per_class[label]["gt"] += len(gt_by_label[label])
            per_class[label]["segments"] += len(segments)
            per_class[label]["coverage_seconds"] += coverage
            per_video.append(
                {
                    "video_id": video_id,
                    "label": label,
                    "matched": matched,
                    "gt": len(gt_by_label[label]),
                    "recall": matched / len(gt_by_label[label]) if gt_by_label[label] else 0.0,
                    "segments": len(segments),
                    "coverage_seconds": coverage,
                }
            )
    for label, item in per_class.items():
        item["recall"] = item["matched"] / item["gt"] if item["gt"] else 0.0
        item["coverage_minutes"] = item["coverage_seconds"] / 60.0
        item["segments_per_gt"] = item["segments"] / item["gt"] if item["gt"] else 0.0
    micro_matched = sum(item["matched"] for item in per_class.values())
    micro_gt = sum(item["gt"] for item in per_class.values())
    micro_segments = sum(item["segments"] for item in per_class.values())
    micro_coverage = sum(item["coverage_seconds"] for item in per_class.values())
    return {
        "run_dirs": [str(path) for path in run_dirs],
        "video_ids": used_video_ids,
        "num_videos": len(used_video_ids),
        "labels": list(labels),
        "thresholds": thresholds,
        "merge_gap_sec": merge_gap_sec,
        "pad_sec": pad_sec,
        "score_mode": score_mode,
        "per_class": per_class,
        "micro": {
            "matched": micro_matched,
            "gt": micro_gt,
            "recall": micro_matched / micro_gt if micro_gt else 0.0,
            "segments": micro_segments,
            "coverage_seconds": micro_coverage,
            "coverage_minutes": micro_coverage / 60.0,
            "segments_per_gt": micro_segments / micro_gt if micro_gt else 0.0,
        },
        "per_video": per_video,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate full-segment recall ceiling for a low-threshold union of football dense generator runs.")
    parser.add_argument("--run-dirs", nargs="+", required=True, type=Path)
    parser.add_argument("--video-id-file", default="")
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--thresholds", default="")
    parser.add_argument("--default-threshold", type=float, default=0.05)
    parser.add_argument("--merge-gap-sec", type=float, default=2.0)
    parser.add_argument("--pad-sec", type=float, default=0.0)
    parser.add_argument("--score-mode", choices=("clip", "clip_or_frame"), default="clip")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    labels = tuple(label.strip() for label in args.labels.split(",") if label.strip())
    thresholds = parse_label_floats(args.thresholds, labels, args.default_threshold)
    video_ids = load_video_ids(args.run_dirs, args.video_id_file)
    report = evaluate(
        args.run_dirs,
        video_ids,
        labels,
        thresholds,
        merge_gap_sec=args.merge_gap_sec,
        pad_sec=args.pad_sec,
        score_mode=args.score_mode,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "micro": report["micro"], "per_class": report["per_class"]}, indent=2))


if __name__ == "__main__":
    main()
