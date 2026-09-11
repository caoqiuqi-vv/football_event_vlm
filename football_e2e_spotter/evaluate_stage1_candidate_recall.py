#!/usr/bin/env python3
"""Strict no-NMS Stage-1 candidate coverage gate for OOF slot exports."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from football_e2e_spotter.set_data import load_set_events
from football_e2e_spotter.set_spotting import SET_LABELS


def ids(path: str) -> list[str]:
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def one_to_one(rows: list[dict], targets: dict[str, list[float]], label: str) -> dict[str, float | int]:
    remaining = {video: list(times) for video, times in targets.items()}
    true_positive = false_positive = 0
    accepted = sorted((row for row in rows if row["label"] == label), key=lambda row: float(row["score"]), reverse=True)
    for row in accepted:
        values = remaining.get(row["video_id"], [])
        if values:
            nearest = min(range(len(values)), key=lambda index: abs(values[index] - float(row["time_sec"])))
            if abs(values[nearest] - float(row["time_sec"])) <= 3.0:
                true_positive += 1
                values.pop(nearest)
                continue
        false_positive += 1
    false_negative = sum(len(values) for values in remaining.values())
    return {
        "tp": true_positive, "fp": false_positive, "fn": false_negative,
        "precision": true_positive / max(true_positive + false_positive, 1),
        "recall": true_positive / max(true_positive + false_negative, 1),
        "candidates": len(accepted),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--ids", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.candidates).read_text().splitlines() if line.strip()]
    video_ids = ids(args.ids)
    targets: dict[int, defaultdict[str, list[float]]] = {index: defaultdict(list) for index in range(len(SET_LABELS))}
    annotations = Path(args.annotations)
    for video_id in video_ids:
        for label, time_sec in load_set_events(annotations / f"{video_id}.json"):
            targets[label][video_id].append(time_sec)
    required = (.95, .90, .90, .90, .90)
    report: dict[str, object] = {
        "no_temporal_nms": True, "tolerance_seconds": 3.0,
        "candidate_threshold": 0.01, "videos": len(video_ids), "classes": {},
    }
    passed = True
    for index, label in enumerate(SET_LABELS):
        result = one_to_one(rows, targets[index], label)
        result["required_recall"] = required[index]
        result["pass"] = bool(result["recall"] >= required[index])
        report["classes"][label] = result
        passed = passed and bool(result["pass"])
    report["pass"] = passed
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
