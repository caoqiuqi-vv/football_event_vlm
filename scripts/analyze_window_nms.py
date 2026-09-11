#!/usr/bin/env python
"""Evaluate score-ordered temporal NMS while preserving window-overlap matching."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_long_video_checkpoint import compute_event_metrics, window_overlap_predictions
from recompute_football_eval_protocols import finalize_totals, init_totals, load_gt_events


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "index": int(row["index"]),
            "start_sec": float(row["start_sec"]),
            "end_sec": float(row["end_sec"]),
            **{key: float(value) for key, value in row.items() if key.startswith("prob_")},
        }
        for row in rows
    ]


def nms_keep_windows(predictions: list[dict], radius_sec: float) -> list[dict]:
    kept: list[dict] = []
    for label in sorted({item["label"] for item in predictions}):
        candidates = sorted(
            (item for item in predictions if item["label"] == label),
            key=lambda item: (-float(item["score"]), float(item["time_sec"])),
        )
        label_kept: list[dict] = []
        for item in candidates:
            if any(abs(float(item["time_sec"]) - float(old["time_sec"])) <= radius_sec for old in label_kept):
                continue
            label_kept.append(item)
        kept.extend(label_kept)
    return kept


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--radius-sec", type=float, default=5.0)
    parser.add_argument("--tolerance-sec", type=float, default=2.0)
    parser.add_argument("--exclude", default="2027572406738604033:set_piece")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    video_dirs = sorted(path for path in args.run_dir.iterdir() if (path / "window_predictions.csv").is_file())
    first = json.loads((video_dirs[0] / "summary.json").read_text())
    labels = list(first["labels"])
    thresholds = {label: float(first["thresholds"][label]) for label in labels}
    exclusions = {tuple(item.split(":", 1)) for item in args.exclude.split(",") if item}
    totals = init_totals(labels)
    per_video = []
    for video_dir in video_dirs:
        rows = read_rows(video_dir / "window_predictions.csv")
        base = window_overlap_predictions(rows, labels, thresholds)
        preds = nms_keep_windows(base, args.radius_sec)
        metrics = compute_event_metrics(
            preds,
            load_gt_events(video_dir / "gt_events.csv", labels),
            args.tolerance_sec,
            matching_mode="window",
            allow_many_predictions_per_gt=True,
        )
        for label in labels:
            item = metrics["per_class"][label]
            per_video.append({"video_id": video_dir.name, "label": label, **item})
            if (video_dir.name, label) in exclusions:
                continue
            for key in totals[label]:
                totals[label][key] += int(item[key])
    per_class, micro = finalize_totals(totals, allow_many_predictions_per_gt=True)
    result = {
        "run_dir": str(args.run_dir),
        "radius_sec": args.radius_sec,
        "tolerance_sec": args.tolerance_sec,
        "thresholds": thresholds,
        "exclusions": sorted([list(item) for item in exclusions]),
        "per_class": per_class,
        "micro": micro,
        "per_video": per_video,
    }
    output = args.output or args.run_dir / f"window_nms{args.radius_sec:g}_tol{args.tolerance_sec:g}.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "per_class": per_class, "micro": micro}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
