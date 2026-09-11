#!/usr/bin/env python3
"""Fail unless a review manifest covers every GT and selected dense window."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


LABELS = ("shot", "save", "set_piece")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--queue-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    manifest = load(args.manifest)
    queue_report = load(args.queue_report)
    selection_by_video = {
        str(row["video_id"]): row for row in queue_report["per_video"]
    }
    missing_gt = []
    missing_windows = []
    gt_counts: Counter[str] = Counter()
    window_counts: Counter[str] = Counter()
    candidate_anchors: dict[str, list[dict]] = {}
    with args.queue.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            candidate_anchors.setdefault(str(row["video_id"]), []).extend(
                anchor for anchor in row.get("anchors", [])
                if anchor.get("source") == "candidate"
            )
    for video in manifest["videos"]:
        video_id = str(video["video_id"])
        selection = selection_by_video[video_id]
        dense_dir = Path(selection["dense_source_dir"])
        intervals = {
            label: [
                (float(event["support_start_sec"]), float(event["support_end_sec"]))
                for event in video["events"]
                if event["label"] == label
            ]
            for label in LABELS
        }
        gt = load(dense_dir / "gt_events.json")
        for event in gt:
            label = str(event["label"])
            time_sec = float(event["time_sec"])
            gt_counts[label] += 1
            if not any(start <= time_sec <= end for start, end in intervals[label]):
                missing_gt.append({"video_id": video_id, "label": label, "time_sec": time_sec})
        for anchor in candidate_anchors.get(video_id, []):
            center = float(anchor["time_sec"])
            for label in anchor.get("labels", []):
                window_counts[label] += 1
                if not any(start <= center <= end for start, end in intervals[label]):
                    missing_windows.append({
                        "video_id": video_id,
                        "window_indices": anchor.get("window_indices", []),
                        "label": label,
                        "center_sec": center,
                    })
    report = {
        "manifest": str(args.manifest.resolve()),
        "videos": len(manifest["videos"]),
        "gt_events_by_label": dict(gt_counts),
        "selected_model_candidates_by_label": dict(window_counts),
        "missing_gt": missing_gt,
        "missing_model_candidates": missing_windows,
        "gt_coverage": 1.0 if not missing_gt else (sum(gt_counts.values()) - len(missing_gt)) / sum(gt_counts.values()),
        "selected_model_candidate_coverage": 1.0 if not missing_windows else (sum(window_counts.values()) - len(missing_windows)) / sum(window_counts.values()),
        "passed": not missing_gt and not missing_windows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "videos", "gt_events_by_label", "selected_model_candidates_by_label",
        "gt_coverage", "selected_model_candidate_coverage", "passed",
    )}, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
