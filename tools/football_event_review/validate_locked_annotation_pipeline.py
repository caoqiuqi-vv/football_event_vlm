#!/usr/bin/env python3
"""Validate the frozen annotation-repair pipeline and its recall provenance."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path


LABELS = ("shot", "save", "set_piece")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def anchor_key(video_id: str, anchor: dict) -> tuple[str, str, str, int]:
    source = str(anchor.get("source", ""))
    family = str(anchor.get("family") or anchor.get("label") or "")
    return video_id, source, family, round(float(anchor["time_sec"]) * 1000)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    config = load(args.config)
    resolve = lambda value: (root / value).resolve()
    manifest = load(resolve(config["manifest"]))
    adaptive = load(resolve(config["adaptive_report"]))
    run_dir = resolve(config["dense_run"])
    stream_root = resolve(config["stream_root"])

    manifest_anchors: set[tuple[str, str, str, int]] = set()
    manifest_event_ids: set[str] = set()
    video_ids: list[str] = []
    for video in manifest["videos"]:
        video_id = str(video["video_id"])
        video_ids.append(video_id)
        for event in video["events"]:
            manifest_event_ids.add(str(event["id"]))
            for anchor in event.get("evidence_anchors", []):
                manifest_anchors.add(anchor_key(video_id, anchor))

    gt_total = 0
    missing_gt: list[dict] = []
    for video_id in video_ids:
        for event in load(run_dir / video_id / "gt_events.json"):
            gt_total += 1
            key = anchor_key(video_id, {
                "source": "gt", "label": event["label"], "time_sec": event["time_sec"]
            })
            if key not in manifest_anchors:
                missing_gt.append({"video_id": video_id, "label": event["label"], "time_sec": event["time_sec"]})

    queue_candidates: set[tuple[str, str, str, int]] = set()
    with resolve(config["candidate_queue"]).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            for anchor in row.get("anchors", []):
                if anchor.get("source") == "candidate":
                    queue_candidates.add(anchor_key(str(row["video_id"]), anchor))
    missing_candidates = sorted(queue_candidates - manifest_anchors)

    connection = sqlite3.connect(f"file:{resolve(config['database'])}?mode=ro", uri=True)
    database_ids = {str(row[0]) for row in connection.execute("SELECT id FROM events")}
    connection.close()
    missing_database_events = sorted(manifest_event_ids - database_ids)
    missing_streams = [
        video_id for video_id in video_ids
        if not (stream_root / f"{video_id}.mp4").is_file()
        or (stream_root / f"{video_id}.mp4").stat().st_size <= 1_000_000
    ]

    model_recall = {}
    for label in LABELS:
        metric = adaptive["per_class"][label]["adaptive_median_logit_loov"]["aggregate"]
        model_recall[label] = {
            "matched_gt": int(metric["num_matched_gt"]),
            "gt": int(metric["num_gt"]),
            "recall": float(metric["recall"]),
        }
    passed = not (missing_gt or missing_candidates or missing_database_events or missing_streams)
    report = {
        "pipeline_id": config["pipeline_id"],
        "scope": config["scope"],
        "videos": len(video_ids),
        "manifest_events": len(manifest_event_ids),
        "existing_gt": gt_total,
        "existing_gt_anchor_coverage": (gt_total - len(missing_gt)) / gt_total if gt_total else 1.0,
        "selected_candidate_anchors": len(queue_candidates),
        "selected_candidate_anchor_coverage": (
            (len(queue_candidates) - len(missing_candidates)) / len(queue_candidates)
            if queue_candidates else 1.0
        ),
        "model_dense_recall": model_recall,
        "missing_gt": missing_gt,
        "missing_candidates": missing_candidates,
        "missing_database_events": missing_database_events,
        "missing_streams": missing_streams,
        "metric_warning": config["metric_warning"],
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
