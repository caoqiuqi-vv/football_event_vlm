#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.retune_all_dense_window_runs import (  # noqa: E402
    aggregate,
    load_gt_events,
    maximum_matching,
    metric,
    normalize_score_columns,
    point_nms_predictions,
    pred_matches_gt,
    read_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune per-class thresholds on a dense eval run using strict PointNMS 1:1 matching.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--video-id-file", type=Path, required=True)
    parser.add_argument("--labels", default="shot,save,set_piece")
    parser.add_argument("--match-tolerance-sec", type=float, default=5.0)
    parser.add_argument("--nms-radius-sec", type=float, default=5.0)
    parser.add_argument("--objective", choices=["f1"], default="f1")
    parser.add_argument("--exclude-setpiece-video", default="2027572406738604033")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_video_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.strip().startswith("#")]


def evaluate_label(video_data: list[dict[str, Any]], label: str, threshold: float, tolerance_sec: float, radius_sec: float) -> dict[str, Any]:
    totals = {"tp": 0, "fp": 0, "gt": 0, "pred": 0}
    labels = [label]
    thresholds = {label: threshold}
    for data in video_data:
        if data["excluded"].get(label, False):
            continue
        proposals = point_nms_predictions(data["rows"], labels, thresholds, radius_sec)
        for proposal in proposals:
            proposal["start_sec"] = float(proposal.get("support_start_sec", proposal.get("time_sec", 0.0)))
            proposal["end_sec"] = float(proposal.get("support_end_sec", proposal.get("time_sec", 0.0)))
        preds = [item for item in proposals if item["label"] == label]
        gts = [item for item in data["gts"] if item["label"] == label]
        edges = [
            [gt_idx for gt_idx, gt in enumerate(gts) if pred_matches_gt(pred, gt, tolerance_sec, matching_mode="point")]
            for pred in preds
        ]
        tp = maximum_matching(edges)
        totals["tp"] += tp
        totals["fp"] += len(preds) - tp
        totals["gt"] += len(gts)
        totals["pred"] += len(preds)
    return metric(totals["tp"], totals["fp"], totals["gt"], num_proposals=totals["pred"], threshold=threshold)


def main() -> None:
    args = parse_args()
    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    video_ids = read_video_ids(args.video_id_file)
    video_data: list[dict[str, Any]] = []
    for video_id in video_ids:
        video_dir = args.run_dir / video_id
        if not video_dir.exists():
            raise FileNotFoundError(f"Missing video eval dir: {video_dir}")
        rows = normalize_score_columns(read_csv(video_dir / "window_predictions.csv"), labels, "prob")
        gts = load_gt_events(video_dir / "gt_events.csv", labels)
        video_data.append({
            "video_id": video_id,
            "rows": rows,
            "gts": gts,
            "excluded": {"set_piece": bool(args.exclude_setpiece_video and video_id == args.exclude_setpiece_video)},
        })

    thresholds: dict[str, float] = {}
    per_class: dict[str, Any] = {}
    curves: dict[str, list[dict[str, Any]]] = {}
    for label in labels:
        values = sorted({float(row.get(f"prob_{label}", row.get(label, 0.0)) or 0.0) for data in video_data for row in data["rows"]}, reverse=True)
        candidates = [1.0000001] + values + [0.0]
        curve = [evaluate_label(video_data, label, thr, args.match_tolerance_sec, args.nms_radius_sec) for thr in candidates]
        best = max(curve, key=lambda item: (float(item["f1"]), float(item["precision"]), float(item["recall"]), float(item["threshold"])))
        thresholds[label] = float(best["threshold"])
        per_class[label] = best
        # Save a compact curve; full unique-score curves can be huge.
        sample = []
        for item in curve:
            if item["threshold"] in {thresholds[label], 0.0, 1.0000001}:
                sample.append(item)
        curves[label] = sample

    payload = {
        "protocol": "point_nms_calibration_v1",
        "run_dir": str(args.run_dir),
        "video_id_file": str(args.video_id_file),
        "labels": labels,
        "objective": args.objective,
        "match_tolerance_sec": args.match_tolerance_sec,
        "nms_radius_sec": args.nms_radius_sec,
        "exclude_setpiece_video": args.exclude_setpiece_video,
        "thresholds": thresholds,
        "threshold_string": ",".join(f"{label}={thresholds[label]:.9g}" for label in labels),
        "per_class": per_class,
        "micro": aggregate(per_class),
        "curve_samples": curves,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(payload["threshold_string"])
    print(args.output)


if __name__ == "__main__":
    main()
