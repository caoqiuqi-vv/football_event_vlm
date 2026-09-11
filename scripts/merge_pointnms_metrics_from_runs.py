#!/usr/bin/env python
"""Merge per-video dense artifacts from multiple eval runs and compute frozen-threshold
PointNMS 1:1 metrics without re-running model inference.

Use case (protocol v2): 35-val = cal29 run ∪ holdout6 run. Thresholds must come from
tune_pointnms_thresholds_from_dense_run.py fitted on the calibration split only.

Matching semantics are reused verbatim from scripts/retune_all_dense_window_runs.py,
so numbers are identical to the official tuning/eval pipeline.
"""
from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True,
                        help="Eval run dirs containing <video_id>/window_predictions.csv")
    parser.add_argument("--video-id-file", type=Path, required=True)
    parser.add_argument("--thresholds-json", type=Path, required=True,
                        help="JSON from tune_pointnms_thresholds_from_dense_run.py")
    parser.add_argument("--labels", default="shot,save,set_piece")
    parser.add_argument("--match-tolerance-sec", type=float, default=5.0)
    parser.add_argument("--nms-radius-sec", type=float, default=5.0)
    parser.add_argument("--exclude-setpiece-video", default="2027572406738604033")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_video_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")]


def find_video_dir(run_dirs: list[Path], video_id: str) -> Path:
    for run_dir in run_dirs:
        video_dir = run_dir / video_id
        if (video_dir / "window_predictions.csv").exists() and (video_dir / "gt_events.csv").exists():
            return video_dir
    raise FileNotFoundError(f"video {video_id} not found in any run dir: {[str(d) for d in run_dirs]}")


def evaluate_video(rows: list[dict[str, Any]], gts: list[dict[str, Any]], label: str,
                   threshold: float, tolerance_sec: float, radius_sec: float) -> dict[str, Any]:
    proposals = point_nms_predictions(rows, [label], {label: threshold}, radius_sec)
    for proposal in proposals:
        proposal["start_sec"] = float(proposal.get("support_start_sec", proposal.get("time_sec", 0.0)))
        proposal["end_sec"] = float(proposal.get("support_end_sec", proposal.get("time_sec", 0.0)))
    preds = [item for item in proposals if item["label"] == label]
    label_gts = [item for item in gts if item["label"] == label]
    edges = [
        [gt_idx for gt_idx, gt in enumerate(label_gts) if pred_matches_gt(pred, gt, tolerance_sec, matching_mode="point")]
        for pred in preds
    ]
    tp = maximum_matching(edges)
    matched_gt = {gt_idx for edge in edges for gt_idx in edge}
    # duplicate ratio: all preds within tolerance of any GT / matched GT (1:1).
    num_near_gt = sum(1 for edge in edges if edge)
    result = metric(tp, len(preds) - tp, len(label_gts), num_proposals=len(preds), threshold=threshold)
    result["num_matched_gt"] = len(matched_gt)
    result["fn"] = len(label_gts) - len(matched_gt)
    result["duplicate_ratio"] = (num_near_gt / len(matched_gt)) if matched_gt else 0.0
    return result


def main() -> None:
    args = parse_args()
    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    video_ids = read_video_ids(args.video_id_file)
    thr_payload = json.loads(args.thresholds_json.read_text())
    thresholds = {label: float(thr_payload["thresholds"][label]) for label in labels}

    per_video_rows: list[dict[str, Any]] = []
    per_class_totals: dict[str, dict[str, Any]] = {}
    total_duration_sec = 0.0
    total_pred = 0
    for video_id in video_ids:
        video_dir = find_video_dir(args.run_dirs, video_id)
        rows = normalize_score_columns(read_csv(video_dir / "window_predictions.csv"), labels, "prob")
        gts = load_gt_events(video_dir / "gt_events.csv", labels)
        try:
            duration_sec = max(float(r.get("end_sec", 0.0) or 0.0) for r in rows)
        except ValueError:
            duration_sec = 0.0
        total_duration_sec += duration_sec
        for label in labels:
            if args.exclude_setpiece_video and label == "set_piece" and video_id == args.exclude_setpiece_video:
                continue
            result = evaluate_video(rows, gts, label, thresholds[label],
                                    args.match_tolerance_sec, args.nms_radius_sec)
            result.update({"video_id": video_id, "label": label})
            per_video_rows.append(result)
            totals = per_class_totals.setdefault(label, {"tp": 0, "fp": 0, "gt": 0, "pred": 0,
                                                         "matched": 0, "near": 0.0})
            totals["tp"] += int(result["tp"])
            totals["fp"] += int(result["fp"])
            totals["gt"] += int(result["num_gt"])
            totals["pred"] += int(result["num_proposals"])
            totals["matched"] += int(result["num_matched_gt"])
            totals["near"] += float(result["duplicate_ratio"]) * int(result["num_matched_gt"])

    per_class: dict[str, Any] = {}
    for label, totals in per_class_totals.items():
        m = metric(totals["tp"], totals["fp"], totals["gt"], num_proposals=totals["pred"],
                   threshold=thresholds[label])
        m["num_matched_gt"] = totals["matched"]
        m["fn"] = totals["gt"] - totals["matched"]
        m["duplicate_ratio"] = (totals["near"] / totals["matched"]) if totals["matched"] else 0.0
        per_class[label] = m

    micro = aggregate(per_class)
    total_pred = sum(int(m["num_proposals"]) for m in per_class.values())
    payload = {
        "protocol": "point_nms_merged_v1",
        "run_dirs": [str(d) for d in args.run_dirs],
        "video_id_file": str(args.video_id_file),
        "thresholds_json": str(args.thresholds_json),
        "thresholds": thresholds,
        "labels": labels,
        "match_tolerance_sec": args.match_tolerance_sec,
        "nms_radius_sec": args.nms_radius_sec,
        "exclude_setpiece_video": args.exclude_setpiece_video,
        "num_videos": len(video_ids),
        "total_duration_sec": total_duration_sec,
        "candidates_per_hour": (total_pred / (total_duration_sec / 3600.0)) if total_duration_sec else 0.0,
        "per_class": per_class,
        "micro": micro,
        "note": "merged from saved window_predictions.csv; no model inference was re-run",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary_metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    import csv as _csv
    fieldnames = ["video_id", "label", "tp", "fp", "fn", "num_pred", "num_gt", "num_matched_gt",
                  "precision", "recall", "f1", "threshold", "duplicate_ratio"]
    with (args.output_dir / "per_video_per_class.csv").open("w", newline="") as fh:
        writer = _csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in per_video_rows:
            row = dict(row)
            row["num_pred"] = row.get("num_proposals", 0)
            writer.writerow(row)
    print(json.dumps({"micro": micro, "candidates_per_hour": payload["candidates_per_hour"]}, indent=2))
    print(args.output_dir / "summary_metrics.json")


if __name__ == "__main__":
    main()
