#!/usr/bin/env python
"""Evaluate duplicate-neutral windows and score-peak merged proposals.

Duplicate-neutral window evaluation counts at most one TP per GT. Additional
positive windows covering an already matched GT are ignored rather than counted
as either TP or FP. Merged proposals use score-ordered 5-second grouping while
retaining the union of the grouped windows as temporal support.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_long_video_checkpoint import (  # noqa: E402
    point_nms_predictions,
    pred_matches_gt,
    prediction_time,
    window_overlap_predictions,
)
from recompute_football_eval_protocols import (  # noqa: E402
    load_gt_events,
    normalize_score_columns,
    read_csv,
)


def maximum_matching(edges: list[list[int]]) -> tuple[dict[int, int], set[int]]:
    """Return a maximum prediction-to-GT bipartite matching."""
    gt_to_pred: dict[int, int] = {}

    def augment(pred_idx: int, seen_gt: set[int]) -> bool:
        for gt_idx in edges[pred_idx]:
            if gt_idx in seen_gt:
                continue
            seen_gt.add(gt_idx)
            old_pred = gt_to_pred.get(gt_idx)
            if old_pred is None or augment(old_pred, seen_gt):
                gt_to_pred[gt_idx] = pred_idx
                return True
        return False

    for pred_idx in range(len(edges)):
        augment(pred_idx, set())
    pred_to_gt = {pred_idx: gt_idx for gt_idx, pred_idx in gt_to_pred.items()}
    return pred_to_gt, set(gt_to_pred)


def duplicate_neutral_metrics(
    predictions: list[dict[str, Any]],
    gt_events: list[dict[str, Any]],
    labels: Sequence[str],
    tolerance_sec: float,
) -> dict[str, dict[str, int | float]]:
    output: dict[str, dict[str, int | float]] = {}
    for label in labels:
        preds = [item for item in predictions if item["label"] == label]
        gts = [item for item in gt_events if item["label"] == label]
        preds = sorted(preds, key=lambda item: (-float(item["score"]), prediction_time(item)))
        edges = [
            [
                idx
                for idx, gt in enumerate(gts)
                if pred_matches_gt(pred, gt, tolerance_sec, matching_mode="window")
            ]
            for pred in preds
        ]
        pred_to_gt, matched_gt = maximum_matching(edges)
        tp = len(pred_to_gt)
        ignored_duplicate = sum(1 for idx, matching in enumerate(edges) if idx not in pred_to_gt and matching)
        fp = sum(1 for idx, matching in enumerate(edges) if idx not in pred_to_gt and not matching)
        fn = len(gts) - len(matched_gt)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / len(gts) if gts else 0.0
        output[label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "num_raw_positive_windows": len(preds),
            "num_ignored_duplicate_windows": ignored_duplicate,
            "num_scored_predictions": tp + fp,
            "num_gt": len(gts),
            "precision": precision,
            "recall": recall,
        }
    return output


def merged_proposal_metrics(
    rows: list[dict[str, Any]],
    gt_events: list[dict[str, Any]],
    labels: Sequence[str],
    thresholds: dict[str, float],
    merge_radius_sec: float,
    tolerance_sec: float,
) -> dict[str, dict[str, int | float]]:
    proposals = point_nms_predictions(rows, labels, thresholds, merge_radius_sec)
    # Keep the representative peak time for display, but evaluate against the
    # union support of all windows absorbed by that peak.
    for proposal in proposals:
        proposal["start_sec"] = float(proposal["support_start_sec"])
        proposal["end_sec"] = float(proposal["support_end_sec"])

    output: dict[str, dict[str, int | float]] = {}
    for label in labels:
        preds = [item for item in proposals if item["label"] == label]
        gts = [item for item in gt_events if item["label"] == label]
        used_gt: set[int] = set()
        tp = fp = 0
        for pred in sorted(preds, key=lambda item: (-float(item["score"]), prediction_time(item))):
            matching = [
                idx
                for idx, gt in enumerate(gts)
                if idx not in used_gt
                and pred_matches_gt(pred, gt, tolerance_sec, matching_mode="window")
            ]
            if matching:
                pred_time = prediction_time(pred)
                chosen = min(matching, key=lambda idx: abs(pred_time - float(gts[idx]["time_sec"])))
                used_gt.add(chosen)
                tp += 1
            else:
                fp += 1
        fn = len(gts) - len(used_gt)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / len(gts) if gts else 0.0
        output[label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "num_proposals": len(preds),
            "num_gt": len(gts),
            "precision": precision,
            "recall": recall,
        }
    return output


def aggregate(
    rows: list[dict[str, Any]], labels: Sequence[str], protocol: str
) -> tuple[dict[str, Any], dict[str, float | int]]:
    per_class: dict[str, Any] = {}
    for label in labels:
        items = [item for item in rows if item["label"] == label and item["protocol"] == protocol]
        tp = sum(int(item["tp"]) for item in items)
        fp = sum(int(item["fp"]) for item in items)
        fn = sum(int(item["fn"]) for item in items)
        gt = sum(int(item["num_gt"]) for item in items)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / gt if gt else 0.0
        per_class[label] = {"tp": tp, "fp": fp, "fn": fn, "num_gt": gt, "precision": precision, "recall": recall}
        if protocol == "window_overlap_duplicate_neutral":
            per_class[label]["num_raw_positive_windows"] = sum(int(item["num_raw_positive_windows"]) for item in items)
            per_class[label]["num_ignored_duplicate_windows"] = sum(int(item["num_ignored_duplicate_windows"]) for item in items)
            per_class[label]["num_scored_predictions"] = tp + fp
        else:
            per_class[label]["num_proposals"] = sum(int(item["num_proposals"]) for item in items)
    tp = sum(int(item["tp"]) for item in per_class.values())
    fp = sum(int(item["fp"]) for item in per_class.values())
    fn = sum(int(item["fn"]) for item in per_class.values())
    gt = sum(int(item["num_gt"]) for item in per_class.values())
    micro = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "num_gt": gt,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / gt if gt else 0.0,
    }
    return per_class, micro


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tolerance-sec", type=float, default=0.0)
    parser.add_argument("--merge-radius-sec", type=float, default=5.0)
    parser.add_argument("--exclude", default="2027572406738604033:set_piece")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    video_dirs = sorted(path for path in args.run_dir.iterdir() if (path / "window_predictions.csv").is_file())
    if not video_dirs:
        raise FileNotFoundError(f"No completed video results under {args.run_dir}")
    summary = json.loads((video_dirs[0] / "summary.json").read_text())
    labels = [str(item) for item in summary["labels"]]
    thresholds = {label: float(summary["thresholds"][label]) for label in labels}
    excluded = {tuple(item.split(":", 1)) for item in args.exclude.split(",") if item.strip()}
    per_video: list[dict[str, Any]] = []
    for video_dir in video_dirs:
        rows = normalize_score_columns(read_csv(video_dir / "window_predictions.csv"), labels, "prob")
        gt = load_gt_events(video_dir / "gt_events.csv", labels)
        windows = window_overlap_predictions(rows, labels, thresholds)
        metrics_by_protocol = {
            "window_overlap_duplicate_neutral": duplicate_neutral_metrics(windows, gt, labels, args.tolerance_sec),
            "merged_proposal": merged_proposal_metrics(
                rows, gt, labels, thresholds, args.merge_radius_sec, args.tolerance_sec
            ),
        }
        for protocol, metrics in metrics_by_protocol.items():
            for label, values in metrics.items():
                if (video_dir.name, label) not in excluded:
                    per_video.append({"protocol": protocol, "video_id": video_dir.name, "label": label, **values})

    protocols = {}
    for protocol in ("window_overlap_duplicate_neutral", "merged_proposal"):
        per_class, micro = aggregate(per_video, labels, protocol)
        protocols[protocol] = {"per_class": per_class, "micro": micro}
    result = {
        "run_dir": str(args.run_dir),
        "thresholds": thresholds,
        "tolerance_sec": args.tolerance_sec,
        "merge_radius_sec": args.merge_radius_sec,
        "excluded_video_label_pairs": sorted([list(item) for item in excluded]),
        "definitions": {
            "window_overlap_duplicate_neutral": "one TP per GT; additional positive windows covering a matched GT are ignored, not FP",
            "merged_proposal": "score-ordered temporal grouping; grouped window union is matched one-to-one with GT",
        },
        "protocols": protocols,
        "per_video": per_video,
    }
    output = args.output or args.run_dir / "duplicate_neutral_and_merged_tol0.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    csv_path = output.with_suffix(".csv")
    keys = sorted({key for row in per_video for key in row})
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(per_video)
    print(json.dumps({"output": str(output), "per_video_csv": str(csv_path), "protocols": protocols}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
