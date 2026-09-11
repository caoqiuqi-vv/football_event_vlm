#!/usr/bin/env python
"""Evaluate multi-label review segments without peak or event-NMS assumptions."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_long_video_checkpoint import compute_event_metrics, window_overlap_predictions  # noqa: E402
from recompute_football_eval_protocols import load_gt_events, normalize_score_columns, read_csv  # noqa: E402

DEFAULT_VIDEOS = (
    "2027564888428580866", "2027572095412195330", "2027572406738604033",
    "2042125172076392450", "2042520971893485569", "2042525152494694401",
)


def parse_thresholds(raw: str, labels: Sequence[str]) -> dict[str, float]:
    result = {item.split("=", 1)[0].strip(): float(item.split("=", 1)[1]) for item in raw.split(",") if item.strip()}
    if set(labels) - set(result):
        raise ValueError(f"Missing thresholds for {sorted(set(labels) - set(result))}")
    return result


def positive_windows(rows: list[dict[str, Any]], labels: Sequence[str], thresholds: dict[str, float]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        active = [label for label in labels if float(row[f"prob_{label}"]) >= thresholds[label]]
        if active:
            result.append({
                "start_sec": float(row["start_sec"]), "end_sec": float(row["end_sec"]),
                "window_index": int(row["index"]), "labels": active,
                "scores": {label: float(row[f"prob_{label}"]) for label in active},
            })
    return result


def make_review_segments(windows: list[dict[str, Any]], max_span_sec: float) -> list[dict[str, Any]]:
    """Union overlapping positive clips, then chunk long runs for honest workload."""
    if not windows:
        return []
    runs: list[dict[str, float]] = []
    for window in sorted(windows, key=lambda item: (item["start_sec"], item["end_sec"])):
        if runs and float(window["start_sec"]) <= runs[-1]["end_sec"]:
            runs[-1]["end_sec"] = max(runs[-1]["end_sec"], float(window["end_sec"]))
        else:
            runs.append({"start_sec": float(window["start_sec"]), "end_sec": float(window["end_sec"])})
    segments: list[dict[str, Any]] = []
    for run in runs:
        start = run["start_sec"]
        while start < run["end_sec"] - 1e-9:
            end = min(start + max_span_sec, run["end_sec"])
            sources = [window for window in windows if float(window["end_sec"]) > start and float(window["start_sec"]) < end]
            labels = sorted({label for window in sources for label in window["labels"]})
            scores = {label: max(float(window["scores"][label]) for window in sources if label in window["scores"]) for label in labels}
            segments.append({
                "segment_index": len(segments), "start_sec": start, "end_sec": end,
                "labels": labels, "scores": scores,
                "source_window_indices": sorted({int(window["window_index"]) for window in sources}),
            })
            start = end
    return segments


def final_counts(tp: int, fp: int, matched_gt: int, num_gt: int, num_pred: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = matched_gt / num_gt if num_gt else 0.0
    return {
        "tp_segment_labels": tp, "fp_segment_labels": fp, "num_predicted_segment_labels": num_pred,
        "num_matched_gt": matched_gt, "fn_gt": num_gt - matched_gt, "num_gt": num_gt,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--videos", default=",".join(DEFAULT_VIDEOS))
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--tolerance-sec", type=float, default=2.0)
    parser.add_argument("--max-review-segment-sec", type=float, default=30.0)
    parser.add_argument("--exclude", default="2027572406738604033:set_piece")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    videos = [item.strip() for item in args.videos.split(",") if item.strip()]
    first = json.loads((args.run_dir / videos[0] / "summary.json").read_text())
    labels = [str(label) for label in first["labels"]]
    thresholds = parse_thresholds(args.thresholds, labels)
    excluded = {tuple(item.split(":", 1)) for item in args.exclude.split(",") if item.strip()}
    out_dir = args.output_dir or args.run_dir / "multilabel_review_segments_tol2"
    out_dir.mkdir(parents=True, exist_ok=True)

    legacy_totals = {label: {"tp": 0, "fp": 0, "fn": 0, "pred": 0, "gt": 0, "matched": 0} for label in labels}
    segment_totals = {label: {"tp": 0, "fp": 0, "pred": 0, "gt": 0, "matched": 0} for label in labels}
    segment_rows: list[dict[str, Any]] = []
    per_video: list[dict[str, Any]] = []
    total_duration = 0.0
    workload = {"review_segments": 0, "segment_label_decisions": 0, "correct_label_confirmations": 0, "wrong_label_deletions": 0, "missing_gt_events_in_visible_segments": 0, "completely_missed_gt_events": 0, "false_only_review_segments": 0, "useful_review_segments": 0, "review_seconds": 0.0}

    for video_id in videos:
        video_dir = args.run_dir / video_id
        summary = json.loads((video_dir / "summary.json").read_text())
        total_duration += float(summary.get("duration_sec") or summary.get("video_duration_sec") or 0.0)
        rows = normalize_score_columns(read_csv(video_dir / "window_predictions.csv"), labels, "prob")
        gt_all = [gt for gt in load_gt_events(video_dir / "gt_events.csv", labels) if (video_id, gt["label"]) not in excluded]
        windows = positive_windows(rows, labels, thresholds)
        # Remove predictions of a known-unlabeled video/class from both metrics and workload.
        for window in windows:
            window["labels"] = [label for label in window["labels"] if (video_id, label) not in excluded]
            window["scores"] = {label: score for label, score in window["scores"].items() if label in window["labels"]}
        windows = [window for window in windows if window["labels"]]
        segments = make_review_segments(windows, args.max_review_segment_sec)

        window_preds = window_overlap_predictions(rows, labels, thresholds)
        for label in labels:
            if (video_id, label) in excluded:
                continue
            gts = [gt for gt in gt_all if gt["label"] == label]
            preds = [pred for pred in window_preds if pred["label"] == label]
            old = compute_event_metrics(preds, gts, args.tolerance_sec, matching_mode="window", allow_many_predictions_per_gt=True)["per_class"][label]
            target = legacy_totals[label]
            target["tp"] += int(old["tp"]); target["fp"] += int(old["fp"]); target["fn"] += int(old["fn"])
            target["pred"] += int(old["num_pred"]); target["gt"] += int(old["num_gt"]); target["matched"] += int(old["num_matched_gt"])

            labeled_segments = [segment for segment in segments if label in segment["labels"]]
            correct = [segment for segment in labeled_segments if any(float(segment["start_sec"]) - args.tolerance_sec <= float(gt["time_sec"]) <= float(segment["end_sec"]) + args.tolerance_sec for gt in gts)]
            matched_gt = sum(any(float(segment["start_sec"]) - args.tolerance_sec <= float(gt["time_sec"]) <= float(segment["end_sec"]) + args.tolerance_sec for segment in labeled_segments) for gt in gts)
            st = segment_totals[label]
            st["tp"] += len(correct); st["fp"] += len(labeled_segments) - len(correct); st["pred"] += len(labeled_segments); st["gt"] += len(gts); st["matched"] += matched_gt
            per_video.append({"video_id": video_id, "label": label, **final_counts(len(correct), len(labeled_segments) - len(correct), matched_gt, len(gts), len(labeled_segments))})

        workload["review_segments"] += len(segments)
        workload["segment_label_decisions"] += sum(len(segment["labels"]) for segment in segments)
        workload["review_seconds"] += sum(float(segment["end_sec"]) - float(segment["start_sec"]) for segment in segments)
        visible_gt: set[int] = set()
        correctly_covered_gt: set[int] = set()
        for segment in segments:
            visible = [idx for idx, gt in enumerate(gt_all) if float(segment["start_sec"]) - args.tolerance_sec <= float(gt["time_sec"]) <= float(segment["end_sec"]) + args.tolerance_sec]
            visible_gt.update(visible)
            if visible: workload["useful_review_segments"] += 1
            else: workload["false_only_review_segments"] += 1
            correct_labels = {gt_all[idx]["label"] for idx in visible}
            workload["correct_label_confirmations"] += len(set(segment["labels"]) & correct_labels)
            workload["wrong_label_deletions"] += len(set(segment["labels"]) - correct_labels)
            for idx in visible:
                if gt_all[idx]["label"] in segment["labels"]:
                    correctly_covered_gt.add(idx)
            segment_rows.append({"video_id": video_id, **segment, "num_labels": len(segment["labels"]), "num_visible_gt": len(visible), "visible_gt_labels": sorted({gt_all[idx]["label"] for idx in visible})})
        workload["missing_gt_events_in_visible_segments"] += len(visible_gt - correctly_covered_gt)
        workload["completely_missed_gt_events"] += len(set(range(len(gt_all))) - visible_gt)

    legacy = {}
    segment_metrics = {}
    for label in labels:
        old = legacy_totals[label]
        precision = old["tp"] / (old["tp"] + old["fp"]) if old["tp"] + old["fp"] else 0.0
        recall = old["matched"] / old["gt"] if old["gt"] else 0.0
        legacy[label] = {**old, "precision": precision, "recall": recall}
        st = segment_totals[label]
        segment_metrics[label] = final_counts(st["tp"], st["fp"], st["matched"], st["gt"], st["pred"])
    hours = total_duration / 3600.0
    total_gt = sum(item["gt"] for item in segment_totals.values())
    workload.update({
        "video_hours": hours,
        "candidate_recall": (total_gt - workload["completely_missed_gt_events"]) / total_gt,
        "review_segments_per_hour": workload["review_segments"] / hours,
        "segment_label_decisions_per_hour": workload["segment_label_decisions"] / hours,
        "wrong_label_deletions_per_hour": workload["wrong_label_deletions"] / hours,
        "review_minutes_per_video_hour": workload["review_seconds"] / 60.0 / hours,
    })
    result = {
        "run_dir": str(args.run_dir), "video_ids": videos, "thresholds": thresholds,
        "tolerance_sec": args.tolerance_sec, "max_review_segment_sec": args.max_review_segment_sec,
        "excluded_video_label_pairs": sorted([list(item) for item in excluded]),
        "definitions": {
            "legacy_window": "every positive window is scored; multiple windows covering one GT may all count TP",
            "multilabel_review_segment": "overlapping positive clips form a max-30s review segment; all active labels retained; no peaks and no event NMS; one segment may contain multiple GT events",
            "segment_label_precision": "correct predicted (segment,label) pairs / all predicted (segment,label) pairs",
            "gt_recall": "GT events covered by at least one review segment carrying the correct label / all GT events",
        },
        "legacy_window": legacy, "multilabel_review_segment": segment_metrics,
        "workload": workload, "per_video_per_class": per_video,
    }
    (out_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for name, rows_out in (("review_segments.csv", segment_rows), ("per_video_per_class.csv", per_video)):
        fields = sorted({key for row in rows_out for key in row})
        with (out_dir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows_out)
    print(json.dumps({"output_dir": str(out_dir), "legacy_window": legacy, "multilabel_review_segment": segment_metrics, "workload": workload}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
