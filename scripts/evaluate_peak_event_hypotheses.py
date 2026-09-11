#!/usr/bin/env python
"""Evaluate multi-label temporal peak hypotheses and UI review workload."""
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
    compute_event_metrics,
    point_nms_predictions,
    pred_matches_gt,
    window_overlap_predictions,
)
from recompute_football_eval_protocols import load_gt_events, normalize_score_columns, read_csv  # noqa: E402

DEFAULT_VIDEOS = (
    "2027564888428580866", "2027572095412195330", "2027572406738604033",
    "2042125172076392450", "2042520971893485569", "2042525152494694401",
)


def parse_thresholds(raw: str, labels: Sequence[str]) -> dict[str, float]:
    values: dict[str, float] = {}
    for item in raw.split(","):
        label, value = item.split("=", 1)
        values[label.strip()] = float(value)
    missing = set(labels) - set(values)
    if missing:
        raise ValueError(f"Missing thresholds for {sorted(missing)}")
    return values


def maximum_matching(edges: list[list[int]]) -> tuple[dict[int, int], set[int]]:
    gt_to_pred: dict[int, int] = {}
    def augment(pred_idx: int, seen: set[int]) -> bool:
        for gt_idx in edges[pred_idx]:
            if gt_idx in seen:
                continue
            seen.add(gt_idx)
            old = gt_to_pred.get(gt_idx)
            if old is None or augment(old, seen):
                gt_to_pred[gt_idx] = pred_idx
                return True
        return False
    for pred_idx in range(len(edges)):
        augment(pred_idx, set())
    return {pred_idx: gt_idx for gt_idx, pred_idx in gt_to_pred.items()}, set(gt_to_pred)


def extract_peaks(rows: list[dict[str, Any]], labels: Sequence[str], thresholds: dict[str, float]) -> list[dict[str, Any]]:
    """Keep every above-threshold local maximum; labels are fully independent."""
    hypotheses: list[dict[str, Any]] = []
    ordered = sorted(rows, key=lambda row: (float(row["start_sec"]), int(row["index"])))
    for label in labels:
        scores = [float(row[f"prob_{label}"]) for row in ordered]
        threshold = thresholds[label]
        start = 0
        while start < len(ordered):
            if scores[start] < threshold:
                start += 1
                continue
            end = start
            while end + 1 < len(ordered) and scores[end + 1] >= threshold:
                end += 1
            # Local maxima within the positive run. Plateaus are represented by
            # their middle/highest-scored window, never by multiple hypotheses.
            candidates: list[int] = []
            idx = start
            while idx <= end:
                plateau_end = idx
                while plateau_end + 1 <= end and scores[plateau_end + 1] == scores[idx]:
                    plateau_end += 1
                left = scores[idx - 1] if idx > start else float("-inf")
                right = scores[plateau_end + 1] if plateau_end < end else float("-inf")
                if scores[idx] > left and scores[idx] > right:
                    candidates.append((idx + plateau_end) // 2)
                idx = plateau_end + 1
            if not candidates:
                candidates = [max(range(start, end + 1), key=lambda pos: scores[pos])]
            for peak_idx in candidates:
                row = ordered[peak_idx]
                # Retain immediate positive neighbours as bounded evidence for
                # this peak. With 10s clips / 5s stride this is at most 20s.
                support_start_idx = max(start, peak_idx - 1)
                support_end_idx = min(end, peak_idx + 1)
                hypotheses.append({
                    "label": label,
                    "score": scores[peak_idx],
                    "anchor_sec": (float(row["start_sec"]) + float(row["end_sec"])) * 0.5,
                    "start_sec": float(ordered[support_start_idx]["start_sec"]),
                    "end_sec": float(ordered[support_end_idx]["end_sec"]),
                    "peak_window_start_sec": float(row["start_sec"]),
                    "peak_window_end_sec": float(row["end_sec"]),
                    "window_index": int(row["index"]),
                    "positive_run_start_sec": float(ordered[start]["start_sec"]),
                    "positive_run_end_sec": float(ordered[end]["end_sec"]),
                })
            start = end + 1
    return sorted(hypotheses, key=lambda item: (item["anchor_sec"], item["label"]))


def evaluate_one_to_one(preds: list[dict[str, Any]], gts: list[dict[str, Any]], tolerance_sec: float) -> tuple[dict[str, Any], dict[int, int]]:
    edges = [[idx for idx, gt in enumerate(gts) if pred_matches_gt(pred, gt, tolerance_sec, matching_mode="window")] for pred in preds]
    pred_to_gt, matched_gt = maximum_matching(edges)
    tp, fp, num_gt = len(pred_to_gt), len(preds) - len(pred_to_gt), len(gts)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / num_gt if num_gt else 0.0
    return {
        "tp": tp, "fp": fp, "fn": num_gt - tp, "num_pred": len(preds), "num_gt": num_gt,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }, pred_to_gt


def make_bundles(hypotheses: list[dict[str, Any]], max_span_sec: float) -> list[dict[str, Any]]:
    """Bundle overlapping playback ranges without altering hypotheses."""
    bundles: list[dict[str, Any]] = []
    for hyp_idx, hyp in sorted(enumerate(hypotheses), key=lambda item: (item[1]["start_sec"], item[1]["end_sec"])):
        if bundles and float(hyp["start_sec"]) <= float(bundles[-1]["end_sec"]) and max(float(bundles[-1]["end_sec"]), float(hyp["end_sec"])) - float(bundles[-1]["start_sec"]) <= max_span_sec:
            bundle = bundles[-1]
            bundle["end_sec"] = max(float(bundle["end_sec"]), float(hyp["end_sec"]))
            bundle["hypothesis_indices"].append(hyp_idx)
            bundle["labels"] = sorted(set(bundle["labels"]) | {hyp["label"]})
        else:
            bundles.append({
                "bundle_index": len(bundles), "start_sec": float(hyp["start_sec"]), "end_sec": float(hyp["end_sec"]),
                "hypothesis_indices": [hyp_idx], "labels": [hyp["label"]],
            })
    return bundles


def add_totals(target: dict[str, int], item: dict[str, Any]) -> None:
    for key in ("tp", "fp", "fn", "num_pred", "num_gt"):
        target[key] += int(item[key])


def finish(item: dict[str, int]) -> dict[str, Any]:
    precision = item["tp"] / (item["tp"] + item["fp"]) if item["tp"] + item["fp"] else 0.0
    recall = (item["num_gt"] - item["fn"]) / item["num_gt"] if item["num_gt"] else 0.0
    return {**item, "precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--videos", default=",".join(DEFAULT_VIDEOS))
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--tolerance-sec", type=float, default=2.0)
    parser.add_argument("--nms-radius-sec", type=float, default=5.0)
    parser.add_argument("--bundle-max-span-sec", type=float, default=30.0)
    parser.add_argument("--exclude", default="2027572406738604033:set_piece")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    videos = [item.strip() for item in args.videos.split(",") if item.strip()]
    first = json.loads((args.run_dir / videos[0] / "summary.json").read_text())
    labels = [str(label) for label in first["labels"]]
    thresholds = parse_thresholds(args.thresholds, labels)
    excluded = {tuple(item.split(":", 1)) for item in args.exclude.split(",") if item.strip()}
    out_dir = args.output_dir or args.run_dir / "peak_event_hypotheses_tol2"
    out_dir.mkdir(parents=True, exist_ok=True)

    totals = {protocol: {label: {key: 0 for key in ("tp", "fp", "fn", "num_pred", "num_gt")} for label in labels} for protocol in ("legacy_window", "peak_hypothesis", "simple_nms")}
    per_video: list[dict[str, Any]] = []
    all_hypotheses: list[dict[str, Any]] = []
    all_bundles: list[dict[str, Any]] = []
    workload = {"review_bundles": 0, "hypothesis_decisions": 0, "confirmations": 0, "deletions": 0, "add_label_in_visible_bundle": 0, "completely_missed_gt": 0, "review_seconds": 0.0}
    total_duration = 0.0

    for video_id in videos:
        video_dir = args.run_dir / video_id
        summary = json.loads((video_dir / "summary.json").read_text())
        total_duration += float(summary.get("duration_sec") or summary.get("video_duration_sec") or 0.0)
        rows = normalize_score_columns(read_csv(video_dir / "window_predictions.csv"), labels, "prob")
        gt_all = load_gt_events(video_dir / "gt_events.csv", labels)
        windows = window_overlap_predictions(rows, labels, thresholds)
        hypotheses = extract_peaks(rows, labels, thresholds)
        hypotheses = [item for item in hypotheses if (video_id, item["label"]) not in excluded]
        nms = point_nms_predictions(rows, labels, thresholds, args.nms_radius_sec)
        for pred in nms:
            pred["start_sec"] = float(pred["support_start_sec"])
            pred["end_sec"] = float(pred["support_end_sec"])

        hyp_matches: dict[int, int] = {}
        valid_gts: list[dict[str, Any]] = []
        gt_global_index: dict[tuple[str, int], int] = {}
        for label in labels:
            if (video_id, label) in excluded:
                continue
            gts = [gt for gt in gt_all if gt["label"] == label]
            label_hyps = [(idx, item) for idx, item in enumerate(hypotheses) if item["label"] == label]
            label_nms = [item for item in nms if item["label"] == label]
            legacy = compute_event_metrics(
                [item for item in windows if item["label"] == label], gts, args.tolerance_sec,
                matching_mode="window", allow_many_predictions_per_gt=True,
            )["per_class"][label]
            legacy_item = {key: int(legacy[key]) for key in ("tp", "fp", "fn", "num_pred", "num_gt")}
            peak_item, matched = evaluate_one_to_one([item for _, item in label_hyps], gts, args.tolerance_sec)
            nms_item, _ = evaluate_one_to_one(label_nms, gts, args.tolerance_sec)
            for protocol, item in (("legacy_window", legacy_item), ("peak_hypothesis", peak_item), ("simple_nms", nms_item)):
                add_totals(totals[protocol][label], item)
                per_video.append({"video_id": video_id, "protocol": protocol, "label": label, **finish({key: int(item[key]) for key in ("tp", "fp", "fn", "num_pred", "num_gt")})})
            for local_pred, local_gt in matched.items():
                hyp_matches[label_hyps[local_pred][0]] = len(valid_gts) + local_gt
            for local_gt, gt in enumerate(gts):
                gt_global_index[(label, local_gt)] = len(valid_gts)
                valid_gts.append(gt)

        bundles = make_bundles(hypotheses, args.bundle_max_span_sec)
        visible_gt: set[int] = set()
        for gt_idx, gt in enumerate(valid_gts):
            if any(float(bundle["start_sec"]) - args.tolerance_sec <= float(gt["time_sec"]) <= float(bundle["end_sec"]) + args.tolerance_sec for bundle in bundles):
                visible_gt.add(gt_idx)
        matched_gt = set(hyp_matches.values())
        matched_hyp = set(hyp_matches)
        workload["review_bundles"] += len(bundles)
        workload["hypothesis_decisions"] += len(hypotheses)
        workload["confirmations"] += len(matched_hyp)
        workload["deletions"] += len(hypotheses) - len(matched_hyp)
        workload["add_label_in_visible_bundle"] += len(visible_gt - matched_gt)
        workload["completely_missed_gt"] += len(set(range(len(valid_gts))) - visible_gt)
        workload["review_seconds"] += sum(float(bundle["end_sec"]) - float(bundle["start_sec"]) for bundle in bundles)
        for idx, hyp in enumerate(hypotheses):
            all_hypotheses.append({"video_id": video_id, "hypothesis_index": idx, "matched": int(idx in matched_hyp), "matched_gt_time_sec": valid_gts[hyp_matches[idx]]["time_sec"] if idx in matched_hyp else "", **hyp})
        for bundle in bundles:
            all_bundles.append({"video_id": video_id, **bundle, "num_hypotheses": len(bundle["hypothesis_indices"])})

    protocols: dict[str, Any] = {}
    for protocol, per_class_raw in totals.items():
        per_class = {label: finish(values) for label, values in per_class_raw.items()}
        micro_raw = {key: sum(int(item[key]) for item in per_class_raw.values()) for key in ("tp", "fp", "fn", "num_pred", "num_gt")}
        protocols[protocol] = {"per_class": per_class, "micro": finish(micro_raw)}
    hours = total_duration / 3600.0
    workload.update({
        "video_hours": hours,
        "review_bundles_per_hour": workload["review_bundles"] / hours,
        "hypothesis_decisions_per_hour": workload["hypothesis_decisions"] / hours,
        "deletions_per_hour": workload["deletions"] / hours,
        "review_minutes_per_video_hour": workload["review_seconds"] / 60.0 / hours,
    })
    result = {
        "run_dir": str(args.run_dir), "video_ids": videos, "thresholds": thresholds,
        "tolerance_sec": args.tolerance_sec, "nms_radius_sec": args.nms_radius_sec,
        "bundle_max_span_sec": args.bundle_max_span_sec,
        "excluded_video_label_pairs": sorted([list(item) for item in excluded]),
        "definitions": {
            "legacy_window": "all matching windows count TP; one GT may yield multiple TP",
            "peak_hypothesis": "each class is independent; every above-threshold local maximum is one event hypothesis; one-to-one GT matching",
            "simple_nms": "score-ordered 5-second per-class NMS, retained for comparison",
            "review_bundle": "overlapping playback ranges grouped only for UI; hypotheses and labels remain independent",
        },
        "protocols": protocols, "workload": workload, "per_video": per_video,
    }
    (out_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for name, rows_out in (("per_video_per_class.csv", per_video), ("event_hypotheses.csv", all_hypotheses), ("review_bundles.csv", all_bundles)):
        fields = sorted({key for row in rows_out for key in row})
        with (out_dir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows_out)
    print(json.dumps({"output_dir": str(out_dir), "protocols": protocols, "workload": workload}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
