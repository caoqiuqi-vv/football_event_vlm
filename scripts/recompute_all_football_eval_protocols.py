#!/usr/bin/env python
"""Recompute duplicate-neutral window and merged-proposal metrics for all runs."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

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

PROTOCOL_WINDOW = "window_overlap_duplicate_neutral"
PROTOCOL_MERGED = "merged_proposal"


def maximum_matching(edges: list[list[int]]) -> tuple[dict[int, int], set[int]]:
    """Maximum-cardinality prediction/GT bipartite matching."""
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


def evaluate_predictions(
    predictions: list[dict[str, Any]],
    gt_events: list[dict[str, Any]],
    labels: Sequence[str],
    tolerance_sec: float,
    *,
    duplicate_neutral: bool,
) -> dict[str, dict[str, int | float]]:
    per_class: dict[str, dict[str, int | float]] = {}
    for label in labels:
        preds = sorted(
            (item for item in predictions if item["label"] == label),
            key=lambda item: (-float(item["score"]), prediction_time(item)),
        )
        gts = [item for item in gt_events if item["label"] == label]
        edges = [
            [
                gt_idx
                for gt_idx, gt in enumerate(gts)
                if pred_matches_gt(pred, gt, tolerance_sec, matching_mode="window")
            ]
            for pred in preds
        ]
        pred_to_gt, matched_gt = maximum_matching(edges)
        tp = len(pred_to_gt)
        if duplicate_neutral:
            ignored = sum(
                pred_idx not in pred_to_gt and bool(matching)
                for pred_idx, matching in enumerate(edges)
            )
            fp = sum(
                pred_idx not in pred_to_gt and not matching
                for pred_idx, matching in enumerate(edges)
            )
        else:
            ignored = 0
            fp = len(preds) - tp
        fn = len(gts) - len(matched_gt)
        per_class[label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "num_gt": len(gts),
            "num_predictions": len(preds),
            "num_ignored_duplicates": ignored,
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / len(gts) if gts else 0.0,
        }
    return per_class


def aggregate(rows: list[dict[str, Any]], labels: Sequence[str], protocol: str) -> dict[str, Any]:
    per_class: dict[str, Any] = {}
    for label in labels:
        selected = [row for row in rows if row["protocol"] == protocol and row["label"] == label]
        totals = {
            key: sum(int(row[key]) for row in selected)
            for key in ("tp", "fp", "fn", "num_gt", "num_predictions", "num_ignored_duplicates")
        }
        totals["precision"] = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else 0.0
        totals["recall"] = totals["tp"] / totals["num_gt"] if totals["num_gt"] else 0.0
        per_class[label] = totals
    micro = {
        key: sum(int(item[key]) for item in per_class.values())
        for key in ("tp", "fp", "fn", "num_gt", "num_predictions", "num_ignored_duplicates")
    }
    micro["precision"] = micro["tp"] / (micro["tp"] + micro["fp"]) if micro["tp"] + micro["fp"] else 0.0
    micro["recall"] = micro["tp"] / micro["num_gt"] if micro["num_gt"] else 0.0
    return {"per_class": per_class, "micro": micro}


def discover_video_dirs(run_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in run_dir.iterdir()
        if path.is_dir()
        and (path / "window_predictions.csv").is_file()
        and (path / "gt_events.csv").is_file()
        and (path / "summary.json").is_file()
    )


def evaluate_run(
    run_dir: Path,
    video_dirs: Sequence[Path],
    tolerance_sec: float,
    merge_radius_sec: float,
    excluded: set[tuple[str, str]],
    score_prefix: str,
) -> dict[str, Any]:
    first_summary = json.loads((video_dirs[0] / "summary.json").read_text())
    labels = [str(label) for label in first_summary["labels"]]
    per_video: list[dict[str, Any]] = []
    thresholds_by_video: dict[str, dict[str, float]] = {}
    for video_dir in video_dirs:
        summary = json.loads((video_dir / "summary.json").read_text())
        current_labels = [str(label) for label in summary["labels"]]
        if current_labels != labels:
            raise ValueError(f"label mismatch: {video_dir}: {current_labels} != {labels}")
        thresholds = {label: float(summary["thresholds"][label]) for label in labels}
        thresholds_by_video[video_dir.name] = thresholds
        rows = normalize_score_columns(read_csv(video_dir / "window_predictions.csv"), labels, score_prefix)
        gt_events = load_gt_events(video_dir / "gt_events.csv", labels)

        windows = window_overlap_predictions(rows, labels, thresholds)
        proposals = point_nms_predictions(rows, labels, thresholds, merge_radius_sec)
        # NMS chooses the UI display peak, while matching uses the complete union
        # of windows absorbed into that proposal.
        for proposal in proposals:
            proposal["start_sec"] = float(proposal["support_start_sec"])
            proposal["end_sec"] = float(proposal["support_end_sec"])

        metrics = {
            PROTOCOL_WINDOW: evaluate_predictions(
                windows, gt_events, labels, tolerance_sec, duplicate_neutral=True
            ),
            PROTOCOL_MERGED: evaluate_predictions(
                proposals, gt_events, labels, tolerance_sec, duplicate_neutral=False
            ),
        }
        for protocol, per_class in metrics.items():
            for label, values in per_class.items():
                if (video_dir.name, label) in excluded:
                    continue
                per_video.append(
                    {"protocol": protocol, "video_id": video_dir.name, "label": label, **values}
                )

    protocols = {
        protocol: aggregate(per_video, labels, protocol)
        for protocol in (PROTOCOL_WINDOW, PROTOCOL_MERGED)
    }
    return {
        "run_dir": str(run_dir),
        "video_ids": [path.name for path in video_dirs],
        "num_videos": len(video_dirs),
        "labels": labels,
        "thresholds_by_video": thresholds_by_video,
        "score_prefix": score_prefix,
        "tolerance_sec": tolerance_sec,
        "merge_radius_sec": merge_radius_sec,
        "excluded_video_label_pairs": sorted([list(item) for item in excluded]),
        "definitions": {
            PROTOCOL_WINDOW: "maximum one-to-one TP matching; unmatched windows overlapping any GT are ignored, not FP",
            PROTOCOL_MERGED: "score-ordered temporal NMS; match using union support of absorbed windows; maximum one-to-one proposal/GT matching",
        },
        "protocols": protocols,
        "per_video": per_video,
    }


def write_run_outputs(result: dict[str, Any], filename: str) -> tuple[Path, Path]:
    run_dir = Path(result["run_dir"])
    output_json = run_dir / filename
    output_csv = output_json.with_suffix(".csv")
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    rows = result["per_video"]
    fieldnames = sorted({key for row in rows for key in row})
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_json, output_csv


def flatten_summary(result: dict[str, Any], protocol: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_name": Path(result["run_dir"]).name,
        "run_dir": result["run_dir"],
        "protocol": protocol,
        "num_videos": result["num_videos"],
        "video_ids": ",".join(result["video_ids"]),
        "tolerance_sec": result["tolerance_sec"],
        "merge_radius_sec": result["merge_radius_sec"],
    }
    metrics = result["protocols"][protocol]
    for label, values in metrics["per_class"].items():
        for key, value in values.items():
            row[f"{label}_{key}"] = value
    for key, value in metrics["micro"].items():
        row[f"micro_{key}"] = value
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/football_eval_runs")
    parser.add_argument("--tolerance-sec", type=float, default=2.0)
    parser.add_argument("--merge-radius-sec", type=float, default=5.0)
    parser.add_argument("--score-prefix", default="prob")
    parser.add_argument("--exclude", default="2027572406738604033:set_piece")
    parser.add_argument("--output-name", default="new_protocols_window_neutral_merged_tol2.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    excluded = {tuple(item.split(":", 1)) for item in args.exclude.split(",") if item.strip()}
    run_dirs = sorted(path for path in args.root.iterdir() if path.is_dir())
    status: list[dict[str, Any]] = []
    leaderboard: list[dict[str, Any]] = []
    for index, run_dir in enumerate(run_dirs, 1):
        video_dirs = discover_video_dirs(run_dir)
        if not video_dirs:
            status.append({"run_name": run_dir.name, "status": "skipped", "reason": "no complete per-video window predictions"})
            continue
        output_path = run_dir / args.output_name
        try:
            if output_path.exists() and not args.overwrite:
                result = json.loads(output_path.read_text())
                state = "existing"
            else:
                result = evaluate_run(
                    run_dir,
                    video_dirs,
                    args.tolerance_sec,
                    args.merge_radius_sec,
                    excluded,
                    args.score_prefix,
                )
                write_run_outputs(result, args.output_name)
                state = "completed"
            for protocol in (PROTOCOL_WINDOW, PROTOCOL_MERGED):
                leaderboard.append(flatten_summary(result, protocol))
            status.append({"run_name": run_dir.name, "status": state, "num_videos": len(video_dirs)})
            print(f"[{index}/{len(run_dirs)}] {state}: {run_dir.name} videos={len(video_dirs)}", flush=True)
        except Exception as exc:  # keep the complete audit instead of aborting all runs
            status.append({"run_name": run_dir.name, "status": "error", "reason": f"{type(exc).__name__}: {exc}"})
            print(f"[{index}/{len(run_dirs)}] error: {run_dir.name}: {type(exc).__name__}: {exc}", flush=True)

    master_csv = args.root / "all_runs_new_protocols_tol2.csv"
    fields = sorted({key for row in leaderboard for key in row})
    with master_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(leaderboard)
    audit_json = args.root / "all_runs_new_protocols_tol2_audit.json"
    audit = {
        "root": str(args.root),
        "tolerance_sec": args.tolerance_sec,
        "merge_radius_sec": args.merge_radius_sec,
        "output_name": args.output_name,
        "summary": {
            state: sum(item["status"] == state for item in status)
            for state in ("completed", "existing", "skipped", "error")
        },
        "runs": status,
    }
    audit_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"master_csv": str(master_csv), "audit_json": str(audit_json), **audit["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
