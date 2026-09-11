#!/usr/bin/env python3
"""Compare the high-resolution residual output with its frozen anchor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from analyze_roi_branch_from_eval import (
    finalize_totals,
    init_totals,
    load_gt_events,
    read_csv,
)
from eval_long_video_checkpoint import compute_event_metrics, window_overlap_predictions, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--video-id-file", required=True)
    parser.add_argument("--labels", default="shot,save,set_piece")
    parser.add_argument("--match-tolerance-sec", type=float, default=3.0)
    parser.add_argument("--output", default="highres_anchor_vs_final_fixed_thresholds.json")
    return parser.parse_args()


def branch_rows(
    rows: list[dict[str, str]], labels: Sequence[str], branch: str
) -> list[dict[str, Any]]:
    score_prefix = "prob" if branch == "final" else "global_prob"
    result: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {
            "index": int(row["index"]),
            "start_sec": float(row["start_sec"]),
            "end_sec": float(row["end_sec"]),
        }
        for label in labels:
            key = f"{score_prefix}_{label}"
            if not row.get(key):
                raise RuntimeError(f"missing {key}; cannot compare {branch} branch")
            item[f"prob_{label}"] = float(row[key])
        result.append(item)
    return result


def add_metrics(
    totals: dict[str, dict[str, int]], metrics: dict[str, Any], labels: Sequence[str]
) -> None:
    for label in labels:
        source = metrics["per_class"][label]
        for key in totals[label]:
            totals[label][key] += int(source[key])


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else 0.0


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    video_ids = [x.strip() for x in Path(args.video_id_file).read_text().splitlines() if x.strip()]
    totals = {branch: init_totals(labels) for branch in ("anchor", "final")}
    per_video: list[dict[str, Any]] = []
    deltas: dict[str, list[float]] = {label: [] for label in labels}
    residuals: dict[str, list[float]] = {label: [] for label in labels}
    changed_decisions = {label: 0 for label in labels}
    thresholds_seen: list[dict[str, float]] = []

    for video_id in video_ids:
        video_dir = run_dir / video_id
        summary = json.loads((video_dir / "summary.json").read_text())
        diagnostic = summary.get("score_diagnostics", {}).get("highres_glimpse", {})
        if diagnostic.get("active") is not True:
            raise RuntimeError(f"high-resolution residual branch inactive for {video_id}: {diagnostic}")
        thresholds = {label: float(summary["thresholds"][label]) for label in labels}
        thresholds_seen.append(thresholds)
        rows = read_csv(video_dir / "window_predictions.csv")
        gt = load_gt_events(video_dir / "gt_events.csv", labels)

        for row in rows:
            for label in labels:
                anchor = float(row[f"global_prob_{label}"])
                final = float(row[f"prob_{label}"])
                residual_key = f"highres_residual_logit_{label}"
                if not row.get(residual_key):
                    raise RuntimeError(f"missing {residual_key} for {video_id}")
                residual = float(row[residual_key])
                deltas[label].append(final - anchor)
                residuals[label].append(residual)
                changed_decisions[label] += int(
                    (anchor >= thresholds[label]) != (final >= thresholds[label])
                )

        for branch in ("anchor", "final"):
            predictions = window_overlap_predictions(
                branch_rows(rows, labels, branch), labels, thresholds
            )
            metrics = compute_event_metrics(
                predictions,
                gt,
                args.match_tolerance_sec,
                matching_mode="window",
                allow_many_predictions_per_gt=True,
            )
            add_metrics(totals[branch], metrics, labels)
            for label in labels:
                item = metrics["per_class"][label]
                per_video.append({
                    "video_id": video_id,
                    "branch": branch,
                    "label": label,
                    "threshold": thresholds[label],
                    **{key: item[key] for key in (
                        "precision", "recall", "f1", "tp", "fp", "fn",
                        "num_pred", "num_gt", "num_matched_gt",
                    )},
                })

    aggregate: dict[str, Any] = {}
    aggregate_rows: list[dict[str, Any]] = []
    for branch in ("anchor", "final"):
        per_class, micro = finalize_totals(totals[branch], True)
        aggregate[branch] = {"per_class": per_class, "micro": micro}
        for label, item in per_class.items():
            aggregate_rows.append({"branch": branch, "label": label, **item})

    activity: dict[str, Any] = {}
    for label in labels:
        abs_residual = [abs(x) for x in residuals[label]]
        abs_delta = [abs(x) for x in deltas[label]]
        count = len(residuals[label])
        activity[label] = {
            "window_count": count,
            "nonzero_residual_fraction": (
                sum(x > 1e-8 for x in abs_residual) / count if count else 0.0
            ),
            "mean_residual_logit": float(np.mean(residuals[label])) if count else 0.0,
            "mean_abs_residual_logit": float(np.mean(abs_residual)) if count else 0.0,
            "p95_abs_residual_logit": percentile(abs_residual, 95),
            "max_abs_residual_logit": max(abs_residual, default=0.0),
            "mean_final_minus_anchor_prob": float(np.mean(deltas[label])) if count else 0.0,
            "mean_abs_probability_delta": float(np.mean(abs_delta)) if count else 0.0,
            "p95_abs_probability_delta": percentile(abs_delta, 95),
            "max_abs_probability_delta": max(abs_delta, default=0.0),
            "changed_threshold_decisions": changed_decisions[label],
            "changed_threshold_fraction": changed_decisions[label] / count if count else 0.0,
        }

    report = {
        "run_dir": str(run_dir),
        "video_ids": video_ids,
        "labels": labels,
        "comparison": "same dense windows, same checkpoint thresholds, anchor=global_logits, final=outputs.logits",
        "postprocess": "window_overlap",
        "match_tolerance_sec": args.match_tolerance_sec,
        "thresholds": thresholds_seen[0] if thresholds_seen else {},
        "aggregate": aggregate,
        "highres_branch_activity": activity,
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = run_dir / output
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    write_csv(output.with_name(output.stem + "_aggregate.csv"), aggregate_rows, list(aggregate_rows[0]))
    write_csv(output.with_name(output.stem + "_per_video.csv"), per_video, list(per_video[0]))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
