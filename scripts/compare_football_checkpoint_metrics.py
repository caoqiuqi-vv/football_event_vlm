#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def load_metrics(path: Path, mode: str) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    metrics = checkpoint.get("metrics", {}).get(mode)
    if not isinstance(metrics, dict) or not isinstance(metrics.get("per_class"), dict):
        raise ValueError(f"Checkpoint has no metrics.{mode}.per_class: {path}")
    return {
        "path": str(path),
        "epoch": checkpoint.get("epoch"),
        "thresholds": checkpoint.get("thresholds", {}),
        "metrics": metrics,
    }


def compare_candidate(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    labels: list[str],
    recall_tolerance: float,
) -> dict[str, Any]:
    baseline_classes = baseline["metrics"]["per_class"]
    candidate_classes = candidate["metrics"]["per_class"]
    per_class: dict[str, Any] = {}
    for label in labels:
        if label not in baseline_classes or label not in candidate_classes:
            raise ValueError(f"Missing label={label} in baseline or candidate metrics")
        base = baseline_classes[label]
        current = candidate_classes[label]
        precision_delta = float(current["precision"]) - float(base["precision"])
        recall_delta = float(current["recall"]) - float(base["recall"])
        per_class[label] = {
            "baseline_precision": float(base["precision"]),
            "precision": float(current["precision"]),
            "precision_delta": precision_delta,
            "baseline_recall": float(base["recall"]),
            "recall": float(current["recall"]),
            "recall_delta": recall_delta,
            "recall_guard_pass": recall_delta >= -recall_tolerance,
            "ap": float(current.get("ap", 0.0)),
            "ap_delta": float(current.get("ap", 0.0)) - float(base.get("ap", 0.0)),
            "threshold": float(current.get("threshold", 0.5)),
        }
    precision_delta_mean = sum(row["precision_delta"] for row in per_class.values()) / len(per_class)
    recall_delta_mean = sum(row["recall_delta"] for row in per_class.values()) / len(per_class)
    recall_guard_pass = all(row["recall_guard_pass"] for row in per_class.values())
    return {
        "path": candidate["path"],
        "epoch": candidate["epoch"],
        "recall_guard_pass": recall_guard_pass,
        "precision_improved": precision_delta_mean > 0.0,
        "precision_delta_mean": precision_delta_mean,
        "recall_delta_mean": recall_delta_mean,
        "mAP": float(candidate["metrics"].get("mAP", 0.0)),
        "mAP_delta": float(candidate["metrics"].get("mAP", 0.0))
        - float(baseline["metrics"].get("mAP", 0.0)),
        "per_class": per_class,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"Baseline: `{report['baseline']['path']}` (epoch {report['baseline']['epoch']})",
        "",
        "| Candidate | Recall guard | Mean P delta | Mean R delta | mAP delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["candidates"]:
        lines.append(
            f"| `{row['path']}` | {'PASS' if row['recall_guard_pass'] else 'FAIL'} | "
            f"{row['precision_delta_mean'] * 100:+.2f}pp | "
            f"{row['recall_delta_mean'] * 100:+.2f}pp | {row['mAP_delta']:+.4f} |"
        )
        lines.extend(
            [
                f"| &nbsp;&nbsp;{label} | {'PASS' if metrics['recall_guard_pass'] else 'FAIL'} | "
                f"{metrics['precision_delta'] * 100:+.2f}pp | "
                f"{metrics['recall_delta'] * 100:+.2f}pp | {metrics['ap_delta']:+.4f} |"
                for label, metrics in row["per_class"].items()
            ]
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare football checkpoint validation precision/recall.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, nargs="+", required=True)
    parser.add_argument("--mode", choices=("default", "tuned"), default="tuned")
    parser.add_argument("--labels", default="shot,save")
    parser.add_argument("--recall-tolerance-pp", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    if not labels:
        raise ValueError("--labels must contain at least one label")
    baseline = load_metrics(args.baseline, args.mode)
    recall_tolerance = float(args.recall_tolerance_pp) / 100.0
    candidates = [
        compare_candidate(baseline, load_metrics(path, args.mode), labels, recall_tolerance)
        for path in args.candidates
    ]
    candidates.sort(
        key=lambda row: (
            bool(row["recall_guard_pass"]),
            float(row["precision_delta_mean"]),
            float(row["recall_delta_mean"]),
            float(row["mAP_delta"]),
        ),
        reverse=True,
    )
    report = {
        "mode": args.mode,
        "labels": labels,
        "recall_tolerance_pp": float(args.recall_tolerance_pp),
        "baseline": {"path": baseline["path"], "epoch": baseline["epoch"]},
        "candidates": candidates,
    }
    markdown = render_markdown(report)
    print(markdown, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
