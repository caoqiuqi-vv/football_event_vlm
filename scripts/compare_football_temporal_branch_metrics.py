#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

if __package__:
    from .compare_football_checkpoint_metrics import (
        compare_candidate,
        render_markdown,
    )
else:
    from compare_football_checkpoint_metrics import (
        compare_candidate,
        render_markdown,
    )


def load_branch_metrics(
    path: Path,
    branch: str,
    mode: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    metrics_root = payload.get("metrics", payload)
    if not isinstance(metrics_root, dict):
        raise ValueError(f"Invalid metrics payload: {path}")
    branch_root = metrics_root.get("temporal_branches", {}).get(branch)
    if not isinstance(branch_root, dict):
        raise ValueError(f"No temporal_branches.{branch}: {path}")
    metrics = branch_root.get(mode)
    if not isinstance(metrics, dict) or not isinstance(
        metrics.get("per_class"), dict
    ):
        raise ValueError(
            f"No temporal_branches.{branch}.{mode}.per_class: {path}"
        )
    return {
        "path": str(path),
        "epoch": payload.get("epoch"),
        "thresholds": branch_root.get("thresholds", {}),
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare validation metrics from one temporal branch."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, nargs="+", required=True)
    parser.add_argument("--branch", choices=("uniform", "event"), required=True)
    parser.add_argument("--mode", choices=("default", "tuned"), default="tuned")
    parser.add_argument("--labels", default="shot,save,set_piece")
    parser.add_argument("--recall-tolerance-pp", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = [
        label.strip() for label in args.labels.split(",") if label.strip()
    ]
    baseline = load_branch_metrics(args.baseline, args.branch, args.mode)
    recall_tolerance = float(args.recall_tolerance_pp) / 100.0
    candidates = [
        compare_candidate(
            baseline,
            load_branch_metrics(path, args.branch, args.mode),
            labels,
            recall_tolerance,
        )
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
        "branch": args.branch,
        "mode": args.mode,
        "labels": labels,
        "recall_tolerance_pp": float(args.recall_tolerance_pp),
        "baseline": {
            "path": baseline["path"],
            "epoch": baseline["epoch"],
        },
        "candidates": candidates,
    }
    markdown = render_markdown(report)
    print(markdown, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
