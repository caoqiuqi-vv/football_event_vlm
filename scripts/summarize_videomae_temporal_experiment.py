#!/usr/bin/env python3
"""Summarize whether VideoMAE temporal residuals improve the global branch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LABELS = ("shot", "save", "set_piece")


def branch_ap(branch: dict[str, Any], label: str) -> float:
    return float(branch["tuned"]["per_class"][label]["ap"])


def summarize_epoch(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    branches = metrics.get("temporal_branches", {})
    global_branch = branches.get("global_action")
    temporal_only_branch = branches.get("temporal_only")
    fused_branch = branches.get("global_plus_temporal")
    if global_branch is None or fused_branch is None:
        raise ValueError(
            f"{path} has no VideoMAE global_action/global_plus_temporal branches"
        )
    global_map = float(global_branch["tuned"]["mAP"])
    fused_map = float(fused_branch["tuned"]["mAP"])
    temporal_only_map = (
        float(temporal_only_branch["tuned"]["mAP"])
        if temporal_only_branch is not None
        else None
    )
    global_ap = {label: branch_ap(global_branch, label) for label in LABELS}
    fused_ap = {label: branch_ap(fused_branch, label) for label in LABELS}
    temporal_only_ap = (
        {label: branch_ap(temporal_only_branch, label) for label in LABELS}
        if temporal_only_branch is not None
        else None
    )
    frame_topk = {
        label: float(item["hit_rate"])
        for label, item in metrics.get("frame_topk_hit_rate", {}).items()
    }
    gates = {
        label: float(item["mean"])
        for label, item in metrics.get("temporal_gate_stats", {}).items()
    }
    return {
        "epoch": int(payload["epoch"]),
        "train_loss": float(payload["train_loss"]),
        "global_mAP": global_map,
        "temporal_only_mAP": temporal_only_map,
        "fused_mAP": fused_map,
        "temporal_mAP_gain": fused_map - global_map,
        "global_ap": global_ap,
        "temporal_only_ap": temporal_only_ap,
        "fused_ap": fused_ap,
        "temporal_ap_gain": {
            label: fused_ap[label] - global_ap[label] for label in LABELS
        },
        "frame_topk_hit_rate": frame_topk,
        "temporal_gate": gates,
        "early_stopping": payload.get("early_stopping"),
    }


def verdict(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"status": "no_metrics"}
    latest = rows[-1]
    action_gain = 0.5 * (
        latest["temporal_ap_gain"]["shot"]
        + latest["temporal_ap_gain"]["save"]
    )
    overall_gain = latest["temporal_mAP_gain"]
    if len(rows) < 2:
        status = "insufficient_epochs"
    elif overall_gain >= 0.005 and action_gain > 0:
        status = "positive_temporal_value"
    elif overall_gain <= -0.005 or action_gain < -0.005:
        status = "temporal_branch_hurts"
    elif abs(overall_gain) < 0.001 and abs(action_gain) < 0.001:
        status = "no_measurable_temporal_value"
    elif overall_gain > 0:
        status = "mixed_or_small_positive"
    else:
        status = "no_measurable_temporal_value"
    return {
        "status": status,
        "latest_epoch": latest["epoch"],
        "latest_temporal_mAP_gain": overall_gain,
        "latest_shot_save_mean_ap_gain": action_gain,
        "latest_global_mAP": latest["global_mAP"],
        "latest_temporal_only_mAP": latest.get("temporal_only_mAP"),
        "latest_fused_mAP": latest["fused_mAP"],
    }


def summarize_directory(output_dir: Path) -> dict[str, Any]:
    rows = []
    errors = []
    for path in sorted(output_dir.glob("metrics_epoch_*.json")):
        try:
            rows.append(summarize_epoch(path))
        except (KeyError, TypeError, ValueError) as error:
            errors.append({"path": str(path), "error": str(error)})
    rows.sort(key=lambda item: item["epoch"])
    return {
        "output_dir": str(output_dir),
        "epochs": rows,
        "verdict": verdict(rows),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()
    summary = summarize_directory(args.output_dir.resolve())
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
