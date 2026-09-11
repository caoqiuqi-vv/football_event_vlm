#!/usr/bin/env python
"""Learning watchdog for the isolated Object Motion Evidence Adapter."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


PAIR = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^ ]+)")
REQUIRED = (
    "loss",
    "grad_norm_head",
    "grad_norm_ball_lora",
    "object_motion_aux_loss",
    "object_motion_heatmap_loss",
    "object_motion_distribution_loss",
    "object_motion_presence_loss",
    "object_motion_coordinate_loss",
    "object_motion_consistency_loss",
    "object_motion_frame_loss",
    "object_motion_dense_rank_loss",
    "object_motion_relation_loss",
    "object_motion_dense_rank_pairs",
    "object_motion_upward_violations",
    "object_motion_downward_violations",
    "object_motion_teacher_valid_fraction",
    "object_motion_event_residual_scale",
    "object_motion_ball_top1_hit",
    "object_motion_goal_top1_hit",
    "object_motion_person_top1_hit",
    "object_motion_ball_heatmap_loss",
    "object_motion_goal_heatmap_loss",
    "object_motion_person_heatmap_loss",
    "object_motion_ball_presence_loss",
    "object_motion_goal_presence_loss",
    "object_motion_person_presence_loss",
    "object_motion_ball_presence_precision",
    "object_motion_ball_presence_recall",
    "object_motion_goal_presence_precision",
    "object_motion_goal_presence_recall",
    "object_motion_clip_residual_abs",
    "object_motion_frame_residual_abs",
    "object_motion_clip_gate_mean",
    "object_motion_frame_gate_mean",
    "object_motion_learned_clip_gate_mean",
    "object_motion_learned_frame_gate_mean",
    "object_motion_residual_energy_loss",
    "object_motion_gate_budget_loss",
    "object_motion_saturation_loss",
    "object_motion_residual_saturation_fraction",
    "object_motion_ball_evidence_gate_mean",
    "object_motion_anchor_local_coverage",
    "object_motion_ball_strong_loss",
    "object_motion_ball_contrastive_loss",
    "object_motion_ball_track_loss",
    "object_motion_ball_preserve_loss",
    "object_motion_ball_background_cosine",
    "object_motion_ball_entropy_mean",
    "object_motion_ball_layer_weight_max",
)
TREND_KEYS = (
    "loss",
    "object_motion_aux_loss",
    "object_motion_heatmap_loss",
    "object_motion_distribution_loss",
    "object_motion_presence_loss",
    "object_motion_coordinate_loss",
    "object_motion_consistency_loss",
    "object_motion_frame_loss",
)


def parse_rows(log_path: Path) -> list[dict[str, float]]:
    if not log_path.exists():
        return []
    rows: list[dict[str, float]] = []
    for line in log_path.read_text(errors="replace").splitlines():
        if "epoch=" not in line or "step=" not in line:
            continue
        row: dict[str, float] = {}
        for key, raw in PAIR.findall(line):
            value = raw.split("/", 1)[0]
            try:
                row[key] = float(value)
            except ValueError:
                continue
        if "epoch" in row and "step" in row:
            rows.append(row)
    return rows


def mean(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else None


def build_status(output_dir: Path) -> dict[str, Any]:
    log_path = output_dir / "train_console.log"
    rows = parse_rows(log_path)
    now = time.time()
    if not rows:
        return {
            "state": "waiting",
            "healthy": None,
            "updated_at": now,
            "log": str(log_path),
            "warnings": ["no training metric rows yet"],
        }
    latest = rows[-1]
    epoch = int(latest["epoch"])
    current = [row for row in rows if int(row["epoch"]) == epoch]
    warnings: list[str] = []
    failures: list[str] = []
    missing = [key for key in REQUIRED if key not in latest]
    if missing:
        failures.append("missing metrics: " + ", ".join(missing))
    nonfinite = [
        key for key, value in latest.items() if not math.isfinite(float(value))
    ]
    if nonfinite:
        failures.append("non-finite metrics: " + ", ".join(nonfinite))
    if latest.get("object_motion_teacher_valid_fraction", 0.0) < 0.99:
        failures.append("online Teacher did not supervise every dense local frame")
    if max((row.get("object_motion_dense_rank_pairs", 0.0) for row in current[-10:]), default=0.0) <= 0:
        failures.append("relation pair queue produced no completed pair in recent rows")
    if latest.get("object_motion_anchor_local_coverage", 0.0) < 0.95:
        failures.append("object-motion full-window coverage is below 0.95")
    if latest.get("grad_norm_head", 0.0) <= 1e-8:
        failures.append("adapter gradient norm is zero")
    if latest.get("grad_norm_ball_lora", 0.0) <= 1e-8:
        failures.append("ball LoRA gradient norm is zero")
    if latest.get("object_motion_ball_background_cosine", 0.0) < 0.90:
        failures.append("non-ball feature-preserve cosine fell below 0.90")
    if latest.get("object_motion_ball_layer_weight_max", 1.0) > 0.700001:
        failures.append("ball multilayer fusion exceeded the 0.7 cap")
    for key in (
        "object_motion_learned_clip_gate_mean",
        "object_motion_learned_frame_gate_mean",
    ):
        value = latest.get(key)
        if value is not None and value >= 0.95:
            failures.append(f"{key} saturated high at {value:.6g}")
    saturation_fraction = latest.get(
        "object_motion_residual_saturation_fraction"
    )
    if saturation_fraction is not None and saturation_fraction > 0.25:
        failures.append(
            "object residual saturation fraction exceeded 0.25: "
            f"{saturation_fraction:.6g}"
        )
    if (
        len(current) >= 10
        and latest.get("object_motion_event_residual_scale", 1.0) > 0.0
        and latest.get("object_motion_clip_residual_abs", 0.0) <= 1e-8
    ):
        warnings.append("clip residual is still exactly zero after ten log intervals")
    if epoch >= 2:
        for object_name in ("ball", "goal"):
            for metric_name in ("presence_precision", "presence_recall"):
                key = f"object_motion_{object_name}_{metric_name}"
                value = latest.get(key)
                if value is not None and value < 0.50:
                    warnings.append(f"{key} remains below 0.50: {value:.6g}")
    age = now - log_path.stat().st_mtime
    if age > 300:
        failures.append(f"training log has not advanced for {age:.0f}s")

    trends: dict[str, Any] = {}
    width = min(max(len(current) // 4, 1), 50)
    if len(current) >= width * 2:
        for key in TREND_KEYS:
            first = mean([row[key] for row in current[:width] if key in row])
            last = mean([row[key] for row in current[-width:] if key in row])
            if first is None or last is None:
                continue
            trends[key] = {
                "early_mean": first,
                "recent_mean": last,
                "relative_change": (last - first) / max(abs(first), 1e-8),
            }
    if len(current) >= 20:
        for key in ("object_motion_heatmap_loss", "object_motion_distribution_loss"):
            trend = trends.get(key)
            if trend is not None and trend["relative_change"] > -0.01:
                warnings.append(
                    f"{key} improved by less than 1% in the current epoch"
                )
    per_epoch: dict[str, dict[str, float | None]] = defaultdict(dict)
    for row_epoch in sorted({int(row["epoch"]) for row in rows}):
        selected = [row for row in rows if int(row["epoch"]) == row_epoch]
        for key in TREND_KEYS:
            per_epoch[str(row_epoch)][key] = mean(
                [row[key] for row in selected if key in row]
            )

    completed = (output_dir / "training_complete.json").exists()
    state = "completed" if completed else ("unhealthy" if failures else "running")
    return {
        "state": state,
        "healthy": not failures,
        "updated_at": now,
        "log_age_sec": age,
        "rows": len(rows),
        "epoch": epoch,
        "step": int(latest["step"]),
        "latest": {key: latest.get(key) for key in REQUIRED if key in latest},
        "trends_current_epoch": trends,
        "means_by_epoch": dict(per_epoch),
        "warnings": warnings,
        "failures": failures,
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--interval-sec", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "object_motion_learning_status.json"
    while True:
        status = build_status(args.output_dir)
        atomic_json(status_path, status)
        print(json.dumps(status, ensure_ascii=False), flush=True)
        if args.once or status["state"] == "completed":
            return
        time.sleep(max(args.interval_sec, 5.0))


if __name__ == "__main__":
    main()
