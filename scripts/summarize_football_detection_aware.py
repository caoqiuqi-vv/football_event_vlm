#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


DEFAULT_RUNS = {
    "baseline_global": "outputs/football_eval_runs/lora_r2_detector47_baseline_global_thr0p5_nms5_tol5",
    "baseline_legacy": "outputs/football_eval_runs/lora_r2_detector47_baseline_legacy_indexed_thr0p5_nms5_tol5",
    "baseline_robust": "outputs/football_eval_runs/lora_r2_detector47_baseline_robust_crop_thr0p5_nms5_tol5",
    "trained_global": "outputs/football_eval_runs/vitl16_detector47_global_control_dense_thr0p5_nms5_tol5",
    "trained_gate": "outputs/football_eval_runs/vitl16_detector47_robust_gate_dense_thr0p5_nms5_tol5",
}
DEFAULT_CHECKPOINTS = {
    "trained_global": "outputs/football_events/vitl16_detector47_global_control/best.pt",
    "trained_gate": "outputs/football_events/vitl16_robust_dual_exp2/best.pt",
}
EXPECTED_BASELINE = {"precision": 0.1953, "recall": 0.7452, "fp": 964}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize detection-aware dense and sampled-validation acceptance metrics.")
    parser.add_argument("--output", default="outputs/football_eval_runs/detection_aware_comparison.json")
    return parser.parse_args()


def load_dense(path: str) -> dict[str, Any] | None:
    summary_path = Path(path) / "summary_metrics.json"
    if not summary_path.exists():
        return None
    payload = json.loads(summary_path.read_text())
    return {"path": str(summary_path), "micro": payload["micro"], "per_class": payload["per_class"]}


def load_sampled_checkpoint(path: str) -> dict[str, Any] | None:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        return None
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    metrics = checkpoint.get("metrics", {})
    default = metrics.get("default", {})
    return {
        "path": str(checkpoint_path),
        "epoch": checkpoint.get("epoch"),
        "macro_mAP": default.get("mAP"),
        "macro_precision": default.get("macro_precision"),
        "macro_recall": default.get("macro_recall"),
        "micro_precision": default.get("micro_precision"),
        "micro_recall": default.get("micro_recall"),
    }


def relative(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    current = candidate["micro"]
    base = baseline["micro"]
    fp_reduction = (float(base["fp"]) - float(current["fp"])) / max(float(base["fp"]), 1.0)
    return {
        "precision_delta": float(current["precision"]) - float(base["precision"]),
        "recall_delta": float(current["recall"]) - float(base["recall"]),
        "fp_reduction": fp_reduction,
        "acceptance": {
            "precision_ge_0p225": float(current["precision"]) >= 0.225,
            "fp_reduction_ge_0p15": fp_reduction >= 0.15,
            "recall_ge_0p715": float(current["recall"]) >= 0.715,
        },
    }


def main() -> None:
    args = parse_args()
    dense = {name: load_dense(path) for name, path in DEFAULT_RUNS.items()}
    sampled = {name: load_sampled_checkpoint(path) for name, path in DEFAULT_CHECKPOINTS.items()}
    comparisons: dict[str, Any] = {}
    baseline = dense.get("baseline_global")
    if baseline is not None:
        for name, item in dense.items():
            if name != "baseline_global" and item is not None:
                comparisons[f"{name}_vs_baseline_global"] = relative(item, baseline)
    baseline_reproduction = None
    if baseline is not None:
        observed = baseline["micro"]
        baseline_reproduction = {
            "expected": EXPECTED_BASELINE,
            "observed": {key: observed[key] for key in ("precision", "recall", "fp")},
            "delta": {
                "precision": float(observed["precision"]) - EXPECTED_BASELINE["precision"],
                "recall": float(observed["recall"]) - EXPECTED_BASELINE["recall"],
                "fp": int(observed["fp"]) - EXPECTED_BASELINE["fp"],
            },
        }
    global_sampled = sampled.get("trained_global")
    gate_sampled = sampled.get("trained_gate")
    if global_sampled and gate_sampled and global_sampled["macro_mAP"] is not None and gate_sampled["macro_mAP"] is not None:
        map_delta = float(gate_sampled["macro_mAP"]) - float(global_sampled["macro_mAP"])
        comparisons["trained_gate_vs_trained_global_sampled"] = {
            "macro_mAP_delta": map_delta,
            "mAP_drop_le_0p01": map_delta >= -0.01,
        }
    result = {
        "reference_baseline_reproduction": baseline_reproduction,
        "dense": dense,
        "sampled_validation": sampled,
        "comparisons": comparisons,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
