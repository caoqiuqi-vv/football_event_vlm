#!/usr/bin/env python3
"""CPU-only audit for Stage-1 ranking/localization and checkpoint updates.

The script deliberately does not instantiate the model or decode video.  It
summarizes epoch JSON files and, when requested, compares only task-trainable
weights between an init checkpoint and a final checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


LABELS = ("shot", "save", "set_piece")


def metric_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("metrics_epoch_*.json")):
        payload = json.loads(path.read_text())
        for source in ("online", "ema"):
            metrics = payload.get("metrics" if source == "online" else "ema_metrics")
            if not isinstance(metrics, dict):
                continue
            row: dict[str, Any] = {
                "epoch": int(payload["epoch"]),
                "source": source,
                "train_loss": float(payload.get("train_loss", math.nan)),
                "mAP": float(metrics["default"]["mAP"]),
                "micro_precision_at_tuned": float(metrics["tuned"]["micro_precision"]),
                "micro_recall_at_tuned": float(metrics["tuned"]["micro_recall"]),
            }
            for label in LABELS:
                sep = metrics["confidence_separation"][label]
                row[f"{label}_ap"] = float(metrics["default"]["per_class"][label]["ap"])
                row[f"{label}_threshold"] = float(metrics["thresholds"][label])
                row[f"{label}_mean_gap"] = float(sep["mean_gap"])
                row[f"{label}_positive_p10"] = float(sep["positive_p10"])
                row[f"{label}_negative_p90"] = float(sep["negative_p90"])
                row[f"{label}_tail_gap"] = float(sep["tail_gap"])
                row[f"{label}_frame_topk"] = float(
                    metrics["frame_topk_hit_rate"][label]["hit_rate"]
                )
            rows.append(row)
    return rows


def group_name(key: str) -> str | None:
    if ".lora_a" in key or ".lora_b" in key:
        return "lora_last6"
    if key.startswith("temporal."):
        return "temporal"
    if key.startswith("head."):
        return "clip_head"
    if key.startswith("frame_proj."):
        return "frame_projection"
    if key.startswith("frame_patch_attn."):
        return "patch_attention"
    if key.startswith("frame_event_head."):
        return "frame_head"
    if key.startswith("response_curve_head.") or key.startswith("response_curve_logit_"):
        return "response_curve"
    return None


def load_selected(path: Path, *, ema: bool = False) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    if ema and isinstance(checkpoint.get("model_ema_trainable"), dict):
        shadow = checkpoint["model_ema_trainable"].get("shadow", {})
        state = dict(state)
        state.update(shadow)
    selected = {
        key: value.detach().float().clone()
        for key, value in state.items()
        if group_name(key) is not None
    }
    del checkpoint, state
    return selected


def checkpoint_delta(init_path: Path, final_path: Path, *, final_ema: bool) -> dict[str, Any]:
    import torch

    initial = load_selected(init_path)
    final = load_selected(final_path, ema=final_ema)
    accum: dict[str, dict[str, float]] = {}
    for key, before in initial.items():
        after = final.get(key)
        if after is None or after.shape != before.shape:
            continue
        group = group_name(key)
        assert group is not None
        stats = accum.setdefault(
            group,
            {"num_tensors": 0.0, "num_parameters": 0.0, "delta_sq": 0.0, "base_sq": 0.0},
        )
        delta = after - before
        stats["num_tensors"] += 1
        stats["num_parameters"] += before.numel()
        stats["delta_sq"] += float(torch.sum(delta * delta))
        stats["base_sq"] += float(torch.sum(before * before))
    result: dict[str, Any] = {}
    for group, stats in accum.items():
        delta_l2 = math.sqrt(stats.pop("delta_sq"))
        base_l2 = math.sqrt(stats.pop("base_sq"))
        result[group] = {
            "num_tensors": int(stats["num_tensors"]),
            "num_parameters": int(stats["num_parameters"]),
            "delta_l2": delta_l2,
            "base_l2": base_l2,
            "relative_delta_l2": delta_l2 / max(base_l2, 1e-12),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--final-checkpoint", type=Path)
    parser.add_argument("--final-ema", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result: dict[str, Any] = {"metric_rows": metric_rows(args.run_dir)}
    if args.init_checkpoint and args.final_checkpoint:
        result["checkpoint_delta"] = checkpoint_delta(
            args.init_checkpoint, args.final_checkpoint, final_ema=args.final_ema
        )
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
