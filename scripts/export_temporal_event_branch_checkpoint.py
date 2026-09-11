#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch


LABELS = ("shot", "save", "set_piece")


def resolve_event_metrics(
    metrics_payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics_root = metrics_payload.get("metrics", metrics_payload)
    if not isinstance(metrics_root, dict):
        raise ValueError("Metrics payload must contain a mapping")
    event_metrics = metrics_root.get("temporal_branches", {}).get("event")
    if not isinstance(event_metrics, dict):
        raise ValueError("Metrics payload has no temporal_branches.event")
    return metrics_root, event_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the event branch of a uniform/event temporal checkpoint"
    )
    parser.add_argument("--input", required=True, help="Source dual-temporal checkpoint")
    parser.add_argument("--output", required=True, help="Standalone event checkpoint")
    parser.add_argument(
        "--branch-metrics",
        default="",
        help="Optional eval JSON containing temporal_branches.event",
    )
    parser.add_argument(
        "--thresholds",
        nargs=3,
        type=float,
        metavar=("SHOT", "SAVE", "SET_PIECE"),
        required=True,
        help="Deployment thresholds in shot/save/set_piece order",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = torch.load(args.input, map_location="cpu", weights_only=False)
    if not isinstance(source, dict) or not isinstance(source.get("model"), dict):
        raise ValueError(f"Unsupported checkpoint: {args.input}")

    checkpoint: dict[str, Any] = copy.deepcopy(source)
    state = checkpoint["model"]
    removed = [
        key
        for key in state
        if key == "uniform_event_gate_logits"
        or key.startswith("uniform_event_gate_adapter.")
        or key.startswith("uniform_temporal.")
        or key.startswith("uniform_head.")
    ]
    checkpoint["model"] = {
        key: value for key, value in state.items() if key not in set(removed)
    }

    config = checkpoint.get("config")
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise ValueError("Checkpoint config.model is missing")
    model_config = config["model"]
    model_config["temporal_fusion"] = "event_topk_transformer"
    model_config["init_checkpoint"] = ""
    model_config["uniform_init_checkpoint"] = ""
    model_config["event_init_checkpoint"] = ""
    model_config["freeze_uniform_reference"] = False
    config["output_dir"] = str(Path(args.output).parent)

    thresholds = {
        label: float(value) for label, value in zip(LABELS, args.thresholds)
    }
    checkpoint["thresholds"] = thresholds
    if args.branch_metrics:
        metrics_payload = json.loads(Path(args.branch_metrics).read_text())
        try:
            metrics_root, event_metrics = resolve_event_metrics(metrics_payload)
        except ValueError as exc:
            raise ValueError(f"{exc} in {args.branch_metrics}") from exc
        checkpoint["metrics"] = {
            "default": event_metrics["default"],
            "tuned": event_metrics["tuned"],
            "default_threshold": float(
                metrics_root.get("default_threshold", 0.5)
            ),
            "thresholds": thresholds,
        }

    for key in ("optimizer", "scheduler", "scaler"):
        checkpoint.pop(key, None)
    checkpoint["export_provenance"] = {
        "source_checkpoint": str(args.input),
        "branch": "event",
        "removed_parameter_count": len(removed),
        "threshold_policy": "same_split_precision_at_no_e1_recall_loss",
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(
        f"exported={output} kept={len(checkpoint['model'])} "
        f"removed={len(removed)} thresholds={thresholds}",
        flush=True,
    )


if __name__ == "__main__":
    main()
