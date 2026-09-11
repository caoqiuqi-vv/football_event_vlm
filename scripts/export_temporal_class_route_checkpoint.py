#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch


LABELS = ("shot", "save", "set_piece")


def metrics_root(payload: dict[str, Any]) -> dict[str, Any]:
    root = payload.get("metrics", payload)
    if not isinstance(root, dict):
        raise ValueError("Metrics payload must contain a mapping")
    return root


def branch_thresholds(
    payload: dict[str, Any], branch: str
) -> dict[str, float]:
    root = metrics_root(payload)
    branch_metrics = root.get("temporal_branches", {}).get(branch)
    if not isinstance(branch_metrics, dict):
        raise ValueError(f"Metrics payload has no temporal_branches.{branch}")
    thresholds = branch_metrics.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError(f"Branch {branch} has no thresholds")
    return {label: float(thresholds[label]) for label in LABELS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a dual-temporal checkpoint with fixed per-class routing"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--event-metrics", required=True)
    parser.add_argument("--uniform-metrics", required=True)
    parser.add_argument("--event-labels", default="shot,save")
    parser.add_argument("--event-gate-prob", type=float, default=0.9999)
    parser.add_argument("--uniform-gate-prob", type=float, default=0.0001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    event_labels = {
        value.strip() for value in args.event_labels.split(",") if value.strip()
    }
    unknown = sorted(event_labels.difference(LABELS))
    if unknown:
        raise ValueError(f"Unknown event labels: {unknown}")
    for name, probability in (
        ("event_gate_prob", args.event_gate_prob),
        ("uniform_gate_prob", args.uniform_gate_prob),
    ):
        if not 0.0 < probability < 1.0:
            raise ValueError(f"{name} must be in (0, 1), got {probability}")

    source = torch.load(args.input, map_location="cpu", weights_only=False)
    if not isinstance(source, dict) or not isinstance(source.get("model"), dict):
        raise ValueError(f"Unsupported checkpoint: {args.input}")
    checkpoint: dict[str, Any] = copy.deepcopy(source)
    state = checkpoint["model"]
    gate_keys = [key for key in state if key.endswith("uniform_event_gate_logits")]
    if len(gate_keys) != 1:
        raise ValueError(
            f"Expected one uniform_event_gate_logits tensor, found {gate_keys}"
        )

    event_payload = json.loads(Path(args.event_metrics).read_text())
    uniform_payload = json.loads(Path(args.uniform_metrics).read_text())
    event_thresholds = branch_thresholds(event_payload, "event")
    uniform_thresholds = branch_thresholds(uniform_payload, "uniform")
    route_probabilities = [
        args.event_gate_prob if label in event_labels else args.uniform_gate_prob
        for label in LABELS
    ]
    gate_key = gate_keys[0]
    source_gate = state[gate_key]
    gate = torch.tensor(route_probabilities, dtype=torch.float32)
    state[gate_key] = torch.logit(gate).to(dtype=source_gate.dtype)

    thresholds = {
        label: (
            event_thresholds[label]
            if label in event_labels
            else uniform_thresholds[label]
        )
        for label in LABELS
    }
    checkpoint["thresholds"] = thresholds
    config = checkpoint.get("config")
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise ValueError("Checkpoint config.model is missing")
    model_config = config["model"]
    if model_config.get("temporal_fusion") != "uniform_event_dual_transformer":
        raise ValueError("Class routing requires uniform_event_dual_transformer")
    model_config["uniform_event_gate_mode"] = "static"
    model_config["uniform_event_gate_init"] = {
        label: route_probabilities[index]
        for index, label in enumerate(LABELS)
    }
    model_config["init_checkpoint"] = ""
    model_config["uniform_init_checkpoint"] = ""
    model_config["event_init_checkpoint"] = ""
    config["output_dir"] = str(Path(args.output).parent)

    for key in ("optimizer", "scheduler", "scaler"):
        checkpoint.pop(key, None)
    checkpoint["export_provenance"] = {
        "source_checkpoint": str(args.input),
        "route": {
            label: "event" if label in event_labels else "uniform"
            for label in LABELS
        },
        "gate_probabilities": dict(zip(LABELS, route_probabilities)),
        "event_metrics": str(args.event_metrics),
        "uniform_metrics": str(args.uniform_metrics),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(
        f"exported={output} gate_key={gate_key} "
        f"route={checkpoint['export_provenance']['route']} "
        f"thresholds={thresholds}",
        flush=True,
    )


if __name__ == "__main__":
    main()
