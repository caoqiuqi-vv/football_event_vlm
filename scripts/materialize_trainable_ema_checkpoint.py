#!/usr/bin/env python
"""Materialize a trainable-parameter EMA into a standalone evaluation checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.input)
    destination = Path(args.output)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Unsupported checkpoint: {source}")
    ema = checkpoint.get("model_ema_trainable")
    if not isinstance(ema, dict) or not isinstance(ema.get("shadow"), dict):
        raise ValueError(f"Checkpoint does not contain model_ema_trainable: {source}")

    model_state = dict(checkpoint["model"])
    replaced = 0
    for name, value in ema["shadow"].items():
        if name not in model_state:
            raise KeyError(f"EMA parameter is absent from model state: {name}")
        model_state[name] = value.detach().cpu()
        replaced += 1

    standalone = {
        "model": model_state,
        "epoch": checkpoint.get("epoch"),
        "pending_evaluation_epoch": checkpoint.get("pending_evaluation_epoch"),
        "best_macro_f1": checkpoint.get("best_macro_f1", -1.0),
        "label_schema": checkpoint.get("label_schema"),
        "labels": checkpoint.get("labels"),
        "thresholds": checkpoint.get("thresholds"),
        "metrics": checkpoint.get("metrics", {}),
        "config": checkpoint.get("config"),
        "ema_materialized": {
            "source": str(source.resolve()),
            "decay": ema.get("decay"),
            "num_updates": ema.get("num_updates"),
            "replaced_parameters": replaced,
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(standalone, destination)
    print(
        f"materialized_ema source={source} output={destination} "
        f"parameters={replaced} updates={ema.get('num_updates')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
