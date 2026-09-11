#!/usr/bin/env python
"""Combine an SSL DINO base with an existing supervised football task head.

The supervised checkpoint supplies the temporal/frame/classification heads and
the task LoRA matrices.  All DINO base tensors (including the base tensors
wrapped by LoRA) are replaced from the merged SSL backbone.  This makes a
backbone-swap audit possible without randomly reinitializing the downstream
model or accidentally restoring the original DINO base from the task
checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch


def state_dict_from_checkpoint(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a state-dict-like checkpoint, got {type(checkpoint)}")
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, dict):
        raise TypeError("Checkpoint has no model/state_dict mapping")
    return state


def ssl_key_for_task_key(task_key: str) -> str | None:
    if not task_key.startswith("backbone."):
        return None
    relative = task_key[len("backbone.") :]
    if relative.endswith(".lora_a") or relative.endswith(".lora_b"):
        return None
    return relative.replace(".base.", ".")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-checkpoint", type=Path, required=True)
    parser.add_argument("--ssl-backbone", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    task_checkpoint = torch.load(
        args.task_checkpoint, map_location="cpu", weights_only=False
    )
    if not isinstance(task_checkpoint, dict) or "model" not in task_checkpoint:
        raise ValueError("Task checkpoint must be a full football checkpoint with model state")
    ssl_checkpoint = torch.load(
        args.ssl_backbone, map_location="cpu", weights_only=False
    )
    ssl_state = state_dict_from_checkpoint(ssl_checkpoint)
    task_state = task_checkpoint["model"]

    hybrid_state: dict[str, torch.Tensor] = {}
    replaced: list[str] = []
    preserved_lora: list[str] = []
    missing: list[str] = []
    shape_mismatch: list[str] = []
    for task_key, task_value in task_state.items():
        ssl_key = ssl_key_for_task_key(task_key)
        if ssl_key is None:
            hybrid_state[task_key] = task_value
            if task_key.startswith("backbone.") and (
                task_key.endswith(".lora_a") or task_key.endswith(".lora_b")
            ):
                preserved_lora.append(task_key)
            continue
        ssl_value = ssl_state.get(ssl_key)
        if ssl_value is None:
            missing.append(f"{task_key} <- {ssl_key}")
            hybrid_state[task_key] = task_value
            continue
        if tuple(ssl_value.shape) != tuple(task_value.shape):
            shape_mismatch.append(
                f"{task_key}: task{tuple(task_value.shape)} ssl{tuple(ssl_value.shape)}"
            )
            hybrid_state[task_key] = task_value
            continue
        hybrid_state[task_key] = ssl_value.to(dtype=task_value.dtype)
        replaced.append(task_key)

    if missing or shape_mismatch:
        details = {
            "missing": missing[:20],
            "shape_mismatch": shape_mismatch[:20],
        }
        raise RuntimeError(
            "SSL backbone did not cover the full task backbone: "
            + json.dumps(details, ensure_ascii=False)
        )

    output_checkpoint = {
        key: value
        for key, value in task_checkpoint.items()
        if key not in {"optimizer", "scheduler", "scaler"}
    }
    output_checkpoint["model"] = hybrid_state
    output_checkpoint["config"] = copy.deepcopy(task_checkpoint.get("config", {}))
    model_config = output_checkpoint["config"].setdefault("model", {})
    model_config["weights"] = str(args.ssl_backbone.resolve())
    model_config["init_checkpoint"] = ""
    model_config["init_checkpoint_strict"] = False
    output_checkpoint["hybrid_metadata"] = {
        "task_checkpoint": str(args.task_checkpoint.resolve()),
        "ssl_backbone": str(args.ssl_backbone.resolve()),
        "replaced_backbone_tensors": len(replaced),
        "preserved_task_lora_tensors": len(preserved_lora),
        "preserved_non_backbone_tensors": sum(
            not key.startswith("backbone.") for key in hybrid_state
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_checkpoint, args.output)
    report = {
        "output": str(args.output.resolve()),
        **output_checkpoint["hybrid_metadata"],
        "output_bytes": args.output.stat().st_size,
    }
    args.output.with_suffix(args.output.suffix + ".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
