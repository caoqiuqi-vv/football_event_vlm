#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import torch


def merge_teacher_checkpoint(
    checkpoint_path: Path,
    output_path: Path,
    *,
    alpha: float,
    rank: int,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("teacher"), dict):
        raise ValueError(f"Expected a teacher checkpoint: {checkpoint_path}")
    teacher = checkpoint["teacher"]
    backbone = {key: value for key, value in teacher.items() if key.startswith("backbone.")}
    if not backbone:
        raise ValueError(f"Checkpoint has no backbone keys: {checkpoint_path}")

    scaling = float(alpha) / int(rank)
    merged: OrderedDict[str, torch.Tensor] = OrderedDict()
    merged_modules: list[str] = []
    for key, value in backbone.items():
        short_key = key.removeprefix("backbone.")
        if short_key.endswith(".lora_a") or short_key.endswith(".lora_b"):
            continue
        if ".base." not in short_key:
            merged[short_key] = value.detach().cpu()
            continue

        module_name, suffix = short_key.split(".base.", 1)
        target_key = f"{module_name}.{suffix}"
        if suffix != "weight":
            merged[target_key] = value.detach().cpu()
            continue

        prefix = f"backbone.{module_name}"
        lora_a = backbone.get(f"{prefix}.lora_a")
        lora_b = backbone.get(f"{prefix}.lora_b")
        if lora_a is None or lora_b is None:
            raise KeyError(f"Missing LoRA tensors for {module_name}")
        update = torch.matmul(lora_b.float(), lora_a.float()).mul_(scaling)
        merged[target_key] = value.detach().cpu() + update.to(dtype=value.dtype)
        merged_modules.append(module_name)

    bad_keys = [key for key in merged if ".base." in key or "lora_" in key]
    if bad_keys:
        raise RuntimeError(f"Unmerged keys remain: {bad_keys[:10]}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output_path)
    return {
        "input": str(checkpoint_path.resolve()),
        "output": str(output_path.resolve()),
        "alpha": float(alpha),
        "rank": int(rank),
        "scaling": scaling,
        "teacher_keys": len(teacher),
        "backbone_input_keys": len(backbone),
        "merged_keys": len(merged),
        "merged_modules": len(merged_modules),
        "output_bytes": output_path.stat().st_size,
    }


def validate_against_reference(output_path: Path, reference_path: Path) -> dict[str, object]:
    candidate = torch.load(output_path, map_location="cpu", weights_only=False)
    reference_obj = torch.load(reference_path, map_location="cpu", weights_only=False)
    reference = reference_obj.get("teacher", reference_obj) if isinstance(reference_obj, dict) else reference_obj
    if not isinstance(candidate, dict) or not isinstance(reference, dict):
        raise ValueError("Candidate/reference checkpoint is not a state dict")
    missing = sorted(set(reference) - set(candidate))
    unexpected = sorted(set(candidate) - set(reference))
    shape_mismatch = sorted(
        key for key in set(candidate) & set(reference)
        if tuple(candidate[key].shape) != tuple(reference[key].shape)
    )
    if missing or unexpected or shape_mismatch:
        raise RuntimeError(
            f"Merged checkpoint incompatible: missing={len(missing)} "
            f"unexpected={len(unexpected)} shape_mismatch={len(shape_mismatch)}"
        )
    return {
        "reference": str(reference_path.resolve()),
        "missing": len(missing),
        "unexpected": len(unexpected),
        "shape_mismatch": len(shape_mismatch),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge football SSL LoRA teacher into a standard DINOv3 backbone.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--rank", type=int, default=16)
    args = parser.parse_args()
    report = merge_teacher_checkpoint(args.checkpoint, args.output, alpha=args.alpha, rank=args.rank)
    if args.reference:
        report["validation"] = validate_against_reference(args.output, args.reference)
    report_path = args.output.with_suffix(args.output.suffix + ".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
