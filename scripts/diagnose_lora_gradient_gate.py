#!/usr/bin/env python3
"""One-real-batch LoRA gradient/update gate for football event training.

This is intentionally separate from the training loop.  It reuses the real
dataset, model, forward path, masked clip BCE, optimizer and AMP settings, but
performs exactly one optimizer step and never writes a training checkpoint.
The process exits non-zero if any LoRA-B tensor has a missing/zero gradient or
does not become non-zero after the step.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

import train_football_events as football


def parse_gpu_ids(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--gpu-ids must contain at least one logical CUDA id")
    if values[0] != 0:
        raise ValueError(
            "Use logical ids after CUDA_VISIBLE_DEVICES; --gpu-ids must start at 0"
        )
    return values


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def lora_modules(model: nn.Module) -> dict[str, football.LoRALinear]:
    result = {
        name: module
        for name, module in unwrap(model).named_modules()
        if isinstance(module, football.LoRALinear)
    }
    if not result:
        raise RuntimeError("No LoRALinear modules found; check model.lora.enabled")
    return result


def tensor_norm(value: torch.Tensor | None) -> float | None:
    if value is None:
        return None
    return float(torch.linalg.vector_norm(value.detach().float()).cpu())


def snapshot_modules(
    modules: dict[str, football.LoRALinear],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, module in modules.items():
        a = module.lora_a.detach().float()
        b = module.lora_b.detach().float()
        base = module.base.weight.detach().float()
        effective = (b @ a) * float(module.scaling)
        result[name] = {
            "lora_a_norm": tensor_norm(a),
            "lora_b_norm": tensor_norm(b),
            "lora_b_abs_max": float(b.abs().max().cpu()),
            "effective_delta_norm": tensor_norm(effective),
            "base_weight_norm": tensor_norm(base),
            "effective_delta_ratio": float(
                torch.linalg.vector_norm(effective)
                / torch.linalg.vector_norm(base).clamp_min(1e-12)
            ),
            "lora_b": b.cpu().clone(),
        }
    return result


def serializable_snapshot(snapshot: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        name: {key: value for key, value in row.items() if key != "lora_b"}
        for name, row in snapshot.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hard-fail LoRA gradient/update gate on one real football batch"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--gpu-ids",
        default="0",
        help="Logical ids inside CUDA_VISIBLE_DEVICES, e.g. 0 or 0,1",
    )
    parser.add_argument(
        "--gradient-checkpointing", choices=("on", "off"), required=True
    )
    parser.add_argument("--per-gpu-batch-size", type=int, default=1)
    parser.add_argument(
        "--diagnostic-num-frames",
        type=int,
        default=4,
        help="Decode fewer real frames while retaining the checkpoint's model max_frames",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "overrides", nargs="*", help="Optional config dotlist overrides"
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this DataParallel/AMP diagnostic")
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    if torch.cuda.device_count() < len(gpu_ids):
        raise RuntimeError(
            f"Requested logical gpu_ids={gpu_ids}, visible CUDA devices={torch.cuda.device_count()}"
        )
    if args.per_gpu_batch_size <= 0 or args.diagnostic_num_frames <= 0:
        raise ValueError("batch size and diagnostic frames must be positive")

    cfg = football.load_config(args.config, args.overrides)
    football.configure_label_schema(cfg)
    football.configure_runtime_threads(cfg)
    football.seed_everything(
        int(cfg.get("seed", 42)), bool(cfg.get("deterministic", True))
    )
    cfg.device = "cuda:0"
    cfg.gpu_ids = gpu_ids
    cfg.model.gradient_checkpointing = args.gradient_checkpointing == "on"
    cfg.train.per_gpu_batch_size = args.per_gpu_batch_size
    cfg.train.grad_accum_steps = 1
    cfg.data.num_workers_per_gpu = 0
    cfg.data.num_workers = args.num_workers
    # This is a fresh one-step diagnostic.  Never import stale optimizer state.
    cfg.train.resume.enabled = False
    cfg.train.resume.checkpoint = ""

    device = torch.device("cuda:0")
    topology = football.resolve_runtime_topology(cfg, device)
    train_dataset, _val_dataset, train_records, _val_records = (
        football.prepare_datasets(cfg, use_cache=False)
    )
    # Keep the model/checkpoint temporal capacity unchanged, but make the gate
    # inexpensive enough to run three ways.  These are still decoded real frames.
    if not hasattr(train_dataset, "num_frames"):
        raise RuntimeError("Training dataset has no num_frames attribute")
    train_dataset.num_frames = min(
        int(args.diagnostic_num_frames), int(cfg.video.num_frames)
    )
    loader = football.make_loader(
        train_dataset,
        cfg,
        is_train=True,
        batch_size=args.per_gpu_batch_size * len(gpu_ids),
    )

    model = football.make_model(cfg, use_cached_features=False, device=device)
    if len(gpu_ids) > 1:
        model = nn.DataParallel(
            model, device_ids=gpu_ids, output_device=gpu_ids[0]
        )
    model.train()
    optimizer = football.build_optimizer(model, cfg)
    optimizer.zero_grad(set_to_none=True)
    modules = lora_modules(model)
    before = snapshot_modules(modules)

    batch = next(iter(loader))
    targets = batch["targets"].to(device, non_blocking=True)
    label_masks = batch["label_masks"].to(device, non_blocking=True)
    pos_weight, pos_weight_mode = football.resolve_pos_weight(train_records, cfg.train)
    pos_weight = pos_weight.to(device)

    with football.autocast_context(
        device, bool(cfg.train.amp), str(cfg.train.amp_dtype)
    ):
        outputs = football.forward_model_batch(
            model, batch, device, return_aux=True
        )
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        loss_matrix = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets.to(logits.dtype),
            pos_weight=pos_weight.to(logits.dtype),
            reduction="none",
        )
        loss = (loss_matrix * label_masks).sum() / label_masks.sum().clamp_min(1.0)

    loss.backward()
    grad_rows: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for name, module in modules.items():
        grad_a = tensor_norm(module.lora_a.grad)
        grad_b = tensor_norm(module.lora_b.grad)
        finite_a = grad_a is not None and math.isfinite(grad_a)
        finite_b = grad_b is not None and math.isfinite(grad_b)
        grad_rows[name] = {
            "lora_a_grad_norm": grad_a,
            "lora_b_grad_norm": grad_b,
            "lora_a_grad_finite": finite_a,
            "lora_b_grad_finite": finite_b,
        }
        if not finite_b or grad_b is None or grad_b <= 0.0:
            failures.append(f"{name}: lora_b grad is missing/nonfinite/zero ({grad_b})")

    grad_clip = float(cfg.train.get("grad_clip_norm", 0.0) or 0.0)
    total_grad_norm = None
    if grad_clip > 0:
        total_grad_norm = float(
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            .detach()
            .float()
            .cpu()
        )
    optimizer.step()
    after = snapshot_modules(modules)

    step_rows: dict[str, dict[str, Any]] = {}
    for name in modules:
        b_delta = after[name]["lora_b"] - before[name]["lora_b"]
        b_delta_norm = tensor_norm(b_delta)
        row = {
            **serializable_snapshot({name: after[name]})[name],
            "lora_b_step_delta_norm": b_delta_norm,
        }
        step_rows[name] = row
        if (
            b_delta_norm is None
            or not math.isfinite(b_delta_norm)
            or b_delta_norm <= 0.0
        ):
            failures.append(f"{name}: lora_b did not change after optimizer.step")
        if row["lora_b_norm"] <= 0.0 or row["effective_delta_ratio"] <= 0.0:
            failures.append(f"{name}: effective LoRA delta remains zero after step")

    result = {
        "status": "PASS" if not failures else "FAIL",
        "config": str(Path(args.config).resolve()),
        "visible_cuda_devices": torch.cuda.device_count(),
        "gpu_ids": gpu_ids,
        "data_parallel": len(gpu_ids) > 1,
        "gradient_checkpointing": bool(cfg.model.gradient_checkpointing),
        "diagnostic_num_frames": int(train_dataset.num_frames),
        "per_gpu_batch_size": args.per_gpu_batch_size,
        "global_batch_size": args.per_gpu_batch_size * len(gpu_ids),
        "runtime_topology": football.to_plain(topology),
        "loss": float(loss.detach().float().cpu()),
        "pos_weight_mode": pos_weight_mode,
        "pos_weight": pos_weight.detach().float().cpu().tolist(),
        "total_grad_norm_before_clip": total_grad_norm,
        "before": serializable_snapshot(before),
        "gradients": grad_rows,
        "after_step": step_rows,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if failures:
        raise RuntimeError(
            f"LoRA gradient gate failed ({len(failures)} failures); see {args.output}"
        )


if __name__ == "__main__":
    main()
