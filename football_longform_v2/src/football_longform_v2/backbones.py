"""DINO backbone construction and verified LoRA checkpoint export.

The long-video locator only consumes plain DINO state dictionaries.  This
module deliberately keeps the task-model checkpoint format at the boundary:
the supervised football checkpoint is converted once, checked against the
EMA LoRA weights that selected ``best.pt``, and then used exactly like an
official frozen DINO checkpoint.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from dinov3.hub.backbones import dinov3_vitl16


SUPPORTED_CONTEXT_ARCHS = {"dinov3_vitl16": dinov3_vitl16}


class BackboneError(RuntimeError):
    """Raised when a frozen visual backbone cannot be reproduced exactly."""


class LoRALinear(nn.Module):
    """Checkpoint-compatible copy of the football training LoRA wrapper."""

    def __init__(self, base: nn.Module, *, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if not hasattr(base, "in_features") or not hasattr(base, "out_features"):
            raise TypeError(f"LoRA target must be Linear-like, got {type(base)}")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.in_features = int(base.in_features)
        self.out_features = int(base.out_features)
        self.rank = int(rank)
        self.scaling = float(alpha) / max(self.rank, 1)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.lora_a = nn.Parameter(torch.empty(self.rank, self.in_features))
        self.lora_b = nn.Parameter(torch.zeros(self.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)

    def forward(self, inputs: Tensor) -> Tensor:
        base_output = self.base(inputs)
        update = F.linear(F.linear(self.dropout(inputs), self.lora_a), self.lora_b)
        return base_output + update * self.scaling


def build_plain_backbone(arch: str, weights: str | Path | None = None) -> nn.Module:
    """Build one supported DINO architecture, optionally strict-load weights."""
    factory = SUPPORTED_CONTEXT_ARCHS.get(str(arch))
    if factory is None:
        supported = ", ".join(sorted(SUPPORTED_CONTEXT_ARCHS))
        raise BackboneError(f"unsupported context arch {arch!r}; supported: {supported}")
    backbone = factory(pretrained=False)
    if weights is not None:
        checkpoint = torch.load(Path(weights), map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise BackboneError(f"plain backbone checkpoint is not a state dict: {weights}")
        missing, unexpected = backbone.load_state_dict(checkpoint, strict=False)
        if missing or unexpected:
            raise BackboneError(
                f"plain backbone mismatch for {weights}: missing={len(missing)} unexpected={len(unexpected)}"
            )
    return backbone


def _resolve_child(module: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    parent = module
    parts = dotted_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_checkpoint_lora(
    backbone: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    target_last_blocks: int,
    target_modules: tuple[str, ...],
) -> list[str]:
    """Inject the exact LoRA layout used by the supervised football run."""
    blocks = getattr(backbone, "blocks", None)
    if blocks is None:
        raise BackboneError("DINO backbone has no .blocks for LoRA injection")
    if target_last_blocks <= 0 or target_last_blocks > len(blocks):
        raise BackboneError(f"invalid target_last_blocks={target_last_blocks}")
    injected: list[str] = []
    for index in range(len(blocks) - target_last_blocks, len(blocks)):
        block = blocks[index]
        for target in target_modules:
            parent, attr = _resolve_child(block, target)
            previous = getattr(parent, attr)
            if not isinstance(previous, nn.Linear):
                raise BackboneError(f"expected Linear at blocks.{index}.{target}, got {type(previous)}")
            setattr(parent, attr, LoRALinear(previous, rank=rank, alpha=alpha, dropout=dropout))
            injected.append(f"blocks.{index}.{target}")
    return injected


def _as_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BackboneError(f"{label} must be a mapping")
    return value


def _checkpoint_lora_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    config = _as_mapping(checkpoint.get("config"), "checkpoint.config")
    model = _as_mapping(config.get("model"), "checkpoint.config.model")
    lora = _as_mapping(model.get("lora"), "checkpoint.config.model.lora")
    expected = {
        "enabled": True,
        "rank": 8,
        "alpha": 16.0,
        "target_last_blocks": 6,
        "target_modules": ["attn.qkv", "attn.proj"],
    }
    for key, wanted in expected.items():
        actual = lora.get(key)
        if actual != wanted:
            raise BackboneError(
                f"football checkpoint LoRA {key}={actual!r}; expected {wanted!r} for clean B"
            )
    if model.get("backbone") != "dinov3_vitl16":
        raise BackboneError(f"football checkpoint backbone={model.get('backbone')!r}, expected dinov3_vitl16")
    return lora


def _backbone_state_from_task_model(model_state: dict[str, Any]) -> dict[str, Tensor]:
    result: dict[str, Tensor] = {}
    for key, value in model_state.items():
        if key.startswith("backbone."):
            if not isinstance(value, Tensor):
                raise BackboneError(f"non-tensor backbone state: {key}")
            result[key.removeprefix("backbone.")] = value.detach().cpu()
    if not result:
        raise BackboneError("task checkpoint has no model.backbone.* tensors")
    return result


def _apply_ema_lora(
    wrapped_state: dict[str, Tensor], checkpoint: dict[str, Any]
) -> tuple[dict[str, Tensor], list[str]]:
    ema = _as_mapping(checkpoint.get("model_ema_trainable"), "checkpoint.model_ema_trainable")
    shadow = _as_mapping(ema.get("shadow"), "checkpoint.model_ema_trainable.shadow")
    expected = {key for key in wrapped_state if key.endswith((".lora_a", ".lora_b"))}
    ema_lora = {
        key.removeprefix("backbone."): value
        for key, value in shadow.items()
        if key.startswith("backbone.") and key.endswith((".lora_a", ".lora_b"))
    }
    if set(ema_lora) != expected:
        missing = sorted(expected - set(ema_lora))
        unexpected = sorted(set(ema_lora) - expected)
        raise BackboneError(
            "EMA LoRA coverage mismatch: "
            f"expected={len(expected)} got={len(ema_lora)} missing={missing[:3]} unexpected={unexpected[:3]}"
        )
    final = dict(wrapped_state)
    for key, value in ema_lora.items():
        if not isinstance(value, Tensor) or tuple(value.shape) != tuple(final[key].shape):
            raise BackboneError(f"EMA tensor mismatch for {key}")
        final[key] = value.detach().cpu()
    return final, sorted(ema_lora)


def merge_lora_state(wrapped_state: dict[str, Tensor], *, alpha: float, rank: int) -> OrderedDict[str, Tensor]:
    """Convert ``*.base`` plus A/B tensors to the canonical 368-key DINO state."""
    merged: OrderedDict[str, Tensor] = OrderedDict()
    scale = float(alpha) / float(rank)
    for key, value in wrapped_state.items():
        if key.endswith((".lora_a", ".lora_b")):
            continue
        if ".base." not in key:
            merged[key] = value.detach().cpu()
            continue
        module_name, suffix = key.split(".base.", 1)
        destination = f"{module_name}.{suffix}"
        if suffix != "weight":
            merged[destination] = value.detach().cpu()
            continue
        a_key, b_key = f"{module_name}.lora_a", f"{module_name}.lora_b"
        if a_key not in wrapped_state or b_key not in wrapped_state:
            raise BackboneError(f"missing LoRA A/B tensors for {module_name}")
        update = torch.matmul(wrapped_state[b_key].float(), wrapped_state[a_key].float()) * scale
        if tuple(update.shape) != tuple(value.shape):
            raise BackboneError(f"LoRA update shape mismatch for {module_name}")
        merged[destination] = value.detach().cpu() + update.to(dtype=value.dtype)
    leftovers = [key for key in merged if ".base." in key or ".lora_" in key]
    if leftovers:
        raise BackboneError(f"LoRA merge left non-plain keys: {leftovers[:3]}")
    return merged


@torch.inference_mode()
def export_supervised_ema_backbone(
    *,
    task_checkpoint: str | Path,
    official_weights: str | Path,
    output_path: str | Path,
    device: str = "cpu",
    verify_image_size: tuple[int, int] = (224, 384),
) -> dict[str, Any]:
    """Export clean supervised best.pt as a plain DINO and verify FP32 equivalence."""
    task_checkpoint = Path(task_checkpoint).resolve()
    official_weights = Path(official_weights).resolve()
    output_path = Path(output_path).resolve()
    checkpoint = torch.load(task_checkpoint, map_location="cpu", weights_only=False)
    checkpoint = _as_mapping(checkpoint, "football checkpoint")
    lora = _checkpoint_lora_config(checkpoint)
    model_state = _as_mapping(checkpoint.get("model"), "checkpoint.model")

    plain = build_plain_backbone("dinov3_vitl16", official_weights)
    wrapped = build_plain_backbone("dinov3_vitl16", official_weights)
    injected = inject_checkpoint_lora(
        wrapped,
        rank=int(lora["rank"]),
        alpha=float(lora["alpha"]),
        dropout=float(lora.get("dropout", 0.0)),
        target_last_blocks=int(lora["target_last_blocks"]),
        target_modules=tuple(lora["target_modules"]),
    )
    expected_modules = [
        f"blocks.{index}.{name}" for index in range(18, 24) for name in ("attn.qkv", "attn.proj")
    ]
    if injected != expected_modules:
        raise BackboneError(f"unexpected injected modules: {injected}")
    task_backbone = _backbone_state_from_task_model(model_state)
    expected_wrapped = set(wrapped.state_dict())
    if set(task_backbone) != expected_wrapped:
        raise BackboneError(
            "task backbone state mismatch: "
            f"missing={len(expected_wrapped - set(task_backbone))} unexpected={len(set(task_backbone) - expected_wrapped)}"
        )
    wrapped_state, ema_lora_keys = _apply_ema_lora(task_backbone, checkpoint)
    wrapped.load_state_dict(wrapped_state, strict=True)
    merged = merge_lora_state(wrapped_state, alpha=float(lora["alpha"]), rank=int(lora["rank"]))
    expected_plain = plain.state_dict()
    if set(merged) != set(expected_plain):
        raise BackboneError(
            "merged DINO keys mismatch: "
            f"missing={len(set(expected_plain) - set(merged))} unexpected={len(set(merged) - set(expected_plain))}"
        )
    shape_mismatch = [key for key in merged if tuple(merged[key].shape) != tuple(expected_plain[key].shape)]
    if shape_mismatch:
        raise BackboneError(f"merged DINO shape mismatch: {shape_mismatch[:3]}")
    plain.load_state_dict(merged, strict=True)

    target = torch.device(device)
    wrapped = wrapped.to(target).eval()
    plain = plain.to(target).eval()
    generator = torch.Generator(device=target.type).manual_seed(20260828)
    height, width = verify_image_size
    inputs = torch.randn((1, 3, height, width), generator=generator, device=target, dtype=torch.float32)
    wrapped_out = wrapped.forward_features(inputs)
    merged_out = plain.forward_features(inputs)
    compare_keys = ("x_norm_clstoken", "x_norm_patchtokens")
    differences: dict[str, dict[str, float | bool]] = {}
    for key in compare_keys:
        left, right = wrapped_out[key].float(), merged_out[key].float()
        max_abs = float((left - right).abs().max().cpu())
        mean_abs = float((left - right).abs().mean().cpu())
        equal = bool(torch.allclose(left, right, rtol=2e-5, atol=2e-5))
        differences[key] = {"max_abs": max_abs, "mean_abs": mean_abs, "allclose": equal}
    if not all(bool(item["allclose"]) for item in differences.values()):
        raise BackboneError(f"wrapped-vs-merged FP32 forward mismatch: {differences}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, output_path)
    return {
        "task_checkpoint": str(task_checkpoint),
        "official_base": str(official_weights),
        "output": str(output_path),
        "arch": "dinov3_vitl16",
        "output_keys": len(merged),
        "lora": {"rank": int(lora["rank"]), "alpha": float(lora["alpha"]), "ema_shadow_keys": len(ema_lora_keys)},
        "injected_modules": injected,
        "fp32_equivalence": differences,
        "output_bytes": output_path.stat().st_size,
    }
