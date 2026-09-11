from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Linear LoRA wrapper that keeps the original module and its buffers."""

    def __init__(self, base: nn.Module, rank: int, alpha: float, dropout: float):
        super().__init__()
        if not hasattr(base, "in_features") or not hasattr(base, "out_features"):
            raise TypeError(f"LoRA target must be linear-like, got {type(base)}")
        self.base = base
        self.in_features = int(base.in_features)
        self.out_features = int(base.out_features)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(self.rank, 1)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.lora_a = nn.Parameter(torch.empty(self.rank, self.in_features, device=base.weight.device))
        self.lora_b = nn.Parameter(torch.empty(self.out_features, self.rank, device=base.weight.device))
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    @property
    def weight(self) -> Tensor:
        return self.base.weight

    @property
    def bias(self) -> Tensor | None:
        return self.base.bias

    def reset_lora_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self, inputs: Tensor) -> Tensor:
        base_output = self.base(inputs)
        update = F.linear(F.linear(self.dropout(inputs), self.lora_a), self.lora_b)
        return base_output + update * self.scaling


def _resolve_child(module: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    parent = module
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_lora(backbone: nn.Module, cfg: Any) -> int:
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    blocks = getattr(backbone, "blocks", None)
    if blocks is None:
        raise ValueError("DINO SSL LoRA requires a ViT backbone with .blocks")
    rank = int(cfg.get("rank", 16))
    alpha = float(cfg.get("alpha", rank * 2))
    dropout = float(cfg.get("dropout", 0.0))
    target_last_blocks = int(cfg.get("target_last_blocks", 12))
    target_modules = list(
        cfg.get("target_modules", ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"])
    )
    selected = list(blocks)[-target_last_blocks:] if target_last_blocks > 0 else list(blocks)
    injected = 0
    for block in selected:
        for target_name in target_modules:
            parent, attribute = _resolve_child(block, target_name)
            original = getattr(parent, attribute)
            if isinstance(original, LoRALinear):
                continue
            setattr(parent, attribute, LoRALinear(original, rank, alpha, dropout))
            injected += 1
    if bool(cfg.get("train_norm", False)):
        for name in ("norm", "cls_norm", "local_cls_norm"):
            module = getattr(backbone, name, None)
            if module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
    return injected


def reset_lora_parameters(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, LoRALinear):
            child.reset_lora_parameters()
