"""Isolated ball-only LoRA overlay for the object-motion DINO path."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint


class BallLoRALinear(nn.Module):
    """A switchable LoRA overlay that leaves the wrapped checkpoint path intact."""

    def __init__(self, base: nn.Module, *, rank: int = 4, alpha: float = 4.0) -> None:
        super().__init__()
        if not hasattr(base, "in_features") or not hasattr(base, "out_features"):
            raise TypeError(f"ball LoRA target must be linear-like, got {type(base)}")
        self.base = base
        for parameter in base.parameters():
            parameter.requires_grad = False
        self.in_features = int(base.in_features)
        self.out_features = int(base.out_features)
        self.rank = int(rank)
        self.scaling = float(alpha) / max(self.rank, 1)
        # Injection happens after the backbone has already been placed on its
        # runtime device. Construct overlays from the wrapped weight so a
        # CUDA/bfloat16 backbone never receives CPU/float32 LoRA tensors.
        self.ball_lora_a = nn.Parameter(
            base.weight.new_empty((self.rank, self.in_features))
        )
        self.ball_lora_b = nn.Parameter(
            base.weight.new_zeros((self.out_features, self.rank))
        )
        nn.init.kaiming_uniform_(self.ball_lora_a, a=5**0.5)
        self.enabled = False

    @property
    def weight(self) -> Tensor:
        return self.base.weight

    @property
    def bias(self) -> Tensor | None:
        return self.base.bias

    def forward(self, inputs: Tensor) -> Tensor:
        result = self.base(inputs)
        if not self.enabled:
            return result
        delta = F.linear(F.linear(inputs, self.ball_lora_a), self.ball_lora_b)
        return result + delta * self.scaling


def _resolve_child(module: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    parent = module
    parts = dotted_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_ball_lora(
    backbone: nn.Module,
    *,
    rank: int = 4,
    alpha: float = 4.0,
    last_blocks: int = 8,
    targets: Sequence[str] = ("attn.qkv", "attn.proj"),
) -> int:
    blocks = getattr(backbone, "blocks", None)
    if blocks is None or len(blocks) < last_blocks:
        raise ValueError("ball LoRA requires a ViT backbone with enough .blocks")
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    injected = 0
    for block in list(blocks)[-int(last_blocks) :]:
        for target in targets:
            parent, attribute = _resolve_child(block, target)
            current = getattr(parent, attribute)
            if isinstance(current, BallLoRALinear):
                continue
            setattr(
                parent,
                attribute,
                BallLoRALinear(current, rank=rank, alpha=alpha),
            )
            injected += 1
    return injected


def ball_lora_modules(backbone: nn.Module) -> list[BallLoRALinear]:
    return [module for module in backbone.modules() if isinstance(module, BallLoRALinear)]


@contextmanager
def ball_lora_enabled(backbone: nn.Module, enabled: bool = True) -> Iterator[None]:
    modules = ball_lora_modules(backbone)
    previous = [module.enabled for module in modules]
    try:
        for module in modules:
            module.enabled = bool(enabled)
        yield
    finally:
        for module, state in zip(modules, previous):
            module.enabled = state


def extract_intermediate_patch_layers(
    owner: nn.Module,
    inputs: Tensor,
    *,
    layers: Sequence[int],
    chunk_size: int = 1,
    checkpoint_trainable_blocks: bool = False,
) -> Tensor:
    """Return normalized DINO patches as [B,T,L,P,D]."""
    backbone = owner.backbone
    processed = owner.preprocess_inputs(inputs)
    batch, frames, channels, height, width = processed.shape
    flat = processed.reshape(batch * frames, channels, height, width)
    chunks: list[Tensor] = []
    resolved_layers = tuple(int(value) for value in layers)
    for chunk in flat.split(max(int(chunk_size), 1), dim=0):
        outputs = (
            _checkpointed_intermediate_patches(backbone, chunk, resolved_layers)
            if checkpoint_trainable_blocks
            else backbone.get_intermediate_layers(
                chunk,
                n=resolved_layers,
                return_class_token=False,
                norm=True,
            )
        )
        if len(outputs) != len(layers) or not all(torch.is_tensor(value) for value in outputs):
            raise RuntimeError("DINO intermediate-layer interface returned an unexpected structure")
        chunks.append(torch.stack(list(outputs), dim=1))
    stacked = torch.cat(chunks, dim=0)
    return stacked.reshape(batch, frames, len(layers), stacked.shape[-2], stacked.shape[-1])


def _checkpointed_intermediate_patches(
    backbone: nn.Module,
    inputs: Tensor,
    layers: tuple[int, ...],
) -> tuple[Tensor, ...]:
    """Capture DINO patches while recomputing only trainable blocks."""
    required = ("prepare_tokens_with_masks", "blocks", "norm", "n_storage_tokens")
    missing = [name for name in required if not hasattr(backbone, name)]
    if missing:
        raise ValueError(
            "Ball-LoRA activation checkpointing requires a DINO ViT backbone; "
            f"missing={missing}"
        )
    requested = set(layers)
    if len(requested) != len(layers):
        raise ValueError("ball feature layers must be unique")
    tokens, (height, width) = backbone.prepare_tokens_with_masks(inputs)
    captured: dict[int, Tensor] = {}
    rope = (
        backbone.rope_embed(H=height, W=width)
        if getattr(backbone, "rope_embed", None) is not None
        else None
    )
    for index, block in enumerate(backbone.blocks):
        if any(parameter.requires_grad for parameter in block.parameters()):
            def run_block(
                value: Tensor,
                module: nn.Module = block,
                rope_value: object = rope,
            ) -> Tensor:
                # Backward recomputation happens after the outer BallLoRA
                # context has exited. Re-enable the overlay locally so the
                # checkpointed forward and recompute graphs are identical.
                with ball_lora_enabled(module, True):
                    return module(value, rope_value)

            tokens = activation_checkpoint(
                run_block,
                tokens,
                use_reentrant=False,
                preserve_rng_state=True,
            )
        else:
            tokens = block(tokens, rope)
        if index in requested:
            captured[index] = tokens
    if set(captured) != requested:
        raise ValueError(
            f"captured layers={sorted(captured)} expected={sorted(requested)}"
        )
    result: list[Tensor] = []
    for index in layers:
        value = captured[index]
        if bool(getattr(backbone, "untie_cls_and_patch_norms", False)):
            value = backbone.norm(
                value[:, int(backbone.n_storage_tokens) + 1 :]
            )
        else:
            value = backbone.norm(value)
            value = value[:, int(backbone.n_storage_tokens) + 1 :]
        result.append(value)
    return tuple(result)


def validate_trainable_allowlist(model: nn.Module) -> tuple[list[str], list[str]]:
    """Fail if anything outside ball LoRA and the new adapter is trainable."""
    ball_names: list[str] = []
    adapter_names: list[str] = []
    illegal: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        normalized = name.removeprefix("module.")
        if normalized.startswith("backbone.") and normalized.endswith(("ball_lora_a", "ball_lora_b")):
            ball_names.append(normalized)
        elif normalized.startswith("object_motion_adapter."):
            adapter_names.append(normalized)
        else:
            illegal.append(normalized)
    if illegal or not ball_names or not adapter_names:
        raise RuntimeError(
            f"invalid object-motion trainable allowlist ball={len(ball_names)} "
            f"adapter={len(adapter_names)} illegal={illegal[:8]}"
        )
    return ball_names, adapter_names

