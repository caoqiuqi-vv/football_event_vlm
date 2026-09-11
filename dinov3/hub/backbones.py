# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from __future__ import annotations

import os
import torch
from enum import Enum
from typing import List, Optional, Union
from urllib.parse import urlparse
from pathlib import Path

from .utils import _DINOV3_BASE_URL, _safe_load_state_dict_from_url


class Weights(Enum):
    LVD1689M = "LVD1689M"
    SAT493M = "SAT493M"


def is_url(path: str) -> bool:
    parsed = urlparse(path)
    return parsed.scheme in ("https", "file")


def convert_path_or_url_to_url(path: str) -> str:
    if is_url(path):
        return path
    return Path(path).expanduser().resolve().as_uri()


def _is_hf_dinov3_vit_state_dict(state_dict: dict) -> bool:
    return "embeddings.cls_token" in state_dict and any(
        key.startswith("layer.0.attention.q_proj.") for key in state_dict
    )


def _convert_hf_dinov3_vit_state_dict(state_dict: dict, model) -> dict:
    converted = {}
    model_state = model.state_dict()

    direct_keys = {
        "embeddings.cls_token": "cls_token",
        "embeddings.mask_token": "mask_token",
        "embeddings.register_tokens": "storage_tokens",
        "embeddings.patch_embeddings.weight": "patch_embed.proj.weight",
        "embeddings.patch_embeddings.bias": "patch_embed.proj.bias",
        "norm.weight": "norm.weight",
        "norm.bias": "norm.bias",
    }
    for source, target in direct_keys.items():
        if source in state_dict:
            tensor = state_dict[source]
            if target in model_state and tensor.shape != model_state[target].shape:
                tensor = tensor.reshape(model_state[target].shape)
            converted[target] = tensor

    for target in model_state:
        if target == "rope_embed.periods":
            converted[target] = model_state[target]
        elif target.endswith("attn.qkv.bias_mask"):
            bias_mask = torch.ones_like(model_state[target])
            width = bias_mask.shape[0]
            bias_mask[width // 3 : 2 * width // 3].zero_()
            converted[target] = bias_mask

    num_blocks = len(getattr(model, "blocks", []))
    for idx in range(num_blocks):
        source_prefix = f"layer.{idx}."
        target_prefix = f"blocks.{idx}."
        q_weight = state_dict[f"{source_prefix}attention.q_proj.weight"]
        k_weight = state_dict[f"{source_prefix}attention.k_proj.weight"]
        v_weight = state_dict[f"{source_prefix}attention.v_proj.weight"]
        converted[f"{target_prefix}attn.qkv.weight"] = torch.cat([q_weight, k_weight, v_weight], dim=0)

        q_bias = state_dict.get(f"{source_prefix}attention.q_proj.bias")
        k_bias = state_dict.get(f"{source_prefix}attention.k_proj.bias")
        v_bias = state_dict.get(f"{source_prefix}attention.v_proj.bias")
        if q_bias is not None or k_bias is not None or v_bias is not None:
            if q_bias is None:
                q_bias = torch.zeros(q_weight.shape[0], dtype=q_weight.dtype)
            if k_bias is None:
                k_bias = torch.zeros(k_weight.shape[0], dtype=k_weight.dtype)
            if v_bias is None:
                v_bias = torch.zeros(v_weight.shape[0], dtype=v_weight.dtype)
            converted[f"{target_prefix}attn.qkv.bias"] = torch.cat([q_bias, k_bias, v_bias], dim=0)

        block_direct = {
            "attention.o_proj.weight": "attn.proj.weight",
            "attention.o_proj.bias": "attn.proj.bias",
            "layer_scale1.lambda1": "ls1.gamma",
            "layer_scale2.lambda1": "ls2.gamma",
            "norm1.weight": "norm1.weight",
            "norm1.bias": "norm1.bias",
            "norm2.weight": "norm2.weight",
            "norm2.bias": "norm2.bias",
            "mlp.gate_proj.weight": "mlp.w1.weight",
            "mlp.gate_proj.bias": "mlp.w1.bias",
            "mlp.up_proj.weight": "mlp.w2.weight",
            "mlp.up_proj.bias": "mlp.w2.bias",
            "mlp.down_proj.weight": "mlp.w3.weight",
            "mlp.down_proj.bias": "mlp.w3.bias",
        }
        for source_suffix, target_suffix in block_direct.items():
            source = f"{source_prefix}{source_suffix}"
            if source in state_dict:
                converted[f"{target_prefix}{target_suffix}"] = state_dict[source]

    return converted


def _make_dinov3_vit_model_arch(
    *,
    patch_size: int = 16,
    compact_arch_name: str = "vitb",
):
    if "plus" in compact_arch_name:
        model_arch = compact_arch_name.replace("plus", f"{patch_size}plus")
    else:
        model_arch = f"{compact_arch_name}{patch_size}"
    return model_arch


def _make_dinov3_vit_model_url(
    *,
    patch_size: int = 16,
    compact_arch_name: str = "vitb",
    version: Optional[str] = None,
    weights: Union[Weights, str] = Weights.LVD1689M,
    hash: Optional[str] = None,
):
    model_name = "dinov3"
    model_arch = _make_dinov3_vit_model_arch(patch_size=patch_size, compact_arch_name=compact_arch_name)
    version_suffix = f"_{version}" if version else ""
    weights_name = weights.value.lower()
    hash_suffix = f"-{hash}" if hash else ""
    model_dir = f"{model_name}_{model_arch}"
    model_filename = f"{model_name}_{model_arch}_pretrain_{weights_name}{version_suffix}{hash_suffix}.pth"
    return os.path.join(_DINOV3_BASE_URL, model_dir, model_filename)


def _make_dinov3_vit(
    *,
    img_size: int = 224,
    patch_size: int = 16,
    in_chans: int = 3,
    compact_arch_name: str = "vitb",
    pos_embed_rope_base: float = 100.0,
    pos_embed_rope_min_period: float | None = None,
    pos_embed_rope_max_period: float | None = None,
    pos_embed_rope_normalize_coords: str = "separate",
    pos_embed_rope_shift_coords: float | None = None,
    pos_embed_rope_jitter_coords: float | None = None,
    pos_embed_rope_rescale_coords: float | None = None,
    pos_embed_rope_dtype: str = "fp32",
    embed_dim: int = 768,
    depth: int = 12,
    num_heads: int = 12,
    ffn_ratio: float = 4.0,
    qkv_bias: bool = True,
    drop_path_rate: float = 0.0,
    layerscale_init: float | None = None,
    norm_layer: str = "layernorm",
    ffn_layer: str = "mlp",
    ffn_bias: bool = True,
    proj_bias: bool = True,
    n_storage_tokens: int = 0,
    mask_k_bias: bool = False,
    pretrained: bool = True,
    version: Optional[str] = None,
    weights: Union[Weights, str] = Weights.LVD1689M,
    hash: Optional[str] = None,
    check_hash: bool = False,
    **kwargs,
):
    from ..models.vision_transformer import DinoVisionTransformer

    vit_kwargs = dict(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        pos_embed_rope_base=pos_embed_rope_base,
        pos_embed_rope_min_period=pos_embed_rope_min_period,
        pos_embed_rope_max_period=pos_embed_rope_max_period,
        pos_embed_rope_normalize_coords=pos_embed_rope_normalize_coords,
        pos_embed_rope_shift_coords=pos_embed_rope_shift_coords,
        pos_embed_rope_jitter_coords=pos_embed_rope_jitter_coords,
        pos_embed_rope_rescale_coords=pos_embed_rope_rescale_coords,
        pos_embed_rope_dtype=pos_embed_rope_dtype,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        ffn_ratio=ffn_ratio,
        qkv_bias=qkv_bias,
        drop_path_rate=drop_path_rate,
        layerscale_init=layerscale_init,
        norm_layer=norm_layer,
        ffn_layer=ffn_layer,
        ffn_bias=ffn_bias,
        proj_bias=proj_bias,
        n_storage_tokens=n_storage_tokens,
        mask_k_bias=mask_k_bias,
    )
    vit_kwargs.update(**kwargs)
    model = DinoVisionTransformer(**vit_kwargs)
    if pretrained:
        if type(weights) is Weights and weights not in {Weights.LVD1689M, Weights.SAT493M}:
            raise ValueError(f"Unsupported weights for the backbone: {weights}")
        elif type(weights) is Weights:
            url = _make_dinov3_vit_model_url(
                patch_size=patch_size,
                compact_arch_name=compact_arch_name,
                version=version,
                weights=weights,
                hash=hash,
            )
        else:
            url = convert_path_or_url_to_url(weights)
        state_dict = _safe_load_state_dict_from_url(url, map_location="cpu", check_hash=check_hash)
        if _is_hf_dinov3_vit_state_dict(state_dict):
            state_dict = _convert_hf_dinov3_vit_state_dict(state_dict, model)
        model.load_state_dict(state_dict, strict=True)
    else:
        model.init_weights()
    return model


def _make_dinov3_convnext_model_url(
    *,
    compact_arch_name: str = "convnext_base",
    weights: Union[Weights, str] = Weights.LVD1689M,
    hash: Optional[str] = None,
):
    model_name = "dinov3"
    weights_name = weights.value.lower()
    hash_suffix = f"-{hash}" if hash else ""

    model_dir = f"{model_name}_{compact_arch_name}"
    model_filename = f"{model_name}_{compact_arch_name}_pretrain_{weights_name}{hash_suffix}.pth"
    return os.path.join(_DINOV3_BASE_URL, model_dir, model_filename)


def _make_dinov3_convnext(
    in_chans: int = 3,
    depths: List[int] = [3, 3, 27, 3],
    dims: List[int] = [128, 256, 512, 1024],
    compact_arch_name: str = "convnext_base",
    drop_path_rate: float = 0.0,
    layer_scale_init_value: float = 1e-6,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    hash: Optional[str] = None,
    **kwargs,
):
    from ..models.convnext import ConvNeXt

    model_kwargs = dict(
        in_chans=in_chans,
        depths=depths,
        dims=dims,
        drop_path_rate=drop_path_rate,
        layer_scale_init_value=layer_scale_init_value,
    )
    model_kwargs.update(**kwargs)
    model = ConvNeXt(**model_kwargs)
    if pretrained:
        if type(weights) is Weights and weights not in {Weights.LVD1689M, Weights.SAT493M}:
            raise ValueError(f"Unsupported weights for the backbone: {weights}")
        elif type(weights) is Weights:
            url = _make_dinov3_convnext_model_url(
                compact_arch_name=compact_arch_name,
                weights=weights,
                hash=hash,
            )
        else:
            url = convert_path_or_url_to_url(weights)
        state_dict = _safe_load_state_dict_from_url(url, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
    return model


def dinov3_vits16(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    if "hash" not in kwargs:
        kwargs["hash"] = "08c60483"
    kwargs["version"] = None
    return _make_dinov3_vit(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        pretrained=pretrained,
        weights=weights,
        compact_arch_name="vits",
        check_hash=check_hash,
        **kwargs,
    )


def dinov3_vits16plus(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    if "hash" not in kwargs:
        kwargs["hash"] = "4057cbaa"
    kwargs["version"] = None
    return _make_dinov3_vit(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=6,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="swiglu",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        pretrained=pretrained,
        weights=weights,
        compact_arch_name="vitsplus",
        check_hash=check_hash,
        **kwargs,
    )


def dinov3_vitb16(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    if "hash" not in kwargs:
        kwargs["hash"] = "73cec8be"
    kwargs["version"] = None
    return _make_dinov3_vit(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        pretrained=pretrained,
        weights=weights,
        compact_arch_name="vitb",
        check_hash=check_hash,
        **kwargs,
    )


def dinov3_vitl16(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    untie_global_and_local_cls_norm = False
    if weights == Weights.LVD1689M:
        if "hash" not in kwargs:
            kwargs["hash"] = "8aa4cbdd"
    elif weights == Weights.SAT493M:
        if "hash" not in kwargs:
            kwargs["hash"] = "eadcf0ff"
        untie_global_and_local_cls_norm = True
    elif type(weights) is str:
        import re

        pattern = r"-(.{8}).pth"
        matches = re.findall(pattern, weights)
        if len(matches) == 1:
            hash = matches[0]
        elif not os.path.isfile(weights):
            raise ValueError(f"Unexpected weights specification for the ViT-L backbone: {weights}")
        if len(matches) == 1 and matches[0] == "eadcf0ff":
            untie_global_and_local_cls_norm = True
    kwargs["version"] = None
    return _make_dinov3_vit(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        untie_global_and_local_cls_norm=untie_global_and_local_cls_norm,
        pretrained=pretrained,
        weights=weights,
        compact_arch_name="vitl",
        check_hash=check_hash,
        **kwargs,
    )


def dinov3_vitl16plus(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    if "hash" not in kwargs:
        kwargs["hash"] = "46503df0"

    return _make_dinov3_vit(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=6.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="swiglu",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        pretrained=pretrained,
        weights=weights,
        compact_arch_name="vitlplus",
        check_hash=check_hash,
        **kwargs,
    )


def dinov3_vith16plus(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    if "hash" not in kwargs:
        kwargs["hash"] = "7c1da9a5"

    return _make_dinov3_vit(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=1280,
        depth=32,
        num_heads=20,
        ffn_ratio=6.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="swiglu",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        pretrained=pretrained,
        weights=weights,
        compact_arch_name="vithplus",
        check_hash=check_hash,
        **kwargs,
    )


def dinov3_vit7b16(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    check_hash: bool = False,
    **kwargs,
):
    if weights == Weights.LVD1689M:
        if "hash" not in kwargs:
            kwargs["hash"] = "a955f4ea"
    elif weights == Weights.SAT493M:
        if "hash" not in kwargs:
            kwargs["hash"] = "a6675841"
    kwargs["version"] = None
    untie_global_and_local_cls_norm = True
    return _make_dinov3_vit(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=4096,
        depth=40,
        num_heads=32,
        ffn_ratio=3,
        qkv_bias=False,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="swiglu64",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        untie_global_and_local_cls_norm=untie_global_and_local_cls_norm,
        pretrained=pretrained,
        weights=weights,
        compact_arch_name="vit7b",
        check_hash=check_hash,
        **kwargs,
    )


def dinov3_convnext_tiny(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    **kwargs,
):
    _hash_convnext = "21b726bb"
    if "hash" not in kwargs:
        kwargs["hash"] = _hash_convnext

    from ..models.convnext import convnext_sizes

    size_dict = convnext_sizes["tiny"]

    model = _make_dinov3_convnext(
        in_chans=3,
        depths=size_dict["depths"],
        dims=size_dict["dims"],
        compact_arch_name="convnext_tiny",
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        pretrained=pretrained,
        weights=weights,
        **kwargs,
    )
    if not pretrained:
        model.init_weights()
    return model


def dinov3_convnext_small(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    **kwargs,
):
    _hash_convnext = "296db49d"
    if "hash" not in kwargs:
        kwargs["hash"] = _hash_convnext

    from ..models.convnext import convnext_sizes

    size_dict = convnext_sizes["small"]

    model = _make_dinov3_convnext(
        in_chans=3,
        depths=size_dict["depths"],
        dims=size_dict["dims"],
        compact_arch_name="convnext_small",
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        pretrained=pretrained,
        weights=weights,
        **kwargs,
    )
    if not pretrained:
        model.init_weights()
    return model


def dinov3_convnext_base(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    **kwargs,
):
    _hash_convnext = "801f2ba9"
    if "hash" not in kwargs:
        kwargs["hash"] = _hash_convnext

    from ..models.convnext import convnext_sizes

    size_dict = convnext_sizes["base"]

    model = _make_dinov3_convnext(
        in_chans=3,
        depths=size_dict["depths"],
        dims=size_dict["dims"],
        compact_arch_name="convnext_base",
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        pretrained=pretrained,
        weights=weights,
        **kwargs,
    )
    if not pretrained:
        model.init_weights()
    return model


def dinov3_convnext_large(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    **kwargs,
):
    _hash_convnext = "61fa432d"
    if "hash" not in kwargs:
        kwargs["hash"] = _hash_convnext

    from ..models.convnext import convnext_sizes

    size_dict = convnext_sizes["large"]

    model = _make_dinov3_convnext(
        in_chans=3,
        depths=size_dict["depths"],
        dims=size_dict["dims"],
        compact_arch_name="convnext_large",
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        pretrained=pretrained,
        weights=weights,
        **kwargs,
    )
    if not pretrained:
        model.init_weights()
    return model
