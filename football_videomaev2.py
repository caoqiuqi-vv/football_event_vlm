"""VideoMAEv2 adapter for the football event training pipeline.

The upstream classifier returns one clip vector. Football event training also
needs frame-level supervision, so this adapter keeps final spatiotemporal
tokens, pools each tubelet spatially, and expands tubelets to sampled frames.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint


def _load_upstream_module(source_root: str) -> ModuleType:
    module_path = Path(source_root) / "models" / "modeling_finetune.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"VideoMAEv2 modeling file not found: {module_path}")
    spec = importlib.util.spec_from_file_location(
        "football_videomaev2_upstream_modeling", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import VideoMAEv2 modeling module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _checkpoint_state(checkpoint: Any) -> dict[str, Tensor]:
    if not isinstance(checkpoint, dict):
        raise ValueError("VideoMAEv2 checkpoint must be a dictionary")
    raw = checkpoint.get(
        "module", checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    )
    if not isinstance(raw, dict):
        raise ValueError("VideoMAEv2 checkpoint has no module/model/state_dict")
    state: dict[str, Tensor] = {}
    for raw_key, value in raw.items():
        if not isinstance(value, Tensor):
            continue
        key = str(raw_key)
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("backbone."):
            key = key[len("backbone.") :]
        state[key] = value
    return state


class VideoMAEV2BackboneAdapter(nn.Module):
    """Expose contextual frame features from a VideoMAEv2 encoder."""

    is_video_backbone = True

    def __init__(
        self,
        *,
        source_root: str,
        architecture: str,
        weights: str,
        image_size: tuple[int, int],
        num_frames: int,
        tubelet_size: int = 2,
        gradient_checkpointing: bool = True,
        drop_path_rate: float = 0.1,
        preserve_global_feature: bool = False,
    ) -> None:
        super().__init__()
        upstream = _load_upstream_module(source_root)
        factories = {
            "videomaev2_vitb16": "vit_base_patch16_224",
            "videomaev2_vitl16": "vit_large_patch16_224",
        }
        if architecture not in factories:
            raise ValueError(
                f"Unsupported VideoMAEv2 architecture {architecture}; "
                f"expected one of {sorted(factories)}"
            )
        factory = getattr(upstream, factories[architecture])
        # Upstream assumes a non-Identity head during initialization.
        self.encoder = factory(
            img_size=image_size,
            all_frames=num_frames,
            tubelet_size=tubelet_size,
            num_classes=3,
            use_mean_pooling=True,
            with_cp=False,
            drop_path_rate=drop_path_rate,
        )
        self.encoder.head = nn.Identity()
        self.num_features = int(self.encoder.num_features)
        # Preserve the exact global action representation used by K710
        # pretraining. The downstream football head splits this prefix from
        # the tubelet feature before applying frame-level supervision.
        self.preserve_global_feature = bool(preserve_global_feature)
        self.global_feature_dim = (
            self.num_features if self.preserve_global_feature else 0
        )
        self.output_feature_dim = self.num_features + self.global_feature_dim
        self.num_frames = int(num_frames)
        self.tubelet_size = int(tubelet_size)
        self.image_size = tuple(int(value) for value in image_size)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self._load_weights(weights)

    @property
    def blocks(self) -> nn.ModuleList:
        return self.encoder.blocks

    @property
    def norm(self) -> nn.Module:
        return self.encoder.norm

    @property
    def fc_norm(self) -> nn.Module | None:
        return self.encoder.fc_norm

    def _load_weights(self, path: str) -> None:
        checkpoint_path = Path(path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"VideoMAEv2 weights not found: {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False, mmap=True
        )
        source = _checkpoint_state(checkpoint)
        target = self.encoder.state_dict()
        matched: dict[str, Tensor] = {}
        skipped: list[str] = []
        for key, value in source.items():
            if key.startswith("head.") or key.startswith("cls_head."):
                continue
            if key not in target:
                continue
            if tuple(value.shape) != tuple(target[key].shape):
                skipped.append(
                    f"{key}: checkpoint{tuple(value.shape)} != model{tuple(target[key].shape)}"
                )
                continue
            matched[key] = value
        missing, unexpected = self.encoder.load_state_dict(matched, strict=False)
        important_missing = [
            key for key in missing if not key.startswith("head.") and key != "pos_embed"
        ]
        print(
            "Loaded VideoMAEv2 checkpoint "
            f"{checkpoint_path}: matched={len(matched)} "
            f"missing={len(important_missing)} unexpected={len(unexpected)} "
            f"skipped_shape={len(skipped)}",
            flush=True,
        )
        if important_missing:
            print(
                "WARN VideoMAEv2 missing weights: "
                + ", ".join(important_missing[:20]),
                flush=True,
            )
        if skipped:
            print(
                "WARN VideoMAEv2 skipped weights: " + "; ".join(skipped[:10]),
                flush=True,
            )

    def _run_blocks(self, tokens: Tensor) -> Tensor:
        for block in self.encoder.blocks:
            trainable = any(parameter.requires_grad for parameter in block.parameters())
            if self.training and self.gradient_checkpointing and trainable:
                tokens = activation_checkpoint(
                    block,
                    tokens,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                tokens = block(tokens)
        return tokens

    def forward_temporal_features(
        self, inputs: Tensor, *, output_frames: int | None = None
    ) -> Tensor:
        """Return B x T x D features from normalized B x T x C x H x W RGB."""
        if inputs.ndim != 5:
            raise ValueError(f"VideoMAEv2 inputs must be BxTxCxHxW, got {inputs.shape}")
        batch, frames, channels, height, width = inputs.shape
        if channels != 3:
            raise ValueError(f"VideoMAEv2 expects RGB inputs, got C={channels}")
        if (height, width) != self.image_size:
            raise ValueError(
                f"VideoMAEv2 expected image_size={self.image_size}, got {(height, width)}"
            )
        if frames != self.num_frames:
            raise ValueError(f"VideoMAEv2 expected {self.num_frames} frames, got {frames}")

        video = inputs.permute(0, 2, 1, 3, 4).contiguous()
        tokens = self.encoder.patch_embed(video)
        pos_embed = self.encoder.pos_embed
        if pos_embed is not None:
            if int(pos_embed.shape[1]) != int(tokens.shape[1]):
                raise ValueError(
                    "VideoMAEv2 positional token count mismatch: "
                    f"pos={pos_embed.shape[1]} tokens={tokens.shape[1]}"
                )
            tokens = tokens + pos_embed.to(
                device=tokens.device, dtype=tokens.dtype
            ).expand(batch, -1, -1)
        tokens = self.encoder.pos_drop(tokens)
        tokens = self._run_blocks(tokens)

        tubelets = frames // self.tubelet_size
        spatial_tokens = (height // 16) * (width // 16)
        if int(tokens.shape[1]) != tubelets * spatial_tokens:
            raise ValueError(
                "Unexpected VideoMAEv2 token layout: "
                f"tokens={tokens.shape[1]} tubelets={tubelets} spatial={spatial_tokens}"
            )
        tube_features = tokens.reshape(
            batch, tubelets, spatial_tokens, self.num_features
        ).mean(dim=2)
        if self.encoder.fc_norm is not None:
            global_feature = self.encoder.fc_norm(tokens.mean(dim=1))
            tube_features = self.encoder.fc_norm(tube_features)
        elif not isinstance(self.encoder.norm, nn.Identity):
            global_feature = self.encoder.norm(tokens.mean(dim=1))
            tube_features = self.encoder.norm(tube_features)
        else:
            global_feature = tokens.mean(dim=1)

        if self.preserve_global_feature:
            expanded_global = global_feature.unsqueeze(1).expand(
                -1, tubelets, -1
            )
            tube_features = torch.cat((expanded_global, tube_features), dim=-1)

        target_frames = int(output_frames or frames)
        frame_features = tube_features.repeat_interleave(self.tubelet_size, dim=1)
        if frame_features.shape[1] < target_frames:
            pad = frame_features[:, -1:].expand(
                -1, target_frames - frame_features.shape[1], -1
            )
            frame_features = torch.cat((frame_features, pad), dim=1)
        return frame_features[:, :target_frames]


def build_videomaev2_backbone(cfg: Any) -> VideoMAEV2BackboneAdapter:
    video_cfg = cfg.video
    model_cfg = cfg.model
    vm_cfg = model_cfg.get("videomaev2", {})
    image_size = tuple(int(value) for value in video_cfg.image_size)
    if len(image_size) != 2:
        raise ValueError("video.image_size must contain [height, width]")
    return VideoMAEV2BackboneAdapter(
        source_root=str(
            vm_cfg.get("source_root", "/home/new_users/qiuqi/code/VideoMAEv2")
        ),
        architecture=str(model_cfg.backbone),
        weights=str(model_cfg.weights),
        image_size=(image_size[0], image_size[1]),
        num_frames=int(video_cfg.num_frames),
        tubelet_size=int(vm_cfg.get("tubelet_size", 2)),
        gradient_checkpointing=bool(model_cfg.get("gradient_checkpointing", True)),
        drop_path_rate=float(vm_cfg.get("drop_path_rate", 0.1)),
        preserve_global_feature=bool(
            vm_cfg.get("preserve_global_feature", False)
        ),
    )
