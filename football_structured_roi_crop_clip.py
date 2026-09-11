from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

import train_football_events as football
from train_football_events_featuremap_structured_roi import (
    load_structured_roi_model,
)


def _gather_shared_time(values: Tensor, indices: Tensor) -> Tensor:
    """Gather B,T,... values with class-specific B,C,S indices."""
    batch, classes, slots = indices.shape
    tail = values.shape[2:]
    expanded = values.unsqueeze(1).expand(
        batch, classes, values.shape[1], *tail
    )
    gather_index = indices.reshape(
        batch, classes, slots, *([1] * len(tail))
    ).expand(batch, classes, slots, *tail)
    return expanded.gather(2, gather_index)


def _gather_class_time(values: Tensor, indices: Tensor) -> Tensor:
    """Gather B,T,C,... values with the matching class's B,C,S indices."""
    batch, frames, classes = values.shape[:3]
    if indices.shape[:2] != (batch, classes):
        raise ValueError(
            f"class-time indices={tuple(indices.shape)} do not match "
            f"values={tuple(values.shape)}"
        )
    ordered = values.permute(0, 2, 1, *range(3, values.ndim))
    tail = ordered.shape[3:]
    gather_index = indices.reshape(
        batch, classes, indices.shape[2], *([1] * len(tail))
    ).expand(batch, classes, indices.shape[2], *tail)
    return ordered.gather(2, gather_index)


class DualQueryOriginalImageCropper(nn.Module):
    """Turn frozen class/query attention into two original-image crops."""

    def __init__(
        self,
        *,
        num_labels: int,
        num_frames: int,
        topk_frames: int,
        exploration_frames: int,
        crop_height: int,
        crop_width: int,
        crop_scale_x: float,
        crop_scale_y: float,
    ):
        super().__init__()
        self.num_labels = int(num_labels)
        self.num_frames = int(num_frames)
        self.topk_frames = min(max(int(topk_frames), 1), self.num_frames)
        self.exploration_frames = min(
            max(int(exploration_frames), 0),
            self.num_frames - self.topk_frames,
        )
        self.crop_height = int(crop_height)
        self.crop_width = int(crop_width)
        self.crop_scale_x = float(crop_scale_x)
        self.crop_scale_y = float(crop_scale_y)
        if self.crop_height % 16 or self.crop_width % 16:
            raise ValueError("ROI crop height/width must be divisible by 16")
        if not (0.0 < self.crop_scale_x <= 1.0):
            raise ValueError("crop_scale_x must be in (0, 1]")
        if not (0.0 < self.crop_scale_y <= 1.0):
            raise ValueError("crop_scale_y must be in (0, 1]")

    @property
    def slot_count(self) -> int:
        return self.topk_frames + self.exploration_frames

    def select_indices(self, global_frame_logits: Tensor) -> Tensor:
        batch, frames, classes = global_frame_logits.shape
        if classes != self.num_labels:
            raise ValueError("global frame logits have wrong class dimension")
        ranking = global_frame_logits.detach().permute(0, 2, 1)
        topk = torch.topk(
            ranking, k=min(self.topk_frames, frames), dim=-1
        ).indices
        if self.exploration_frames == 0:
            return topk

        occupied = torch.zeros(
            batch,
            classes,
            frames,
            dtype=torch.bool,
            device=global_frame_logits.device,
        )
        occupied.scatter_(2, topk, True)
        if self.training:
            priority = torch.rand(
                batch,
                classes,
                frames,
                device=global_frame_logits.device,
            )
        else:
            anchors = torch.linspace(
                0,
                frames - 1,
                self.exploration_frames + 2,
                device=global_frame_logits.device,
            )[1:-1]
            positions = torch.arange(
                frames,
                device=global_frame_logits.device,
                dtype=anchors.dtype,
            )
            base = -torch.stack(
                [(positions - anchor).abs() for anchor in anchors], dim=0
            ).amin(dim=0)
            priority = base.reshape(1, 1, frames).expand(
                batch, classes, -1
            )
        priority = priority.masked_fill(occupied, -1.0e9)
        exploration = torch.topk(
            priority, k=self.exploration_frames, dim=-1
        ).indices
        return torch.cat([topk, exploration], dim=-1)

    def forward(
        self,
        normalized_inputs: Tensor,
        attention: Tensor,
        indices: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch, frames, channels, height, width = normalized_inputs.shape
        if attention.shape[:3] != (
            batch,
            frames,
            self.num_labels,
        ):
            raise ValueError(
                f"attention={tuple(attention.shape)} does not match "
                f"inputs={tuple(normalized_inputs.shape)}"
            )
        queries = attention.shape[3]
        patches = attention.shape[4]
        patch_h, patch_w = height // 16, width // 16
        if patch_h * patch_w != patches:
            raise ValueError(
                f"attention patches={patches} cannot map to "
                f"{patch_h}x{patch_w}"
            )

        selected_attention = _gather_class_time(attention, indices)
        selected_attention = selected_attention / selected_attention.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        x_coords = torch.linspace(
            -1.0 + 1.0 / patch_w,
            1.0 - 1.0 / patch_w,
            patch_w,
            device=attention.device,
            dtype=attention.dtype,
        )
        y_coords = torch.linspace(
            -1.0 + 1.0 / patch_h,
            1.0 - 1.0 / patch_h,
            patch_h,
            device=attention.device,
            dtype=attention.dtype,
        )
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        flat_x = grid_x.reshape(-1)
        flat_y = grid_y.reshape(-1)
        center_x = (selected_attention * flat_x).sum(dim=-1)
        center_y = (selected_attention * flat_y).sum(dim=-1)

        scale_x = torch.full_like(center_x, self.crop_scale_x)
        scale_y = torch.full_like(center_y, self.crop_scale_y)
        translate_x = (1.0 - scale_x) * center_x.clamp(-1.0, 1.0)
        translate_y = (1.0 - scale_y) * center_y.clamp(-1.0, 1.0)
        flat_count = batch * self.num_labels * indices.shape[2] * queries
        theta = attention.new_zeros((flat_count, 2, 3))
        theta[:, 0, 0] = scale_x.reshape(-1)
        theta[:, 1, 1] = scale_y.reshape(-1)
        theta[:, 0, 2] = translate_x.reshape(-1)
        theta[:, 1, 2] = translate_y.reshape(-1)

        selected_inputs = _gather_shared_time(
            normalized_inputs, indices
        ).unsqueeze(3)
        selected_inputs = selected_inputs.expand(
            -1, -1, -1, queries, -1, -1, -1
        ).reshape(flat_count, channels, height, width)
        sampling_grid = F.affine_grid(
            theta,
            (
                flat_count,
                channels,
                self.crop_height,
                self.crop_width,
            ),
            align_corners=False,
        )
        crops = F.grid_sample(
            selected_inputs,
            sampling_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        ).reshape(
            batch,
            self.num_labels * indices.shape[2] * queries,
            channels,
            self.crop_height,
            self.crop_width,
        )
        crop_params = torch.stack(
            [center_x, center_y, scale_x, scale_y], dim=-1
        )
        return crops, crop_params, selected_attention


class ROICropClipTemporalFusion(nn.Module):
    """Classify dual-query crop tokens and add a bounded clip-only residual."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_labels: int,
        num_frames: int,
        slots: int,
        queries: int,
        num_heads: int,
        layers: int,
        dropout: float,
        gate_init: float,
        max_delta: float,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_labels = int(num_labels)
        self.num_frames = int(num_frames)
        self.slots = int(slots)
        self.queries = int(queries)
        self.max_delta = float(max_delta)
        self.class_token = nn.Parameter(
            torch.zeros(1, num_labels, 1, hidden_dim)
        )
        self.class_embedding = nn.Parameter(
            torch.zeros(1, num_labels, 1, 1, hidden_dim)
        )
        self.slot_embedding = nn.Parameter(
            torch.zeros(1, 1, slots, 1, hidden_dim)
        )
        self.query_embedding = nn.Parameter(
            torch.zeros(1, 1, 1, queries, hidden_dim)
        )
        self.time_embedding = nn.Parameter(
            torch.zeros(1, num_frames, hidden_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=max(int(layers), 1)
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.roi_classifier = nn.Linear(hidden_dim, 1)
        self.delta_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, max(hidden_dim // 2, 64)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(hidden_dim // 2, 64), 1),
        )
        self.gate_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, max(hidden_dim // 2, 64)),
            nn.GELU(),
            nn.Linear(max(hidden_dim // 2, 64), 1),
        )
        for parameter in (
            self.class_token,
            self.class_embedding,
            self.slot_embedding,
            self.query_embedding,
            self.time_embedding,
        ):
            nn.init.trunc_normal_(parameter, std=0.02)
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)
        gate_init = min(max(float(gate_init), 1e-4), 1.0 - 1e-4)
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(
            self.gate_head[-1].bias,
            torch.logit(torch.tensor(gate_init)).item(),
        )

    def forward(
        self,
        *,
        crop_tokens: Tensor,
        indices: Tensor,
        global_logits: Tensor,
    ) -> dict[str, Tensor]:
        batch = crop_tokens.shape[0]
        expected = self.num_labels * self.slots * self.queries
        if crop_tokens.shape[1:] != (expected, self.hidden_dim):
            raise ValueError(
                f"crop tokens={tuple(crop_tokens.shape)} expected "
                f"(B,{expected},{self.hidden_dim})"
            )
        local = crop_tokens.reshape(
            batch,
            self.num_labels,
            self.slots,
            self.queries,
            self.hidden_dim,
        )
        time = self.time_embedding[0][indices].unsqueeze(3)
        local = (
            local
            + time
            + self.class_embedding
            + self.slot_embedding
            + self.query_embedding
        )
        local = local.reshape(
            batch * self.num_labels,
            self.slots * self.queries,
            self.hidden_dim,
        )
        class_token = self.class_token.expand(
            batch, -1, -1, -1
        ).reshape(batch * self.num_labels, 1, self.hidden_dim)
        encoded = self.encoder(torch.cat([class_token, local], dim=1))
        feature = self.norm(encoded[:, 0]).reshape(
            batch, self.num_labels, self.hidden_dim
        )
        roi_logits = self.roi_classifier(feature).squeeze(-1)
        raw_delta = self.max_delta * torch.tanh(
            self.delta_head(feature).squeeze(-1)
        )
        gate = torch.sigmoid(self.gate_head(feature).squeeze(-1))
        correction = gate * raw_delta
        return {
            "logits": global_logits.detach() + correction,
            "roi_logits": roi_logits,
            "roi_raw_delta": raw_delta,
            "roi_correction": correction,
            "roi_gate": gate,
            "roi_feature": feature,
        }


class StructuredROICropClipModel(nn.Module):
    """Frozen E2 ROI proposer + original-image crop encoder + clip-only fusion."""

    def __init__(self, base: nn.Module, cfg: Any):
        super().__init__()
        self.base = base
        crop_cfg = cfg.model.roi_crop_clip
        queries = int(cfg.model.spatial_attention.queries_per_class)
        self.cropper = DualQueryOriginalImageCropper(
            num_labels=len(football.LABELS),
            num_frames=int(cfg.video.num_frames),
            topk_frames=int(crop_cfg.get("topk_frames", 4)),
            exploration_frames=int(
                crop_cfg.get("exploration_frames", 2)
            ),
            crop_height=int(crop_cfg.get("crop_height", 256)),
            crop_width=int(crop_cfg.get("crop_width", 448)),
            crop_scale_x=float(crop_cfg.get("crop_scale_x", 0.35)),
            crop_scale_y=float(crop_cfg.get("crop_scale_y", 0.35)),
        )
        self.fusion = ROICropClipTemporalFusion(
            hidden_dim=int(cfg.model.hidden_dim),
            num_labels=len(football.LABELS),
            num_frames=int(cfg.video.num_frames),
            slots=self.cropper.slot_count,
            queries=queries,
            num_heads=int(crop_cfg.get("num_heads", 8)),
            layers=int(crop_cfg.get("temporal_layers", 2)),
            dropout=float(cfg.model.dropout),
            gate_init=float(crop_cfg.get("gate_init", 0.02)),
            max_delta=float(crop_cfg.get("max_delta", 1.0)),
        )
        self.encode_chunk_size = max(
            int(crop_cfg.get("encode_chunk_size", 12)), 1
        )
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def _encode_crops(self, crops: Tensor) -> Tensor:
        pieces: list[Tensor] = []
        with torch.no_grad():
            for start in range(0, crops.shape[1], self.encode_chunk_size):
                chunk = crops[:, start : start + self.encode_chunk_size]
                features = self.base.encode_frames(chunk)
                pieces.append(self.base.frame_proj(features))
        return torch.cat(pieces, dim=1)

    def forward(
        self,
        inputs: Tensor,
        *,
        return_aux: bool = False,
        **_: Any,
    ) -> Tensor | dict[str, Tensor]:
        with torch.no_grad():
            global_features, patch_tokens = (
                self.base.encode_frames_with_patch_tokens(inputs)
            )
            global_outputs = self.base._global_branch_outputs(
                global_features
            )
            spatial_outputs = self.base.spatial_attention(
                global_outputs["frame_tokens"], patch_tokens
            )
            attention = spatial_outputs.get("attention_maps")
            if attention is None:
                raise RuntimeError(
                    "structured ROI checkpoint must return attention maps"
                )
            indices = self.cropper.select_indices(
                global_outputs["frame_event_logits"]
            )
            normalized_inputs = self.base.preprocess_inputs(inputs)
            crops, crop_params, selected_attention = self.cropper(
                normalized_inputs, attention, indices
            )
            crop_tokens = self._encode_crops(crops)

        fused = self.fusion(
            crop_tokens=crop_tokens,
            indices=indices,
            global_logits=global_outputs["logits"],
        )
        if not return_aux:
            return fused["logits"]
        return {
            "logits": fused["logits"],
            "global_logits": global_outputs["logits"],
            "frame_event_logits": global_outputs["frame_event_logits"],
            "global_frame_event_logits": global_outputs[
                "frame_event_logits"
            ],
            "spatial_clip_logits": fused["roi_logits"],
            "spatial_fused_logits": fused["logits"],
            "spatial_fusion_delta": fused["roi_correction"],
            "spatial_fusion_gate": fused["roi_gate"],
            "roi_raw_delta": fused["roi_raw_delta"],
            "roi_indices": indices,
            "roi_crop_params": crop_params,
            "roi_selected_attention": selected_attention,
            "retention_reference_logits": global_outputs["logits"],
        }


def load_structured_roi_crop_clip_model(
    cfg: Any, device: torch.device
) -> StructuredROICropClipModel:
    checkpoint_path = str(
        cfg.model.featuremap_structured.init_checkpoint
    )
    base = load_structured_roi_model(cfg, device)
    model = StructuredROICropClipModel(base, cfg).to(device)
    trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable or any(
        not name.startswith("fusion.") for name in trainable
    ):
        raise RuntimeError(
            "ROI crop clip experiment must train only fusion parameters: "
            f"{trainable[:20]}"
        )
    print(
        f"Loaded frozen structured ROI proposer={checkpoint_path} "
        f"trainable={len(trainable)} crop="
        f"{model.cropper.crop_height}x{model.cropper.crop_width} "
        f"slots={model.cropper.slot_count} "
        f"queries={model.fusion.queries}",
        flush=True,
    )
    return model
