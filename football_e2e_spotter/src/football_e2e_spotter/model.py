from __future__ import annotations

"""End-to-end pixel/audio model for full-timeline football event spotting."""

import math
from collections.abc import Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.models import RegNet_Y_400MF_Weights, regnet_y_400mf


LABELS = ("shot", "save", "corner", "freekick")
FAMILIES = ("shot_chain", "restart")


class SpatialTemporalRefinement(nn.Module):
    """Refine a RegNet stage with spatial and short-term temporal evidence.

    This is deliberately attached to trainable feature maps rather than to a
    frozen clip embedding.  The global long-range dependency is modeled later
    by the bidirectional GRU; this block preserves location-dependent motion.
    """

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        inner = max(int(channels) // int(reduction), 16)
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=True)
        self.local_temporal = nn.Sequential(
            nn.Conv3d(channels, inner, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(inner),
            nn.SiLU(inplace=True),
            nn.Conv3d(inner, channels, kernel_size=1, bias=True),
        )
        self.global_temporal = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=7, padding=3, groups=channels, bias=False),
            nn.Conv1d(channels, channels, kernel_size=1, bias=True),
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 5:
            raise ValueError("refinement input must be [batch,time,channels,height,width]")
        batch, steps, channels, height, width = inputs.shape
        flat = inputs.reshape(batch * steps, channels, height, width)
        spatial_summary = torch.cat(
            (flat.mean(dim=1, keepdim=True), flat.amax(dim=1, keepdim=True)), dim=1
        )
        spatial_gate = torch.sigmoid(self.spatial(spatial_summary)).reshape(
            batch, steps, 1, height, width
        )
        channel_first = inputs.permute(0, 2, 1, 3, 4).contiguous()
        local_gate = torch.sigmoid(self.local_temporal(channel_first)).permute(0, 2, 1, 3, 4)
        pooled = inputs.mean(dim=(-1, -2)).transpose(1, 2)
        global_gate = torch.sigmoid(self.global_temporal(pooled)).transpose(1, 2)
        global_gate = global_gate.unsqueeze(-1).unsqueeze(-1)
        enhanced = inputs * (1.0 + spatial_gate) * (1.0 + local_gate) * (1.0 + global_gate)
        return inputs + torch.tanh(self.residual_scale) * (enhanced - inputs)


class RegNetTemporalEncoder(nn.Module):
    """Trainable per-frame RegNet with temporal refinement after selected stages."""

    STAGE_CHANNELS = {"block1": 48, "block2": 104, "block3": 208, "block4": 440}

    def __init__(
        self,
        *,
        imagenet_initialization: bool = True,
        refinement_stages: Iterable[str] = ("block2", "block3", "block4"),
    ) -> None:
        super().__init__()
        weights = RegNet_Y_400MF_Weights.DEFAULT if imagenet_initialization else None
        backbone = regnet_y_400mf(weights=weights)
        self.stem = backbone.stem
        self.stages = backbone.trunk_output
        self.output_dim = int(backbone.fc.in_features)
        requested = tuple(str(item) for item in refinement_stages)
        unknown = sorted(set(requested) - set(self.STAGE_CHANNELS))
        if unknown:
            raise ValueError(f"unknown RegNet refinement stages: {unknown}")
        self.refinement = nn.ModuleDict({
            name: SpatialTemporalRefinement(self.STAGE_CHANNELS[name]) for name in requested
        })

    def forward(self, frames: Tensor) -> Tensor:
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError("frames must be [batch,time,3,height,width]")
        batch, steps, channels, height, width = frames.shape
        hidden = self.stem(frames.reshape(batch * steps, channels, height, width))
        for name, stage in self.stages.named_children():
            hidden = stage(hidden)
            if name in self.refinement:
                _, stage_channels, stage_height, stage_width = hidden.shape
                sequence = hidden.reshape(
                    batch, steps, stage_channels, stage_height, stage_width
                )
                hidden = self.refinement[name](sequence).reshape(
                    batch * steps, stage_channels, stage_height, stage_width
                )
        return hidden.mean(dim=(-1, -2)).reshape(batch, steps, self.output_dim)


class AudioEncoder(nn.Module):
    def __init__(self, mel_bins: int = 64, output_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.mel_bins = int(mel_bins)
        self.output_dim = int(output_dim)
        self.norm = nn.LayerNorm(self.mel_bins)
        self.network = nn.Sequential(
            nn.Conv1d(self.mel_bins, output_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(output_dim, output_dim, kernel_size=5, padding=4, dilation=2),
            nn.GELU(),
        )

    def forward(self, log_mel: Tensor) -> Tensor:
        if log_mel.ndim != 3 or log_mel.shape[-1] != self.mel_bins:
            raise ValueError(f"audio must be [batch,time,{self.mel_bins}]")
        return self.network(self.norm(log_mel).transpose(1, 2)).transpose(1, 2)


class CausalShotContext(nn.Module):
    """Past-only shot evidence used by the save head, never as a hard gate."""

    def __init__(self, history_steps: int, half_lives: tuple[float, ...] = (1.0, 2.0, 4.0)):
        super().__init__()
        self.history_steps = max(int(history_steps), 1)
        ages = torch.flip(torch.arange(self.history_steps, dtype=torch.float32), dims=(0,))
        kernels = []
        for half_life in half_lives:
            weights = torch.exp(-math.log(2.0) * ages / max(float(half_life), 1e-3))
            kernels.append(weights / weights.sum().clamp_min(1e-9))
        self.register_buffer("kernels", torch.stack(kernels).unsqueeze(1), persistent=True)

    @property
    def output_dim(self) -> int:
        return 1 + int(self.kernels.shape[0])

    def forward(self, shot_probability: Tensor) -> Tensor:
        values = shot_probability.transpose(1, 2)
        padded = F.pad(values, (self.history_steps - 1, 0))
        peak = F.max_pool1d(padded, kernel_size=self.history_steps, stride=1).transpose(1, 2)
        traces = F.conv1d(padded, self.kernels).transpose(1, 2)
        return torch.cat((peak, traces), dim=-1)


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size=5, padding=2 * dilation,
            dilation=dilation, groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, 2 * channels, kernel_size=1)
        self.output = nn.Sequential(nn.Dropout(dropout), nn.Linear(channels, channels))
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = self.depthwise(self.norm(inputs).transpose(1, 2))
        value, gate = self.pointwise(hidden).chunk(2, dim=1)
        hidden = (value * torch.sigmoid(gate)).transpose(1, 2)
        return inputs + torch.tanh(self.scale) * self.output(hidden)


class FootballE2ESpotter(nn.Module):
    """Four-class dense spotter trained directly from RGB pixels and audio."""

    LABELS = LABELS
    FAMILIES = FAMILIES

    def __init__(
        self,
        *,
        sample_fps: float = 2.0,
        mel_bins: int = 64,
        audio_dim: int = 128,
        hidden_dim: int = 256,
        gru_layers: int = 1,
        dropout: float = 0.15,
        shot_history_seconds: float = 6.0,
        max_offset_seconds: float = 2.0,
        imagenet_initialization: bool = True,
        refinement_stages: Iterable[str] = ("block2", "block3", "block4"),
    ) -> None:
        super().__init__()
        if sample_fps <= 0:
            raise ValueError("sample_fps must be positive")
        self.sample_fps = float(sample_fps)
        self.max_offset_seconds = float(max_offset_seconds)
        self.visual = RegNetTemporalEncoder(
            imagenet_initialization=imagenet_initialization,
            refinement_stages=refinement_stages,
        )
        self.audio = AudioEncoder(mel_bins, audio_dim, dropout)
        visual_dim = self.visual.output_dim
        fusion_input = 3 * visual_dim + audio_dim
        temporal_dim = 2 * int(hidden_dim)
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_input),
            nn.Linear(fusion_input, temporal_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            temporal_dim,
            hidden_dim,
            num_layers=int(gru_layers),
            batch_first=True,
            bidirectional=True,
            dropout=dropout if int(gru_layers) > 1 else 0.0,
        )
        self.temporal = nn.Sequential(*[
            TemporalResidualBlock(temporal_dim, dilation, dropout)
            for dilation in (1, 2, 4, 8)
        ])
        self.norm = nn.LayerNorm(temporal_dim)
        self.family_head = nn.Linear(temporal_dim, len(FAMILIES))
        self.shot_head = nn.Linear(temporal_dim, 1)
        history_steps = max(int(round(shot_history_seconds * self.sample_fps)), 1)
        self.shot_context = CausalShotContext(history_steps)
        self.save_head = nn.Sequential(
            nn.LayerNorm(temporal_dim + self.shot_context.output_dim),
            nn.Linear(temporal_dim + self.shot_context.output_dim, hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )
        self.restart_head = nn.Sequential(
            nn.LayerNorm(temporal_dim), nn.Linear(temporal_dim, hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 2),
        )
        self.family_residual_scale = nn.Parameter(torch.full((len(LABELS),), 0.2))
        self.offset_head = nn.Linear(temporal_dim, len(LABELS))
        self.embedding_head = nn.Linear(temporal_dim, 128)

    def encode_visual(self, frames: Tensor) -> Tensor:
        return self.visual(frames)

    def forward_from_embeddings(
        self,
        visual: Tensor,
        audio: Tensor,
        valid_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if visual.ndim != 3 or audio.shape[:2] != visual.shape[:2]:
            raise ValueError("visual/audio timelines disagree")
        delta = F.pad(visual[:, 1:] - visual[:, :-1], (0, 0, 1, 0))
        fused = self.fusion(torch.cat((visual, delta, delta.abs(), self.audio(audio)), dim=-1))
        if valid_mask is not None:
            fused = fused * valid_mask.to(fused.dtype).unsqueeze(-1)
        hidden, _ = self.gru(fused)
        hidden = self.norm(self.temporal(hidden))
        family_logits = self.family_head(hidden)
        shot_logits = self.shot_head(hidden)
        shot_context = self.shot_context(torch.sigmoid(shot_logits).detach())
        save_logits = self.save_head(torch.cat((hidden, shot_context), dim=-1))
        restart_logits = self.restart_head(hidden)
        scale = torch.tanh(self.family_residual_scale)
        class_logits = torch.cat((
            shot_logits + scale[0] * family_logits[..., 0:1],
            save_logits + scale[1] * family_logits[..., 0:1],
            restart_logits + scale[2:].reshape(1, 1, 2) * family_logits[..., 1:2],
        ), dim=-1)
        offsets = self.max_offset_seconds * torch.tanh(self.offset_head(hidden))
        embeddings = F.normalize(self.embedding_head(hidden), dim=-1)
        if valid_mask is not None:
            invalid = ~valid_mask.bool()
            class_logits = class_logits.masked_fill(invalid.unsqueeze(-1), -20.0)
            family_logits = family_logits.masked_fill(invalid.unsqueeze(-1), -20.0)
            offsets = offsets.masked_fill(invalid.unsqueeze(-1), 0.0)
        return {
            "class_logits": class_logits,
            "family_logits": family_logits,
            "offsets": offsets,
            "embeddings": embeddings,
            "shot_context": shot_context,
            "visual_embeddings": visual,
        }

    def forward(
        self,
        frames: Tensor,
        audio: Tensor,
        valid_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        return self.forward_from_embeddings(self.encode_visual(frames), audio, valid_mask)

