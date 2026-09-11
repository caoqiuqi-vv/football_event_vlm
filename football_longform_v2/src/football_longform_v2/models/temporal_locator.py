from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


MODEL_SCHEMA = "football_longform_v2.temporal_locator.conditional_classes.v2"


class FourierTimeEncoding(nn.Module):
    def __init__(self, hidden_dim: int, bands: int) -> None:
        super().__init__()
        frequencies = torch.tensor([2.0 ** (-index) for index in range(bands)])
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.projection = nn.Linear(4 * bands, hidden_dim)

    def forward(self, timestamps: Tensor) -> Tensor:
        if timestamps.ndim != 2:
            raise ValueError("timestamps must have shape [batch, time]")
        relative = timestamps - timestamps[:, :1]
        delta = torch.zeros_like(relative)
        delta[:, 1:] = timestamps[:, 1:] - timestamps[:, :-1]
        rel_phase = 2.0 * math.pi * relative.unsqueeze(-1) * self.frequencies
        delta_phase = 2.0 * math.pi * delta.unsqueeze(-1) * self.frequencies
        encoded = torch.cat(
            [rel_phase.sin(), rel_phase.cos(), delta_phase.sin(), delta_phase.cos()], dim=-1
        )
        return self.projection(encoded)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.norm = nn.LayerNorm(hidden_dim)
        self.depthwise = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size, padding=padding, groups=hidden_dim
        )
        self.channel_mixer = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim * 4, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim * 4, hidden_dim, 1),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        residual = inputs
        hidden = self.norm(inputs.transpose(1, 2)).transpose(1, 2)
        hidden = self.depthwise(hidden)
        hidden = self.channel_mixer(hidden)
        return residual + hidden


class TemporalLocator(nn.Module):
    """Dense point locator with an RGB-defined main path and bounded entity residual."""

    def __init__(
        self,
        context_dim: int,
        motion_dim: int,
        *,
        family_count: int = 3,
        output_labels: tuple[str, ...] = ("shot", "save", "corner", "penalty", "freekick", "kickoff"),
        family_names: tuple[str, ...] = ("shot_chain", "restart", "generic_event"),
        timeline_hz: float = 5.0,
        save_lookback_seconds: float = 3.0,
        hidden_dim: int = 256,
        levels: int = 4,
        blocks_per_level: int = 2,
        kernel_size: int = 5,
        dropout: float = 0.1,
        time_fourier_bands: int = 8,
        entity_dim: int | None = None,
        max_entity_residual_logit: float = 0.15,
    ) -> None:
        super().__init__()
        if levels < 2:
            raise ValueError("levels must be at least 2")
        self.family_count = family_count
        self.output_labels = tuple(output_labels)
        self.family_names = tuple(family_names)
        self.timeline_hz = float(timeline_hz)
        self.save_lookback_steps = max(int(round(save_lookback_seconds * timeline_hz)) + 1, 2)
        self.max_entity_residual_logit = float(max_entity_residual_logit)
        self.context_projection = nn.Linear(context_dim, hidden_dim)
        self.motion_projection = nn.Linear(motion_dim, hidden_dim)
        self.time_encoding = FourierTimeEncoding(hidden_dim, time_fourier_bands)
        self.input_norm = nn.LayerNorm(hidden_dim)

        def stage() -> nn.Sequential:
            return nn.Sequential(
                *[
                    ResidualTemporalBlock(hidden_dim, kernel_size, dropout)
                    for _ in range(blocks_per_level)
                ]
            )

        self.encoder_stages = nn.ModuleList([stage() for _ in range(levels)])
        self.downsamples = nn.ModuleList(
            [nn.Conv1d(hidden_dim, hidden_dim, 3, stride=2, padding=1) for _ in range(levels - 1)]
        )
        self.decoder_merges = nn.ModuleList(
            [nn.Conv1d(hidden_dim * 2, hidden_dim, 1) for _ in range(levels - 1)]
        )
        self.decoder_stages = nn.ModuleList([stage() for _ in range(levels - 1)])
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.heatmap_head = nn.Linear(hidden_dim, family_count)
        self.offset_head = nn.Linear(hidden_dim, family_count)
        self.quality_head = nn.Linear(hidden_dim, family_count)
        self.class_head = nn.Linear(hidden_dim, len(self.output_labels))
        self.state_head = nn.Linear(hidden_dim, len(self.output_labels))
        self.shot_label_index = (
            self.output_labels.index("shot") if "shot" in self.output_labels else None
        )
        self.save_label_index = (
            self.output_labels.index("save") if "save" in self.output_labels else None
        )
        self.restart_family_index = (
            self.family_names.index("restart") if "restart" in self.family_names else None
        )
        restart_names = ("corner", "penalty", "freekick", "kickoff")
        self.restart_label_indices = tuple(
            self.output_labels.index(label) for label in restart_names if label in self.output_labels
        )
        self.save_conditioner = None
        if self.shot_label_index is not None and self.save_label_index is not None:
            self.save_conditioner = nn.Sequential(
                nn.Linear(hidden_dim + 2, hidden_dim // 2), nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )
        self.restart_conditioner = None
        if self.restart_family_index is not None and self.restart_label_indices:
            self.restart_conditioner = nn.Sequential(
                nn.Linear(hidden_dim + 1 + len(self.restart_label_indices), hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, len(self.restart_label_indices)),
            )

        self.entity_projection = None
        self.entity_residual_head = None
        if entity_dim is not None:
            self.entity_projection = nn.Sequential(
                nn.LayerNorm(entity_dim), nn.Linear(entity_dim, hidden_dim), nn.GELU()
            )
            self.entity_residual_head = nn.Linear(hidden_dim * 2, family_count)

    @classmethod
    def from_config(cls, config: dict) -> "TemporalLocator":
        features = config["features"]
        model = config["model"]
        task = config["task"]
        entity = features.get("entity", {})
        return cls(
            context_dim=int(features["context"]["dim"]),
            motion_dim=int(features["motion"]["dim"]),
            family_count=len(task["proposal_families"]),
            output_labels=tuple(task["output_labels"]),
            family_names=tuple(task["proposal_families"]),
            timeline_hz=float(features["timeline_hz"]),
            hidden_dim=int(model["hidden_dim"]),
            levels=int(model["encoder_levels"]),
            blocks_per_level=int(model["blocks_per_level"]),
            kernel_size=int(model["kernel_size"]),
            dropout=float(model["dropout"]),
            time_fourier_bands=int(model["time_fourier_bands"]),
            entity_dim=int(entity["dim"]) if entity.get("enabled", False) else None,
            max_entity_residual_logit=float(entity.get("max_residual_logit", 0.15)),
        )

    def forward(
        self,
        context: Tensor,
        motion: Tensor,
        timestamps: Tensor,
        *,
        context_valid: Tensor | None = None,
        motion_valid: Tensor | None = None,
        entity: Tensor | None = None,
        entity_valid: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if context.shape[:2] != motion.shape[:2] or context.shape[:2] != timestamps.shape:
            raise ValueError("context, motion and timestamps must share [batch, time]")
        context_features = self.context_projection(context)
        motion_features = self.motion_projection(motion)
        if context_valid is not None:
            context_features = context_features * context_valid.unsqueeze(-1).to(context.dtype)
        if motion_valid is not None:
            motion_features = motion_features * motion_valid.unsqueeze(-1).to(motion.dtype)
        hidden = self.input_norm(
            context_features + motion_features + self.time_encoding(timestamps)
        ).transpose(1, 2)

        skips: list[Tensor] = []
        for level, encoder in enumerate(self.encoder_stages):
            hidden = encoder(hidden)
            skips.append(hidden)
            if level < len(self.downsamples):
                hidden = self.downsamples[level](hidden)
        for merge, decoder, skip in zip(
            self.decoder_merges, self.decoder_stages, reversed(skips[:-1])
        ):
            hidden = F.interpolate(hidden, size=skip.shape[-1], mode="linear", align_corners=False)
            hidden = merge(torch.cat([hidden, skip], dim=1))
            hidden = decoder(hidden)

        features = self.output_norm(hidden.transpose(1, 2))
        rgb_logits = self.heatmap_head(features)
        logits = rgb_logits
        entity_delta = torch.zeros_like(rgb_logits)
        if entity is not None:
            if self.entity_projection is None or self.entity_residual_head is None:
                raise ValueError("entity input was provided but entity support is disabled")
            if entity.shape[:2] != context.shape[:2]:
                raise ValueError("entity must share [batch, time]")
            entity_features = self.entity_projection(entity)
            if entity_valid is not None:
                entity_features = entity_features * entity_valid.unsqueeze(-1).to(entity.dtype)
            raw_delta = self.entity_residual_head(torch.cat([features, entity_features], dim=-1))
            entity_delta = self.max_entity_residual_logit * torch.tanh(raw_delta)
            logits = rgb_logits + entity_delta

        state_logits = self.state_head(features)
        base_class_logits = self.class_head(features)
        class_parts = [base_class_logits[..., index] for index in range(len(self.output_labels))]
        shot_condition_max = features.new_zeros(features.shape[:2])
        shot_condition_lag_seconds = features.new_zeros(features.shape[:2])
        if self.save_conditioner is not None:
            assert self.shot_label_index is not None and self.save_label_index is not None
            shot_probability = base_class_logits[..., self.shot_label_index].sigmoid().detach()
            padded = F.pad(shot_probability.unsqueeze(1), (self.save_lookback_steps - 1, 0))
            windows = padded.unfold(2, self.save_lookback_steps, 1).squeeze(1)
            shot_condition_max, peak_index = windows.max(dim=-1)
            shot_condition_lag_seconds = (
                self.save_lookback_steps - 1 - peak_index
            ).to(features.dtype) / self.timeline_hz
            save_delta = self.save_conditioner(torch.cat([
                features, shot_condition_max.unsqueeze(-1),
                shot_condition_lag_seconds.unsqueeze(-1),
            ], dim=-1)).squeeze(-1)
            class_parts[self.save_label_index] = (
                base_class_logits[..., self.save_label_index] + save_delta
            )
        if self.restart_conditioner is not None:
            assert self.restart_family_index is not None
            restart_evidence = rgb_logits[..., self.restart_family_index].sigmoid().detach()
            restart_state = state_logits[..., self.restart_label_indices].sigmoid().detach()
            restart_delta = self.restart_conditioner(torch.cat([
                features, restart_evidence.unsqueeze(-1), restart_state,
            ], dim=-1))
            for local_index, label_index in enumerate(self.restart_label_indices):
                class_parts[label_index] = base_class_logits[..., label_index] + restart_delta[..., local_index]
        class_logits = torch.stack(class_parts, dim=-1)

        return {
            "rgb_logits": rgb_logits,
            "logits": logits,
            "entity_delta": entity_delta,
            "offsets": self.offset_head(features),
            "quality_logits": self.quality_head(features),
            "base_class_logits": base_class_logits,
            "state_logits": state_logits,
            "class_logits": class_logits,
            "shot_condition_max": shot_condition_max,
            "shot_condition_lag_seconds": shot_condition_lag_seconds,
            "features": features,
        }
