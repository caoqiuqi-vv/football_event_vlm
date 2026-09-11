from __future__ import annotations

import torch
from torch import Tensor, nn


def _conv_block(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True),
    )


class RGBMotionEncoder(nn.Module):
    """Small full-frame RGB encoder; it never consumes detector boxes or tracks."""

    def __init__(self, output_dim: int = 256) -> None:
        super().__init__()
        self.spatial = nn.Sequential(
            _conv_block(6, 32, stride=2),
            _conv_block(32, 64, stride=2),
            _conv_block(64, 128, stride=2),
            _conv_block(128, 192, stride=2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(192, output_dim)
        self.temporal = nn.Sequential(
            nn.Conv1d(output_dim, output_dim, 5, padding=2, groups=output_dim),
            nn.GELU(),
            nn.Conv1d(output_dim, output_dim, 1),
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, frames: Tensor) -> Tensor:
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError("frames must have shape [batch, time, 3, height, width]")
        previous = torch.cat([frames[:, :1], frames[:, :-1]], dim=1)
        difference = frames - previous
        rgb_motion = torch.cat([frames, difference], dim=2)
        batch, steps = rgb_motion.shape[:2]
        encoded = self.spatial(rgb_motion.flatten(0, 1)).flatten(1)
        encoded = self.projection(encoded).reshape(batch, steps, -1)
        mixed = self.temporal(encoded.transpose(1, 2)).transpose(1, 2)
        return self.norm(encoded + mixed)

