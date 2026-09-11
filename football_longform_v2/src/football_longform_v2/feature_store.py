from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class AlignedTimeline:
    timestamps: Tensor
    context: Tensor
    motion: Tensor
    context_valid: Tensor
    motion_valid: Tensor
    entity: Tensor | None = None
    entity_valid: Tensor | None = None


def align_feature_stream(
    source_times: Tensor,
    source_features: Tensor,
    target_times: Tensor,
    *,
    max_gap_seconds: float | None = None,
) -> tuple[Tensor, Tensor]:
    """Linearly align a timestamped feature stream to a common timeline."""
    if source_times.ndim != 1 or target_times.ndim != 1:
        raise ValueError("source_times and target_times must be 1D")
    if source_features.ndim != 2 or source_features.shape[0] != source_times.numel():
        raise ValueError("source_features must have shape [source_time, feature]")
    if source_times.numel() == 0:
        shape = (target_times.numel(), source_features.shape[-1])
        return source_features.new_zeros(shape), torch.zeros_like(target_times, dtype=torch.bool)
    if source_times.numel() > 1 and not torch.all(source_times[1:] >= source_times[:-1]):
        raise ValueError("source_times must be sorted")

    right = torch.searchsorted(source_times, target_times).clamp(max=source_times.numel() - 1)
    left = (right - 1).clamp(min=0)
    left_t = source_times[left]
    right_t = source_times[right]
    denominator = (right_t - left_t).clamp_min(1e-8)
    weight = ((target_times - left_t) / denominator).clamp(0.0, 1.0).unsqueeze(-1)
    aligned = source_features[left] + weight * (source_features[right] - source_features[left])
    nearest_gap = torch.minimum((target_times - left_t).abs(), (right_t - target_times).abs())
    valid = torch.ones_like(target_times, dtype=torch.bool)
    if max_gap_seconds is not None:
        valid = nearest_gap <= float(max_gap_seconds)
        aligned = aligned * valid.unsqueeze(-1).to(aligned.dtype)
    return aligned, valid


def load_aligned_npz(path: str | Path) -> AlignedTimeline:
    """Load the stable on-disk contract used by locator training."""
    with np.load(Path(path), allow_pickle=False) as data:
        required = ("timestamps", "context", "motion", "context_valid", "motion_valid")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"timeline cache is missing keys: {missing}")
        entity = torch.from_numpy(data["entity"]).float() if "entity" in data else None
        entity_valid = (
            torch.from_numpy(data["entity_valid"]).bool() if "entity_valid" in data else None
        )
        return AlignedTimeline(
            timestamps=torch.from_numpy(data["timestamps"]).float(),
            context=torch.from_numpy(data["context"]).float(),
            motion=torch.from_numpy(data["motion"]).float(),
            context_valid=torch.from_numpy(data["context_valid"]).bool(),
            motion_valid=torch.from_numpy(data["motion_valid"]).bool(),
            entity=entity,
            entity_valid=entity_valid,
        )



def read_timeline_provenance(path: str | Path) -> dict[str, str]:
    """Read scalar provenance without materializing the large feature arrays."""
    with np.load(Path(path), allow_pickle=False) as data:
        required = ("backbone_arch", "backbone_id", "backbone_weights", "source_video", "config_sha256", "weights_sha256")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"timeline cache is missing provenance keys: {missing}")
        return {key: str(np.asarray(data[key]).item()) for key in required}


def assert_timeline_provenance(
    path: str | Path, *, expected_backbone_arch: str, expected_backbone_id: str,
    expected_source_video: str | Path, expected_config_sha256: str,
    expected_weights_sha256: str,
) -> None:
    provenance = read_timeline_provenance(path)
    expected = {
        "backbone_arch": str(expected_backbone_arch),
        "backbone_id": str(expected_backbone_id),
        "source_video": str(Path(expected_source_video).resolve()),
        "config_sha256": str(expected_config_sha256),
        "weights_sha256": str(expected_weights_sha256),
    }
    errors = {key: {"expected": value, "actual": provenance.get(key)}
              for key, value in expected.items() if provenance.get(key) != value}
    if errors:
        raise ValueError(f"timeline provenance mismatch for {path}: {errors}")
