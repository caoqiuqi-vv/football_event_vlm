from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor


LABELS = ("shot", "save", "corner", "freekick")
FAMILIES = ("shot_chain", "restart")
LABEL_MAP = {
    "射门": "shot", "其他射门类型": "shot", "扑救": "save",
    "角球": "corner", "任意球": "freekick",
}
FAMILY = {"shot": "shot_chain", "save": "shot_chain", "corner": "restart", "freekick": "restart"}


@dataclass(frozen=True)
class Event:
    label: str
    family: str
    timestamp: float


@dataclass(frozen=True)
class Supervision:
    accepted: tuple[Event, ...]
    rejected: tuple[Event, ...]


def parse_timestamp(value: object) -> float | None:
    if isinstance(value, (int, float)):
        value = float(value)
        return value if value >= 0 else None
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        value = float(text)
    except ValueError:
        pieces = text.split(":")
        if len(pieces) not in (2, 3):
            return None
        try:
            values = [float(piece) for piece in pieces]
        except ValueError:
            return None
        if any(item < 0 for item in values):
            return None
        value = values[-1] + 60 * values[-2]
        if len(values) == 3:
            value += 3600 * values[0]
    return value if value >= 0 else None


def load_annotations(path: str | Path) -> Supervision:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("data", payload.get("events", payload.get("annotations", [])))
    if not isinstance(payload, list):
        raise ValueError(f"annotation payload is not a list: {path}")
    accepted: list[Event] = []
    rejected: list[Event] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        label = LABEL_MAP.get(str(item.get("label", "")))
        if label is None:
            continue
        timestamp = parse_timestamp(item.get("startTime", item.get("timestamp")))
        if timestamp is None:
            continue
        event = Event(label, FAMILY[label], timestamp)
        (rejected if item.get("label_correct") is False else accepted).append(event)
    return Supervision(
        tuple(sorted(accepted, key=lambda item: item.timestamp)),
        tuple(sorted(rejected, key=lambda item: item.timestamp)),
    )


def gaussian_targets(
    timestamps: Tensor,
    events: tuple[Event, ...],
    names: tuple[str, ...],
    sigma_seconds: dict[str, float],
    *,
    by_family: bool = False,
) -> Tensor:
    targets = timestamps.new_zeros((timestamps.numel(), len(names)))
    for index, name in enumerate(names):
        times = [
            event.timestamp for event in events
            if (event.family if by_family else event.label) == name
        ]
        if not times:
            continue
        event_times = timestamps.new_tensor(times)
        distance = (timestamps[:, None] - event_times[None, :]).abs().amin(dim=1)
        sigma = max(float(sigma_seconds[name]), 1e-3)
        targets[:, index] = torch.exp(-0.5 * (distance / sigma).square())
    return targets


def supervision_mask(
    timestamps: Tensor,
    rejected: tuple[Event, ...],
    names: tuple[str, ...],
    radius_seconds: dict[str, float],
    *,
    by_family: bool = False,
) -> Tensor:
    mask = torch.ones((timestamps.numel(), len(names)), dtype=torch.bool)
    for index, name in enumerate(names):
        times = [
            event.timestamp for event in rejected
            if (event.family if by_family else event.label) == name
        ]
        if not times:
            continue
        event_times = timestamps.new_tensor(times)
        distance = (timestamps[:, None] - event_times[None, :]).abs().amin(dim=1)
        mask[:, index] = distance > max(float(radius_seconds[name]), 0.0)
    return mask


def nearest_offsets(
    timestamps: Tensor,
    events: tuple[Event, ...],
    labels: tuple[str, ...],
    max_seconds: float,
) -> Tensor:
    offsets = timestamps.new_zeros((timestamps.numel(), len(labels)))
    for index, label in enumerate(labels):
        times = [event.timestamp for event in events if event.label == label]
        if not times:
            continue
        delta = timestamps.new_tensor(times)[:, None] - timestamps[None, :]
        nearest = delta.abs().argmin(dim=0)
        offsets[:, index] = delta[nearest, torch.arange(timestamps.numel())].clamp(
            -float(max_seconds), float(max_seconds)
        )
    return offsets

