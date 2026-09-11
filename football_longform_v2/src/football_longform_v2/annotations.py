from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor


LABEL_TO_CANONICAL = {
    "射门": "shot",
    "其他射门类型": "shot",
    "扑救": "save",
    "角球": "corner",
    "点球": "penalty",
    "任意球": "freekick",
    "中圈开球": "kickoff",
}
LABEL_TO_FAMILY = {
    "shot": "shot_chain",
    "save": "shot_chain",
    "corner": "restart",
    "penalty": "restart",
    "freekick": "restart",
    "kickoff": "restart",
}


@dataclass(frozen=True)
class Event:
    label: str
    family: str
    timestamp: float


@dataclass(frozen=True)
class EventAnnotations:
    """Accepted anchors plus reviewed rejected anchors that must never become negatives."""

    accepted: tuple[Event, ...]
    rejected: tuple[Event, ...]


def parse_timestamp(value: object) -> float | None:
    """Parse seconds or the reviewed export's ``HH:MM:SS.sss`` notation."""
    if isinstance(value, (int, float)):
        seconds = float(value)
        return seconds if seconds >= 0.0 else None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        seconds = float(text)
    except ValueError:
        pieces = text.split(":")
        if len(pieces) not in (2, 3):
            return None
        try:
            values = [float(piece) for piece in pieces]
        except ValueError:
            return None
        if any(number < 0.0 for number in values):
            return None
        if len(values) == 2:
            minutes, seconds = values
            seconds += 60.0 * minutes
        else:
            hours, minutes, seconds = values
            seconds += 3600.0 * hours + 60.0 * minutes
    return seconds if seconds >= 0.0 else None


def _items(payload: object) -> list[dict]:
    if isinstance(payload, dict):
        payload = payload.get("data", payload.get("events", payload.get("annotations", [])))
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def load_event_annotations(
    path: str | Path, *, require_reviewed: bool = False
) -> EventAnnotations:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    accepted: list[Event] = []
    rejected: list[Event] = []
    for item in _items(payload):
        label = LABEL_TO_CANONICAL.get(str(item.get("label", "")))
        if label is None:
            continue
        reviewed = item.get("label_correct")
        if require_reviewed and not isinstance(reviewed, bool):
            raise ValueError(f"mapped event is missing boolean label_correct in {path}")
        timestamp = parse_timestamp(item.get("startTime", item.get("timestamp")))
        if timestamp is None:
            continue
        event = Event(label, LABEL_TO_FAMILY[label], timestamp)
        (rejected if reviewed is False else accepted).append(event)
    return EventAnnotations(
        tuple(sorted(accepted, key=lambda event: event.timestamp)),
        tuple(sorted(rejected, key=lambda event: event.timestamp)),
    )


def load_events(path: str | Path) -> tuple[Event, ...]:
    """Load accepted event anchors; retained as the evaluation-facing API."""
    return load_event_annotations(path).accepted


def family_event_times(events: tuple[Event, ...], family: str) -> list[float]:
    """Return semantic proposal anchors, avoiding duplicate shot+save chain targets."""
    if family == "shot_chain":
        shot_times = sorted(event.timestamp for event in events if event.label == "shot")
        orphan_saves = sorted(
            event.timestamp for event in events
            if event.label == "save"
            and (not shot_times or min(abs(event.timestamp - shot) for shot in shot_times) > 5.0)
        )
        return sorted([*shot_times, *orphan_saves])
    if family == "restart":
        return sorted(event.timestamp for event in events if event.family == "restart")
    if family == "generic_event":
        restart_times = [event.timestamp for event in events if event.family == "restart"]
        return sorted([*family_event_times(events, "shot_chain"), *restart_times])
    return sorted(event.timestamp for event in events if event.family == family)


def build_family_targets(
    timestamps: Tensor,
    events: tuple[Event, ...],
    families: tuple[str, ...],
    *,
    sigma_seconds: dict[str, float] | None = None,
) -> tuple[Tensor, Tensor]:
    """Create dense Gaussian point targets and nearest-event offsets."""
    sigma_seconds = sigma_seconds or {
        "shot_chain": 1.0,
        "restart": 2.0,
        "generic_event": 1.0,
    }
    targets = timestamps.new_zeros((timestamps.numel(), len(families)))
    offsets = timestamps.new_zeros(targets.shape)
    for family_index, family in enumerate(families):
        family_events = family_event_times(events, family)
        if not family_events:
            continue
        event_times = timestamps.new_tensor(family_events)
        delta = event_times[:, None] - timestamps[None, :]
        nearest_index = delta.abs().argmin(dim=0)
        nearest_delta = delta[nearest_index, torch.arange(timestamps.numel())]
        sigma = max(float(sigma_seconds.get(family, 1.0)), 1e-3)
        targets[:, family_index] = torch.exp(-0.5 * (nearest_delta / sigma).square())
        offsets[:, family_index] = nearest_delta
    return targets, offsets


def build_family_supervision_mask(
    timestamps: Tensor,
    rejected_events: tuple[Event, ...],
    families: tuple[str, ...],
    *,
    ignore_radius_seconds: dict[str, float] | None = None,
) -> Tensor:
    """Mask reviewed false anchors so they cannot be learned as dense negatives."""
    ignore_radius_seconds = ignore_radius_seconds or {
        "shot_chain": 4.5,
        "restart": 8.0,
        "generic_event": 3.0,
    }
    mask = torch.ones(
        (timestamps.numel(), len(families)), dtype=torch.bool, device=timestamps.device
    )
    for family_index, family in enumerate(families):
        radius = max(float(ignore_radius_seconds.get(family, 0.0)), 0.0)
        if radius == 0.0:
            continue
        rejected_times = [
            event.timestamp
            for event in rejected_events
            if event.family == family or family == "generic_event"
        ]
        if not rejected_times:
            continue
        event_times = timestamps.new_tensor(rejected_times)
        near_rejected = (timestamps[:, None] - event_times[None, :]).abs().amin(dim=1) <= radius
        mask[near_rejected, family_index] = False
    return mask


def build_label_targets(
    timestamps: Tensor,
    events: tuple[Event, ...],
    labels: tuple[str, ...],
    *,
    sigma_seconds: dict[str, float] | None = None,
) -> Tensor:
    """Dense class-specific point targets used by the conditional verifier heads."""
    sigma_seconds = sigma_seconds or {
        "shot": 0.8, "save": 1.0, "corner": 2.0, "penalty": 2.0,
        "freekick": 2.0, "kickoff": 2.0,
    }
    targets = timestamps.new_zeros((timestamps.numel(), len(labels)))
    for label_index, label in enumerate(labels):
        label_times = [event.timestamp for event in events if event.label == label]
        if not label_times:
            continue
        event_times = timestamps.new_tensor(label_times)
        nearest = (event_times[:, None] - timestamps[None, :]).abs().amin(dim=0)
        sigma = max(float(sigma_seconds.get(label, 1.0)), 1e-3)
        targets[:, label_index] = torch.exp(-0.5 * (nearest / sigma).square())
    return targets


def build_label_supervision_mask(
    timestamps: Tensor,
    rejected_events: tuple[Event, ...],
    labels: tuple[str, ...],
    *,
    ignore_radius_seconds: dict[str, float] | None = None,
) -> Tensor:
    """Class-specific ignore mask around explicitly rejected annotation candidates."""
    ignore_radius_seconds = ignore_radius_seconds or {
        "shot": 4.5, "save": 4.5, "corner": 8.0, "penalty": 8.0,
        "freekick": 8.0, "kickoff": 8.0,
    }
    mask = torch.ones(
        (timestamps.numel(), len(labels)), dtype=torch.bool, device=timestamps.device
    )
    for label_index, label in enumerate(labels):
        radius = max(float(ignore_radius_seconds.get(label, 0.0)), 0.0)
        rejected_times = [event.timestamp for event in rejected_events if event.label == label]
        if radius == 0.0 or not rejected_times:
            continue
        event_times = timestamps.new_tensor(rejected_times)
        near_rejected = (timestamps[:, None] - event_times[None, :]).abs().amin(dim=1) <= radius
        mask[near_rejected, label_index] = False
    return mask
