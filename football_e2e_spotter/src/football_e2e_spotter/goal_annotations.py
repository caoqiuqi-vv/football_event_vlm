"""Canonical five-class supervision for the goal-oriented retriever.

This module is deliberately independent from the legacy four-class parser.  A
penalty is a shot and simultaneous/nearby rows remain separate instances.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


LABELS = ("shot", "save", "freekick", "corner", "kickoff")
FAMILIES = ("shot_chain", "restart")
LABEL_MAP = {
    "射门": "shot",
    "其他射门类型": "shot",
    "点球": "shot",
    "扑救": "save",
    "任意球": "freekick",
    "角球": "corner",
    "中圈开球": "kickoff",
    "开球": "kickoff",
}
FAMILY_MAP = {
    "shot": "shot_chain",
    "save": "shot_chain",
    "freekick": "restart",
    "corner": "restart",
    "kickoff": "restart",
}


@dataclass(frozen=True)
class GoalEvent:
    label: str
    timestamp: float
    event_id: str
    accepted: bool
    raw_label: str

    @property
    def family(self) -> str:
        return FAMILY_MAP[self.label]


def parse_timestamp(value: object) -> float | None:
    if isinstance(value, (int, float)):
        number = float(value)
        return number if number >= 0 else None
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        number = float(text)
    except ValueError:
        pieces = text.split(":")
        if len(pieces) not in (2, 3):
            return None
        try:
            numbers = [float(piece) for piece in pieces]
        except ValueError:
            return None
        if any(number < 0 for number in numbers):
            return None
        number = numbers[-1] + 60.0 * numbers[-2]
        if len(numbers) == 3:
            number += 3600.0 * numbers[0]
    return number if number >= 0 else None


def load_goal_annotations(path: str | Path) -> tuple[GoalEvent, ...]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("data", payload.get("events", payload.get("annotations", [])))
    if not isinstance(payload, list):
        raise ValueError(f"annotation payload is not a list: {source}")
    events: list[GoalEvent] = []
    for row_index, item in enumerate(payload):
        if not isinstance(item, dict):
            continue
        raw_label = str(item.get("label", item.get("event_type", ""))).strip()
        label = LABEL_MAP.get(raw_label)
        if label is None:
            continue
        timestamp = parse_timestamp(item.get("startTime", item.get("timestamp", item.get("time_sec"))))
        if timestamp is None:
            continue
        events.append(GoalEvent(
            label=label,
            timestamp=timestamp,
            event_id=str(item.get("id", f"{source.stem}_{row_index:05d}")),
            accepted=item.get("label_correct") is not False,
            raw_label=raw_label,
        ))
    return tuple(sorted(events, key=lambda event: (event.timestamp, event.event_id)))

