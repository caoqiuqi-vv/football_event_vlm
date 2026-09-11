from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class EventOntology:
    output_labels: tuple[str, ...] = (
        "shot", "save", "corner", "penalty", "freekick", "kickoff"
    )
    proposal_families: tuple[str, ...] = (
        "shot_chain", "restart", "generic_event"
    )

    @property
    def family_members(self) -> dict[str, tuple[str, ...]]:
        return {
            "shot_chain": ("shot", "save"),
            "restart": ("corner", "penalty", "freekick", "kickoff"),
            "generic_event": self.output_labels,
        }


def read_video_ids(path: str | Path) -> tuple[str, ...]:
    values = tuple(
        line.strip() for line in Path(path).read_text().splitlines() if line.strip()
    )
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate video ids in {path}")
    return values


def assert_disjoint_splits(*splits: Iterable[str]) -> None:
    seen: set[str] = set()
    for split in splits:
        current = set(split)
        overlap = seen & current
        if overlap:
            raise ValueError(f"video leakage across splits: {sorted(overlap)[:5]}")
        seen.update(current)
