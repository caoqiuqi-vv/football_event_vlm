"""Shared cache contract for football long-video score interpretation."""

from __future__ import annotations

from typing import Any, Mapping


SCORE_SEMANTICS_VERSION = "final_clip_logits_v2"


def has_current_score_semantics(summary: Mapping[str, Any]) -> bool:
    return summary.get("score_semantics_version") == SCORE_SEMANTICS_VERSION
