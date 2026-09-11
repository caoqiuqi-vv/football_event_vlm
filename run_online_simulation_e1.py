"""Compatibility entry point for the Online-simulation E1 experiment."""

from __future__ import annotations

import sys
import types

import train_football_events as base

# The online wrapper only needs a dict-like default container. Keep that small
# compatibility local to this entry point instead of adding a new dependency to
# the training environment.
compat = types.ModuleType("ml_collections")
compat.ConfigDict = base.ConfigDict
sys.modules.setdefault("ml_collections", compat)

import train_football_events_online_simulation  # noqa: E402,F401


if __name__ == "__main__":
    base.main()
