#!/usr/bin/env python
"""v2 entrypoint: recall-preserving loss and supervised review duration."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "football_e2e_spotter" / "src"
for path in (ROOT, PKG, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_goal_retriever as training  # noqa: E402
from football_e2e_spotter.goal_matching_v2 import retriever_loss  # noqa: E402


if __name__ == "__main__":
    training.retriever_loss = retriever_loss
    training.main()

