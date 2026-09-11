#!/usr/bin/env python
"""Final v5 entrypoint for the goal retriever experiment."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "football_e2e_spotter" / "src"
for path in (ROOT, PKG, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_goal_retriever as training  # noqa: E402
from football_e2e_spotter.goal_feature_data_v2 import ClassAwareGoalFeatureWindowDataset  # noqa: E402
from football_e2e_spotter.goal_matching_v3 import retriever_loss  # noqa: E402
from football_e2e_spotter.goal_retriever_eval_v2 import evaluate_predictions_adaptive  # noqa: E402


if __name__ == "__main__":
    training.GoalFeatureWindowDataset = ClassAwareGoalFeatureWindowDataset
    training.retriever_loss = retriever_loss
    training.evaluate_predictions = evaluate_predictions_adaptive
    training.main()

