#!/usr/bin/env python
"""Resume-safe v6: tolerate legitimately empty-positive DDP batches."""

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


OriginalDDP = training.DistributedDataParallel


def positive_sparse_ddp(*args, **kwargs):
    # A rank can legitimately receive an all-background batch.  In that batch
    # the temporal/family heads have no target and therefore no gradient.
    kwargs["find_unused_parameters"] = True
    return OriginalDDP(*args, **kwargs)


if __name__ == "__main__":
    training.GoalFeatureWindowDataset = ClassAwareGoalFeatureWindowDataset
    training.retriever_loss = retriever_loss
    training.evaluate_predictions = evaluate_predictions_adaptive
    training.DistributedDataParallel = positive_sparse_ddp
    training.main()

