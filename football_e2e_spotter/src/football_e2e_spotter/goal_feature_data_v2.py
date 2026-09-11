"""Class-aware schedule: every GT once, rare classes receive extra views."""

from __future__ import annotations

import random

from .goal_annotations import LABELS
from .goal_feature_data import GoalFeatureWindowDataset, ScheduledWindow


class ClassAwareGoalFeatureWindowDataset(GoalFeatureWindowDataset):
    def __init__(self, *args, minimum_focus_per_class: int = 600, **kwargs) -> None:
        self.minimum_focus_per_class = int(minimum_focus_per_class)
        super().__init__(*args, **kwargs)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        rng = random.Random(self.seed + 1_000_003 * self.epoch)
        by_label = {label: [] for label in LABELS}
        for video_id, events in self.events.items():
            for ordinal, event in enumerate(events):
                if event.accepted:
                    by_label[event.label].append(ScheduledWindow(video_id, event, ordinal))
        positives = []
        for label, windows in by_label.items():
            positives.extend(windows)  # hard guarantee: every GT appears once
            if windows:
                for extra_index in range(max(self.minimum_focus_per_class - len(windows), 0)):
                    positives.append(windows[extra_index % len(windows)])
        rng.shuffle(positives)
        video_ids = sorted(self.items)
        backgrounds = [
            ScheduledWindow(video_ids[index % len(video_ids)], None, index)
            for index in range(int(round(len(positives) * self.background_ratio)))
        ]
        rng.shuffle(backgrounds)
        self.schedule = positives + backgrounds
        rng.shuffle(self.schedule)

