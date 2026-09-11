"""Frame-count-safe multi-rate dataset."""

from __future__ import annotations

from .goal_multirate_data import MultiRateGoalBlockDataset


class SafeMultiRateGoalBlockDataset(MultiRateGoalBlockDataset):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        for video,(motion,camera,audio) in tuple(self.high.items()):
            length=min(len(motion),len(camera),len(audio));self.high[video]=(motion[:length],camera[:length],audio[:length])

