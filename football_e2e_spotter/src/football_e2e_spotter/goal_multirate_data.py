"""Aligned 1/4 FPS feature windows for the multi-rate block retriever."""

from __future__ import annotations

import numpy as np
import torch

from .goal_block_data import GoalBlockDataset


class MultiRateGoalBlockDataset(GoalBlockDataset):
    def __init__(self,*args,motion4_root,pixel_root,**kwargs):
        super().__init__(*args,**kwargs);self.motion4_root=self.feature_root.parent.parent.__class__(motion4_root)/self.feature_root.name;self.pixel_root=self.feature_root.parent.parent.__class__(pixel_root)/self.feature_root.name
        self.high={}
        for video in self.timelines:
            motion=np.load(self.motion4_root/video/"motion4.npy",mmap_mode="r");camera=np.load(self.motion4_root/video/"camera4.npy",mmap_mode="r");audio=np.load(self.pixel_root/video/"audio_logmel.npy",mmap_mode="r");self.high[video]=(motion,camera,audio)

    def __getitem__(self,index):
        item=super().__getitem__(index);video,core_start=self.items[index];context_start=core_start-self.geometry.context_left;motion,camera,audio=self.high[video]
        source=np.floor(context_start*4+np.arange(720)).astype(np.int64);valid=(source>=0)&(source<len(motion));clipped=np.clip(source,0,max(len(motion)-1,0))
        values=[np.asarray(x[clipped],dtype=np.float32) for x in (motion,camera,audio)]
        high=np.concatenate(values,axis=-1);high[~valid]=0.;item["high"]=torch.from_numpy(high.copy());item["high_valid"]=torch.from_numpy(valid);return item

