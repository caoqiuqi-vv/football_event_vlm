#!/usr/bin/env python
"""Train 1 FPS DINO + 4 FPS residual-motion/audio block retriever."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];PKG=ROOT/"football_e2e_spotter/src"
for path in (ROOT,PKG,Path(__file__).resolve().parent):
    if str(path) not in sys.path:sys.path.insert(0,str(path))

parser=argparse.ArgumentParser(add_help=False);parser.add_argument("--motion4-root",required=True);parser.add_argument("--pixel-root",required=True);extra,remaining=parser.parse_known_args();sys.argv=[sys.argv[0],*remaining]

import train_goal_block_retriever as training  # noqa:E402
from football_e2e_spotter.goal_multirate_block import MultiRateGoalBlockRetriever  # noqa:E402
from football_e2e_spotter.goal_multirate_data import MultiRateGoalBlockDataset  # noqa:E402
from football_e2e_spotter.goal_block_loss_v2 import block_retriever_loss_v2  # noqa:E402


class Dataset(MultiRateGoalBlockDataset):
    def __init__(self,*args,**kwargs):super().__init__(*args,motion4_root=extra.motion4_root,pixel_root=extra.pixel_root,**kwargs)
    def __getitem__(self,index):
        item=super().__getitem__(index);item["motion"]=item["high"];return item


class Model(MultiRateGoalBlockRetriever):
    def forward(self,appearance,motion,audio,valid=None):return super().forward(appearance,motion,valid)


if __name__=="__main__":
    training.GoalBlockDataset=Dataset;training.GoalBlockRetriever=Model;training.block_retriever_loss=block_retriever_loss_v2;training.main()

