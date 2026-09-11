"""20-second retriever with 1 FPS DINO and independent 4 FPS action/audio."""

from __future__ import annotations

import torch
from torch import Tensor,nn

from .goal_annotations import LABELS
from .goal_block_retriever import BlockGeometry
from .goal_retriever import MultiScaleMemory,ResidualTemporalBlock


class MultiRateGoalBlockRetriever(nn.Module):
    def __init__(self,appearance_dim:int,high_dim:int=131,hidden_dim:int=384,dropout:float=.15,geometry:BlockGeometry=BlockGeometry()):
        super().__init__();self.geometry=geometry
        self.appearance=nn.Sequential(nn.LayerNorm(appearance_dim),nn.Linear(appearance_dim,hidden_dim))
        self.high=nn.Sequential(nn.LayerNorm(high_dim),nn.Linear(high_dim,hidden_dim),nn.GELU())
        self.high_tcn=nn.ModuleList(ResidualTemporalBlock(hidden_dim,d,dropout) for d in (1,2,4,8))
        self.high_down=nn.Conv1d(hidden_dim,hidden_dim,7,stride=4,padding=3)
        self.position=nn.Parameter(torch.randn(geometry.context_steps,hidden_dim)*.02);self.memory=MultiScaleMemory(hidden_dim,dropout)
        self.class_queries=nn.Parameter(torch.randn(len(LABELS),hidden_dim)*.02);self.any_query=nn.Parameter(torch.randn(1,hidden_dim)*.02);self.block_position=nn.Parameter(torch.randn(3,hidden_dim)*.02)
        self.attention=nn.MultiheadAttention(hidden_dim,8,dropout=dropout,batch_first=True);self.class_head=nn.Linear(hidden_dim,1);self.any_head=nn.Linear(hidden_dim,1)
        self.dense_head=nn.Linear(hidden_dim,len(LABELS));self.whistle_head=nn.Linear(hidden_dim,1)

    def forward(self,appearance:Tensor,high:Tensor,valid:Tensor|None=None)->dict[str,Tensor]:
        if appearance.shape[1]!=180 or high.shape[1]!=720: raise ValueError("expected appearance 180x1FPS and high 720x4FPS")
        high_hidden=self.high(high)
        for block in self.high_tcn: high_hidden=block(high_hidden)
        whistle=self.whistle_head(high_hidden).squeeze(-1)
        high_1fps=self.high_down(high_hidden.transpose(1,2)).transpose(1,2)
        memory=self.memory(self.appearance(appearance)+high_1fps+self.position.unsqueeze(0));core=memory[:,60:120]
        block_logits=[];any_logits=[]
        for index in range(3):
            tokens=core[:,20*index:20*(index+1)];cq=self.class_queries.unsqueeze(0).expand(len(appearance),-1,-1)+self.block_position[index]
            hidden,_=self.attention(cq,tokens,tokens,need_weights=False);block_logits.append(self.class_head(hidden).squeeze(-1))
            aq=self.any_query.unsqueeze(0).expand(len(appearance),-1,-1)+self.block_position[index];ah,_=self.attention(aq,tokens,tokens,need_weights=False);any_logits.append(self.any_head(ah).squeeze(-1).squeeze(-1))
        return {"block_logits":torch.stack(block_logits,1),"any_logits":torch.stack(any_logits,1),"dense_logits":self.dense_head(core),"whistle_logits":whistle[:,240:480],"memory":memory}

