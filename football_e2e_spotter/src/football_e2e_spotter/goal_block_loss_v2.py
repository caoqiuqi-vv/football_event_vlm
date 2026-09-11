"""Multi-rate block loss with 4 FPS whistle MIL."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F
from .goal_block_loss import asymmetric_focal


def block_retriever_loss_v2(output:dict[str,Tensor],block_targets:Tensor,dense_targets:Tensor):
    counts=block_targets.sum((0,1)).clamp_min(1);weight=(counts.sum()/counts).sqrt().clamp(1,6).reshape(1,1,-1)
    block=asymmetric_focal(output["block_logits"],block_targets,weight);any_event=asymmetric_focal(output["any_logits"],block_targets.amax(-1));dense=asymmetric_focal(output["dense_logits"],dense_targets)
    restart=block_targets[...,2:].amax(-1); whistle=output["whistle_logits"].reshape(len(block_targets),3,80).logsumexp(-1)-torch.tensor(80.,device=block_targets.device).log()
    whistle_loss=(restart*F.softplus(-whistle)+.08*(1-restart)*F.softplus(whistle)).mean();total=block+.35*any_event+.25*dense+.1*whistle_loss
    return total,{"loss":float(total.detach()),"block":float(block.detach()),"any":float(any_event.detach()),"dense":float(dense.detach()),"whistle":float(whistle_loss.detach())}

