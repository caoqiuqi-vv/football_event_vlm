"""CPU/Gloo regression for padding, accumulation, warmup and synchronized readers.

Run: python -m torch.distributed.run --standalone --nproc_per_node=2 tests/stage2_online_ddp_smoke.py
This is a synthetic engineering check, not a football accuracy experiment.
"""
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from contextlib import nullcontext
from football_stage2_joint import JointTemporalReader
from football_events.stage2.online import shard_order, paired_forward_batch, optimizer_step


class Temporal(nn.Module):
    def __init__(self):
        super().__init__()
        self.mix=nn.Linear(512,512)
    def forward(self,x):
        return self.mix(x).tanh().mean(1)


class Event(nn.Module):
    def __init__(self):
        super().__init__()
        self.frame_proj=nn.Linear(2048,512)
        self.temporal=Temporal()
        self.head=nn.Linear(512,3)


def main():
    torch.set_num_threads(1)
    dist.init_process_group('gloo')
    rank,world=dist.get_rank(),dist.get_world_size()
    assert world==2
    cfg=dict(hidden=32,adapter_warmup_epochs=0,lr=2e-4,temporal_lr=2e-5,warmup_optimizer_steps=1)
    torch.manual_seed(7)
    reader=JointTemporalReader(cfg,event=Event());reader.set_epoch(1)
    model=DistributedDataParallel(reader)
    opt=torch.optim.AdamW(reader.optimizer_groups(),weight_decay=.01)
    original_head={k:v.clone() for k,v in reader.head.state_dict().items()}
    original_teacher={k:v.clone() for k,v in reader.teacher.state_dict().items()}
    data=dict(global_features=torch.randn(9,2,2048),tokens=torch.randn(9,2,26,1024),
              descriptors=torch.randn(9,2,26,11),valid=torch.ones(9,2,26,dtype=torch.bool),
              anchor=torch.randn(9,3),frame_times=torch.tensor([[0.,1.]]).expand(9,-1))
    labels=torch.randn(9,3).sigmoid()
    order,local=shard_order(9,42,world,rank,2)
    parts=[None]*world;dist.all_gather_object(parts,local)
    real=[i for part in parts for i in part if i>=0]
    assert sorted(real)==list(range(9)) and sum(i<0 for part in parts for i in part)==3
    updates=0
    for step in range(len(local)//2):
        ids=torch.tensor(local[step*2:(step+1)*2]);keep=ids>=0;safe=ids.clamp_min(0)
        batch={k:v[safe].clone() for k,v in data.items()};batch['valid'] &= keep[:,None,None]
        group=step//2;group_ids=order[group*8:(group+1)*8];count=int((group_ids>=0).sum())
        boundary=(step+1)%2==0 or step+1==len(local)//2
        with model.no_sync() if not boundary else nullcontext():
            result=model(**paired_forward_batch(batch))
            loss=((result['logits'][:2]-labels[safe]).square().mean(1)*keep).sum()*world/count
            loss += .01*(result['delta'][2:].square().mean(1)*(keep & keep.roll(1,0))).sum()*world/count
            loss.backward()
        if boundary:
            # DDP must produce identical reduced gradients, even when a rank has only padding.
            for parameter in reader.parameters():
                if parameter.requires_grad:
                    assert parameter.grad is not None
                    other=parameter.grad.clone();dist.broadcast(other,src=0)
                    assert torch.equal(parameter.grad,other)
            optimizer_step(reader,opt,cfg,updates,2);updates+=1
            if updates==1:
                assert all(torch.equal(v,original_head[k]) for k,v in reader.head.state_dict().items())
                assert all(p not in opt.state for p in opt.param_groups[1]['params'])
    assert updates==2
    assert any(not torch.equal(v,original_head[k]) for k,v in reader.head.state_dict().items())
    assert all(torch.equal(v,original_teacher[k]) for k,v in reader.teacher.state_dict().items())
    for parameter in reader.parameters():
        other=parameter.detach().clone();dist.broadcast(other,src=0)
        assert torch.equal(parameter,other)
    if rank==0:
        print('PASS: unpadded real-row coverage, accumulation synchronization, all-padding rank, warmup optimizer state, joint head update, immutable teacher',flush=True)
    dist.destroy_process_group()

if __name__=='__main__':main()
