import sys,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from football_stage2_joint import JointTemporalReader
class Temporal(nn.Module):
    def __init__(self):super().__init__();self.proj=nn.Linear(512,512)
    def forward(self,x):return self.proj(x).tanh().mean(1)
class Event(nn.Module):
    def __init__(self):
        super().__init__();self.frame_proj=nn.Linear(2048,512);self.temporal=Temporal();self.head=nn.Sequential(nn.LayerNorm(512),nn.Linear(512,3))
def check():
    torch.set_num_threads(1);torch.manual_seed(7)
    cfg={'hidden':32,'adapter_warmup_epochs':1,'lr':2e-4,'temporal_lr':2e-5}
    for joint in [True,False]:
        model=JointTemporalReader(cfg,joint,event=Event());teacher={k:v.clone() for k,v in model.teacher.state_dict().items()};start={k:v.clone() for k,v in model.temporal.state_dict().items()}
        b={'global_features':torch.randn(2,4,2048),'tokens':torch.randn(2,4,26,1024),'descriptors':torch.randn(2,4,26,11),'valid':torch.ones(2,4,26,dtype=torch.bool),'anchor':torch.randn(2,3),'frame_times':torch.arange(4).float()[None].expand(2,-1)}
        with torch.no_grad():assert torch.allclose(model(**b)['logits'],b['anchor'],atol=1e-6,rtol=0)
        opt=torch.optim.AdamW(model.optimizer_groups());model.set_epoch(1);model.train()
        opt.zero_grad();model(**b)['logits'].square().mean().backward();opt.step()
        assert all(torch.equal(v,start[k]) for k,v in model.temporal.state_dict().items())
        model.set_epoch(2);opt.zero_grad();model(**b)['logits'].square().mean().backward()
        assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.temporal.parameters())==joint
        opt.step();changed=any(not torch.equal(v,start[k]) for k,v in model.temporal.state_dict().items());assert changed==joint
        assert all(torch.equal(v,teacher[k]) for k,v in model.teacher.state_dict().items())
        empty={**b,'valid':torch.zeros_like(b['valid']),'tokens':torch.full_like(b['tokens'],float('nan'))}
        assert torch.equal(model(**empty)['logits'],b['anchor'])
        clone=JointTemporalReader(cfg,joint,event=copy.deepcopy(model.teacher));clone.load_learned(model.learned_state());clone.set_epoch(2)
        assert torch.equal(clone(**b)['logits'],model(**b)['logits'])
    print('joint warmup/update/teacher/null/checkpoint checks passed')
if __name__=='__main__':check()
