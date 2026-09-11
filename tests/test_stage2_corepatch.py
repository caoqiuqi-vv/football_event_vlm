import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from football_stage2_corepatch import core_tokens,CoreEventReader

def test_exact_core_and_boundary():
    patches=torch.arange(3600*4,dtype=torch.float32).reshape(1,3600,4)
    for peak in [0,79,3520,3599,22*80+40]:
        logits=torch.zeros(1,3600);logits[0,peak]=100
        r=core_tokens(patches,logits)
        assert r['tokens'].shape==(1,26,4) and r['descriptors'].shape==(1,26,11)
        assert r['patch_indices'][0,4]==peak
        assert torch.equal(r['tokens'][0,4],patches[0,peak])
        for i,ix in enumerate(r['patch_indices'][0]):
            if ix>=0:assert torch.equal(r['tokens'][0,i],patches[0,ix])
        assert not r['valid'][0,0] if peak==0 else True
        assert torch.isfinite(r['tokens']).all()

class ToyTemporal(nn.Module):
    def __init__(self):super().__init__();self.mix=nn.Linear(512,512)
    def forward(self,x):return self.mix(x).tanh().mean(1)
class ToyEvent(nn.Module):
    def __init__(self):
        super().__init__();self.frame_proj=nn.Linear(2048,512);self.temporal=ToyTemporal();self.head=nn.Sequential(nn.LayerNorm(512),nn.Linear(512,3))

def test_initialization_null_gradients_and_invalid():
    torch.set_num_threads(1)
    for fusion in ['temporal','late']:
        torch.manual_seed(42);model=CoreEventReader({'hidden':32},fusion,event=ToyEvent())
        b={'global_features':torch.randn(2,4,2048),'tokens':torch.randn(2,4,26,1024),'descriptors':torch.randn(2,4,26,11),'valid':torch.ones(2,4,26,dtype=torch.bool),'anchor':torch.randn(2,3),'frame_times':torch.arange(4).float()[None].expand(2,-1)}
        assert torch.equal(model(**b)['logits'],b['anchor'])
        opt=torch.optim.Adam(model.adapter.parameters(),lr=.001)
        for i in range(3):
            opt.zero_grad();r=model(**b);r['logits'].square().mean().backward()
            if i>0:
                assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model.adapter.parameters())
            opt.step()
        assert all(p.grad is None for p in model.event.parameters())
        assert torch.equal(model(**b,enabled=False)['logits'],b['anchor'])
        empty={**b,'valid':torch.zeros_like(b['valid']),'tokens':torch.full_like(b['tokens'],float('nan')),'descriptors':torch.full_like(b['descriptors'],float('inf'))}
        assert torch.equal(model(**empty)['logits'],b['anchor'])
        mixed={**b,'valid':b['valid'].clone(),'tokens':b['tokens'].clone(),'descriptors':b['descriptors'].clone()}
        mixed['valid'][:,1,:]=False
        ordinary=model(**mixed)['logits']
        mixed['tokens'][:,1,:]=float('nan');mixed['descriptors'][:,1,:]=float('inf')
        assert torch.equal(model(**mixed)['logits'],ordinary)
        # No wrapped candidate from the last frame may reach the first frame.
        with torch.no_grad():
            r1=model.adapter(b['global_features'].new_zeros(2,4,512),b['tokens'],b['descriptors'],b['valid'],b['frame_times'])[0]
            altered=b['tokens'].clone();altered[:,-1]+=100
            r2=model.adapter(b['global_features'].new_zeros(2,4,512),altered,b['descriptors'],b['valid'],b['frame_times'])[0]
            assert torch.equal(r1[:,0],r2[:,0])

if __name__=="__main__":
    test_exact_core_and_boundary()
    test_initialization_null_gradients_and_invalid()
    print("corepatch CPU checks passed")
