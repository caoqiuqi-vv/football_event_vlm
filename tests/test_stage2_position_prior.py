"""Candidate safety, exact original core preservation, and extractor routing tests."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from football_stage2_position_prior import height_prior,select_peaks,tokens_at_peaks,PositionPriorExtractor
from football_stage2_corepatch import core_tokens

PRIOR={'mode':'soft_top','top_weight':.25,'top_end':.25,'full_weight_start':.4}


def check():
    torch.set_num_threads(1);torch.manual_seed(3)
    weights=height_prior(PRIOR,'cpu').reshape(45,80)
    assert weights.min()==.25 and weights.max()==1 and (weights>0).all()
    assert (weights[1:]>=weights[:-1]).all() and (weights[18:]==1).all()
    # Highest peak must survive, even at the edge. Only the second candidate changes.
    logits=torch.full((2,3600),-20.)
    logits[0,0]=10;logits[0,8*80+20]=9;logits[0,25*80+30]=8.5
    logits[1,44*80+79]=10;logits[1,30*80+30]=9
    selected=select_peaks(logits,PRIOR)
    assert selected[0].tolist()==[0,25*80+30]
    assert selected[1,0]==44*80+79
    plain=select_peaks(logits,{**PRIOR,'mode':'none'})
    assert plain[0].tolist()==[0,8*80+20]
    patches=torch.arange(3600).float()[None,:,None].expand(2,-1,3).clone()
    result=tokens_at_peaks(patches,logits,selected)
    for j in range(2):
        ix=j*13+4
        assert torch.equal(result['patch_indices'][:,ix],selected[:,j])
        assert torch.equal(result['tokens'][:,ix],patches[torch.arange(2),selected[:,j]])
    assert not result['valid'][0,:4].any()  # clipped upper-left 3x3 cells are invalid, not duplicated
    assert result['valid'][0,4] and torch.isfinite(result['descriptors']).all()
    # With a flat prior, gathering and every descriptor match the old extractor exactly.
    expected=core_tokens(patches,logits)
    actual=tokens_at_peaks(patches,logits,plain)
    assert all(torch.equal(actual[k],expected[k]) for k in expected)
    try:select_peaks(torch.full((1,3600),float('nan')),PRIOR)
    except ValueError:pass
    else:raise AssertionError('Nonfinite heatmaps must fail explicitly')
    # Distinguishable original/adapted branches catch accidentally reading adapted patches.
    class Block(nn.Module):
        def __init__(self,value):super().__init__();self.value=value
        def forward(self,x,rope):return x+self.value
    class Backbone(nn.Module):
        def __init__(self):super().__init__();self.blocks=nn.ModuleList([Block(9.)]);self.n_storage_tokens=0
        def prepare_tokens_with_masks(self,x):return torch.zeros(len(x),3601,3),(45,80)
        def rope_embed(self,**kw):return None
    class Head(nn.Module):
        def forward(self,x):return logits[:len(x),:,None]
    extractor=PositionPriorExtractor.__new__(PositionPriorExtractor);nn.Module.__init__(extractor)
    extractor.prior=PRIOR;extractor.rgb_mean=0.;extractor.rgb_std=1.
    extractor.backbone=Backbone();extractor.teacher_tail=nn.ModuleList([Block(2.)])
    extractor.start=0;extractor.adapt_head=Head();extractor.norm_tokens=lambda x:x
    output=extractor.extract_frames(torch.zeros(1,3,720,1280,dtype=torch.uint8))
    assert (output['stage1_tokens'][0,4]==2).all()
    assert (output['legacy_tokens']==9).all()
    assert (output['global_features']==2).all()
    print('PASS: soft-prior coverage, global safeguard, exact center/boundary, flat-prior equivalence, original-vs-adapted routing')

if __name__=='__main__':check()
