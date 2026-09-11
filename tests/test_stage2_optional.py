import torch
from football_stage2_optional import OptionalEvidence

def inputs():
 torch.manual_seed(17)
 return dict(global_features=torch.randn(3,5,64),tokens=torch.randn(3,5,4,32),descriptors=torch.randn(3,5,4,7),valid=torch.ones(3,5,4,dtype=torch.bool),anchor=torch.randn(3,3))

def test_empty_and_disabled_exact_anchor():
 m=OptionalEvidence(dim=32,global_dim=64,hidden=32);x=inputs();x['valid'].zero_();r=m(**x)
 assert torch.equal(r['logits'],x['anchor']);assert torch.equal(r['null_mass'],torch.ones_like(r['null_mass']))
 x['valid'].fill_(True);assert torch.equal(m(**x,enabled=False)['logits'],x['anchor'])

def test_invalid_content_cannot_leak():
 m=OptionalEvidence(dim=32,global_dim=64,hidden=32).eval();x=inputs();x['valid'][:,:,1:]=False;a=m(**x)
 x['tokens'][:,:,1:]=torch.nan;x['descriptors'][:,:,1:]=torch.inf;b=m(**x)
 assert torch.equal(a['logits'],b['logits']);assert torch.isfinite(b['logits']).all()
 x['tokens'].fill_(float('nan'));assert torch.equal(m(**x)['logits'],x['anchor'])

def test_partial_absence_and_gradient_and_bound():
 m=OptionalEvidence(dim=32,global_dim=64,hidden=32);x=inputs();x['valid'][0].zero_();x['valid'][1,2:].zero_();r=m(**x)
 assert torch.equal(r['logits'][0],x['anchor'][0]);assert (r['delta'].abs()<=1.5).all();assert torch.isfinite(r['logits']).all()
 torch.nn.functional.binary_cross_entropy_with_logits(r['logits'],torch.ones(3,3)).backward()
 for name in ['appearance.1.weight','key.weight','query.weight','readout.weight','null_bias','scale_logits','temporal.layers.0.linear1.weight']:
  p=dict(m.named_parameters())[name];assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0,name

if __name__=='__main__':
 torch.set_num_threads(1)
 for t in [test_empty_and_disabled_exact_anchor,test_invalid_content_cannot_leak,test_partial_absence_and_gradient_and_bound]:t();print(t.__name__,'PASS')

def test_event_curve_ties_and_many_windows_per_event():
 import numpy as np
 from football_stage2_metrics import EventCurves
 rows=[{'video_id':'v','start_sec':s,'end_sec':s+10,'label_mask':[1,1,1]} for s in [0,5,30]]
 meta={'videos':{'v':{'label_mask':[1,1,1],'duration':3600}},'gt':{'v':{k:[7.] for k in ['shot','save','set_piece']}}}
 e=EventCurves(rows,meta);p=np.array([[.8]*3,[.8]*3,[.9]*3]);th,m=e.tune(p,[1,1,1]);assert th==[.8,.8,.8]
 assert m['shot']['recall']==1. and m['shot']['precision']==2/3 and m['shot']['false_positive_windows']==1
 assert e.evaluate(p,[1,1,1])['shot']['predicted_windows']==0
