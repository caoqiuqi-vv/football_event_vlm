"""CPU orchestration smoke with synthetic frames; never claims DINO/GPU validation."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from football_stage2_joint import JointTemporalReader
from football_events.stage2 import online


class Temporal(nn.Module):
    def __init__(self):
        super().__init__();self.linear=nn.Linear(512,512)
    def forward(self,x):return self.linear(x).tanh().mean(1)


class Event(nn.Module):
    def __init__(self):
        super().__init__();self.frame_proj=nn.Linear(2048,512);self.temporal=Temporal();self.head=nn.Linear(512,3)


class Extractor(nn.Module):
    pass


def check():
    torch.set_num_threads(1);torch.manual_seed(7);event=Event().eval()
    original_tensor=torch.tensor
    def tensor(*args,**kwargs):
        if kwargs.get('device')=='cuda':kwargs['device']='cpu'
        return original_tensor(*args,**kwargs)
    def reader_factory(cfg,joint=True):return JointTemporalReader(cfg,joint,event=copy.deepcopy(event))
    def batches(records,indices,cfg):
        for start in range(0,len(indices),cfg['micro_batch_size']):
            ids=indices[start:start+cfg['micro_batch_size']]
            yield original_tensor(ids,dtype=torch.float32)[:,None,None].expand(-1,16,1),original_tensor(ids)
    @torch.no_grad()
    def features(extractor,frames,times,chunk):
        value=frames[:,0,0].clamp_min(0)/10
        global_features=value[:,None,None].expand(-1,16,2048).clone()
        anchor=event.head(event.temporal(event.frame_proj(global_features)))
        return dict(global_features=global_features,anchor=anchor,frame_times=times,
                    tokens=value[:,None,None,None].expand(-1,16,26,1024).clone(),
                    descriptors=torch.zeros(len(value),16,26,11),valid=torch.ones(len(value),16,26,dtype=torch.bool))
    with tempfile.TemporaryDirectory() as temp:
        out=Path(temp)/'run';dummy=Path(temp)/'source.json';dummy.write_text('{}')
        cfg=dict(epochs=2,seed=42,world_size=1,micro_batch_size=2,accumulation_steps=1,frame_chunk=8,
                 hidden=16,adapter_warmup_epochs=0,warmup_optimizer_steps=1,lr=2e-4,temporal_lr=2e-5,
                 weight_decay=.01,clip_drop_probability=.15,frame_drop_probability=.2,candidate_drop_probability=.1,
                 pos_weight=[1.,2.5,4.],residual_l2_weight=.01,corrupt_consistency_weight=.05,
                 recall_floors=[.84,.8,.72],output_dir=str(out))
        for key in ['source_checkpoint','stage1_checkpoint','control_checkpoint','event_config','manifest_source']:cfg[key]=str(dummy)
        rows=[dict(video_id='v',start_sec=start,end_sec=start+10,label_mask=[1.,1.,1.],
                   labels=[float(start<10)]*3,frame_times=list(range(16))) for start in [0,5,30]]
        manifest=dict(splits={key:rows for key in ['train','calibration','development']},
                      videos={'v':{'duration':3600,'label_mask':[1,1,1]}},
                      gt={'v':{label:[7.] for label in ['shot','save','set_piece']}})
        with patch.object(torch.cuda,'set_device',lambda *args:None), \
             patch.object(torch.Tensor,'cuda',lambda self,*args,**kwargs:self), \
             patch.object(nn.Module,'cuda',lambda self,*args,**kwargs:self), \
             patch.object(torch,'tensor',tensor), \
             patch.object(online,'PositionPriorExtractor',lambda cfg:Extractor()), \
             patch.object(online,'JointTemporalReader',reader_factory), \
             patch.object(online,'loader',batches), \
             patch.object(online,'extract_batch',features):
            online.train(cfg,manifest)
            assert (out/'COMPLETE.json').exists() and (out/'FINAL_REPORT.md').exists()
            assert not (out/'cache').exists() and not (out/'arrays').exists()
            state=torch.load(out/'resume.pt',weights_only=True,map_location='cpu')
            assert state['epoch']==2 and state['optimizer_updates']==4
            assert set(state['learned'])=={'adapter','temporal','head'}
            assert all((out/f'epoch_{epoch:03d}.json').exists() for epoch in [1,2])
            stamp=(out/'resume.pt').stat().st_mtime_ns
            online.train(cfg,manifest)
            assert (out/'resume.pt').stat().st_mtime_ns==stamp
            final=json.loads((out/'FINAL_SUMMARY.json').read_text())
            assert final['feature_cache_used'] is False
    print('PASS: CPU synthetic online workflow, two epochs, predictions/report/checkpoints, completion reentry, no feature-cache artifacts')

if __name__=='__main__':check()
