"""Joint temporal adaptation with an immutable original-model fallback."""
import copy
import torch
from torch import nn
from football_stage2_corepatch import load_event,EvidenceAdapter,CoreExtractor


class JointTemporalReader(nn.Module):
    def __init__(self,cfg,joint=True,event=None):
        super().__init__();self.cfg=cfg;self.joint=joint;self.epoch=0
        self.teacher=load_event(cfg) if event is None else event.requires_grad_(False).eval()
        self.temporal=copy.deepcopy(self.teacher.temporal)
        self.head=copy.deepcopy(self.teacher.head)
        self.adapter=EvidenceAdapter(hidden=cfg['hidden'])
        self.set_epoch(0)
    def set_epoch(self,epoch):
        self.epoch=epoch
        enabled=self.joint and epoch>self.cfg['adapter_warmup_epochs']
        self.temporal.requires_grad_(enabled);self.head.requires_grad_(enabled)
        self.teacher.requires_grad_(False).eval()
        self.temporal.eval();self.head.eval()
    def train(self,mode=True):
        super().train(mode)
        # Keep dropout disabled in both comparisons; eval does NOT disable grads.
        self.teacher.eval();self.temporal.eval();self.head.eval();return self
    def optimizer_groups(self):
        groups=[{'params':list(self.adapter.parameters()),'lr':self.cfg['lr'],'base_lr':self.cfg['lr'],'name':'adapter'}]
        if self.joint:groups.append({'params':list(self.temporal.parameters())+list(self.head.parameters()),'lr':self.cfg['temporal_lr'],'base_lr':self.cfg['temporal_lr'],'name':'temporal_and_classifier'})
        return groups
    def learned_state(self):
        return {'adapter':self.adapter.state_dict(),'temporal':self.temporal.state_dict(),'head':self.head.state_dict()}
    def load_learned(self,state):
        self.adapter.load_state_dict(state['adapter']);self.temporal.load_state_dict(state['temporal']);self.head.load_state_dict(state['head'])
    def forward(self,global_features,tokens,descriptors,valid,anchor,frame_times,enabled=True):
        if not enabled:valid=torch.zeros_like(valid)
        with torch.no_grad():
            frames=self.teacher.frame_proj(global_features.float())
            reference=self.teacher.head(self.teacher.temporal(frames))
        evidence,null,valid=self.adapter(frames,tokens,descriptors,valid,frame_times)
        predicted=self.head(self.temporal(frames+evidence))
        # Independent teacher is crucial: subtracting the *updated* student's
        # own no-evidence prediction would cancel useful temporal adaptation.
        delta=predicted-reference
        delta=torch.where(valid.flatten(1).any(1,keepdim=True),delta,torch.zeros_like(delta))
        return {'logits':anchor.float()+delta,'delta':delta,'null_mass':null,'valid':valid}


class JointTemporalEventModel(nn.Module):
    def __init__(self,checkpoint):
        super().__init__();s=torch.load(checkpoint,weights_only=True,map_location='cpu')
        self.cfg=s['config'];self.enabled=s['best']['enabled']
        self.extractor=CoreExtractor(self.cfg)
        self.reader=JointTemporalReader(self.cfg,joint=self.cfg['arms'][s['arm']]['joint'])
        self.reader.load_learned(s['learned']);self.reader.set_epoch(s['epoch']);self.eval()
        self.register_buffer('normal_thresholds',torch.tensor(s['best']['thresholds']))
        self.register_buffer('baseline_thresholds',torch.tensor(s['baseline_thresholds']))
    @torch.no_grad()
    def forward(self,frames,valid=None,frame_times=None):
        assert frames.dtype==torch.uint8 and frames.shape[1:]==(16,3,720,1280)
        rows=[];anchors=[]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            for clip in frames:
                r=self.extractor.extract_clip(clip,8);rows.append(r)
                anchors.append(self.extractor.event._global_branch_outputs(r['global_features'].float()[None])['logits'][0].float())
        tokens=torch.stack([r['stage1_tokens'].half().float() for r in rows]);desc=torch.stack([r['stage1_descriptors'].float() for r in rows]);mask=torch.stack([r['stage1_valid'] for r in rows])
        if valid is not None:mask &= valid.bool()
        if frame_times is None:frame_times=torch.linspace(0,10,16,device=tokens.device)[None].expand(len(rows),-1)
        with torch.autocast('cuda',enabled=False):
            result=self.reader(torch.stack([r['global_features'].float() for r in rows]),tokens,desc,mask,torch.stack(anchors),frame_times,enabled=self.enabled)
        thresholds=torch.where(result['valid'].flatten(1).any(1,keepdim=True),self.normal_thresholds[None],self.baseline_thresholds[None])
        result['probabilities']=result['logits'].sigmoid();result['thresholds']=thresholds;result['decisions']=result['probabilities']>=thresholds
        return result
