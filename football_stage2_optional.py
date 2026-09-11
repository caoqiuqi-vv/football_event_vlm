"""Optional local visual evidence; object absence is a normal input state."""
from __future__ import annotations
import json,math
from pathlib import Path
from collections import OrderedDict
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.ops import roi_align
from scripts.train_football_localization_stage1 import LocalizationModel
import train_football_events as football


class OptionalEvidence(nn.Module):
    """Global anchor plus candidate-dependent residual, with a zero-value null slot.

    Saliency descriptors are NOT calibrated object-presence probabilities.
    No-candidate input returns anchor logits exactly, independently of parameters.
    """
    def __init__(self,dim=1024,global_dim=2048,hidden=128,classes=3,max_delta=1.5):
        super().__init__();self.max_delta=max_delta
        self.context=nn.Sequential(nn.LayerNorm(global_dim),nn.Linear(global_dim,hidden))
        self.appearance=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,hidden))
        self.position=nn.Linear(7,hidden);self.key=nn.Linear(hidden,hidden,bias=False)
        self.query=nn.Linear(hidden,hidden,bias=False);self.value=nn.Linear(hidden,hidden,bias=False)
        self.null_key=nn.Parameter(torch.zeros(hidden));self.null_bias=nn.Parameter(torch.tensor(1.5))
        layer=nn.TransformerEncoderLayer(hidden,4,hidden*2,dropout=0.,batch_first=True,norm_first=True)
        self.temporal=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.readout=nn.Linear(hidden,classes,bias=False)
        self.scale_logits=nn.Parameter(torch.full((classes,),-1.5))
        self.time=nn.Linear(3,hidden,bias=False)

    def forward(self,global_features,tokens,descriptors,valid,anchor,frame_times=None,enabled=True):
        valid=valid.bool() & torch.isfinite(tokens).all(-1) & torch.isfinite(descriptors).all(-1)
        if not enabled:valid=torch.zeros_like(valid)
        # Sanitize before projections: masking an eventual NaN score is insufficient.
        safe=torch.where(valid[...,None],tokens,torch.zeros_like(tokens)).float()
        desc=torch.where(valid[...,None],descriptors,torch.zeros_like(descriptors)).float()
        ctx=self.context(global_features.float());local=self.appearance(safe)+self.position(desc)
        q=self.query(ctx);scores=(q[:,:,None]*self.key(local)).sum(-1)/math.sqrt(q.shape[-1])
        scores=scores.masked_fill(~valid,-torch.inf)
        null=(q*self.null_key).sum(-1,keepdim=True)/math.sqrt(q.shape[-1])+self.null_bias
        attn=torch.cat([scores,null],-1).softmax(-1);mass=attn[...,:-1].sum(-1)
        pooled=(attn[...,:-1,None]*self.value(local)).sum(-2)
        b,t,_=ctx.shape
        if frame_times is None:times=torch.linspace(0,1,t,device=ctx.device)[None].expand(b,-1)
        else:
            span=(frame_times[:,-1:]-frame_times[:,:1]).clamp_min(1e-6);times=(frame_times-frame_times[:,:1])/span
        ctx=ctx+self.time(torch.stack([times,torch.sin(math.pi*times),torch.cos(math.pi*times)],-1).float())
        paired=self.temporal(torch.cat([ctx+pooled,ctx],0))
        raw=self.readout(paired[:b].mean(1))-self.readout(paired[b:].mean(1))
        delta=self.max_delta*torch.sigmoid(self.scale_logits)*torch.tanh(raw)*mass.mean(1,keepdim=True)
        available=valid.flatten(1).any(1,keepdim=True)
        delta=torch.where(available,delta,torch.zeros_like(delta))
        return {'logits':anchor.float()+delta,'delta':delta,'null_mass':attn[...,-1],
                'valid':valid,'attention':attn,'scale':torch.sigmoid(self.scale_logits)}


def candidate_features(patches,logits,uniform=False):
    """Two separated candidate centers × two context sizes; no existence assertion."""
    b,n,d=patches.shape;assert n==3600
    prob=logits.float().softmax(1);work=logits.float().reshape(b,45,80).clone()
    yy,xx=torch.meshgrid(torch.arange(45,device=patches.device),torch.arange(80,device=patches.device),indexing='ij')
    centers=[];gaps=[]
    for j in range(2):
        if uniform:
            # Fixed space-filling control, same candidate count and context sizes.
            cx=torch.full((b,),.3 if j==0 else .7,device=patches.device)*80
            cy=torch.full((b,),.5,device=patches.device)*45
        else:
            peak=work.flatten(1).argmax(1);cx=(peak%80).float()+.5;cy=(peak//80).float()+.5
            suppress=((xx[None]+.5-cx[:,None,None]).abs()<=3)&((yy[None]+.5-cy[:,None,None]).abs()<=3)
            work.masked_fill_(suppress,-torch.inf)
        centers.append(torch.stack([cx,cy],-1))
    entropy=-(prob*prob.clamp_min(1e-12).log()).sum(1)/math.log(n)
    top=prob.topk(2,dim=1).values;peakp=top[:,0];gap=top[:,0]-top[:,1]
    fmap=patches.transpose(1,2).reshape(b,d,45,80).float();features=[];descriptors=[]
    for xy in centers:
        for size in [5.,11.]:
            lo=(xy-size/2).clamp_min(0);hi=torch.minimum(xy+size/2,xy.new_tensor([80.,45.]))
            rois=torch.cat([torch.arange(b,device=xy.device)[:,None],lo,hi],-1)
            features.append(roi_align(fmap,rois,(2,2),spatial_scale=1.,sampling_ratio=2,aligned=True).mean((-2,-1)))
            saliency=torch.stack([entropy,peakp,gap],-1) if not uniform else torch.zeros(b,3,device=xy.device)
            descriptors.append(torch.cat([xy/xy.new_tensor([80.,45.]),torch.full((b,1),size/11.,device=xy.device),saliency,torch.ones(b,1,device=xy.device)],-1))
    return torch.stack(features,1),torch.stack(descriptors,1)


class FrozenStage2Extractor(LocalizationModel):
    def __init__(self,cfg):
        super().__init__({'checkpoint':cfg['source_checkpoint'],'temperature':2.})
        for arm,key in [('adapt','stage1_checkpoint'),('control','control_checkpoint')]:
            s=torch.load(cfg[key],weights_only=True,map_location='cpu')
            if arm=='adapt':
                with torch.no_grad():
                    params=dict(self.backbone.named_parameters())
                    assert set(s['backbone_lora'])=={n for n,p in params.items() if p.requires_grad}
                    for n,v in s['backbone_lora'].items():params[n].copy_(v)
            getattr(self,arm+'_head').load_state_dict(s[arm+'_head'])
        source=torch.load(cfg['source_checkpoint'],weights_only=True,map_location='cpu',mmap=True)
        event_cfg=football.to_config(json.loads(Path(cfg['event_config']).read_text()))
        football.configure_label_schema(event_cfg)
        self.event=football.make_model(event_cfg,use_cached_features=True,device=torch.device('cpu'))
        non_backbone={k:v for k,v in source['model'].items() if not k.startswith('backbone.')}
        self.event.load_state_dict(non_backbone,strict=True)
        self.requires_grad_(False);self.eval()

    @torch.no_grad()
    def extract_global_pair(self,frames):
        """Return matched original/adapted global frame features."""
        assert frames.shape[-2:]==(720,1280)
        x=(frames.float()/255-self.rgb_mean)/self.rgb_std
        x,(h,w)=self.backbone.prepare_tokens_with_masks(x);rope=self.backbone.rope_embed(H=h,W=w)
        for block in self.backbone.blocks[:self.start]:x=block(x,rope)
        teacher=x;adapted=x
        for block in self.teacher_tail:teacher=block(teacher,rope)
        for block in self.backbone.blocks[self.start:]:adapted=block(adapted,rope)
        teacher=self.norm_tokens(teacher);adapted=self.norm_tokens(adapted)
        n=1+self.backbone.n_storage_tokens
        return {
            'original_global':torch.cat([teacher[:,0],teacher[:,n:].mean(1)],-1),
            'adapted_global':torch.cat([adapted[:,0],adapted[:,n:].mean(1)],-1),
        }

    @torch.no_grad()
    def extract_frames(self,frames):
        assert frames.shape[-2:]==(720,1280)
        x=(frames.float()/255-self.rgb_mean)/self.rgb_std
        x,(h,w)=self.backbone.prepare_tokens_with_masks(x);rope=self.backbone.rope_embed(H=h,W=w)
        for block in self.backbone.blocks[:self.start]:x=block(x,rope)
        t=x;s=x
        for block in self.teacher_tail:t=block(t,rope)
        for block in self.backbone.blocks[self.start:]:s=block(s,rope)
        t=self.norm_tokens(t);s=self.norm_tokens(s);n=1+self.backbone.n_storage_tokens
        tp=t[:,n:];sp=s[:,n:];base=torch.cat([t[:,0],tp.mean(1)],-1)
        cl=self.control_head(tp)[:,:,0];al=self.adapt_head(sp)[:,:,0]
        result={'global_features':base}
        for name,p,l,u in [('ordinary',tp,cl,True),('frozen',tp,cl,False),('stage1',sp,al,False)]:
            a,b=candidate_features(p,l,u);result[name+'_tokens']=a;result[name+'_descriptors']=b
        return result

    @torch.no_grad()
    def extract_clip(self,frames,chunk=8):
        pieces=[self.extract_frames(frames[i:i+chunk]) for i in range(0,len(frames),chunk)]
        r={k:torch.cat([p[k] for p in pieces]) for k in pieces[0]}
        # Anchor is computed from exactly the features being cached; retained in FP32.
        gf=r['global_features'].float();outputs=self.event._global_branch_outputs(gf[None])
        r['anchor']=outputs['logits'][0].float()
        return r


class WindowFrames(Dataset):
    def __init__(self,records):self.records=records;self.caps=OrderedDict()
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        cv2.setNumThreads(0);r=self.records[index];path=r['video_path'];cap=self.caps.pop(path,None)
        if cap is None:cap=cv2.VideoCapture(path,cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
        self.caps[path]=cap
        if len(self.caps)>2:_,old=self.caps.popitem(last=False);old.release()
        frames=[]
        for f in r['frame_indices']:
            pos=int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
            if 0<=f-pos<=128:
                for _ in range(f-pos):
                    if not cap.grab():raise RuntimeError(f'advance failed {path} frame={f}')
            else:cap.set(cv2.CAP_PROP_POS_FRAMES,f)
            ok,x=cap.read()
            if not ok or int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))-1!=f:raise RuntimeError(f'exact decode failed {path} frame={f}')
            h,w=x.shape[:2]
            if h<720 or w<1280:raise RuntimeError(f'source below 720P {path}')
            # Match original event model preprocessing, including cubic resize.
            x=cv2.resize(cv2.cvtColor(x,cv2.COLOR_BGR2RGB),(1280,720),interpolation=cv2.INTER_CUBIC)
            frames.append(torch.from_numpy(x.copy()).permute(2,0,1))
        return torch.stack(frames),index


class CachedEvidence(Dataset):
    def __init__(self,records,cache_dir,arm):self.records=records;self.cache_dir=Path(cache_dir);self.arm=arm
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        r=self.records[index];z=torch.load(self.cache_dir/(r['key']+'.pt'),weights_only=True,map_location='cpu')
        return {'global_features':z['global_features'].float(),'tokens':z[self.arm+'_tokens'].float(),
                'descriptors':z[self.arm+'_descriptors'].float(),'anchor':z['anchor'].float(),
                'frame_times':torch.tensor(r['frame_times'],dtype=torch.float32),'targets':torch.tensor(r['labels']),
                'label_masks':torch.tensor(r['label_mask']),'index':index}


class Stage2EventModel(nn.Module):
    """Load a selected reader and run the same optional pathway on raw 720P clips."""
    def __init__(self,checkpoint):
        super().__init__();state=torch.load(checkpoint,weights_only=True,map_location='cpu')
        self.extractor=FrozenStage2Extractor(state['config']);self.reader=OptionalEvidence(hidden=state['config']['hidden'])
        self.reader.load_state_dict(state['model']);self.arm=state['arm'];self.enabled=state['best']['enabled'];self.register_buffer('decision_thresholds',torch.tensor(state['best']['thresholds'],dtype=torch.float32));self.eval()
    @torch.no_grad()
    def forward(self,frames,valid=None,frame_times=None):
        assert frames.dtype==torch.uint8 and frames.shape[1:]==(16,3,720,1280)
        with torch.autocast('cuda',dtype=torch.bfloat16):rows=[self.extractor.extract_clip(x) for x in frames]
        g=torch.stack([r['global_features'].float() for r in rows]);a=torch.stack([r['anchor'].float() for r in rows])
        t=torch.stack([r[self.arm+'_tokens'].half().float() for r in rows]);d=torch.stack([r[self.arm+'_descriptors'].float() for r in rows])
        if valid is None:valid=torch.ones(t.shape[:-1],device=t.device,dtype=torch.bool)
        with torch.autocast('cuda',enabled=False):return self.reader(g,t,d,valid,a,frame_times,enabled=self.enabled)
