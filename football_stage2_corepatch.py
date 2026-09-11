"""Stage2 core-preserving evidence and zero-initialized event temporal adapters."""
from __future__ import annotations
import json, math
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import roi_align
import train_football_events as football
from football_stage2_optional import FrozenStage2Extractor, candidate_features


def core_tokens(patches, logits, uniform=False):
    """Two peaks, each nine EXACT patch vectors + four spatial context vectors.

    Out-of-grid core cells are masked, not clamped duplicates. Index 4 in each
    13-token group is always its exact center. No learned preprocessing here.
    Descriptor: x,y,dx,dy,scale,kind,relative_salience,entropy,peakp,gap,candidate.
    """
    b,n,d=patches.shape
    assert n==3600 and logits.shape==(b,n)
    device=patches.device;prob=logits.float().softmax(1);work=logits.float().clone()
    yy,xx=torch.meshgrid(torch.arange(45,device=device),torch.arange(80,device=device),indexing='ij')
    yy,xx=yy.flatten(),xx.flatten();entropy=-(prob*prob.clamp_min(1e-12).log()).sum(1)/math.log(n)
    top=prob.topk(2,dim=1).values
    dy,dx=torch.meshgrid(torch.arange(-1,2,device=device),torch.arange(-1,2,device=device),indexing='ij')
    dx,dy=dx.flatten(),dy.flatten();features=[];descs=[];masks=[];indices=[]
    fmap=patches.transpose(1,2).reshape(b,d,45,80).float()
    for candidate in range(2):
        if uniform:
            cx=torch.full((b,),24 if candidate==0 else 56,device=device,dtype=torch.long)
            cy=torch.full((b,),22,device=device,dtype=torch.long)
        else:
            peak=work.argmax(1);cx,cy=peak%80,peak//80
            work.masked_fill_(((xx[None]-cx[:,None]).abs()<=3)&((yy[None]-cy[:,None]).abs()<=3),-torch.inf)
        gx,gy=cx[:,None]+dx,cy[:,None]+dy
        valid=(gx>=0)&(gx<80)&(gy>=0)&(gy<45)
        ix=(gy.clamp(0,44)*80+gx.clamp(0,79)).long()
        raw=patches.gather(1,ix[...,None].expand(-1,-1,d))
        raw=torch.where(valid[...,None],raw,torch.zeros_like(raw))
        center=torch.stack([cx+.5,cy+.5],-1).float()
        lo=(center-5.5).clamp_min(0);hi=torch.minimum(center+5.5,center.new_tensor([80.,45.]))
        rois=torch.cat([torch.arange(b,device=device)[:,None],lo,hi],-1)
        context=roi_align(fmap,rois,(2,2),spatial_scale=1.,sampling_ratio=0,aligned=True).flatten(2).transpose(1,2)
        qx,qy=torch.meshgrid(torch.tensor([.25,.75],device=device),torch.tensor([.25,.75],device=device),indexing='xy')
        xycontext=lo[:,None]+torch.stack([qx.flatten(),qy.flatten()],-1)[None]*(hi-lo)[:,None]
        xycore=torch.stack([gx+.5,gy+.5],-1).float();xy=torch.cat([xycore,xycontext],1)
        relative=(xy-center[:,None])/11
        pos=xy/xy.new_tensor([80.,45.]);scale=torch.cat([torch.full((b,9,1),1/11,device=device),torch.ones(b,4,1,device=device)],1)
        kind=torch.cat([torch.zeros(b,9,1,device=device),torch.ones(b,4,1,device=device)],1)
        salience=torch.cat([torch.log1p(3600*prob.gather(1,ix)),torch.log1p(3600*prob.gather(1,(cy*80+cx)[:,None])).expand(-1,4)],1)[...,None]
        stats=torch.stack([entropy,top[:,0],top[:,0]-top[:,1]],-1)[:,None].expand(-1,13,-1)
        if uniform:salience=torch.zeros_like(salience);stats=torch.zeros_like(stats)
        desc=torch.cat([pos,relative,scale,kind,salience,stats,torch.full((b,13,1),float(candidate),device=device)],-1)
        features.append(torch.cat([raw.float(),context],1));descs.append(desc)
        masks.append(torch.cat([valid,torch.ones(b,4,device=device,dtype=torch.bool)],1))
        indices.append(torch.cat([torch.where(valid,ix,-torch.ones_like(ix)),torch.full((b,4),-1,device=device,dtype=torch.long)],1))
    return {'tokens':torch.cat(features,1),'descriptors':torch.cat(descs,1),'valid':torch.cat(masks,1),'patch_indices':torch.cat(indices,1)}


def legacy_descriptors(d):
    """Map legacy 7D descriptors to the same documented 11D interface."""
    out=torch.zeros(*d.shape[:-1],11,device=d.device,dtype=d.dtype)
    out[...,:2]=d[...,:2];out[...,4]=d[...,2];out[...,5]=2.
    out[...,6]=torch.log1p(3600*d[...,4]);out[...,7:10]=d[...,3:6]
    out[...,10]=torch.arange(d.shape[-2],device=d.device).div(2,rounding_mode='floor')
    return out


class CoreExtractor(FrozenStage2Extractor):
    @torch.no_grad()
    def extract_frames(self,frames):
        assert frames.shape[-2:]==(720,1280)
        x=(frames.float()/255-self.rgb_mean)/self.rgb_std
        x,(h,w)=self.backbone.prepare_tokens_with_masks(x);rope=self.backbone.rope_embed(H=h,W=w)
        for block in self.backbone.blocks[:self.start]:x=block(x,rope)
        t,s=x,x
        for block in self.teacher_tail:t=block(t,rope)
        for block in self.backbone.blocks[self.start:]:s=block(s,rope)
        t,s=self.norm_tokens(t),self.norm_tokens(s);n=1+self.backbone.n_storage_tokens
        tp,sp=t[:,n:],s[:,n:];cl=self.control_head(tp)[:,:,0];al=self.adapt_head(sp)[:,:,0]
        result={}
        for name,p,l,u in [('ordinary',tp,cl,True),('frozen',tp,cl,False),('stage1',sp,al,False)]:
            r=core_tokens(p,l,u)
            for k,v in r.items():result[name+'_'+k]=v
        # A small checksum-equivalence target against the old cache establishes
        # identical backbone and localization extraction on the same RGB frames.
        result['global_features']=torch.cat([t[:,0],tp.mean(1)],-1)
        a,d=candidate_features(sp,al)
        result['legacy_tokens']=a;result['legacy_descriptors']=d
        return result
    @torch.no_grad()
    def extract_clip(self,frames,chunk=8):
        parts=[self.extract_frames(frames[i:i+chunk]) for i in range(0,len(frames),chunk)]
        return {k:torch.cat([p[k] for p in parts]) for k in parts[0]}


def load_event(cfg):
    c=football.to_config(json.loads(Path(cfg['event_config']).read_text()));football.configure_label_schema(c)
    event=football.make_model(c,use_cached_features=True,device=torch.device('cpu'))
    state=torch.load(cfg['source_checkpoint'],weights_only=True,map_location='cpu',mmap=True)
    event.load_state_dict({k:v for k,v in state['model'].items() if not k.startswith('backbone.')},strict=True)
    assert event.fusion=='cls_transformer'
    return event.requires_grad_(False).eval()


class EvidenceAdapter(nn.Module):
    def __init__(self,dim=1024,event_dim=512,hidden=128):
        super().__init__();self.hidden=hidden
        self.local=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,hidden))
        self.position=nn.Linear(11,hidden);self.query=nn.Sequential(nn.LayerNorm(event_dim),nn.Linear(event_dim,hidden))
        self.key=nn.Linear(hidden,hidden,bias=False);self.value=nn.Linear(hidden,hidden,bias=False)
        self.time=nn.Linear(3,hidden,bias=False)
        self.null_key=nn.Parameter(torch.zeros(hidden));self.null_bias=nn.Parameter(torch.tensor(1.5))
        self.output=nn.Linear(hidden,event_dim,bias=False);nn.init.zeros_(self.output.weight)
    def forward(self,frame_tokens,tokens,descriptors,valid,frame_times):
        valid=valid.bool()&torch.isfinite(tokens).all(-1)&torch.isfinite(descriptors).all(-1)
        safe=torch.where(valid[...,None],tokens,0.).float();desc=torch.where(valid[...,None],descriptors,0.).float()
        z=self.local(safe)+self.position(desc)
        times=(frame_times-frame_times[:,:1])/(frame_times[:,-1:]-frame_times[:,:1]).clamp_min(1e-6)
        temporal=self.time(torch.stack([times,torch.sin(math.pi*times),torch.cos(math.pi*times)],-1))
        q=self.query(frame_tokens)+temporal
        # Soft association to previous/current/next frame candidates, with actual
        # key timestamps; no hard trajectory or interpolation across absence.
        z=z+temporal[:,:,None]
        bank=[];bank_valid=[]
        for offset in [-1,0,1]:
            shifted=z.roll(offset,1);mask=valid.roll(offset,1).clone()
            if offset<0:mask[:,offset:]=False
            if offset>0:mask[:,:offset]=False
            bank.append(shifted);bank_valid.append(mask)
        z=torch.cat(bank,2);usable=torch.cat(bank_valid,2)
        score=(q[:,:,None]*self.key(z)).sum(-1)/math.sqrt(self.hidden)
        score=score.masked_fill(~usable,-torch.inf)
        null=(q*self.null_key).sum(-1,keepdim=True)/math.sqrt(self.hidden)+self.null_bias
        attention=torch.cat([score,null],-1).softmax(-1)
        pooled=(attention[...,:-1,None]*self.value(z)).sum(-2)
        delta=self.output(pooled)
        delta=torch.where(usable.any(-1,keepdim=True),delta,torch.zeros_like(delta))
        return delta,attention[...,-1],valid


class CoreEventReader(nn.Module):
    """Same adapter capacity; factorial varies only token source and fusion site.

    Temporal: add evidence to original projected frames before original 4-layer
    event transformer. Late: add averaged evidence before original classifier.
    In both cases only the adapter trains. Paired identical source computations
    make zero-initialization and explicit null return cached anchor exactly.
    """
    def __init__(self,cfg,fusion,event=None):
        super().__init__();assert fusion in ['temporal','late'];self.fusion=fusion
        self.event=load_event(cfg) if event is None else event.requires_grad_(False).eval()
        self.adapter=EvidenceAdapter(hidden=cfg['hidden'])
    def train(self,mode=True):
        super().train(mode);self.event.eval();return self
    def forward(self,global_features,tokens,descriptors,valid,anchor,frame_times,enabled=True):
        self.event.eval()
        if not enabled:valid=torch.zeros_like(valid)
        with torch.no_grad():frames=self.event.frame_proj(global_features.float())
        evidence,null,valid=self.adapter(frames,tokens,descriptors,valid,frame_times)
        b=len(frames)
        if self.fusion=='temporal':
            states=self.event.temporal(torch.cat([frames+evidence,frames],0))
            scores=self.event.head(states);residual=scores[:b]-scores[b:]
        else:
            with torch.no_grad():states=self.event.temporal(frames)
            scores=self.event.head(torch.cat([states+evidence.mean(1),states],0));residual=scores[:b]-scores[b:]
        residual=torch.where(valid.flatten(1).any(1,keepdim=True),residual,torch.zeros_like(residual))
        return {'logits':anchor.float()+residual,'delta':residual,'null_mass':null,'valid':valid}


class CorePatchEventModel(nn.Module):
    """Inference entry: raw uint8 [B,16,3,720,1280] and optional token mask."""
    def __init__(self,checkpoint):
        super().__init__();s=torch.load(checkpoint,weights_only=True,map_location='cpu')
        self.cfg=s['config'];self.arm=self.cfg['arms'][s['arm']];self.enabled=s['best']['enabled']
        self.extractor=CoreExtractor(self.cfg);self.reader=CoreEventReader(self.cfg,self.arm['fusion'])
        self.reader.adapter.load_state_dict(s['adapter'])
        self.register_buffer('normal_thresholds',torch.tensor(s['best']['thresholds']))
        self.register_buffer('baseline_thresholds',torch.tensor(s['baseline_thresholds']));self.eval()
    @torch.no_grad()
    def forward(self,frames,valid=None,frame_times=None):
        assert frames.dtype==torch.uint8 and frames.shape[1:]==(16,3,720,1280)
        rows=[];anchors=[]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            for clip in frames:
                r=self.extractor.extract_clip(clip,8);rows.append(r)
                anchors.append(self.extractor.event._global_branch_outputs(r['global_features'].float()[None])['logits'][0].float())
        source=self.arm['source'];roi=self.arm['representation']=='roi'
        tokens=torch.stack([r['legacy_tokens' if roi else source+'_tokens'].half().float() for r in rows])
        desc=torch.stack([r['legacy_descriptors' if roi else source+'_descriptors'].float() for r in rows])
        if roi:desc=legacy_descriptors(desc)
        intrinsic=torch.ones(tokens.shape[:-1],device=tokens.device,dtype=torch.bool) if roi else torch.stack([r[source+'_valid'] for r in rows])
        if valid is not None:intrinsic &= valid.bool()
        if frame_times is None:frame_times=torch.linspace(0,10,16,device=tokens.device)[None].expand(len(rows),-1)
        with torch.autocast('cuda',enabled=False):
            result=self.reader(torch.stack([r['global_features'].float() for r in rows]),tokens,desc,intrinsic,torch.stack(anchors),frame_times,enabled=self.enabled)
        available=result['valid'].flatten(1).any(1,keepdim=True)
        thresholds=torch.where(available,self.normal_thresholds[None],self.baseline_thresholds[None])
        result['probabilities']=result['logits'].sigmoid();result['thresholds']=thresholds;result['decisions']=result['probabilities']>=thresholds
        return result
