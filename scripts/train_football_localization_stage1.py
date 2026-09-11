#!/usr/bin/env python
"""720P ball/goal localization curriculum with frozen-checkpoint feature KL.

This standalone experiment never updates event heads. A matched frozen-feature
probe is trained on exactly the same batches. Validation measures agreement with
held-out automatic labels, not human-confirmed detection accuracy.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_football_events as football


def feature_kl(student, teacher, temperature=2.0):
    """Mean token KL(teacher || student) over channel distributions, FP32.

    Per-token standardization makes the distribution temperature interpretable;
    this controls relative feature structure, not feature norm or all semantics.
    """
    def standardize(x):
        x = x.float()
        return (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True, unbiased=False).clamp_min(1e-6)
    s = F.log_softmax(standardize(student) / temperature, dim=-1)
    t = F.softmax(standardize(teacher.detach()) / temperature, dim=-1)
    return F.kl_div(s, t, reduction='none').sum(-1).mean() * temperature**2


def spatial_targets(boxes, valid):
    """Ball Gaussian / goal box density on native 45x80 patch lattice."""
    yy, xx = torch.meshgrid(torch.arange(45, device=boxes.device) + .5,
                            torch.arange(80, device=boxes.device) + .5, indexing='ij')
    x, y = xx.reshape(1, -1), yy.reshape(1, -1)
    targets = []
    for c in range(2):
        b = boxes[:, c]
        if c == 0:
            cx = ((b[:, 0, 0] + b[:, 0, 2]) * 40)[:, None]
            cy = ((b[:, 0, 1] + b[:, 0, 3]) * 22.5)[:, None]
            heat = torch.exp(-((x-cx)**2 + (y-cy)**2) / (2*.75**2))
        else:
            heat = torch.zeros((len(boxes), 3600), device=boxes.device)
            for k in range(b.shape[1]):
                box = b[:, k]
                active = (box[:, 2] > box[:, 0]) & (box[:, 3] > box[:, 1])
                # Soft boundary keeps a target even for narrow distant goals.
                dx = torch.maximum(torch.maximum(box[:, 0, None]*80-x, x-box[:, 2, None]*80), torch.zeros_like(x))
                dy = torch.maximum(torch.maximum(box[:, 1, None]*45-y, y-box[:, 3, None]*45), torch.zeros_like(y))
                h = torch.exp(-(dx**2+dy**2)/(2*.5**2)) * active[:, None]
                heat = torch.maximum(heat, h)
        heat = heat / heat.sum(-1, keepdim=True).clamp_min(1e-12)
        targets.append(heat * valid[:, c, None])
    return torch.stack(targets, dim=-1)


def localization_loss(logits, targets, weights):
    losses = -(targets * logits.float().log_softmax(1)).sum(1)
    # Confidence is a loss weight, never the height of a positive target.
    return (losses*weights).sum() / weights.sum().clamp_min(1e-6)


def read_ids(path):
    return [s.strip() for s in Path(path).read_text().splitlines() if s.strip() and not s.startswith('#')]


def sample_spaced(rows, count):
    spaced = []
    for row in sorted(rows, key=lambda r: r['time']):
        if not spaced or row['time'] - spaced[-1]['time'] >= 2:
            spaced.append(row)
    if len(spaced) > count:
        spaced = [spaced[i] for i in np.linspace(0, len(spaced)-1, count).round().astype(int)]
    return spaced


def build_manifest(cfg, out):
    sets = {s: read_ids(cfg[f'{s}_ids']) for s in ('train', 'val')}
    test = set(read_ids(cfg['test_ids']))
    assert not set(sets['train']) & set(sets['val'])
    assert not (set(sets['train']) | set(sets['val'])) & test
    manifest = {'config': cfg, 'label_scope': 'positive automatic detections only; missing objects unknown',
                'excluded': [], 'records': {}}
    fingerprints = {}
    # Catch byte-identical automatic-track aliases across train/validation.
    for split in ('val', 'train'):
        records = []
        for vid in sets[split]:
            video = Path(cfg['video_root']) / f'{vid}.mp4'
            ball_path = Path(cfg['ball_index']) / f'{vid}.npz'
            if vid in cfg.get('excluded_videos', []) or not video.is_file():
                manifest['excluded'].append([split, vid, 'excluded_or_missing_video']); continue
            if ball_path.is_file():
                digest = hashlib.sha256(ball_path.read_bytes()).hexdigest()
                if digest in fingerprints:
                    manifest['excluded'].append([split, vid, 'duplicate_index', fingerprints[digest]]); continue
                fingerprints[digest] = [split, vid]
            count = cfg[f'{split}_frames_per_object_video']
            ball_rows, goal_rows = [], []
            if ball_path.is_file():
                with np.load(ball_path) as archive:
                    d = {k: archive[k] for k in archive.files}
                    good = (d['source_code'] == 1) & ((d['flags'] & 1) > 0) & (d['confidence'] >= cfg['ball_confidence'])
                    chosen = []
                    for i in np.flatnonzero(good):
                        if not chosen or float(d['timestamp_sec'][i])-float(d['timestamp_sec'][chosen[-1]]) >= 2:
                            chosen.append(int(i))
                    if len(chosen)>count:
                        chosen=[chosen[i] for i in np.linspace(0,len(chosen)-1,count).round().astype(int)]
                    for i in chosen:
                        box = d['bbox_xyxy_norm'][i].astype(float)
                        if not np.isfinite(box).all() or (box < 0).any() or (box > 1).any() or box[2] <= box[0] or box[3] <= box[1]:
                            continue
                        ball_rows.append({'time': float(d['timestamp_sec'][i]), 'boxes': [box.tolist()], 'object': 0, 'weight': float(d['confidence'][i]), 'source': 'observed_ball_native'})
            goal_path = Path(cfg['goal_index']) / f'{vid}.pt'
            if goal_path.is_file():
                d = torch.load(goal_path, weights_only=True, map_location='cpu')
                w, h = d['image_size']['width'], d['image_size']['height']
                scale = torch.tensor([w, h, w, h])
                object_ok = ((d['classes'] == 2) & (d['confidences'] >= cfg['goal_confidence'])).numpy()
                object_indices = np.flatnonzero(object_ok)
                frame_indices = np.unique(np.searchsorted(d['frame_offsets'].numpy()[1:], object_indices, side='right'))
                selected_frames=[]
                for i in frame_indices:
                    if not selected_frames or float(d['frame_ids'][i]-d['frame_ids'][selected_frames[-1]])/float(d['fps']) >= 2:
                        selected_frames.append(int(i))
                if len(selected_frames)>count:
                    selected_frames=[selected_frames[j] for j in np.linspace(0,len(selected_frames)-1,count).round().astype(int)]
                for i in selected_frames:
                    frame = d['frame_ids'][i]
                    lo, hi = int(d['frame_offsets'][i]), int(d['frame_offsets'][i+1])
                    good = (d['classes'][lo:hi] == 2) & (d['confidences'][lo:hi] >= cfg['goal_confidence'])
                    if not good.any(): continue
                    boxes = (d['boxes'][lo:hi][good] / scale).clamp(0, 1)
                    boxes = boxes[(boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])][:4]
                    if not len(boxes): continue
                    goal_rows.append({'time': float(frame)/float(d['fps']), 'boxes': boxes.tolist(), 'object': 1,
                                      'weight': float(d['confidences'][lo:hi][good].mean()), 'source': 'goal_native_sparse'})
            selected = sample_spaced(ball_rows, count) + sample_spaced(goal_rows, count)
            for row in selected:
                row.update(video_id=vid, video_path=str(video))
            records.extend(selected)
            if len(records) and len(records) % 320 == 0:
                print(f'PREP {split} frames={len(records)} latest={vid}',flush=True)
        manifest['records'][split] = records
    for split, records in manifest['records'].items():
        assert records and all(any(r['object'] == c for r in records) for c in (0, 1)), split
    manifest['counts'] = {s: {'frames':len(r), 'videos':len({x['video_id'] for x in r}),
                         'ball':sum(x['object']==0 for x in r), 'goal':sum(x['object']==1 for x in r)} for s,r in manifest['records'].items()}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


class Frames(Dataset):
    def __init__(self, records):
        self.records = records
        self.cache = None

    def __len__(self): return len(self.records)

    def __getitem__(self, i):
        cv2.setNumThreads(0)
        r = self.records[i]
        if self.cache is None: self.cache = football.VideoCaptureCache(2)
        cap = self.cache.get(r['video_path'])
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        assert fps > 0
        frame_idx = round(r['time']*fps)
        # Decode exactly the nearest source frame; never substitute a random frame.
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok: raise RuntimeError(f"decode failed {r['video_id']} frame={frame_idx}")
        h,w = frame.shape[:2]
        if h < 720 or w < 1280 or abs(w/h - 1280/720) > .02:
            raise RuntimeError(f"source below 720P or wrong aspect ratio: {frame.shape} in {r['video_path']}")
        if (h,w) != (720,1280):
            frame = cv2.resize(frame,(1280,720),interpolation=cv2.INTER_AREA)
        assert frame.shape[:2] == (720,1280)
        frame = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy()).permute(2,0,1)
        boxes = torch.zeros(2, 4, 4)
        boxes[r['object'], :len(r['boxes'])] = torch.tensor(r['boxes'])
        weights = torch.zeros(2); weights[r['object']] = r['weight']
        return {'frame':frame, 'boxes':boxes, 'weights':weights, 'index':i}


class LocalizationModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        ckpt = torch.load(cfg['checkpoint'], weights_only=True, map_location='cpu', mmap=True)
        model_cfg = copy.deepcopy(ckpt['config']['model'])
        model_cfg.update(pretrained=False, weights='')
        self.backbone = football.build_backbone(football.to_config({'model': model_cfg}))
        football.inject_lora(self.backbone, model_cfg['lora'])
        state = {k[len('backbone.'):]: v for k,v in ckpt['model'].items() if k.startswith('backbone.')}
        self.backbone.load_state_dict(state, strict=True)
        self.backbone.requires_grad_(False)
        self.start = len(self.backbone.blocks)-int(model_cfg['lora']['target_last_blocks'])
        self.teacher_tail = copy.deepcopy(self.backbone.blocks[self.start:]).requires_grad_(False)
        for name, p in self.backbone.named_parameters():
            if name.endswith(('lora_a', 'lora_b')): p.requires_grad_(True)
        dim = int(self.backbone.embed_dim)
        self.adapt_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 128), nn.GELU(), nn.Linear(128, 2))
        self.control_head = copy.deepcopy(self.adapt_head)
        self.register_buffer('rgb_mean', torch.tensor([.485,.456,.406]).view(1,3,1,1))
        self.register_buffer('rgb_std', torch.tensor([.229,.224,.225]).view(1,3,1,1))
        self.source_epoch = int(ckpt['epoch'])
        self.temperature = cfg['temperature']

    def norm_tokens(self, x):
        b = self.backbone
        if b.untie_cls_and_patch_norms:
            return torch.cat([b.cls_norm(x[:,:1+b.n_storage_tokens]), b.norm(x[:,1+b.n_storage_tokens:])],1)
        return b.norm(x)

    def forward(self, frames, adapt=True):
        # Eval mode disables inherited LoRA dropout: teacher/student discrepancy
        # then measures parameter changes, not random augmentation noise.
        self.backbone.eval(); self.teacher_tail.eval()
        x = (frames.float()/255-self.rgb_mean)/self.rgb_std
        with torch.no_grad():
            x, (h,w) = self.backbone.prepare_tokens_with_masks(x)
            rope = self.backbone.rope_embed(H=h,W=w)
            for block in self.backbone.blocks[:self.start]: x = block(x, rope)
            t = x
            for block in self.teacher_tail: t = block(t, rope)
            t = self.norm_tokens(t)
        if adapt:
            s = x.detach()
            for block in self.backbone.blocks[self.start:]:
                if self.training:
                    s = checkpoint(block, s, rope, use_reentrant=False)
                else: s = block(s, rope)
            s = self.norm_tokens(s)
        else: s = t
        n = 1+self.backbone.n_storage_tokens
        sp, tp = s[:,n:], t[:,n:]
        assert sp.shape[1] == 3600, sp.shape
        control_logits = self.control_head(tp)
        adapt_logits = self.adapt_head(sp) if adapt else control_logits
        return {'adapt_logits':adapt_logits, 'control_logits':control_logits,
                'patch_kl':feature_kl(sp,tp,self.temperature), 'cls_kl':feature_kl(s[:,:1],t[:,:1],self.temperature),
                'patch_cosine':F.cosine_similarity(sp.float(),tp.float(),dim=-1).mean(),
                'cls_cosine':F.cosine_similarity(s[:,0].float(),t[:,0].float(),dim=-1).mean()}


@torch.no_grad()
def evaluate(model, loader, device, adapt):
    model.eval(); rows=[]
    for batch in loader:
        frames, boxes = batch['frame'].to(device), batch['boxes'].to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16): result = model(frames,adapt=adapt)
        for j, idx in enumerate(batch['index'].tolist()):
            c = int(batch['weights'][j].argmax())
            row = {'index':idx, 'class':c}
            for key in ('patch_kl','cls_kl','patch_cosine','cls_cosine'): row[key]=float(result[key])
            for arm in ('adapt','control'):
                scores=result[f'{arm}_logits'][j,:,c].float()
                peak=int(scores.argmax()); xy=torch.tensor([(peak%80+.5)/80,(peak//80+.5)/45],device=device)
                b=boxes[j,c]; valid=(b[:,2]>b[:,0]) & (b[:,3]>b[:,1]); b=b[valid]
                centers=(b[:,:2]+b[:,2:])/2
                distance=float((((centers-xy)*torch.tensor([1280,720],device=device))**2).sum(-1).sqrt().min())
                inside=bool(((xy>=b[:,:2]) & (xy<=b[:,2:])).all(-1).any())
                row[arm]={'center_error_px':distance,'within_16px':distance<=16,'within_32px':distance<=32,'peak_inside_box':inside}
            rows.append(row)
    if dist.is_initialized():
        gathered=[None]*dist.get_world_size(); dist.all_gather_object(gathered,rows)
        # DistributedSampler pads validation; count each real sample once.
        rows=list({r['index']:r for part in gathered for r in part}.values())
    summary={k:float(np.mean([r[k] for r in rows])) for k in ('patch_kl','cls_kl','patch_cosine','cls_cosine')}
    for arm in ('adapt','control'):
        summary[arm]={}
        for c,name in enumerate(('ball','goal')):
            part=[r[arm] for r in rows if r['class']==c]
            summary[arm][name]={'count':len(part), **{k:float(np.mean([r[k] for r in part])) for k in part[0]}}
    return summary, rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True); parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--smoke',action='store_true'); args=parser.parse_args()
    torch.set_num_threads(1)
    cfg=json.loads(Path(args.config).read_text()); out=Path(cfg['output_dir']); out.mkdir(parents=True,exist_ok=True)
    rank=int(os.environ.get('RANK','0')); world=int(os.environ.get('WORLD_SIZE','1')); local=int(os.environ.get('LOCAL_RANK','0'))
    if args.prepare_only:
        manifest=build_manifest(cfg,out); print(json.dumps(manifest['counts'])); return
    torch.cuda.set_device(local); device=torch.device('cuda',local)
    if world>1: dist.init_process_group('nccl',device_id=device)
    torch.set_num_threads(1); torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed']); random.seed(cfg['seed'])
    if rank==0:
        if not (out/'manifest.json').exists(): build_manifest(cfg,out)
    if world>1: dist.barrier()
    manifest=json.loads((out/'manifest.json').read_text())
    if manifest['config'] != cfg: raise RuntimeError('manifest/config mismatch; create a new output directory')
    train_records, val_records=manifest['records']['train'],manifest['records']['val']
    if args.smoke:
        train_records=[next(r for r in train_records if r['object']==c) for c in (0,1)]*2
        val_records=[next(r for r in val_records if r['object']==c) for c in (0,1)]
    trainset,valset=Frames(train_records),Frames(val_records)
    samplers=[DistributedSampler(d,num_replicas=world,rank=rank,shuffle=s,seed=cfg['seed']) for d,s in ((trainset,True),(valset,False))]
    kwargs={'batch_size':cfg['batch_size'],'num_workers':0 if args.smoke else cfg['workers'],'pin_memory':True}
    if kwargs['num_workers']: kwargs.update(persistent_workers=True,prefetch_factor=1,multiprocessing_context='spawn')
    loaders=[DataLoader(d,sampler=s,**kwargs) for d,s in zip((trainset,valset),samplers)]
    owner=LocalizationModel(cfg).to(device)
    model=DDP(owner,device_ids=[local],find_unused_parameters=True) if world>1 else owner
    lora=[p for n,p in owner.backbone.named_parameters() if p.requires_grad]
    opt=torch.optim.AdamW([{'params':lora,'lr':cfg['lora_lr']}, {'params':list(owner.adapt_head.parameters())+list(owner.control_head.parameters()),'lr':cfg['head_lr']}],weight_decay=cfg['weight_decay'])
    if rank==0:
        (out/'trainable_parameters.json').write_text(json.dumps({n:p.numel() for n,p in owner.named_parameters() if p.requires_grad},indent=2))
        print(f"loaded source epoch={owner.source_epoch}; image=720x1280 patches=3600; counts={manifest['counts']}; world={world}",flush=True)
    best=-float('inf'); start=time.time()
    for epoch in range(1,(2 if args.smoke else cfg['epochs'])+1):
        adapt=epoch>cfg['warmup_epochs']; model.train(); samplers[0].set_epoch(epoch)
        totals={}; steps=0; grad_seen=0.0
        for step,batch in enumerate(loaders[0],1):
            frames=batch['frame'].to(device); boxes=batch['boxes'].to(device); weights=batch['weights'].to(device)
            targets=spatial_targets(boxes,weights>0)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                result=model(frames,adapt=adapt)
                loc=localization_loss(result['adapt_logits'],targets,weights)
                control=localization_loss(result['control_logits'],targets,weights)
                loss=(loc+control if adapt else control)+cfg['patch_kl_weight']*result['patch_kl']+cfg['cls_kl_weight']*result['cls_kl']
            if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
            loss.backward()
            if adapt:
                g=sum(float(p.grad.float().norm()) for p in lora if p.grad is not None)
                grad_seen=max(grad_seen,g)
            nn.utils.clip_grad_norm_([p for p in owner.parameters() if p.requires_grad],1.0); opt.step()
            metrics={'loss':float(loss.detach()),'loc':float(loc.detach()),'control_loc':float(control.detach()),
                     **{k:float(result[k].detach()) for k in ('patch_kl','cls_kl','patch_cosine','cls_cosine')}}
            for k,v in metrics.items(): totals[k]=totals.get(k,0)+v
            steps+=1
            if rank==0 and (step==1 or step%cfg['log_interval']==0):
                print(json.dumps({'epoch':epoch,'adapt':adapt,'step':step,'steps':len(loaders[0]),'elapsed_sec':round(time.time()-start,1),'lora_grad_norm_sum_max':grad_seen,**metrics}),flush=True)
        if adapt and grad_seen<=0: raise RuntimeError('LoRA received no gradient')
        # Warmup arms start identical and must stay identical on frozen features.
        if not adapt:
            # One shared warmup probe, then exact copies of weights and Adam state.
            owner.adapt_head.load_state_dict(owner.control_head.state_dict())
            for a,b in zip(owner.adapt_head.parameters(),owner.control_head.parameters()):
                if b in opt.state: opt.state[a]=copy.deepcopy(opt.state[b])
        summary,rows=evaluate(model,loaders[1],device,adapt)
        score=(summary['adapt']['ball']['within_16px']+summary['adapt']['goal']['peak_inside_box'])/2
        preserved=summary['patch_cosine']>=cfg['min_patch_cosine'] and summary['cls_cosine']>=cfg['min_cls_cosine']
        payload={'epoch':epoch,'adapt':adapt,'train':{k:v/steps for k,v in totals.items()},'val':summary,'selection_score':score,'feature_preservation_pass':preserved,'label_scope':manifest['label_scope']}
        if rank==0:
            suffix='smoke_' if args.smoke else ''
            (out/f'{suffix}metrics_epoch_{epoch:03d}.json').write_text(json.dumps(payload,indent=2))
            (out/f'{suffix}predictions_epoch_{epoch:03d}.json').write_text(json.dumps(rows))
            state={'config':cfg,'epoch':epoch,'source_checkpoint':cfg['checkpoint'],'source_epoch':owner.source_epoch,
                   'backbone_lora':{n:p.detach().cpu() for n,p in owner.backbone.named_parameters() if p.requires_grad},
                   'adapt_head':owner.adapt_head.state_dict(),'control_head':owner.control_head.state_dict(),'metrics':payload,'optimizer':opt.state_dict()}
            torch.save(state,out/f'{suffix}last.pt')
            if preserved and score>best:
                best=score; torch.save(state,out/f'{suffix}best.pt')
            if epoch==cfg['warmup_epochs']: torch.save(state,out/f'{suffix}warmup.pt')
            print('VALIDATION '+json.dumps(payload),flush=True)
        if world>1: dist.barrier()
    if rank==0: (out/('smoke_COMPLETE.json' if args.smoke else 'COMPLETE.json')).write_text(json.dumps({'epochs':epoch,'elapsed_sec':time.time()-start,'stage2_started':False}))
    if world>1: dist.destroy_process_group()


if __name__=='__main__': main()
