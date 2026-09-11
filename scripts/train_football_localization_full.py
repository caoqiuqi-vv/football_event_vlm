#!/usr/bin/env python
"""Full-frame Stage1: exact coverage, separate control, atomic recovery and full validation."""
from __future__ import annotations
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from football_localization_full import FullFrames,CoverageSampler,global_localization_loss,atomic_json,digest
from scripts.train_football_localization_stage1 import LocalizationModel,spatial_targets


class FullModel(LocalizationModel):
    def __init__(self,cfg):
        super().__init__({**cfg,'checkpoint':cfg.get('checkpoint_snapshot',cfg['checkpoint'])});self.use_checkpoint=cfg['gradient_checkpointing']

    def forward(self,frames):
        self.backbone.eval();self.teacher_tail.eval()
        with torch.no_grad():
            x=(frames.float()/255-self.rgb_mean)/self.rgb_std
            x,(h,w)=self.backbone.prepare_tokens_with_masks(x);rope=self.backbone.rope_embed(H=h,W=w)
            for block in self.backbone.blocks[:self.start]:x=block(x,rope)
            t=x
            for block in self.teacher_tail:t=block(t,rope)
            t=self.norm_tokens(t)
        s=x.detach()
        for block in self.backbone.blocks[self.start:]:
            s=checkpoint(block,s,rope,use_reentrant=False) if self.use_checkpoint and self.training else block(s,rope)
        s=self.norm_tokens(s);n=1+self.backbone.n_storage_tokens;sp,tp=s[:,n:],t[:,n:]
        assert sp.shape[1]==3600
        def kl(a,b):
            def z(v):
                v=v.float();return (v-v.mean(-1,keepdim=True))/v.std(-1,keepdim=True,unbiased=False).clamp_min(1e-6)
            return F.kl_div(F.log_softmax(z(a)/self.temperature,-1),F.softmax(z(b.detach())/self.temperature,-1),reduction='none').sum(-1).mean(-1)*self.temperature**2
        stats=torch.stack([kl(sp,tp),kl(s[:,:1],t[:,:1]),F.cosine_similarity(sp.float(),tp.float(),dim=-1).mean(-1),F.cosine_similarity(s[:,0].float(),t[:,0].float(),dim=-1)],-1)
        return {'adapt_logits':self.adapt_head(sp),'control_logits':self.control_head(tp),'stats':stats}


def global_frame_mean(values,mask):
    count=mask.sum().detach().float();world=1
    if dist.is_initialized():world=dist.get_world_size();dist.all_reduce(count)
    return world*(values*mask).sum()/count.clamp_min(1)


def save_torch(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp.pt');torch.save(value,tmp);tmp.replace(path)


def loader_for(data,cfg,rank,world,epoch,train,start_step=0):
    sampler=CoverageSampler([d['frames'] for d in data.descriptors],cfg['batch_size'],rank,world,cfg['seed'],epoch,cfg['block_size'],start_step,shuffle=train)
    kw={'batch_size':cfg['batch_size'],'num_workers':cfg['workers'],'pin_memory':True}
    if cfg['workers']:kw.update(prefetch_factor=1,multiprocessing_context='spawn')
    return DataLoader(data,sampler=sampler,**kw),sampler


@torch.no_grad()
def evaluate(owner,data,cfg,out,tag,rank,world,device):
    owner.eval();loader,sampler=loader_for(data,cfg,rank,world,0,False);pieces=[];start=time.time()
    dtype=np.dtype([('index','i8'),('video','i4'),('valid','?',(2,)),('peaks','i4',(2,2)),('error','f4',(2,2)),('inside','?',(2,2)),('stats','f4',(4,))])
    for step,batch in enumerate(loader,1):
        frames=batch['frame'].to(device,non_blocking=True);boxes=batch['boxes'].to(device,non_blocking=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):r=owner(frames)
        real=batch['index'].numpy()>=0;row=np.zeros(int(real.sum()),dtype=dtype)
        row['index']=batch['index'].numpy()[real];row['video']=batch['video_index'].numpy()[real];row['valid']=batch['weights'].numpy()[real]>0;row['stats']=r['stats'].float().cpu().numpy()[real]
        for arm,name in enumerate(('adapt','control')):
            peaks=r[name+'_logits'].float().argmax(1)
            xy=torch.stack([(peaks%80+.5)*16,(peaks//80+.5)*16],-1)
            centers=(boxes[:,:,:,:2]+boxes[:,:,:,2:])/2*torch.tensor([1280,720],device=device)
            validboxes=(boxes[:,:,:,2]>boxes[:,:,:,0])&(boxes[:,:,:,3]>boxes[:,:,:,1])
            distances=(centers-xy[:,:,None]).square().sum(-1).sqrt().masked_fill(~validboxes,float('inf')).amin(-1)
            norm=xy/torch.tensor([1280,720],device=device)
            inside=((norm[:,:,None]>=boxes[:,:,:,:2])&(norm[:,:,None]<=boxes[:,:,:,2:])).all(-1)&validboxes
            row['peaks'][:,arm]=peaks.cpu().numpy()[real];row['error'][:,arm]=distances.cpu().numpy()[real];row['inside'][:,arm]=inside.any(-1).cpu().numpy()[real]
        pieces.append(row)
        if rank==0 and (step==1 or step%100==0):
            atomic_json(out/'heartbeat.json',{'phase':'validation','tag':tag,'step':step,'steps':len(loader),'elapsed_sec':time.time()-start,'updated_unix':time.time()})
            print(f'EVAL {tag} {step}/{len(loader)} elapsed={time.time()-start:.1f}',flush=True)
    array=np.concatenate(pieces) if pieces else np.empty(0,dtype=dtype)
    np.save(out/f'{tag}_rank{rank}.npy',array)
    if world>1:dist.barrier()
    summary=None
    if rank==0:
        all_rows=np.concatenate([np.load(out/f'{tag}_rank{r}.npy') for r in range(world)])
        order=np.argsort(all_rows['index']);all_rows=all_rows[order]
        assert np.array_equal(all_rows['index'],np.arange(len(data))),f'{tag}: validation coverage not exact'
        np.save(out/f'{tag}_predictions.npy',all_rows)
        summary={'frames':len(all_rows),'videos':len(data.descriptors),'drift':{},'arms':{},'scope':'complete qualified automatic-positive validation; not human detection accuracy'}
        for k,name in enumerate(('patch_kl','cls_kl','patch_cosine','cls_cosine')):
            v=all_rows['stats'][:,k];summary['drift'][name]={'mean':float(v.mean()),'p01':float(np.quantile(v,.01)),'p99':float(np.quantile(v,.99))}
        for arm,name in enumerate(('adapt','control')):
            summary['arms'][name]={}
            for c,cls in enumerate(('ball','goal')):
                valid=all_rows['valid'][:,c];errors=all_rows['error'][valid,arm,c]
                if not valid.any():
                    summary['arms'][name][cls]={'count':0}
                    continue
                summary['arms'][name][cls]={'count':int(valid.sum()),'within_16px':float((errors<=16).mean()),'within_32px':float((errors<=32).mean()),'peak_inside_box':float(all_rows['inside'][valid,arm,c].mean()),'mean_error_px':float(errors.mean()),'median_error_px':float(np.median(errors)),'p90_error_px':float(np.quantile(errors,.9))}
        atomic_json(out/f'{tag}_metrics.json',summary)
        for r in range(world):(out/f'{tag}_rank{r}.npy').unlink()
    payload=[summary]
    if world>1:dist.broadcast_object_list(payload,src=0)
    return payload[0]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--resume',action='store_true');ap.add_argument('--evaluate-selected',action='store_true');ap.add_argument('--smoke',action='store_true');ap.add_argument('--smoke-stop-at-step',type=int,default=0);args=ap.parse_args()
    cfg=json.loads(Path(args.config).read_text());out=Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0));torch.cuda.set_device(local);device=torch.device('cuda',local)
    if world>1:dist.init_process_group('nccl',device_id=device)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
    manifest=json.loads((out/'manifest.json').read_text());assert manifest['config']==cfg
    fingerprint=digest(out/'manifest.json')
    train_desc,val_desc=manifest['splits']['train'],manifest['splits']['val']
    if args.smoke:
        cfg={**cfg,'block_size':16,'checkpoint_interval':2}
        train_desc=[{**train_desc[0],'frames':world*cfg['batch_size']*2+3}]
        val_desc=[{**val_desc[0],'frames':world*cfg['batch_size']+3}]
        out=out/'smoke';out.mkdir(exist_ok=True)
    train=FullFrames(train_desc,cfg['max_goal_boxes']);val=FullFrames(val_desc,cfg['max_goal_boxes'])
    owner=FullModel(cfg).to(device)
    if args.evaluate_selected:
        adapted=torch.load(out/'best_adapt.pt',weights_only=True,map_location='cpu')
        control=torch.load(out/'best_control.pt',weights_only=True,map_location='cpu')
        assert adapted['manifest_sha256']==fingerprint and control['manifest_sha256']==fingerprint
        with torch.no_grad():
            params=dict(owner.backbone.named_parameters())
            for n,p in adapted['backbone_lora'].items():params[n].copy_(p)
        owner.adapt_head.load_state_dict(adapted['adapt_head']);owner.control_head.load_state_dict(control['control_head'])
        expert=json.loads((out/'expert_reference_manifest.json').read_text())
        refs=FullFrames(expert['descriptors'],cfg['max_goal_boxes'])
        evaluate(owner,refs,cfg,out,'expert_selected',rank,world,device)
        if world>1:dist.destroy_process_group()
        return
    warm=torch.load(cfg['warmup_checkpoint'],weights_only=True,map_location='cpu')
    assert warm['source_checkpoint']==cfg['checkpoint']
    owner.adapt_head.load_state_dict(warm['control_head']);owner.control_head.load_state_dict(warm['control_head'])
    model=owner
    synchronized_params=[p for p in owner.parameters() if p.requires_grad]
    lora=[p for p in owner.backbone.parameters() if p.requires_grad];adapt_params=lora+list(owner.adapt_head.parameters());control_params=list(owner.control_head.parameters())
    opt=torch.optim.AdamW([{'params':lora,'lr':cfg['lora_lr']},{'params':owner.adapt_head.parameters(),'lr':cfg['head_lr']},{'params':owner.control_head.parameters(),'lr':cfg['head_lr']}],weight_decay=cfg['weight_decay'])
    base_lrs=[cfg['lora_lr'],cfg['head_lr'],cfg['head_lr']];begin_epoch=1;begin_step=0;best={'adapt':-float('inf'),'control':-float('inf')};restored=False
    def state(epoch,step):
        return {'config':cfg,'manifest_sha256':fingerprint,'world_size':world,'epoch':epoch,'next_step':step,'best_scores':best,
                'source_checkpoint':cfg['checkpoint'],'source_epoch':owner.source_epoch,'backbone_lora':{n:p.detach().cpu() for n,p in owner.backbone.named_parameters() if p.requires_grad},
                'adapt_head':owner.adapt_head.state_dict(),'control_head':owner.control_head.state_dict(),'optimizer':opt.state_dict()}
    if args.resume and (out/'resume.pt').exists():
        saved=torch.load(out/'resume.pt',weights_only=True,map_location='cpu');assert saved['manifest_sha256']==fingerprint and saved['world_size']==world and saved['config']==cfg
        with torch.no_grad():
            params=dict(owner.backbone.named_parameters())
            for n,p in saved['backbone_lora'].items():params[n].copy_(p)
        owner.adapt_head.load_state_dict(saved['adapt_head']);owner.control_head.load_state_dict(saved['control_head']);opt.load_state_dict(saved['optimizer'])
        begin_epoch=saved['epoch'];begin_step=saved['next_step'];best=saved['best_scores'];restored=True
        print(f'RESUME rank={rank} epoch={begin_epoch} next_step={begin_step}',flush=True)
    def consider(summary,epoch,step):
        if rank==0:
            preserved=summary['drift']['patch_cosine']['mean']>=cfg['min_patch_cosine'] and summary['drift']['cls_cosine']['mean']>=cfg['min_cls_cosine']
            for arm in ('adapt','control'):
                a=summary['arms'][arm];score=(a['ball']['within_16px']+a['goal']['peak_inside_box'])/2
                if (arm=='control' or preserved) and score>best[arm]:
                    best[arm]=score;s=state(epoch,step);s['selection_metrics']=summary;s['selected_arm']=arm;save_torch(out/f'best_{arm}.pt',s)
        values=[best]
        if world>1:dist.broadcast_object_list(values,0)
        best.update(values[0])
    if not restored:
        initial=evaluate(owner,val,cfg,out,'val_epoch_000',rank,world,device);consider(initial,0,0)
        if rank==0:save_torch(out/'resume.pt',state(1,0))
    epochs=1 if args.smoke else cfg['epochs'];started=time.time()
    for epoch in range(begin_epoch,epochs+1):
        start_step=begin_step if epoch==begin_epoch else 0
        loader,sampler=loader_for(train,cfg,rank,world,epoch,True,start_step);seen=np.zeros(len(train),np.uint8)
        prefix=sampler.all_values[:start_step*cfg['batch_size']];seen[prefix[prefix>=0]]=1
        model.train();grad_seen=0.
        for local_step,batch in enumerate(loader):
            step=start_step+local_step
            expected=sampler.all_values[step*cfg['batch_size']:(step+1)*cfg['batch_size']]
            assert np.array_equal(batch['index'].numpy(),expected),'actual batches differ from deterministic coverage plan'
            ratio=((epoch-1)*sampler.total_steps+step)/max(epochs*sampler.total_steps,1);factor=.1+.9*.5*(1+math.cos(math.pi*ratio))
            for group,lr in zip(opt.param_groups,base_lrs):group['lr']=lr*factor
            frames=batch['frame'].to(device,non_blocking=True);boxes=batch['boxes'].to(device,non_blocking=True);weights=batch['weights'].to(device,non_blocking=True);real=batch['index'].to(device)>=0
            targets=spatial_targets(boxes,weights>0);opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                result=model(frames);loc=global_localization_loss(result['adapt_logits'],targets,weights);control=global_localization_loss(result['control_logits'],targets,weights)
                klp=global_frame_mean(result['stats'][:,0],real);klc=global_frame_mean(result['stats'][:,1],real)
                loss=loc+control+cfg['patch_kl_weight']*klp+cfg['cls_kl_weight']*klc
            if not torch.isfinite(loss):raise RuntimeError('nonfinite full training loss')
            loss.backward()
            if world>1:
                # Fixed packing order avoids DDP bucket rebuild differences on resume.
                flat=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in synchronized_params])
                dist.all_reduce(flat);flat.div_(world);offset=0
                for p in synchronized_params:
                    if p.grad is None:p.grad=torch.zeros_like(p)
                    p.grad.copy_(flat[offset:offset+p.numel()].view_as(p));offset+=p.numel()
            gn=nn.utils.clip_grad_norm_(adapt_params,1.);nn.utils.clip_grad_norm_(control_params,1.);opt.step();grad_seen=max(grad_seen,float(gn))
            indices=batch['index'].numpy();indices=indices[indices>=0]
            assert not seen[indices].any(),'repeated real training row within epoch';seen[indices]=1
            if rank==0 and (step==0 or (step+1)%cfg['log_interval']==0):
                h={'phase':'train','epoch':epoch,'epochs':epochs,'step':step+1,'steps':sampler.total_steps,'real_frames_seen_rank0':int(seen.sum()),'loss_rank0_scaled':float(loss.detach()),'patch_kl_rank0_scaled':float(klp.detach()),'cls_kl_rank0_scaled':float(klc.detach()),'elapsed_since_start_sec':time.time()-started,'updated_unix':time.time()}
                atomic_json(out/'heartbeat.json',h);print(json.dumps(h),flush=True)
            if (step+1)%cfg['checkpoint_interval']==0:
                if world>1:dist.barrier()
                if rank==0:save_torch(out/'resume.pt',state(epoch,step+1))
                if world>1:dist.barrier()
                if args.smoke and args.smoke_stop_at_step==step+1:
                    if world>1:dist.destroy_process_group()
                    raise SystemExit(85)
        if start_step<sampler.total_steps and grad_seen<=0:raise RuntimeError('no student gradient')
        counts=torch.from_numpy(seen.astype(np.int32)).to(device)
        if world>1:dist.all_reduce(counts)
        assert bool((counts==1).all()),'full coverage must be exactly once per real frame'
        if rank==0:
            np.save(out/f'coverage_epoch_{epoch:03d}.npy',counts.cpu().numpy())
            atomic_json(out/f'coverage_epoch_{epoch:03d}.json',{'eligible_frames':len(train),'visited_exactly_once':int((counts==1).sum()),'missing':int((counts==0).sum()),'duplicates':int((counts>1).sum()),'manifest_sha256':fingerprint})
            save_torch(out/'resume.pt',state(epoch,sampler.total_steps))
        if world>1:dist.barrier()
        summary=evaluate(owner,val,cfg,out,f'val_epoch_{epoch:03d}',rank,world,device);consider(summary,epoch,sampler.total_steps)
        if rank==0:
            save_torch(out/f'epoch_{epoch:03d}.pt',state(epoch,sampler.total_steps));save_torch(out/'resume.pt',state(epoch+1,0))
        if world>1:dist.barrier()
    if rank==0:atomic_json(out/'TRAINING_COMPLETE.json',{'epochs':epochs,'train_frames_per_epoch':len(train),'validation_frames_per_pass':len(val),'coverage_pass':True,'stage2_started':False,'final_report_ready':False})
    if world>1:dist.destroy_process_group()


if __name__=='__main__':main()
