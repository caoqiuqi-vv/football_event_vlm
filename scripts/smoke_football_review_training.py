#!/usr/bin/env python3
"""Real decode and optimizer checks for frozen review data (no saved model)."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
import train_football_events as t
from football_review_data import window_supervision, policy_for_record


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    cfg=t.load_config(a.config,[]);t.configure_runtime_threads(cfg);t.configure_label_schema(cfg)
    t.seed_everything(cfg.seed, cfg.deterministic);device=torch.device(cfg.device)
    t.resolve_runtime_topology(cfg,device)
    dataset,val_dataset,records,val_records=t.prepare_datasets(cfg,use_cache=False)
    summary={'train_rows':len(records),'train_videos':len({r.video_id for r in records}),
             'val_rows':len(val_records),'val_videos':len({r.video_id for r in val_records}),
             'mask_support':{},'val_temporal_support':{lab:len({(r.video_id,v) for r in val_records for v in r.online_gt_anchors[i]}) for i,lab in enumerate(t.LABELS)}}
    selectors={}
    for i,r in enumerate(records):
        targets,masks,frame=window_supervision(policy_for_record(r),r.base_clip_start,r.base_clip_end)
        for kind,match in [('strong',any(v>0 and f>0 for v,f in zip(targets,frame))),
                           ('weak',any(v>0 and f==0 for v,f in zip(targets,frame))),
                           ('negative',not any(targets) and any(masks))]:
            if match:selectors.setdefault(kind,i)
        for index,lab in enumerate(t.LABELS):
            for key,enabled in [('positive',targets[index]>0 and masks[index]>0),('negative',targets[index]==0 and masks[index]>0),('frame_class',frame[index]>0)]:
                stat=lab+'_'+key;summary['mask_support'][stat]=summary['mask_support'].get(stat,0)+int(enabled)
    assert set(selectors)=={'strong','weak','negative'},selectors
    print(json.dumps(summary),flush=True)
    # Decode all three supervision types before allocating the large model.
    batches={kind:t.football_collate([dataset[index] for _ in range(int(cfg.train.per_gpu_batch_size))]) for kind,index in selectors.items()}
    for kind,b in batches.items():
        assert tuple(b['inputs'].shape[2:])==(3,720,1280),b['inputs'].shape
        assert b['sample_loss_weights'].sum()>0, b['meta']
        print('DECODE_OK',kind,b['meta'][0]['sampled_clip_start'],b['meta'][0]['sampled_clip_end'],b['label_masks'].tolist(),b['frame_label_masks'].tolist(),flush=True)
    model=t.make_model(cfg,use_cached_features=False,device=device);model.train()
    optimizer=t.build_optimizer(model,cfg)
    updates=[]
    for stage,kind,weight in [('E1','strong',0.0),('E2','strong',0.15),('E2','weak',0.15),('E2','negative',0.15)]:
        b=batches[kind];optimizer.zero_grad(set_to_none=True);start=time.monotonic()
        with t.autocast_context(device,True,'bf16'):
            outputs=t.forward_model_batch(model,b,device,return_aux=(weight>0))
            logits=outputs['logits'] if isinstance(outputs,dict) else outputs
            clip=t.masked_mean(F.binary_cross_entropy_with_logits(logits,b['targets'].to(device),reduction='none',pos_weight=logits.new_tensor(cfg.train.pos_weight)),b['label_masks'].to(device))
            frame,components=t.frame_detection_loss(outputs,b,cfg,device) if weight else (clip.new_zeros(()),{})
            loss=clip+weight*frame
        assert torch.isfinite(loss)
        loss.backward()
        gradients={'head':0.0,'lora':0.0,'frame':0.0}
        for name,param in model.named_parameters():
            if param.grad is None:continue
            value=float(param.grad.float().norm())
            if 'lora_' in name:gradients['lora']+=value
            elif name.startswith('frame_event_head'):gradients['frame']+=value
            elif name.startswith('head'):gradients['head']+=value
        assert gradients['head']>0 and gradients['lora']>0,gradients
        if stage=='E1':assert gradients['frame']==0,gradients
        if stage=='E2' and kind in ('strong','negative'):assert gradients['frame']>0,gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.train.grad_clip_norm);optimizer.step();torch.cuda.synchronize()
        result={'stage':stage,'kind':kind,'loss':float(loss.detach()),'frame_loss':float(frame.detach()),'gradients':gradients,'seconds':time.monotonic()-start,'peak_memory_gib':torch.cuda.max_memory_allocated()/2**30}
        updates.append(result);print('OPTIMIZER_OK',json.dumps(result),flush=True)
    summary['updates']=updates;Path(a.output).write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
