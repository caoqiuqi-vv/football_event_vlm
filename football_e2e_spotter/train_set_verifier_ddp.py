#!/usr/bin/env python3
"""Four-GPU DDP trainer for the no-NMS DINO verifier (one sample/GPU)."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,DistributedSampler
from dinov3.hub.backbones import dinov3_vitl16
from football_e2e_spotter.set_spotting import NO_EVENT_INDEX
from football_e2e_spotter.train_set_verifier import VerifierCandidateDataset,assign_oof_targets,read_rows
from football_e2e_spotter.verifier import DinoVerifier
from football_e2e_spotter.verifier_lora_runtime import configure
def main():
 p=argparse.ArgumentParser();p.add_argument('--candidates',required=True);p.add_argument('--annotations',required=True);p.add_argument('--weights',default='checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth');p.add_argument('--output',required=True);p.add_argument('--epochs',type=int,default=6);p.add_argument('--warmup-epochs',type=int,default=2);a=p.parse_args()
 dist.init_process_group('nccl');rank=dist.get_rank();local=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(local);device=torch.device('cuda',local)
 rows=assign_oof_targets(read_rows(a.candidates),a.annotations);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 if rank==0:(out/'oof_target_summary.json').write_text(json.dumps({'rows':len(rows),'matched':sum(r['matched'] for r in rows),'no_temporal_nms':True},indent=2))
 dataset=VerifierCandidateDataset(rows,512,896);sampler=DistributedSampler(dataset,shuffle=True,drop_last=False);loader=DataLoader(dataset,batch_size=1,sampler=sampler,num_workers=0,pin_memory=True)
 backbone=dinov3_vitl16(pretrained=True,weights=a.weights,check_hash=False);model=DinoVerifier(backbone,audio_dim=64).to(device);configure(backbone,warmup=False,lora_rank=16);enabled={n for n,v in backbone.named_parameters() if v.requires_grad}
 for q in backbone.parameters():q.requires_grad_(False)
 model=DDP(model,device_ids=[local],output_device=local,broadcast_buffers=False);optim=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.02)
 for epoch in range(a.epochs):
  if epoch==a.warmup_epochs:
   for n,q in model.module.backbone.named_parameters():q.requires_grad_(n in enabled)
  sampler.set_epoch(epoch);model.train();total=torch.zeros((),device=device)
  for batch in loader:
   b={k:v.to(device,non_blocking=True) for k,v in batch.items()}
   with torch.autocast('cuda',dtype=torch.bfloat16):
    z=model(b['frames'],b['candidate'],b['shared'],b['audio']);positive=b['label']!=NO_EVENT_INDEX;loss=F.cross_entropy(z['class_logits'],b['label'])+.5*F.binary_cross_entropy_with_logits(z['quality_logits'],positive.float())
    if positive.any():loss=loss+2*F.smooth_l1_loss(z['time_delta_sec'][positive],b['delta'][positive].clamp(-2,2))
    loss=loss+.01*F.relu(.15-(z['crop_boxes'][:,0,:2]-z['crop_boxes'][:,1,:2]).norm(dim=-1)).mean()
   optim.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optim.step();total+=loss.detach()
  dist.all_reduce(total);total/=dist.get_world_size()
  if rank==0:
   torch.save({'epoch':epoch,'model':model.module.state_dict(),'no_temporal_nms':True,'world_size':dist.get_world_size()},out/'last.pt');print(json.dumps({'epoch':epoch,'mean_rank_loss':float(total/max(len(loader),1)),'world_size':dist.get_world_size()}),flush=True)
  dist.barrier()
 dist.destroy_process_group()
if __name__=='__main__':main()
