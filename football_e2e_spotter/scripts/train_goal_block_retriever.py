#!/usr/bin/env python
"""Six-GPU training/evaluation for the KPI-aligned 20-second retriever."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

ROOT = Path(__file__).resolve().parents[2]; PKG = ROOT / "football_e2e_spotter/src"
for path in (ROOT, PKG):
    if str(path) not in sys.path: sys.path.insert(0, str(path))

from football_e2e_spotter.goal_annotations import LABELS, load_goal_annotations  # noqa: E402
from football_e2e_spotter.goal_block_data import GoalBlockDataset  # noqa: E402
from football_e2e_spotter.goal_block_loss import block_retriever_loss  # noqa: E402
from football_e2e_spotter.goal_block_retriever import GoalBlockRetriever  # noqa: E402
from football_e2e_spotter.goal_retriever_eval_v2 import adaptive_predictions, robust_location_scale, logits  # noqa: E402


def setup():
    world=int(os.environ.get("WORLD_SIZE","1")); rank=int(os.environ.get("RANK","0")); local=int(os.environ.get("LOCAL_RANK","0"))
    if world>1: dist.init_process_group("nccl")
    torch.cuda.set_device(local); return rank,world,local,torch.device(f"cuda:{local}")


def block_curve(rows, gt_by_video, floor):
    ordered=sorted(rows,key=lambda row:(-row["score"],row["video_id"],row["start"]))
    total=sum(len(v) for v in gt_by_video.values()); captured={v:set() for v in gt_by_video}; positive_blocks=0; best=None
    for index,row in enumerate(ordered,1):
        video=row["video_id"]; hits=[i for i,t in enumerate(gt_by_video.get(video,())) if row["start"]<=t<row["end"]]
        new=[i for i in hits if i not in captured.setdefault(video,set())]
        captured[video].update(new); positive_blocks += bool(hits)
        recall=sum(len(v) for v in captured.values())/max(total,1); precision=positive_blocks/index
        candidate={"threshold":row["score"],"recall":recall,"precision":precision,"selected_blocks":index,"captured_events":sum(len(v) for v in captured.values()),"total_events":total}
        if recall>=floor and (best is None or (precision,candidate["threshold"])>(best["precision"],best["threshold"])): best=candidate
    if best is None:
        best={"threshold":ordered[-1]["score"] if ordered else 1.0,"recall":0.0,"precision":0.0,"selected_blocks":len(ordered),"captured_events":0,"total_events":total}
        reachable=False
    else: reachable=True
    return {**best,"recall_floor":floor,"recall_floor_reachable":reachable}


def evaluate(model,dataset,device,batch_size,floors):
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=0)
    rows_by_label={label:[] for label in LABELS}; durations={}; model.eval()
    with torch.inference_mode():
        for batch in loader:
            with torch.autocast("cuda",dtype=torch.bfloat16):
                output=model(batch["appearance"].to(device),batch["motion"].to(device),batch["audio"].to(device),batch["valid"].to(device))
                score=(output["block_logits"].sigmoid()*output["any_logits"].sigmoid().unsqueeze(-1).sqrt()).float().cpu().numpy()
            for b,(video,start) in enumerate(zip(batch["video_id"],batch["core_start"].numpy())):
                durations[video]=max(durations.get(video,0.0),float(start)+60.0)
                for block in range(3):
                    for label_index,label in enumerate(LABELS):
                        rows_by_label[label].append({"video_id":video,"start":float(start+20*block),"end":float(start+20*(block+1)),"score":float(score[b,block,label_index])})
    gt={label:{} for label in LABELS}
    manifest_by_id={str(item["media_id"]):item for item in dataset.manifest}
    for video in durations:
        events=load_goal_annotations(manifest_by_id[video]["annotation_path"])
        for label in LABELS: gt[label][video]=[e.timestamp for e in events if e.accepted and e.label==label]
    selected={}; transformed_selected={}; searches={}
    for label in LABELS:
        raw=rows_by_label[label]; raw_logits=logits(np.asarray([r["score"] for r in raw])); median,scale=robust_location_scale(raw_logits)
        variants=[]; transformed={}
        for alpha in (0.,.25,.5,.75,1.):
            values=adaptive_predictions(raw,alpha,median,scale); transformed[alpha]=values
            metric=block_curve(values,gt[label],floors[label]); variants.append({**metric,"adaptive_alpha":alpha,"global_logit_median":median,"global_logit_scale":scale})
        chosen=max(variants,key=lambda x:(x["recall_floor_reachable"],x["precision"],x["recall"],x["threshold"])); selected[label]=chosen; searches[label]=variants
        transformed_selected[label]=[r for r in transformed[chosen["adaptive_alpha"]] if r["score"]>=chosen["threshold"]]
    intervals=set()
    for label,values in transformed_selected.items():
        intervals.update((r["video_id"],r["start"],r["end"]) for r in values)
    total=sum(durations.values()); reviewed=sum(min(end,durations[video])-start for video,start,end in intervals)
    return {"protocol":{"disjoint_20s_blocks":True,"no_nms":True,"review_union":True},"classes":selected,"adaptive_search":searches,"selected_union_blocks":len(intervals),"review_seconds":reviewed,"total_video_seconds":total,"review_ratio":reviewed/max(total,1),"gate_pass":all(v["recall_floor_reachable"] for v in selected.values()) and reviewed/max(total,1)<.35}


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--feature-root",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=15); p.add_argument("--batch-size",type=int,default=32); p.add_argument("--eval-batch-size",type=int,default=64); p.add_argument("--lr",type=float,default=3e-4); p.add_argument("--hidden-dim",type=int,default=384); p.add_argument("--seed",type=int,default=5519); return p.parse_args()


def main():
    args=parse_args(); args.manifest=args.manifest.resolve(); args.feature_root=args.feature_root.resolve(); args.output=args.output.resolve()
    rank,world,local,device=setup(); random.seed(args.seed+rank); np.random.seed(args.seed+rank); torch.manual_seed(args.seed+rank)
    train=GoalBlockDataset(manifest=args.manifest,feature_root=args.feature_root,split="train"); cal=GoalBlockDataset(manifest=args.manifest,feature_root=args.feature_root,split="calibration") if rank==0 else None
    sampler=DistributedSampler(train,num_replicas=world,rank=rank,shuffle=True,seed=args.seed) if world>1 else None
    loader=DataLoader(train,batch_size=args.batch_size,sampler=sampler,shuffle=sampler is None,num_workers=0,pin_memory=True,drop_last=True)
    appearance_dim=train[0]["appearance"].shape[-1]; model=GoalBlockRetriever(appearance_dim,hidden_dim=args.hidden_dim).to(device)
    wrapped=DDP(model,device_ids=[local],broadcast_buffers=False) if world>1 else model
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.05,betas=(.9,.98)); total=args.epochs*len(loader); step=0; best=None
    if rank==0: args.output.mkdir(parents=True,exist_ok=True)
    floors={label:(.95 if label=="shot" else .90) for label in LABELS}
    for epoch in range(1,args.epochs+1):
        if sampler: sampler.set_epoch(epoch)
        wrapped.train(); sums={}; seen=0; started=time.time()
        for batch in loader:
            progress=step/max(total,1); lr=args.lr*(.05+.95*.5*(1+math.cos(math.pi*progress)))
            for group in optimizer.param_groups: group["lr"]=lr
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                output=wrapped(batch["appearance"].to(device),batch["motion"].to(device),batch["audio"].to(device),batch["valid"].to(device)); loss,stats=block_retriever_loss(output,batch["block_targets"].to(device),batch["dense_targets"].to(device))
            loss.backward(); torch.nn.utils.clip_grad_norm_(wrapped.parameters(),2.0); optimizer.step(); step+=1; seen+=1
            for k,v in stats.items(): sums[k]=sums.get(k,0)+v
        if world>1: dist.barrier()
        if rank==0:
            report=evaluate(model,cal,device,args.eval_batch_size,floors); report["epoch"]=epoch
            (args.output/f"calibration_epoch{epoch:02d}.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
            score=(report["gate_pass"],-report["review_ratio"],sum(v["recall"] for v in report["classes"].values())); is_best=best is None or score>best; best=max(best,score) if best is not None else score
            state={"schema":"football.goal_block_retriever.v1","epoch":epoch,"model":model.state_dict(),"optimizer":optimizer.state_dict(),"config":{"appearance_dim":appearance_dim,"hidden_dim":args.hidden_dim},"calibration":report,"best_score":best}; torch.save(state,args.output/"last.pt")
            if is_best: torch.save(state,args.output/"best.pt")
            print(json.dumps({"epoch":epoch,"seconds":round(time.time()-started,1),"train":{k:v/max(seen,1) for k,v in sums.items()},"review_ratio":report["review_ratio"],"recall":{k:v["recall"] for k,v in report["classes"].items()},"precision":{k:v["precision"] for k,v in report["classes"].items()},"gate_pass":report["gate_pass"],"best":is_best},ensure_ascii=False),flush=True)
        if world>1: dist.barrier()
    if world>1: dist.destroy_process_group()


if __name__=="__main__": main()

