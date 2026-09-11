#!/usr/bin/env python
"""Evaluate one shared any-event block proposal stream for all five classes."""

from __future__ import annotations

import argparse,json,sys
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[2];PKG=ROOT/"football_e2e_spotter/src"
for p in (ROOT,PKG):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
from football_e2e_spotter.goal_annotations import LABELS,load_goal_annotations  # noqa:E402
from football_e2e_spotter.goal_block_data import GoalBlockDataset  # noqa:E402
from football_e2e_spotter.goal_block_retriever import GoalBlockRetriever  # noqa:E402
from football_e2e_spotter.goal_retriever_eval_v2 import adaptive_predictions,logits,robust_location_scale  # noqa:E402


def choose(rows,gt,floors):
    captured={label:{video:set() for video in values} for label,values in gt.items()};total={label:sum(len(x) for x in values.values()) for label,values in gt.items()};positive=0
    for rank,row in enumerate(sorted(rows,key=lambda x:-x["score"]),1):
        has=False
        for label in LABELS:
            for index,t in enumerate(gt[label].get(row["video_id"],())):
                if row["start"]<=t<row["end"]:captured[label][row["video_id"]].add(index);has=True
        positive+=has;recall={label:sum(len(x) for x in captured[label].values())/max(total[label],1) for label in LABELS}
        if all(recall[label]>=floors[label] for label in LABELS):return {"threshold":row["score"],"selected_blocks":rank,"block_precision":positive/rank,"recall":recall,"reachable":True}
    return {"threshold":-1e9,"selected_blocks":len(rows),"block_precision":positive/max(len(rows),1),"recall":{label:sum(len(x) for x in captured[label].values())/max(total[label],1) for label in LABELS},"reachable":False}


def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--manifest",type=Path,required=True);p.add_argument("--feature-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="cuda:0");args=p.parse_args();device=torch.device(args.device)
    ck=torch.load(args.checkpoint,map_location="cpu");model=GoalBlockRetriever(ck["config"]["appearance_dim"],hidden_dim=ck["config"]["hidden_dim"]).to(device);model.load_state_dict(ck["model"]);model.eval();data=GoalBlockDataset(manifest=args.manifest,feature_root=args.feature_root,split="calibration")
    rows=[]
    with torch.inference_mode():
        for b in DataLoader(data,batch_size=64,shuffle=False,num_workers=0):
            with torch.autocast("cuda",dtype=torch.bfloat16):out=model(b["appearance"].to(device),b["motion"].to(device),b["audio"].to(device),b["valid"].to(device));scores=out["any_logits"].sigmoid().float().cpu().numpy()
            for i,(video,start) in enumerate(zip(b["video_id"],b["core_start"].numpy())):
                for block in range(3):rows.append({"video_id":video,"start":float(start+20*block),"end":float(start+20*(block+1)),"score":float(scores[i,block])})
    items={str(x["media_id"]):x for x in data.manifest};gt={label:{} for label in LABELS}
    for video in data.timelines:
        events=load_goal_annotations(items[video]["annotation_path"])
        for label in LABELS:gt[label][video]=[e.timestamp for e in events if e.accepted and e.label==label]
    raw=logits(np.asarray([r["score"] for r in rows]));median,scale=robust_location_scale(raw);floors={label:(.95 if label=="shot" else .90) for label in LABELS};variants=[]
    for alpha in (0.,.25,.5,.75,1.):
        transformed=adaptive_predictions(rows,alpha,median,scale);metric=choose(transformed,gt,floors);metric.update({"alpha":alpha,"review_ratio":metric["selected_blocks"]/max(len(rows),1)});variants.append(metric)
    chosen=min(variants,key=lambda x:(not x["reachable"],x["review_ratio"],-x["block_precision"]));report={"protocol":{"shared_any_event_stream":True,"no_nms":True,"disjoint_20s_blocks":True},"floors":floors,"variants":variants,"selected":chosen,"gate_pass":chosen["reachable"] and chosen["review_ratio"]<.35}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n");print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":main()

