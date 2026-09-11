#!/usr/bin/env python3
from __future__ import annotations
import os
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import Dataset
import train_football_events as football

class TinyDataset(Dataset):
    def __len__(self):
        return 17
    def __getitem__(self, index):
        target=torch.tensor([index%2, (index//2)%2, (index//3)%2],dtype=torch.float32)
        logits=(target*2.0-1.0)*2.0 + float(index)*0.001
        return {"inputs":logits,"targets":target,"label_masks":torch.ones(3),"cache_key":str(index),"meta":{"source":"tiny","video_id":str(index%3),"sample_id":str(index)}}

class IdentityLogits(nn.Module):
    def forward(self, inputs, return_aux=False, **kwargs):
        return inputs

def main():
    dist.init_process_group("gloo")
    rank=dist.get_rank(); world=dist.get_world_size()
    football.configure_label_schema(football.to_config({"task":{"label_schema":"set_piece"}}))
    cfg=football.to_config({"seed":42,"data":{"num_workers":0,"pin_memory":False},"train":{"batch_size":2,"amp":False,"amp_dtype":"bf16","frame_eval_topk":4},"eval":{"threshold":0.5,"tuned_min_recall":0.0},"model":{"temporal_fusion":"cls_transformer","spatial_attention":{"enabled":False}}})
    dataset=TinyDataset()
    loader=football.make_loader(dataset,cfg,is_train=False,batch_size=2,distributed=True)
    local_indices=list(iter(loader.sampler))
    result=football.evaluate(IdentityLogits(),loader,cfg,torch.device("cpu"))
    gathered=[None for _ in range(world)] if rank==0 else None
    dist.gather_object(local_indices,gathered,dst=0)
    if rank==0:
        flat=[value for values in gathered for value in values]
        evaluated=sum(row["num_samples"] for row in result["per_video_default"])
        summary={"world_size":world,"shard_lengths":[len(v) for v in gathered],"unique_indices":len(set(flat)),"total_indices":len(flat),"evaluated_samples":evaluated,"status":"PASS" if len(flat)==len(set(flat))==evaluated==len(dataset) else "FAIL"}
        print(summary,flush=True)
        if summary["status"] != "PASS": raise RuntimeError(summary)
    dist.barrier(); dist.destroy_process_group()

if __name__=="__main__": main()
