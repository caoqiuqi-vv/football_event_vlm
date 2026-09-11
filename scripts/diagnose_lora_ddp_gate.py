#!/usr/bin/env python3
"""Two-rank torchrun hard gate for LoRA gradients on the real football path."""
from __future__ import annotations
import argparse, json, math, os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import train_football_events as football


def norm(x):
    return None if x is None else float(torch.linalg.vector_norm(x.detach().float()).cpu())


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--gradient-checkpointing", choices=("on","off"), default="on")
    ap.add_argument("--diagnostic-num-frames", type=int, default=4)
    args=ap.parse_args()
    local_rank=int(os.environ["LOCAL_RANK"]); world=int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank); dist.init_process_group("nccl")
    device=torch.device(f"cuda:{local_rank}")
    cfg=football.load_config(args.config, [])
    cfg["device"]=str(device); cfg["gpu_ids"]=list(range(world))
    cfg.model.gradient_checkpointing=args.gradient_checkpointing=="on"
    cfg.train.per_gpu_batch_size=1; cfg.train.grad_accum_steps=1
    cfg.train.resume.enabled=False; cfg.data.num_workers_per_gpu=0
    football.configure_label_schema(cfg); football.configure_runtime_threads(cfg)
    football.seed_everything(int(cfg.get("seed",42))+dist.get_rank(), bool(cfg.get("deterministic",True)))
    topology=football.resolve_runtime_topology(cfg, device); cfg.data.num_workers=0
    ds,_vd,records,_vr=football.prepare_datasets(cfg,use_cache=False)
    ds.num_frames=min(args.diagnostic_num_frames,int(cfg.video.num_frames))
    loader=football.make_loader(ds,cfg,is_train=True,batch_size=1,distributed=True)
    model=football.make_model(cfg,use_cached_features=False,device=device)
    model=DDP(model,device_ids=[local_rank],output_device=local_rank,broadcast_buffers=False,find_unused_parameters=True)
    optimizer=football.build_optimizer(model,cfg); optimizer.zero_grad(set_to_none=True); model.train()
    modules={n:m for n,m in model.module.named_modules() if isinstance(m,football.LoRALinear)}
    before={n:m.lora_b.detach().clone() for n,m in modules.items()}
    batch=next(iter(loader)); targets=batch["targets"].to(device); masks=batch["label_masks"].to(device)
    pos,_=football.resolve_pos_weight(records,cfg.train); pos=pos.to(device)
    with football.autocast_context(device,bool(cfg.train.amp),str(cfg.train.amp_dtype)):
        out=football.forward_model_batch(model,batch,device,return_aux=True); logits=out["logits"]
        matrix=torch.nn.functional.binary_cross_entropy_with_logits(logits,targets.to(logits.dtype),pos_weight=pos.to(logits.dtype),reduction="none")
        loss=(matrix*masks).sum()/masks.sum().clamp_min(1)
    loss.backward()
    grad={n:norm(m.lora_b.grad) for n,m in modules.items()}
    optimizer.step()
    update={n:norm(m.lora_b.detach()-before[n]) for n,m in modules.items()}
    effective={n:norm((m.lora_b.detach().float()@m.lora_a.detach().float())*float(m.scaling)) for n,m in modules.items()}
    local_fail=[n for n in modules if not grad[n] or not math.isfinite(grad[n]) or not update[n] or not effective[n]]
    fail_tensor=torch.tensor([len(local_fail)],device=device,dtype=torch.int64); dist.all_reduce(fail_tensor)
    result={"status":"PASS" if int(fail_tensor.item())==0 else "FAIL","world_size":world,"gradient_checkpointing":bool(cfg.model.gradient_checkpointing),"loss":float(loss.detach()),"lora_b_grad_norm":grad,"lora_b_update_norm":update,"effective_delta_norm":effective,"local_failures":local_fail,"topology":football.to_plain(topology)}
    if dist.get_rank()==0:
        args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n"); print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    dist.barrier(); dist.destroy_process_group()
    if int(fail_tensor.item())!=0: raise RuntimeError("DDP LoRA gate failed")

if __name__=="__main__": main()
