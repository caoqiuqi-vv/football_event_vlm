#!/usr/bin/env python
"""Watch motion4 train -> motion4 calibration -> multirate block training."""

from __future__ import annotations
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];MANIFEST=ROOT/"football_longform_v2/experiments/lf_a0_official_fullscale/canonical_manifest.json";MOTION=Path("/mnt/data_16t/football/goal_motion4_v1");EXPERIMENT=ROOT/"football_e2e_spotter/experiments/goal_multirate_block_v1";OUTPUT=ROOT/"outputs/football_goal_retriever/block20_multirate_dino_motion4_audio4_v1_20260830";GPUS=(0,1,2,4,5,7)
def ids(split):return [str(x["media_id"]) for x in json.loads(MANIFEST.read_text())[split]]
def ready(split):return sum((MOTION/split/v/"metadata.json").is_file() for v in ids(split))
def status(stage,**kw):
    EXPERIMENT.mkdir(parents=True,exist_ok=True);payload={"stage":stage,"time":time.time(),"train_ready":ready("train"),"calibration_ready":ready("calibration"),**kw};tmp=EXPERIMENT/".status.tmp";tmp.write_text(json.dumps(payload,indent=2)+"\n");os.replace(tmp,EXPERIMENT/"status.json");print(json.dumps(payload),flush=True)
def main():
    while ready("train")<len(ids("train")):status("waiting_motion4_train");time.sleep(30)
    status("motion4_train_complete")
    if ready("calibration")<len(ids("calibration")):
        processes=[]
        for shard in range(6):
            handle=(EXPERIMENT/f"motion4_cal_shard{shard}.log").open("a");cmd=[sys.executable,str(ROOT/"football_e2e_spotter/scripts/build_goal_motion4_ffmpeg.py"),"--manifest",str(MANIFEST),"--output-root",str(MOTION),"--split","calibration","--shard-index",str(shard),"--num-shards","6"];processes.append((subprocess.Popen(cmd,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True),handle))
        status("motion4_calibration_running",pids=[p.pid for p,h in processes])
        failures=[]
        for process,handle in processes:
            code=process.wait();handle.close();failures.append(code) if code else None
        if failures:raise RuntimeError(f"cal motion failures {failures}")
    status("motion4_calibration_complete")
    OUTPUT.mkdir(parents=True,exist_ok=True);log=(EXPERIMENT/"train.log").open("a");env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=",".join(map(str,GPUS));cmd=["torchrun","--standalone","--nproc_per_node=6",str(ROOT/"football_e2e_spotter/scripts/train_goal_multirate_block_v2.py"),"--motion4-root",str(MOTION),"--pixel-root","/mnt/data_16t/football/set_spotter_4fps_288x512","--manifest",str(MANIFEST),"--feature-root","/mnt/data_16t/football/goal_feature_bank_dino_v1","--output",str(OUTPUT),"--epochs","15","--batch-size","16","--eval-batch-size","32","--lr","0.0003","--hidden-dim","384"]
    process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);status("multirate_training",pid=process.pid,log=str(EXPERIMENT/"train.log"));code=process.wait();log.close()
    if code:raise RuntimeError(f"training failed {code}")
    reports=[json.loads(p.read_text()) for p in sorted(OUTPUT.glob("calibration_epoch*.json"))];best=min(reports,key=lambda x:(not x["gate_pass"],x["review_ratio"])) if reports else None;status("complete",best=best)
if __name__=="__main__":main()

