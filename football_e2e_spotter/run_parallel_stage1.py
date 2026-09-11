#!/usr/bin/env python3
"""GPU-parallel, restart-safe OOF and full Stage-1 executor; never uses NMS."""
from __future__ import annotations
import argparse,json,os,subprocess,time
from pathlib import Path
def lines(p):return [x.strip() for x in Path(p).read_text().splitlines() if x.strip()]
def run(command,gpu,log):
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1')
 return subprocess.Popen(command,shell=True,env=env,stdout=log,stderr=subprocess.STDOUT)
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--work',required=True);p.add_argument('--gpus',default='0,1,2,6');p.add_argument('--epochs',type=int,default=12);p.add_argument('--wait-cache',action='store_true');a=p.parse_args();import yaml
 c=yaml.safe_load(Path(a.config).read_text());w=Path(a.work);f=w/'folds';f.mkdir(parents=True,exist_ok=True);state=w/'parallel_stage1_state.json';log_path=w/'parallel_stage1.log';gpus=[x for x in a.gpus.split(',') if x];train=lines(c['data']['train_ids']);calib=c['data']['calibration_ids'];jobs=[]
 for k in range(5):
  va=train[k::5];tr=[x for i,x in enumerate(train) if i%5!=k];tp=f/f'fold{k}_train.txt';vp=f/f'fold{k}_valid.txt';tp.write_text('\n'.join(tr)+'\n');vp.write_text('\n'.join(va)+'\n');ck=w/f'stage1_fold{k}';out=w/'oof'/f'fold{k}.jsonl';cmd=f'PYTHONPATH=.:football_e2e_spotter/src python football_e2e_spotter/train_set_spotter_ids.py --config {a.config} --ids {tp} --output {ck} --epochs {a.epochs} && PYTHONPATH=.:football_e2e_spotter/src python football_e2e_spotter/run_set_candidates.py --config {a.config} --checkpoint {ck}/last.pt --ids {vp} --split train --output {out} --threshold 0.01';jobs.append((f'fold{k}',cmd))
 full=w/'stage1_full';jobs.append(('full',f'PYTHONPATH=.:football_e2e_spotter/src python football_e2e_spotter/train_set_spotter_ids.py --config {a.config} --ids {c["data"]["train_ids"]} --output {full} --epochs {a.epochs} && PYTHONPATH=.:football_e2e_spotter/src python football_e2e_spotter/run_set_candidates.py --config {a.config} --checkpoint {full}/last.pt --ids {calib} --split calibration --output {w}/calibration/candidates.jsonl --threshold 0.01'))
 if a.wait_cache:
  expected=len(train)+len(lines(calib));root=Path(c['data']['frame_store']);
  while len(list(root.glob('*/*/metadata.json')))<expected:time.sleep(60)
 completed=set(json.loads(state.read_text()).get('completed',[]) if state.exists() and state.stat().st_size else []);pending=[j for j in jobs if j[0] not in completed];active={}
 with log_path.open('a') as log:
  while pending or active:
   while pending and len(active)<len(gpus):
    name,cmd=pending.pop(0);gpu=next(x for x in gpus if x not in [y[0] for y in active.values()]);active[name]=(gpu,run(cmd,gpu,log));log.write(f'{name} gpu={gpu}\n');log.flush()
   time.sleep(20)
   for name,(gpu,proc) in list(active.items()):
    code=proc.poll()
    if code is not None:
     if code:raise RuntimeError(f'{name} failed: {code}')
     completed.add(name);state.write_text(json.dumps({'completed':sorted(completed),'no_temporal_nms':True},indent=2)+'\n');del active[name]
 print(json.dumps({'status':'complete','completed':sorted(completed),'no_temporal_nms':True}))
if __name__=='__main__':main()
