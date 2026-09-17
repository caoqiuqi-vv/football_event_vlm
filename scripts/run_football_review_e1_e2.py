#!/usr/bin/env python3
"""Durable single-GPU E1 -> E2 queue; stop on failure, preserve all logs."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
import signal
import subprocess
import time
from datetime import datetime,timezone
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-root',required=True)
    p.add_argument('--source-root',required=True);p.add_argument('--python',required=True);p.add_argument('--gpu',type=int,default=7)
    p.add_argument('--annotations',required=True);p.add_argument('--video-root',required=True);a=p.parse_args()
    root=Path(a.run_root).resolve();source=Path(a.source_root).resolve();lock=open(root/'queue.lock','w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state={'pipeline_pid':os.getpid(),'gpu':a.gpu,'status':'waiting_for_gpu','stage':'E1','completed':[],
           'started_at':datetime.now(timezone.utc).isoformat()}
    def save():
        state['updated_at']=datetime.now(timezone.utc).isoformat()
        temp=root/'state.json.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(root/'state.json')
    child=None
    def terminate(signum,frame):
        state['status']='interrupted';save()
        if child is not None and child.poll() is None:child.terminate()
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    save()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH=str(source),PYTHONUNBUFFERED='1',
             OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    def wait_for_gpu(stage):
        while True:
            result=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
            values=[line.split(',') for line in result.stdout.splitlines()]
            selected=next(v for v in values if int(v[0])==a.gpu)
            if int(selected[1])<1500 and int(selected[2])<10:return
            state.update(status='waiting_for_gpu',stage=stage);save();time.sleep(30)
    try:
        for stage in ('E1','E2'):
            wait_for_gpu(stage)
            output=root/stage;output.mkdir(exist_ok=False)
            cmd=[a.python,str(source/'train_football_events.py'),'--config',str(root/(stage+'.yaml'))]
            state.update(status='training',stage=stage,command=cmd);save()
            with open(output/'train_console.log','w') as log:
                child=subprocess.Popen(cmd,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
                state['child_pid']=child.pid;save();code=child.wait()
            if code:raise RuntimeError(f'{stage} training exited {code}; see {output}/train_console.log')
            if not (output/'best.pt').is_file():raise RuntimeError(f'{stage} ended without best.pt')
            state['completed'].append(stage+'_training');save()
        # Both matched training experiments run before external evaluation.
        for stage in ('E1','E2'):
            wait_for_gpu(stage+'_test18')
            output=root/stage;cmd=[a.python,str(source/'scripts/evaluate_football_review_test18.py'),
                '--checkpoint',str(output/'best.pt'),'--annotations',a.annotations,
                '--video-root',a.video_root,'--output',str(output/'test18')]
            state.update(status='evaluating_test18',stage=stage,command=cmd);save()
            with open(output/'test18_console.log','w') as log:
                child=subprocess.Popen(cmd,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
                state['child_pid']=child.pid;save();code=child.wait()
            if code:raise RuntimeError(f'{stage} test18 exited {code}; see {output}/test18_console.log')
            state['completed'].append(stage+'_test18');save()
        state.update(status='complete',child_pid=None);save()
    except Exception as exc:
        state.update(status='failed',error=str(exc),child_pid=None);save();raise


if __name__=='__main__':main()
