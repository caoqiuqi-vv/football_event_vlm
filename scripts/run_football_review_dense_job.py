#!/usr/bin/env python3
"""Run fixed-checkpoint dense test18 and cached-score LOOV diagnosis durably."""
import argparse,fcntl,json,os,signal,subprocess
from pathlib import Path
from datetime import datetime,timezone

def main():
    p=argparse.ArgumentParser();p.add_argument('--job-root',required=True);p.add_argument('--source-root',required=True);p.add_argument('--python',required=True);p.add_argument('--gpu',type=int,default=4);p.add_argument('--checkpoint',required=True);p.add_argument('--annotations',required=True);p.add_argument('--video-root',required=True);a=p.parse_args()
    root=Path(a.job_root).resolve();source=Path(a.source_root).resolve();lock=(root/'job.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state={'pid':os.getpid(),'gpu':a.gpu,'checkpoint':a.checkpoint,'started_at':datetime.now(timezone.utc).isoformat(),'status':'starting'};child=None
    def save():
        state['updated_at']=datetime.now(timezone.utc).isoformat();tmp=root/'state.json.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(root/'state.json')
    def stop(signum,frame):
        state['status']='interrupted';save()
        if child and child.poll() is None:child.terminate()
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),PYTHONPATH=str(source),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    evaluation=root/'evaluation'
    stages=[('dense_evaluation',[a.python,str(source/'scripts/evaluate_football_review_test18.py'),'--checkpoint',a.checkpoint,'--annotations',a.annotations,'--video-root',a.video_root,'--output',str(evaluation)]),('loov_diagnosis',[a.python,str(source/'scripts/summarize_football_review_test18.py'),'--evaluation',str(evaluation),'--checkpoint',a.checkpoint])]
    try:
        for stage,command in stages:
            state.update(status=stage,command=command);save()
            with (root/(stage+'.log')).open('w') as log:
                child=subprocess.Popen(command,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT);state['child_pid']=child.pid;save();code=child.wait()
            if code:raise RuntimeError(f'{stage} exited {code}; see {stage}.log')
        state.update(status='complete',child_pid=None,report=str(evaluation/'dense_analysis/REPORT.md'));save()
    except Exception as exc:
        state.update(status='failed',error=str(exc),child_pid=None);save();raise

if __name__=='__main__':main()
