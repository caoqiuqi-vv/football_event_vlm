#!/usr/bin/env python
"""Supervise the authorized full Stage1 run through recovery and final reporting."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from football_localization_full import atomic_json,digest


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);args=ap.parse_args();cfg=json.loads(Path(args.config).read_text());out=Path(cfg['output_dir'])
    lock=(out/'pipeline.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.environ['STAGE1_FULL_GPUS']=','.join(map(str,cfg['gpus']))
    registry=json.loads((out/'run_provenance.json').read_text())
    def integrity():
        for p,sha in registry['training_code_sha256'].items():
            if digest(p)!=sha:raise RuntimeError('training code changed: '+p)
        if digest(out/'manifest.json')!=registry['manifest_sha256']:raise RuntimeError('manifest changed')
        if digest(cfg['checkpoint_snapshot'])!=registry['source_checkpoint_sha256']:raise RuntimeError('source snapshot changed')
    def status(phase,**kw):atomic_json(out/'pipeline_status.json',{'phase':phase,'updated_unix':time.time(),'supervisor_pid':os.getpid(),**kw})
    attempts=[]
    try:
        for attempt in range(cfg['resume_max_attempts']):
            integrity()
            if (out/'TRAINING_COMPLETE.json').exists():break
            command=['bash','scripts/run_football_localization_full.sh']
            if (out/'resume.pt').exists():command.append('--resume')
            with (out/'train_console.log').open('a',buffering=1) as log:
                log.write(f'\nPIPELINE ATTEMPT {attempt+1}\n');child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                status('training',attempt=attempt+1,child_pid=child.pid)
                while child.poll() is None:
                    # Silence while progressing; state is available in heartbeat.json.
                    time.sleep(20)
                code=child.returncode
            attempts.append({'attempt':attempt+1,'exit_code':code});atomic_json(out/'attempts.json',attempts)
            if code==0 and (out/'TRAINING_COMPLETE.json').exists():break
            if attempt+1==cfg['resume_max_attempts']:raise RuntimeError('training failed after bounded recovery attempts; no completion report generated')
            status('recovering',attempt=attempt+1,exit_code=code);time.sleep(10)
        integrity();status('selected_reference_evaluation')
        if not (out/'expert_selected_metrics.json').exists():
            with (out/'reference_console.log').open('a') as log:
                completed=subprocess.run(['bash','scripts/run_football_localization_full.sh','--evaluate-selected'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            if completed.returncode:raise RuntimeError('selected reference evaluation failed')
        integrity();status('final_audit')
        from scripts.finalize_football_localization_full import finalize
        summary=finalize(out)
        status('complete',report=str(out/'FINAL_REPORT.md'),real_detection_accuracy_verified=summary['real_detection_accuracy_verified'],stage2_started=False)
    except Exception as e:
        status('failed',error=str(e));atomic_json(out/'PIPELINE_FAILED.json',{'error':str(e),'attempts':attempts,'not_complete':True});raise


if __name__=='__main__':main()
