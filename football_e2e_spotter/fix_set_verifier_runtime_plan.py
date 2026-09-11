#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args();q=Path(a.plan);x=json.loads(q.read_text())
 for s in x['stages']:
  s['command']=s['command'].replace('train_set_verifier.py','train_set_verifier_runtime.py').replace('evaluate_set_pipeline.py','evaluate_set_pipeline_runtime.py')
 q.write_text(json.dumps(x,indent=2)+'\n')
if __name__=='__main__':main()
