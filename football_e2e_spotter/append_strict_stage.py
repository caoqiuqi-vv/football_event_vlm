#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--annotations',required=True);p.add_argument('--work',required=True);a=p.parse_args(); q=Path(a.plan); x=json.loads(q.read_text()); x['stages'].append({'name':'strict_calibration_metrics','command':f'PYTHONPATH=.:football_e2e_spotter/src python football_e2e_spotter/strict_set_metrics.py --predictions {a.work}/calibration/predictions.jsonl --annotations {a.annotations} --output {a.work}/calibration/strict_metrics.json'});q.write_text(json.dumps(x,indent=2)+'\n')
if __name__=='__main__':main()
