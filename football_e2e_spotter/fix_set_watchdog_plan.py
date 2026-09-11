#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',required=True);a=p.parse_args();q=Path(a.plan);x=json.loads(q.read_text())
 for stage in x['stages']:
  if stage['name']=='cache_4fps':stage['command']=stage['command']+' --split train && PYTHONPATH=.:football_e2e_spotter/src python football_e2e_spotter/scripts/build_pixel_audio_store.py --config football_e2e_spotter/configs/set_spotter_v1_cache.yaml --workers 8 --split calibration'
  if stage['name']=='strict_calibration_metrics':stage['command']=stage['command'].replace('strict_set_metrics.py','strict_set_metrics_v2.py')
 q.write_text(json.dumps(x,indent=2)+'\n')
if __name__=='__main__':main()
