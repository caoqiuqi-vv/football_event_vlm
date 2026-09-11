#!/usr/bin/env python3
"""Re-select thresholds using every GT, including events missed by Stage-1."""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
from football_e2e_spotter.set_data import load_set_events
from football_e2e_spotter.set_spotting import SET_LABELS
from football_e2e_spotter.evaluate_set_pipeline import one_to_one
def main():
 p=argparse.ArgumentParser(); p.add_argument('--predictions',required=True); p.add_argument('--annotations',required=True); p.add_argument('--output',required=True); a=p.parse_args(); rows=[json.loads(x) for x in Path(a.predictions).read_text().splitlines() if x.strip()]; targets={i:defaultdict(list) for i in range(5)}
 seen={r['video_id']:str(r.get('annotation_id',r['video_id'])) for r in rows}
 for video,ann in seen.items():
  for label,time in load_set_events(Path(a.annotations)/f'{ann}.json'): targets[label][video].append(time)
 required=[.90,.85,.85,.85,.85]; report={'no_temporal_nms':True,'tolerance_seconds':3.0,'classes':{}}
 for label,name in enumerate(SET_LABELS):
  best=None
  for threshold in sorted({r['final_score'] for r in rows if r['final_label']==label},reverse=True):
   result=one_to_one(rows,label,threshold,targets[label]); candidate={'threshold':threshold,**result}
   if result['recall']>=required[label] and (best is None or result['precision']>best['precision']): best=candidate
  report['classes'][name]=best or {'threshold':1.0,**one_to_one(rows,label,1.0,targets[label])}
 Path(a.output).write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
