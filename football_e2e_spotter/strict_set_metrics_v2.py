#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from collections import defaultdict
from pathlib import Path
from football_e2e_spotter.set_data import load_set_events
from football_e2e_spotter.set_spotting import SET_LABELS
def metric(rows,label,threshold,targets):
 left={v:list(x) for v,x in targets.items()};tp=fp=0
 for r in sorted((x for x in rows if x['final_label']==label and x['final_score']>=threshold),key=lambda x:x['final_score'],reverse=True):
  a=left.get(r['video_id'],[])
  if a:
   i=min(range(len(a)),key=lambda i:abs(a[i]-r['time_sec']))
   if abs(a[i]-r['time_sec'])<=3:tp+=1;a.pop(i);continue
  fp+=1
 fn=sum(len(x) for x in left.values());return {'tp':tp,'fp':fp,'fn':fn,'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1)}
def main():
 p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);p.add_argument('--annotations',required=True);p.add_argument('--output',required=True);a=p.parse_args();rows=[json.loads(x) for x in Path(a.predictions).read_text().splitlines() if x.strip()];targets={i:defaultdict(list) for i in range(5)}
 for video,ann in {r['video_id']:str(r.get('annotation_id',r['video_id'])) for r in rows}.items():
  for label,time in load_set_events(Path(a.annotations)/f'{ann}.json'):targets[label][video].append(time)
 required=(.90,.85,.85,.85,.85);out={'no_temporal_nms':True,'tolerance_seconds':3,'classes':{}}
 for label,name in enumerate(SET_LABELS):
  best=None
  for t in sorted({r['final_score'] for r in rows if r['final_label']==label},reverse=True):
   m=metric(rows,label,t,targets[label])
   if m['recall']>=required[label] and (best is None or m['precision']>best['precision']):best={'threshold':t,**m}
  out['classes'][name]=best or {'threshold':1.,**metric(rows,label,1.,targets[label])}
 Path(a.output).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
