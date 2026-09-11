#!/usr/bin/env python3
"""Compare current review tasks with provenance-aware 3-second multi-label bundles."""
from __future__ import annotations
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path
FAMILY={"shot":"shot_save","save":"shot_save","set_piece":"set_piece"}
FAMILY_LABELS={"shot_save":{"shot","save"},"set_piece":{"set_piece"}}
RAW={"射门":"shot","扑救":"save","任意球":"free_kick","点球":"penalty","角球":"corner","中圈开球":"kickoff"}
def union_seconds(rows):
 total=0.;end=-1.
 for a,b in sorted(rows):
  if b<=a:continue
  if a>end:total+=b-a;end=b
  elif b>end:total+=b-end;end=b
 return total
def main():
 p=argparse.ArgumentParser();p.add_argument('--queue',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--tolerance-sec',type=float,default=3.0);p.add_argument('--context-sec',type=float,default=5.0);a=p.parse_args()
 manifest=json.loads(a.manifest.read_text());videos={str(v['video_id']):v for v in manifest['videos']};rows=[json.loads(x) for x in a.queue.read_text().splitlines() if x.strip()]
 anchors=defaultdict(dict)
 for row in rows:
  vid=str(row['video_id']);rowlabels=set(row.get('labels',[]))
  for x in row.get('anchors',[]):
   t=round(float(x['time_sec']),3)
   if x.get('source')=='gt':key=('gt',str(x['label']),t);labels={str(x['label'])};family=FAMILY.get(str(x['label']),str(x['label']))
   else:key=('candidate',str(x.get('family','')),t);family=str(x.get('family',''));labels=rowlabels & FAMILY_LABELS.get(family,rowlabels)
   z=anchors[vid].setdefault(key,{'source':key[0],'time_sec':t,'family':family,'labels':set(),'support':[]})
   z['labels']|=labels;z['support'].append((float(row['start_sec']),float(row['end_sec'])))
 current={};current_sum=0.;current_union=0.;current_tasks=0
 for vid,v in videos.items():
  seg={}
  for e in v['events']:
   z=seg.setdefault(str(e.get('segment_id') or e['id']),[float(e['support_start_sec']),float(e['support_end_sec'])]);z[0]=min(z[0],float(e['support_start_sec']));z[1]=max(z[1],float(e['support_end_sec']))
  iv=list(map(tuple,seg.values()));current[vid]=iv;current_tasks+=len(iv);current_sum+=sum(b-a for a,b in iv);current_union+=union_seconds(iv)
 proposed={};proposed_sum=0.;proposed_union=0.;proposed_tasks=0;attached_candidate_to_gt=0
 for vid,bykey in anchors.items():
  vals=list(bykey.values());gts=sorted((x for x in vals if x['source']=='gt'),key=lambda x:x['time_sec']);cands=sorted((x for x in vals if x['source']=='candidate'),key=lambda x:x['time_sec']);tasks=[]
  # Distinct same-label GT instances never share one task. Different labels may share one playback/multi-label decision.
  for x in gts:
   options=[q for q in tasks if x['labels'].isdisjoint(q['gt_labels']) and max([x['time_sec'],*q['times']])-min([x['time_sec'],*q['times']])<=a.tolerance_sec]
   if options:q=min(options,key=lambda z:abs(sum(z['times'])/len(z['times'])-x['time_sec']));q['anchors'].append(x);q['times'].append(x['time_sec']);q['gt_labels']|=x['labels']
   else:tasks.append({'anchors':[x],'times':[x['time_sec']],'gt_labels':set(x['labels']),'candidate_families':set()})
  for x in cands:
   gt_options=[q for q in tasks if q['gt_labels'] and min(abs(x['time_sec']-t) for t in q['times'])<=a.tolerance_sec and max([x['time_sec'],*q['times']])-min([x['time_sec'],*q['times']])<=a.tolerance_sec]
   if gt_options:
    q=min(gt_options,key=lambda z:min(abs(x['time_sec']-t) for t in z['times']));attached_candidate_to_gt+=1
   else:
    cand_options=[q for q in tasks if not q['gt_labels'] and x['family'] not in q['candidate_families'] and max([x['time_sec'],*q['times']])-min([x['time_sec'],*q['times']])<=a.tolerance_sec]
    if cand_options:q=min(cand_options,key=lambda z:abs(sum(z['times'])/len(z['times'])-x['time_sec']))
    else:q={'anchors':[],'times':[],'gt_labels':set(),'candidate_families':set()};tasks.append(q)
   q['anchors'].append(x);q['times'].append(x['time_sec']);q['candidate_families'].add(x['family'])
  iv=[]
  duration=float(videos[vid]['duration_sec'])
  for q in tasks:iv.append((max(0.,min(q['times'])-a.context_sec),min(duration,max(q['times'])+a.context_sec)))
  proposed[vid]=iv;proposed_tasks+=len(iv);proposed_sum+=sum(b-a for a,b in iv);proposed_union+=union_seconds(iv)
 # Candidate coverage is computed independently before/after bundling; bundling moves evidence but never drops it.
 recall=defaultdict(lambda:[0,0]);subrec=defaultdict(lambda:[0,0])
 for vid in videos:
  vals=list(anchors.get(vid,{}).values());cands=[x for x in vals if x['source']=='candidate'];gtrows=json.loads((a.run_dir/vid/'gt_events.json').read_text())
  for g in gtrows:
   label=str(g['label']);t=float(g['time_sec']);recall[label][1]+=1
   hit=any(label in x['labels'] and abs(x['time_sec']-t)<=a.tolerance_sec for x in cands)
   recall[label][0]+=int(hit)
   raw=str(g.get('raw_label') or g.get('event_type') or '');sem=RAW.get(raw,label);subrec[sem][1]+=1;subrec[sem][0]+=int(hit)
 orig=sum(float(x['duration_sec']) for x in videos.values())
 result={'policy':{'candidate_gt_tolerance_sec':a.tolerance_sec,'context_each_side_sec':a.context_sec,'distinct_same_label_gt_never_merged':True,'candidate_evidence_removed':False},'baseline':{'tasks':current_tasks,'sum_playback_hours':current_sum/3600,'sum_playback_vs_original_pct':100*current_sum/orig,'unique_video_hours':current_union/3600,'unique_video_vs_original_pct':100*current_union/orig},'proposed':{'tasks':proposed_tasks,'sum_playback_hours':proposed_sum/3600,'sum_playback_vs_original_pct':100*proposed_sum/orig,'unique_video_hours':proposed_union/3600,'unique_video_vs_original_pct':100*proposed_union/orig,'candidate_anchors_attached_to_gt_tasks':attached_candidate_to_gt},'compression':{'task_reduction_pct':100*(1-proposed_tasks/current_tasks),'sum_playback_reduction_pct':100*(1-proposed_sum/current_sum),'unique_video_reduction_pct':100*(1-proposed_union/current_union)},'candidate_recall_at_3s':{k:{'hit':x[0],'gt':x[1],'recall':x[0]/x[1] if x[1] else 0} for k,x in sorted(recall.items())},'candidate_recall_by_semantic_at_3s':{k:{'hit':x[0],'gt':x[1],'recall':x[0]/x[1] if x[1] else 0} for k,x in sorted(subrec.items())},'gt_anchor_coverage':{'all':sum(x[1] for x in recall.values()),'covered':sum(x[1] for x in recall.values()),'recall':1.0}}
 if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
