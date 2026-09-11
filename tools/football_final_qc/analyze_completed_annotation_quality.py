#!/usr/bin/env python3
"""Count missed and incorrect annotations on fully first-pass-reviewed videos."""
from __future__ import annotations
import argparse,json
from collections import Counter
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cases',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 data=json.loads(a.cases.read_text());completed={vid for vid,row in data['videos'].items() if row.get('first_pass_complete')}
 tolerance=float(data.get('source',{}).get('match_tolerance_sec',5.0))
 cases=[x for x in data['cases'] if x['video_id'] in completed];per_video={vid:{'missed':Counter(),'wrong':Counter()} for vid in completed}
 seen_new=set();seen_wrong=set();details=[];offset_exclusions=[]
 for case in cases:
  vid=case['video_id'];gt={x['id']:x for x in case['original_gt']}
  mapped={gid:[] for gid in gt}
  for event in case['first_pass_events']:
   gids=list(event.get('lineage_gt_ids') or [])
   if not gids and event.get('inferred_gt_id'):gids=[event['inferred_gt_id']]
   valid=[gid for gid in gids if gid in gt]
   for gid in valid:mapped[gid].append(event)
   for gid in valid:
    event_gt=gt[gid];delta=abs(float(event['time_sec'])-float(event_gt['time_sec']))
    if event['semantic_label']==event_gt['semantic_label'] and 3.0 < delta <= tolerance:
     offset_exclusions.append({'video_id':vid,'case_id':case['id'],'event_id':event['id'],'gt_id':gid,'label':event['semantic_label'],'gt_time_sec':event_gt['time_sec'],'human_time_sec':event['time_sec'],'abs_delta_sec':delta})
   if not valid and event['semantic_label']!='throw_in' and event['id'] not in seen_new:
    seen_new.add(event['id']);per_video[vid]['missed'][event['semantic_label']]+=1
    details.append({'kind':'missed_annotation','video_id':vid,'case_id':case['id'],'event_id':event['id'],'label':event['semantic_label'],'time_sec':event['time_sec']})
  for gid,event_gt in gt.items():
   if gid in seen_wrong:continue
   events=mapped.get(gid,[]);subtype=None
   if not events:subtype='gt_deleted_no_human_event'
   elif not any(x['semantic_label']==event_gt['semantic_label'] for x in events):subtype='wrong_label'
   elif not any(abs(float(x['time_sec'])-float(event_gt['time_sec']))<=tolerance for x in events if x['semantic_label']==event_gt['semantic_label']):subtype=f'wrong_time_over_{tolerance:g}s'
   if subtype:
    seen_wrong.add(gid);per_video[vid]['wrong'][subtype]+=1
    details.append({'kind':'incorrect_gt','subtype':subtype,'video_id':vid,'case_id':case['id'],'gt_id':gid,'label':event_gt['semantic_label'],'time_sec':event_gt['time_sec']})
 aggregate_missed=sum((row['missed'] for row in per_video.values()),Counter());aggregate_wrong=sum((row['wrong'] for row in per_video.values()),Counter())
 gt_run=Path(data['source']['gt_run_dir']);gt_counts=Counter()
 for vid in completed:gt_counts.update(x['label'] for x in json.loads((gt_run/vid/'gt_events.json').read_text()))
 gt_total=sum(gt_counts.values());missed_total=sum(aggregate_missed.values());wrong_total=sum(aggregate_wrong.values());core_missed=missed_total-aggregate_missed['back_pass']
 # A single event can occur in overlapping source segments. Keep the audit
 # count provenance-unique so a timestamp-offset duplicate is reported once.
 unique_offset_exclusions={row['event_id']:row for row in offset_exclusions}
 result={'schema_version':3,'source_cases':str(a.cases.resolve()),'source_created_at':data.get('created_at'),'status':'first-pass quality audit; counts remain provisional until final QC completes','matching':f'semantic class, provenance-aware, {tolerance:g} second QC tolerance (formal model evaluation remains 3 seconds)','completed_videos':len(completed),'completed_video_ids':sorted(completed),'completed_hours':sum(float(data['videos'][vid]['duration_sec']) for vid in completed)/3600,'original_gt':{'total':gt_total,'by_parent_label':dict(sorted(gt_counts.items()))},'missed_annotations':{'total':missed_total,'core_football_events_excluding_back_pass':core_missed,'by_label':dict(sorted(aggregate_missed.items())),'ratio_vs_original_gt':missed_total/gt_total if gt_total else None,'time_offset_candidates_excluded_from_missed':len(unique_offset_exclusions),'time_offset_interval_sec':'(3, 5]','time_offset_details':list(unique_offset_exclusions.values())},'incorrect_gt':{'total':wrong_total,'by_type':dict(sorted(aggregate_wrong.items())),'ratio_of_original_gt':wrong_total/gt_total if gt_total else None},'per_video':{vid:{'missed_annotations':{'total':sum(row['missed'].values()),'by_label':dict(sorted(row['missed'].items()))},'incorrect_gt':{'total':sum(row['wrong'].values()),'by_type':dict(sorted(row['wrong'].items()))}} for vid,row in sorted(per_video.items())},'details':details}
 a.output.parent.mkdir(parents=True,exist_ok=True)
 if a.output.exists():raise FileExistsError(f'refusing to overwrite {a.output}; use a new versioned path')
 a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('completed_videos','completed_video_ids','completed_hours','missed_annotations','incorrect_gt')},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
