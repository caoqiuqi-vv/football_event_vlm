#!/usr/bin/env python3
# Absorb unreviewed same-family candidates within tolerance into an unreviewed GT task.
from __future__ import annotations
import argparse,json,sqlite3
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
def family(label):return 'shot_save' if label in {'shot','save'} else 'set_piece'
def uniq(items,key):
 out=[];seen=set()
 for x in items:
  k=key(x)
  if k not in seen:seen.add(k);out.append(x)
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--db',type=Path,required=True);p.add_argument('--tolerance-sec',type=float,default=3);p.add_argument('--apply',action='store_true');a=p.parse_args()
 c=sqlite3.connect(a.db);c.row_factory=sqlite3.Row
 rows=c.execute('''select e.*,r.status,r.reviewer,r.revision from events e join reviews r on r.event_id=e.id order by e.video_id,e.source_time_sec''').fetchall();groups=defaultdict(list)
 for row in rows:
  d=dict(row);d['payload']=json.loads(d['payload_json']);groups[(str(d['video_id']),str(d['payload'].get('segment_id') or d['id']))].append(d)
 info={}
 for key,items in groups.items():
  gt=[];cand=[]
  for x in items:
   q=x['payload'];gt += [(str(x['source_label']),float(t)) for t in q.get('matching_gt_times',[])];gt += [(str(z.get('label')),float(z['time_sec'])) for z in q.get('evidence_anchors',[]) if z.get('source')=='gt'];cand += [(str(z.get('family','')),float(z['time_sec']),z) for z in q.get('evidence_anchors',[]) if z.get('source')=='candidate']
  info[key]={'items':items,'gt':uniq(gt,lambda z:(z[0],round(z[1],3))),'cand':uniq(cand,lambda z:(z[0],round(z[1],3))),'unreviewed':all(x['status']=='unreviewed' for x in items)}
 plans=[]
 for ck,z in info.items():
  if z['gt'] or not z['cand'] or not z['unreviewed']:continue
  opts=[]
  for gk,g in info.items():
   if gk[0]!=ck[0] or not g['gt'] or not g['unreviewed']:continue
   for cf,ct,_ in z['cand']:
    for gl,gt in g['gt']:
     if cf==family(gl) and abs(ct-gt)<=a.tolerance_sec:opts.append((abs(ct-gt),gk))
  if opts:plans.append((ck,min(opts)[1],min(opts)[0]))
 print(json.dumps([{'candidate_segment':x[0][1],'gt_segment':x[1][1],'delta_sec':x[2]} for x in plans],ensure_ascii=False,indent=2))
 if not a.apply:return
 ts=datetime.now(timezone.utc).isoformat()
 with c:
  for ck,gk,delta in plans:
   source=info[ck];target=info[gk];candidate_anchors=[x[2] for x in source['cand']];candidate_times=sorted({x[1] for x in source['cand']});candidate_labels=sorted({str(x['source_label']) for x in source['items']});source_starts=[float(x['payload'].get('support_start_sec',x['payload'].get('start_sec',x['source_time_sec']))) for x in source['items']];source_ends=[float(x['payload'].get('support_end_sec',x['payload'].get('end_sec',x['source_time_sec']))) for x in source['items']]
   for x in target['items']:
    q=x['payload'];q['evidence_anchors']=uniq([*q.get('evidence_anchors',[]),*candidate_anchors],lambda z:(str(z.get('source')),str(z.get('family') or z.get('label')),round(float(z.get('time_sec',0)),3)));q['candidate_times']=sorted({*map(float,q.get('candidate_times',[])),*candidate_times});q['candidate_labels']=sorted({*map(str,q.get('candidate_labels',[])),*candidate_labels});q['review_sources']=sorted({*map(str,q.get('review_sources',[])),'candidate','gt'});q['support_start_sec']=min([float(q.get('support_start_sec',q.get('start_sec',x['source_time_sec']))),*source_starts]);q['support_end_sec']=max([float(q.get('support_end_sec',q.get('end_sec',x['source_time_sec']))),*source_ends]);q.setdefault('absorbed_candidate_segments',[]);q['absorbed_candidate_segments']=sorted({*q['absorbed_candidate_segments'],ck[1]});c.execute('update events set payload_json=? where id=?',(json.dumps(q,ensure_ascii=False),x['id']))
   for x in source['items']:
    current=c.execute('select * from reviews where event_id=?',(x['id'],)).fetchone();c.execute('insert into review_history(event_id,revision,state_json,created_at) values(?,?,?,?)',(x['id'],current['revision'],json.dumps(dict(current)),ts));q=x['payload'];q['absorbed_into_gt_segment_id']=gk[1];q['absorbed_delta_sec']=delta;c.execute('update events set payload_json=? where id=?',(json.dumps(q,ensure_ascii=False),x['id']));c.execute("update reviews set status='deleted',note=?,reviewer='system_gt_merge_v34',revision=revision+1,updated_at=? where event_id=? and status='unreviewed'",(f'模型候选已并入 GT 任务 {gk[1]}，无需重复审核',ts,x['id']))
 print(json.dumps({'applied':len(plans),'events_absorbed':sum(len(info[x[0]]['items']) for x in plans)},ensure_ascii=False))
 c.close()
if __name__=='__main__':main()
