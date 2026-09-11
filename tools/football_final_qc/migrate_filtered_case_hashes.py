#!/usr/bin/env python3
"""Migrate final-QC hashes after a semantics-preserving case filter."""
from __future__ import annotations
import argparse,json,sqlite3
from datetime import datetime,timezone
from pathlib import Path

def now():return datetime.now(timezone.utc).isoformat()
def main():
 p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,required=True);p.add_argument('--db',type=Path,required=True);a=p.parse_args()
 cases={x['id']:x for x in json.loads(a.cases.read_text())['cases']};changed=[]
 with sqlite3.connect(a.db) as c:
  c.row_factory=sqlite3.Row
  for cid,case in cases.items():
   row=c.execute('select * from final_decisions where case_id=?',(cid,)).fetchone()
   if not row or row['source_hash']==case['source_hash']:continue
   selected=json.loads(row['selected_events_json'] or '[]')
   allowed={str(x['id']) for x in case['first_pass_events']}|{str(x['id']) for x in case['recommended_events']}
   selected_source={str(x.get('source_id')) for x in selected if x.get('source_id') and not str(x.get('source_id')).startswith('manual_')}
   missing=selected_source-allowed
   if row['status'] not in {'unreviewed','stale'} and missing:
    raise RuntimeError(f'{cid}: reviewed decision references filtered events: {sorted(missing)}')
   state=json.dumps(dict(row),ensure_ascii=False,separators=(',',':'))
   c.execute('insert into final_history(case_id,revision,action,state_json,created_at) values(?,?,?,?,?)',(cid,row['revision'],'semantics_preserving_case_filter',state,now()))
   c.execute('update final_decisions set source_hash=?,revision=revision+1,updated_at=? where case_id=?',(case['source_hash'],now(),cid))
   changed.append({'case_id':cid,'status':row['status']})
  c.commit()
 print(json.dumps({'migrated':len(changed),'by_status':{s:sum(x['status']==s for x in changed) for s in sorted({x['status'] for x in changed})}},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
