#!/usr/bin/env python3
"""Validate and export only fully adjudicated, non-stale football event labels."""
from __future__ import annotations
import argparse,json,sqlite3,sys
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
FINAL={"confirmed","keep_both","deleted"}
def load_decisions(path):
 with sqlite3.connect(f"file:{path.resolve()}?mode=ro",uri=True) as c:
  c.row_factory=sqlite3.Row;rows=c.execute("SELECT * FROM final_decisions").fetchall()
 out={}
 for r in rows:
  d=dict(r);d["selected_events"]=json.loads(d.pop("selected_events_json") or "[]");out[d["case_id"]]=d
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument("--cases",type=Path,required=True);p.add_argument("--db",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--video-id",action="append");p.add_argument("--allow-incomplete-video",action="store_true");a=p.parse_args()
 data=json.loads(a.cases.read_text());ds=load_decisions(a.db);requested=set(a.video_id or data["videos"]);errors=[];warnings=[];events=defaultdict(list);dispositions={};case_audit=[]
 for source in (a.cases,a.db):
  if source.stat().st_mode&0o222:errors.append(f"mutable source rejected: {source}; create an immutable QC snapshot first")
 unknown=requested-set(data["videos"])
 if unknown:errors.append(f"unknown videos: {sorted(unknown)}")
 for vid in sorted(requested&set(data["videos"])):
  if not data["videos"][vid].get("first_pass_complete") and not a.allow_incomplete_video:errors.append(f"{vid}: first pass is incomplete")
 visible_cases=[x for x in data["cases"] if x["video_id"] in requested]
 auto_cases=[x for x in data.get("auto_resolved_cases",[]) if x["video_id"] in requested]
 cases=visible_cases+auto_cases
 for case in cases:
  auto=case.get("auto_resolution")=="gt_human_agree_one_to_one"
  d={"status":"confirmed","selected_events":case["recommended_events"],"reviewer":"auto:gt_human_agree","revision":0,"updated_at":data.get("created_at"),"note":"","provisional":not case.get("video_first_pass_complete",False),"source_hash":case["source_hash"]} if auto else ds.get(case["id"])
  if not d:errors.append(f"{case['id']}: decision missing");continue
  if d["source_hash"]!=case["source_hash"]:errors.append(f"{case['id']}: source hash mismatch")
  if d["status"] not in FINAL:errors.append(f"{case['id']}: status={d['status']}")
  if d.get("provisional") and not a.allow_incomplete_video:errors.append(f"{case['id']}: provisional decision")
  selected=d["selected_events"] if d["status"] in FINAL else []
  selected_ids={x.get("source_id") for x in selected}
  for gt in case["original_gt"]:
   mapped=[e for e in selected if gt["id"] in e.get("lineage_gt_ids",[])]
   dispositions[gt["id"]]={"video_id":gt["video_id"],"original":gt,"disposition":"deleted" if not mapped else "retained_or_modified","final_events":mapped,"case_id":case["id"]}
  for e in selected:
   normalized={**e,"source_id":e.get("source_id") or e.get("id"),"case_id":case["id"]}
   if not normalized["source_id"]:errors.append(f"{case['id']}: selected event missing source_id/id")
   events[case["video_id"]].append(normalized)
  case_audit.append({"case_id":case["id"],"video_id":case["video_id"],"source_hash":case["source_hash"],"status":d["status"],"reviewer":d["reviewer"],"revision":d["revision"],"updated_at":d["updated_at"],"selected_count":len(selected),"note":d["note"]})
 # Every original GT in the selected complete videos must have an explicit disposition.
 source_gt={g["id"]:g for c in cases for g in c["original_gt"]}
 missing=set(source_gt)-set(dispositions)
 if missing:errors.append(f"original GT without disposition: {len(missing)}")
 # A same-class <= tolerance pair is legal only after an explicit keep_both decision.
 tol=float(data["source"]["duplicate_window_sec"])
 case_status={x["case_id"]:x["status"] for x in case_audit}
 for vid,rows in events.items():
  rows.sort(key=lambda x:(x["time_sec"],x["semantic_label"],x["source_id"]))
  for i,x in enumerate(rows):
   for y in rows[i+1:]:
    if y["time_sec"]-x["time_sec"]>tol:break
    if x["semantic_label"]==y["semantic_label"]:
     x_lineage=set(x.get("lineage_gt_ids") or [])
     y_lineage=set(y.get("lineage_gt_ids") or [])
     distinct_gt_instances=bool(x_lineage and y_lineage and x_lineage.isdisjoint(y_lineage))
     explicitly_kept=x["case_id"]==y["case_id"] and case_status.get(x["case_id"])=="keep_both"
     if not (distinct_gt_instances or explicitly_kept):
      errors.append(f"{vid}: unresolved nearby duplicate {x['semantic_label']} at {x['time_sec']}/{y['time_sec']}")
 summary={"requested_videos":len(requested),"visible_cases":len(visible_cases),"auto_resolved_cases":len(auto_cases),"cases":len(cases),"events":sum(map(len,events.values())),"gt_dispositions":len(dispositions),"deleted_gt":sum(x["disposition"]=="deleted" for x in dispositions.values()),"errors":len(errors),"warnings":len(warnings)}
 payload={"schema_version":"football_final_labels_v1","created_at":datetime.now(timezone.utc).isoformat(),"source_cases":str(a.cases.resolve()),"source_db":str(a.db.resolve()),"summary":summary,"errors":errors,"warnings":warnings,"videos":{vid:{"events":events[vid],"source":data["videos"][vid]} for vid in sorted(requested&set(data["videos"]))},"gt_dispositions":dispositions,"case_audit":case_audit}
 if errors:
  print(json.dumps(summary,ensure_ascii=False,indent=2));print("EXPORT BLOCKED",file=sys.stderr)
  for x in errors[:30]:print("-",x,file=sys.stderr)
  raise SystemExit(2)
 if a.output.exists():raise FileExistsError(f"refusing to overwrite immutable export: {a.output}; choose a new versioned path")
 a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix(a.output.suffix+".tmp");tmp.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n");tmp.replace(a.output);print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
