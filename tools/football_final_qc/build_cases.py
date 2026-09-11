#!/usr/bin/env python3
'''Build non-overlapping final-adjudication cases from first-pass reviews.

This is intentionally not temporal NMS. Original GT instances are immutable source
nodes; only multiple outputs mapped to the same GT may be safely collapsed.
'''
from __future__ import annotations
import argparse, hashlib, json, sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SET_TYPES={"corner","free_kick","kickoff","penalty"}
RAW_TO_SEMANTIC={"角球":"corner","任意球":"free_kick","中圈开球":"kickoff","点球":"penalty"}
FINAL={"accepted","modified","deleted"}

def semantic(label:str, secondary:list[str]|None=None, raw_label:str="")->str:
    if label!="set_piece": return label
    for item in secondary or []:
        if item in SET_TYPES:return item
    return RAW_TO_SEMANTIC.get(raw_label,"set_piece")

def parent(label:str)->str:return "set_piece" if label in SET_TYPES else label

def stable_sample(key:str,rate:float)->bool:
    return int(hashlib.sha1(key.encode()).hexdigest()[:12],16)/(16**12-1)<rate

def content_hash(value:Any)->str:
    payload=json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"))
    return hashlib.sha256(payload.encode()).hexdigest()

def equivalent_events(gt:list[dict[str,Any]],human:list[dict[str,Any]],tolerance:float)->bool:
    """One-to-one semantic agreement; close timestamps are the same event."""
    if len(gt)!=len(human):return False
    by_label_gt:dict[str,list[float]]=defaultdict(list);by_label_human:dict[str,list[float]]=defaultdict(list)
    for row in gt:by_label_gt[str(row["semantic_label"])].append(float(row["time_sec"]))
    for row in human:by_label_human[str(row["semantic_label"])].append(float(row["time_sec"]))
    if set(by_label_gt)!=set(by_label_human):return False
    for label,times in by_label_gt.items():
        remaining=sorted(by_label_human[label])
        if len(times)!=len(remaining):return False
        for time_sec in sorted(times):
            options=[(abs(value-time_sec),index) for index,value in enumerate(remaining) if abs(value-time_sec)<=tolerance]
            if not options:return False
            _,index=min(options);remaining.pop(index)
    return True

def filter_final_cases(cases:list[dict[str,Any]],tolerance:float,excluded:set[str])->tuple[list[dict[str,Any]],list[dict[str,Any]],Counter[str]]:
    """Keep only genuine GT/human disagreements after removing ignored labels."""
    stats:Counter[str]=Counter();kept=[];auto_resolved=[]
    for case in cases:
        before=sum(len(case[key]) for key in ("original_gt","first_pass_events","recommended_events"))
        case["original_gt"]=[x for x in case["original_gt"] if x["semantic_label"] not in excluded]
        case["first_pass_events"]=[x for x in case["first_pass_events"] if x["semantic_label"] not in excluded]
        case["recommended_events"]=[x for x in case["recommended_events"] if x["semantic_label"] not in excluded]
        allowed_ids={x["id"] for x in case["first_pass_events"]}
        case["duplicate_pairs"]=[pair for pair in case["duplicate_pairs"] if all(item in allowed_ids for item in pair)]
        stats["excluded_label_event_references"]+=before-sum(len(case[key]) for key in ("original_gt","first_pass_events","recommended_events"))
        if not case["original_gt"] and not case["recommended_events"]:
            stats["dropped_gt_empty_human_empty"]+=1;continue
        if equivalent_events(case["original_gt"],case["recommended_events"],tolerance):
            case["auto_resolution"]="gt_human_agree_one_to_one"
            auto_resolved.append(case);stats["dropped_gt_human_agree"]+=1;continue
        kept.append(case)
    stats["kept_disagreement_cases"]=len(kept)
    stats["preserved_auto_resolved_cases"]=len(auto_resolved)
    return kept,auto_resolved,stats

def segment_status(items:list[dict[str,Any]])->str:
    states=[x["status"] for x in items]
    if "unreviewed" in states:return "unreviewed"
    if "needs_confirmation" in states:return "needs_confirmation"
    if all(x=="deleted" for x in states):return "deleted"
    if any(x in {"modified","deleted"} for x in states):return "modified"
    return "accepted"

class UnionFind:
    def __init__(self,n:int):self.p=list(range(n))
    def find(self,x:int)->int:
        while self.p[x]!=x:self.p[x]=self.p[self.p[x]];x=self.p[x]
        return x
    def union(self,a:int,b:int)->None:
        a,b=self.find(a),self.find(b)
        if a!=b:self.p[b]=a

def load_gt(run_dir:Path,video_id:str)->list[dict[str,Any]]:
    rows=json.loads((run_dir/video_id/"gt_events.json").read_text());out=[]
    for i,row in enumerate(rows):
        raw=str(row.get("raw_label") or row.get("event_type") or "");label=str(row["label"]);t=float(row["time_sec"])
        out.append({"id":str(row.get("event_id") or f"{video_id}:gt:{i}"),"video_id":video_id,
                    "parent_label":label,"semantic_label":semantic(label,raw_label=raw),
                    "time_sec":t,"raw_label":raw,"source":"original_gt"})
    return out

def find_gt(gt:list[dict[str,Any]],label:str,time_sec:float)->dict[str,Any]|None:
    candidates=[x for x in gt if x["parent_label"]==parent(label) and abs(x["time_sec"]-time_sec)<=.02]
    return min(candidates,key=lambda x:abs(x["time_sec"]-time_sec)) if candidates else None

def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True);p.add_argument("--review-db",type=Path,required=True)
    p.add_argument("--gt-run-dir",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    # Final annotation QC uses a wider association window than formal model
    # evaluation. Five seconds absorbs window-centre / annotator timestamp
    # jitter without changing the official +/-3 s evaluation protocol.
    p.add_argument("--match-tolerance-sec",type=float,default=5.0);p.add_argument("--duplicate-window-sec",type=float,default=5.0)
    p.add_argument("--rejected-candidate-sample-rate",type=float,default=.15);a=p.parse_args()
    manifest=json.loads(a.manifest.read_text());videos={str(v["video_id"]):v for v in manifest["videos"]}
    gt_by_video={vid:load_gt(a.gt_run_dir,vid) for vid in videos};gt_index={x["id"]:x for rows in gt_by_video.values() for x in rows}
    con=sqlite3.connect(a.review_db);con.row_factory=sqlite3.Row
    sql=("SELECT e.id,e.video_id,e.source_label,e.source_time_sec,e.source_score,e.payload_json,"
         "r.status,r.corrected_label,r.corrected_time_sec,r.note,r.reviewer,r.secondary_labels_json,r.updated_at "
         "FROM events e JOIN reviews r ON r.event_id=e.id ORDER BY e.video_id,e.source_time_sec,e.source_label")
    dbrows=con.execute(sql).fetchall();con.close();grouped:dict[tuple[str,str],list[dict[str,Any]]]=defaultdict(list)
    for row in dbrows:
        d=dict(row);d["payload"]=json.loads(d.pop("payload_json"));d["secondary"]=json.loads(d.pop("secondary_labels_json") or "[]")
        grouped[(str(d["video_id"]),str(d["payload"].get("segment_id") or d["id"]))].append(d)
    manifest_segment_ids={vid:{str(e.get("segment_id") or e["id"]) for e in v["events"]} for vid,v in videos.items()}
    segment_states={(vid,sid):segment_status(items) for (vid,sid),items in grouped.items()}
    video_progress={}
    for vid in videos:
        expected=manifest_segment_ids[vid];counts=Counter(segment_states.get((vid,sid),"missing") for sid in expected)
        incomplete=sum(counts[x] for x in ("unreviewed","needs_confirmation","missing"))
        video_progress[vid]={"total_segments":len(expected),"accepted":counts["accepted"],"modified":counts["modified"],
            "deleted":counts["deleted"],"needs_confirmation":counts["needs_confirmation"],"unreviewed":counts["unreviewed"],
            "missing":counts["missing"],"first_pass_complete":incomplete==0}
    bases=[];gt_task_lookup={};event_evidence={}
    for (vid,sid),items in grouped.items():
        state=segment_states[(vid,sid)];lineages=[]
        for item in items:
            for t in item["payload"].get("matching_gt_times",[]):
                found=find_gt(gt_by_video[vid],str(item["source_label"]),float(t))
                if found and found["id"] not in lineages:lineages.append(found["id"])
        has_gt=bool(lineages)
        for gid in lineages:gt_task_lookup[gid]=(vid,sid,state)
        include=state!="unreviewed" and (has_gt or state=="needs_confirmation")
        if not has_gt and state in FINAL:
            kept=any(x["status"] in {"accepted","modified"} for x in items)
            include=kept or stable_sample(f"{vid}|{sid}",a.rejected_candidate_sample_rate)
        if not include:continue
        finals=[]
        for item in items:
            if item["status"] not in {"accepted","modified"}:continue
            payload=item["payload"];label=str(item["corrected_label"] or item["source_label"]);lineage=[]
            event_evidence[str(item["id"])]={
                "window_indices":{int(x) for x in payload.get("window_indices",[])},
                "support_start_sec":float(payload.get("support_start_sec",payload.get("start_sec",item["source_time_sec"]-5))),
                "support_end_sec":float(payload.get("support_end_sec",payload.get("end_sec",item["source_time_sec"]+5))),
                "score":float(item["source_score"]),
            }
            for t in payload.get("matching_gt_times",[]):
                found=find_gt(gt_by_video[vid],str(item["source_label"]),float(t))
                if found:lineage.append(found["id"])
            finals.append({"id":str(item["id"]),"video_id":vid,"segment_id":sid,"label":label,
                "semantic_label":semantic(label,item["secondary"]),"time_sec":float(item["corrected_time_sec"] if item["corrected_time_sec"] is not None else item["source_time_sec"]),
                "source_time_sec":float(item["source_time_sec"]),"score":float(item["source_score"]),"lineage_gt_ids":sorted(set(lineage)),
                "human_added":bool(payload.get("human_added")),"reviewer":str(item["reviewer"] or ""),"note":str(item["note"] or ""),"source":"first_pass"})
        starts=[float(x["payload"].get("support_start_sec",x["payload"].get("start_sec",x["source_time_sec"]-5))) for x in items]
        ends=[float(x["payload"].get("support_end_sec",x["payload"].get("end_sec",x["source_time_sec"]+5))) for x in items]
        bases.append({"video_id":vid,"segment_ids":[sid],"source_statuses":[state],"original_gt_ids":lineages,"first_pass_events":finals,
                      "start_sec":max(0,min(starts)),"end_sec":min(float(videos[vid]["duration_sec"]),max(ends)),"sampled_rejected_candidate":not has_gt and not finals})
    represented={gid for b in bases for e in b["first_pass_events"] for gid in e["lineage_gt_ids"]};candidates=[]
    for bi,b in enumerate(bases):
        for ei,e in enumerate(b["first_pass_events"]):
            if e["lineage_gt_ids"]:continue
            for gt in gt_by_video[e["video_id"]]:
                if gt["id"] in represented:continue
                task=gt_task_lookup.get(gt["id"])
                if task and task[2] in FINAL and gt["semantic_label"]==e["semantic_label"] and abs(gt["time_sec"]-e["time_sec"])<=a.match_tolerance_sec:
                    candidates.append((abs(gt["time_sec"]-e["time_sec"]),bi,ei,gt["id"]))
    used_events=set()
    for _,bi,ei,gid in sorted(candidates):
        if (bi,ei) in used_events or gid in represented:continue
        bases[bi]["first_pass_events"][ei]["inferred_gt_id"]=gid;represented.add(gid);used_events.add((bi,ei))
    explicit_gt_events=defaultdict(list)
    for b in bases:
        for e in b["first_pass_events"]:
            for gid in e["lineage_gt_ids"]:explicit_gt_events[gid].append(e)

    def same_dense_response(event:dict[str,Any],reference:dict[str,Any])->bool:
        """Strong evidence that two timestamps came from one dense response.

        This safely bridges a small gap beyond the generic QC tolerance while
        preserving genuinely dense shots: shared source windows, overlapping
        support and effectively identical class score are all required.
        """
        left=event_evidence.get(event["id"],{});right=event_evidence.get(reference["id"],{})
        shared=left.get("window_indices",set())&right.get("window_indices",set())
        overlap=min(left.get("support_end_sec",-1),right.get("support_end_sec",-1))-max(left.get("support_start_sec",0),right.get("support_start_sec",0))
        return len(shared)>=2 and overlap>0 and abs(left.get("score",-100)-right.get("score",100))<=1e-6

    # Second pass deliberately allows an additional nearby output to map onto
    # an already represented GT. This is the small-time-offset duplicate that
    # ordinary evaluation matching often hides and ordinary NMS handles unsafely.
    for bi,b in enumerate(bases):
        for ei,e in enumerate(b["first_pass_events"]):
            if e["lineage_gt_ids"] or e.get("inferred_gt_id"):continue
            nearby=[]
            for gt in gt_by_video[e["video_id"]]:
                task=gt_task_lookup.get(gt["id"])
                delta=abs(gt["time_sec"]-e["time_sec"])
                response_match=delta<=8.0 and any(same_dense_response(e,ref) for ref in explicit_gt_events.get(gt["id"],[]))
                if task and task[2] in FINAL and gt["semantic_label"]==e["semantic_label"] and (delta<=a.match_tolerance_sec or response_match):
                    nearby.append((delta,gt["id"],"shared_dense_response" if response_match and delta>a.match_tolerance_sec else "time_tolerance"))
            if nearby:
                _,e["inferred_gt_id"],e["inferred_gt_reason"]=min(nearby)
    uf=UnionFind(len(bases));nodes=[]
    # Outputs associated with one immutable GT belong to one adjudication case
    # even when their timestamps sit just outside the generic duplicate window.
    bases_by_gt=defaultdict(list)
    for bi,b in enumerate(bases):
        for e in b["first_pass_events"]:
            for gid in e["lineage_gt_ids"] or ([e["inferred_gt_id"]] if e.get("inferred_gt_id") else []):bases_by_gt[gid].append(bi)
    for values in bases_by_gt.values():
        for bi in values[1:]:uf.union(values[0],bi)
    for bi,b in enumerate(bases):
        for e in b["first_pass_events"]:nodes.append((bi,e["video_id"],e["semantic_label"],e["time_sec"]))
        for gid in b["original_gt_ids"]:
            g=gt_index[gid];nodes.append((bi,g["video_id"],g["semantic_label"],g["time_sec"]))
    by_label=defaultdict(list)
    for node in nodes:by_label[(node[1],node[2])].append(node)
    for values in by_label.values():
        values.sort(key=lambda x:x[3])
        for i,left in enumerate(values):
            for right in values[i+1:]:
                if right[3]-left[3]>a.duplicate_window_sec:break
                if left[0]!=right[0]:uf.union(left[0],right[0])
    components=defaultdict(list)
    for i,b in enumerate(bases):components[uf.find(i)].append(b)
    cases=[]
    for members in components.values():
        vid=members[0]["video_id"];segs=sorted({s for b in members for s in b["segment_ids"]});originals=[];seen=set()
        for gid in [g for b in members for g in b["original_gt_ids"]]:
            if gid not in seen:originals.append(gt_index[gid]);seen.add(gid)
        finals=[];seen=set()
        for e in [e for b in members for e in b["first_pass_events"]]:
            if e["id"] not in seen:finals.append(e);seen.add(e["id"])
        risks=set();final_by_gt=defaultdict(list)
        for e in finals:
            gids=e["lineage_gt_ids"] or ([e["inferred_gt_id"]] if e.get("inferred_gt_id") else [])
            for gid in gids:final_by_gt[gid].append(e)
        for gt in originals:
            mapped=final_by_gt.get(gt["id"],[])
            if not mapped:risks.add("gt_deleted")
            else:
                if not any(e["semantic_label"]==gt["semantic_label"] for e in mapped):risks.add("gt_label_changed")
                elif not any(e["semantic_label"]==gt["semantic_label"] and abs(e["time_sec"]-gt["time_sec"])<=a.match_tolerance_sec for e in mapped):risks.add("gt_time_changed")
                else:risks.add("gt_unchanged")
        for e in finals:
            if not e["lineage_gt_ids"] and not e.get("inferred_gt_id"):risks.add("new_event")
        duplicate_pairs=[]
        for i,x in enumerate(finals):
            for y in finals[i+1:]:
                if x["semantic_label"]==y["semantic_label"] and abs(x["time_sec"]-y["time_sec"])<=a.duplicate_window_sec and x["segment_id"]!=y["segment_id"]:
                    duplicate_pairs.append([x["id"],y["id"]]);risks.add("near_duplicate")
        if any(s=="needs_confirmation" for b in members for s in b["source_statuses"]):risks.add("source_pending")
        if any(b["sampled_rejected_candidate"] for b in members):risks.add("rejected_candidate_sample")
        drop=set()
        for gid,events in final_by_gt.items():
            if len(events)>1:
                ranked=sorted(events,key=lambda e:(gid not in e["lineage_gt_ids"],abs(e["time_sec"]-gt_index[gid]["time_sec"]),-e["score"]))
                drop.update(e["id"] for e in ranked[1:]);risks.add("same_gt_duplicate")
        # Adjacent source segments can expose the exact same dense response at
        # slightly different anchor times. Collapse those response-identical
        # events as well; unlike temporal NMS this requires shared source
        # windows and identical scores, and never collapses distinct GT ids.
        for i,x in enumerate(finals):
            for y in finals[i+1:]:
                if x["semantic_label"]!=y["semantic_label"] or not same_dense_response(x,y):continue
                xg=set(x["lineage_gt_ids"] or ([x["inferred_gt_id"]] if x.get("inferred_gt_id") else []));yg=set(y["lineage_gt_ids"] or ([y["inferred_gt_id"]] if y.get("inferred_gt_id") else []))
                if xg and yg and xg.isdisjoint(yg):continue
                ranked=sorted((x,y),key=lambda e:(not bool(e["lineage_gt_ids"]),bool(e.get("human_added")),e["id"]))
                drop.add(ranked[1]["id"]);risks.add("same_response_duplicate")
        recommended=[dict(e) for e in finals if e["id"] not in drop]
        # A duplicate group that is proven to reference one GT has one
        # canonical annotation. Keep all raw source evidence above for audit,
        # but publish the immutable GT timestamp and lineage in the terminal
        # recommendation. Do not snap ordinary single human corrections.
        recommended_by_id={e["id"]:e for e in recommended}
        for gid,events in final_by_gt.items():
            if len(events)<=1:continue
            retained=[recommended_by_id[e["id"]] for e in events if e["id"] in recommended_by_id]
            gt=gt_index[gid]
            if len(retained)!=1 or retained[0]["semantic_label"]!=gt["semantic_label"]:continue
            event=retained[0];source_time=float(event["time_sec"])
            if abs(source_time-float(gt["time_sec"]))<=.001:continue
            event["time_sec"]=float(gt["time_sec"]);event["lineage_gt_ids"]=[gid]
            event["canonicalization"]={"policy":"same_gt_duplicate_gt_time","gt_id":gid,"source_time_sec":source_time}
        priority=100 if risks&{"source_pending","gt_deleted","gt_label_changed","gt_time_changed","near_duplicate","same_gt_duplicate"} else 60 if "new_event" in risks else 10
        case_id=f"{vid}_final_{hashlib.sha1('|'.join(segs).encode()).hexdigest()[:14]}"
        case={"id":case_id,"video_id":vid,"segment_ids":segs,"priority":priority,"risk_types":sorted(risks),
          "start_sec":max(0,min(b["start_sec"] for b in members)-2),"end_sec":min(float(videos[vid]["duration_sec"]),max(b["end_sec"] for b in members)+2),
          "original_gt":sorted(originals,key=lambda x:(x["time_sec"],x["semantic_label"])),"first_pass_events":sorted(finals,key=lambda x:(x["time_sec"],x["semantic_label"])),
          "recommended_events":sorted(recommended,key=lambda x:(x["time_sec"],x["semantic_label"])),"duplicate_pairs":duplicate_pairs,
          "source_complete":not any(s in {"unreviewed","needs_confirmation"} for b in members for s in b["source_statuses"]),
          "video_first_pass_complete":video_progress[vid]["first_pass_complete"]}
        case["source_hash"]=content_hash({k:case[k] for k in ("video_id","segment_ids","risk_types","original_gt","first_pass_events","recommended_events")})
        cases.append(case)
    cases_before_filter=len(cases)
    cases,auto_resolved_cases,filter_stats=filter_final_cases(cases,a.match_tolerance_sec,{"throw_in"})
    for case in cases:
        case["risk_types"]=[risk for risk in case["risk_types"] if risk not in {"gt_unchanged","rejected_candidate_sample"}]
        if not case["risk_types"]:case["risk_types"]=["gt_human_disagreement"]
        case["priority"]=100 if set(case["risk_types"])&{"source_pending","gt_deleted","gt_label_changed","gt_time_changed","near_duplicate","same_gt_duplicate"} else 60
        case["source_hash"]=content_hash({k:case[k] for k in ("video_id","segment_ids","risk_types","original_gt","first_pass_events","recommended_events")})
    cases.sort(key=lambda x:(-x["priority"],x["video_id"],x["start_sec"],x["id"]));counts=Counter(r for c in cases for r in c["risk_types"])
    output={"schema_version":"football_final_qc_v1","created_at":datetime.now(timezone.utc).isoformat(),
      "source":{"manifest":str(a.manifest.resolve()),"review_db":str(a.review_db.resolve()),"gt_run_dir":str(a.gt_run_dir.resolve()),"match_tolerance_sec":a.match_tolerance_sec,"duplicate_window_sec":a.duplicate_window_sec,"rejected_candidate_sample_rate":a.rejected_candidate_sample_rate,"dedup_policy":"one-to-one provenance-aware association; never standard temporal NMS; never auto-collapse distinct GT lineage","prior_policy":{"qc_timestamp_jitter_sec":a.match_tolerance_sec,"same_semantic_event_within_jitter":"associate to the original GT and retain the GT timestamp","distinct_original_gt_instances":"always retain, even inside the jitter window","shot_save_cooccurrence":"allowed and retained as independent labels","dense_shot_or_save":"never hard-suppress by a football cooldown prior","restart_burst_beyond_jitter":"review hint only; never automatic deletion"},"formal_evaluation_tolerance_sec":3.0,"final_case_filter":{"gt_human_agreement":"one-to-one same semantic label within match tolerance","drop_both_empty":True,"excluded_semantic_labels":["throw_in"]}},
      "videos":{vid:{"video_path":v["video_path"],"duration_sec":v["duration_sec"],**video_progress[vid]} for vid,v in videos.items()},
      "summary":{"cases_before_filter":cases_before_filter,"cases":len(cases),"auto_resolved_cases":len(auto_resolved_cases),"filter":dict(filter_stats),"high_priority":sum(c["priority"]>=100 for c in cases),"source_complete_cases":sum(c["source_complete"] for c in cases),
        "first_pass_complete_videos":sum(v["first_pass_complete"] for v in video_progress.values()),"by_risk":dict(sorted(counts.items()))},"auto_resolved_cases":auto_resolved_cases,"cases":cases}
    a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix(a.output.suffix+".tmp");tmp.write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n");tmp.replace(a.output);print(json.dumps(output["summary"],ensure_ascii=False,indent=2))
if __name__=="__main__":main()
