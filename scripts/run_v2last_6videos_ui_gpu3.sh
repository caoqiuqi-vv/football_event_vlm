#!/usr/bin/env bash
set -euo pipefail
cd /home/new_users/qiuqi/code/dinov3-main

CKPT="outputs/football_events/vitl16_weekend_event_anchor_512x896_sym20_st_xbotgo0807_pos_ft_v2_anchor3_7_rawmissing/last.pt"
RUN_NAME="vitl16_weekend_st_xbotgo0807_pos_ft_v2_last_6videos_window_overlap_tol2_frame"
RUN_DIR="outputs/football_eval_runs/${RUN_NAME}"
LOG="${RUN_DIR}/run_ui_eval.log"
VIDEOS="2027564888428580866,2027572095412195330,2027572406738604033,2042125172076392450,2042520971893485569,2042525152494694401"
THRESHOLDS="shot=0.30000001192092896,save=0.44999998807907104,set_piece=0.4000000059604645"
WHISTLE_DIR="/home/new_users/qiuqi/code/det_and_track/outputs/whistle_detection"
mkdir -p "$RUN_DIR"
exec > >(tee -a "$LOG") 2>&1

echo "[$(date '+%F %T')] start v2-last 6-video UI eval on GPU3"
echo "checkpoint=$CKPT"
echo "run_dir=$RUN_DIR"

if [[ ! -f "$CKPT" ]]; then
  echo "[$(date '+%F %T')] ERROR missing checkpoint: $CKPT"
  exit 2
fi

run_dense() {
  local batch="$1"
  echo "[$(date '+%F %T')] dense eval batch_size=$batch"
  python scripts/evaluate_football_model.py \
    --checkpoint "$CKPT" \
    --mode dense \
    --video-ids "$VIDEOS" \
    --gt-dir /home/new_users/qiuqi/code/football_events_human_repair \
    --video-root xbotgo_0608=/mnt/data_16t/football/raw_video_720P \
    --output-root outputs/football_eval_runs \
    --run-name "$RUN_NAME" \
    --clip-sec 10 \
    --stride-sec 5 \
    --thresholds checkpoint \
    --match-tolerance-sec 2 \
    --prediction-postprocess window_overlap \
    --save-frame-event-logits \
    --frame-event-topk 8 \
    --batch-size "$batch" \
    --num-workers 1 \
    --device cuda:3 \
    --gpu-ids 3 \
    --force
}

if ! run_dense 6; then
  echo "[$(date '+%F %T')] dense batch_size=6 failed; retry batch_size=4"
  run_dense 4
fi

echo "[$(date '+%F %T')] build multilabel review segments"
python scripts/evaluate_multilabel_review_segments.py \
  --run-dir "$RUN_DIR" \
  --videos "$VIDEOS" \
  --thresholds "$THRESHOLDS" \
  --tolerance-sec 2 \
  --max-review-segment-sec 30 \
  --exclude 2027572406738604033:set_piece \
  --output-dir "$RUN_DIR/multilabel_review_segments_recall_preserving_tol2"

echo "[$(date '+%F %T')] analyze whistle UI"
python scripts/analyze_whistle_setpiece_ui.py \
  --run-dir "$RUN_DIR" \
  --review-segments "$RUN_DIR/multilabel_review_segments_recall_preserving_tol2/review_segments.csv" \
  --whistle-dir "$WHISTLE_DIR" \
  --videos "$VIDEOS" \
  --exclude 2027572406738604033:set_piece \
  --tolerance-sec 2 \
  --output-dir "$RUN_DIR/whistle_setpiece_ui_analysis_tol2"

echo "[$(date '+%F %T')] compute capped10/whistle threshold audit"
python - <<'PY_AUDIT'
import csv,json,ast
from pathlib import Path
from collections import Counter
RUN=Path('outputs/football_eval_runs/vitl16_weekend_st_xbotgo0807_pos_ft_v2_last_6videos_window_overlap_tol2_frame')
GT_RUN=RUN
SEG=RUN/'multilabel_review_segments_recall_preserving_tol2/review_segments.csv'
WH=RUN/'whistle_setpiece_ui_analysis_tol2/whistle_candidates_selected.csv'
SUMMARY=json.loads((RUN/'multilabel_review_segments_recall_preserving_tol2/summary.json').read_text())
VIDEOS=SUMMARY['video_ids']; LABELS=['shot','save','set_piece']; TOL=2.0; EXCLUDED={('2027572406738604033','set_piece')}
def read_csv(p):
    with p.open(newline='') as f: return list(csv.DictReader(f))
def parse_list(s):
    try:
        v=ast.literal_eval(s); return v if isinstance(v,list) else []
    except Exception: return []
def clamp(x,lo,hi): return max(lo,min(hi,x))
def covers(t,a,b): return a-TOL<=t<=b+TOL
win_by_vid={vid:{int(r['index']):r for r in read_csv(RUN/vid/'window_predictions.csv')} for vid in VIDEOS}
segments=[]
for r in read_csv(SEG):
    vid=r['video_id']; labs=parse_list(r['labels']); src=parse_list(r['source_window_indices']); s=float(r['start_sec']); e=float(r['end_sec'])
    best=None; label_best={}
    for idx in src:
        w=win_by_vid.get(vid,{}).get(int(idx))
        if not w: continue
        wc=(float(w['start_sec'])+float(w['end_sec']))/2
        for lab in labs:
            sc=float(w.get(f'prob_{lab}',-1e9))
            if best is None or sc>best[0]: best=(sc,wc,lab)
            if lab not in label_best or sc>label_best[lab][0]: label_best[lab]=(sc,wc)
    segments.append({'video_id':vid,'start':s,'end':e,'labels':labs,'global_peak':best[1] if best else (s+e)/2,'label_peak':{k:v[1] for k,v in label_best.items()}})
gts=[]
for vid in VIDEOS:
    for r in read_csv(RUN/vid/'gt_events.csv'):
        lab=r['label']
        if lab in LABELS and (vid,lab) not in EXCLUDED:
            gts.append({'video_id':vid,'label':lab,'time':float(r['time_sec']),'id':r.get('event_id') or f'{vid}:{lab}:{r["time_sec"]}'})
wh_rows=[]
if WH.exists():
    for r in read_csv(WH):
        wh_rows.append({'video_id':r['video_id'],'start':float(r['start_sec']),'end':float(r['end_sec']),'score':float(r['score']),'action':r['ui_action']})
def seg_interval(seg,mode,cap,lab):
    s,e=seg['start'],seg['end']
    if mode=='full' or e-s<=cap: return s,e
    if mode=='start': return s,s+cap
    c=seg['global_peak'] if mode=='global' else seg['label_peak'].get(lab,seg['global_peak'])
    a=clamp(c-cap/2,s,e-cap); return a,a+cap
def eval_recall(mode,cap,whistle_thr=None):
    per={}
    kept_wh=[w for w in wh_rows if whistle_thr is not None and w['score']>=whistle_thr]
    for lab in LABELS:
        m=0; n=0
        for gt in [g for g in gts if g['label']==lab]:
            n+=1; ok=False
            for seg in segments:
                if seg['video_id']!=gt['video_id']: continue
                a,b=seg_interval(seg,mode,cap,lab)
                if covers(gt['time'],a,b): ok=True; break
            if not ok and lab=='set_piece' and whistle_thr is not None:
                for w in kept_wh:
                    if w['video_id']==gt['video_id'] and covers(gt['time'],w['start'],w['end']): ok=True; break
            if ok: m+=1
        per[lab]={'matched':m,'gt':n,'recall':m/n if n else 0.0}
    mic={'matched':sum(v['matched'] for v in per.values()),'gt':sum(v['gt'] for v in per.values())}; mic['recall']=mic['matched']/mic['gt'] if mic['gt'] else 0.0
    return {'per_class':per,'micro':mic}
durs=[seg['end']-seg['start'] for seg in segments]
workload={'full_minutes':sum(durs)/60,'cap10_minutes':sum(min(d,10) for d in durs)/60,'cap15_minutes':sum(min(d,15) for d in durs)/60,'segments':len(segments)}
# threshold sweep for whistle: cost uses new_review_segment only, consistent with analyze_whistle_setpiece_ui.
wh_sweep=[]
base_set=eval_recall('full',None,None)['per_class']['set_piece']
for thr in [0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90]:
    kept=[w for w in wh_rows if w['score']>=thr]
    new=[w for w in kept if w['action']=='new_review_segment']
    rec=eval_recall('full',None,thr)['per_class']['set_piece']
    wh_sweep.append({'thr':thr,'kept':len(kept),'new_segments':len(new),'new_minutes':sum(w['end']-w['start'] for w in new)/60,'set_piece_recall':rec['recall'],'set_piece_matched':rec['matched'],'set_piece_gt':rec['gt']})
report={'run_dir':str(RUN),'gt_counts':dict(Counter(g['label'] for g in gts)),'workload':workload,'recall':{'full':eval_recall('full',None,None),'cap10_start':eval_recall('start',10,None),'cap10_global_peak':eval_recall('global',10,None),'cap10_label_peak':eval_recall('label',10,None),'cap15_global_peak':eval_recall('global',15,None),'full_whistle_thr045':eval_recall('full',None,0.45),'cap10_global_whistle_thr065':eval_recall('global',10,0.65),'cap10_label_whistle_thr065':eval_recall('label',10,0.65)},'whistle_threshold_sweep':wh_sweep}
out=RUN/'ui_capped10_recall_and_whistle_threshold_audit.json'
out.write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))
PY_AUDIT

echo "[$(date '+%F %T')] complete v2-last 6-video UI eval"
