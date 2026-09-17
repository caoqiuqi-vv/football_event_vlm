#!/usr/bin/env python3
"""Export dense per-video scores and separate fixed-test results from LOOV diagnosis."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import train_football_events as t
from analyze_dense_pr_ceiling import load_run_scores,evaluate_data,combine_metrics
from analyze_video_adaptive_dense_thresholds import analyze

LABELS=['shot','save','set_piece']
FLOORS={'shot':.90,'save':.85,'set_piece':.85}
ZH={'shot':'射门','save':'扑救','set_piece':'定位球'}

def write_json(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def write_csv(path,rows,fields):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evaluation',required=True);p.add_argument('--checkpoint',required=True);a=p.parse_args()
    root=Path(a.evaluation);output=root/'dense_analysis';output.mkdir(exist_ok=False)
    metrics=json.loads((root/'metrics.json').read_text());truth=json.loads((root/'test18_annotation_snapshot.json').read_text())
    cache=np.load(root/'test18_predictions.npz');metas=json.loads((root/'test18_predictions.meta.json').read_text())
    assert cache['labels'].tolist()==LABELS
    assert len(metas)==len(cache['probs']) and np.all(cache['masks']>.5)
    ids=[x['video_id'] for x in truth];assert len(ids)==len(set(ids))==18
    thresholds=metrics['protocol']['thresholds'];threshold_array=np.array([thresholds[x] for x in LABELS])
    config=t.to_config({'task':{'label_schema':'set_piece'}});t.configure_label_schema(config)
    per_event=[]
    for video in truth:
        vid=video['video_id'];rows=np.where(cache['video_ids']==vid)[0];assert len(rows)>0
        d=output/vid;d.mkdir()
        windows=[{'window_index':int(j),'start_sec':float(cache['clip_starts'][i]),'end_sec':float(cache['clip_ends'][i]),**{'prob_'+label:float(cache['probs'][i,k]) for k,label in enumerate(LABELS)}} for j,i in enumerate(rows)]
        write_csv(d/'window_predictions.csv',windows,['window_index','start_sec','end_sec',*['prob_'+x for x in LABELS]])
        events=[{'label':label,'time_sec':e['anchor_time'],'event_id':e['event_id'],'raw_label':e['raw_label']} for e in video['events'] for k,label in enumerate(LABELS) if e['labels'][k]>0]
        write_csv(d/'gt_events.csv',events,['label','time_sec','event_id','raw_label'])
        write_json(d/'summary.json',{'video_id':vid,'checkpoint':a.checkpoint,'duration_sec':video['duration_sec'],'num_windows':len(rows),'thresholds':thresholds,'frame_event':{'enabled':False}})
        event=t.online_event_metrics(cache['probs'][rows],cache['candidate_times'][rows],[metas[i] for i in rows],threshold_array,masks=cache['masks'][rows],nms_radius_sec=5.,tolerance_sec=3.,capped_clip_sec=10.,stride_sec=5.)
        for label in LABELS:
            v=event['per_class'][label];per_event.append({'video_id':vid,'label':label,**{k:v[k] for k in ['precision','recall','f1','tp','fp','fn','support','threshold']}})
    for label in LABELS:
        rows=[x for x in per_event if x['label']==label];aggregate=metrics['online_event']['per_class'][label]
        for key in ['tp','fp','fn','support']:assert sum(x[key] for x in rows)==aggregate[key],(label,key)
    write_csv(output/'per_video_event_metrics.csv',per_event,list(per_event[0]))
    write_json(output/'run_config.json',{'video_ids':ids,'checkpoint':a.checkpoint,'threshold_source':'checkpoint_internal_validation','annotations':metrics['protocol']})
    args=SimpleNamespace(run_dir=str(output),labels=','.join(LABELS),video_id_file='',match_tolerance_sec=3.,recall_floors='shot=0.90,save=0.85,set_piece=0.85',alpha_min=-.5,alpha_max=1.5,alpha_step=.1)
    print('Starting CPU LOOV threshold diagnosis from dense score cache',flush=True)
    loov=analyze(args);write_json(output/'video_adaptive_thresholds_loov_tol3.json',loov)
    data,loaded=load_run_scores(output,labels=LABELS,branch='fused',tolerance_sec=0.,exclusions=set());assert loaded==ids
    exact={label:combine_metrics(evaluate_data(x,thresholds[label]) for x in data if x.label==label) for label in LABELS}
    for label in LABELS:
        assert exact[label]['num_gt']==metrics['online_event']['per_class'][label]['support'],('GT projection mismatch',label)
        assert loov['per_class'][label]['fixed_global']['num_gt']==exact[label]['num_gt']
    comparisons={}
    for label in LABELS:
        r=loov['per_class'][label];fixed=r['fixed_global'];adaptive=r['adaptive_median_logit_loov']['aggregate'];oracle=r['per_video_oracle']['aggregate']
        precision_gain=adaptive['precision']-fixed['precision'];recall_gain=adaptive['recall']-fixed['recall']
        # Predeclared diagnostic criterion: preserve recall (within1pp or reach floor)
        # while improving precision by at least2pp; compare only the same window protocol.
        calibration_helpful=precision_gain>=.02 and (adaptive['recall']>=FLOORS[label] or recall_gain>=-.01)
        if calibration_helpful:advice='LOOV 显示阈值校准有收益；下一步在训练/验证视频做校准，再冻结后到独立测试集验证。'
        elif fixed['recall']>=FLOORS[label] and metrics['online_event']['per_class'][label]['recall']<FLOORS[label]:advice='窗口覆盖已达召回目标，事件级仍不足；优先检查时间定位和 NMS，LOOV 不能替代定位改进。'
        elif adaptive['recall']<FLOORS[label]:advice='LOOV 仍未达到召回目标；需要检查低分漏检及视频间偏移，不宜仅靠逐视频阈值。'
        else:advice='当前 LOOV 未显示明确的精度收益，先看 E1 后续轮次及 E2 的同口径结果。'
        comparisons[label]={'fixed_window_tol3':fixed,'fixed_window_tol0':exact[label],'loov_window_tol3':adaptive,'oracle_window_tol3_upper_bound':oracle,'precision_change':precision_gain,'recall_change':recall_gain,'calibration_helpful':calibration_helpful,'recommendation':advice}
    report={'checkpoint':a.checkpoint,'videos':18,'windows':len(metas),'fixed_event_tol3':metrics['online_event'],'comparisons':comparisons,'loov_note':'LOOV在其他17个test18视频拟合alpha和阈值；只对留出视频计分。属于test18内部交叉验证诊断，不能替代独立固定阈值测试。','oracle_note':'逐视频oracle读取该视频GT，仅表示校准上限。','helpful_rule':'同窗口±3秒协议下，P提升至少2个百分点，且R达到目标或下降不超过1个百分点；这是诊断标准。'}
    write_json(output/'analysis_report.json',report)
    folds=[{'label':label,**{k:f[k] for k in ['held_out_video_id','alpha','video_median_logit','adaptive_threshold','train_recall_floor_feasible']}} for label in LABELS for f in loov['per_class'][label]['adaptive_median_logit_loov']['folds']]
    write_csv(output/'loov_thresholds.csv',folds,list(folds[0]))
    lines=[f'# E1 test18 dense 评测与 LOOV 诊断\n\n模型：`{a.checkpoint}`。18个完整视频，{len(metas)}个窗口，10秒窗口、5秒步长。使用冻结的 v4 修正标签；固定测试阈值来自模型内部验证集。\n','## 固定阈值事件级结果（±3秒，NMS5秒，严格一对一）\n','|类别|P|R|TP|FP|FN|\n|---|---:|---:|---:|---:|---:|']
    for label in LABELS:
        r=metrics['online_event']['per_class'][label];lines.append(f"|{ZH[label]}|{r['precision']:.2%}|{r['recall']:.2%}|{r['tp']}|{r['fp']}|{r['fn']}|")
    lines+=['\n## 候选窗口覆盖与 LOOV（多个窗口可覆盖同一GT，不使用NMS）\n','|类别|固定P ±3秒|固定R ±3秒|固定R 无扩展|LOOV P ±3秒|LOOV R ±3秒|\n|---|---:|---:|---:|---:|---:|']
    for label in LABELS:
        r=comparisons[label];f=r['fixed_window_tol3'];l=r['loov_window_tol3'];lines.append(f"|{ZH[label]}|{f['precision']:.2%}|{f['recall']:.2%}|{r['fixed_window_tol0']['recall']:.2%}|{l['precision']:.2%}|{l['recall']:.2%}|")
    lines+=['\n## 是否需要 LOOV\n',report['helpful_rule']+'\n']
    for label in LABELS:lines.append(f"- {ZH[label]}：{comparisons[label]['recommendation']}")
    lines+=['\n'+report['loov_note']+' '+report['oracle_note'],'\n窗口覆盖和事件级口径分别报告，不能将LOOV窗口召回与固定事件召回直接作改善对比。逐视频事件指标见 per_video_event_metrics.csv；完整窗口分数和GT投影在逐视频目录。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(comparisons,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
