#!/usr/bin/env python
"""Finalize only after full training, exact coverage and selected reference evaluation."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from football_localization_full import atomic_json,digest


def compare(rows, video_ids):
    result={};rng=np.random.default_rng(20260907)
    for c,name in enumerate(('ball','goal')):
        valid=rows['valid'][:,c]
        if not valid.any():result[name]={'count':0};continue
        hit=(rows['error'][:,:,c]<=16) if c==0 else rows['inside'][:,:,c]
        part=rows[valid];hit=hit[valid]
        a=hit[:,0].astype(float);b=hit[:,1].astype(float);groups=[];per_video={}
        for vid in np.unique(part['video']):
            select=part['video']==vid;n=int(select.sum());delta=float((a[select]-b[select]).sum())
            groups.append((n,delta));per_video[video_ids[int(vid)]]={'frames':n,'delta_pp':delta/n*100}
        groups=np.asarray(groups);indices=rng.integers(len(groups),size=(10000,len(groups)));draw=groups[indices]
        boot=draw[:,:,1].sum(1)/draw[:,:,0].sum(1)*100
        result[name]={'count':int(valid.sum()),'adapt':float(a.mean()),'control':float(b.mean()),'pooled_delta_pp':float((a-b).mean()*100),
            'cluster_video_bootstrap_95CI_pp':np.percentile(boot,[2.5,97.5]).tolist(),
            'equal_video_mean_delta_pp':float(np.mean(groups[:,1]/groups[:,0])*100),'videos':len(groups),
            'improved_videos':sum(r['delta_pp']>0 for r in per_video.values()),'per_video':per_video}
    return result


def finalize(out):
    torch.set_num_threads(1);out=Path(out);manifest=json.loads((out/'manifest.json').read_text());cfg=manifest['config'];done=json.loads((out/'TRAINING_COMPLETE.json').read_text())
    assert done['epochs']==cfg['epochs'] and done['coverage_pass']
    n=manifest['counts']['train']['unique_annotated_rgb_frames'];coverage=[]
    for e in range(1,cfg['epochs']+1):
        report=json.loads((out/f'coverage_epoch_{e:03d}.json').read_text());a=np.load(out/f'coverage_epoch_{e:03d}.npy',mmap_mode='r')
        assert len(a)==n and np.all(a==1) and report['missing']==0 and report['duplicates']==0
        assert report['manifest_sha256']==digest(out/'manifest.json');coverage.append(report)
    provenance=json.loads((out/'source_provenance.json').read_text());assert digest(provenance['snapshot'])==provenance['checkpoint_sha256']
    assert digest(cfg['warmup_checkpoint'])==provenance['warmup_snapshot_sha256']
    for split in manifest['splits'].values():
        for d in split:
            for name,sha in d['array_sha256'].items():assert digest(Path(d['array_dir'])/(name+'.npy'))==sha
    adapted=torch.load(out/'best_adapt.pt',weights_only=True,map_location='cpu');control=torch.load(out/'best_control.pt',weights_only=True,map_location='cpu')
    assert adapted['manifest_sha256']==control['manifest_sha256']==digest(out/'manifest.json')
    ar=np.load(out/f"val_epoch_{adapted['epoch']:03d}_predictions.npy");cr=np.load(out/f"val_epoch_{control['epoch']:03d}_predictions.npy")
    assert np.array_equal(ar['index'],cr['index']) and np.array_equal(ar['valid'],cr['valid']) and np.array_equal(ar['video'],cr['video'])
    rows=ar.copy()
    for key in ('peaks','error','inside'):rows[key][:,1]=cr[key][:,1]
    np.save(out/'selected_pair_predictions.npy',rows)
    ids=[d['video_id'] for d in manifest['splits']['val']];comparison=compare(rows,ids)
    expert_meta=json.loads((out/'expert_reference_manifest.json').read_text());expert_rows=np.load(out/'expert_selected_predictions.npy')
    assert len(expert_rows)==expert_meta['frames']
    for d in expert_meta['descriptors']:
        for name,sha in d['array_sha256'].items():assert digest(Path(d['array_dir'])/(name+'.npy'))==sha
    expert=compare(expert_rows,[d['video_id'] for d in expert_meta['descriptors']])
    stats=rows['stats'];preserved=float(stats[:,2].mean())>=cfg['min_patch_cosine'] and float(stats[:,3].mean())>=cfg['min_cls_cosine']
    directional=adapted['epoch']>0 and preserved and all(comparison[c]['cluster_video_bootstrap_95CI_pp'][0]>0 for c in ('ball','goal')) and expert['ball']['pooled_delta_pp']>=0
    summary={'execution_complete':True,'epochs':cfg['epochs'],'source_checkpoint':cfg['checkpoint'],'unique_train_frames':n,'total_committed_training_frame_visits':n*cfg['epochs'],
        'validation_frames':len(rows),'coverage':coverage,'adapt_selected_epoch':adapted['epoch'],'control_selected_epoch':control['epoch'],
        'comparison':comparison,'other_teacher_reference':expert,'feature_preservation_pass':preserved,
        'drift':{name:{'mean':float(stats[:,i].mean()),'p01':float(np.quantile(stats[:,i],.01)),'p99':float(np.quantile(stats[:,i],.99))} for i,name in enumerate(('patch_kl','cls_kl','patch_cosine','cls_cosine'))},
        'automatic_reference_gain_supported_for_both_classes':directional,'real_detection_accuracy_verified':False,
        'reason_real_accuracy_unverified':'No independent human-audited positive, absent and ambiguous-object labels were available in this run.',
        'stage2_started':False,'stage2_acceptance_pass':False,'source_snapshot_verified':True}
    atomic_json(out/'FINAL_SUMMARY.json',summary)
    lines=['# 全量 Stage 1 最终报告','',f"已完成 {cfg['epochs']} 轮全量微调：每轮 {n:,} 个唯一合格标注帧，合计 {n*cfg['epochs']:,} 次已提交训练访问。每轮覆盖审计均为零遗漏、零重复。完整验证 {len(rows):,} 帧。",'',
        '输入始终 720×1280，源 DINO 为用户指定的 720P fromlast best（不可变快照）。定位头共用先前预热权重；本次没有用采样阶段的适配 LoRA 替代原始初始化。', '',
        f"微调与冻结对照分别在完整验证集选点：epoch {adapted['epoch']} / {control['epoch']}。下面比较各自最佳模型，避免对照后期退化造成虚假收益。",'',
        '|自动标注定位指标|最佳冻结对照|最佳微调|差值|','|---|---:|---:|---:|']
    for c,label in [('ball','足球中心≤16px'),('goal','球门峰值落入任一已知框')]:
        r=comparison[c];lines.append(f"|{label}|{r['control']*100:.2f}%|{r['adapt']*100:.2f}%|{r['pooled_delta_pp']:+.2f}pp|")
    lines+=['','按视频簇重采样的配对区间（对应上表按帧合并的差值）：']
    for c in ('ball','goal'):
        r=comparison[c];lo,hi=r['cluster_video_bootstrap_95CI_pp'];lines.append(f"- {c}：95% 条件区间 [{lo:+.2f}, {hi:+.2f}]pp，改善视频 {r['improved_videos']}/{r['videos']}；等视频权重差值 {r['equal_video_mean_delta_pp']:+.2f}pp。")
    r=expert['ball'];lines+=['',f"另一足球检测器 RF-DETR 的全部 {r['count']:,} 个可用参考帧：冻结 {r['control']*100:.2f}%，微调 {r['adapt']*100:.2f}%，差值 {r['pooled_delta_pp']:+.2f}pp。它来自相同开发视频的稀疏事件样本，是跨教师复核，不是新的独立测试或人工真值。",'',
        f"选中模型 patch/CLS 余弦均值：{stats[:,2].mean():.6f}/{stats[:,3].mean():.6f}；KL 均值：{stats[:,0].mean():.6g}/{stats[:,1].mean():.6g}。预设均值保真门槛：{'通过' if preserved else '未通过'}。分位数见 FINAL_SUMMARY.json。",'',
        '**最终判断**：'+('两类定位在自动参考上的增量证据满足本轮保守统计检查。' if directional else '本轮未得到同时支持足球和球门稳定增益的充分自动参考证据。'),'',
        '**不能据此宣布真实检测精度或 Stage 1 最终验收通过。** 验证标签仍为自动检测结果，缺少人工核查的无目标、遮挡、备用球及目标身份真值；球门指标不是完整框检测 AP。没有独立比赛分组，视频簇之间也可能相关。区间没有覆盖检查点选择和多轮开发历史。', '',
        'KL 与特征相似度只衡量本输入域的输出漂移；本轮没有无 KL 对照，也没有评测原事件头或通用语义能力，因此不把保真或增益全部归因于 KL，不声称事件识别已提高。', '',
        'Stage 2 未启动。训练和现有数据上的评估已经完整结束；可靠定位验收仍需同源人工核查。原始数据的质量筛选、缺标和坏媒体排除详见 manifest.json，不把所有原视频帧或未标注目标都称作已监督样本。', '',
        '产物：best_adapt.pt、best_control.pt、各 epoch 权重、逐帧预测、逐轮 coverage 文件、FINAL_SUMMARY.json、源权重/代码校验和。定位检查点需要结合指定源模型加载，不是可直接交给事件评估脚本的完整事件模型。']
    (out/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
    atomic_json(out/'PIPELINE_COMPLETE.json',{'complete':True,'final_report':str(out/'FINAL_REPORT.md'),'stage2_started':False})
    return summary


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',required=True);a=ap.parse_args();s=finalize(a.output_dir);print(json.dumps({'execution_complete':s['execution_complete'],'real_detection_accuracy_verified':s['real_detection_accuracy_verified'],'stage2_started':False}))
