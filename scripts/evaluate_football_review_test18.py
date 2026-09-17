#!/usr/bin/env python3
"""Evaluate frozen test18 with validation thresholds, export E2 frame curves."""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time as wall_time
from dataclasses import asdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import train_football_events as t
from football_review_data import parent, clean_final_events


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--annotations',required=True)
    p.add_argument('--video-root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--config');p.add_argument('--preflight-only',action='store_true')
    a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    cfg=t.load_config(a.config,[]) if a.config else t.to_config(checkpoint['config'])
    cfg['device']='cuda:0';cfg['gpu_ids']=[0]
    cfg.model['init_checkpoint']=str(Path(a.checkpoint).resolve());cfg.model['init_checkpoint_strict']=True
    cfg.eval['candidate_time_mode']='window_center'
    cfg.eval['external_audit']=t.to_config({'enabled':False,'online_mode':{'nms_radius_sec':5.,'tolerance_sec':3.,'window_stride_sec':5.,'capped_clip_sec':10.,'score_fusion':'clip'}})
    assert t.resolve_model_init_checkpoint(cfg)[0]==str(Path(a.checkpoint).resolve())
    t.configure_runtime_threads(cfg);t.configure_label_schema(cfg);t.resolve_runtime_topology(cfg,torch.device(cfg.device))
    annotations=Path(a.annotations);manifest=json.loads((annotations/'manifest.json').read_text())
    source_files=manifest.get('files',[])
    if len(source_files)!=18:raise ValueError('Expected exactly 18 test videos')
    records,events_by_video,snapshot,hashes=[],{},[],{}
    cleaning=[]
    training_ids=set(Path(cfg.data.long_video.split_files['train'][0]).read_text().split())
    for item in source_files:
        vid=item['video_id'];path=annotations/item.get('file',vid+'.json');raw=path.read_bytes()
        digest=hashlib.sha256(raw).hexdigest()
        if item.get('sha256') and digest!=item['sha256']:raise ValueError(f'Test labels changed: {vid}')
        hashes[vid]=digest;payload=json.loads(raw)
        if vid in training_ids:raise ValueError(f'Train/test leakage: {vid}')
        video=Path(a.video_root)/(vid+'.mp4')
        if not video.is_file():raise FileNotFoundError(video)
        duration=float(payload['video_source']['duration_sec']);events=[]
        cleaned,decisions=clean_final_events(payload['events'])
        cleaning.extend({'video_id':vid,**d} for d in decisions)
        for index,event in enumerate(cleaned):
            raw_label=event['semantic_label'];lab=parent(raw_label)
            # goal/own_goal never become additional shot instances.
            if lab not in t.LABELS:continue
            time=float(event['time_sec']);values=tuple(float(name==lab) for name in t.LABELS)
            if not 0<=time<=duration:raise ValueError((vid,time,duration))
            events.append(t.FootballEvent(source='xbotgo_0608',video_id=vid,
                event_id=str(event.get('id') or event.get('source_id') or f'{vid}:{index}'),
                event_type='',raw_label=raw_label,start_time=time,end_time=time,anchor_time=time,labels=values))
        events_by_video[('xbotgo_0608',vid)]=events
        records.append(t.LongVideoRecord(source='xbotgo_0608',split='val',video_id=vid,sample_id='template',
            video_path=str(video),annotation_path=str(path),anchor_time=5.,base_clip_start=0.,base_clip_end=10.,
            video_duration=duration,is_negative=False,labels=(0.,)*3,label_mask=(1.,)*3))
        snapshot.append({'video_id':vid,'events':[asdict(e) for e in events],'cleaned_source_events':cleaned,'duration_sec':duration})
    records=t.build_online_validation_records(records,events_by_video,clip_duration=10.,stride_sec=5.)
    # External evaluation uses frozen thresholds and does not tune on test18.
    from dataclasses import replace
    records=[replace(r,sample_id=r.sample_id.replace('online_val_','online_eval_',1)) for r in records]
    (root/'test18_cleaning_decisions.json').write_text(json.dumps(cleaning,ensure_ascii=False,indent=2)+'\n')
    if a.preflight_only:
        from collections import Counter
        counts=Counter(t.LABELS[i] for events in events_by_video.values() for e in events for i,v in enumerate(e.labels) if v>0)
        duplicates=[]
        for key,events in events_by_video.items():
            counter=Counter((tuple(e.labels),round(e.anchor_time,4)) for e in events)
            duplicates.extend((key,value,n) for value,n in counter.items() if n>1)
        report={'videos':len(events_by_video),'windows':len(records),'events_by_class':dict(counts),
                'exact_same_class_time_duplicates':duplicates,'cleaning_merges':len(cleaning),'annotation_hashes':hashes,'train_test_overlap':False}
        (root/'preflight.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2));return
    frame_enabled=float(cfg.train.frame_det_loss_weight)>0
    dataset=t.FootballLongVideoDataset(records,events_by_video,num_frames=16,image_size=(720,1280),
        clip_duration=10.,event_margin=1.,temporal_jitter_sec=0.,is_train=False,hflip_prob=0.,
        normalize_on_cpu=False,decode_strategy=str(cfg.data.get('video_decode_strategy','single_seek')),video_reader_cache_size=2,
        frame_supervision=frame_enabled)
    loader=t.make_loader(dataset,cfg,is_train=False,batch_size=int(cfg.eval.batch_size),distributed=False)
    model=t.make_model(cfg,use_cached_features=False,device=torch.device(cfg.device))
    print(f'test18_dense videos={len(snapshot)} windows={len(records)} checkpoint_epoch={checkpoint.get("epoch")} decoder={dataset.decode_strategy}',flush=True)
    def progress_loader():
        started=wall_time.monotonic();done=0
        for batch_index,batch in enumerate(loader,1):
            yield batch
            done+=len(batch['meta'])
            if batch_index%20==0 or done==len(records):
                progress={'completed_windows':done,'total_windows':len(records),'elapsed_sec':wall_time.monotonic()-started}
                temporary=root/'progress.json.tmp';temporary.write_text(json.dumps(progress)+'\n');temporary.replace(root/'progress.json')
                print('test18_progress '+json.dumps(progress),flush=True)
    frame_curves,frame_times=[],[]
    def capture(module,args,kwargs,outputs):
        if not isinstance(outputs,dict) or 'frame_event_logits' not in outputs:raise ValueError('Missing frame curves')
        frame_curves.append(outputs['frame_event_logits'].float().cpu().numpy())
        frame_times.append(kwargs['global_frame_times'].float().cpu().numpy())
    handle=model.register_forward_hook(capture,with_kwargs=True) if frame_enabled else None
    metrics=t.evaluate(model,progress_loader(),cfg,torch.device(cfg.device),fixed_thresholds=checkpoint['thresholds'],
        online_cache_path=root/'test18_predictions.npz')
    if handle:handle.remove()
    metrics['protocol']={'annotations':str(annotations.resolve()),'annotation_hashes':hashes,
        'thresholds':checkpoint['thresholds'],'threshold_source':'checkpoint_internal_validation',
        'candidate_time_mode':'window_center','tolerance_sec':3.,'nms_radius_sec':5.,
        'scope':'test18_full_videos; fixed_v4_core_labels; no_test_threshold_tuning'}
    if frame_enabled:
        curves=np.concatenate(frame_curves);times=np.concatenate(frame_times)
        assert len(curves)==len(records)
        peaks=curves.argmax(axis=1);peak_times=np.take_along_axis(times,peaks,axis=1)
        np.savez_compressed(root/'frame_detection_curves.npz',frame_event_logits=curves,frame_times=times,
            video_ids=np.asarray([r.video_id for r in records]),clip_starts=np.asarray([r.base_clip_start for r in records]))
        cache=np.load(root/'test18_predictions.npz')
        supplemental=t.online_event_metrics(cache['probs'],peak_times,[asdict(r) for r in records],
            np.asarray([checkpoint['thresholds'][lab] for lab in t.LABELS]),masks=np.ones_like(cache['probs']),
            nms_radius_sec=5.,tolerance_sec=3.,capped_clip_sec=10.,stride_sec=5.)
        metrics['frame_timestamp_ablation']={'score':'clip','thresholds':'same_frozen_clip_thresholds',
            'timestamp':'frame_peak','metrics':supplemental}
    (root/'test18_annotation_snapshot.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n')
    (root/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(metrics['online_event'],ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
