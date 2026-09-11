#!/usr/bin/env python
import sys,json,hashlib,copy
from pathlib import Path
from dataclasses import asdict
import cv2,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import train_football_events as f
from football_localization_full import atomic_json,digest

def main():
 out=ROOT/'outputs/football_localization_stage2/720p_optional_ball_roi_from_stage1e2_20260909'
 cfg0=json.loads((out/'event_config.json').read_text());cfg=f.to_config(cfg0);f.configure_label_schema(cfg)
 records={};events={}
 for split in ['train','val']:records[split],events[split]=f.load_long_video_records(cfg,split)
 vids=sorted(events['val'],key=lambda x:hashlib.sha256(x[1].encode()).hexdigest());calib=set(vids[:7])
 metadata={};corrections=[]
 for rs in records.values():
  for r in rs:
   if r.video_id in metadata:continue
   c=cv2.VideoCapture(r.video_path,cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);nf=int(c.get(cv2.CAP_PROP_FRAME_COUNT));fps=c.get(cv2.CAP_PROP_FPS);w=c.get(cv2.CAP_PROP_FRAME_WIDTH);h=c.get(cv2.CAP_PROP_FRAME_HEIGHT)
   assert h>=720 and w>=1280 and fps>0
   for last in range(nf-1,max(nf-17,0),-1):
    c.set(cv2.CAP_PROP_POS_FRAMES,last);ok,img=c.read()
    if ok and int(round(c.get(cv2.CAP_PROP_POS_FRAMES)))-1==last:break
   else:raise RuntimeError('unverified video tail '+r.video_id)
   c.release()
   if last+1!=nf:corrections.append({'video_id':r.video_id,'container_frames':nf,'verified_frames':last+1})
   metadata[r.video_id]={'fps':fps,'frames':last+1,'source_size':[h,w]}
 chunks={'train':[],'calibration':[],'development':[]};gt={};video_meta={}
 for split in ['train','val']:
  ev=events[split]
  ds=f.FootballLongVideoDataset(records[split],ev,num_frames=16,image_size=(720,1280),clip_duration=10.,event_margin=1.,temporal_jitter_sec=0.,is_train=False,hflip_prob=0.,normalize_on_cpu=False)
  rs=records[split] if split=='train' else f.build_online_validation_records(records[split],ev,clip_duration=10.,stride_sec=5.)
  availability={k:np.max([r.label_mask for r in records[split] if (r.source,r.video_id)==k],axis=0).tolist() for k in ev}
  for key,ee in ev.items():
   vid=key[1];gt[vid]={label:sorted({float(e.anchor_time) for e in ee if not e.is_ignored and e.labels[i]>.5}) for i,label in enumerate(f.LABELS)}
   template=next(r for r in records[split] if (r.source,r.video_id)==key);video_meta[vid]={'duration':template.video_duration,'label_mask':availability[key],'annotation_path':template.annotation_path,'annotation_sha256':digest(template.annotation_path),'source':key[0]}
  for r in rs:
   ee=ev[(r.source,r.video_id)];group='train' if split=='train' else ('calibration' if (r.source,r.video_id) in calib else 'development')
   if split=='train':start,end=ds._sample_window(r,ee)
   else:start,end=r.base_clip_start,r.base_clip_end
   labels=f.labels_for_window(ee,start,end);mask=f.label_mask_for_sample(r.label_mask if split=='train' else tuple(availability[(r.source,r.video_id)]),labels)
   mask=f.rejected_set_piece_label_mask(ee,start,end,labels,mask,0.)
   v=metadata[r.video_id];frames=f.segment_frame_indices(v['frames'],v['fps'],16,False,start_sec=start,end_sec=end)
   key=hashlib.sha256(json.dumps([group,r.video_id,start,end,frames]).encode()).hexdigest()[:24]
   chunks[group].append({'key':key,'video_id':r.video_id,'video_path':r.video_path,'start_sec':start,'end_sec':end,'frame_indices':frames,'frame_times':[i/v['fps'] for i in frames],'labels':list(labels),'label_mask':list(mask),'sample_id':r.sample_id})
 # Preserve source rows and supervision; duplicate identical windows are one cache read but may repeat in training.
 assert not set(r['video_id'] for r in chunks['train']) & set(r['video_id'] for r in chunks['calibration']+chunks['development'])
 s1=ROOT/'outputs/football_localization_stage1/720p_full_native_kl_from720best_20260907'
 config={'output_dir':str(out),'event_config':str(out/'event_config.json'),'source_checkpoint':str(s1/'source_720p_best.pt'),'stage1_checkpoint':str(s1/'best_adapt.pt'),'control_checkpoint':str(s1/'best_control.pt'),'gpus':[1,3,4,6],'epochs':6,'seeds':[42,43],'arms':['ordinary','frozen','stage1'],'hidden':128,'batch_size':64,'lr':0.0002,'weight_decay':0.01,'pos_weight':[1.,2.5,4.],'recall_floors':[.84,.8,.72],'clip_drop_probability':.15,'frame_drop_probability':.2,'candidate_drop_probability':.1,'corrupt_consistency_weight':.05,'residual_l2_weight':.01,'workers':2,'cache_frame_chunk':8,'num_frames':16,'image_size':[720,1280],'event_protocol':{'mode':'all_positive_windows_overlap','stride_sec':5.,'clip_sec':10.,'tolerance_sec':2.,'no_nms':True},'scope':'frozen-feature Stage2 controlled experiment; all eligible event training records with fixed temporal sampling and no RGB augmentation; calibration/development video split predeclared; development videos have project history'}
 manifest={'config':config,'splits':chunks,'gt':gt,'videos':video_meta,'media_metadata':metadata,'tail_corrections':corrections,'counts':{k:{'windows':len(v),'unique_windows':len(set(r['key'] for r in v)),'videos':len(set(r['video_id'] for r in v))} for k,v in chunks.items()},'source_sha256':{name:digest(config[name]) for name in ['source_checkpoint','stage1_checkpoint','control_checkpoint']}}
 atomic_json(out/'config.json',config);atomic_json(out/'manifest.json',manifest);print(json.dumps(manifest['counts'],indent=2));print('tail corrections',corrections)
if __name__=='__main__':main()
