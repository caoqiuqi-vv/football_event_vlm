"""Full native-frame localization data with explicit coverage and resumable order."""
from __future__ import annotations
import bisect
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def atomic_json(path, value):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)); tmp.replace(path)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def ids(path):
    return [s.strip() for s in Path(path).read_text().splitlines() if s.strip() and not s.startswith('#')]


def goal_arrays(path, fps, floor):
    if not path.is_file():return np.empty(0,np.int64),np.empty((0,4),np.float32),np.empty(0,np.float32),{}
    d=torch.load(path,weights_only=True,map_location='cpu')
    classes=d['classes'].numpy();conf=d['confidences'].float().numpy(); boxes=d['boxes'].float().numpy()
    good=(classes==2)&(conf>=floor)&np.isfinite(boxes).all(1)
    positions=np.flatnonzero(good)
    frame_positions=np.searchsorted(d['frame_offsets'].numpy()[1:],positions,side='right')
    frame_ids=np.rint(d['frame_ids'].numpy()[frame_positions]/float(d['fps'])*fps).astype(np.int64)
    size=d['image_size'];boxes=np.clip(boxes[good]/np.array([size['width'],size['height']]*2,np.float32),0,1)
    valid=(boxes[:,2]>boxes[:,0])&(boxes[:,3]>boxes[:,1])
    return frame_ids[valid],boxes[valid],conf[good][valid],{'raw_detections':len(classes),'qualified_boxes':int(valid.sum()),'sha256':digest(path)}


def prepare_full(cfg):
    out=Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True)
    sets={s:ids(cfg[s+'_ids']) for s in ('train','val')};test=set(ids(cfg['test_ids']))
    assert not set(sets['train'])&set(sets['val'])
    assert not (set(sets['train'])|set(sets['val']))&test
    manifest={'schema':'full-native-localization-v1','config':cfg,'splits':{},'excluded':[],
              'scope':'all qualifying native annotated frames; no temporal spacing or per-video cap; missing objects unknown'}
    fingerprints={}
    for split in ('val','train'):
        descriptors=[]
        for vid in sets[split]:
            video=Path(cfg['video_root'])/(vid+'.mp4'); ball=Path(cfg['ball_index'])/(vid+'.npz')
            if vid in cfg['excluded_videos'] or not video.is_file():
                manifest['excluded'].append([split,vid,'declared_bad_media_or_missing']);continue
            fingerprint=digest(ball) if ball.is_file() else None
            if fingerprint and fingerprint in fingerprints:
                manifest['excluded'].append([split,vid,'identical_ball_index_alias',fingerprints[fingerprint]]);continue
            if fingerprint:fingerprints[fingerprint]=[split,vid]
            cap=cv2.VideoCapture(str(video));fps=float(cap.get(cv2.CAP_PROP_FPS));nf=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));cap.release()
            if fps<=0 or nf<=0 or min(w/1280,h/720)<1 or abs(w/h-1280/720)>.02:
                raise RuntimeError(f'invalid source metadata {vid}: {fps,nf,w,h}')
            bf=np.empty(0,np.int64);bb=np.empty((0,4),np.float32);bc=np.empty(0,np.float32);raw_ball=0
            if ball.is_file():
                with np.load(ball) as z:d={k:z[k] for k in z.files}
                raw_ball=len(d['timestamp_sec']);boxes=d['bbox_xyxy_norm'].astype(np.float32)
                good=(d['source_code']==1)&((d['flags']&1)>0)&(d['confidence']>=cfg['ball_confidence'])
                good &= np.isfinite(boxes).all(1)&(boxes>=0).all(1)&(boxes<=1).all(1)&(boxes[:,2]>boxes[:,0])&(boxes[:,3]>boxes[:,1])
                bf=np.rint(d['timestamp_sec'][good]*fps).astype(np.int64);bb=boxes[good];bc=d['confidence'][good].astype(np.float32)
                valid=(bf>=0)&(bf<nf);bf,bb,bc=bf[valid],bb[valid],bc[valid]
                # Multiple detector timestamps mapping to one RGB frame are one example.
                order=np.lexsort((-bc,bf));bf,bb,bc=bf[order],bb[order],bc[order]
                _,unique=np.unique(bf,return_index=True);bf,bb,bc=bf[unique],bb[unique],bc[unique]
            gf,gb,gc,ga=goal_arrays(Path(cfg['goal_index'])/(vid+'.pt'),fps,cfg['goal_confidence'])
            sf,sb,sc,sa=goal_arrays(Path(cfg['goal_sparse_index'])/(vid+'.pt'),fps,cfg['goal_confidence'])
            # Sparse re-detection takes precedence for the same source frame;
            # all other full-video goal frames remain included.
            keep=~np.isin(gf,np.unique(sf));gf=np.concatenate([gf[keep],sf]);gb=np.concatenate([gb[keep],sb]);gc=np.concatenate([gc[keep],sc])
            valid=(gf>=0)&(gf<nf);gf,gb,gc=gf[valid],gb[valid],gc[valid]
            order=np.argsort(gf,kind='stable');gf,gb,gc=gf[order],gb[order],gc[order]
            frames=np.union1d(bf,gf).astype(np.int64)
            if not len(frames):manifest['excluded'].append([split,vid,'no_qualified_labels']);continue
            weights=np.zeros((len(frames),2),np.float32);ball_boxes=np.zeros((len(frames),4),np.float32)
            bi=np.searchsorted(frames,bf);weights[bi,0]=bc;ball_boxes[bi]=bb
            gi=np.searchsorted(frames,gf);counts=np.bincount(gi,minlength=len(frames));offsets=np.r_[0,np.cumsum(counts)].astype(np.int64)
            if counts.max(initial=0)>cfg['max_goal_boxes']:raise RuntimeError(f'{vid}: too many goals; refusing silent truncation')
            sums=np.bincount(gi,weights=gc,minlength=len(frames));weights[:,1]=sums/np.maximum(counts,1)
            root=out/'data'/split/vid;root.mkdir(parents=True,exist_ok=True)
            arrays={'frames':frames,'weights':weights,'ball_boxes':ball_boxes,'goal_offsets':offsets,'goal_boxes':gb.astype(np.float32)}
            hashes={}
            for name,array in arrays.items():
                p=root/(name+'.npy');np.save(p,array);hashes[name]=digest(p)
            entry={'video_id':vid,'video_path':str(video),'fps':fps,'source_frames':nf,'source_size':[h,w],
                   'array_dir':str(root),'frames':len(frames),'ball_frames':int((weights[:,0]>0).sum()),'goal_frames':int((weights[:,1]>0).sum()),
                   'raw_ball_rows':raw_ball,'ball_source_sha256':fingerprint,'full_goal_source':ga,'sparse_goal_source':sa,'array_sha256':hashes}
            descriptors.append(entry);print(f'INDEX {split} {vid} frames={len(frames)} ball={entry["ball_frames"]} goal={entry["goal_frames"]}',flush=True)
        manifest['splits'][split]=descriptors
    manifest['counts']={s:{'videos':len(v),'unique_annotated_rgb_frames':sum(d['frames'] for d in v),
                          'ball_frames':sum(d['ball_frames'] for d in v),'goal_frames':sum(d['goal_frames'] for d in v)} for s,v in manifest['splits'].items()}
    for counts in manifest['counts'].values():assert counts['ball_frames'] and counts['goal_frames']
    atomic_json(out/'manifest.json',manifest);return manifest


class FullFrames(Dataset):
    def __init__(self, descriptors, max_goals=16):
        self.descriptors=descriptors;self.ends=np.cumsum([d['frames'] for d in descriptors]);self.starts=np.r_[0,self.ends[:-1]]
        self.max_goals=max_goals;self.arrays=OrderedDict();self.captures=OrderedDict()

    def __len__(self):return int(self.ends[-1])

    def arrays_for(self, vid):
        if vid in self.arrays:self.arrays.move_to_end(vid);return self.arrays[vid]
        p=Path(self.descriptors[vid]['array_dir']);d={k:np.load(p/(k+'.npy'),mmap_mode='r') for k in ('frames','weights','ball_boxes','goal_offsets','goal_boxes')}
        self.arrays[vid]=d
        if len(self.arrays)>8:self.arrays.popitem(last=False)
        return d

    def __getitem__(self, index):
        if index<0:return {'frame':torch.zeros(3,720,1280,dtype=torch.uint8),'boxes':torch.zeros(2,self.max_goals,4),'weights':torch.zeros(2),'index':-1,'video_index':-1}
        cv2.setNumThreads(0);vi=int(np.searchsorted(self.ends,index,side='right'));local=int(index-self.starts[vi]);d=self.arrays_for(vi);desc=self.descriptors[vi]
        cap=self.captures.pop(vi,None)
        if cap is None:cap=cv2.VideoCapture(desc['video_path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
        self.captures[vi]=cap
        if len(self.captures)>4:_,old=self.captures.popitem(last=False);old.release()
        wanted=int(d['frames'][local]);position=int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))
        if 0<=wanted-position<=128:
            for _ in range(wanted-position):
                if not cap.grab():raise RuntimeError(f'decode advance failed {desc["video_id"]} frame={wanted}')
        else:cap.set(cv2.CAP_PROP_POS_FRAMES,wanted)
        ok,frame=cap.read()
        if not ok:raise RuntimeError(f'decode failed {desc["video_id"]} frame={wanted}; no substitution allowed')
        actual=int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))-1
        if actual!=wanted:raise RuntimeError(f'decode alignment {desc["video_id"]}: wanted={wanted},actual={actual}')
        h,w=frame.shape[:2]
        if h<720 or w<1280 or abs(w/h-1280/720)>.02:raise RuntimeError(f'actual source below 720P or invalid aspect: {desc["video_id"]} frame={wanted} size={h,w}')
        if (h,w)!=(720,1280):frame=cv2.resize(frame,(1280,720),interpolation=cv2.INTER_AREA)
        frame=torch.from_numpy(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB).copy()).permute(2,0,1)
        boxes=torch.zeros(2,self.max_goals,4);boxes[0,0]=torch.from_numpy(d['ball_boxes'][local].copy())
        lo,hi=int(d['goal_offsets'][local]),int(d['goal_offsets'][local+1]);boxes[1,:hi-lo]=torch.from_numpy(d['goal_boxes'][lo:hi].copy())
        return {'frame':frame,'boxes':boxes,'weights':torch.from_numpy(d['weights'][local].copy()),'index':int(index),'video_index':vi}


class CoverageSampler(Sampler):
    """Shuffle sequential blocks, cover each real row exactly once across ranks.

    Padding is -1 (masked), never a duplicated training example. Deterministic
    offsets permit exact mid-epoch recovery without repeating optimizer steps.
    """
    def __init__(self,lengths,batch,rank,world,seed,epoch,block=128,start_step=0,shuffle=True):
        pieces=[];offset=0
        for n in lengths:
            pieces.extend((offset+i,min(block,n-i)) for i in range(0,n,block));offset+=n
        order=np.arange(len(pieces))
        if shuffle:np.random.default_rng(seed+epoch).shuffle(order)
        counts=[sum(pieces[j][1] for j in order[r::world]) for r in range(world)]
        steps=(max(counts)+batch-1)//batch
        own=[np.arange(pieces[j][0],pieces[j][0]+pieces[j][1],dtype=np.int64) for j in order[rank::world]]
        values=np.concatenate(own) if own else np.empty(0,np.int64)
        self.all_values=np.pad(values,(0,steps*batch-len(values)),constant_values=-1)
        self.values=self.all_values[start_step*batch:];self.total_steps=steps;self.real_count=counts[rank]
    def __iter__(self):return iter(self.values.tolist())
    def __len__(self):return len(self.values)


def global_localization_loss(logits, targets, weights):
    """Equal per-class global means, independent of rank and missing classes."""
    terms=-(targets*logits.float().log_softmax(1)).sum(1)
    denom=weights.sum(0).detach().clone();world=1
    if torch.distributed.is_initialized():
        world=torch.distributed.get_world_size();torch.distributed.all_reduce(denom)
    active=denom>0
    return world*((terms*weights).sum(0)/denom.clamp_min(1e-12)*active).sum()/active.sum().clamp_min(1)


def prepare_expert_reference(cfg):
    """All available RF-DETR ball frames; automatic development references, not GT."""
    out=Path(cfg['output_dir']);main=json.loads((out/'manifest.json').read_text());descriptors=[]
    for entry in main['splits']['val']:
        vid=entry['video_id'];path=Path(cfg['goal_sparse_index'])/(vid+'.pt')
        if not path.exists():continue
        d=torch.load(path,weights_only=True,map_location='cpu');c=d['classes'].numpy();q=d['confidences'].float().numpy();boxes=d['boxes'].float().numpy()
        good=(c==1)&(q>=cfg['ball_confidence'])&np.isfinite(boxes).all(1);positions=np.flatnonzero(good)
        fi=np.searchsorted(d['frame_offsets'].numpy()[1:],positions,side='right')
        frames=np.rint(d['frame_ids'].numpy()[fi]/float(d['fps'])*entry['fps']).astype(np.int64)
        size=d['image_size'];b=np.clip(boxes[good]/np.array([size['width'],size['height']]*2,np.float32),0,1);conf=q[good]
        valid=(frames>=0)&(frames<entry['source_frames'])&(b[:,2]>b[:,0])&(b[:,3]>b[:,1]);frames,b,conf=frames[valid],b[valid],conf[valid]
        order=np.lexsort((-conf,frames));frames,b,conf=frames[order],b[order],conf[order]
        _,keep=np.unique(frames,return_index=True);frames,b,conf=frames[keep],b[keep],conf[keep]
        if not len(frames):continue
        root=out/'data'/'expert_reference'/vid;root.mkdir(parents=True,exist_ok=True);weights=np.zeros((len(frames),2),np.float32);weights[:,0]=conf
        arrays={'frames':frames,'weights':weights,'ball_boxes':b.astype(np.float32),'goal_offsets':np.zeros(len(frames)+1,np.int64),'goal_boxes':np.empty((0,4),np.float32)}
        for name,array in arrays.items():np.save(root/(name+'.npy'),array)
        descriptors.append({**entry,'array_dir':str(root),'frames':len(frames),'ball_frames':len(frames),'goal_frames':0,'reference_source_sha256':digest(path)})
    result={'descriptors':descriptors,'frames':sum(d['frames'] for d in descriptors),'videos':len(descriptors),'scope':'all available sparse RF-DETR positive ball references; same development videos; not human GT; highest confidence box per frame'}
    if not result['frames']:raise RuntimeError('no independent-teacher references available')
    atomic_json(out/'expert_reference_manifest.json',result);return result
