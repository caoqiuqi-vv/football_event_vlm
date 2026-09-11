"""Fixed full-window overlap protocol: window precision, event recall, FP windows/hour."""
import numpy as np
LABELS=['shot','save','set_piece']

class EventCurves:
    def __init__(self,records,manifest,tolerance=2.):
        self.records=records;self.classes=[]
        videos=sorted(set(r['video_id'] for r in records))
        for c,label in enumerate(LABELS):
            offsets={};all_gt=[];hours=0.
            for vid in videos:
                if manifest['videos'][vid]['label_mask'][c]<=0:continue
                offsets[vid]=len(all_gt);all_gt.extend(manifest['gt'][vid][label]);hours+=manifest['videos'][vid]['duration']/3600
            hits=[];valid=[]
            for i,r in enumerate(records):
                vid=r['video_id']
                if r['label_mask'][c]<=0 or vid not in offsets:continue
                gt=np.asarray(manifest['gt'][vid][label]);lo=np.searchsorted(gt,r['start_sec']-tolerance,'left');hi=np.searchsorted(gt,r['end_sec']+tolerance,'right')
                hits.append(np.arange(lo,hi,dtype=np.int64)+offsets[vid]);valid.append(i)
            self.classes.append({'valid':np.asarray(valid,np.int64),'hits':hits,'gt_count':len(all_gt),'hours':hours})

    def curve(self,prob,c):
        meta=self.classes[c];scores=prob[meta['valid'],c];order=np.argsort(-scores,kind='stable');first=np.full(meta['gt_count'],len(order),np.int64);tp=[]
        for rank,j in enumerate(order):
            hit=meta['hits'][j];tp.append(bool(len(hit)))
            if len(hit):first[hit]=np.minimum(first[hit],rank)
        count=np.arange(1,len(order)+1);tps=np.cumsum(tp,dtype=np.int64);fps=count-tps
        hit_counts=np.bincount(first[first<len(order)],minlength=len(order));recall=np.cumsum(hit_counts)/max(meta['gt_count'],1)
        return {'scores':scores[order],'count':count,'tp_windows':tps,'fp_windows':fps,'precision':tps/np.maximum(count,1),'recall':recall,'gt_count':meta['gt_count'],'hours':meta['hours']}

    def tune(self,prob,floors):
        thresholds=[]
        for c in range(3):
            curve=self.curve(prob,c);scores=curve['scores']
            if not len(scores):thresholds.append(1.);continue
            ends=np.r_[scores[:-1]!=scores[1:],True];eligible=np.flatnonzero(ends & (curve['recall']>=floors[c]))
            if not len(eligible):eligible=np.flatnonzero(ends & (curve['recall']>=curve['recall'].max()))
            chosen=eligible[np.argmax(curve['precision'][eligible])];thresholds.append(float(scores[chosen]))
        return thresholds,self.evaluate(prob,thresholds)

    def evaluate(self,prob,thresholds):
        results={}
        for c,label in enumerate(LABELS):
            curve=self.curve(prob,c);n=int((curve['scores']>=thresholds[c]).sum());i=n-1
            results[label]={'threshold':float(thresholds[c]),'precision':float(curve['precision'][i]) if n else 0.,'recall':float(curve['recall'][i]) if n else 0.,'false_positive_windows':int(curve['fp_windows'][i]) if n else 0,'predicted_windows':n,'fp_windows_per_hour':float(curve['fp_windows'][i]/max(curve['hours'],1e-12)) if n else 0.,'gt_events':curve['gt_count'],'evaluated_video_hours':curve['hours']}
        results['macro_precision']=float(np.mean([results[k]['precision'] for k in LABELS]));results['macro_recall']=float(np.mean([results[k]['recall'] for k in LABELS]));return results
