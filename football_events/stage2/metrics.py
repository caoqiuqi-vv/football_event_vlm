"""Window AP and paired-video uncertainty under the existing event protocol.

EventCurves remains the single definition of window precision/event recall.
These metrics are not event-spotting mAP and development videos are not a blind test.
"""
import numpy as np
from sklearn.metrics import average_precision_score
from football_stage2_metrics import EventCurves, LABELS

def window_ap(curves, prob):
    per = []
    for c, meta in enumerate(curves.classes):
        y = np.array([bool(len(x)) for x in meta['hits']])
        per.append(float(average_precision_score(y, prob[meta['valid'], c])) if y.any() else 0.0)
    return {'per_class': dict(zip(LABELS, per)), 'macro': float(np.mean(per))}

def paired_video_bootstrap(records, curves, left, right, repeats=500):
    videos = sorted({r['video_id'] for r in records})
    vi = {v: i for i, v in enumerate(videos)}
    rng = np.random.default_rng(20260909)
    deltas = []
    lp = 1 / (1 + np.exp(-np.clip(left, -50, 50)))
    rp = 1 / (1 + np.exp(-np.clip(right, -50, 50)))
    metas = []
    for c, meta in enumerate(curves.classes):
        y = np.array([bool(len(h)) for h in meta['hits']])
        ids = np.array([vi[records[j]['video_id']] for j in meta['valid']])
        metas.append((y, ids, meta['valid']))
    for _ in range(repeats):
        counts = np.bincount(rng.integers(0, len(videos), len(videos)), minlength=len(videos))
        scores = []
        for c, (y, ids, valid) in enumerate(metas):
            weight = counts[ids]
            if not weight[y].sum():
                break
            scores.append(average_precision_score(y, lp[valid, c], sample_weight=weight) - average_precision_score(y, rp[valid, c], sample_weight=weight))
        if len(scores) == 3:
            deltas.append(float(np.mean(scores) * 100))
    return {
        'metric': 'macro window AP delta pp',
        'videos': len(videos),
        'bootstrap_repeats': repeats,
        'valid_replicates': len(deltas),
        'percentile_95CI_pp': np.quantile(deltas, [0.025, 0.975]).tolist() if deltas else None,
        'interpretation': 'paired video resampling on historically used development videos; not independent test evidence',
    }
