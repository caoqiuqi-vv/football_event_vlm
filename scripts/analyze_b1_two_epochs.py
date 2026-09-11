"""Recompute cached B1/E16 validation with identical masks and event policy."""
import json
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from football_online_evaluation import tune_online_event_thresholds


def main():
    base = ROOT / 'outputs/football_events/vitl16_online_grid_dual_e16_512_from_last6r8'
    run = ROOT / 'outputs/football_events/vitl16_b1_external_trajectory_camcomp_symgoal_zerogate_4gpu_from_e16v2_20260907'
    out = ROOT / 'outputs/football_b1_two_epoch_audit_20260907'
    out.mkdir(exist_ok=True)
    entries = [('E16', base, 1), ('B1_epoch1', run, 1), ('B1_epoch2', run, 2)]
    loaded = []
    for name, folder, epoch in entries:
        path = folder / f'val15_online_ema_epoch_{epoch:03d}.npz'
        with np.load(path) as z:
            d = {k: z[k] for k in z.files}
        metas = json.loads(path.with_suffix('.meta.json').read_text())
        keys = [(str(v), round(float(s), 4), round(float(e), 4)) for v, s, e in zip(d['video_ids'], d['clip_starts'], d['clip_ends'])]
        assert len(set(keys)) == len(keys)
        order = sorted(range(len(keys)), key=keys.__getitem__)
        n = len(keys)
        d = {k: v[order] if v.ndim and v.shape[0] == n else v for k, v in d.items()}
        loaded.append((name, d, [metas[i] for i in order], sorted(keys)))
    _, ref, metas, keys = loaded[0]
    for name, d, m, k in loaded:
        assert k == keys, name
        assert np.array_equal(d['labels'], ref['labels'])
        assert np.array_equal(d['targets'], ref['targets']), name
    masks = np.minimum.reduce([d['masks'] for _, d, _, _ in loaded])
    # Compare GT unions: baseline stores local anchors; B1 repeats whole-video anchors.
    def anchors(ms):
        result = {}
        for m in ms:
            for c, ts in enumerate(m['online_gt_anchors']):
                result.setdefault((m['source'], m['video_id'], c), set()).update(round(float(t), 4) for t in ts)
        return result
    gt = anchors(metas)
    for name, _, ms, _ in loaded:
        assert anchors(ms) == gt, f'GT changed: {name}'
    result = {'protocol': {'windows': len(keys), 'videos': len(set(ref['video_ids'])), 'identical_targets_and_gt_unions': True,
        'masks': 'intersection across caches; incomplete video/class excluded from strict event metrics',
        'min_recalls': [.85, .8, .8], 'nms_radius_sec': 5, 'tolerance_sec': 5, 'threshold_candidates': 401,
        'scope': 'internal validation; thresholds tuned on evaluated split, not independent test'}, 'models': {}}
    labels = ref['labels'].tolist()
    for name, d, _, _ in loaded:
        _, events = tune_online_event_thresholds(d['online_probs'], d['candidate_times'], metas, labels,
            ['precision'] * 3, [.85, .8, .8], masks, nms_radius_sec=5, tolerance_sec=5, max_candidates=401)
        ap = {label: float(average_precision_score(d['targets'][masks[:, c] > .5, c], d['probs'][masks[:, c] > .5, c])) for c, label in enumerate(labels)}
        pervideo = {}
        for c, label in enumerate(labels):
            pervideo[label] = {}
            for vid in np.unique(ref['video_ids']):
                valid = (ref['video_ids'] == vid) & (masks[:, c] > .5)
                if valid.any() and ref['targets'][valid, c].sum() > 0:
                    pervideo[label][str(vid)] = float(average_precision_score(d['targets'][valid, c], d['probs'][valid, c]))
        result['models'][name] = {'strict_events': events, 'pooled_window_AP': ap, 'macro_window_AP': float(np.mean(list(ap.values()))), 'per_video_window_AP': pervideo,
            'original_mask_valid_counts': d['masks'].sum(0).tolist(),
            'candidate_time_change_fraction_vs_E16': np.mean(np.abs(d['candidate_times']-ref['candidate_times']) > .01, axis=0).tolist()}
    (out / 'metrics.json').write_text(json.dumps(result, indent=2))
    for name, r in result['models'].items():
        print(name, 'window AP', r['pooled_window_AP'])
        print('events', {k: {s: v[s] for s in ['precision', 'recall', 'tp', 'fp', 'fn', 'support', 'complete_video_count']} for k,v in r['strict_events'].items()})
    print(out)


if __name__ == '__main__':
    main()
