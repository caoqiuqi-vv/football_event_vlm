#!/usr/bin/env python
"""Temporal heatmap rendering around real shot events; never updates model or experiment files.

For a clip window around an event anchor, samples frames at a fixed stride and
renders ball/goal heatmaps (before: frozen DINO + control head, after: Stage1
DINO + adapted head) as a time-ordered grid.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from football_localization_full import digest
from scripts.train_football_localization_full import FullModel
from scripts.render_football_localization_full_comparison import extract, decorate


def ts2sec(t):
    h, m, s = t.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)


def load_ball_index(path, vid):
    p = Path(path) / f'{vid}.npz'
    if not p.is_file():
        return None
    with np.load(p) as d:
        return {k: d[k] for k in d.files}


def ball_ref_at(ball, t, tol=0.34):
    if ball is None or not len(ball['timestamp_sec']):
        return None
    i = int(np.searchsorted(ball['timestamp_sec'], t))
    best, best_dt = None, tol
    for j in (i - 1, i):
        if 0 <= j < len(ball['timestamp_sec']):
            dt = abs(float(ball['timestamp_sec'][j]) - t)
            if dt <= best_dt:
                best, best_dt = j, dt
    if best is None:
        return None
    return {'box': ball['bbox_xyxy_norm'][best].tolist(),
            'conf': float(ball['confidence'][best]),
            'source': int(ball['source_code'][best]),
            'heatmap_usable': bool(int(ball['flags'][best]) & 1)}


def goal_refs_at(goals, frame_id, tol=30, conf_floor=0.4):
    if goals is None:
        return []
    ids = goals['frame_ids'].numpy()
    i = int(np.searchsorted(ids, frame_id))
    best, best_dt = None, tol + 1
    for j in (i - 1, i):
        if 0 <= j < len(ids):
            dt = abs(int(ids[j]) - frame_id)
            if dt < best_dt:
                best, best_dt = j, dt
    if best is None:
        return []
    lo, hi = int(goals['frame_offsets'][best]), int(goals['frame_offsets'][best + 1])
    cls = goals['classes'][lo:hi].numpy()
    conf = goals['confidences'][lo:hi].float().numpy()
    boxes = goals['boxes'][lo:hi].numpy()
    size = goals['image_size']
    w, h = size['width'], size['height']
    out = []
    for c, cf, b in zip(cls, conf, boxes):
        if c == 2 and cf >= conf_floor and b[2] > b[0] and b[3] > b[1]:
            out.append(([float(b[0] / w), float(b[1] / h), float(b[2] / w), float(b[3] / h)], float(cf)))
    return out


def auto_select_events(val_ids, ann_dir, ball_root, goal_root, per_video=1, min_gap_sec=20):
    picked = []
    for vid in val_ids:
        ann = Path(ann_dir) / f'{vid}.json'
        if not ann.is_file():
            continue
        events = [e for e in json.loads(ann.read_text()) if e['label'] == '射门' and e.get('label_correct', True)]
        ball = load_ball_index(ball_root, vid)
        goals = None
        gp = Path(goal_root) / f'{vid}.pt'
        if gp.is_file():
            goals = torch.load(gp, map_location='cpu', weights_only=False)
        fps = float(goals['fps']) if goals is not None else None
        best = None
        for e in events:
            t = ts2sec(e['timestamp'])
            br = ball_ref_at(ball, t)
            if not br or br['source'] != 1 or br['conf'] < 0.5:
                continue
            gr = goal_refs_at(goals, round(t * fps), tol=45) if fps else []
            if not gr:
                continue
            score = br['conf'] + max(g[1] for g in gr)
            if any(abs(t - p['anchor_sec']) < min_gap_sec for p in picked if p['video_id'] == vid):
                continue
            if best is None or score > best[0]:
                best = (score, t, br, len(gr))
        if best is not None:
            picked.append({'video_id': vid, 'anchor_sec': best[1], 'score': best[0]})
        if len(picked) >= per_video * len(val_ids):
            break
    return picked


def render_clip(model, cap_ctx, ball, goals, fps, vid, anchor, pre, post, stride, out, event_label='射门'):
    times = np.arange(anchor - pre, anchor + post + 1e-6, stride)
    frames, probs_all, refs = [], [], []
    cap = cap_ctx
    for t in times:
        fi = int(round(t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, img = cap.read()
        if not ok:
            print(f'WARN decode failed {vid} frame={fi}, skipped', flush=True)
            continue
        if img.shape[:2] != (720, 1280):
            img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1)
        probability, _, _ = extract(model, tensor)
        br = ball_ref_at(ball, t)
        gr = goal_refs_at(goals, fi)
        frames.append((t, fi, rgb))
        probs_all.append(probability)
        refs.append((br, gr))
    for c, name in enumerate(('ball', 'goal')):
        n = len(frames)
        fig, axes = plt.subplots(n, 3, figsize=(18, n * 3.2), squeeze=False)
        for row, ((t, fi, rgb), probability, (br, gr)) in enumerate(zip(frames, probs_all, refs)):
            boxes = np.zeros((2, 16, 4))
            if br:
                boxes[0, 0] = br['box']
            for k, (gb, _) in enumerate(gr[:16]):
                boxes[1, k] = gb
            titles = ['720P RGB / automatic reference',
                      'Before: frozen DINO + control head e1',
                      'After: Stage1 DINO + adapted head e2']
            arms = [None, 0, 1]
            for ax, arm, title in zip(axes[row], arms, titles):
                decorate(ax, rgb, boxes, c, None if arm is None else probability[arm, :, :, c])
                ax.set_title(title if row == 0 else '', fontsize=10)
            off = t - anchor
            tag = f't={t:.2f}s ({off:+.2f}s)'
            if abs(off) < stride / 2:
                tag += f' <= {event_label} anchor'
            br_tag = 'ball-ref src%d conf%.2f' % (br['source'], br['conf']) if br else 'no ball-ref'
            axes[row][0].text(0, -.06, f'{tag} | {br_tag} | goal-refs={len(gr)}',
                              transform=axes[row][0].transAxes, fontsize=8)
        fig.suptitle(f'{name.upper()} temporal heatmap | {vid} | {event_label} anchor={anchor:.2f}s\n'
                     'Green: automatic reference (ball box / goal boxes); cyan: peak. Fixed heat scale; not object presence.', fontsize=12)
        fig.subplots_adjust(left=.015, right=.985, bottom=.05, top=.94, hspace=.25, wspace=.035)
        from matplotlib.colors import Normalize
        colorax = fig.add_axes([.32, .015, .36, .012])
        bar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 12), cmap='magma'), cax=colorax, orientation='horizontal')
        bar.set_label('Fixed heat scale: log2(1 + 3600 x spatial probability)', fontsize=9)
        fig.savefig(out / f'{vid}_{anchor:.1f}s_{name}_temporal.png', dpi=130)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage1-dir', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--video-id', default=None)
    ap.add_argument('--anchor-sec', type=float, default=None)
    ap.add_argument('--auto-select', type=int, default=0, help='auto pick N shot events from val videos')
    ap.add_argument('--pre-sec', type=float, default=5.0)
    ap.add_argument('--post-sec', type=float, default=2.0)
    ap.add_argument('--stride-sec', type=float, default=0.5)
    args = ap.parse_args()
    src, out = args.stage1_dir, args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((src / 'manifest.json').read_text())
    cfg = manifest['config']
    fps_by_vid = {d['video_id']: d['fps'] for d in manifest['splits']['val']}
    model = FullModel(cfg)
    adapt = torch.load(src / 'best_adapt.pt', map_location='cpu', weights_only=True, mmap=True)
    control = torch.load(src / 'best_control.pt', map_location='cpu', weights_only=True, mmap=True)
    assert adapt['manifest_sha256'] == control['manifest_sha256'] == digest(src / 'manifest.json')
    params = dict(model.backbone.named_parameters())
    with torch.no_grad():
        for name, value in adapt['backbone_lora'].items():
            params[name].copy_(value)
    model.adapt_head.load_state_dict(adapt['adapt_head'])
    model.control_head.load_state_dict(control['control_head'])
    model.requires_grad_(False).cuda().eval()
    if args.video_id and args.anchor_sec is not None:
        clips = [{'video_id': args.video_id, 'anchor_sec': args.anchor_sec}]
    else:
        val_ids = [d['video_id'] for d in manifest['splits']['val']]
        clips = auto_select_events(val_ids, '/mnt/data_16t/football/football_events_human_repair',
                                   cfg['ball_index'], cfg['goal_index'])[:max(args.auto_select, 1)]
    (out / 'clips.json').write_text(json.dumps(clips, indent=2))
    for clip in clips:
        vid, anchor = clip['video_id'], clip['anchor_sec']
        video_path = Path(cfg['video_root']) / f'{vid}.mp4'
        cap = cv2.VideoCapture(str(video_path))
        goals = None
        gp = Path(cfg['goal_index']) / f'{vid}.pt'
        if gp.is_file():
            goals = torch.load(gp, map_location='cpu', weights_only=False)
        ball = load_ball_index(cfg['ball_index'], vid)
        render_clip(model, cap, ball, goals, fps_by_vid[vid], vid, anchor,
                    args.pre_sec, args.post_sec, args.stride_sec, out)
        cap.release()
        print(f'CLIP DONE {vid} anchor={anchor:.2f}s', flush=True)
    print(f'ALL DONE out={out}', flush=True)


if __name__ == '__main__':
    main()
