#!/usr/bin/env python
"""Extended heatmap gallery for the full Stage1 run; never updates model or experiment files.

Renders per-class, per-diagnostic-group grids (multiple samples each, distinct
videos preferred) comparing frozen-DINO control arm vs Stage1 adapted arm.
Reuses the verified inference/decorators from render_football_localization_full_comparison.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from football_localization_full import FullFrames, digest
from scripts.train_football_localization_full import FullModel
from scripts.render_football_localization_full_comparison import extract, comparison

GROUPS = ('improved', 'regressed', 'both_correct', 'both_missed')
GROUP_TITLE = {
    'improved': 'After correct, before missed',
    'regressed': 'After missed, before correct',
    'both_correct': 'Both correct',
    'both_missed': 'Both missed',
}


def group_mask(rows, c, group):
    ok = rows['error'][:, :, c] <= 16 if c == 0 else rows['inside'][:, :, c]
    after, before = ok[:, 0], ok[:, 1]
    return {'improved': after & ~before, 'regressed': ~after & before,
            'both_correct': after & before, 'both_missed': ~after & ~before}[group]


def select_gallery(rows, per_group, seed):
    rng = np.random.default_rng(seed)
    selection = []
    # Random panel: 8 distinct videos, one random frame with both references each.
    eligible = np.flatnonzero(rows['valid'].all(1))
    videos = rng.choice(np.unique(rows['video'][eligible]), min(8, len(np.unique(rows['video'][eligible]))), replace=False)
    for video in videos:
        idx = int(rng.choice(eligible[rows['video'][eligible] == video]))
        selection.append({'index': idx, 'group': 'random_both_references', 'class': None})
    for c, name in enumerate(('ball', 'goal')):
        for group in GROUPS:
            pool = np.flatnonzero(rows['valid'][:, c] & group_mask(rows, c, group))
            pool = pool[~np.isin(pool, [r['index'] for r in selection])]
            # Round-robin over distinct videos for coverage, then random fill.
            chosen, used_videos = [], set()
            by_video = {}
            for idx in pool:
                by_video.setdefault(int(rows['video'][idx]), []).append(int(idx))
            vids = list(by_video)
            rng.shuffle(vids)
            for vid in vids:
                if len(chosen) >= per_group:
                    break
                chosen.append(int(rng.choice(by_video[vid])))
                used_videos.add(vid)
            if len(chosen) < per_group:
                rest = [i for i in pool if i not in chosen]
                if rest:
                    chosen += [int(i) for i in rng.choice(rest, min(per_group - len(chosen), len(rest)), replace=False)]
            for idx in chosen:
                selection.append({'index': idx, 'group': group, 'class': name,
                                  'eligible_category_frames': int(len(pool))})
    return selection


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage1-dir', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=20260909)
    ap.add_argument('--per-group', type=int, default=6)
    args = ap.parse_args()
    src, out = args.stage1_dir, args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / 'arrays').mkdir(exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    manifest = json.loads((src / 'manifest.json').read_text())
    rows = np.load(src / 'selected_pair_predictions.npy', mmap_mode='r')
    assert np.array_equal(rows['index'], np.arange(len(rows)))
    selected = select_gallery(rows, args.per_group, args.seed)
    (out / 'selection_before_render.json').write_text(json.dumps(selected, indent=2))
    model = FullModel(manifest['config'])
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
    dataset = FullFrames(manifest['splits']['val'], manifest['config']['max_goal_boxes'])
    records = []
    for sample in selected:
        idx = sample['index']
        batch = dataset[idx]
        vi = batch['video_index']
        desc = dataset.descriptors[vi]
        source_frame = int(dataset.arrays_for(vi)['frames'][idx - dataset.starts[vi]])
        probability, tp, sp = extract(model, batch['frame'])
        peaks = probability.reshape(3, 3600, 2).argmax(1)
        saved_peaks = rows[idx]['peaks'][[1, 0]]
        record = {**sample, 'video_id': desc['video_id'], 'source_frame': source_frame,
                  'time_sec': source_frame / desc['fps'], 'valid': rows[idx]['valid'].tolist(),
                  'render_peaks_before_after_fixed': peaks.tolist(),
                  'full_evaluation_peaks_before_after': saved_peaks.tolist(),
                  'peaks_match_full_evaluation': bool(np.array_equal(peaks[:2], saved_peaks)),
                  'full_evaluation_error_px_before_after': [[float(v) if np.isfinite(v) else None for v in arm] for arm in rows[idx]['error'][[1, 0]]],
                  'full_evaluation_inside_before_after': rows[idx]['inside'][[1, 0]].tolist(),
                  'patch_cosine': float(torch.nn.functional.cosine_similarity(torch.from_numpy(tp), torch.from_numpy(sp), dim=-1).mean()),
                  'boxes': batch['boxes'].numpy(), 'rgb': batch['frame'].permute(1, 2, 0).numpy()}
        np.savez_compressed(out / 'arrays' / f'{idx}.npz', probabilities=probability,
                            rgb=record['rgb'], boxes=record['boxes'])
        records.append(record)
        print(f"RENDER index={idx} class={sample['class']} group={sample['group']} video={desc['video_id']} frame={source_frame}", flush=True)
    del model
    torch.cuda.empty_cache()
    random_records = [r for r in records if r['group'] == 'random_both_references']
    for c, name in enumerate(('ball', 'goal')):
        comparison(out, random_records, c, f'{name}_random.png')
        for group in GROUPS:
            group_records = [r for r in records if r['class'] == name and r['group'] == group]
            if not group_records:
                continue
            comparison(out, group_records, c, f'{name}_{group}.png')
            comparison(out, group_records, c, f'{name}_{group}_zoom.png', zoom=True)
    clean = [{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in r.items() if k != 'rgb'} for r in records]
    report = {'seed': args.seed, 'per_group': args.per_group,
              'artifact_sha256': {n: digest(src / n) for n in ['best_adapt.pt', 'best_control.pt', 'manifest.json', 'selected_pair_predictions.npy']},
              'samples': clean}
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    matched = sum(1 for r in records if r['peaks_match_full_evaluation'])
    (out / 'README.md').write_text(
        '# Stage1 扩展 heatmap 画廊\n\n'
        '列含义与 visualizations_20260909/README.md 相同:before=冻结 DINO+最佳冻结对照头;'
        'after=Stage1 epoch2 DINO+对应头。绿色框为自动参考标注,青色十字为响应峰;'
        '热图统一 log2(1+3600p) 0-12 色阶。空间 softmax 在无对象时也会出现峰,不能解释为存在概率。\n\n'
        f"每类×每分组最多 {args.per_group} 张,优先覆盖不同验证视频。分组按完整评测判定"
        '(球:中心误差≤16px;球门:峰落入任一参考框),类别刻意平衡,不能据此估计总体增益。\n'
        f"峰位复现核验:{matched}/{len(records)} 与全量评测一致(单帧 BF16 并列可能造成差异,详见 report.json)。\n"
        'zoom 图为原始整帧推理结果围绕参考框中心的 192×192 裁剪。\n')
    groups_html = []
    for c, name in enumerate(('ball', 'goal')):
        files = [f'{name}_random.png'] + [f'{name}_{g}.png' for g in GROUPS] + [f'{name}_{g}_zoom.png' for g in GROUPS]
        imgs = '\n'.join(f'<h3>{f}</h3><img src="{f}" style="max-width:100%">' for f in files if (out / f).exists())
        groups_html.append(f'<h2>{name.upper()}</h2>{imgs}')
    (out / 'index.html').write_text('<html><body style="background:#111;color:#ddd">' + '\n'.join(groups_html) + '</body></html>')
    print(f"GALLERY DONE records={len(records)} peaks_matched={matched} out={out}", flush=True)


if __name__ == '__main__':
    main()
