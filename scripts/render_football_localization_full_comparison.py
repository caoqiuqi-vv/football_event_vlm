#!/usr/bin/env python
"""Reproducible 720P diagnostics; never updates model or experiment files."""
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
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from football_localization_full import FullFrames, digest
from scripts.train_football_localization_full import FullModel


@torch.inference_mode()
def extract(model, frame):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        x = (frame[None].cuda().float() / 255 - model.rgb_mean) / model.rgb_std
        x, (h, w) = model.backbone.prepare_tokens_with_masks(x)
        rope = model.backbone.rope_embed(H=h, W=w)
        for block in model.backbone.blocks[:model.start]:
            x = block(x, rope)
        t = x
        for block in model.teacher_tail:
            t = block(t, rope)
        s = x
        for block in model.backbone.blocks[model.start:]:
            s = block(s, rope)
        n = 1 + model.backbone.n_storage_tokens
        tp, sp = model.norm_tokens(t)[:, n:], model.norm_tokens(s)[:, n:]
        logits = torch.stack([model.control_head(tp), model.adapt_head(sp), model.control_head(sp)])[:, 0].float()
    return (logits.softmax(1).cpu().numpy().reshape(3, 45, 80, 2),
            tp[0].float().cpu().numpy(), sp[0].float().cpu().numpy())


def select(rows, seed):
    rng = np.random.default_rng(seed)
    selection = []
    # Random reference panel: choose four distinct videos, then one uniformly
    # sampled frame per video that has both automatic object references.
    eligible = np.flatnonzero(rows['valid'].all(1))
    videos = rng.choice(np.unique(rows['video'][eligible]), 4, replace=False)
    for video in videos:
        idx = int(rng.choice(eligible[rows['video'][eligible] == video]))
        selection.append({'index': idx, 'group': 'random_both_references', 'class': None})
    for c, name in enumerate(('ball', 'goal')):
        ok = rows['error'][:, :, c] <= 16 if c == 0 else rows['inside'][:, :, c]
        for label, mask in [('improved', ok[:, 0] & ~ok[:, 1]),
                            ('regressed', ~ok[:, 0] & ok[:, 1]),
                            ('both_correct', ok[:, 0] & ok[:, 1]),
                            ('both_missed', ~ok[:, 0] & ~ok[:, 1])]:
            eligible = np.flatnonzero(rows['valid'][:, c] & mask)
            eligible = eligible[~np.isin(eligible, [r['index'] for r in selection])]
            idx = int(rng.choice(eligible))
            selection.append({'index': idx, 'group': label, 'class': name,
                              'eligible_category_frames': int(len(eligible))})
    return selection


def decorate(ax, frame, boxes, c, probability=None, peak=True):
    ax.imshow(frame)
    if probability is not None:
        # Fixed scale for every frame, class and arm. Spatial probability is
        # expressed relative to a uniform distribution over 3600 patches.
        salience = np.log2(1 + probability * 3600)
        ax.imshow(salience, extent=(0, 1280, 720, 0), cmap='magma',
                  norm=Normalize(0, 12), interpolation='nearest', alpha=.58)
        if peak:
            p = int(probability.argmax())
            ax.plot((p % 80 + .5) * 16, (p // 80 + .5) * 16,
                    '+', color='cyan', markersize=12, markeredgewidth=1.5)
    for box in boxes[c]:
        x1, y1, x2, y2 = box * [1280, 720, 1280, 720]
        if x2 > x1 and y2 > y1:
            ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1,
                                   fill=False, edgecolor='lime', linewidth=1.2))
    ax.set_xlim(0, 1280); ax.set_ylim(720, 0); ax.axis('off')


def comparison(out, records, c, filename, fixed=False, zoom=False):
    fig, axes = plt.subplots(len(records), 3, figsize=(18, len(records)*3.7), squeeze=False)
    for axes_row, r in zip(axes, records):
        probs = np.load(out / 'arrays' / f"{r['index']}.npz")['probabilities']
        arms = [None, 0, 2 if fixed else 1]
        titles = ['720P RGB / automatic reference', 'Before: frozen DINO + control head e1',
                  'After: Stage1 DINO + SAME control head e1' if fixed else 'After: Stage1 DINO + adapted head e2']
        for ax, arm, title in zip(axes_row, arms, titles):
            decorate(ax, r['rgb'], r['boxes'], c, None if arm is None else probs[arm, :, :, c])
            if zoom:
                b = r['boxes'][c, 0] * [1280, 720, 1280, 720]
                x, y = (b[:2] + b[2:]) / 2
                left, top = np.clip(x-96, 0, 1088), np.clip(y-96, 0, 528)
                ax.set_xlim(left, left+192); ax.set_ylim(top+192, top)
            ax.set_title(title, fontsize=10)
        axes_row[0].text(0, -.055, f"{r['group']} | {r['video_id']} | frame {r['source_frame']} | {r['time_sec']:.2f}s",
                         transform=axes_row[0].transAxes, fontsize=8)
    fig.suptitle(f"{'BALL' if c == 0 else 'GOAL'} | {'fixed-head feature diagnostic' if fixed else 'selected checkpoint comparison'}"
                 + (' | 192px reference-centered crop of full-frame inference' if zoom else '')
                 + '\nGreen: automatic reference; cyan: peak. Missing labels do not establish absence.', fontsize=12)
    fig.subplots_adjust(left=.015, right=.985, bottom=.075, top=.935, hspace=.22, wspace=.035)
    colorax = fig.add_axes([.32, .025, .36, .012])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 12), cmap='magma'), cax=colorax, orientation='horizontal')
    bar.set_label('Fixed heat scale: log2(1 + 3600 x spatial probability); NOT object presence confidence', fontsize=9)
    fig.savefig(out / filename, dpi=145)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage1-dir', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=20260909)
    args = ap.parse_args(); src, out = args.stage1_dir, args.output_dir
    out.mkdir(parents=True, exist_ok=True); (out/'arrays').mkdir(exist_ok=True)
    torch.set_num_threads(2); torch.manual_seed(args.seed)
    manifest = json.loads((src/'manifest.json').read_text())
    rows = np.load(src/'selected_pair_predictions.npy', mmap_mode='r')
    assert np.array_equal(rows['index'], np.arange(len(rows)))
    selected = select(rows, args.seed)
    (out/'selection_before_render.json').write_text(json.dumps(selected, indent=2))
    model = FullModel(manifest['config'])
    adapt = torch.load(src/'best_adapt.pt', map_location='cpu', weights_only=True, mmap=True)
    control = torch.load(src/'best_control.pt', map_location='cpu', weights_only=True, mmap=True)
    assert adapt['epoch'] == 2 and control['epoch'] == 1
    assert adapt['manifest_sha256'] == control['manifest_sha256'] == digest(src/'manifest.json')
    params = dict(model.backbone.named_parameters())
    with torch.no_grad():
        for name, value in adapt['backbone_lora'].items():
            params[name].copy_(value)
    model.adapt_head.load_state_dict(adapt['adapt_head'])
    model.control_head.load_state_dict(control['control_head'])
    model.requires_grad_(False).cuda().eval()
    dataset = FullFrames(manifest['splits']['val'], manifest['config']['max_goal_boxes'])
    records, teacher_features, student_features = [], [], []
    for sample in selected:
        idx = sample['index']; batch = dataset[idx]; vi = batch['video_index']
        desc = dataset.descriptors[vi]
        source_frame = int(dataset.arrays_for(vi)['frames'][idx-dataset.starts[vi]])
        probability, tp, sp = extract(model, batch['frame'])
        peaks = probability.reshape(3, 3600, 2).argmax(1)
        saved_peaks = rows[idx]['peaks'][[1, 0]]
        exact = bool(np.array_equal(peaks[:2], saved_peaks))
        valid_exact = bool(np.array_equal(peaks[:2, rows[idx]['valid']], saved_peaks[:, rows[idx]['valid']]))
        # Batch-size-dependent BF16 ties must be disclosed, never hidden.
        record = {**sample, 'video_id': desc['video_id'], 'source_frame': source_frame,
                  'time_sec': source_frame/desc['fps'], 'valid': rows[idx]['valid'].tolist(),
                  'render_peaks_before_after_fixed': peaks.tolist(),
                  'full_evaluation_peaks_before_after': saved_peaks.tolist(),
                  'peaks_match_full_evaluation': exact,
                  'valid_reference_peaks_match': valid_exact,
                  'full_evaluation_error_px_before_after': [[float(v) if np.isfinite(v) else None for v in arm] for arm in rows[idx]['error'][[1, 0]]],
                  'full_evaluation_inside_before_after': rows[idx]['inside'][[1, 0]].tolist(),
                  'patch_cosine': float(torch.nn.functional.cosine_similarity(torch.from_numpy(tp), torch.from_numpy(sp), dim=-1).mean()),
                  'boxes': batch['boxes'].numpy(), 'rgb': batch['frame'].permute(1, 2, 0).numpy()}
        np.savez_compressed(out/'arrays'/f'{idx}.npz', probabilities=probability,
                            rgb=record['rgb'], boxes=record['boxes'])
        records.append(record)
        if sample['group'] == 'random_both_references':
            teacher_features.append(tp); student_features.append(sp)
        print(f"RENDER index={idx} group={sample['group']} peaks_match={exact}", flush=True)
    del model; torch.cuda.empty_cache()
    random_records = records[:4]
    for c, name in enumerate(('ball', 'goal')):
        diagnostic = [r for r in records if r['class'] == name]
        comparison(out, random_records, c, f'{name}_random.png')
        comparison(out, diagnostic, c, f'{name}_diagnostic.png')
        comparison(out, random_records, c, f'{name}_fixed_head.png', fixed=True)
    comparison(out, random_records, 0, 'ball_random_zoom.png', zoom=True)
    comparison(out, [r for r in records if r['class']=='ball'], 0, 'ball_diagnostic_zoom.png', zoom=True)
    # Fit one basis to original teacher features ONLY, then use exactly that
    # center, basis and robust RGB scale for both conditions on every frame.
    teacher = np.concatenate(teacher_features)
    center = teacher.mean(0); fitted = torch.from_numpy(teacher-center)
    _, _, basis = torch.pca_lowrank(fitted, q=3, center=False, niter=5)
    basis = basis.numpy(); projected = (teacher-center) @ basis
    lo, hi = np.quantile(projected, [.01, .99], axis=0)
    np.savez(out/'pca_basis.npz', center=center, basis=basis, rgb_low=lo, rgb_high=hi)
    fig, axes = plt.subplots(4, 3, figsize=(18, 14))
    for i, r in enumerate(random_records):
        axes[i, 0].imshow(r['rgb'])
        for j, feature in enumerate((teacher_features[i], student_features[i]), 1):
            colors = np.clip(((feature-center) @ basis-lo)/(hi-lo), 0, 1).reshape(45, 80, 3)
            axes[i, j].imshow(colors, interpolation='nearest')
        for ax, title in zip(axes[i], ['720P RGB', 'Before: original DINO features', 'After: Stage1 DINO features']):
            ax.set_title(title, fontsize=11); ax.axis('off')
        axes[i, 0].text(0, -.065, f"{r['video_id']} | {r['time_sec']:.2f}s | patch cosine {r['patch_cosine']:.5f}", transform=axes[i, 0].transAxes, fontsize=8)
    fig.suptitle('Raw patch features | ONE teacher-fitted PCA basis and shared RGB scale\nPCA colors are not ball/goal scores; coarse global similarity can hide local changes.', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(out/'feature_pca.png', dpi=145); plt.close(fig)
    clean = [{k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in r.items() if k != 'rgb'} for r in records]
    report = {'seed': args.seed, 'input_hw': [720, 1280], 'patch_hw': [45, 80],
              'source_checkpoint': manifest['config'].get('checkpoint_snapshot', manifest['config']['checkpoint']),
              'artifact_sha256': {name: digest(src/name) for name in ['best_adapt.pt', 'best_control.pt', 'manifest.json', 'selected_pair_predictions.npy']},
              'script_sha256': digest(__file__), 'peak_matches': sum(r['peaks_match_full_evaluation'] for r in records),
              'valid_reference_peak_matches': sum(r['valid_reference_peaks_match'] for r in records),
              'samples': clean, 'scope': 'Automatic positive development references, not human GT or representative accuracy estimation.'}
    (out/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    text = '''# Stage1 720P 特征与定位可视化

所有图使用相同的 1280×720 RGB 输入，沿用 Stage1 的 INTER_AREA 预处理；网络输出为 45×80 patch。显示和局部放大不改变网络输入。

## 对照含义

- before：用户指定 720P best 的冻结 DINO 特征 + 最佳冻结对照定位头（epoch 1）。这不是未训练的 DINO 自带的足球/球门概率。
- after：Stage1 epoch 2 的 DINO 特征 + 对应定位头。
- fixed_head：前后使用同一个冻结对照定位头，只更换 DINO 特征。该图用于分离特征变化与定位头变化；固定头可能与微调特征不匹配，不能代替完整模型评测。
- feature_pca：仅用四张随机样本的原始 DINO patch 特征拟合一次 PCA，前后共用中心、投影和 RGB 色阶。颜色不是对象置信度；大体相似也不意味着事件语义完全保持。

## 读图与取样

绿色框是自动参考标注，青色十字是响应峰。热图取每类 3600 patch 上的空间 softmax，统一使用 log2(1+3600p) 的 0–12 色阶；不做逐图归一化。空间 softmax 即使画面无对象也会出现峰，因此不能解释成对象存在概率。局部图为原始整帧推理结果的 192×192 参考框中心裁剪。

random：种子 20260909，先从有双类参考的视频中随机选择四个不同视频，再分别随机抽一张双类参考帧。仅是受约束的随机展示，不代表完整分布。diagnostic：按完整评测的提升、退化、均正确、均失败分别随机抽样；足球判据为中心误差 ≤16px，球门判据为峰位于任一参考框内。类别刻意平衡，不能据此估计总体增益。

完整评测与本次单帧 BF16 推理的峰位复现核验见 report.json；如不完全一致，原始峰和本次峰均保留，案例分类以完整评测为准。无法从缺失标注判断对象实际不存在。

## 可靠结论的边界

全量配对评测中，足球中心 ≤16px 为 80.586% → 84.907%（+4.321 个百分点）；球门峰入框为 76.033% → 79.015%（+2.982 个百分点），但球门按视频 bootstrap 的增益区间跨零。图像用于解释这些结果，不能独立证明检测准确率、目标缺失鲁棒性或 Stage2 事件收益。

report.json 保存采样、帧号、时间、参考框、峰位、模型与脚本摘要；arrays/ 保留未经渲染的空间概率和原帧；pca_basis.npz 保存共享 PCA 参数。
'''
    (out/'README.md').write_text(text)
    files = ['ball_random.png', 'ball_random_zoom.png', 'goal_random.png', 'feature_pca.png',
             'ball_fixed_head.png', 'goal_fixed_head.png', 'ball_diagnostic.png', 'ball_diagnostic_zoom.png', 'goal_diagnostic.png']
    html = '<!doctype html><meta charset="utf-8"><title>Stage1 720P comparison</title><style>body{max-width:1500px;margin:30px auto;font-family:sans-serif}img{width:100%}</style><h1>Stage1 720P 可视化</h1><p>自动参考标注；统一色阶；包含退化与失败案例。方法和局限见 <a href="README.md">README</a>，复现核验见 <a href="report.json">report.json</a>。</p>'
    html += ''.join(f'<h2>{name}</h2><a href="{name}"><img src="{name}" loading="lazy"></a>' for name in files)
    (out/'index.html').write_text(html)
    print(json.dumps({'output_dir': str(out), 'frames': len(records), 'peak_matches': report['peak_matches']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
