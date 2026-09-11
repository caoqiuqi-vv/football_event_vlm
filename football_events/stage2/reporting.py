"""Predeclared development checks and honest final reporting for one Stage2 run."""
import json
import numpy as np
from .artifacts import atomic_json, digest
from .metrics import EventCurves, LABELS, paired_video_bootstrap

def report(out, cfg, manifest, seed):
    dest = out / f'joint_temporal_seed{seed}'
    assert (dest / 'COMPLETE.json').exists()
    result = json.loads((dest / 'RESULT.json').read_text())
    assert result['manifest_sha256'] == digest(out / 'manifest.json')
    pred = np.load(dest / 'development_predictions.npz')
    ci = paired_video_bootstrap(manifest['splits']['development'], EventCurves(manifest['splits']['development'], manifest, 2.0), pred['logits'], pred['baseline_logits'])
    base, dev = (result['baseline_development'], result['development_retuned'])
    checks = {
        'trained_model_selected': bool(result['selected']['enabled']),
        'macro_window_AP_improved': result['development_AP']['macro'] > result['baseline_development_AP']['macro'],
        'macro_precision_improved': dev['macro_precision'] > base['macro_precision'],
        'recall_and_false_positive_guard': all((dev[c]['recall'] >= base[c]['recall'] - 0.01 and dev[c]['fp_windows_per_hour'] <= base[c]['fp_windows_per_hour'] for c in LABELS)),
        'video_AP_interval_positive': ci['percentile_95CI_pp'] is not None and ci['percentile_95CI_pp'][0] > 0,
        'synthetic_robustness_guard': all((result['selected_enabled_stress'][s]['operating_metrics'][c]['recall'] >= base[c]['recall'] - 0.01 and result['selected_enabled_stress'][s]['operating_metrics'][c]['precision'] >= base[c]['precision'] - 0.01 for s in ['half_missing', 'cross_window'] for c in LABELS)),
        'explicit_empty_logits_exact': result['explicit_empty_logits_exact'],
    }
    final = {
        'seed': seed,
        'checks': checks,
        'all_development_checks_passed': all(checks.values()),
        'paired_video_bootstrap': ci,
        'result': result,
        'interpretation': 'single-seed exploratory result on historically used development videos',
        'prior_causal_gain_established': False,
        'natural_absence_verified': False,
        'temporal_only_control_run': False,
        'independent_test_claimed': False,
    }
    case = cfg.get('known_teacher_error')
    if case:
        index = json.loads((out / 'arrays/index.json').read_text())
        mapping = {k: i for i, k in enumerate(index['keys'])}
        selected = np.load(out / 'arrays/selected_peaks.npy', mmap_mode='r')
        plain = np.load(out / 'arrays/unweighted_peaks.npy', mmap_mode='r')
        rows = []
        for r in manifest['splits'][case['split']]:
            if r['video_id'] != case['video_id'] or r['end_sec'] < case['start_sec'] or r['start_sec'] > case['end_sec']:
                continue
            i = mapping[r['key']]

            def xy(peaks):
                return np.stack([(peaks % 80 + 0.5) * 16, (peaks // 80 + 0.5) * 16], -1).tolist()
            rows.append({
                'key': r['key'],
                'frame_times': r['frame_times'],
                'selected_xy_pixels': xy(selected[i]),
                'without_prior_xy_pixels': xy(plain[i]),
            })
        atomic_json(out / f'KNOWN_CASE_CANDIDATES_seed{seed}.json', {
            'scope': 'calibration diagnostic only; no known-case labels used in training; retained maxima can still be lamps',
            'rows': rows,
        })
    atomic_json(out / f'FINAL_SUMMARY_seed{seed}.json', final)
    text = [f'# 原始 patch＋软位置先验＋联合时序微调：seed {seed}', '', f"选中 epoch：{result['selected']['epoch']}；开发检查全部通过：{all(checks.values())}。", f"开发窗口宏 AP：{result['baseline_development_AP']['macro'] * 100:.3f}% → {result['development_AP']['macro'] * 100:.3f}%。", f"配对视频宏 AP 差的 95% 区间（百分点）：{ci['percentile_95CI_pp']}。", '', '逐类 P/R、误报窗口/小时、原阈值和重新校准阈值结果、所选与末轮模型的空输入/半帧缺失/跨窗错配/时间反转结果见 FINAL_SUMMARY JSON。', '', '此结果不能单独证明位置先验有效：还改变了局部特征来源并联合适配时序层。', '窗口 AP 不是事件 spotting mAP；历史开发集不是独立盲测。单种子结果和合成缺失不能证明自然无球片段鲁棒性。', '若开发检查通过，下一步同配置 seed43 复现，再进行人工核验的自然无球/高球/近景分层及独立测试；不自动开展。']
    (out / f'FINAL_REPORT_seed{seed}.md').write_text('\n'.join(text) + '\n')
