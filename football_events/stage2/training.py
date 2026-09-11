"""Shared joint/frozen temporal training, calibration selection and stress evaluation.

Only the adapter updates during warmup. Subsequent joint epochs update the
original temporal transformer and classifier against an immutable teacher.
The selected and last enabled models are both audited; epoch zero is a valid
fallback candidate, not a successful newly trained model.
"""
import fcntl
import json
import math
import time
import numpy as np
import torch
from football_stage2_joint import JointTemporalReader
from .artifacts import atomic_json, digest, save_torch
from .data import Bank
from .metrics import EventCurves, window_ap

def train(out, cfg, m, arm, seed):
    dest = out / f'{arm}_seed{seed}'
    dest.mkdir(exist_ok=True)
    lock = (dest / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    if (dest / 'COMPLETE.json').exists():
        return
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    bank = Bank(out, cfg, m, arm)
    model = JointTemporalReader(cfg, joint=cfg['arms'][arm]['joint']).cuda()
    groups = model.optimizer_groups()
    params = [p for g in groups for p in g['params']]
    opt = torch.optim.AdamW(groups, weight_decay=cfg['weight_decay'])
    sha = digest(out / 'manifest.json')
    curves = {s: EventCurves(m['splits'][s], m, 2.0) for s in ['calibration', 'development']}

    @torch.no_grad()
    def predict(group, enabled=True, stress=None):
        model.eval()
        parts = []
        null = []
        deltas = []
        indices = bank.groups[group]
        for i, b in enumerate(bank.batches(indices, cfg['eval_batch_size'])):
            if stress == 'empty':
                b['valid'].zero_()
            elif stress == 'half_missing':
                b['valid'][:, ::2] = False
            elif stress == 'reverse_local_time':
                for k in ['tokens', 'descriptors', 'valid']:
                    b[k] = b[k].flip(1)
            elif stress == 'cross_window':
                start = i * cfg['eval_batch_size']
                wrong = indices[(np.arange(start, start + len(b['tokens'])) + len(indices) // 2) % len(indices)]
                corrupt = {k: v.cuda(non_blocking=True) for k, v in bank.cpu(wrong).items()}
                for k in ['tokens', 'descriptors', 'valid']:
                    b[k] = corrupt[k]
            r = model(**b, enabled=enabled)
            parts.append(r['logits'].cpu().numpy())
            null.append(r['null_mass'].cpu().numpy())
            deltas.append(r['delta'].cpu().numpy())
        logits = np.concatenate(parts)
        return {
            'logits': logits,
            'prob': 1 / (1 + np.exp(-np.clip(logits, -50, 50))),
            'delta': np.concatenate(deltas),
            'null_mass': np.concatenate(null),
        }
    basecal = predict('calibration', False)
    base_th, base_metrics = curves['calibration'].tune(basecal['prob'], cfg['recall_floors'])
    baseap = window_ap(curves['calibration'], basecal['prob'])
    best = {'epoch': 0, 'enabled': False, 'score': baseap['macro'], 'thresholds': base_th, 'calibration': base_metrics}
    begin = 1

    def state(epoch):
        return {
            'epoch': epoch,
            'learned': model.learned_state(),
            'optimizer': opt.state_dict(),
            'best': best,
            'baseline_thresholds': base_th,
            'config': cfg,
            'manifest_sha256': sha,
            'arm': arm,
            'seed': seed,
        }
    if (dest / 'resume.pt').exists():
        s = torch.load(dest / 'resume.pt', weights_only=True, map_location='cpu')
        assert s['manifest_sha256'] == sha and s['config'] == cfg
        model.load_learned(s['learned'])
        model.set_epoch(s['epoch'])
        opt.load_state_dict(s['optimizer'])
        best = s['best']
        begin = s['epoch'] + 1
    else:
        initial = predict('calibration', True)
        assert np.allclose(initial['logits'], basecal['logits'], atol=1e-06, rtol=0), 'initialization differs from original model'
        save_torch(dest / 'best.pt', state(0))
    rows = m['splits']['train']
    targets = torch.tensor([r['labels'] for r in rows], device='cuda', dtype=torch.float32)
    masks = torch.tensor([r['label_mask'] for r in rows], device='cuda', dtype=torch.float32)
    n = len(rows)
    weights = torch.tensor(cfg['pos_weight'], device='cuda')
    start_all = time.time()
    for epoch in range(begin, cfg['epochs'] + 1):
        model.set_epoch(epoch)
        torch.manual_seed(seed * 100 + epoch)
        order = np.random.default_rng(seed * 100 + epoch).permutation(n)
        model.train()
        losses = []
        grad_max = 0.0
        start = time.time()
        for step, b in enumerate(bank.batches(bank.groups['train'][order], cfg['batch_size'])):
            take = torch.as_tensor(order[step * cfg['batch_size']:(step + 1) * cfg['batch_size']], device='cuda')
            count = len(take)
            keep_clip = torch.rand(count, 1, 1, device='cuda') >= cfg['clip_drop_probability']
            keep_frame = torch.rand(count, 16, 1, device='cuda') >= cfg['frame_drop_probability']
            keep_candidate = torch.rand(count, 16, 2, device='cuda') >= cfg['candidate_drop_probability']
            which = b['descriptors'][..., 10].long().clamp(0, 1)
            b['valid'] &= keep_clip & keep_frame & keep_candidate.gather(2, which)
            ratio = ((epoch - 1) * n + step * cfg['batch_size']) / (cfg['epochs'] * n)
            decay = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * ratio))
            for group in opt.param_groups:
                group['lr'] = group['base_lr'] * decay
            r = model(**b)
            mask = masks[take]
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(r['logits'], targets[take], pos_weight=weights, reduction='none') * mask).sum() / mask.sum().clamp_min(1)
            loss = loss + cfg['residual_l2_weight'] * r['delta'].square().mean()
            corrupt = {k: v[:max(2, count // 4)] for k, v in b.items()}
            for k in ['tokens', 'descriptors', 'valid']:
                corrupt[k] = corrupt[k].roll(1, 0)
            loss = loss + cfg['corrupt_consistency_weight'] * model(**corrupt)['delta'].square().mean()
            assert torch.isfinite(loss)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            assert torch.isfinite(gn)
            opt.step()
            losses.append(float(loss.detach()))
            grad_max = max(grad_max, float(gn))
            if step % 25 == 0:
                h = {
                    'phase': 'training',
                    'arm': arm,
                    'seed': seed,
                    'epoch': epoch,
                    'step': step + 1,
                    'steps': math.ceil(n / cfg['batch_size']),
                    'loss': float(loss.detach()),
                    'elapsed_sec': time.time() - start,
                    'updated_unix': time.time(),
                }
                atomic_json(dest / 'heartbeat.json', h)
                print(json.dumps(h), flush=True)
        assert grad_max > 0
        pred = predict('calibration')
        ap = window_ap(curves['calibration'], pred['prob'])
        thresholds, metrics = curves['calibration'].tune(pred['prob'], cfg['recall_floors'])
        fixed = curves['calibration'].evaluate(pred['prob'], base_th)
        if ap['macro'] > best['score'] + 1e-08:
            best = {'epoch': epoch, 'enabled': True, 'score': ap['macro'], 'thresholds': thresholds, 'calibration': metrics}
            save_torch(dest / 'best.pt', state(epoch))
        summary = {
            'epoch': epoch,
            'loss_mean': float(np.mean(losses)),
            'gradient_norm_max': grad_max,
            'temporal_updates_enabled': bool(model.joint and epoch > cfg['adapter_warmup_epochs']),
            'optimizer_lrs': {g['name']: g['lr'] for g in opt.param_groups},
            'calibration_window_AP': ap,
            'calibration_retuned': metrics,
            'calibration_fixed_thresholds': fixed,
            'selection': 'calibration macro window AP',
            'best_epoch': best['epoch'],
            'null_mass_mean': float(pred['null_mass'].mean()),
            'elapsed_sec': time.time() - start,
        }
        atomic_json(dest / f'epoch_{epoch:03d}.json', summary)
        save_torch(dest / f'epoch_{epoch:03d}.pt', state(epoch))
        save_torch(dest / 'resume.pt', state(epoch))
    last = predict('development', True)
    last_stress = {}
    for stress in ['empty', 'half_missing', 'cross_window', 'reverse_local_time']:
        pred = predict('development', True, stress)
        if stress == 'empty':
            assert np.array_equal(pred['logits'], np.asarray(bank.arrays['anchor'][bank.groups['development']]))
        last_stress[stress] = {
            'window_AP': window_ap(curves['development'], pred['prob']),
            'fixed_thresholds': curves['development'].evaluate(pred['prob'], base_th),
            'mean_absolute_residual': float(np.abs(pred['delta']).mean()),
        }
        np.savez_compressed(dest / f'last_enabled_{stress}.npz', logits=pred['logits'])
    np.savez_compressed(dest / 'last_enabled.npz', logits=last['logits'])
    chosen = torch.load(dest / 'best.pt', weights_only=True, map_location='cpu')['best']
    selected_state = torch.load(dest / 'best.pt', weights_only=True, map_location='cpu')
    model.load_learned(selected_state['learned'])
    model.set_epoch(selected_state['epoch'])
    dev = predict('development', chosen['enabled'])
    baseline = predict('development', False)
    empty = predict('development', True, 'empty')
    assert np.array_equal(empty['logits'], baseline['logits'])
    selected_stress = {}
    for stress in ['empty', 'half_missing', 'cross_window', 'reverse_local_time']:
        pred = predict('development', True, stress)
        thresholds = base_th if stress == 'empty' else chosen['thresholds']
        selected_stress[stress] = {
            'window_AP': window_ap(curves['development'], pred['prob']),
            'operating_metrics': curves['development'].evaluate(pred['prob'], thresholds),
            'mean_absolute_residual': float(np.abs(pred['delta']).mean()),
        }
        np.savez_compressed(dest / f'selected_enabled_{stress}.npz', logits=pred['logits'])
    known_error = []
    case = cfg.get('known_teacher_error')
    if case:
        group = case['split']
        selected_prob = predict(group, chosen['enabled'])
        indices = bank.groups[group]
        for i, row in enumerate(m['splits'][group]):
            if row['video_id'] != case['video_id'] or row['end_sec'] < case['start_sec'] or row['start_sec'] > case['end_sec']:
                continue
            descriptor = bank.cpu(indices[i:i + 1])['descriptors'][0]
            known_error.append({
                'video_id': row['video_id'],
                'start_sec': row['start_sec'],
                'end_sec': row['end_sec'],
                'frame_times': row['frame_times'],
                'first_candidate_xy_normalized': descriptor[:, 4, :2].tolist(),
                'baseline_logits': np.asarray(bank.arrays['anchor'][indices[i]]).tolist(),
                'selected_logits': selected_prob['logits'][i].tolist(),
                'null_mass_per_frame': selected_prob['null_mass'][i].tolist(),
                'scope': 'known contaminated calibration reference; no labels used to train this case; null mass is not presence probability',
            })
        atomic_json(dest / 'known_teacher_error_probe.json', {'cases': known_error, 'selected_epoch': chosen['epoch'], 'selected_enabled': chosen['enabled']})
    result = {
        'arm': arm,
        'seed': seed,
        'known_teacher_error_probe': known_error,
        'selected_enabled_stress': selected_stress,
        'selected_stress_note': 'adapter forced enabled even if selection chose epoch0; epoch0 adapter is zero initialized',
        'selected': chosen,
        'baseline_thresholds': base_th,
        'baseline_calibration_AP': baseap,
        'development_AP': window_ap(curves['development'], dev['prob']),
        'baseline_development_AP': window_ap(curves['development'], baseline['prob']),
        'development_retuned': curves['development'].evaluate(dev['prob'], chosen['thresholds']),
        'development_fixed': curves['development'].evaluate(dev['prob'], base_th),
        'baseline_development': curves['development'].evaluate(baseline['prob'], base_th),
        'last_enabled_AP': window_ap(curves['development'], last['prob']),
        'last_enabled_stress': last_stress,
        'explicit_empty_logits_exact': True,
        'empty_decision_rule': 'use baseline thresholds when all candidates explicitly invalid; normal inputs use selected thresholds',
        'natural_absence_verified': False,
        'optimized_parameters': sum((p.numel() for p in params)),
        'adapter_parameters': sum((p.numel() for p in model.adapter.parameters())),
        'joint_temporal_training': model.joint,
        'warmup_epochs': cfg['adapter_warmup_epochs'],
        'manifest_sha256': sha,
        'elapsed_sec': time.time() - start_all,
    }
    np.savez_compressed(dest / 'development_predictions.npz', logits=dev['logits'], baseline_logits=baseline['logits'])
    atomic_json(dest / 'RESULT.json', result)
    atomic_json(dest / 'COMPLETE.json', {'complete': True, 'epochs': cfg['epochs'], 'manifest_sha256': sha})
