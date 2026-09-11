"""Two-epoch online Stage2 probe: frozen DINO forwards, no feature-cache files.

Run with torchrun. Only the reader participates in DDP. Epoch checkpoints,
predictions and metrics are persisted; extracted visual features are transient.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import fcntl
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from football_stage2_optional import WindowFrames
from football_stage2_joint import JointTemporalReader
from .model import PositionPriorExtractor
from .artifacts import atomic_json, digest, save_torch
from .metrics import EventCurves, LABELS, window_ap, paired_video_bootstrap

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / 'configs/football/stage2_online_softprior_2ep_20260911.json'


def shard_order(count, seed, world, rank, batch_size):
    """Every real row once; -1 pads only the final DDP microbatch, never repeats rows."""
    order = np.random.default_rng(seed).permutation(count)
    unit = world * batch_size
    order = np.pad(order, (0, (-count) % unit), constant_values=-1)
    return order, order.reshape(-1, world, batch_size)[:, rank].reshape(-1).tolist()


class OnlineFrames(WindowFrames):
    def __getitem__(self, index):
        if index == -1:
            return torch.zeros(16, 3, 720, 1280, dtype=torch.uint8), -1
        return super().__getitem__(index)


def loader(records, indices, cfg):
    options = dict(num_workers=cfg['workers'], pin_memory=True)
    if cfg['workers']:
        options.update(prefetch_factor=1, multiprocessing_context='spawn')
    return DataLoader(OnlineFrames(records), batch_size=cfg['micro_batch_size'],
                      sampler=indices, **options)


@torch.no_grad()
def extract_batch(extractor, frames, times, chunk):
    """No disk reads/writes of features; same rounding as the verified inference API."""
    rows, anchors = [], []
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for clip in frames:
            row = extractor.extract_clip(clip, chunk)
            anchors.append(extractor.event._global_branch_outputs(
                row['global_features'].float()[None])['logits'][0].float())
            rows.append(row)
    return {
        'global_features': torch.stack([r['global_features'].float() for r in rows]),
        'tokens': torch.stack([r['stage1_tokens'].half().float() for r in rows]),
        'descriptors': torch.stack([r['stage1_descriptors'].float() for r in rows]),
        'valid': torch.stack([r['stage1_valid'] for r in rows]),
        'anchor': torch.stack(anchors),
        'frame_times': times,
    }


def drop_evidence(batch, cfg):
    b, t, _ = batch['valid'].shape
    device = batch['valid'].device
    clip = torch.rand(b, 1, 1, device=device) >= cfg['clip_drop_probability']
    frame = torch.rand(b, t, 1, device=device) >= cfg['frame_drop_probability']
    candidate = torch.rand(b, t, 2, device=device) >= cfg['candidate_drop_probability']
    which = batch['descriptors'][..., 10].long().clamp(0, 1)
    batch['valid'] &= clip & frame & candidate.gather(2, which)


def paired_forward_batch(batch):
    """One DDP forward for clean and cross-window evidence, including accumulated steps."""
    wrong = dict(batch)
    for key in ['tokens', 'descriptors', 'valid']:
        wrong[key] = batch[key].roll(1, 0)
    return {key: torch.cat([value, wrong[key]], 0) for key, value in batch.items()}


def optimizer_step(model, optimizer, cfg, updates, total_updates):
    """Warmup suppresses temporal/head optimizer state and weight decay as well as updates."""
    decay = .1 + .9 * .5 * (1 + math.cos(math.pi * updates / max(total_updates, 1)))
    for group in optimizer.param_groups:
        group['lr'] = group['base_lr'] * decay
        if updates < cfg['warmup_optimizer_steps'] and group['name'] == 'temporal_and_classifier':
            for parameter in group['params']:
                parameter.grad = None
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    if not torch.isfinite(norm):
        raise RuntimeError('Nonfinite reader gradient')
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return float(norm)


@torch.no_grad()
def evaluate(extractor, reader, records, cfg, rank, world, stress=False):
    """Unpadded evaluation shards; baseline and model share every decoded frame."""
    reader.eval()
    rows = []
    indices = list(range(rank, len(records), world))
    for frames, ids in loader(records, indices, cfg):
        times = torch.tensor([records[i]['frame_times'] for i in ids.tolist()], device='cuda')
        batch = extract_batch(extractor, frames.cuda(non_blocking=True), times, cfg['frame_chunk'])
        values = {'logits': reader(**batch)['logits'], 'baseline_logits': batch['anchor']}
        if stress:
            empty = dict(batch, valid=torch.zeros_like(batch['valid']))
            values['empty'] = reader(**empty)['logits']
            assert torch.equal(values['empty'], batch['anchor'])
            half = dict(batch, valid=batch['valid'].clone())
            half['valid'][:, ::2] = False
            values['half_missing'] = reader(**half)['logits']
            reverse = dict(batch)
            for key in ['tokens', 'descriptors', 'valid']:
                reverse[key] = batch[key].flip(1)
            values['reverse_local_time'] = reader(**reverse)['logits']
        values = {key: value.cpu().numpy() for key, value in values.items()}
        rows.extend((int(index), {key: value[i] for key, value in values.items()})
                    for i, index in enumerate(ids))
    if world > 1:
        gathered = [None] * world if rank == 0 else None
        dist.gather_object(rows, gathered, dst=0)
        if rank:
            return None
        rows = [row for shard in gathered for row in shard]
    rows.sort(key=lambda row: row[0])
    assert [row[0] for row in rows] == list(range(len(records))), 'Evaluation coverage mismatch'
    return {key: np.stack([row[1][key] for row in rows]) for key in rows[0][1]}


def probability(logits):
    return 1 / (1 + np.exp(-np.clip(logits, -50, 50)))


def metrics(prediction, curves, thresholds):
    prob = probability(prediction)
    return {'window_AP': window_ap(curves, prob), 'operating_metrics': curves.evaluate(prob, thresholds)}


def preflight(cfg):
    assert cfg['epochs'] in (1, 2) and cfg['num_frames'] == 16 and cfg['image_size'] == [720, 1280]
    assert cfg['micro_batch_size'] >= 2, 'Cross-window corruption needs at least two local clips'
    assert cfg['accumulation_steps'] >= 1 and cfg['warmup_optimizer_steps'] >= 0
    manifest = json.loads(Path(cfg['manifest_source']).read_text())
    for key in ['source_checkpoint', 'stage1_checkpoint', 'control_checkpoint', 'event_config']:
        assert Path(cfg[key]).is_file(), f'Missing {key}: {cfg[key]}'
    for group in ['train', 'calibration', 'development']:
        assert manifest['splits'][group]
        assert all(len(r['frame_indices']) == 16 for r in manifest['splits'][group])
    effective = cfg['world_size'] * cfg['micro_batch_size'] * cfg['accumulation_steps']
    assert cfg['warmup_optimizer_steps'] < math.ceil(len(manifest['splits']['train']) / effective) * cfg['epochs'], 'Warmup would consume the entire probe'
    paths = {r['video_path'] for rows in manifest['splits'].values() for r in rows}
    missing = [p for p in paths if not Path(p).is_file()]
    assert not missing, f'Missing videos: {missing[:5]}'
    return manifest


def train(cfg, source_manifest):
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    assert world == cfg['world_size'], 'Launch with the world size specified in the config'
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group('nccl')
    torch.set_num_threads(1)
    torch.manual_seed(cfg['seed'])
    torch.use_deterministic_algorithms(True)
    out = Path(cfg['output_dir'])
    lock = None
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        lock = (out/'run.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if world > 1:
        dist.barrier()
    manifest = dict(source_manifest, config=cfg)
    if rank == 0:
        for name, value in [('config.json', cfg), ('manifest.json', manifest)]:
            path = out/name
            if path.exists():
                assert json.loads(path.read_text()) == value, f'Changed run configuration: {path}'
            else:
                atomic_json(path, value)
        dependencies = list((ROOT/'football_events').rglob('*.py'))
        dependencies += [ROOT/name for name in [
            'football_stage2_joint.py', 'football_stage2_corepatch.py', 'football_stage2_optional.py',
            'football_stage2_metrics.py', 'train_football_events.py',
            'scripts/train_football_localization_stage1.py', 'football_localization_full.py']]
        dependencies += list((ROOT/'dinov3').rglob('*.py'))
        dependencies += [Path(cfg[k]) for k in ['source_checkpoint','stage1_checkpoint','control_checkpoint','event_config','manifest_source']]
        proof = {str(p): digest(p) for p in dependencies}
        path = out/'source_hashes.json'
        if path.exists():
            assert json.loads(path.read_text()) == proof, 'Dependencies changed; use a new output directory'
        else:
            atomic_json(path, proof)
    if world > 1:
        dist.barrier()
    if (out/'COMPLETE.json').exists():
        if rank == 0:
            print('Already complete; no training restarted', flush=True)
        if world > 1:
            dist.destroy_process_group()
        return
    extractor = PositionPriorExtractor(cfg).cuda().requires_grad_(False).eval()
    reader = JointTemporalReader(cfg, joint=True).cuda()
    # Keep DDP's trainable set fixed; optimizer_step suppresses head gradients during warmup.
    reader.set_epoch(cfg['adapter_warmup_epochs'] + 1)
    initial = {group: {key: value.detach().cpu().clone() for key, value in state.items()}
               for group, state in reader.learned_state().items()}
    optimizer = torch.optim.AdamW(reader.optimizer_groups(), weight_decay=cfg['weight_decay'])
    model = DistributedDataParallel(reader, device_ids=[local_rank], broadcast_buffers=False) if world > 1 else reader
    best = None
    baseline_thresholds = None
    updates = 0
    begin = 1
    if (out/'resume.pt').exists():
        state = torch.load(out/'resume.pt', weights_only=True, map_location='cpu')
        assert state['config'] == cfg
        reader.load_learned(state['learned'])
        optimizer.load_state_dict(state['optimizer'])
        best, baseline_thresholds = state['best'], state['baseline_thresholds']
        updates, begin = state['optimizer_updates'], state['epoch'] + 1
    records = manifest['splits']['train']
    targets = torch.tensor([r['labels'] for r in records], device='cuda', dtype=torch.float32)
    masks = torch.tensor([r['label_mask'] for r in records], device='cuda', dtype=torch.float32)
    weights = torch.tensor(cfg['pos_weight'], device='cuda')
    effective = world * cfg['micro_batch_size'] * cfg['accumulation_steps']
    steps_per_epoch = math.ceil(len(records) / effective)
    total_updates = steps_per_epoch * cfg['epochs']
    curves = {s: EventCurves(manifest['splits'][s], manifest, 2.) for s in ['calibration', 'development']}
    for epoch in range(begin, cfg['epochs'] + 1):
        torch.manual_seed(cfg['seed'] + epoch * 1000 + rank)
        order, local = shard_order(len(records), cfg['seed'] + epoch, world, rank, cfg['micro_batch_size'])
        microsteps = len(local) // cfg['micro_batch_size']
        model.train()
        optimizer.zero_grad(set_to_none=True)
        started = time.time()
        for step, (frames, ids) in enumerate(loader(records, local, cfg)):
            ids = ids.cuda(non_blocking=True)
            real = ids >= 0
            safe = ids.clamp_min(0)
            times = torch.tensor([records[i]['frame_times'] for i in safe.tolist()], device='cuda')
            batch = extract_batch(extractor, frames.cuda(non_blocking=True), times, cfg['frame_chunk'])
            batch['valid'] &= real[:, None, None]
            if epoch == 1 and step == 0:
                with torch.no_grad():
                    assert torch.allclose(reader(**batch)['logits'], batch['anchor'], atol=1e-6, rtol=0)
            drop_evidence(batch, cfg)
            group = step // cfg['accumulation_steps']
            group_ids = order[group*effective:(group+1)*effective]
            group_ids = torch.tensor(group_ids[group_ids >= 0], device='cuda')
            denominator = masks[group_ids].sum().clamp_min(1)
            count = max(len(group_ids), 1)
            sync = (step + 1) % cfg['accumulation_steps'] == 0 or step + 1 == microsteps
            context = model.no_sync() if world > 1 and not sync else nullcontext()
            with context:
                result = model(**paired_forward_batch(batch))
                b = len(ids)
                loss_mask = masks[safe] * real[:, None]
                task = (torch.nn.functional.binary_cross_entropy_with_logits(
                    result['logits'][:b], targets[safe], pos_weight=weights, reduction='none') * loss_mask).sum()
                residual = (result['delta'][:b].square().mean(1) * real).sum()
                valid_pair = real & real.roll(1, 0)
                wrong = (result['delta'][b:].square().mean(1) * valid_pair).sum()
                loss = world * (task / denominator + cfg['residual_l2_weight'] * residual / count
                                + cfg['corrupt_consistency_weight'] * wrong / count)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite online loss')
                loss.backward()
            if sync:
                optimizer_step(reader, optimizer, cfg, updates, total_updates)
                updates += 1
            if rank == 0 and (step == 0 or (step + 1) % 20 == 0):
                status = dict(phase='online_training', epoch=epoch, microstep=step+1,
                              microsteps=microsteps, optimizer_updates=updates,
                              temporal_updates_enabled=updates > cfg['warmup_optimizer_steps'],
                              elapsed_sec=time.time()-started, updated_unix=time.time())
                atomic_json(out/'heartbeat.json', status)
                print(json.dumps(status), flush=True)
        if rank == 0:
            atomic_json(out/'heartbeat.json', dict(phase='online_evaluation', epoch=epoch, updated_unix=time.time()))
        calibration = evaluate(extractor, reader, manifest['splits']['calibration'], cfg, rank, world)
        development = evaluate(extractor, reader, manifest['splits']['development'], cfg, rank, world, stress=True)
        if rank == 0:
            base_prob = probability(calibration['baseline_logits'])
            base_th, _ = curves['calibration'].tune(base_prob, cfg['recall_floors'])
            base_ap = window_ap(curves['calibration'], base_prob)
            prob = probability(calibration['logits'])
            thresholds, _ = curves['calibration'].tune(prob, cfg['recall_floors'])
            ap = window_ap(curves['calibration'], prob)
            if best is None:
                baseline_thresholds = base_th
                best = dict(epoch=0, enabled=False, score=base_ap['macro'], thresholds=base_th)
                save_torch(out/'best.pt', dict(epoch=0, learned=initial, best=best, config=cfg,
                                             baseline_thresholds=base_th, arm='joint_temporal', seed=cfg['seed']))
            assert base_th == baseline_thresholds, 'Original-model calibration changed between epochs'
            if ap['macro'] > best['score'] + 1e-8:
                best = dict(epoch=epoch, enabled=True, score=ap['macro'], thresholds=thresholds)
                save_torch(out/'best.pt', dict(epoch=epoch, learned=reader.learned_state(), best=best,
                                             config=cfg, baseline_thresholds=base_th, arm='joint_temporal', seed=cfg['seed']))
            summary = dict(
                epoch=epoch, optimizer_updates=updates, effective_batch_size=effective,
                calibration=metrics(calibration['logits'], curves['calibration'], thresholds),
                baseline_calibration=metrics(calibration['baseline_logits'], curves['calibration'], base_th),
                development=metrics(development['logits'], curves['development'], thresholds),
                development_original_thresholds=metrics(development['logits'], curves['development'], base_th),
                baseline_development=metrics(development['baseline_logits'], curves['development'], base_th),
                stress={key: metrics(development[key], curves['development'], base_th if key=='empty' else thresholds)
                        for key in ['empty','half_missing','reverse_local_time']},
                selected=best, elapsed_sec=time.time()-started,
                interpretation='Exploratory epoch metrics; select only on calibration, not development.')
            atomic_json(out/f'epoch_{epoch:03d}.json', summary)
            np.savez_compressed(out/f'calibration_epoch{epoch:03d}.npz', **calibration)
            np.savez_compressed(out/f'development_epoch{epoch:03d}.npz', **development)
            save_torch(out/'resume.pt', dict(epoch=epoch, learned=reader.learned_state(),
                       optimizer=optimizer.state_dict(), optimizer_updates=updates, config=cfg,
                       best=best, baseline_thresholds=baseline_thresholds))
            print(json.dumps(summary), flush=True)
        if world > 1:
            dist.barrier()
    if rank == 0:
        chosen = torch.load(out/'best.pt', weights_only=True, map_location='cpu')['best']
        selected_epoch = chosen['epoch'] or cfg['epochs']
        pred = np.load(out/f'development_epoch{selected_epoch:03d}.npz')
        logits = pred['logits'] if chosen['enabled'] else pred['baseline_logits']
        final = dict(selected=chosen,
                     development=metrics(logits, curves['development'], chosen['thresholds']),
                     baseline=metrics(pred['baseline_logits'], curves['development'], baseline_thresholds),
                     paired_video_bootstrap=paired_video_bootstrap(manifest['splits']['development'], curves['development'], logits, pred['baseline_logits']),
                     feature_cache_used=False, independent_test_claimed=False, natural_absence_verified=False,
                     interpretation='1–2 epoch screening only. No improvement here does not prove this route ineffective.')
        dev = final['development']['operating_metrics']
        base = final['baseline']['operating_metrics']
        interval = final['paired_video_bootstrap']['percentile_95CI_pp']
        final['development_checks'] = {
            'trained_model_selected': bool(chosen['enabled']),
            'macro_window_AP_improved': final['development']['window_AP']['macro'] > final['baseline']['window_AP']['macro'],
            'macro_precision_improved': dev['macro_precision'] > base['macro_precision'],
            'recall_and_false_positive_guard': all(dev[c]['recall'] >= base[c]['recall']-.01 and dev[c]['fp_windows_per_hour'] <= base[c]['fp_windows_per_hour'] for c in LABELS),
            'video_AP_interval_positive': interval is not None and interval[0] > 0,
        }
        final['screening_signal'] = ('positive_candidate_requires_reproduction'
                                     if all(final['development_checks'].values()) else 'not_supported_by_all_checks')
        atomic_json(out/'FINAL_SUMMARY.json', final)
        (out/'FINAL_REPORT.md').write_text(
            '# 在线 Stage2 快速验证\n\n'
            f"完成 {cfg['epochs']} epochs；校准选中 epoch {chosen['epoch']}。\n\n"
            f"筛查结论：{final['screening_signal']}。\n\n"
            f"开发窗口宏 AP：{final['baseline']['window_AP']['macro']*100:.3f}% → "
            f"{final['development']['window_AP']['macro']*100:.3f}%。\n\n"
            f"视频配对 AP 差 95% 区间（百分点）：{final['paired_video_bootstrap']['percentile_95CI_pp']}。\n\n"
            '未生成或读取特征缓存。逐类指标、原阈值结果和缺失/时间反转检查见 epoch JSON。\n'
            '这是单种子、历史开发集上的短训练筛查，不能证明自然无球鲁棒性或独立测试收益；'
            '两轮没有收益也不能据此断定路线无效。\n')
        atomic_json(out/'COMPLETE.json', dict(epochs=cfg['epochs'], feature_cache_used=False))
        atomic_json(out/'heartbeat.json', dict(phase='complete', updated_unix=time.time()))
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    manifest = preflight(cfg)
    if args.preflight:
        print(json.dumps(dict(preflight='passed', feature_cache_required=False,
                              split_windows={k:len(v) for k,v in manifest['splits'].items()},
                              epochs=cfg['epochs'], world_size=cfg['world_size'])))
        return
    train(cfg, manifest)


if __name__ == '__main__':
    main()
