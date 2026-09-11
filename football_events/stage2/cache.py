"""Resumable original-patch extraction, compaction and cache integrity checks."""
import fcntl
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from football_stage2_optional import WindowFrames
from .model import PositionPriorExtractor
from .artifacts import atomic_json, digest, save_torch, unique_records
SPECS = {
    'stage1_tokens': ((16, 26, 1024), 'float16'),
    'stage1_descriptors': ((16, 26, 11), 'float32'),
    'stage1_valid': ((16, 26), 'bool'),
    'stage1_patch_indices': ((16, 26), 'int16'),
    'selected_peaks': ((16, 2), 'int16'),
    'unweighted_peaks': ((16, 2), 'int16'),
}

def cache(out, cfg, manifest, rank):
    assert 0 <= rank < len(cfg['gpus'])
    lock = (out / f'cache_rank{rank}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    sha = digest(out / 'manifest.json')
    done = out / f'cache_rank{rank}_complete.json'
    if done.exists():
        assert json.loads(done.read_text())['manifest_sha256'] == sha
        return
    torch.set_num_threads(1)
    torch.manual_seed(42)
    records = unique_records(manifest)[rank::len(cfg['gpus'])]
    dest = out / 'cache'
    dest.mkdir(exist_ok=True)
    pending = []
    for r in records:
        path = dest / (r['key'] + '.pt')
        if not path.exists():
            pending.append(r)
            continue
        z = torch.load(path, weights_only=True, map_location='cpu')
        assert z['manifest_sha256'] == sha and z['key'] == r['key']
    if pending:
        old = Path(cfg['base_dir']) / 'arrays'
        mapping = {k: i for i, k in enumerate(json.loads((old / 'index.json').read_text())['keys'])}
        refs = {k: np.load(old / (k + '.npy'), mmap_mode='r') for k in ['global_features', 'stage1_tokens', 'stage1_descriptors']}
        model = PositionPriorExtractor(cfg).cuda().eval()
        kwargs = {'num_workers': cfg['workers'], 'pin_memory': True}
        if cfg['workers']:
            kwargs.update(prefetch_factor=1, multiprocessing_context='spawn')
        loader = DataLoader(WindowFrames(pending), batch_size=1, **kwargs)
        start = time.time()
        for step, (frames, index) in enumerate(loader, 1):
            r = pending[int(index[0])]
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                z = model.extract_clip(frames[0].cuda(non_blocking=True), cfg['cache_frame_chunk'])
            ix = mapping[r['key']]
            for actual, reference, dtype in [
                ('global_features', 'global_features', torch.float32),
                ('legacy_tokens', 'stage1_tokens', torch.float16),
                ('legacy_descriptors', 'stage1_descriptors', torch.float32),
            ]:
                assert torch.equal(z[actual].cpu().to(dtype), torch.from_numpy(np.array(refs[reference][ix], copy=True))), f"{r['key']}: {reference} changed"
            result = {k: z[k].detach().cpu().to(getattr(torch, dtype)) for k, (_, dtype) in SPECS.items()}
            assert all((torch.isfinite(v).all() for v in result.values()))
            result.update(key=r['key'], manifest_sha256=sha, token_source='original_event_dino')
            save_torch(dest / (r['key'] + '.pt'), result)
            if step == 1 or step % 20 == 0:
                status = {
                    'phase': 'new_original_patch_cache',
                    'rank': rank,
                    'completed_this_run': step,
                    'pending_at_start': len(pending),
                    'elapsed_sec': time.time() - start,
                    'updated_unix': time.time(),
                }
                atomic_json(out / f'cache_rank{rank}_heartbeat.json', status)
                print(json.dumps(status), flush=True)
    atomic_json(done, {'manifest_sha256': sha, 'assigned': len(records), 'complete': True})

def compact(out, cfg, manifest):
    lock = (out / 'compact.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    sha = digest(out / 'manifest.json')
    marker = out / 'COMPACT_COMPLETE.json'
    if marker.exists():
        assert json.loads(marker.read_text())['manifest_sha256'] == sha
        return
    records = unique_records(manifest)
    assert all(((out / 'cache' / (r['key'] + '.pt')).exists() for r in records)), 'Cache incomplete'
    dest = out / 'arrays'
    dest.mkdir(exist_ok=True)
    arrays = {k: np.lib.format.open_memmap(dest / (k + '.tmp.npy'), mode='w+', dtype=dtype, shape=(len(records), *shape)) for k, (shape, dtype) in SPECS.items()}
    changed = 0
    for i, r in enumerate(records):
        z = torch.load(out / 'cache' / (r['key'] + '.pt'), weights_only=True, map_location='cpu')
        assert z['manifest_sha256'] == sha and z['key'] == r['key'] and (z['token_source'] == 'original_event_dino')
        for k, a in arrays.items():
            a[i] = z[k].numpy()
        assert torch.equal(z['selected_peaks'][:, 0], z['unweighted_peaks'][:, 0])
        changed += int((z['selected_peaks'][:, 1] != z['unweighted_peaks'][:, 1]).sum())
    for k, a in arrays.items():
        a.flush()
        (dest / (k + '.tmp.npy')).replace(dest / (k + '.npy'))
    index = json.loads((Path(cfg['base_dir']) / 'arrays/index.json').read_text())
    assert index['keys'] == [r['key'] for r in records]
    atomic_json(dest / 'index.json', {**index, 'manifest_sha256': sha})
    atomic_json(marker, {
        'manifest_sha256': sha,
        'windows': len(records),
        'unrestricted_max_always_retained': True,
        'second_candidate_changed_frames': changed,
        'frame_slots': len(records) * 16,
        'arrays': {k: {'sha256': digest(dest / (k + '.npy')), 'shape': list(a.shape)} for k, a in arrays.items()},
    })

def validate_compacted_cache(out, manifest):
    marker = json.loads((out / 'COMPACT_COMPLETE.json').read_text())
    assert marker['manifest_sha256'] == digest(out / 'manifest.json')
    index = json.loads((out / 'arrays/index.json').read_text())
    assert index['keys'] == [r['key'] for r in unique_records(manifest)]
    for name, spec in marker['arrays'].items():
        path = out / 'arrays' / (name + '.npy')
        assert digest(path) == spec['sha256'], f'Cache hash mismatch: {path}'
        assert list(np.load(path, mmap_mode='r').shape) == spec['shape']
