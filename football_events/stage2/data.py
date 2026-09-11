"""Memory-mapped local/global features with one CPU prefetch thread.

The public Bank preserves the joint-training cache contract, including original
window ordering, timestamps, ROI compatibility, and explicit validity masks.
"""
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from football_stage2_corepatch import legacy_descriptors

class Bank:

    def __init__(self, out, cfg, m, arm):
        out = Path(cfg['core_cache_dir'])
        self.arm = cfg['arms'][arm]
        old = Path(cfg['base_dir']) / 'arrays'
        index = json.loads((old / 'index.json').read_text())
        mapping = {k: i for i, k in enumerate(index['keys'])}
        self.groups = {s: np.array([mapping[r['key']] for r in rows]) for s, rows in m['splits'].items()}
        self.times = np.array(index['frame_times'], np.float32)
        self.arrays = {k: np.load(old / (k + '.npy'), mmap_mode='r') for k in ['global_features', 'anchor']}
        name = self.arm['source']
        path = old if self.arm['representation'] == 'roi' else out / 'arrays'
        for k in ['tokens', 'descriptors']:
            self.arrays[k] = np.load(path / f'{name}_{k}.npy', mmap_mode='r')
        self.valid = None if self.arm['representation'] == 'roi' else np.load(path / f'{name}_valid.npy', mmap_mode='r')

    def cpu(self, indices):
        b = {k: torch.from_numpy(np.array(a[indices], copy=True)) for k, a in self.arrays.items()}
        if self.arm['representation'] == 'roi':
            b['descriptors'] = legacy_descriptors(b['descriptors'])
        b['valid'] = torch.ones(b['tokens'].shape[:-1], dtype=torch.bool) if self.valid is None else torch.from_numpy(np.array(self.valid[indices], copy=True))
        b['frame_times'] = torch.from_numpy(self.times[indices].copy())
        return {k: v.pin_memory() for k, v in b.items()}

    def batches(self, indices, size):
        slices = [indices[i:i + size] for i in range(0, len(indices), size)]
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(self.cpu, slices[0]) if slices else None
            for i in range(len(slices)):
                b = pending.result()
                if i + 1 < len(slices):
                    pending = pool.submit(self.cpu, slices[i + 1])
                yield {k: v.cuda(non_blocking=True) for k, v in b.items()}
