"""Experiment preparation, read-only preflight and explicit process orchestration.

Default execution is a preflight; only explicit pipeline/cache/train phases
start work. Historical experiment supervisors are never adopted or resumed.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
from .artifacts import atomic_json, digest, unique_records
from .cache import SPECS, cache, compact, validate_compacted_cache
from .training import train
from .reporting import report
ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'outputs/football_localization_stage2/720p_originalpatch_softprior_joint_20260909'
DEPENDENCIES = [
    'football_stage2_position_prior.py',
    'football_stage2_joint.py',
    'football_stage2_corepatch.py',
    'football_stage2_optional.py',
    'football_stage2_metrics.py',
    'football_localization_full.py',
    'scripts/run_football_stage2_position_prior.py',
    'scripts/train_football_localization_stage1.py',
    'train_football_events.py',
]
DEPENDENCIES += sorted((str(p.relative_to(ROOT)) for p in (ROOT / 'football_events').rglob('*.py')))

def prepare(out, config_file):
    cfg = json.loads(config_file.read_text())
    assert Path(cfg['output_dir']) == out and Path(cfg['core_cache_dir']) == out
    base = Path(cfg['base_dir'])
    manifest = json.loads((base / 'manifest.json').read_text())
    manifest['config'] = cfg
    out.mkdir(parents=True, exist_ok=True)
    for name, value in [('config.json', cfg), ('manifest.json', manifest)]:
        path = out / name
        if path.exists():
            assert json.loads(path.read_text()) == value, f'Refusing to overwrite {path}'
        else:
            atomic_json(path, value)
    files = [ROOT / p for p in DEPENDENCIES]
    files += list((ROOT / 'dinov3').rglob('*.py'))
    files += [out / 'config.json', out / 'manifest.json', Path(cfg['event_config'])]
    files += [Path(cfg[k]) for k in ['source_checkpoint', 'stage1_checkpoint', 'control_checkpoint']]
    files += [base / 'arrays/index.json', base / 'arrays/global_features.npy', base / 'arrays/anchor.npy']
    proof = {
        'sha256': {str(p.resolve()): digest(p) for p in sorted(set(files))},
        'token_source': 'original event DINO; Stage1 used only for heatmap guidance',
        'formal_training_started': False,
    }
    path = out / 'run_provenance.json'
    if path.exists():
        assert json.loads(path.read_text()) == proof, 'Prepared dependencies changed; use a new output directory'
    else:
        atomic_json(path, proof)
        snap = out / 'code_snapshot'
        for name in DEPENDENCIES:
            dest = snap / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dest)
    print(json.dumps({'prepared': str(out), 'training_started': False}), flush=True)

def integrity(out):
    proof = json.loads((out / 'run_provenance.json').read_text())
    for path, sha in proof['sha256'].items():
        if digest(path) != sha:
            raise RuntimeError(f'Pinned dependency changed: {path}')

def preflight(out, cfg, manifest):
    integrity(out)
    records = unique_records(manifest)
    index = json.loads((Path(cfg['base_dir']) / 'arrays/index.json').read_text())
    assert index['keys'] == [r['key'] for r in records], 'Cache/window ordering mismatch'
    assert cfg['num_frames'] == 16 and cfg['image_size'] == [720, 1280]
    assert all((len(r['frame_indices']) == 16 for r in records))
    assert cfg['arms'] == {'joint_temporal': {'representation': 'core', 'source': 'stage1', 'fusion': 'temporal', 'joint': True}}
    n = len(records)
    size = sum((np.prod(shape) * np.dtype(dtype).itemsize for shape, dtype in SPECS.values()))
    committed = sum(((out / 'cache' / (r['key'] + '.pt')).exists() for r in records))
    required = 0 if (out / 'COMPACT_COMPLETE.json').exists() else (2 * n - committed) * size + 5 * 1024 ** 3
    free = shutil.disk_usage(out).free
    assert free > required, f'Need ~{required / 1024 ** 3:.1f} GiB free; have {free / 1024 ** 3:.1f}'
    print(json.dumps({
        'preflight': 'passed',
        'windows': n,
        'committed': committed,
        'additional_disk_GiB_estimate': float(required / 1024 ** 3),
        'free_GiB': free / 1024 ** 3,
        'training_started': False,
    }), flush=True)

def pipeline(out, cfg, manifest, seed):
    lock = (out / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    preflight(out, cfg, manifest)
    children = []
    logs = []

    def status(phase, **extra):
        atomic_json(out / 'pipeline_status.json', {'phase': phase, 'pid': os.getpid(), 'updated_unix': time.time(), **extra})

    def stop(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def free(gpus):
        raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
        available = {int(line.split(',')[0]) for line in raw.splitlines() if int(line.split(',')[1]) < 256}
        if not set(gpus) <= available:
            raise RuntimeError(f'Requested GPUs busy: {sorted(set(gpus) - available)}. No unrelated job stopped.')

    def launch(phase, gpu, extra, tag):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
        log = (out / (tag + '.log')).open('a')
        logs.append(log)
        child = subprocess.Popen([
            sys.executable,
            str(ROOT / 'scripts/run_football_stage2_position_prior.py'),
            '--output-dir',
            str(out),
            '--phase',
            phase,
            *extra,
        ], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(child)
        return child

    def wait_all():
        while any((p.poll() is None for p in children)):
            failed = [p.pid for p in children if p.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f'Worker failed: {failed}; see worker logs')
            time.sleep(2)
        assert all((p.returncode == 0 for p in children)), 'Worker failed; see logs'
        children.clear()
    try:
        if not (out / 'COMPACT_COMPLETE.json').exists():
            ranks = [r for r in range(len(cfg['gpus'])) if not (out / f'cache_rank{r}_complete.json').exists()]
            free([cfg['gpus'][r] for r in ranks])
            for rank in ranks:
                launch('cache', cfg['gpus'][rank], ['--rank', str(rank)], f'cache_rank{rank}')
            status('cache', children=[p.pid for p in children])
            wait_all()
            status('compact')
            compact(out, cfg, manifest)
        integrity(out)
        if not (out / f'joint_temporal_seed{seed}/COMPLETE.json').exists():
            gpu = cfg['gpus'][0]
            free([gpu])
            p = launch('train', gpu, ['--seed', str(seed)], f'joint_temporal_seed{seed}')
            status('training', child_pid=p.pid, seed=seed, gpu=gpu)
            wait_all()
        integrity(out)
        status('report')
        report(out, cfg, manifest, seed)
        status('complete', seed=seed, report=str(out / f'FINAL_REPORT_seed{seed}.md'))
    except BaseException as error:
        for p in children:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for p in children:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
        status('paused' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error), automatic_restart=False)
        raise
    finally:
        for log in logs:
            log.close()

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', type=Path, default=DEFAULT)
    ap.add_argument('--config', type=Path, default=ROOT / 'configs/football/stage2_originalpatch_softprior_joint_20260909.json')
    ap.add_argument('--phase', choices=['prepare', 'preflight', 'pipeline', 'cache', 'compact', 'train', 'report'], default='preflight')
    ap.add_argument('--rank', type=int, default=0)
    ap.add_argument('--seed', type=int, choices=[42, 43], default=42)
    a = ap.parse_args()
    out = a.output_dir.resolve()
    if a.phase == 'prepare':
        prepare(out, a.config)
        return
    cfg = json.loads((out / 'config.json').read_text())
    manifest = json.loads((out / 'manifest.json').read_text())
    assert cfg == manifest['config'] and Path(cfg['output_dir']) == out and (Path(cfg['core_cache_dir']) == out)
    if a.phase == 'preflight':
        preflight(out, cfg, manifest)
    elif a.phase == 'pipeline':
        pipeline(out, cfg, manifest, a.seed)
    elif a.phase == 'cache':
        cache(out, cfg, manifest, a.rank)
    elif a.phase == 'compact':
        compact(out, cfg, manifest)
    elif a.phase == 'train':
        integrity(out)
        validate_compacted_cache(out, manifest)
        train(out, cfg, manifest, 'joint_temporal', a.seed)
    else:
        report(out, cfg, manifest, a.seed)
