#!/usr/bin/env python3
"""Reject runtime artifacts/secrets and large files in the Git index before publishing."""
import fnmatch
import re
import subprocess
import sys

FORBIDDEN_DIRS = {'outputs', 'checkpoints', 'weights', 'pretrained', 'datasets', 'logs',
                  'wandb', 'runs', '__pycache__', '.cache', 'node_modules', '.venv',
                  'venv', 'archive', 'experiments', '.agents', '.codex', '.claude'}
FORBIDDEN_SUFFIXES = ('.pt', '.pth', '.safetensors', '.ckpt', '.onnx', '.engine', '.bin',
                      '.npy', '.npz', '.pkl', '.pickle', '.h5', '.hdf5', '.parquet', '.arrow',
                      '.mp4', '.mov', '.mkv', '.avi', '.webm', '.wav', '.mp3', '.flac',
                      '.jsonl', '.csv', '.tsv', '.log', '.pid', '.pem', '.key', '.crt', '.p12', '.pfx')
SECRET = re.compile(rb'(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')

def main():
    records = subprocess.check_output(['git', 'ls-files', '--stage', '-z']).split(b'\0')
    errors = []; count = total = 0
    for record in records:
        if not record: continue
        head, name = record.split(b'\t', 1)
        mode, oid, stage = head.decode().split()
        path = name.decode(); parts = path.split('/'); lower = path.lower()
        if stage != '0': errors.append(f'{path}: unresolved merge'); continue
        if any(p in FORBIDDEN_DIRS for p in parts) or lower.endswith(FORBIDDEN_SUFFIXES) or re.search(r'\.(?:sqlite\d*|db)(?:-|$)', lower):
            errors.append(f'{path}: generated data/model/runtime artifact')
        if parts[-1] in ('access_config.json', 'config.yaml') or fnmatch.fnmatch(parts[-1], '*credentials*.json') or (parts[-1].startswith('.env') and parts[-1] != '.env.example'):
            errors.append(f'{path}: private configuration')
        data = subprocess.check_output(['git', 'cat-file', 'blob', oid])
        count += 1; total += len(data)
        if len(data) > 2 * 1024 * 1024: errors.append(f'{path}: exceeds 2 MiB; review before tracking')
        if SECRET.search(data): errors.append(f'{path}: possible embedded secret (value suppressed)')
        if mode == '120000' and (data.startswith(b'/') or b'outputs' in data.split(b'/')):
            errors.append(f'{path}: machine-specific symlink')
    print(f'Index: {count} files, {total / 1024 / 1024:.2f} MiB')
    for error in errors: print(error, file=sys.stderr)
    return bool(errors)

if __name__ == '__main__':
    raise SystemExit(main())
