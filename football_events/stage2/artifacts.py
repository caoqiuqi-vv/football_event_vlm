"""Atomic experiment artifacts and canonical window ordering."""
import hashlib
import json
from pathlib import Path
import torch

def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def save_torch(p, s):
    p = Path(p)
    tmp = p.with_suffix('.tmp.pt')
    torch.save(s, tmp)
    tmp.replace(p)

def unique_records(m):
    d = {r['key']: r for rows in m['splits'].values() for r in rows}
    return sorted(d.values(), key=lambda r: (r['video_id'], r['start_sec'], r['key']))
