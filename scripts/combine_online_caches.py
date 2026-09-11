#!/usr/bin/env python
"""Combine multiple online prediction caches (same model, different video sets)
into one NPZ+meta so downstream tuning/analysis sees the union (e.g. Val15 +
extra10 = Val25)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ARRAY_KEYS = [
    "logits", "probs", "frame_peak_probs", "online_probs", "targets", "masks",
    "candidate_times", "video_ids", "sample_ids", "clip_starts", "clip_ends",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("caches", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if len(args.caches) < 2:
        raise SystemExit("need at least two caches")

    merged: dict[str, list[np.ndarray]] = {k: [] for k in ARRAY_KEYS}
    metas: list[dict] = []
    labels = None
    for cache in args.caches:
        payload = np.load(cache, allow_pickle=False)
        if labels is None:
            labels = payload["labels"]
        elif not np.array_equal(labels, payload["labels"]):
            raise ValueError(f"label mismatch in {cache}")
        for key in ARRAY_KEYS:
            if key not in payload:
                raise ValueError(f"{cache} missing key {key}")
            merged[key].append(payload[key])
        metas.extend(json.loads(cache.with_suffix(".meta.json").read_text(encoding="utf-8")))
    out = {key: np.concatenate(values) for key, values in merged.items()}
    out["labels"] = labels
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **out)
    args.output.with_suffix(".meta.json").write_text(
        json.dumps(metas, ensure_ascii=False), encoding="utf-8"
    )
    print(f"combined {len(args.caches)} caches -> {args.output} rows={len(metas)}")


if __name__ == "__main__":
    main()
