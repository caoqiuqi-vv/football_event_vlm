#!/usr/bin/env python
"""Export review thumbnails for mined dense hard negatives.

For quick human verification of a hard-negative manifest before/while it is
used in training: dump a small JPG grid (few frames per window) for the
top-scoring mined windows per class.  CPU-only; direct cv2 seeks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LABELS = ["shot", "save", "set_piece"]


def grab_frame(path: str, sec: float) -> np.ndarray | None:
    cap = cv2.VideoCapture(path)
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(sec, 0.0) * 1000.0)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top-per-class", type=int, default=120)
    parser.add_argument("--thumb-width", type=int, default=448)
    args = parser.parse_args()

    import train_football_events as base

    cfg = base.load_config(args.config, [])
    roots = cfg["data"]["long_video"]["roots"]
    video_dirs = {r["source"]: r["videos_dir"] for r in roots}
    overrides = {}
    for r in roots:
        overrides.update(r.get("video_path_overrides", {}) or {})

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = manifest["entries"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for label in LABELS:
        picked = [
            (max(e["scores"].get(label, 0.0) for e in [e]), e)
            for e in entries
            if label in e["mined_for"]
        ]
        picked = sorted(
            ((e["scores"][label], e) for _, e in picked),
            key=lambda item: item[0], reverse=True,
        )[: args.top_per_class]
        out_dir = args.output_dir / label
        out_dir.mkdir(exist_ok=True)
        written = 0
        for score, entry in picked:
            vid = entry["video_id"]
            path = overrides.get(vid) or str(Path(video_dirs[entry["source"]]) / f"{vid}.mp4")
            if not Path(path).exists():
                continue
            mid = 0.5 * (float(entry["start"]) + float(entry["end"]))
            times = [float(entry["start"]) + 2.0, mid, float(entry["end"]) - 2.0]
            frames = [grab_frame(path, t) for t in times]
            frames = [f for f in frames if f is not None]
            if not frames:
                continue
            ratio = args.thumb_width / frames[0].shape[1]
            thumbs = [
                cv2.resize(f, (args.thumb_width, int(f.shape[0] * ratio)))
                for f in frames
            ]
            strip = np.concatenate(thumbs, axis=1)
            name = f"{score:.3f}_{vid}_{int(entry['start'])}s.jpg"
            cv2.imwrite(str(out_dir / name), strip)
            written += 1
        print(f"{label}: wrote {written} review strips to {out_dir}")


if __name__ == "__main__":
    main()
