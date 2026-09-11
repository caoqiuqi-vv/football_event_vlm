#!/usr/bin/env python
"""Build a per-video audio feature index (whistle / crowd side-channel).

Motivation: XbotGo match videos carry stereo audio; set-piece anchors show
clearly elevated 2-4.5 kHz narrowband energy (whistle) versus background, and
crowd reaction follows shot/save outcomes.  These cues are orthogonal to the
2.4 fps visual stream (which physically cannot resolve ball flight).

Output ``<output_dir>/<video_id>.npz``:
  times : float32 [M]   seconds (5 Hz grid, hop 0.2 s)
  feats : float32 [M, 5]
    0 whistle_peakiness  1 log_energy  2 hf_band_ratio  3 energy_flux
    4 audio_valid
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np

SR = 16000
HOP = 3200          # 0.2 s
WIN = 4096          # 0.256 s analysis window


def load_audio(path: Path) -> np.ndarray | None:
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
         "-ar", str(SR), "-f", "f32le", "-"],
        capture_output=True,
    )
    if out.returncode != 0 or not out.stdout:
        return None
    return np.frombuffer(out.stdout, dtype=np.float32)


def audio_features(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_frames = max((len(x) - WIN) // HOP + 1, 0)
    if n_frames == 0:
        return np.zeros(0, dtype=np.float32), np.zeros((0, 5), dtype=np.float32)
    window = np.hanning(WIN).astype(np.float32)
    freqs = np.fft.rfftfreq(WIN, 1.0 / SR)
    whistle_band = (freqs >= 2000) & (freqs <= 4500)
    low_band = (freqs >= 500) & (freqs < 2000)
    rumble_band = (freqs >= 60) & (freqs < 500)
    times = (np.arange(n_frames) * HOP + WIN // 2) / SR
    feats = np.zeros((n_frames, 5), dtype=np.float32)
    prev_energy = 0.0
    for i in range(n_frames):
        seg = x[i * HOP: i * HOP + WIN] * window
        power = np.abs(np.fft.rfft(seg)) ** 2
        total = float(power.sum()) + 1e-12
        wp = power[whistle_band]
        peak = float(wp.max()) / (float(np.median(wp)) + 1e-9)
        band_vs_low = float(wp.sum()) / (float(power[low_band].sum()) + 1e-9)
        energy = float(np.log1p(total))
        flux = max(energy - prev_energy, 0.0)
        prev_energy = energy
        feats[i] = (
            np.log1p(peak) + np.log1p(band_vs_low),
            energy,
            float(power[whistle_band].sum()) / total,
            flux,
            1.0,
        )
    # robust per-video normalization (leave valid flag untouched)
    for dim in range(4):
        col = feats[:, dim]
        med = np.median(col)
        scale = np.percentile(np.abs(col - med), 95) + 1e-6
        feats[:, dim] = np.clip((col - med) / scale, -4.0, 4.0)
    return times.astype(np.float32), feats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video-ids", type=Path, default=None)
    parser.add_argument(
        "--path-override", action="append", default=[],
        help="video_id=/abs/path.mp4 for videos living outside --video-dir",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overrides = dict(
        item.split("=", 1) for item in args.path_override
    )
    if args.video_ids:
        video_ids = [l.strip() for l in args.video_ids.read_text().splitlines() if l.strip()]
    else:
        video_ids = sorted(p.stem for p in args.video_dir.glob("*.mp4"))
    done = failed = 0
    for vid in video_ids:
        dst = args.output_dir / f"{vid}.npz"
        src = Path(overrides.get(vid, args.video_dir / f"{vid}.mp4"))
        if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
            done += 1
            continue
        audio = load_audio(src)
        if audio is None or len(audio) < SR * 30:
            failed += 1
            print(f"  audio extract failed: {vid}")
            continue
        times, feats = audio_features(audio)
        np.savez_compressed(dst, times=times, feats=feats)
        done += 1
    print(f"audio index: wrote/kept {done}, failed {failed}, dir={args.output_dir}")


if __name__ == "__main__":
    main()
