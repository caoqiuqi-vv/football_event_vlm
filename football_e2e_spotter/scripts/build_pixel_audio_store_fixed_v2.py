#!/usr/bin/env python
"""Tail-safe pixel/audio cache builder (importlib-compatible entrypoint)."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import librosa
import numpy as np
import torch


ORIGINAL = Path(__file__).with_name("build_pixel_audio_store.py")
SPEC = importlib.util.spec_from_file_location("football_pixel_audio_builder_base", ORIGINAL)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load {ORIGINAL}")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)


def extract_log_mel_fixed(
    video: Path, *, frame_count: int, sample_fps: float,
    sample_rate: int, mel_bins: int,
) -> np.ndarray:
    samples_per_step = int(round(sample_rate / sample_fps))
    process = subprocess.Popen([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
        "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "s16le", "pipe:1",
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdout is None:
        raise RuntimeError("ffmpeg did not expose stdout")
    mel_filter = torch.from_numpy(librosa.filters.mel(
        sr=sample_rate, n_fft=512, n_mels=mel_bins, fmin=40.0,
        fmax=min(7600.0, 0.5 * sample_rate),
    ).astype(np.float32))
    window = torch.hann_window(512)
    rows: list[np.ndarray] = []
    try:
        for start in range(0, frame_count, 128):
            count = min(128, frame_count - start)
            raw = base.read_exact(process.stdout, count * samples_per_step * 2)
            samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            expected = count * samples_per_step
            if samples.size < expected:
                samples = np.pad(samples, (0, expected - samples.size))
            values = torch.from_numpy(samples[:expected].reshape(count, samples_per_step))
            spectrum = torch.stft(
                values, n_fft=512, hop_length=160, win_length=512,
                window=window, center=True, return_complex=True,
            ).abs().square()
            mel = torch.matmul(mel_filter, spectrum).mean(dim=-1)
            rows.append(torch.log1p(mel).numpy().astype(np.float16))
        while process.stdout.read(1024 * 1024):
            pass
    finally:
        process.stdout.close()
    stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"ffmpeg audio decode failed ({return_code}): {stderr[-1000:]}")
    output = np.concatenate(rows, axis=0) if rows else np.zeros((0, mel_bins), np.float16)
    if output.shape != (frame_count, mel_bins):
        raise RuntimeError(f"audio feature shape mismatch: {output.shape}")
    return output


if __name__ == "__main__":
    base.extract_log_mel = extract_log_mel_fixed
    base.main()
