"""Bounded timestamp sampling directly from high-resolution source videos."""
from __future__ import annotations

import re
import subprocess
from typing import Sequence

import numpy as np


def decode_timestamp_frames(
    path: str,
    indices: Sequence[int],
    fps: float,
    image_size: tuple[int, int],
    *,
    timeout_sec: float = 45.0,
) -> tuple[list[np.ndarray], list[float]]:
    """Select the first source frame at each requested time, retaining its PTS.

    FFmpeg scales the decoded source to the requested size using bicubic
    interpolation. No intermediate encoded video or alternate source is used.
    A subprocess timeout also bounds native decoder/seek stalls.
    """
    if not indices:
        return [], []
    unique = sorted(set(int(index) for index in indices))
    if fps <= 0 or unique[0] < 0:
        raise ValueError("invalid frame indices or fps")
    height, width = image_size
    if height <= 0 or width <= 0 or timeout_sec <= 0:
        raise ValueError("invalid output geometry or decode timeout")
    start = unique[0] / fps
    expression = "+".join(
        f"eq(selected_n,{position})*gte(t,{(index - unique[0]) / fps:.9f})"
        for position, index in enumerate(unique)
    )
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "info", "-threads", "2",
        "-ss", f"{start:.9f}", "-i", str(path), "-an", "-sn", "-dn",
        "-vf", f"select='{expression}',showinfo,scale={width}:{height}:flags=bicubic",
        "-filter_threads", "1", "-frames:v", str(len(unique)),
        "-fps_mode", "passthrough", "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True, timeout=timeout_sec)
    stderr = result.stderr.decode("utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError(f"ffmpeg decode failed ({result.returncode}): {stderr[-1200:]}")
    pts = [float(value) for value in re.findall(r"\bn:\s*\d+.*?pts_time:([\d.eE+-]+)", stderr)]
    expected_bytes = len(unique) * height * width * 3
    if len(result.stdout) != expected_bytes or len(pts) != len(unique):
        raise RuntimeError(
            f"ffmpeg incomplete decode: bytes={len(result.stdout)}/{expected_bytes} "
            f"timestamps={len(pts)}/{len(unique)}"
        )
    actual_times = [start + value for value in pts]
    # Refuse silent timestamp drift, including incorrect seeking on malformed media.
    tolerance = max(0.1, 2.0 / fps)
    if any(abs(actual - index / fps) > tolerance for actual, index in zip(actual_times, unique)):
        raise RuntimeError("ffmpeg sampled frames drifted from requested timestamps")
    pixels = np.frombuffer(result.stdout, dtype=np.uint8).reshape(len(unique), height, width, 3)
    positions = {index: position for position, index in enumerate(unique)}
    return (
        [pixels[positions[int(index)]].copy() for index in indices],
        [actual_times[positions[int(index)]] for index in indices],
    )
