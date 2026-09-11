"""Per-frame geometric evidence provider (ball/goal/keeper/crowd).

Loads the per-video evidence index built by ``scripts/build_evidence_index.py``
and returns, for any clip, a ``[T, 19]`` float tensor sampled at the clip's
absolute frame times.  Videos without an index (or frames beyond the lookup
tolerance) return zeroed rows with ``evidence_valid=0`` so the model can learn
to fall back to visual features only — detector absence is never treated as
"no ball".
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

EVIDENCE_FEATURE_DIM = 19


class EvidenceProvider:
    def __init__(
        self,
        index_dir: str | Path,
        *,
        feature_dim: int = EVIDENCE_FEATURE_DIM,
        max_time_gap_sec: float = 0.25,
        train_dropout: float = 0.0,
        is_train: bool = False,
        cache_size: int = 4,
        audio_index_dir: str | Path | None = None,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.audio_index_dir = Path(audio_index_dir) if audio_index_dir else None
        self.audio_dim = 5 if self.audio_index_dir is not None else 0
        self.feature_dim = int(feature_dim) + self.audio_dim
        self.max_time_gap_sec = float(max_time_gap_sec)
        self.train_dropout = float(train_dropout)
        self.is_train = bool(is_train)
        self.cache_size = max(int(cache_size), 1)
        self._cache: OrderedDict[str, tuple[np.ndarray, float, np.ndarray]] = (
            OrderedDict()
        )
        self._audio_cache: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = (
            OrderedDict()
        )
        self._lock = threading.Lock()
        self._rng = np.random.default_rng()

    @classmethod
    def from_config(cls, cfg: Any, *, is_train: bool) -> "EvidenceProvider | None":
        ev_cfg = cfg.get("model", {}).get("evidence_features", {})
        if not bool(ev_cfg.get("enabled", False)):
            return None
        index_dir = str(ev_cfg.get("index_dir", "") or "")
        if not index_dir:
            raise ValueError("model.evidence_features.enabled requires index_dir")
        return cls(
            index_dir,
            feature_dim=int(ev_cfg.get("feature_dim", EVIDENCE_FEATURE_DIM)),
            max_time_gap_sec=float(ev_cfg.get("max_time_gap_sec", 0.25)),
            train_dropout=(
                float(ev_cfg.get("train_dropout", 0.0)) if is_train else 0.0
            ),
            is_train=is_train,
            cache_size=int(ev_cfg.get("cache_size", 4)),
            audio_index_dir=ev_cfg.get("audio_index_dir", None) or None,
        )

    def _load(self, video_id: str) -> tuple[np.ndarray, float, np.ndarray] | None:
        with self._lock:
            hit = self._cache.get(video_id)
            if hit is not None:
                self._cache.move_to_end(video_id)
                return hit
        path = self.index_dir / f"{video_id}.npz"
        if not path.exists():
            return None
        with np.load(path) as payload:
            entry = (
                payload["frame_ids"].astype(np.int64),
                float(payload["fps"]),
                payload["feats"].astype(np.float32),
            )
        if entry[2].shape[1] != self.feature_dim - self.audio_dim:
            raise ValueError(
                f"evidence index {path} has dim {entry[2].shape[1]}, "
                f"expected {self.feature_dim - self.audio_dim}"
            )
        with self._lock:
            self._cache[video_id] = entry
            self._cache.move_to_end(video_id)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return entry

    def _load_audio(self, video_id: str) -> tuple[np.ndarray, np.ndarray] | None:
        with self._lock:
            hit = self._audio_cache.get(video_id)
            if hit is not None:
                self._audio_cache.move_to_end(video_id)
                return hit
        if self.audio_index_dir is None:
            return None
        path = self.audio_index_dir / f"{video_id}.npz"
        if not path.exists():
            return None
        with np.load(path) as payload:
            entry = (
                payload["times"].astype(np.float64),
                payload["feats"].astype(np.float32),
            )
        with self._lock:
            self._audio_cache[video_id] = entry
            self._audio_cache.move_to_end(video_id)
            while len(self._audio_cache) > self.cache_size:
                self._audio_cache.popitem(last=False)
        return entry

    def features(
        self, video_id: str, abs_times: Sequence[float]
    ) -> torch.Tensor:
        times = np.asarray(list(abs_times), dtype=np.float64)
        out = np.zeros((len(times), self.feature_dim), dtype=np.float32)
        entry = self._load(str(video_id))
        video_dim = self.feature_dim - self.audio_dim
        if entry is not None and entry[0].size:
            frame_ids, fps, feats = entry
            approx = times * fps
            pos = np.searchsorted(frame_ids, approx)
            pos = np.clip(pos, 0, len(frame_ids) - 1)
            prev = np.clip(pos - 1, 0, len(frame_ids) - 1)
            choose_prev = np.abs(frame_ids[prev] - approx) <= np.abs(frame_ids[pos] - approx)
            nearest = np.where(choose_prev, prev, pos)
            gap = np.abs(frame_ids[nearest] / fps - times)
            valid = gap <= self.max_time_gap_sec
            out[valid, :video_dim] = feats[nearest[valid]]
        if self.audio_index_dir is not None:
            audio = self._load_audio(str(video_id))
            if audio is not None and audio[0].size:
                audio_times, audio_feats = audio
                pos = np.searchsorted(audio_times, times)
                pos = np.clip(pos, 0, len(audio_times) - 1)
                prev = np.clip(pos - 1, 0, len(audio_times) - 1)
                choose_prev = (
                    np.abs(audio_times[prev] - times) <= np.abs(audio_times[pos] - times)
                )
                nearest = np.where(choose_prev, prev, pos)
                gap = np.abs(audio_times[nearest] - times)
                valid = gap <= max(self.max_time_gap_sec, 0.15)
                out[valid, video_dim:] = audio_feats[nearest[valid]]
        if self.is_train and self.train_dropout > 0.0:
            if self._rng.random() < self.train_dropout:
                out[:] = 0.0
        return torch.from_numpy(out)


def evidence_features_enabled(cfg: Any) -> bool:
    return bool(
        cfg.get("model", {}).get("evidence_features", {}).get("enabled", False)
    )


def evidence_feature_dim(cfg: Any) -> int:
    ev_cfg = cfg.get("model", {}).get("evidence_features", {})
    if not bool(ev_cfg.get("enabled", False)):
        return 0
    return int(ev_cfg.get("feature_dim", EVIDENCE_FEATURE_DIM))
