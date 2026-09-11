#!/usr/bin/env python
"""Rescore cached online predictions without another model forward."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import train_football_events as football
from football_online_evaluation import tune_online_event_thresholds


def per_video_robust_z(scores: np.ndarray, payload: np.lib.npyio.NpzFile) -> np.ndarray:
    """Per-video robust z-score on the logit scale, rescaled globally.

    Background score levels differ strongly across videos (camera, turf,
    broadcast style); a single global threshold lets "hot" videos dominate
    the FP budget.  Standardizing per video (median/MAD over that video's
    own windows -- >95% background) and mapping back through the global
    logit mean/std keeps one global threshold applicable.
    """
    eps = 1e-9
    logits = np.log(np.clip(scores, eps, 1.0 - eps)) - np.log(
        np.clip(1.0 - scores, eps, 1.0)
    )
    video_ids = payload["video_ids"]
    clip_starts = payload["clip_starts"].astype(np.float64)
    out = np.zeros_like(logits)
    for c in range(logits.shape[1]):
        z = np.zeros(len(logits), dtype=np.float64)
        for vid in {str(v) for v in video_ids}:
            idx = np.where(video_ids == vid)[0]
            idx = idx[np.argsort(clip_starts[idx])]
            values = logits[idx, c]
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median))) * 1.4826
            scale = mad if mad > 1e-6 else float(values.std() + 1e-3)
            z[idx] = (values - median) / scale
        out[:, c] = 1.0 / (
            1.0 + np.exp(-(z * logits[:, c].std() + logits[:, c].mean()))
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", type=Path)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shot-recall-floor", type=float, default=0.85)
    parser.add_argument("--save-recall-floor", type=float, default=None)
    parser.add_argument("--set-piece-recall-floor", type=float, default=None)
    parser.add_argument(
        "--score-transform",
        choices=["raw", "pvz"],
        default="raw",
        help="raw: cached probs as-is; pvz: per-video robust z-score on the "
        "logit scale (median/MAD per video+class), rescaled to the global "
        "logit distribution.  Use --probs-key online_probs to apply it to "
        "the fused score.",
    )
    parser.add_argument(
        "--probs-key",
        default="probs",
        help="NPZ array to rescore (probs | online_probs | frame_peak_probs)",
    )
    parser.add_argument("--nms-radius-sec", type=float, default=5.0)
    parser.add_argument("--tolerance-sec", type=float, default=5.0)
    parser.add_argument("--capped-clip-sec", type=float, default=10.0)
    parser.add_argument("--max-threshold-candidates", type=int, default=401)
    args = parser.parse_args()

    metadata_path = args.metadata or args.cache.with_suffix(".meta.json")
    payload = np.load(args.cache, allow_pickle=False)
    metas = json.loads(metadata_path.read_text(encoding="utf-8"))
    labels = [str(value) for value in payload["labels"].tolist()]
    if labels != ["shot", "save", "set_piece"]:
        raise ValueError(f"Expected set-piece schema, got labels={labels}")
    football.LABELS = labels
    football.LABEL_TO_INDEX = {label: index for index, label in enumerate(labels)}
    scores = payload[args.probs_key].astype(np.float64)
    if args.score_transform == "pvz":
        scores = per_video_robust_z(scores, payload)
    thresholds, tuning = tune_online_event_thresholds(
        scores,
        payload["candidate_times"],
        metas,
        labels,
        ["precision", "precision", "precision"],
        [
            float(args.shot_recall_floor),
            None if args.save_recall_floor is None else float(args.save_recall_floor),
            None
            if args.set_piece_recall_floor is None
            else float(args.set_piece_recall_floor),
        ],
        masks=payload["masks"] if "masks" in payload else None,
        nms_radius_sec=float(args.nms_radius_sec),
        tolerance_sec=float(args.tolerance_sec),
        max_candidates=int(args.max_threshold_candidates),
    )
    metrics = football.online_event_metrics(
        scores,
        payload["candidate_times"],
        metas,
        thresholds,
        masks=payload["masks"] if "masks" in payload else None,
        nms_radius_sec=float(args.nms_radius_sec),
        tolerance_sec=float(args.tolerance_sec),
        capped_clip_sec=float(args.capped_clip_sec),
    )
    metrics["score_transform"] = args.score_transform
    metrics["probs_key"] = args.probs_key
    metrics["threshold_tuning"] = tuning
    metrics["thresholds"] = {
        label: float(thresholds[index]) for index, label in enumerate(labels)
    }
    metrics["source_cache"] = str(args.cache)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(metrics, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
