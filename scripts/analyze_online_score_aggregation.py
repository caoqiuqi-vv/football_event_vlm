#!/usr/bin/env python
"""Offline analysis of dense online prediction caches.

For each cached val15 online NPZ, evaluate score-transform variants under the
strict PointNMS / 1:1 / tol=5s protocol and report, per class:
  - max precision subject to recall >= floor (user operating targets)
  - best F1 operating point
  - candidate recall ceiling (threshold -> 0)

Score transforms (all pure functions of cached per-window scores):
  clip         : raw clip probs
  fused        : clip probs * frame_peak_probs (training-time fusion)
  nbmax_aXX    : fused ** a * max(neighbor fused within +-stride) ** (1-a)
  nbmean_aXX   : fused ** a * mean(neighbor fused) ** (1-a)
  nbmax_clip   : same neighbor max applied to clip-only scores

Windows sit on a fixed stride grid per video, so neighbors are index +-1
after sorting by clip_start.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from football_online_evaluation import tune_online_event_thresholds

LABELS = ["shot", "save", "set_piece"]


def group_indices(video_ids: np.ndarray, clip_starts: np.ndarray) -> list[np.ndarray]:
    groups: dict[str, list[int]] = {}
    for index, vid in enumerate(video_ids):
        groups.setdefault(str(vid), []).append(index)
    ordered = []
    for vid, indices in groups.items():
        order = sorted(indices, key=lambda i: float(clip_starts[i]))
        ordered.append(np.asarray(order, dtype=np.int64))
    return ordered


def neighbor_stat(scores: np.ndarray, groups: list[np.ndarray], stat: str) -> np.ndarray:
    out = np.zeros_like(scores)
    for idx in groups:
        s = scores[idx]
        left = np.concatenate([s[:1], s[:-1]])
        right = np.concatenate([s[1:], s[-1:]])
        if stat == "max":
            m = np.maximum(left, right)
            # do not count self: edge windows keep their own value only if isolated
            m = np.where(np.arange(len(s)) > 0, m, right)
            m = np.where(np.arange(len(s)) < len(s) - 1, m, left)
        else:
            count = np.full(len(s), 2.0)
            count[0] = 1.0
            count[-1] = 1.0
            m = (left + right) / count
        out[idx] = m
    return out


def evaluate(scores: np.ndarray, candidate_times: np.ndarray, masks: np.ndarray,
             metas: list[dict], floors: dict[str, float], nms_radius: float,
             tolerance: float) -> dict:
    objectives = ["precision", "precision", "precision"]
    min_recalls = [floors[label] for label in LABELS]
    thresholds, tuning = tune_online_event_thresholds(
        scores, candidate_times, metas, LABELS, objectives, min_recalls, masks,
        nms_radius_sec=nms_radius, tolerance_sec=tolerance, max_candidates=801,
    )
    report = {}
    _, tuning_f1_all = tune_online_event_thresholds(
        scores, candidate_times, metas, LABELS, ["f1", "f1", "f1"],
        [None, None, None], masks,
        nms_radius_sec=nms_radius, tolerance_sec=tolerance, max_candidates=801,
    )
    for i, label in enumerate(LABELS):
        t = tuning.get(label, {})
        f1info = tuning_f1_all.get(label, {})
        report[label] = {
            "P_at_floor": t.get("precision"),
            "R_at_floor": t.get("recall"),
            "FP_at_floor": t.get("fp"),
            "threshold_at_floor": t.get("threshold"),
            "floor": floors[label],
            "floor_reachable": t.get("recall_floor_reachable"),
            "bestF1": f1info.get("f1"),
            "bestF1_P": f1info.get("precision"),
            "bestF1_R": f1info.get("recall"),
            "recall_ceiling": t.get("candidate_recall_ceiling"),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("caches", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--shot-floor", type=float, default=0.85)
    parser.add_argument("--save-floor", type=float, default=0.80)
    parser.add_argument("--set-piece-floor", type=float, default=0.80)
    parser.add_argument("--nms-radius-sec", type=float, default=5.0)
    parser.add_argument("--tolerance-sec", type=float, default=5.0)
    args = parser.parse_args()

    floors = {"shot": args.shot_floor, "save": args.save_floor,
              "set_piece": args.set_piece_floor}
    all_results = {}
    for cache in args.caches:
        payload = np.load(cache, allow_pickle=False)
        meta_path = cache.with_suffix(".meta.json")
        metas = json.loads(meta_path.read_text(encoding="utf-8"))
        probs = payload["probs"].astype(np.float64)
        if "frame_peak_probs" in payload:
            fused = probs * payload["frame_peak_probs"].astype(np.float64)
        else:
            fused = probs.copy()
        candidate_times = payload["candidate_times"].astype(np.float64)
        masks = payload["masks"].astype(np.float64) if "masks" in payload else None
        video_ids = payload["video_ids"]
        clip_starts = payload["clip_starts"].astype(np.float64)
        groups = group_indices(video_ids, clip_starts)

        variants: dict[str, np.ndarray] = {"clip": probs, "fused": fused}
        eps = 1e-9
        logit_fused = np.log(np.clip(fused, eps, 1.0 - eps)) - np.log(
            np.clip(1.0 - fused, eps, 1.0)
        )
        pvz = np.zeros_like(fused)
        for c in range(len(LABELS)):
            for idx in groups:
                values = logit_fused[idx, c]
                median = np.median(values)
                mad = np.median(np.abs(values - median)) * 1.4826
                scale = mad if mad > 1e-6 else (values.std() + 1e-3)
                pvz[idx, c] = (values - median) / scale
        # rescale per-video z-scores back to the global logit distribution so a
        # single global threshold still applies
        for c in range(len(LABELS)):
            gstd = logit_fused[:, c].std()
            gmean = logit_fused[:, c].mean()
            pvz[:, c] = 1.0 / (1.0 + np.exp(-(pvz[:, c] * gstd + gmean)))
        variants["fused_pvz"] = pvz
        for base_name, base in (("fused", fused), ("clip", probs)):
            for stat in ("max", "mean"):
                nb = np.stack(
                    [neighbor_stat(base[:, c], groups, stat) for c in range(len(LABELS))],
                    axis=1,
                )
                for alpha in (0.5, 0.7):
                    tag = f"{base_name}_nb{stat}_a{int(alpha*100)}"
                    variants[tag] = (base ** alpha) * (nb ** (1.0 - alpha))

        result = {}
        for name, scores in variants.items():
            result[name] = evaluate(scores, candidate_times, masks, metas, floors,
                                    args.nms_radius_sec, args.tolerance_sec)
        all_results[str(cache)] = result

        print(f"\n===== {cache} =====")
        header = f"{'variant':<22}" + "".join(
            f"{l+':P@R'+str(floors[l]):>18}{'R':>7}{'FP':>6}" for l in LABELS)
        print(header)
        for name, rep in result.items():
            row = f"{name:<22}"
            for label in LABELS:
                r = rep[label]
                p = r["P_at_floor"]
                row += (f"{p if p is not None else float('nan'):>18.4f}"
                        f"{r['R_at_floor'] if r['R_at_floor'] is not None else float('nan'):>7.3f}"
                        f"{r['FP_at_floor'] if r['FP_at_floor'] is not None else 0:>6}")
            print(row)
        print(f"{'':<22}" + "".join(
            f"  bestF1={rep[l]['bestF1']:.3f}/P={rep[l]['bestF1_P']:.3f}/R={rep[l]['bestF1_R']:.3f} ceil={rep[l]['recall_ceiling']:.3f}      "
            for l, rep in ((l, result['fused']) for l in LABELS)))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
