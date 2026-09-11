#!/usr/bin/env python
"""Temporal score rescoring over dense online prediction caches.

A lightweight 1D conv net re-scores each window using the score *time series*
of the whole video (all three classes), so it can learn structures the
per-window heads cannot express: shot peaks followed by save peaks, true
events persisting across overlapping windows vs isolated FP spikes, and the
wider temporal footprint of set pieces.

Train split usage only: the model is trained on the dense train-split cache
produced by ``mine_dense_hard_negatives.py predict`` and validated on a val15
online cache.  It predicts a per-class logit residual on top of the fused
online score, so at init it is exactly the identity.

Eval protocol: strict PointNMS / 1:1 / tol=5s via tune_online_event_thresholds
with the product operating floors (shot R>=0.85, save/set_piece R>=0.80,
objective precision) — thresholds are tuned on the val cache only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LABELS = ["shot", "save", "set_piece"]


def logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 1e-6, 1.0 - 1e-6)
    return np.log(x) - np.log1p(-x)


def load_cache(cache: Path):
    payload = np.load(cache, allow_pickle=False)
    metas = json.loads(cache.with_suffix(".meta.json").read_text(encoding="utf-8"))
    return payload, metas


def build_sequences(payload, metas, tolerance: float):
    """Group windows into per-video sequences aligned by clip_start."""
    keys = np.asarray(
        [f"{m.get('source','')}|{m.get('video_id','')}" for m in metas]
    )
    starts = payload["clip_starts"].astype(np.float64)
    fused = payload["online_probs"].astype(np.float64)
    clip = payload["probs"].astype(np.float64)
    frame = payload["frame_peak_probs"].astype(np.float64)
    candidate_times = payload["candidate_times"].astype(np.float64)
    masks = payload["masks"].astype(np.float64)

    seqs = []
    for key in sorted(set(keys.tolist())):
        idx = np.where(keys == key)[0]
        idx = idx[np.argsort(starts[idx])]
        anchors = [set() for _ in LABELS]
        for i in idx:
            gt = metas[i].get("online_gt_anchors", ()) or ()
            if len(gt) == len(LABELS):
                for c in range(len(LABELS)):
                    anchors[c].update(float(v) for v in gt[c])
        # target: candidate peak within tolerance of a same-class anchor
        targets = np.zeros((len(idx), len(LABELS)), dtype=np.float64)
        for c in range(len(LABELS)):
            if not anchors[c]:
                continue
            arr = np.asarray(sorted(anchors[c]))
            for j, i in enumerate(idx):
                if np.min(np.abs(arr - candidate_times[i, c])) <= tolerance:
                    targets[j, c] = 1.0
        features = np.stack(
            [logit(fused[idx]), logit(clip[idx]), logit(frame[idx])], axis=-1
        )  # [L, C, 3]
        seqs.append(
            {
                "key": key,
                "features": features.reshape(len(idx), -1),  # [L, C*3]
                "base_logit": logit(fused[idx]),  # [L, C]
                "targets": targets,
                "masks": masks[idx],
                "indices": idx,
            }
        )
    return seqs


class Rescorer:
    def __init__(self, in_dim: int, num_classes: int, hidden: int = 32):
        import torch

        self.torch = torch
        self.model = torch.nn.Sequential(
            torch.nn.Conv1d(in_dim, hidden, kernel_size=5, dilation=1, padding=2),
            torch.nn.GELU(),
            torch.nn.Conv1d(hidden, hidden, kernel_size=5, dilation=2, padding=4),
            torch.nn.GELU(),
            torch.nn.Conv1d(hidden, num_classes, kernel_size=5, dilation=4, padding=8),
        )
        # zero-init the last layer: exact identity at start
        torch.nn.init.zeros_(self.model[-1].weight)
        torch.nn.init.zeros_(self.model[-1].bias)

    def fit(self, seqs, *, device: str, epochs: int, lr: float, pos_weight_cap: float):
        torch = self.torch
        self.model.to(device)
        counts = np.zeros((2, len(LABELS)))
        for seq in seqs:
            for c in range(len(LABELS)):
                valid = seq["masks"][:, c] > 0.5
                counts[0, c] += float((valid & (seq["targets"][:, c] <= 0.5)).sum())
                counts[1, c] += float((valid & (seq["targets"][:, c] > 0.5)).sum())
        pos_weight = np.minimum(counts[0] / np.maximum(counts[1], 1.0), pos_weight_cap)
        print(f"train positives per class: {counts[1].astype(int).tolist()}, "
              f"pos_weight={np.round(pos_weight, 1).tolist()}")
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        pw = torch.tensor(pos_weight, dtype=torch.float32, device=device)
        for epoch in range(int(epochs)):
            order = np.random.permutation(len(seqs))
            total_loss = 0.0
            for si in order:
                seq = seqs[si]
                x = torch.tensor(
                    seq["features"], dtype=torch.float32, device=device
                ).permute(1, 0).unsqueeze(0)
                delta = self.model(x).squeeze(0).permute(1, 0)
                base = torch.tensor(
                    seq["base_logit"], dtype=torch.float32, device=device
                )
                logits = base + delta
                t = torch.tensor(seq["targets"], dtype=torch.float32, device=device)
                m = torch.tensor(seq["masks"], dtype=torch.float32, device=device)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, t, reduction="none", pos_weight=pw.unsqueeze(0)
                )
                loss = (loss * m).sum() / m.sum().clamp_min(1.0)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += float(loss.detach().cpu())
            if (epoch + 1) % max(epochs // 4, 1) == 0 or epoch == 0:
                print(f"  epoch {epoch + 1}/{epochs} loss={total_loss / len(seqs):.4f}")

    def predict_logits(self, seq, device: str) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            x = torch.tensor(
                seq["features"], dtype=torch.float32, device=device
            ).permute(1, 0).unsqueeze(0)
            delta = self.model(x).squeeze(0).permute(1, 0).cpu().numpy()
        return seq["base_logit"] + delta


def evaluate_scores(scores: np.ndarray, candidate_times: np.ndarray, metas,
                    masks: np.ndarray, nms_radius: float, tolerance: float,
                    floors: dict) -> dict:
    from football_online_evaluation import tune_online_event_thresholds

    _, tuning = tune_online_event_thresholds(
        scores, candidate_times, metas, LABELS, ["precision"] * 3,
        [floors[l] for l in LABELS], masks,
        nms_radius_sec=nms_radius, tolerance_sec=tolerance, max_candidates=801,
    )
    return {
        label: {
            "P": tuning[label]["precision"],
            "R": tuning[label]["recall"],
            "FP": tuning[label]["fp"],
            "ceiling": tuning[label]["candidate_recall_ceiling"],
        }
        for label in LABELS
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", required=True, type=Path)
    parser.add_argument("--val-cache", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--pos-weight-cap", type=float, default=20.0)
    parser.add_argument("--nms-radius-sec", type=float, default=5.0)
    parser.add_argument("--tolerance-sec", type=float, default=5.0)
    args = parser.parse_args()

    floors = {"shot": 0.85, "save": 0.80, "set_piece": 0.80}
    train_payload, train_metas = load_cache(args.train_cache)
    val_payload, val_metas = load_cache(args.val_cache)
    train_seqs = build_sequences(train_payload, train_metas, args.tolerance_sec)
    val_seqs = build_sequences(val_payload, val_metas, args.tolerance_sec)
    print(f"train videos={len(train_seqs)} val videos={len(val_seqs)}")

    rescorer = Rescorer(in_dim=9, num_classes=len(LABELS), hidden=args.hidden)
    rescorer.fit(train_seqs, device=args.device, epochs=args.epochs, lr=args.lr,
                 pos_weight_cap=args.pos_weight_cap)

    # re-score the val cache
    new_scores = np.zeros_like(val_payload["online_probs"], dtype=np.float64)
    for seq in val_seqs:
        logits = rescorer.predict_logits(seq, args.device)
        new_scores[seq["indices"]] = 1.0 / (1.0 + np.exp(-logits))

    val_masks = val_payload["masks"].astype(np.float64)
    candidate_times = val_payload["candidate_times"].astype(np.float64)
    report = {
        "baseline_fused": evaluate_scores(
            val_payload["online_probs"].astype(np.float64), candidate_times,
            val_metas, val_masks, args.nms_radius_sec, args.tolerance_sec, floors),
        "rescored": evaluate_scores(
            new_scores, candidate_times, val_metas, val_masks,
            args.nms_radius_sec, args.tolerance_sec, floors),
    }
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
