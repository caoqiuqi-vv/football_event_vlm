#!/usr/bin/env python
"""Train the D1 candidate examiner head and run the V2/D1/concat comparison.

Spec: docs/football_tail_first_fullchain_plan_20260910.md §5.

Head (per label): 48-frame sequence -> Linear(in_dim->256) + learned positional
embedding -> 2x TransformerEncoderLayer(d_model=256, nhead=8, ffn=512, gelu,
norm_first) -> mean pool -> Linear(256->1) logit.  Loss: BCE with
pos_weight = n_neg/n_pos of the fold's trainable rows (is_ignored rows never
enter the loss but are scored for evaluation, same as V2).  Optimizer: AdamW
lr=1e-3, wd=1e-4, epochs<=20, early stop on the fold's val P@Rfloor (patience 5).

Protocol (identical to scripts/fit_football_verifier_v2.py, whose evaluation
functions are reused verbatim):
  - train129: GroupKFold(5, by video) OOF
  - val15:    per-class threshold = max precision s.t. recall >= floor
              (shot 0.85 / save 0.80 / set_piece 0.80), window_overlap tol=3s
  - test18:   frozen transfer of the val15 thresholds, GT = test18_final_repaired_v2

Comparison arms:
  v2     — Verifier V2 scores (OOF from oof_predictions.csv; val15/test18
           recomputed deterministically from verifier.json via apply_model)
  d1     — this head's scores
  concat — 2-d stacker: LogisticRegression(class_weight=balanced) on
           [logit(v2), logit(d1)], trained on train129 OOF pairs with the same
           GroupKFold for its own OOF; final stacker fit on all OOF pairs and
           applied to val15/test18. Score-level concatenation keeps the two
           evidence sources calibrated instead of mixing raw feature spaces.

Decision gate (pre-registered in §5): OOF save P@R80 improvement of the best
D1 arm over v2 alone, reported in report["decision_gate"] (>= +3pp = go).

Features come from extract_d1_candidate_features.py npz files; candidates whose
video npz is missing are dropped from ALL arms' evaluation rows (fair
comparison on the covered subset) and counted under report["missing_features"].

`--smoke-random-features` replaces npz features with deterministic random
tensors — used to smoke-test the full train/OOF/eval chain on CPU before the
GPU extraction exists.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import fit_football_verifier_v2 as v2

LABELS = v2.LABELS
RECALL_FLOORS = v2.RECALL_FLOORS
NUM_FRAMES = 48

DEFAULT_MANIFEST = Path("outputs/football_d1_examiner/manifest_20260911.parquet")
DEFAULT_FEATURES_DIR = Path("outputs/football_d1_examiner/features")
DEFAULT_V2_DIR = Path("outputs/football_candidate_verifiers/v2_full166_20260910")
DEFAULT_RUN_DIR = Path("outputs/football_full_review/full166_fromlast_e8_best_dense_s5_20260902")
DEFAULT_TEST_GT = Path(
    "outputs/football_event_annotations/test18_final_repaired_v2_20260907T123656Z/final_labels.json"
)


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------


class FeatureProvider:
    """candidate_idx -> float16 [48, D] features, from npz files or RNG (smoke)."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        features_dir: Path,
        *,
        smoke_random: bool,
        smoke_dim: int,
    ):
        self.smoke_random = smoke_random
        self.smoke_dim = smoke_dim
        self.missing_videos: list[str] = []
        self.feats: dict[int, np.ndarray] = {}
        self.in_dim = smoke_dim
        if smoke_random:
            return
        by_video: dict[str, list[int]] = {}
        for video_id, group in manifest.groupby("video_id"):
            by_video[video_id] = [int(i) for i in group["candidate_idx"]]
        for video_id, cand_ids in sorted(by_video.items()):
            path = features_dir / f"{video_id}.npz"
            if not path.is_file():
                self.missing_videos.append(video_id)
                continue
            with np.load(path) as data:
                stored = {int(i): j for j, i in enumerate(data["candidate_idx"])}
                feats = data["feats"]
                self.in_dim = int(feats.shape[-1])
                for cand_id in cand_ids:
                    j = stored.get(cand_id)
                    if j is not None:
                        self.feats[cand_id] = feats[j]

    def available(self, candidate_idx: int) -> bool:
        return self.smoke_random or candidate_idx in self.feats

    def get(self, candidate_idx: int) -> np.ndarray:
        if self.smoke_random:
            rng = np.random.default_rng(candidate_idx * 1009 + 7)
            return rng.standard_normal((NUM_FRAMES, self.smoke_dim), dtype=np.float32).astype(np.float16)
        return self.feats[candidate_idx]

    def stack(self, candidate_ids: list[int]) -> np.ndarray:
        return np.stack([self.get(i) for i in candidate_ids]).astype(np.float32)


# ---------------------------------------------------------------------------
# head
# ---------------------------------------------------------------------------


def build_head(in_dim: int, d_model: int, nhead: int, ffn: int, layers: int, dropout: float):
    import torch
    from torch import nn

    class D1ExaminerHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(in_dim, d_model)
            self.pos = nn.Parameter(torch.zeros(1, NUM_FRAMES, d_model))
            nn.init.normal_(self.pos, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=ffn,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=layers, norm=nn.LayerNorm(d_model))
            self.out = nn.Linear(d_model, 1)

        def forward(self, x):  # x: [B, 48, in_dim] float32
            h = self.proj(x) + self.pos
            h = self.encoder(h)
            return self.out(h.mean(dim=1)).squeeze(-1)

    return D1ExaminerHead()


def batches(ids: list[int], batch_size: int, rng: np.random.Generator | None):
    order = np.arange(len(ids))
    if rng is not None:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield [ids[int(i)] for i in order[start : start + batch_size]]


def score_candidates(model, provider: FeatureProvider, ids: list[int], batch_size: int, device: str) -> np.ndarray:
    import torch

    model.eval()
    out = np.zeros(len(ids), dtype=np.float64)
    with torch.no_grad():
        for offset, chunk in enumerate(batches(ids, batch_size, None)):
            x = torch.from_numpy(provider.stack(chunk)).to(device)
            logits = model(x).float().cpu().numpy()
            out[offset * batch_size : offset * batch_size + len(chunk)] = 1.0 / (1.0 + np.exp(-logits))
    return out


def train_head_fold(
    provider: FeatureProvider,
    train_ids: list[int],
    train_targets: dict[int, int],  # keyed by candidate_idx; -1 = ignored (never in loss)
    val_rows: list[v2.Candidate],
    val_ids: list[int],
    gt_all: dict,
    label: str,
    val_video_ids: list[str],
    args: argparse.Namespace,
    seed: int,
    select_on_val: bool = True,
) -> tuple[object, dict[tuple[str, str, int], float], int, list[dict]]:
    """Returns (model, val scores by candidate key, best epoch, history).

    select_on_val=False (final all-train model): run exactly args.epochs epochs,
    no per-epoch val evaluation, return the last state — val15 must not drive
    model selection for the transfer model.
    """
    import torch
    from torch import nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = args.device
    model = build_head(provider.in_dim, args.d_model, args.nhead, args.ffn, args.layers, args.dropout).to(device)

    fit_ids = [i for i in train_ids if train_targets[i] >= 0]
    n_pos = sum(train_targets[i] for i in fit_ids)
    n_neg = len(fit_ids) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    rng = np.random.default_rng(seed)
    best = {"key": (-1.0, -1.0), "epoch": -1, "state": None, "point": None}
    history: list[dict] = []
    epochs_since_best = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for chunk in batches(fit_ids, args.batch_size, rng):
            x = torch.from_numpy(provider.stack(chunk)).to(device)
            y = torch.tensor([train_targets[i] for i in chunk], dtype=torch.float32, device=device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach())
            n_batches += 1
        history.append({"epoch": epoch, "train_loss": round(total_loss / max(n_batches, 1), 5)})
        if not select_on_val:
            best = {"key": (0.0, 0.0), "epoch": epoch, "state": None, "point": None}
            continue
        val_scores = score_candidates(model, provider, val_ids, args.batch_size, device)
        score_map = {row.key: float(s) for row, s in zip(val_rows, val_scores)}
        point, _ = v2.select_threshold(val_rows, score_map, gt_all, label, val_video_ids, RECALL_FLOORS[label])
        history[-1].update(
            {
                "val_p_at_floor": round(float(point["precision"]), 5),
                "val_r": round(float(point["recall"]), 5),
            }
        )
        key = (float(point["precision"]), float(point["recall"]))
        if key > best["key"]:
            best = {
                "key": key,
                "epoch": epoch,
                "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
                "point": point,
            }
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= args.patience:
                break
    if select_on_val:
        assert best["state"] is not None
        model.load_state_dict(best["state"])
    final_scores = score_candidates(model, provider, val_ids, args.batch_size, device)
    return model, {row.key: float(s) for row, s in zip(val_rows, final_scores)}, best["epoch"], history


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------


def prob_logit(value: float) -> float:
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


def load_v2_scores(v2_dir: Path, rows: list[v2.Candidate]) -> dict[tuple[str, str, int], float]:
    """V2 verifier scores for any split rows (recomputed from verifier.json)."""
    payload = json.loads((v2_dir / "verifier.json").read_text())
    models = payload["models"]
    final_feature_names = payload["final_feature_names"]
    scores: dict[tuple[str, str, int], float] = {}
    for label in LABELS:
        label_rows = [row for row in rows if row.label == label]
        if not label_rows:
            continue
        scores.update(v2.apply_model(label_rows, final_feature_names[label], models[label]))
    return scores


def load_v2_oof_scores(v2_dir: Path) -> dict[tuple[str, str, int], float]:
    scores: dict[tuple[str, str, int], float] = {}
    with (v2_dir / "oof_predictions.csv").open(newline="", encoding="utf-8") as handle:
        import csv

        for row in csv.DictReader(handle):
            if row["oof_score"] in ("", None):
                continue
            key = (row["video_id"], row["label"], int(float(row["window_index"])))
            scores[key] = float(row["oof_score"])
    return scores


def fit_concat_stacker(
    rows: list[v2.Candidate],
    v2_scores: dict,
    d1_scores: dict,
    *,
    folds: int,
) -> dict[tuple[str, str, int], float]:
    """OOF scores of a 2-d logit stacker, GroupKFold by video (same protocol).

    Folds are defined over ALL rows (like the D1 head); only non-ignored rows
    enter the fit, but every val-fold row (ignored included) receives a score —
    the window_overlap evaluation needs them. A fold whose train side has a
    single class (tiny debug subsamples only) falls back to the D1 score.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    x = np.asarray([[prob_logit(v2_scores[row.key]), prob_logit(d1_scores[row.key])] for row in rows])
    y = np.asarray([-1 if row.target is None else int(row.target) for row in rows], dtype=np.int64)
    groups = np.asarray([row.video_id for row in rows])
    out: dict[tuple[str, str, int], float] = {}
    n_splits = min(max(2, folds), len(np.unique(groups)))
    for tr, va in GroupKFold(n_splits=n_splits).split(x, y, groups):
        fit_mask = y[tr] >= 0
        single_class = len(np.unique(y[tr][fit_mask])) < 2
        model = None
        if not single_class:
            model = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000, random_state=42)
            model.fit(x[tr][fit_mask], y[tr][fit_mask])
        for idx in va:
            key = rows[int(idx)].key
            if model is None:
                out[key] = float(d1_scores[key])
            else:
                out[key] = float(model.predict_proba(x[int(idx) : int(idx) + 1])[:, 1][0])
    return out


def _fit_lr(x: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000, random_state=42)
    if len(np.unique(y)) < 2:
        return None
    model.fit(x, y)
    return model


def fit_concat_stacker_final(
    rows: list[v2.Candidate],
    v2_scores: dict,
    d1_scores: dict,
):
    fit_rows = [row for row in rows if row.target is not None]
    x = np.asarray([[prob_logit(v2_scores[row.key]), prob_logit(d1_scores[row.key])] for row in fit_rows])
    y = np.asarray([int(row.target) for row in fit_rows], dtype=np.int64)
    return _fit_lr(x, y)


def apply_concat_stacker(model, rows: list[v2.Candidate], v2_scores: dict, d1_scores: dict) -> dict:
    if model is None:  # single-class fallback (debug subsamples only)
        return {row.key: float(d1_scores[row.key]) for row in rows}
    x = np.asarray([[prob_logit(v2_scores[row.key]), prob_logit(d1_scores[row.key])] for row in rows])
    prob = model.predict_proba(x)[:, 1]
    return {row.key: float(p) for row, p in zip(rows, prob)}


# ---------------------------------------------------------------------------
# evaluation helpers (V2 protocol)
# ---------------------------------------------------------------------------


def oof_point(rows, scores, gt_all, label, video_ids):
    best, _ = v2.select_threshold(rows, scores, gt_all, label, video_ids, RECALL_FLOORS[label])
    return {**v2.metric_short(best), "recall_shortfall": bool(best.get("recall_shortfall", False))}


def frozen_transfer(rows, scores, gt_all, label, val_ids, test_ids):
    """val15 threshold selection, then frozen test18 eval (V2 protocol)."""
    vrows = [row for row in rows if row.split == "val15"]
    trows = [row for row in rows if row.split == "test18"]
    best, _ = v2.select_threshold(vrows, scores, gt_all, label, val_ids, RECALL_FLOORS[label])
    threshold = float(best["threshold"])
    tpoint = v2.eval_frozen(trows, scores, gt_all, label, test_ids, threshold)
    return {
        "val15": {**v2.metric_short(best), "recall_shortfall": bool(best.get("recall_shortfall", False))},
        "threshold": threshold,
        "test18": v2.metric_short(tpoint),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR)
    parser.add_argument("--v2-dir", type=Path, default=DEFAULT_V2_DIR)
    parser.add_argument("--candidates-csv", type=Path, default=DEFAULT_V2_DIR / "candidates.csv")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--test-gt-json", type=Path, default=DEFAULT_TEST_GT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/football_d1_examiner/train_20260911"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-folds", type=int, default=0, help="debug: stop after N folds (0=all)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--ffn", type=int, default=512)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke-random-features", action="store_true", help="RNG features, no npz needed")
    parser.add_argument("--smoke-dim", type=int, default=2048)
    parser.add_argument("--max-candidates-per-label", type=int, default=0, help="debug subsample (train split)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(args.manifest)
    rows_all = v2.load_candidates(args.candidates_csv)
    rows_all = [row for row in rows_all if row.split in ("train", "val15", "test18")]
    key_to_cidx = {
        (r.video_id, r.label, r.window_index): int(r.candidate_idx)
        for r in manifest.itertuples()
    }

    train_rows = [row for row in rows_all if row.split == "train"]
    val_videos = sorted({row.video_id for row in rows_all if row.split == "val15"})
    test_videos = sorted({row.video_id for row in rows_all if row.split == "test18"})
    train_videos = sorted({row.video_id for row in train_rows})
    leakage = set(train_videos) & (set(val_videos) | set(test_videos))
    if leakage:
        raise ValueError(f"split leakage: {sorted(leakage)}")

    run_gt = v2.load_run_gt(args.run_dir, train_videos + val_videos)
    test_gt = v2.load_test_gt(args.test_gt_json)
    gt_all = {
        **run_gt,
        **{vid: test_gt.get(vid, {l: [] for l in LABELS}) for vid in test_videos},
    }

    provider = FeatureProvider(
        manifest, args.features_dir, smoke_random=args.smoke_random_features, smoke_dim=args.smoke_dim
    )
    covered_rows = [
        row for row in rows_all if provider.available(key_to_cidx[row.key])
    ]
    coverage = {
        "rows_total": len(rows_all),
        "rows_with_features": len(covered_rows),
        "missing_videos": provider.missing_videos,
    }
    print(f"feature coverage: {coverage['rows_with_features']}/{coverage['rows_total']}", flush=True)

    v2_scores_covered = load_v2_scores(args.v2_dir, covered_rows)
    v2_oof_all = load_v2_oof_scores(args.v2_dir)

    report: dict = {
        "params": {
            "folds": args.folds, "epochs": args.epochs, "patience": args.patience,
            "lr": args.lr, "weight_decay": args.weight_decay, "batch_size": args.batch_size,
            "head": {"d_model": args.d_model, "nhead": args.nhead, "ffn": args.ffn,
                     "layers": args.layers, "dropout": args.dropout, "in_dim": provider.in_dim},
            "smoke_random_features": bool(args.smoke_random_features),
            "recall_floors": RECALL_FLOORS, "match_tolerance_sec": v2.MATCH_TOLERANCE_SEC,
        },
        "coverage": coverage,
        "gt_counts": {
            split: {label: sum(len(gt_all.get(vid, {}).get(label, [])) for vid in vids) for label in LABELS}
            for split, vids in (("train", train_videos), ("val15", val_videos), ("test18", test_videos))
        },
    }
    prediction_rows: list[dict] = []
    oof_d1_by_label: dict[str, dict] = {}
    oof_concat_by_label: dict[str, dict] = {}

    for label in LABELS:
        label_rows = [row for row in covered_rows if row.label == label]
        label_train = [row for row in label_rows if row.split == "train"]
        if args.max_candidates_per_label:
            # stratified by position: candidates.csv is grouped by video, so a
            # head() cut would leave GroupKFold a single group
            take = np.linspace(0, len(label_train) - 1, args.max_candidates_per_label).astype(int)
            label_train = [label_train[int(i)] for i in sorted(set(take))]
        train_targets = {
            key_to_cidx[row.key]: (-1 if row.target is None else int(row.target)) for row in label_train
        }
        train_ids = [key_to_cidx[row.key] for row in label_train]
        groups = np.asarray([row.video_id for row in label_train])

        # --- OOF ---
        from sklearn.model_selection import GroupKFold

        n_splits = min(max(2, args.folds), len(np.unique(groups)))
        oof_scores: dict = {}
        best_epochs: list[int] = []
        x_dummy = np.zeros(len(label_train))
        y_dummy = np.zeros(len(label_train), dtype=np.int64)  # GroupKFold ignores y
        fold_histories: list[dict] = []
        for fold, (tr, va) in enumerate(GroupKFold(n_splits=n_splits).split(x_dummy, y_dummy, groups)):
            if args.max_folds and fold >= args.max_folds:
                break
            tr_rows = [label_train[int(i)] for i in tr]
            va_rows = [label_train[int(i)] for i in va]
            tr_ids = [key_to_cidx[row.key] for row in tr_rows]
            va_ids = [key_to_cidx[row.key] for row in va_rows]
            va_video_ids = sorted({row.video_id for row in va_rows})
            _model, scores, best_epoch, history = train_head_fold(
                provider, tr_ids, train_targets, va_rows, va_ids, gt_all, label, va_video_ids,
                args, seed=args.seed + fold,
            )
            oof_scores.update(scores)
            best_epochs.append(best_epoch)
            fold_histories.append({"fold": fold, "best_epoch": best_epoch, "history": history})
            print(f"{label} fold {fold}: best_epoch={best_epoch} history_last={history[-1]}", flush=True)
        oof_d1_by_label[label] = oof_scores

        # OOF rows limited to candidates scored in the (possibly truncated) folds
        oof_rows = [row for row in label_train if row.key in oof_scores]
        v2_oof_scores = {row.key: v2_oof_all[row.key] for row in oof_rows if row.key in v2_oof_all}
        oof_rows_v2 = [row for row in oof_rows if row.key in v2_oof_scores]
        concat_oof = fit_concat_stacker(
            oof_rows_v2,
            v2_oof_scores,
            {k: v for k, v in oof_scores.items() if k in v2_oof_scores},
            folds=n_splits,
        )
        oof_concat_by_label[label] = concat_oof
        train_video_ids_oof = sorted({row.video_id for row in oof_rows_v2})
        report.setdefault("oof", {})[label] = {
            "v2": oof_point(oof_rows_v2, v2_oof_scores, gt_all, label, train_video_ids_oof),
            "d1": oof_point(oof_rows_v2, {k: oof_scores[k] for k in v2_oof_scores}, gt_all, label, train_video_ids_oof),
            "concat": oof_point(oof_rows_v2, concat_oof, gt_all, label, train_video_ids_oof),
            "n_rows": len(oof_rows_v2),
            "best_epochs": best_epochs,
        }

        # --- final model on all train129 (epochs = median of fold best epochs) ---
        final_epochs = int(np.median(best_epochs)) if best_epochs else args.epochs
        final_epochs = max(final_epochs, 1)
        saved_epochs = args.epochs
        args.epochs = final_epochs
        final_model, _, _, _ = train_head_fold(
            provider, train_ids, train_targets,
            [row for row in label_rows if row.split == "val15"],
            [key_to_cidx[row.key] for row in label_rows if row.split == "val15"],
            gt_all, label, val_videos, args, seed=args.seed + 1000,
            select_on_val=False,
        )
        args.epochs = saved_epochs

        eval_rows = [row for row in label_rows if row.split in ("val15", "test18")]
        eval_ids = [key_to_cidx[row.key] for row in eval_rows]
        d1_eval = score_candidates(final_model, provider, eval_ids, args.batch_size, args.device)
        d1_eval_scores = {row.key: float(s) for row, s in zip(eval_rows, d1_eval)}
        v2_eval_scores = {row.key: v2_scores_covered[row.key] for row in eval_rows}
        stacker = fit_concat_stacker_final(oof_rows_v2, v2_oof_scores, {k: oof_scores[k] for k in v2_oof_scores})
        concat_eval_scores = apply_concat_stacker(stacker, eval_rows, v2_eval_scores, d1_eval_scores)

        report.setdefault("transfer", {})[label] = {
            "final_model_epochs": final_epochs,
            "v2": frozen_transfer(eval_rows, v2_eval_scores, gt_all, label, val_videos, test_videos),
            "d1": frozen_transfer(eval_rows, d1_eval_scores, gt_all, label, val_videos, test_videos),
            "concat": frozen_transfer(eval_rows, concat_eval_scores, gt_all, label, val_videos, test_videos),
        }
        for row in oof_rows_v2:
            prediction_rows.append({
                "video_id": row.video_id, "split": row.split, "label": label,
                "window_index": row.window_index, "peak_time_sec": row.peak_time_sec,
                "target": "" if row.target is None else row.target,
                "v2_score": v2_oof_scores[row.key], "d1_score": oof_scores[row.key],
                "concat_score": concat_oof[row.key],
            })
        for row in eval_rows:
            prediction_rows.append({
                "video_id": row.video_id, "split": row.split, "label": label,
                "window_index": row.window_index, "peak_time_sec": row.peak_time_sec,
                "target": "" if row.target is None else row.target,
                "v2_score": v2_eval_scores[row.key], "d1_score": d1_eval_scores[row.key],
                "concat_score": concat_eval_scores[row.key],
            })

    # --- decision gate: OOF save P@R80 improvement vs V2 ----------------------
    save_oof = report["oof"]["save"]
    gate = {
        "metric": "OOF save precision at recall floor 0.80 (window_overlap tol=3s)",
        "v2": save_oof["v2"]["precision"],
        "d1": save_oof["d1"]["precision"],
        "concat": save_oof["concat"]["precision"],
        "delta_d1_vs_v2_pp": round((save_oof["d1"]["precision"] - save_oof["v2"]["precision"]) * 100, 2),
        "delta_concat_vs_v2_pp": round((save_oof["concat"]["precision"] - save_oof["v2"]["precision"]) * 100, 2),
    }
    gate["best_arm"] = "concat" if save_oof["concat"]["precision"] >= save_oof["d1"]["precision"] else "d1"
    gate["best_delta_pp"] = gate[f"delta_{gate['best_arm']}_vs_v2_pp"]
    gate["go"] = bool(gate["best_delta_pp"] >= 3.0)
    report["decision_gate"] = gate

    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(prediction_rows).to_csv(args.output_dir / "predictions.csv", index=False)
    print(json.dumps(report["oof"], ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(report["decision_gate"], ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
