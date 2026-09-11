#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


LABELS = ("shot", "save", "set_piece")
EXPERIMENTS = ("p0", "p1", "p2", "p3")


@dataclass
class Record:
    video_id: str
    label: str
    label_index: int
    target: float
    group: str
    window_index: int
    start_sec: float
    end_sec: float
    checkpoint_probability: float
    store_index: int = -1


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def binary_auc(targets: np.ndarray, scores: np.ndarray) -> float:
    targets = targets.astype(np.int64)
    positives = int(targets.sum())
    negatives = int(len(targets) - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    rank_sum = float(ranks[targets == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def best_balanced_threshold(targets: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    candidates = np.unique(scores)
    if len(candidates) > 200:
        candidates = np.quantile(scores, np.linspace(0, 1, 201))
    best = (0.0, -1.0)
    for threshold in candidates:
        pred = scores >= threshold
        tpr = float(pred[targets == 1].mean()) if np.any(targets == 1) else 0.0
        tnr = float((~pred[targets == 0]).mean()) if np.any(targets == 0) else 0.0
        balanced = 0.5 * (tpr + tnr)
        if balanced > best[1]:
            best = (float(threshold), balanced)
    return best


def balanced_accuracy(targets: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    pred = scores >= threshold
    tpr = float(pred[targets == 1].mean()) if np.any(targets == 1) else float("nan")
    tnr = float((~pred[targets == 0]).mean()) if np.any(targets == 0) else float("nan")
    return 0.5 * (tpr + tnr)


class TokenStore:
    def __init__(
        self,
        cache_root: Path,
        records: list[Record],
        device: torch.device,
        store_device: str,
    ):
        manifest = json.loads((cache_root / "manifest.json").read_text())
        entries = {
            (str(item["video_id"]), int(item["window_index"])): item
            for item in manifest["windows"]
        }
        keys = sorted({(record.video_id, record.window_index) for record in records})
        missing = [key for key in keys if key not in entries]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} cached windows; first={missing[:5]}")
        first = torch.load(cache_root / entries[keys[0]]["cache_path"], map_location="cpu", weights_only=False)
        frames, patches, dim = map(int, first["patches"].shape)
        self.num_frames = frames
        self.num_patches = patches
        self.feature_dim = dim
        self.device = device if store_device == "cuda" else torch.device("cpu")
        bytes_required = len(keys) * frames * patches * dim * 2
        if self.device.type == "cuda":
            free_bytes, _ = torch.cuda.mem_get_info(device)
            if bytes_required > free_bytes * 0.82:
                raise RuntimeError(
                    f"Patch cache needs {bytes_required / 2**30:.2f} GiB but GPU has "
                    f"{free_bytes / 2**30:.2f} GiB free; rerun with --store-device cpu"
                )
        print(
            f"loading token store windows={len(keys)} shape=({frames},{patches},{dim}) "
            f"estimated={bytes_required / 2**30:.2f}GiB device={self.device}",
            flush=True,
        )
        self.patches = torch.empty(
            (len(keys), frames, patches, dim), dtype=torch.float16, device=self.device
        )
        self.cls = torch.empty((len(keys), frames, dim), dtype=torch.float16, device=self.device)
        self.frame_times = torch.empty((len(keys), frames), dtype=torch.float32, device=self.device)
        self.keys = keys
        key_to_index = {key: index for index, key in enumerate(keys)}
        for position, key in enumerate(keys, start=1):
            payload = first if key == keys[0] else torch.load(
                cache_root / entries[key]["cache_path"], map_location="cpu", weights_only=False
            )
            target = key_to_index[key]
            self.patches[target].copy_(payload["patches"].to(self.device))
            self.cls[target].copy_(payload["cls"].to(self.device))
            self.frame_times[target].copy_(payload["frame_times"].to(self.device))
            if position == 1 or position % 20 == 0 or position == len(keys):
                print(f"loaded token windows={position}/{len(keys)}", flush=True)
        for record in records:
            record.store_index = key_to_index[(record.video_id, record.window_index)]

    def batch(
        self,
        records: Sequence[Record],
        frame_indices: Tensor,
        target_device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor]:
        indices = torch.tensor([record.store_index for record in records], device=self.device)
        patches = self.patches.index_select(0, indices).index_select(1, frame_indices.to(self.device))
        cls = self.cls.index_select(0, indices).index_select(1, frame_indices.to(self.device))
        if self.device != target_device:
            patches = patches.to(target_device, non_blocking=True)
            cls = cls.to(target_device, non_blocking=True)
        labels = torch.tensor([record.label_index for record in records], device=target_device)
        return patches, cls, labels


class TemporalCLS(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, num_heads: int, max_frames: int):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_frames + 1, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            hidden_dim,
            num_heads,
            hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, tokens: Tensor) -> Tensor:
        batch, frames, _ = tokens.shape
        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_embed[:, : frames + 1]
        return self.norm(self.encoder(tokens)[:, 0])


class PatchReadout(nn.Module):
    def __init__(
        self,
        experiment: str,
        feature_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        max_frames: int,
        micro_clips: int,
        frames_per_micro: int,
    ):
        super().__init__()
        self.experiment = experiment
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_labels = len(LABELS)
        self.micro_clips = micro_clips
        self.frames_per_micro = frames_per_micro
        self.num_queries = {"p0": 0, "p1": 1, "p2": 4, "p3": 4}[experiment]
        if self.num_queries:
            self.patch_queries = nn.Parameter(
                torch.empty(self.num_labels, self.num_queries, feature_dim)
            )
            nn.init.trunc_normal_(self.patch_queries, std=0.02)
        input_dim = feature_dim * (2 if experiment == "p0" else self.num_queries + 1)
        self.frame_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.class_embed = nn.Embedding(self.num_labels, hidden_dim)
        if experiment in {"p0", "p1"}:
            self.temporal = TemporalCLS(hidden_dim, num_layers, num_heads, max_frames)
            output_dim = self.num_labels if experiment == "p0" else 1
            self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, output_dim))
        elif experiment == "p2":
            layer = nn.TransformerEncoderLayer(
                hidden_dim,
                num_heads,
                hidden_dim * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.pos_embed = nn.Parameter(torch.zeros(1, max_frames, hidden_dim))
            self.temporal_queries = nn.Embedding(self.num_labels, hidden_dim)
            self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.motion_proj = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
            )
            self.micro_queries = nn.Embedding(self.num_labels, hidden_dim)
            self.micro_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, max(hidden_dim // 2, 32)),
                nn.GELU(),
                nn.Linear(max(hidden_dim // 2, 32), 1),
            )

    def query_pool(self, patches: Tensor, labels: Tensor) -> Tensor:
        queries = F.normalize(self.patch_queries.index_select(0, labels), dim=-1)
        normalized_patches = F.normalize(patches.float(), dim=-1)
        scores = torch.einsum("btnd,bqd->btqn", normalized_patches, queries)
        weights = torch.softmax(scores / 0.07, dim=-1).to(patches.dtype)
        return torch.einsum("btqn,btnd->btqd", weights, patches)

    def frame_tokens(self, patches: Tensor, cls: Tensor, labels: Tensor) -> Tensor:
        if self.experiment == "p0":
            spatial = torch.cat([cls, patches.mean(dim=2)], dim=-1)
        else:
            pooled = self.query_pool(patches, labels)
            spatial = torch.cat([cls.unsqueeze(2), pooled], dim=2).flatten(2)
        tokens = self.frame_proj(spatial.float())
        if self.experiment != "p0":
            tokens = tokens + self.class_embed(labels).unsqueeze(1)
        return tokens

    def forward(self, patches: Tensor, cls: Tensor, labels: Tensor) -> dict[str, Tensor]:
        tokens = self.frame_tokens(patches, cls, labels)
        if self.experiment == "p0":
            all_logits = self.head(self.temporal(tokens))
            logits = all_logits.gather(1, labels.unsqueeze(1)).squeeze(1)
            return {"logits": logits}
        if self.experiment == "p1":
            return {"logits": self.head(self.temporal(tokens)).squeeze(-1)}
        if self.experiment == "p2":
            frames = tokens.shape[1]
            encoded = self.encoder(tokens + self.pos_embed[:, :frames])
            query = F.normalize(self.temporal_queries(labels), dim=-1)
            weights = torch.softmax(
                torch.einsum("bth,bh->bt", F.normalize(encoded, dim=-1), query) / 0.1,
                dim=1,
            )
            pooled = (encoded * weights.unsqueeze(-1)).sum(dim=1)
            return {"logits": self.head(pooled).squeeze(-1), "temporal_attention": weights}

        if tokens.shape[1] != self.micro_clips * self.frames_per_micro:
            raise ValueError(
                f"P3 needs {self.micro_clips * self.frames_per_micro} frames, got {tokens.shape[1]}"
            )
        previous = torch.cat([tokens[:, :1], tokens[:, :-1]], dim=1)
        motion = tokens - previous
        motion_tokens = self.motion_proj(torch.cat([tokens, motion], dim=-1))
        batch = tokens.shape[0]
        micro = motion_tokens.reshape(
            batch, self.micro_clips, self.frames_per_micro, self.hidden_dim
        )
        query = F.normalize(self.micro_queries(labels), dim=-1)
        weights = torch.softmax(
            torch.einsum("bmfh,bh->bmf", F.normalize(micro, dim=-1), query) / 0.1,
            dim=2,
        )
        micro_features = (micro * weights.unsqueeze(-1)).sum(dim=2)
        micro_logits = self.micro_head(micro_features).squeeze(-1)
        temperature = 0.5
        logits = temperature * torch.logsumexp(micro_logits / temperature, dim=1)
        logits = logits - temperature * math.log(self.micro_clips)
        return {"logits": logits, "micro_logits": micro_logits, "micro_attention": weights}

    def diversity_loss(self) -> Tensor:
        if self.num_queries <= 1:
            return next(self.parameters()).new_zeros(())
        queries = F.normalize(self.patch_queries, dim=-1)
        gram = torch.einsum("cqd,ckd->cqk", queries, queries)
        identity = torch.eye(self.num_queries, device=gram.device).unsqueeze(0)
        return ((gram - identity) ** 2).mean()


def parse_records(path: Path) -> list[Record]:
    labels = {label: index for index, label in enumerate(LABELS)}
    records: list[Record] = []
    for row in read_csv(path):
        if row["group"] not in {"TP", "FP"} or row["label"] not in labels:
            continue
        records.append(
            Record(
                video_id=row["video_id"],
                label=row["label"],
                label_index=labels[row["label"]],
                target=float(row["group"] == "TP"),
                group=row["group"],
                window_index=int(row["window_index"]),
                start_sec=float(row["start_sec"]),
                end_sec=float(row["end_sec"]),
                checkpoint_probability=float(row["probability"]),
            )
        )
    return records


def batches(indices: Sequence[int], batch_size: int, shuffle: bool) -> list[list[int]]:
    values = list(indices)
    if shuffle:
        random.shuffle(values)
    return [values[start : start + batch_size] for start in range(0, len(values), batch_size)]


def predict(
    model: PatchReadout,
    store: TokenStore,
    records: list[Record],
    indices: Sequence[int],
    frame_indices: Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    scores: list[np.ndarray] = []
    with torch.inference_mode():
        for batch_indices in batches(indices, batch_size, False):
            items = [records[index] for index in batch_indices]
            patches, cls, labels = store.batch(items, frame_indices, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(patches, cls, labels)["logits"]
            scores.append(logits.float().cpu().numpy())
    return np.concatenate(scores) if scores else np.zeros(0, dtype=np.float32)


def train_fold(
    experiment: str,
    seed: int,
    held_video: str,
    store: TokenStore,
    records: list[Record],
    frame_indices: Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float, list[float]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_indices = [index for index, record in enumerate(records) if record.video_id != held_video]
    test_indices = [index for index, record in enumerate(records) if record.video_id == held_video]
    model = PatchReadout(
        experiment,
        store.feature_dim,
        args.hidden_dim,
        args.temporal_layers,
        args.temporal_heads,
        int(frame_indices.numel()),
        args.micro_clips,
        args.frames_per_micro,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history: list[float] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch_indices in batches(train_indices, args.batch_size, True):
            items = [records[index] for index in batch_indices]
            patches, cls, labels = store.batch(items, frame_indices, device)
            targets = torch.tensor([record.target for record in items], device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                outputs = model(patches, cls, labels)
                loss = F.binary_cross_entropy_with_logits(outputs["logits"].float(), targets)
                if experiment == "p3":
                    negative = targets == 0
                    if bool(negative.any()):
                        loss = loss + args.negative_micro_weight * F.binary_cross_entropy_with_logits(
                            outputs["micro_logits"][negative].float(),
                            torch.zeros_like(outputs["micro_logits"][negative].float()),
                        )
                loss = loss + args.query_diversity_weight * model.diversity_loss()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)))
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"train exp={experiment} seed={seed} held={held_video} epoch={epoch}/{args.epochs} "
                f"loss={history[-1]:.5f}",
                flush=True,
            )
    train_scores = predict(model, store, records, train_indices, frame_indices, device, args.batch_size)
    test_scores = predict(model, store, records, test_indices, frame_indices, device, args.batch_size)
    train_targets = np.asarray([records[index].target for index in train_indices])
    threshold, _ = best_balanced_threshold(train_targets, train_scores)
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.asarray(test_indices), test_scores, threshold, history


def summarize_predictions(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    experiments = sorted({row["experiment"] for row in rows})
    seeds = sorted({int(row["seed"]) for row in rows})
    for experiment in experiments:
        for seed in seeds:
            selected = [row for row in rows if row["experiment"] == experiment and int(row["seed"]) == seed]
            if not selected:
                continue
            for label in LABELS:
                label_rows = [row for row in selected if row["label"] == label]
                targets = np.asarray([int(row["target"]) for row in label_rows])
                scores = np.asarray([float(row["score"]) for row in label_rows])
                thresholds = np.asarray([float(row["threshold"]) for row in label_rows])
                predictions = scores >= thresholds
                tpr = float(predictions[targets == 1].mean())
                tnr = float((~predictions[targets == 0]).mean())
                summary.append(
                    {
                        "experiment": experiment,
                        "seed": seed,
                        "label": label,
                        "num_tp": int((targets == 1).sum()),
                        "num_fp": int((targets == 0).sum()),
                        "auc": binary_auc(targets, scores),
                        "balanced_accuracy": 0.5 * (tpr + tnr),
                        "tp_recall": tpr,
                        "fp_rejection": tnr,
                    }
                )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train P0-P3 frozen-DINO patch-token readout ceiling experiments with leave-one-video-out CV.")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiments", default=",".join(EXPERIMENTS))
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--temporal-layers", type=int, default=4)
    parser.add_argument("--temporal-heads", type=int, default=8)
    parser.add_argument("--micro-clips", type=int, default=5)
    parser.add_argument("--frames-per-micro", type=int, default=4)
    parser.add_argument("--negative-micro-weight", type=float, default=0.3)
    parser.add_argument("--query-diversity-weight", type=float, default=0.01)
    parser.add_argument("--store-device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = parse_records(Path(args.cases).expanduser().resolve())
    device = torch.device(args.device)
    store = TokenStore(
        Path(args.cache_root).expanduser().resolve(), records, device, args.store_device
    )
    experiments = [item.strip() for item in args.experiments.split(",") if item.strip()]
    invalid = [item for item in experiments if item not in EXPERIMENTS]
    if invalid:
        raise ValueError(f"Unknown experiments={invalid}; supported={EXPERIMENTS}")
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    videos = sorted({record.video_id for record in records})
    frame_indices_16 = torch.linspace(0, store.num_frames - 1, 16).round().long()
    frame_indices_20 = torch.arange(store.num_frames).long()
    if store.num_frames != args.micro_clips * args.frames_per_micro:
        raise ValueError(
            f"P3 configuration needs {args.micro_clips * args.frames_per_micro} cached frames, "
            f"manifest has {store.num_frames}"
        )

    predictions: list[dict[str, Any]] = []
    histories: dict[str, Any] = {}
    for seed in seeds:
        # A-best checkpoint scores are the non-trained P0 reference on the exact
        # 16-frame evaluation inputs.
        for record in records:
            predictions.append(
                {
                    "experiment": "p0_checkpoint",
                    "seed": seed,
                    "held_video": record.video_id,
                    "video_id": record.video_id,
                    "label": record.label,
                    "target": int(record.target),
                    "group": record.group,
                    "window_index": record.window_index,
                    "start_sec": record.start_sec,
                    "end_sec": record.end_sec,
                    "score": record.checkpoint_probability,
                    "threshold": 0.5,
                }
            )
        for experiment in experiments:
            frame_indices = frame_indices_20 if experiment == "p3" else frame_indices_16
            for held_video in videos:
                started = time.time()
                test_indices, test_scores, threshold, history = train_fold(
                    experiment,
                    seed,
                    held_video,
                    store,
                    records,
                    frame_indices,
                    args,
                    device,
                )
                histories[f"{experiment}/seed{seed}/{held_video}"] = history
                for record_index, score in zip(test_indices.tolist(), test_scores.tolist()):
                    record = records[record_index]
                    predictions.append(
                        {
                            "experiment": experiment,
                            "seed": seed,
                            "held_video": held_video,
                            "video_id": record.video_id,
                            "label": record.label,
                            "target": int(record.target),
                            "group": record.group,
                            "window_index": record.window_index,
                            "start_sec": record.start_sec,
                            "end_sec": record.end_sec,
                            "score": float(score),
                            "threshold": float(threshold),
                        }
                    )
                print(
                    f"fold done exp={experiment} seed={seed} held={held_video} "
                    f"elapsed={time.time() - started:.1f}s",
                    flush=True,
                )
                write_csv(output_dir / "predictions.csv", predictions)
                (output_dir / "histories.json").write_text(json.dumps(histories, indent=2))

    summary_rows = summarize_predictions(predictions)
    write_csv(output_dir / "summary.csv", summary_rows)
    aggregate: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in summary_rows:
        aggregate[row["experiment"]].setdefault(row["label"], []).append(row)
    report = {
        experiment: {
            label: {
                metric: float(np.mean([float(row[metric]) for row in rows]))
                for metric in ("auc", "balanced_accuracy", "tp_recall", "fp_rejection")
            }
            for label, rows in labels.items()
        }
        for experiment, labels in aggregate.items()
    }
    (output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
