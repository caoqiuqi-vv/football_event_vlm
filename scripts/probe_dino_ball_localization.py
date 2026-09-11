#!/usr/bin/env python
"""Probe whether frozen DINO patch tokens can localize a visible football.

Only high-confidence, directly observed tracking rows are admitted.  Videos,
not frames, are split between train and validation.  Token-wise linear and
shallow probes are trained together with shuffled-coordinate controls so that
an apparent result caused by the pitch/camera position prior is visible.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor, nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train_football_events as football  # noqa: E402
from scripts.eval_long_video_checkpoint import load_checkpoint_model  # noqa: E402


@dataclass(frozen=True)
class FrameRecord:
    video_id: str
    video_path: str
    timestamp_sec: float
    center_x: float
    center_y: float
    confidence: float
    width_px: float
    height_px: float
    shuffled_center_x: float = 0.0
    shuffled_center_y: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--index-root", required=True)
    parser.add_argument("--video-root", default="/mnt/data_16t/football/raw_video_720P")
    parser.add_argument("--video-ids", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", nargs="+", type=int, default=[5, 11, 17, 23])
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--val-video-fraction", type=float, default=0.2)
    parser.add_argument("--max-train-frames-per-video", type=int, default=80)
    parser.add_argument("--max-val-frames-per-video", type=int, default=60)
    parser.add_argument("--min-time-gap-sec", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", choices=("fp16", "bf16", "none"), default="fp16")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def distributed_setup() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, torch.device("cuda", local_rank)


def read_video_ids(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def spaced_indices(times: np.ndarray, candidates: np.ndarray, maximum: int, gap: float) -> np.ndarray:
    if not len(candidates):
        return candidates
    selected: list[int] = []
    last = -float("inf")
    for index in candidates:
        current = float(times[index])
        if current - last >= gap:
            selected.append(int(index))
            last = current
    if len(selected) <= maximum:
        return np.asarray(selected, dtype=np.int64)
    positions = np.linspace(0, len(selected) - 1, maximum).round().astype(np.int64)
    return np.asarray(selected, dtype=np.int64)[positions]


def records_for_videos(
    video_ids: Iterable[str],
    *,
    index_root: Path,
    video_root: Path,
    confidence: float,
    maximum: int,
    min_gap: float,
) -> list[FrameRecord]:
    records: list[FrameRecord] = []
    for video_id in video_ids:
        index_path = index_root / f"{video_id}.npz"
        video_path = video_root / f"{video_id}.mp4"
        if not index_path.is_file() or not video_path.is_file():
            continue
        with np.load(index_path, allow_pickle=False) as data:
            valid = (
                (data["confidence"].astype(np.float32) >= confidence)
                & (data["source_code"] == 1)
                & ((data["flags"] & 1) > 0)
            )
            candidate = np.flatnonzero(valid)
            chosen = spaced_indices(
                data["timestamp_sec"], candidate, maximum, min_gap
            )
            for index in chosen:
                box = data["bbox_xyxy_norm"][index].astype(np.float32)
                x1, y1, x2, y2 = (float(value) for value in box)
                records.append(
                    FrameRecord(
                        video_id=str(video_id),
                        video_path=str(video_path),
                        timestamp_sec=float(data["timestamp_sec"][index]),
                        center_x=0.5 * (x1 + x2),
                        center_y=0.5 * (y1 + y2),
                        confidence=float(data["confidence"][index]),
                        width_px=(x2 - x1) * 1280.0,
                        height_px=(y2 - y1) * 720.0,
                    )
                )
    return records


def attach_shuffled_targets(records: list[FrameRecord], seed: int) -> list[FrameRecord]:
    if not records:
        return []
    order = list(range(len(records)))
    random.Random(seed).shuffle(order)
    if len(order) > 1 and all(index == value for index, value in enumerate(order)):
        order = order[1:] + order[:1]
    return [
        replace(
            record,
            shuffled_center_x=records[source].center_x,
            shuffled_center_y=records[source].center_y,
        )
        for record, source in zip(records, order)
    ]


def split_records(args: argparse.Namespace) -> tuple[list[FrameRecord], list[FrameRecord], dict[str, Any]]:
    ids = read_video_ids(Path(args.video_ids).expanduser())
    eligible = [
        value
        for value in ids
        if (Path(args.index_root).expanduser() / f"{value}.npz").is_file()
        and (Path(args.video_root).expanduser() / f"{value}.mp4").is_file()
    ]
    # Some exported media IDs are byte-identical aliases of the same long
    # video. Split content groups rather than IDs so a probe cannot memorize
    # one alias and score on another.
    content_groups: dict[str, list[str]] = {}
    for value in eligible:
        payload = (Path(args.index_root).expanduser() / f"{value}.npz").read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        content_groups.setdefault(digest, []).append(value)
    groups = [sorted(values) for _, values in sorted(content_groups.items())]
    rng = random.Random(args.seed)
    rng.shuffle(groups)
    val_count = max(1, round(len(eligible) * args.val_video_fraction))
    val_groups: list[list[str]] = []
    train_groups: list[list[str]] = []
    current_val = 0
    for group in groups:
        if current_val < val_count:
            val_groups.append(group)
            current_val += len(group)
        else:
            train_groups.append(group)
    val_ids = sorted(value for group in val_groups for value in group)
    train_ids = sorted(value for group in train_groups for value in group)
    if args.smoke:
        train_ids, val_ids = train_ids[:4], val_ids[:2]
    train_records = records_for_videos(
        train_ids,
        index_root=Path(args.index_root).expanduser(),
        video_root=Path(args.video_root).expanduser(),
        confidence=args.confidence,
        maximum=(4 if args.smoke else args.max_train_frames_per_video),
        min_gap=args.min_time_gap_sec,
    )
    val_records = records_for_videos(
        val_ids,
        index_root=Path(args.index_root).expanduser(),
        video_root=Path(args.video_root).expanduser(),
        confidence=args.confidence,
        maximum=(4 if args.smoke else args.max_val_frames_per_video),
        min_gap=args.min_time_gap_sec,
    )
    if not train_records or not val_records:
        raise RuntimeError(
            f"empty probe split: train={len(train_records)} val={len(val_records)}"
        )
    train_records = attach_shuffled_targets(train_records, args.seed + 1009)
    manifest = {
        "schema": "dino-ball-localization-probe-v1",
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "index_root": str(Path(args.index_root).expanduser().resolve()),
        "video_root": str(Path(args.video_root).expanduser().resolve()),
        "confidence_threshold": args.confidence,
        "source": "track_observed",
        "split_grouping": "exact_offline_index_sha256",
        "duplicate_content_groups": [
            values for values in sorted(content_groups.values()) if len(values) > 1
        ],
        "layers": list(args.layers),
        "train_video_ids": train_ids,
        "val_video_ids": val_ids,
        "train_frames": len(train_records),
        "val_frames": len(val_records),
    }
    return train_records, val_records, manifest


class BallFrameDataset(Dataset[dict[str, Any]]):
    def __init__(self, records: list[FrameRecord], image_size: tuple[int, int]) -> None:
        self.records = records
        self.image_size = image_size
        self._cache: football.VideoCaptureCache | None = None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[int(index)]
        if self._cache is None:
            self._cache = football.VideoCaptureCache(2)
        cap = self._cache.get(record.video_path)
        fps = float(cap.get(5) or 0.0)
        if fps <= 0:
            raise RuntimeError(f"invalid fps: {record.video_path}")
        frame_index = max(int(round(record.timestamp_sec * fps)), 0)
        frame = football.read_video_segment(
            record.video_path,
            1,
            self.image_size,
            False,
            0.0,
            frame_indices=[frame_index],
            normalize=False,
            cap_cache=self._cache,
            decode_strategy="multi_seek",
        )[0]
        return {
            "frame": frame,
            "center": torch.tensor([record.center_x, record.center_y], dtype=torch.float32),
            "shuffled_center": torch.tensor(
                [record.shuffled_center_x, record.shuffled_center_y], dtype=torch.float32
            ),
            "confidence": torch.tensor(record.confidence, dtype=torch.float32),
            "size_px": torch.tensor(
                [record.width_px, record.height_px], dtype=torch.float32
            ),
            "video_id": record.video_id,
        }


class ProbeBank(nn.Module):
    def __init__(self, layers: list[int], dimension: int, hidden: int) -> None:
        super().__init__()
        self.layers = list(layers)
        self.heads = nn.ModuleDict()
        for layer in layers:
            for architecture in ("linear", "shallow"):
                for control in ("true", "shuffled"):
                    name = f"{architecture}_l{layer}_{control}"
                    self.heads[name] = (
                        nn.Linear(dimension, 1)
                        if architecture == "linear"
                        else nn.Sequential(
                            nn.LayerNorm(dimension),
                            nn.Linear(dimension, hidden),
                            nn.GELU(),
                            nn.Linear(hidden, 1),
                        )
                    )

    def forward(self, features: list[Tensor]) -> dict[str, Tensor]:
        result: dict[str, Tensor] = {}
        for layer, tokens in zip(self.layers, features):
            for architecture in ("linear", "shallow"):
                for control in ("true", "shuffled"):
                    name = f"{architecture}_l{layer}_{control}"
                    result[name] = self.heads[name](tokens).squeeze(-1)
        return result


def target_indices(center: Tensor, grid_h: int, grid_w: int) -> Tensor:
    x = (center[:, 0].clamp(0, 1 - 1e-7) * grid_w).long()
    y = (center[:, 1].clamp(0, 1 - 1e-7) * grid_h).long()
    return y * grid_w + x


def autocast_context(args: argparse.Namespace):
    if args.amp == "none":
        return torch.autocast("cuda", enabled=False)
    dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def patch_features(owner: nn.Module, frames: Tensor, layers: list[int], args: argparse.Namespace) -> list[Tensor]:
    inputs = owner.preprocess_inputs(frames.unsqueeze(1))[:, 0]
    with torch.no_grad(), autocast_context(args):
        outputs = owner.backbone.get_intermediate_layers(
            inputs, n=tuple(layers), return_class_token=False, norm=True
        )
    return [value.detach() for value in outputs]


def make_loader(
    dataset: Dataset,
    *,
    sampler: DistributedSampler,
    args: argparse.Namespace,
    train: bool,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "sampler": sampler,
        "num_workers": args.num_workers,
        "pin_memory": True,
        "drop_last": train,
    }
    if args.num_workers > 0:
        kwargs.update(
            {
                "persistent_workers": True,
                "prefetch_factor": 2,
                "multiprocessing_context": "spawn",
            }
        )
    return DataLoader(dataset, **kwargs)


def reduce_mean(value: float, count: int, device: torch.device) -> float:
    tensor = torch.tensor([value, count], dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(tensor)
    return float(tensor[0] / max(float(tensor[1]), 1.0))


def train_epoch(
    owner: nn.Module,
    bank: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    rank: int,
) -> float:
    bank.train()
    total, batches = 0.0, 0
    started = time.time()
    for step, batch in enumerate(loader, start=1):
        frames = batch["frame"].to(device, non_blocking=True)
        center = batch["center"].to(device, non_blocking=True)
        shuffled = batch["shuffled_center"].to(device, non_blocking=True)
        features = patch_features(owner, frames, args.layers, args)
        patches = int(features[0].shape[1])
        grid_h = args.image_height // 16
        grid_w = args.image_width // 16
        if grid_h * grid_w != patches:
            raise RuntimeError(f"unexpected DINO grid: {patches} != {grid_h}x{grid_w}")
        true_target = target_indices(center, grid_h, grid_w)
        shuffled_target = target_indices(shuffled, grid_h, grid_w)
        with autocast_context(args):
            logits = bank(features)
            losses = [
                nn.functional.cross_entropy(
                    value,
                    shuffled_target if name.endswith("_shuffled") else true_target,
                )
                for name, value in logits.items()
            ]
            loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.detach())
        batches += 1
        if rank == 0 and step % args.log_interval == 0:
            print(
                f"epoch={epoch} step={step}/{len(loader)} loss={total / batches:.5f} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    return reduce_mean(total, batches, device)


def summarize_errors(errors: list[dict[str, float]]) -> dict[str, float]:
    distance = np.asarray([row["distance"] for row in errors], dtype=np.float32)
    return {
        "count": int(len(distance)),
        "mean_error_patches": float(distance.mean()),
        "median_error_patches": float(np.median(distance)),
        "p90_error_patches": float(np.quantile(distance, 0.9)),
        "exact_top1": float(np.mean(distance < 1e-6)),
        "within_1_patch": float(np.mean(distance <= 1.0)),
        "within_sqrt2_patches": float(np.mean(distance <= math.sqrt(2.0))),
    }


def evaluate(
    owner: nn.Module,
    bank: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    rank: int,
) -> dict[str, Any]:
    bank.eval()
    local: dict[str, list[dict[str, float]]] = {}
    grid_h, grid_w = args.image_height // 16, args.image_width // 16
    with torch.no_grad():
        for batch in loader:
            frames = batch["frame"].to(device, non_blocking=True)
            center = batch["center"].to(device, non_blocking=True)
            features = patch_features(owner, frames, args.layers, args)
            with autocast_context(args):
                logits = bank(features)
            target = target_indices(center, grid_h, grid_w)
            target_y, target_x = target // grid_w, target % grid_w
            confidence = batch["confidence"].numpy()
            size = batch["size_px"].amax(dim=1).numpy()
            for name, values in logits.items():
                prediction = values.argmax(dim=1)
                pred_y, pred_x = prediction // grid_w, prediction % grid_w
                distance = torch.sqrt(
                    (pred_x - target_x).float().square()
                    + (pred_y - target_y).float().square()
                ).cpu().numpy()
                rows = local.setdefault(name, [])
                rows.extend(
                    {
                        "distance": float(d),
                        "confidence": float(c),
                        "size_px": float(s),
                    }
                    for d, c, s in zip(distance, confidence, size)
                )
    gathered: list[dict[str, list[dict[str, float]]]] = [dict() for _ in range(args.world_size)]
    if dist.is_initialized():
        dist.all_gather_object(gathered, local)
    else:
        gathered = [local]
    if rank != 0:
        return {}
    merged: dict[str, list[dict[str, float]]] = {}
    for shard in gathered:
        for name, rows in shard.items():
            merged.setdefault(name, []).extend(rows)
    result: dict[str, Any] = {}
    for name, rows in sorted(merged.items()):
        metrics: dict[str, Any] = {"all": summarize_errors(rows)}
        metrics["confidence_ge_0.7"] = summarize_errors(
            [row for row in rows if row["confidence"] >= 0.7]
        ) if any(row["confidence"] >= 0.7 for row in rows) else None
        metrics["ball_max_size_lt_8px"] = summarize_errors(
            [row for row in rows if row["size_px"] < 8.0]
        ) if any(row["size_px"] < 8.0 for row in rows) else None
        metrics["ball_max_size_ge_16px"] = summarize_errors(
            [row for row in rows if row["size_px"] >= 16.0]
        ) if any(row["size_px"] >= 16.0 for row in rows) else None
        result[name] = metrics
    return result


def spatial_prior_metrics(train: list[FrameRecord], val: list[FrameRecord], height: int, width: int) -> dict[str, float]:
    histogram = np.zeros((height, width), dtype=np.int64)
    for record in train:
        x = min(int(record.center_x * width), width - 1)
        y = min(int(record.center_y * height), height - 1)
        histogram[y, x] += 1
    prior_y, prior_x = np.unravel_index(int(histogram.argmax()), histogram.shape)
    errors = []
    for record in val:
        x = min(int(record.center_x * width), width - 1)
        y = min(int(record.center_y * height), height - 1)
        errors.append(
            {"distance": math.hypot(x - prior_x, y - prior_y), "confidence": record.confidence, "size_px": max(record.width_px, record.height_px)}
        )
    result = summarize_errors(errors)
    result.update({"prior_x": int(prior_x), "prior_y": int(prior_y)})
    return result


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank, device = distributed_setup()
    args.world_size = world_size
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    output_dir = Path(args.output_dir).expanduser()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    train_records, val_records, manifest = split_records(args)
    owner, cfg, _labels, _thresholds = load_checkpoint_model(
        str(Path(args.checkpoint).expanduser()), device, []
    )
    owner.to(device).eval()
    for parameter in owner.parameters():
        parameter.requires_grad = False
    image_size = tuple(int(value) for value in cfg.video.image_size)
    args.image_height, args.image_width = image_size
    if image_size != (720, 1280):
        raise ValueError(f"probe contract expects 720x1280, got {image_size}")

    train_dataset = BallFrameDataset(train_records, image_size)
    val_dataset = BallFrameDataset(val_records, image_size)
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )
    train_loader = make_loader(train_dataset, sampler=train_sampler, args=args, train=True)
    val_loader = make_loader(val_dataset, sampler=val_sampler, args=args, train=False)
    dimension = int(owner.backbone.num_features)
    bank: nn.Module = ProbeBank(args.layers, dimension, args.hidden_dim).to(device)
    if world_size > 1:
        bank = DistributedDataParallel(bank, device_ids=[local_rank])
    optimizer = torch.optim.AdamW(
        bank.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    if rank == 0:
        manifest.update(
            {
                "world_size": world_size,
                "batch_size_per_rank": args.batch_size,
                "epochs": (1 if args.smoke else args.epochs),
                "criterion": "dense_patch_cross_entropy",
                "controls": ["shuffled_coordinate_probe", "spatial_prior"],
            }
        )
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(manifest, ensure_ascii=False), flush=True)

    best_score = (float("inf"), float("inf"), float("inf"))
    epochs = 1 if args.smoke else args.epochs
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        train_sampler.set_epoch(epoch)
        train_loss = train_epoch(
            owner, bank, train_loader, optimizer, device, args, epoch, rank
        )
        metrics = evaluate(owner, bank, val_loader, device, args, rank)
        if rank == 0:
            prior = spatial_prior_metrics(
                train_records, val_records, image_size[0] // 16, image_size[1] // 16
            )
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "spatial_prior": prior,
                "probes": metrics,
            }
            history.append(row)
            (output_dir / f"metrics_epoch_{epoch:03d}.json").write_text(
                json.dumps(row, indent=2), encoding="utf-8"
            )
            true_heads = [
                (
                    value["all"]["median_error_patches"],
                    value["all"]["mean_error_patches"],
                    -value["all"]["within_1_patch"],
                )
                for name, value in metrics.items()
                if name.endswith("_true")
            ]
            current_score = min(true_heads)
            payload = {
                "epoch": epoch,
                "model": (bank.module if hasattr(bank, "module") else bank).state_dict(),
                "args": vars(args),
                "metrics": row,
            }
            torch.save(payload, output_dir / "last.pt")
            torch.save(payload, output_dir / f"epoch_{epoch:03d}.pt")
            if current_score < best_score:
                best_score = current_score
                torch.save(payload, output_dir / "best.pt")
                (output_dir / "best_metrics.json").write_text(
                    json.dumps(row, indent=2), encoding="utf-8"
                )
            print(
                f"epoch={epoch} train_loss={train_loss:.5f} "
                f"best_true_median_error_patches={current_score[0]:.4f} "
                f"best_true_mean_error_patches={current_score[1]:.4f}",
                flush=True,
            )
        if dist.is_initialized():
            dist.barrier()

    if rank == 0:
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
