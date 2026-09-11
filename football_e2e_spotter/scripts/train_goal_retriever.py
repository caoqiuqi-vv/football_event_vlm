#!/usr/bin/env python
"""Multi-GPU training for the long-context goal-oriented retriever."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "football_e2e_spotter" / "src"
for path in (ROOT, PKG):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from football_e2e_spotter.goal_feature_data import GoalFeatureWindowDataset, collate_goal_windows  # noqa: E402
from football_e2e_spotter.goal_matching import retriever_loss  # noqa: E402
from football_e2e_spotter.goal_retriever import GoalLongContextRetriever, RetrieverGeometry  # noqa: E402
from football_e2e_spotter.goal_retriever_eval import evaluate_predictions, scan_video  # noqa: E402


def distributed_setup() -> tuple[int, int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return rank, world, local_rank, torch.device(f"cuda:{local_rank}")


def seed_all(seed: int, rank: int) -> None:
    value = seed + rank * 10_007
    random.seed(value); np.random.seed(value); torch.manual_seed(value); torch.cuda.manual_seed_all(value)


def cosine_multiplier(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def atomic_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def validate(model: GoalLongContextRetriever, args: argparse.Namespace, device: torch.device, epoch: int) -> dict:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    ready_ids = []
    for item in manifest["calibration"]:
        video_id = str(item["media_id"])
        base = args.feature_root / "calibration" / video_id
        if (base / "timeline.npz").is_file() or (base / "metadata.json").is_file():
            ready_ids.append(video_id)
    predictions, durations = [], {}
    model.eval()
    for video_id in ready_ids:
        rows, duration = scan_video(
            model, feature_base=args.feature_root / "calibration" / video_id,
            video_id=video_id, device=device, batch_size=args.eval_batch_size,
        )
        predictions.extend(rows); durations[video_id] = duration
    report = evaluate_predictions(
        predictions, manifest=args.manifest, split="calibration", durations=durations,
        recall_floors={"shot": args.shot_recall_floor, "save": args.other_recall_floor,
                       "freekick": args.other_recall_floor, "corner": args.other_recall_floor,
                       "kickoff": args.other_recall_floor},
    )
    report["epoch"] = epoch
    report["calibration_videos"] = len(ready_ids)
    output = args.output / f"calibration_epoch{epoch:02d}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def score_report(report: dict) -> tuple:
    classes = report["classes"]
    reached = sum(bool(row["recall_floor_reachable"]) for row in classes.values())
    recall = sum(float(row["recall"]) for row in classes.values())
    precision = sum(float(row["precision"]) for row in classes.values())
    return (bool(report["gate_pass"]), reached, -float(report["review_ratio"]), recall, precision)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=12, help="per GPU")
    parser.add_argument("--eval-batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--shot-recall-floor", type=float, default=0.97)
    parser.add_argument("--other-recall-floor", type=float, default=0.94)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.manifest = args.manifest.resolve(); args.feature_root = args.feature_root.resolve(); args.output = args.output.resolve()
    rank, world, local_rank, device = distributed_setup()
    seed_all(args.seed, rank)
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "run_config.json").write_text(json.dumps({
            **vars(args), "manifest": str(args.manifest), "feature_root": str(args.feature_root),
            "output": str(args.output), "resume": str(args.resume) if args.resume else None,
            "world_size": world, "protocol": "180s context / 60s core / 2 anchors per 2s / Hungarian / no NMS",
        }, default=str, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    dataset = GoalFeatureWindowDataset(
        manifest=args.manifest, feature_root=args.feature_root, split="train", seed=args.seed,
    )
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=args.seed) if world > 1 else None
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler, shuffle=sampler is None,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
        collate_fn=collate_goal_windows, drop_last=True,
    )
    geometry = RetrieverGeometry()
    model = GoalLongContextRetriever(
        dataset.appearance_dim, dataset.motion_dim, dataset.audio_dim,
        hidden_dim=args.hidden_dim, geometry=geometry,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    start_epoch = 0; best_score = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"]); optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]); best_score = tuple(checkpoint.get("best_score", ())) or None
    wrapped = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=False) if world > 1 else model
    scaler = torch.amp.GradScaler("cuda")
    total_steps = max(args.epochs * len(loader), 1); warmup = max(int(0.05 * total_steps), 100)
    global_step = start_epoch * len(loader)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        dataset.set_epoch(epoch)
        if sampler is not None:
            sampler.set_epoch(epoch)
        wrapped.train(); sums = {}; seen = 0; started = time.time()
        for batch in loader:
            multiplier = cosine_multiplier(global_step, total_steps, warmup)
            for group in optimizer.param_groups:
                group["lr"] = args.lr * multiplier
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = wrapped(
                    batch["appearance"].to(device, non_blocking=True),
                    batch["motion"].to(device, non_blocking=True),
                    batch["audio"].to(device, non_blocking=True),
                    batch["valid"].to(device, non_blocking=True),
                )
                loss, stats = retriever_loss(output, batch["targets"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(wrapped.parameters(), 2.0)
            scaler.step(optimizer); scaler.update(); global_step += 1
            for key, value in stats.items():
                sums[key] = sums.get(key, 0.0) + value
            seen += 1
            if rank == 0 and seen % 50 == 0:
                print(json.dumps({"epoch": epoch, "step": seen, "lr": optimizer.param_groups[0]["lr"], **{key: value / seen for key, value in sums.items()}}, ensure_ascii=False), flush=True)
        if world > 1:
            dist.barrier()
        if rank == 0:
            report = validate(model, args, device, epoch)
            current_score = score_report(report)
            is_best = best_score is None or current_score > best_score
            if is_best:
                best_score = current_score
            payload = {
                "schema": "football.goal_long_context_retriever.v1", "epoch": epoch,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "model_config": {"appearance_dim": dataset.appearance_dim, "motion_dim": dataset.motion_dim,
                                 "audio_dim": dataset.audio_dim, "hidden_dim": args.hidden_dim},
                "best_score": best_score, "calibration": report,
            }
            atomic_checkpoint(args.output / "last.pt", payload)
            if is_best:
                atomic_checkpoint(args.output / "best.pt", payload)
            print(json.dumps({
                "epoch": epoch, "training_seconds": round(time.time() - started, 1),
                "review_ratio": report["review_ratio"], "gate_pass": report["gate_pass"],
                "recall": {label: row["recall"] for label, row in report["classes"].items()},
                "best": is_best,
            }, ensure_ascii=False), flush=True)
        if world > 1:
            dist.barrier()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

