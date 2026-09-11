from __future__ import annotations

"""Train the long-context locator on once-computed VideoMAE tubelets."""

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from football_longform_v2.canonical import load_canonical_split  # noqa: E402
from football_longform_v2.chunk_data import SequentialChunkDataset  # noqa: E402
from football_longform_v2.chunk_evaluation import evaluate_sequential_chunk_locator  # noqa: E402
from football_longform_v2.data import VideoCohortBatchSampler  # noqa: E402
from football_longform_v2.losses import sequential_chunk_locator_loss  # noqa: E402
from football_longform_v2.models import SequentialChunkLocator  # noqa: E402


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("config must be a mapping")
    return payload


def resolve(raw: str) -> Path:
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_device_ids(raw: str | None, device: str) -> tuple[int, ...]:
    if not raw:
        return ()
    ids = tuple(int(item) for item in raw.split(",") if item.strip())
    if not ids or len(ids) != len(set(ids)) or min(ids) < 0:
        raise ValueError("--device-ids must be unique non-negative CUDA indices")
    if device != f"cuda:{ids[0]}":
        raise ValueError("--device must equal the first data-parallel device")
    return ids


def read_id_file(path: Path) -> tuple[str, ...]:
    values = tuple(
        line.strip() for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not values or len(values) != len(set(values)):
        raise ValueError(f"ID file is empty or contains duplicates: {path}")
    return values


def dataset_from_config(
    config: dict[str, Any],
    split: str,
    *,
    canonical_split: str | None = None,
    store_split: str | None = None,
    video_ids_override: tuple[str, ...] | None = None,
) -> SequentialChunkDataset:
    data = config["data"]
    canonical_name = canonical_split or split
    store_name = store_split or split
    canonical = load_canonical_split(resolve(data["canonical_manifest"]), canonical_name)
    video_ids = tuple(video_ids_override or canonical.media_ids)
    unknown = sorted(set(video_ids) - set(canonical.media_ids))
    if unknown:
        raise ValueError(
            f"{split} override contains non-{canonical_name} canonical IDs: {unknown[:5]}"
        )
    targets = config.get("targets", {})
    training = config["train"]
    return SequentialChunkDataset(
        video_ids,
        store_root=resolve(data["store_root"]),
        split=store_name,
        annotations=resolve(data["annotations"]),
        annotation_id_by_video_id=canonical.annotation_id_by_media_id,
        labels=tuple(config["task"]["labels"]),
        families=tuple(config["task"]["families"]),
        block_seconds=float(training["block_seconds"]),
        blocks_per_video=int(training["blocks_per_video"]),
        positive_probability=float(training["positive_probability"]),
        jitter_seconds=float(training["jitter_seconds"]),
        seed=int(config["seed"]),
        max_offset_seconds=float(config["model"]["max_offset_seconds"]),
        class_sigma_seconds=targets.get("class_sigma_seconds"),
        family_sigma_seconds=targets.get("family_sigma_seconds"),
        class_ignore_radius_seconds=targets.get("class_ignore_radius_seconds"),
        family_ignore_radius_seconds=targets.get("family_ignore_radius_seconds"),
        require_all_videos=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--device-ids")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--train-id-file")
    parser.add_argument("--validation-id-file")
    parser.add_argument(
        "--validation-canonical-split", choices=("train", "calibration"),
        default="calibration",
    )
    parser.add_argument(
        "--validation-store-split", choices=("train", "calibration"),
        default="calibration",
    )
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--evaluate-only-final",
        action="store_true",
        help="Use a fixed epoch budget and only run held-out inference at the final epoch.",
    )
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_yaml(config_path)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device_ids = parse_device_ids(args.device_ids, args.device)
    device = torch.device(args.device)
    train_id_path = Path(args.train_id_file).expanduser().resolve() if args.train_id_file else None
    validation_id_path = (
        Path(args.validation_id_file).expanduser().resolve()
        if args.validation_id_file else None
    )
    train_ids = read_id_file(train_id_path) if train_id_path else None
    validation_ids = read_id_file(validation_id_path) if validation_id_path else None
    train_dataset = dataset_from_config(
        config, "train", video_ids_override=train_ids
    )
    calibration_dataset = dataset_from_config(
        config,
        "calibration",
        canonical_split=args.validation_canonical_split,
        store_split=args.validation_store_split,
        video_ids_override=validation_ids,
    )
    overlap = sorted(set(train_dataset.video_ids) & set(calibration_dataset.video_ids))
    if overlap:
        raise RuntimeError(f"train/validation video leakage: {overlap[:5]}")
    if train_dataset.feature_dim != calibration_dataset.feature_dim:
        raise RuntimeError("train/calibration feature dimensions disagree")
    if train_dataset.checkpoint_sha256 != calibration_dataset.checkpoint_sha256:
        raise RuntimeError("train/calibration stores came from different VideoMAE checkpoints")
    labels = tuple(config["task"]["labels"])
    families = tuple(config["task"]["families"])
    if labels != SequentialChunkLocator.LABELS or families != SequentialChunkLocator.FAMILIES:
        raise RuntimeError("config label/family order disagrees with model schema")

    training = config["train"]
    workers = int(args.workers if args.workers is not None else training["workers"])
    batch_sampler = VideoCohortBatchSampler(
        len(train_dataset.video_ids), train_dataset.blocks_per_video,
        batch_size=int(training["batch_size"]), worker_count=workers, seed=seed,
    )
    loader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    model_config = config["model"]
    model = SequentialChunkLocator(
        input_dim=train_dataset.feature_dim,
        hidden_dim=int(model_config["hidden_dim"]),
        dilations=tuple(model_config["dilations"]),
        dropout=float(model_config["dropout"]),
        shot_history_steps=int(model_config["shot_history_steps"]),
        max_offset_seconds=float(model_config["max_offset_seconds"]),
        detach_shot_context=bool(model_config["detach_shot_context"]),
        tubelets_per_chunk=int(model_config["tubelets_per_chunk"]),
    ).to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=list(device_ids), output_device=device_ids[0])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    epochs = int(args.epochs if args.epochs is not None else training["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=float(training.get("minimum_learning_rate", 0.0))
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir else resolve(config["output_dir"])
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    loss_config = config["loss"]
    evaluation = config["evaluation"]
    deployment_gate = config["deployment_gate"]
    amp_dtype = torch.bfloat16 if str(training.get("amp_dtype", "bf16")) == "bf16" else torch.float16
    amp_enabled = device.type == "cuda" and bool(training.get("amp", True))
    best_selection: tuple[float, ...] | None = None
    best_epoch = -1
    patience = int(training.get("early_stopping_patience", epochs))

    for epoch in range(epochs):
        train_dataset.set_epoch(epoch)
        batch_sampler.set_epoch(epoch)
        model.train()
        running = {key: 0.0 for key in (
            "loss", "class_heatmap_loss", "family_heatmap_loss", "offset_loss", "hierarchy_loss"
        )}
        for step, batch in enumerate(loader, start=1):
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items() if key != "video_index"
            }
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                outputs = model(
                    batch["features"], batch["valid"], batch["chunk_phase"]
                )
                losses = sequential_chunk_locator_loss(
                    outputs,
                    class_targets=batch["class_targets"],
                    family_targets=batch["family_targets"],
                    offset_targets=batch["offset_targets"],
                    class_valid_mask=batch["class_valid"],
                    family_valid_mask=batch["family_valid"],
                    alpha=float(loss_config["positive_alpha"]),
                    gamma=float(loss_config["focal_gamma"]),
                    family_weight=float(loss_config["family_weight"]),
                    offset_weight=float(loss_config["offset_weight"]),
                    hierarchy_weight=float(loss_config["hierarchy_weight"]),
                )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["grad_clip_norm"]))
            optimizer.step()
            for key in running:
                running[key] += float(losses[key].detach())
            if step % int(training["log_interval"]) == 0:
                values = " ".join(f"{key}={running[key] / step:.6f}" for key in running)
                print(f"epoch={epoch + 1} step={step}/{len(loader)} {values}", flush=True)
        scheduler.step()
        means = {key: value / max(len(loader), 1) for key, value in running.items()}
        if args.evaluate_only_final and epoch + 1 < epochs:
            print(
                f"epoch={epoch + 1} mean_loss={means['loss']:.6f} "
                "validation=deferred_fixed_epoch_oof",
                flush=True,
            )
            continue
        base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        report = evaluate_sequential_chunk_locator(
            base_model,
            calibration_dataset,
            device=device,
            nms_radius_seconds=evaluation["nms_radius_seconds"],
            target_recall=evaluation["target_recall"],
            minimum_precision_at_target_recall=deployment_gate[
                "minimum_precision_at_target_recall"
            ],
            maximum_fp_per_minute=deployment_gate["maximum_fp_per_minute"],
            tolerances_seconds=tuple(float(item) for item in evaluation["tolerances_seconds"]),
            operating_tolerance_seconds=float(evaluation["operating_tolerance_seconds"]),
            uniform_baseline_intervals_seconds=tuple(
                float(item) for item in evaluation.get(
                    "uniform_baseline_intervals_seconds", (3.0, 4.0)
                )
            ),
            amp_dtype=amp_dtype if amp_enabled else None,
        )
        report["checkpoint_epoch"] = epoch + 1
        report["feature_checkpoint_sha256"] = train_dataset.checkpoint_sha256
        selection = tuple(float(item) for item in report["selection_tuple"])
        checkpoint = {
            "schema": "football_longform_v2.sequential_chunk_locator.checkpoint.v1",
            "epoch": epoch + 1,
            "model": base_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "model_config": model_config,
            "labels": labels,
            "families": families,
            "feature_dim": train_dataset.feature_dim,
            "tubelet_seconds": train_dataset.step_seconds,
            "tubelets_per_chunk": train_dataset.tubelets_per_chunk,
            "feature_checkpoint_sha256": train_dataset.checkpoint_sha256,
            "train_video_ids": train_dataset.video_ids,
            "calibration_video_ids": calibration_dataset.video_ids,
            "train_id_file": str(train_id_path) if train_id_path else None,
            "validation_id_file": str(validation_id_path) if validation_id_path else None,
            "validation_canonical_split": args.validation_canonical_split,
            "validation_store_split": args.validation_store_split,
            "config": config,
            "config_sha256": sha256_file(config_path),
            "mean_train_losses": means,
            "selection_tuple": selection,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "hard_negative_mining": False,
            "detection_or_tracking_input": False,
            "evaluate_only_final": bool(args.evaluate_only_final),
        }
        epoch_path = output_dir / f"epoch_{epoch + 1:03d}.pt"
        torch.save(checkpoint, epoch_path)
        (output_dir / f"calibration_epoch_{epoch + 1:03d}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        improved = best_selection is None or selection > best_selection
        if improved:
            best_selection = selection
            best_epoch = epoch + 1
            torch.save(checkpoint, output_dir / "best.pt")
        (output_dir / "checkpoint_selection.json").write_text(json.dumps({
            "best_epoch": best_epoch,
            "best_selection_tuple": best_selection,
            "latest_epoch": epoch + 1,
            "selection_rule": report["selection_rule"],
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(
            f"epoch={epoch + 1} mean_loss={means['loss']:.6f} "
            f"selection={selection} best_epoch={best_epoch}", flush=True
        )
        if epoch + 1 - best_epoch >= patience:
            print(f"early_stop epoch={epoch + 1} patience={patience}", flush=True)
            break


if __name__ == "__main__":
    main()
