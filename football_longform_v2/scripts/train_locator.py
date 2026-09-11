from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from football_longform_v2.canonical import load_canonical_split
from football_longform_v2.config import load_config
from football_longform_v2.data import VideoBalancedTimelineDataset, VideoCohortBatchSampler
from football_longform_v2.feature_store import assert_timeline_provenance
from football_longform_v2.losses import locator_loss
from football_longform_v2.models import TemporalLocator
from football_longform_v2.models.temporal_locator import MODEL_SCHEMA
from football_longform_v2.schema import read_video_ids


def resolve(root: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_data_parallel_device_ids(raw: str | None, primary_device: str) -> tuple[int, ...]:
    if not raw:
        return ()
    try:
        device_ids = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError as error:
        raise ValueError("--device-ids must be a comma-separated integer list") from error
    if not device_ids or len(device_ids) != len(set(device_ids)) or min(device_ids) < 0:
        raise ValueError("--device-ids must contain unique non-negative GPU indices")
    expected_primary = f"cuda:{device_ids[0]}"
    if primary_device != expected_primary:
        raise ValueError(
            f"--device must equal first data-parallel device {expected_primary}"
        )
    return device_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--device-ids",
        help="Optional physical CUDA indices for single-process DataParallel, e.g. 6,7.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--blocks-per-video", type=int)
    parser.add_argument("--feature-split", default="train")
    parser.add_argument("--expected-ready-count", type=int)
    parser.add_argument(
        "--train-id-file",
        help="Optional canonical-train subset for leakage-safe video-grouped OOF training.",
    )
    parser.add_argument("--checkpoint-prefix", default="locator")
    parser.add_argument("--positive-alpha", type=float, default=0.75)
    parser.add_argument("--density-balanced-focal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--class-loss-weight", type=float, default=0.5)
    parser.add_argument("--state-loss-weight", type=float, default=0.1)
    args = parser.parse_args()
    data_parallel_device_ids = parse_data_parallel_device_ids(args.device_ids, args.device)
    if data_parallel_device_ids:
        if not torch.cuda.is_available():
            raise RuntimeError("--device-ids requires CUDA")
        unavailable = [
            index for index in data_parallel_device_ids
            if index >= torch.cuda.device_count()
        ]
        if unavailable:
            raise RuntimeError(f"requested unavailable CUDA device IDs: {unavailable}")

    if args.feature_split != "train":
        raise ValueError("training is train-only; calibration and thirdparty18 are forbidden")
    config = load_config(args.config)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    root = Path(config["_project_root"])
    paths = config["paths"]
    train = config["train"]
    canonical = load_canonical_split(resolve(root, paths["canonical_manifest"]), "train")
    configured_ids = read_video_ids(resolve(root, paths["train_ids"]))
    if canonical.media_ids != configured_ids:
        raise RuntimeError("canonical manifest and train ID list disagree")
    train_id_path = resolve(root, args.train_id_file) if args.train_id_file else None
    training_ids = read_video_ids(train_id_path) if train_id_path else configured_ids
    if not training_ids:
        raise RuntimeError("training ID subset is empty")
    unknown_training_ids = sorted(set(training_ids) - set(configured_ids))
    if unknown_training_ids:
        raise RuntimeError(
            f"training ID subset contains non-canonical IDs: {unknown_training_ids[:5]}"
        )
    held_out_train_ids = tuple(video_id for video_id in configured_ids if video_id not in set(training_ids))
    training_ids_sha256 = hashlib.sha256(
        ("\n".join(training_ids) + "\n").encode("utf-8")
    ).hexdigest()
    config_sha256 = sha256_file(Path(config["_config_path"]))
    context_config = config["features"]["context"]
    weights_path = resolve(root, str(context_config["weights"]))
    weights_sha256 = sha256_file(weights_path)
    feature_root = resolve(root, paths["feature_store"]) / args.feature_split
    for video_id in training_ids:
        assert_timeline_provenance(
            feature_root / video_id / "timeline.npz",
            expected_backbone_arch=str(context_config["arch"]),
            expected_backbone_id=str(context_config["backbone_id"]),
            expected_source_video=canonical.source_video_by_media_id[video_id],
            expected_config_sha256=config_sha256,
            expected_weights_sha256=weights_sha256,
        )
    targets_config = config.get("targets", {})
    dataset = VideoBalancedTimelineDataset(
        training_ids,
        feature_store=resolve(root, paths["feature_store"]),
        annotations=resolve(root, paths["annotations"]),
        families=tuple(config["task"]["proposal_families"]),
        timeline_hz=float(config["features"]["timeline_hz"]),
        sequence_seconds=float(train["sequence_seconds"]),
        labels=tuple(config["task"]["output_labels"]),
        blocks_per_video=(
            args.blocks_per_video
            if args.blocks_per_video is not None
            else int(train["blocks_per_video"])
        ),
        positive_probability=float(train["positive_probability"]),
        jitter_seconds=float(train["temporal_jitter_seconds"]),
        seed=int(config["seed"]),
        feature_split=args.feature_split,
        annotation_id_by_video_id={
            video_id: canonical.annotation_id_by_media_id[video_id]
            for video_id in training_ids
        },
        sigma_seconds=targets_config.get("sigma_seconds"),
        rejected_ignore_radius_seconds=targets_config.get("ignore_radius_seconds"),
        class_sigma_seconds=targets_config.get("class_sigma_seconds"),
        rejected_class_ignore_radius_seconds=targets_config.get("class_ignore_radius_seconds"),
        class_state_sigma_seconds=targets_config.get("class_state_sigma_seconds"),
        require_reviewed_annotations=False,
        require_all_videos=True,
    )
    if args.expected_ready_count is not None and len(dataset.video_ids) != args.expected_ready_count:
        raise RuntimeError(
            f"ready train cache count {len(dataset.video_ids)} != expected {args.expected_ready_count}; "
            "refusing partial or aliased training"
        )
    batch_sampler = VideoCohortBatchSampler(
        len(dataset.video_ids), dataset.blocks_per_video,
        batch_size=int(train["batch_size"]), worker_count=args.workers, seed=int(config["seed"]),
    )
    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"),
    )
    device = torch.device(args.device)
    model = TemporalLocator.from_config(config).to(device)
    if len(data_parallel_device_ids) > 1:
        model = torch.nn.DataParallel(
            model, device_ids=list(data_parallel_device_ids),
            output_device=data_parallel_device_ids[0],
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train["learning_rate"]),
        weight_decay=float(train["weight_decay"]),
    )
    output_dir = resolve(root, paths["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        dataset.set_epoch(epoch)
        batch_sampler.set_epoch(epoch)
        model.train()
        running = {
            key: 0.0 for key in (
                "loss", "heatmap_loss", "offset_loss",
                "class_heatmap_loss", "state_heatmap_loss",
            )
        }
        for step, batch in enumerate(loader, start=1):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            outputs = model(
                batch["context"],
                batch["motion"],
                batch["timestamps"],
                context_valid=batch["context_valid"],
                motion_valid=batch["motion_valid"],
            )
            losses = locator_loss(
                outputs,
                batch["targets"],
                offset_targets=batch["offset_targets"],
                valid_mask=batch["valid"],
                alpha=float(args.positive_alpha),
                gamma=float(train["focal_gamma"]),
                density_balanced=bool(args.density_balanced_focal),
                class_targets=batch["class_targets"],
                class_valid_mask=batch["class_valid"],
                class_weight=float(args.class_loss_weight),
                state_targets=batch["class_state_targets"],
                state_valid_mask=batch["class_valid"],
                state_weight=float(args.state_loss_weight),
            )
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            optimizer.step()
            for key in running:
                running[key] += float(losses[key].detach())
            if step % 20 == 0:
                means = {key: value / step for key, value in running.items()}
                print(
                    f"epoch={epoch} step={step} loss={means['loss']:.6f} "
                    f"family={means['heatmap_loss']:.6f} class={means['class_heatmap_loss']:.6f} "
                    f"state={means['state_heatmap_loss']:.6f} offset={means['offset_loss']:.6f}",
                    flush=True,
                )
        epoch_steps = max(len(loader), 1)
        mean_losses = {key: value / epoch_steps for key, value in running.items()}
        checkpoint = {
            "epoch": epoch,
            "model_schema": MODEL_SCHEMA,
            "output_labels": tuple(config["task"]["output_labels"]),
            "proposal_families": tuple(config["task"]["proposal_families"]),
            "model": (
                model.module.state_dict()
                if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            ),
            "training_device_ids": data_parallel_device_ids or (device.index,),
            "data_parallel": len(data_parallel_device_ids) > 1,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "optimizer": optimizer.state_dict(),
            "train_steps": len(loader),
            "mean_train_losses": mean_losses,
            "config": config,
            "train_video_count": len(dataset.video_ids),
            "canonical_train_video_count": len(configured_ids),
            "training_media_ids": tuple(training_ids),
            "training_media_ids_sha256": training_ids_sha256,
            "train_id_file": str(train_id_path) if train_id_path else None,
            "held_out_train_video_ids": held_out_train_ids,
            "requested_train_video_count": len(dataset.requested_video_ids),
            "missing_train_video_ids": dataset.missing_video_ids,
            "feature_split": args.feature_split,
            "feature_config_sha256": config_sha256,
            "feature_weights_sha256": weights_sha256,
            "training_objective": {
                "positive_alpha": float(args.positive_alpha),
                "focal_gamma": float(train["focal_gamma"]),
                "density_balanced_focal": bool(args.density_balanced_focal),
                "class_loss_weight": float(args.class_loss_weight),
                "state_loss_weight": float(args.state_loss_weight),
                "restart_state_supervision": "broad_gaussian_8s_penalty10s",
                "positive_sampling_unit": "event_label_within_video",
                "batch_sampler": "video_cohort_worker_affine_lru_v1",
                "save_conditioning": "past_3s_shot_max_and_peak_lag",
                "rejected_event_policy": "family_radius_ignore",
                "reviewed_annotations_required": False,
                "reviewed_false_anchors_respected": True,
            },
        }
        torch.save(checkpoint, output_dir / f"{args.checkpoint_prefix}_epoch_{epoch:03d}.pt")
        print(
            f"epoch={epoch} mean_loss={mean_losses['loss']:.6f} "
            f"family={mean_losses['heatmap_loss']:.6f} "
            f"class={mean_losses['class_heatmap_loss']:.6f} "
            f"state={mean_losses['state_heatmap_loss']:.6f} "
            f"offset={mean_losses['offset_loss']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
