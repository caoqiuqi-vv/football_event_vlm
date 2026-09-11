#!/usr/bin/env python
"""Audit how football SSL changes global, patch and detector-ROI features."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_football_events as train_mod  # noqa: E402
from scripts.analyze_football_representation_failures import (  # noqa: E402
    DetectionIndex,
    binary_auc,
    fit_fold_ridge_probe,
    grouped_probe,
)
from scripts.audit_football_backbone_adaptation import (  # noqa: E402
    TASK_GROUPS,
    find_video,
    read_cases,
    unique_windows,
)
from scripts.eval_long_video_checkpoint import SlidingWindowVideoDataset, collate_windows  # noqa: E402

ROI_SPECS = {
    "ball": {"class_id": 1, "threshold": 0.10, "topk": 3, "expand": 3.0, "min_patches": 3},
    "goal": {"class_id": 2, "threshold": 0.40, "topk": 1, "expand": 1.15, "min_patches": 2},
    "player": {"class_id": 0, "threshold": 0.35, "topk": 10, "expand": 1.10, "min_patches": 2},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="outputs/football_diagnostics/a_best_fp_root_cause_v1/cases.csv")
    parser.add_argument("--task-checkpoint", default="checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt")
    parser.add_argument("--base-backbone", default="checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
    parser.add_argument("--ssl-backbone", default="outputs/football_ssl/vitl16_fullft_wds_v2_head10x_8k/export/dinov3_vitl16_football_ssl_step7999_merged.pth")
    parser.add_argument("--video-root", default="/mnt/data_16t/football/raw_video_720P")
    parser.add_argument("--detection-root", default="/mnt/data_16t/football/roi_indices/conditional_goal_crowd_v1")
    parser.add_argument("--output-dir", default="outputs/football_ssl/vitl16_fullft_wds_v2_head10x_8k/representation_audit_v2")
    parser.add_argument("--layers", default="0,5,11,17,20,23")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--probe-components", type=int, default=32)
    parser.add_argument("--probe-ridge", type=float, default=1.0)
    parser.add_argument("--min-roi-frames", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reuse-features", action="store_true")
    return parser.parse_args()


def temporal_pool(values: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    values = values.float()
    if valid is None:
        mean = values.mean(1)
        std = values.std(1, unbiased=False)
        motion = (values[:, 1:] - values[:, :-1]).abs().mean(1)
    else:
        weights = valid.to(values.dtype).unsqueeze(-1)
        count = weights.sum(1).clamp_min(1.0)
        mean = (values * weights).sum(1) / count
        std = (((values - mean[:, None]) ** 2 * weights).sum(1) / count).sqrt()
        pairs = (valid[:, 1:] & valid[:, :-1]).to(values.dtype).unsqueeze(-1)
        motion = ((values[:, 1:] - values[:, :-1]).abs() * pairs).sum(1) / pairs.sum(1).clamp_min(1.0)
    return torch.cat((mean, std, motion), dim=-1)


def expanded_patch_bounds(
    box: np.ndarray,
    index: DetectionIndex,
    grid_h: int,
    grid_w: int,
    spec: dict[str, Any],
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = map(float, box)
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 2.0:
        x1, x2 = x1 / max(index.width, 1.0), x2 / max(index.width, 1.0)
        y1, y2 = y1 / max(index.height, 1.0), y2 / max(index.height, 1.0)
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    width = max((x2 - x1) * float(spec["expand"]), float(spec["min_patches"]) / grid_w)
    height = max((y2 - y1) * float(spec["expand"]), float(spec["min_patches"]) / grid_h)
    left = int(np.clip(math.floor((cx - width / 2) * grid_w), 0, grid_w - 1))
    right = int(np.clip(math.ceil((cx + width / 2) * grid_w), left + 1, grid_w))
    top = int(np.clip(math.floor((cy - height / 2) * grid_h), 0, grid_h - 1))
    bottom = int(np.clip(math.ceil((cy + height / 2) * grid_h), top + 1, grid_h))
    return left, top, right, bottom


def roi_pool(
    patches: torch.Tensor,
    frame_times: torch.Tensor,
    detection: DetectionIndex | None,
    grid_h: int,
    grid_w: int,
    roi_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, frames, _, dim = patches.shape
    output = torch.zeros((batch, frames, dim), device=patches.device, dtype=patches.dtype)
    valid = torch.zeros((batch, frames), device=patches.device, dtype=torch.bool)
    if detection is None:
        return output, valid
    spec = ROI_SPECS[roi_name]
    all_frame_ids = torch.round(frame_times.cpu() * max(detection.fps, 1.0)).long().numpy()
    for batch_index in range(batch):
        for frame_index, frame_id in enumerate(all_frame_ids[batch_index]):
            boxes, classes, confidences = detection.objects(int(frame_id))
            keep = np.flatnonzero((classes == spec["class_id"]) & (confidences >= spec["threshold"]))
            if not len(keep):
                continue
            keep = keep[np.argsort(confidences[keep])[::-1][: int(spec["topk"])]]
            mask = torch.zeros((grid_h, grid_w), device=patches.device, dtype=torch.bool)
            for object_index in keep:
                left, top, right, bottom = expanded_patch_bounds(
                    boxes[object_index], detection, grid_h, grid_w, spec
                )
                mask[top:bottom, left:right] = True
            if mask.any():
                output[batch_index, frame_index] = patches[batch_index, frame_index, mask.flatten()].mean(0)
                valid[batch_index, frame_index] = True
    return output, valid


def extract_layers(
    backbone: torch.nn.Module,
    inputs: torch.Tensor,
    frame_times: torch.Tensor,
    detection: DetectionIndex | None,
    layers: Sequence[int],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    batch, frames, channels, height, width = inputs.shape
    outputs = backbone.get_intermediate_layers(
        inputs.reshape(batch * frames, channels, height, width),
        n=tuple(layers),
        return_class_token=True,
        norm=True,
    )
    features: dict[str, torch.Tensor] = {}
    validity: dict[str, torch.Tensor] = {}
    for layer, (patches, cls) in zip(layers, outputs):
        patch_count = patches.shape[1]
        grid_h = height // 16
        grid_w = width // 16
        if grid_h * grid_w != patch_count:
            grid_h = int(round(math.sqrt(patch_count * height / width)))
            grid_w = patch_count // grid_h
        patches = patches.reshape(batch, frames, patch_count, -1)
        cls = cls.reshape(batch, frames, -1)
        features[f"layer{layer:02d}__cls"] = temporal_pool(cls)
        features[f"layer{layer:02d}__patch"] = temporal_pool(patches.mean(2))
        for roi_name in ROI_SPECS:
            roi, valid = roi_pool(patches, frame_times, detection, grid_h, grid_w, roi_name)
            features[f"layer{layer:02d}__{roi_name}"] = temporal_pool(roi, valid)
            validity[f"layer{layer:02d}__{roi_name}"] = valid.sum(1)
    return features, validity


def build_backbone(cfg: Any, weights: Path, device: torch.device) -> torch.nn.Module:
    model_cfg = copy.deepcopy(cfg)
    model_cfg.model["weights"] = str(weights)
    model_cfg.model.pretrained = True
    backbone = train_mod.build_backbone(model_cfg).to(device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    return backbone


def extract_all(
    cases: Sequence[dict[str, Any]],
    cfg: Any,
    backbones: dict[str, torch.nn.Module],
    layers: Sequence[int],
    video_root: Path,
    detection_root: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    case_lookup: dict[tuple[str, int], list[int]] = defaultdict(list)
    for case_index, case in enumerate(cases):
        case_lookup[(case["video_id"], case["window_index"])].append(case_index)
    values: dict[str, list[np.ndarray | None]] = {}
    valid_values: dict[str, list[float | None]] = {}
    image_size = train_mod.parse_image_size(cfg.video.image_size)
    num_frames = train_mod.effective_num_frames(cfg)
    windows_by_video = unique_windows(cases)
    for video_number, (video_id, windows) in enumerate(windows_by_video.items(), 1):
        index_path = detection_root / f"{video_id}.pt"
        detection = DetectionIndex(index_path) if index_path.exists() else None
        dataset = SlidingWindowVideoDataset(
            video_path=str(find_video(video_root, video_id)),
            video_id=video_id,
            windows=windows,
            num_frames=num_frames,
            image_size=image_size,
            normalize_on_cpu=True,
            decode_strategy="single_seek",
            video_reader_cache_size=1,
            window_cropper=None,
            view_mode="single",
            global_image_size=image_size,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=num_workers > 0,
            collate_fn=collate_windows,
        )
        with torch.inference_mode():
            for batch in loader:
                inputs = batch["inputs"].to(device, non_blocking=True)
                frame_times = batch["frame_times"]
                for backbone_name, backbone in backbones.items():
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        extracted, validity = extract_layers(
                            backbone, inputs, frame_times, detection, layers
                        )
                    for name, tensor in extracted.items():
                        key = f"{backbone_name}__{name}"
                        values.setdefault(key, [None] * len(cases))
                        for batch_index, meta in enumerate(batch["meta"]):
                            for case_index in case_lookup[(video_id, int(meta["index"]))]:
                                values[key][case_index] = tensor[batch_index].cpu().numpy().astype(np.float32)
                    for name, tensor in validity.items():
                        key = f"valid__{name}"
                        valid_values.setdefault(key, [None] * len(cases))
                        for batch_index, meta in enumerate(batch["meta"]):
                            for case_index in case_lookup[(video_id, int(meta["index"]))]:
                                valid_values[key][case_index] = float(tensor[batch_index].cpu())
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            f"extract video={video_id} windows={len(windows)} detection={detection is not None} "
            f"progress={video_number}/{len(windows_by_video)}",
            flush=True,
        )
    if any(item is None for items in values.values() for item in items):
        raise RuntimeError("Some cases did not receive features")
    features = {key: np.stack([item for item in items if item is not None]) for key, items in values.items()}
    validity = {key: np.asarray(items, dtype=np.float32) for key, items in valid_values.items()}
    return features, validity


def paired_geometry(base: np.ndarray, ssl: np.ndarray) -> dict[str, float]:
    numerator = np.sum(base * ssl, axis=1)
    denominator = np.linalg.norm(base, axis=1) * np.linalg.norm(ssl, axis=1)
    cosine = numerator / np.maximum(denominator, 1e-8)
    relative_l2 = np.linalg.norm(ssl - base, axis=1) / np.maximum(np.linalg.norm(base, axis=1), 1e-8)
    return {
        "paired_cosine_mean": float(np.mean(cosine)),
        "paired_cosine_p05": float(np.quantile(cosine, 0.05)),
        "relative_l2_mean": float(np.mean(relative_l2)),
    }


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    left = left.astype(np.float64) - left.mean(0, keepdims=True)
    right = right.astype(np.float64) - right.mean(0, keepdims=True)
    gram_left = left @ left.T
    gram_right = right @ right.T
    numerator = float(np.sum(gram_left * gram_right))
    denominator = math.sqrt(float(np.sum(gram_left**2) * np.sum(gram_right**2)))
    return numerator / max(denominator, 1e-12)


def stratified_random_probe(
    features: np.ndarray,
    targets: np.ndarray,
    max_components: int,
    ridge: float,
    folds: int = 5,
) -> dict[str, float]:
    fold_ids = np.full(len(targets), -1, dtype=np.int64)
    rng = np.random.default_rng(20260817)
    for target in (0, 1):
        indices = np.flatnonzero(targets == target)
        rng.shuffle(indices)
        fold_ids[indices] = np.arange(len(indices)) % folds
    scores = np.full(len(targets), np.nan, dtype=np.float64)
    used_folds = 0
    for fold in range(folds):
        test = fold_ids == fold
        train = ~test
        if not test.any() or len(np.unique(targets[train])) < 2:
            continue
        scores[test] = fit_fold_ridge_probe(features[train], targets[train], features[test], max_components, ridge)
        used_folds += 1
    valid = np.isfinite(scores)
    return {"random_probe_auc": binary_auc(targets[valid], scores[valid]), "random_probe_folds": used_folds}


def in_sample_probe(
    features: np.ndarray,
    targets: np.ndarray,
    max_components: int,
    ridge: float,
) -> float:
    scores = fit_fold_ridge_probe(features, targets, features, max_components, ridge)
    return binary_auc(targets, scores)


def geometry_report(
    features: dict[str, np.ndarray],
    validity: dict[str, np.ndarray],
    layers: Sequence[int],
    min_roi_frames: int,
) -> list[dict[str, Any]]:
    rows = []
    for layer in layers:
        for representation in ("cls", "patch", *ROI_SPECS):
            suffix = f"layer{layer:02d}__{representation}"
            base = features[f"original__{suffix}"]
            ssl = features[f"ssl__{suffix}"]
            valid = np.ones(len(base), dtype=bool)
            if representation in ROI_SPECS:
                valid &= validity[f"valid__{suffix}"] >= min_roi_frames
            unique = np.asarray([True] + [False] * (len(base) - 1))
            # cases.csv duplicates windows for different event labels; geometry must not overweight them.
            seen: set[tuple[bytes, bytes]] = set()
            for index, (left, right) in enumerate(zip(base, ssl)):
                key = (left[:8].tobytes(), right[:8].tobytes())
                unique[index] = key not in seen
                seen.add(key)
            mask = valid & unique
            if mask.sum() < 3:
                continue
            rows.append({
                "layer": layer,
                "representation": representation,
                "samples": int(mask.sum()),
                **paired_geometry(base[mask], ssl[mask]),
                "linear_cka": linear_cka(base[mask], ssl[mask]),
            })
    return rows


def probe_report(
    cases: Sequence[dict[str, Any]],
    features: dict[str, np.ndarray],
    validity: dict[str, np.ndarray],
    layers: Sequence[int],
    min_roi_frames: int,
    max_components: int,
    ridge: float,
) -> list[dict[str, Any]]:
    labels = np.asarray([case["label"] for case in cases])
    groups = np.asarray([case["group"] for case in cases])
    videos = np.asarray([case["video_id"] for case in cases])
    rows = []
    for task_name, (positive_groups, negative_groups) in TASK_GROUPS.items():
        allowed = positive_groups | negative_groups
        for label in sorted(set(labels.tolist())):
            task_mask = (labels == label) & np.isin(groups, list(allowed))
            for layer in layers:
                for representation in ("cls", "patch", *ROI_SPECS):
                    suffix = f"layer{layer:02d}__{representation}"
                    mask = task_mask.copy()
                    if representation in ROI_SPECS:
                        mask &= validity[f"valid__{suffix}"] >= min_roi_frames
                    targets = np.isin(groups[mask], list(positive_groups)).astype(np.int64)
                    if len(targets) < 8 or len(np.unique(targets)) < 2:
                        continue
                    for backbone_name in ("original", "ssl"):
                        values = features[f"{backbone_name}__{suffix}"][mask]
                        grouped = grouped_probe(values, targets, videos[mask], max_components, ridge)
                        random = stratified_random_probe(values, targets, max_components, ridge)
                        rows.append({
                            "task": task_name,
                            "label": label,
                            "backbone": backbone_name,
                            "layer": layer,
                            "representation": representation,
                            "num_positive": int(targets.sum()),
                            "num_negative": int((targets == 0).sum()),
                            **grouped,
                            **random,
                            "in_sample_auc": in_sample_probe(values, targets, max_components, ridge),
                            "random_minus_grouped_auc": random["random_probe_auc"] - grouped["probe_auc"],
                        })
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(geometry: Sequence[dict[str, Any]], probes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"geometry": {}, "probe_best": {}}
    for row in geometry:
        summary["geometry"][f"layer{row['layer']:02d}_{row['representation']}"] = row
    for task in TASK_GROUPS:
        summary["probe_best"][task] = {}
        for label in sorted({row["label"] for row in probes if row["task"] == task}):
            summary["probe_best"][task][label] = {}
            for representation in ("cls", "patch", *ROI_SPECS):
                selected = [row for row in probes if row["task"] == task and row["label"] == label and row["representation"] == representation]
                if not selected:
                    continue
                by_backbone = {}
                for backbone in ("original", "ssl"):
                    candidates = [row for row in selected if row["backbone"] == backbone]
                    if candidates:
                        by_backbone[backbone] = max(candidates, key=lambda item: item["probe_auc"])
                if "original" in by_backbone and "ssl" in by_backbone:
                    by_backbone["ssl_grouped_auc_gain"] = by_backbone["ssl"]["probe_auc"] - by_backbone["original"]["probe_auc"]
                summary["probe_best"][task][label][representation] = by_backbone
    return summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "features_fp16.npz"
    cases = read_cases(Path(args.cases).expanduser().resolve())
    layers = tuple(int(item) for item in args.layers.split(",") if item.strip())
    checkpoint = torch.load(args.task_checkpoint, map_location="cpu", weights_only=False)
    cfg = train_mod.to_config(checkpoint["config"])
    device = torch.device(args.device)
    metadata = {
        "cases": str(Path(args.cases).resolve()),
        "base_backbone": str(Path(args.base_backbone).resolve()),
        "ssl_backbone": str(Path(args.ssl_backbone).resolve()),
        "num_cases": len(cases),
        "num_unique_windows": sum(len(items) for items in unique_windows(cases).values()),
        "videos": sorted({case["video_id"] for case in cases}),
        "layers": list(layers),
        "roi_specs": ROI_SPECS,
        "min_roi_frames": args.min_roi_frames,
        "image_size": list(train_mod.parse_image_size(cfg.video.image_size)),
        "num_frames": train_mod.effective_num_frames(cfg),
    }
    (output_dir / "audit_config.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    if args.reuse_features and feature_path.exists():
        loaded = np.load(feature_path)
        features = {key: loaded[key].astype(np.float32) for key in loaded.files if not key.startswith("valid__")}
        validity = {key: loaded[key].astype(np.float32) for key in loaded.files if key.startswith("valid__")}
    else:
        backbones = {
            "original": build_backbone(cfg, Path(args.base_backbone), device),
            "ssl": build_backbone(cfg, Path(args.ssl_backbone), device),
        }
        features, validity = extract_all(
            cases, cfg, backbones, layers, Path(args.video_root), Path(args.detection_root),
            device, args.batch_size, args.num_workers,
        )
        np.savez_compressed(
            feature_path,
            **{key: value.astype(np.float16) for key, value in {**features, **validity}.items()},
        )
    geometry = geometry_report(features, validity, layers, args.min_roi_frames)
    probes = probe_report(
        cases, features, validity, layers, args.min_roi_frames, args.probe_components, args.probe_ridge
    )
    write_csv(output_dir / "geometry_report.csv", geometry)
    write_csv(output_dir / "probe_report.csv", probes)
    summary = summarize(geometry, probes)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float))
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=float), flush=True)
    print(f"wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
