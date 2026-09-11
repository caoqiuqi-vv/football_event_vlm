#!/usr/bin/env python
"""Audit base-DINO vs football-LoRA separability across backbone depths.

The audit intentionally uses only full-image frames and existing TP/FP/FN/
hard-TN cases. It answers whether supervised football adaptation improved the
backbone representation, and whether useful evidence lives in intermediate
layers but is lost by the current last-layer readout.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
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
    grouped_probe,
)
from scripts.eval_long_video_checkpoint import (  # noqa: E402
    SlidingWindowVideoDataset,
    WindowRecord,
    collate_windows,
    load_checkpoint_model,
)


VALID_GROUPS = ("TP", "FP", "FN", "hard_TN")
TASK_GROUPS = {
    "tp_vs_fp": ({"TP"}, {"FP"}),
    "all_positive_vs_all_negative": ({"TP", "FN"}, {"FP", "hard_TN"}),
    "fn_vs_hard_tn": ({"FN"}, {"hard_TN"}),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        default="outputs/football_diagnostics/a_best_fp_root_cause_v1/cases.csv",
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/football_events/vitl16_strong_lora12_mlp_16f_hr_e1/best.pt",
    )
    parser.add_argument(
        "--base-backbone",
        default="",
        help="Optional plain DINO backbone used as the reference instead of checkpoint config model.weights.",
    )
    parser.add_argument(
        "--adapted-backbone",
        default="",
        help="Optional plain adapted backbone. When set, the supervised checkpoint is used only for data config.",
    )
    parser.add_argument("--adapted-name", default="football_ssl")
    parser.add_argument("--video-root", default="/mnt/data_16t/football/raw_video_720P")
    parser.add_argument(
        "--output-dir",
        default="outputs/football_diagnostics/backbone_adaptation_audit_v1",
    )
    parser.add_argument("--layers", default="0,5,11,17,20,23")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--probe-components", type=int, default=32)
    parser.add_argument("--probe-ridge", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--save-features", action="store_true")
    return parser.parse_args()


def read_cases(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = []
    for row in rows:
        if row["group"] not in VALID_GROUPS:
            continue
        result.append(
            {
                **row,
                "window_index": int(row["window_index"]),
                "start_sec": float(row["start_sec"]),
                "end_sec": float(row["end_sec"]),
            }
        )
    return result


def find_video(video_root: Path, video_id: str) -> Path:
    for suffix in (".mp4", ".mov", ".mkv", ".avi", ".MP4", ".MOV"):
        path = video_root / f"{video_id}{suffix}"
        if path.exists():
            return path
    matches = sorted(video_root.glob(f"{video_id}.*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Missing video={video_id} under {video_root}")


def unique_windows(cases: Sequence[dict[str, Any]]) -> dict[str, list[WindowRecord]]:
    grouped: dict[str, dict[int, WindowRecord]] = defaultdict(dict)
    for case in cases:
        grouped[case["video_id"]][case["window_index"]] = WindowRecord(
            index=case["window_index"],
            start_sec=case["start_sec"],
            end_sec=case["end_sec"],
        )
    return {
        video_id: [items[index] for index in sorted(items)]
        for video_id, items in grouped.items()
    }


def temporal_statistics(frame_features: torch.Tensor) -> torch.Tensor:
    values = frame_features.float()
    mean = values.mean(dim=1)
    maximum = values.amax(dim=1)
    std = values.std(dim=1, unbiased=False)
    motion = (
        (values[:, 1:] - values[:, :-1]).abs().mean(dim=1)
        if values.shape[1] > 1
        else torch.zeros_like(mean)
    )
    return torch.cat((mean, maximum, std, motion), dim=-1)


def extract_layers(
    backbone: torch.nn.Module,
    inputs: torch.Tensor,
    layers: Sequence[int],
) -> dict[int, torch.Tensor]:
    batch, frames, channels, height, width = inputs.shape
    flat = inputs.reshape(batch * frames, channels, height, width)
    outputs = backbone.get_intermediate_layers(
        flat,
        n=tuple(layers),
        return_class_token=True,
        norm=True,
    )
    result = {}
    for layer, (patches, cls) in zip(layers, outputs):
        frame_features = torch.cat((cls, patches.mean(dim=1)), dim=-1)
        frame_features = frame_features.reshape(batch, frames, -1)
        result[layer] = temporal_statistics(frame_features)
    return result


def extract_all(
    cases: Sequence[dict[str, Any]],
    backbones: dict[str, torch.nn.Module],
    cfg: Any,
    layers: Sequence[int],
    video_root: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> dict[str, np.ndarray]:
    case_lookup: dict[tuple[str, int], list[int]] = defaultdict(list)
    for case_index, case in enumerate(cases):
        case_lookup[(case["video_id"], case["window_index"])].append(case_index)
    values: dict[str, list[np.ndarray | None]] = {
        f"{backbone_name}_layer{layer:02d}": [None] * len(cases)
        for backbone_name in backbones
        for layer in layers
    }
    image_size = train_mod.parse_image_size(cfg.video.image_size)
    num_frames = train_mod.effective_num_frames(cfg)
    for video_number, (video_id, windows) in enumerate(unique_windows(cases).items(), start=1):
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
                for backbone_name, backbone in backbones.items():
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=device.type == "cuda",
                    ):
                        layer_values = extract_layers(backbone, inputs, layers)
                    for batch_index, meta in enumerate(batch["meta"]):
                        key = (video_id, int(meta["index"]))
                        for case_index in case_lookup[key]:
                            for layer, features in layer_values.items():
                                values[f"{backbone_name}_layer{layer:02d}"][case_index] = (
                                    features[batch_index].cpu().numpy().astype(np.float32)
                                )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            f"extract video={video_id} windows={len(windows)} "
            f"progress={video_number}/{len(unique_windows(cases))}",
            flush=True,
        )
    if any(item is None for items in values.values() for item in items):
        raise RuntimeError("Some cases did not receive backbone features")
    return {
        name: np.stack([item for item in items if item is not None])
        for name, items in values.items()
    }


def evaluate(
    cases: Sequence[dict[str, Any]],
    features: dict[str, np.ndarray],
    max_components: int,
    ridge: float,
) -> list[dict[str, Any]]:
    labels = np.asarray([case["label"] for case in cases])
    groups = np.asarray([case["group"] for case in cases])
    videos = np.asarray([case["video_id"] for case in cases])
    rows: list[dict[str, Any]] = []
    for task_name, (positive_groups, negative_groups) in TASK_GROUPS.items():
        allowed_groups = positive_groups | negative_groups
        for label in sorted(set(labels.tolist())):
            mask = (labels == label) & np.isin(groups, list(allowed_groups))
            targets = np.isin(groups[mask], list(positive_groups)).astype(np.int64)
            if len(np.unique(targets)) < 2:
                continue
            for representation, all_values in features.items():
                metrics = grouped_probe(
                    all_values[mask],
                    targets,
                    videos[mask],
                    max_components,
                    ridge,
                )
                backbone_name, raw_layer = representation.rsplit("_layer", 1)
                rows.append(
                    {
                        "task": task_name,
                        "label": label,
                        "backbone": backbone_name,
                        "layer": int(raw_layer),
                        "num_positive": int(targets.sum()),
                        "num_negative": int((targets == 0).sum()),
                        **metrics,
                    }
                )
    return rows


def summarize(
    rows: Sequence[dict[str, Any]],
    *,
    base_name: str = "base_dino",
    adapted_name: str = "football_lora",
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for task in TASK_GROUPS:
        task_rows = [row for row in rows if row["task"] == task]
        task_result: dict[str, Any] = {}
        for label in sorted({row["label"] for row in task_rows}):
            label_rows = [row for row in task_rows if row["label"] == label]
            best_by_backbone = {}
            for backbone in (base_name, adapted_name):
                matches = [row for row in label_rows if row["backbone"] == backbone]
                if not matches:
                    continue
                best = max(matches, key=lambda row: float(row["probe_auc"]))
                last = max(matches, key=lambda row: int(row["layer"]))
                best_by_backbone[backbone] = {
                    "best_layer": int(best["layer"]),
                    "best_auc": float(best["probe_auc"]),
                    "last_layer_auc": float(last["probe_auc"]),
                }
            base = best_by_backbone.get(base_name, {})
            adapted = best_by_backbone.get(adapted_name, {})
            if base and adapted:
                adapted["best_auc_gain_over_base"] = adapted["best_auc"] - base["best_auc"]
                adapted["last_auc_gain_over_base"] = adapted["last_layer_auc"] - base["last_layer_auc"]
                adapted["intermediate_gain_over_last"] = adapted["best_auc"] - adapted["last_layer_auc"]
            task_result[label] = best_by_backbone
        result[task] = task_result
    return result


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    layers = tuple(int(item.strip()) for item in args.layers.split(",") if item.strip())
    cases = read_cases(Path(args.cases).expanduser().resolve())
    adapted_name = str(args.adapted_name).strip() or "football_ssl"
    if args.adapted_backbone:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("config"), dict):
            raise ValueError("--checkpoint must contain a football task config")
        cfg = train_mod.to_config(checkpoint["config"])
        del checkpoint
        adapted_cfg = copy.deepcopy(cfg)
        adapted_cfg.model["weights"] = str(Path(args.adapted_backbone).expanduser().resolve())
        adapted_cfg.model["pretrained"] = True
        current_backbone = train_mod.build_backbone(adapted_cfg).to(device).eval()
    else:
        current_model, cfg, _, _ = load_checkpoint_model(args.checkpoint, device, [])
        current_backbone = current_model.backbone
        adapted_name = "football_lora"
        if current_backbone is None:
            raise RuntimeError("Football checkpoint has no backbone")
    base_cfg = copy.deepcopy(cfg)
    if args.base_backbone:
        base_cfg.model["weights"] = str(Path(args.base_backbone).expanduser().resolve())
        base_cfg.model["pretrained"] = True
    base_backbone = train_mod.build_backbone(base_cfg).to(device).eval()
    for backbone in (current_backbone, base_backbone):
        backbone.eval()
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
    total_layers = len(current_backbone.blocks)
    if not layers or min(layers) < 0 or max(layers) >= total_layers:
        raise ValueError(f"layers={layers} invalid for backbone blocks={total_layers}")
    metadata = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "base_weights": str(base_cfg.model.weights),
        "adapted_weights": (
            str(Path(args.adapted_backbone).expanduser().resolve())
            if args.adapted_backbone
            else str(Path(args.checkpoint).resolve())
        ),
        "adapted_name": adapted_name,
        "cases": str(Path(args.cases).resolve()),
        "num_cases": len(cases),
        "num_unique_windows": sum(len(items) for items in unique_windows(cases).values()),
        "videos": sorted({case["video_id"] for case in cases}),
        "layers": list(layers),
        "image_size": list(train_mod.parse_image_size(cfg.video.image_size)),
        "num_frames": train_mod.effective_num_frames(cfg),
    }
    (output_dir / "audit_config.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(json.dumps(metadata, ensure_ascii=False), flush=True)
    features = extract_all(
        cases,
        {"base_dino": base_backbone, adapted_name: current_backbone},
        cfg,
        layers,
        Path(args.video_root),
        device,
        args.batch_size,
        args.num_workers,
    )
    if args.save_features:
        np.savez_compressed(
            output_dir / "features.npz",
            **{name: values.astype(np.float16) for name, values in features.items()},
        )
    rows = evaluate(cases, features, args.probe_components, args.probe_ridge)
    write_csv(output_dir / "layer_probe_report.csv", rows)
    summary = summarize(rows, base_name="base_dino", adapted_name=adapted_name)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
