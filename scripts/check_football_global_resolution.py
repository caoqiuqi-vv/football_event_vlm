#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Subset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_football_events as train_mod


def parse_size(value: str) -> tuple[int, int]:
    return train_mod.parse_image_size(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether a frozen football checkpoint preserves sampled validation quality at low global resolution."
    )
    parser.add_argument(
        "--config",
        default="configs/football/dinov3_vitl16_robust_dual_exp2.yaml",
        help="Uses this config's fixed detector-complete validation split and sampling protocol.",
    )
    parser.add_argument("--checkpoint", default="/mnt/data_16t/football/qiuqi/checkpoints/lora_r2_best.pt")
    parser.add_argument("--reference-size", default="384,640")
    parser.add_argument("--candidate-size", default="192,320")
    parser.add_argument("--max-map-drop", type=float, default=0.01)
    parser.add_argument("--max-recall-drop", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1024,
        help="Balanced paired validation subset; 0 evaluates the full validation set.",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        default="outputs/football_resolution_check/lora_r2_detector58.json",
    )
    return parser.parse_args()


def balanced_subset_indices(records: list[Any], max_samples: int, seed: int) -> list[int]:
    if max_samples <= 0 or len(records) <= max_samples:
        return list(range(len(records)))
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[(record.video_id, bool(record.is_negative), tuple(record.labels))].append(index)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    selected: list[int] = []
    ordered_keys = sorted(groups, key=str)
    while len(selected) < max_samples:
        wrote = False
        for key in ordered_keys:
            values = groups[key]
            if values:
                selected.append(values.pop())
                wrote = True
                if len(selected) >= max_samples:
                    break
        if not wrote:
            break
    return sorted(selected)


def validation_metrics(
    base_cfg: Any,
    model: torch.nn.Module,
    device: torch.device,
    image_size: tuple[int, int],
    batch_size: int,
    max_samples: int,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    # ConfigDict supports attribute reads but not attribute writes. Use mapping
    # assignment so downstream `.get(...)` calls see the requested override.
    cfg.video["image_size"] = list(image_size)
    _, val_dataset, _, val_records = train_mod.prepare_datasets(cfg, use_cache=False)
    selected = balanced_subset_indices(val_records, max_samples, int(cfg.seed))
    if len(selected) < len(val_records):
        val_dataset = Subset(val_dataset, selected)
        val_records = [val_records[index] for index in selected]
    loader = train_mod.make_loader(val_dataset, cfg, is_train=False, batch_size=batch_size)
    metrics = train_mod.evaluate(model, loader, cfg, device)
    return {
        "image_size": list(image_size),
        "num_samples": len(val_records),
        "num_videos": len({record.video_id for record in val_records}),
        "default": metrics["default"],
        "per_video_default": metrics["per_video_default"],
    }


def main() -> None:
    args = parse_args()
    cfg = train_mod.load_config(args.config, [])
    cfg["device"] = args.device
    cfg["gpu_ids"] = [0] if str(args.device).startswith("cuda") else []
    cfg.data["num_workers"] = args.num_workers
    cfg.data["persistent_workers"] = False
    cfg.eval["batch_size"] = args.batch_size
    cfg.model["init_checkpoint"] = args.checkpoint
    cfg.model["view_fusion"] = "single"
    cfg.model["freeze_global_branch"] = False
    cfg.model["freeze_loaded_backbone"] = True
    cfg.spatial_crop["mode"] = "none"
    cfg.spatial_crop["output_view"] = "roi_only"
    train_mod.configure_runtime_threads(cfg)
    train_mod.configure_label_schema(cfg)
    train_mod.seed_everything(int(cfg.seed), bool(cfg.deterministic))

    device = torch.device(args.device)
    model = train_mod.make_model(cfg, use_cached_features=False, device=device)
    if model.dual_view:
        raise RuntimeError("Resolution check must use the checkpoint's single global branch")
    if str(cfg.spatial_crop.get("mode")) != "none":
        raise RuntimeError("Resolution check must disable detector-aware spatial cropping")
    model.eval()
    reference = validation_metrics(
        cfg, model, device, parse_size(args.reference_size), args.batch_size, args.max_samples
    )
    candidate = validation_metrics(
        cfg, model, device, parse_size(args.candidate_size), args.batch_size, args.max_samples
    )
    map_drop = float(reference["default"]["mAP"]) - float(candidate["default"]["mAP"])
    recall_drop = float(reference["default"]["micro_recall"]) - float(candidate["default"]["micro_recall"])
    passed = map_drop <= args.max_map_drop and recall_drop <= args.max_recall_drop
    result = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "threshold": float(cfg.eval.threshold),
        "protocol": {
            "view_fusion": str(cfg.model.get("view_fusion")),
            "spatial_crop_mode": str(cfg.spatial_crop.get("mode")),
            "video_roots": [str(root.get("videos_dir")) for root in cfg.data.long_video.roots],
        },
        "reference": reference,
        "candidate": candidate,
        "map_drop": map_drop,
        "recall_drop": recall_drop,
        "max_map_drop": args.max_map_drop,
        "max_recall_drop": args.max_recall_drop,
        "use_low_resolution": passed,
        "selected_global_image_size": candidate["image_size"] if passed else reference["image_size"],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(
        f"reference mAP={reference['default']['mAP']:.4f} recall={reference['default']['micro_recall']:.4f}",
        flush=True,
    )
    print(
        f"candidate mAP={candidate['default']['mAP']:.4f} recall={candidate['default']['micro_recall']:.4f} "
        f"map_drop={map_drop:.4f} recall_drop={recall_drop:.4f} use_low_resolution={passed}",
        flush=True,
    )
    print(f"wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
