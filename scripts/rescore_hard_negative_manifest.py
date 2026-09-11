#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

import train_football_events as train_mod
from scripts import eval_long_video_checkpoint as ev

VIDEO_EXTENSIONS = (".mp4", ".MP4", ".mov", ".MOV", ".mkv", ".MKV", ".avi", ".AVI")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rescore hard-negative windows with the exact target checkpoint to avoid cross-model mining."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-root", action="append", default=["xbotgo_0608=/mnt/data_16t/football/raw_video_720P"])
    parser.add_argument("--labels", default="shot,save")
    parser.add_argument("--min-shot-prob", type=float, default=0.05)
    parser.add_argument("--min-save-prob", type=float, default=0.05)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--gpu-ids", default="1")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--decode-strategy", default="single_seek", choices=["multi_seek", "single_seek"])
    parser.add_argument("--video-reader-cache-size", type=int, default=2)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_video_roots(values: list[str]) -> list[tuple[str, Path]]:
    roots: list[tuple[str, Path]] = []
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid --video-root {item!r}; expected source=/path")
        source, path = item.split("=", 1)
        roots.append((source.strip(), Path(path).expanduser()))
    return roots


def find_video(video_id: str, roots: list[tuple[str, Path]]) -> tuple[str, Path]:
    for source, root in roots:
        for ext in VIDEO_EXTENSIONS:
            path = root / f"{video_id}{ext}"
            if path.exists():
                return source, path
    searched = ", ".join(str(root) for _, root in roots)
    raise FileNotFoundError(f"Could not find video {video_id} in {searched}")


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        items = payload.get("hard_negatives", payload.get("windows", []))
    else:
        items = payload
        payload = {"hard_negatives": items}
    if not isinstance(items, list):
        raise ValueError(f"Manifest {path} has no list hard_negatives/windows")
    payload["hard_negatives"] = [item for item in items if isinstance(item, dict)]
    return payload


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(f"Output exists: {args.output}; pass --force")
    labels_requested = [label.strip() for label in args.labels.split(",") if label.strip()]
    payload = load_manifest(args.manifest)
    items = payload["hard_negatives"]
    if args.max_items > 0:
        items = items[: args.max_items]
    video_roots = parse_video_roots(args.video_root)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()] if device.type == "cuda" else []

    model, cfg, ckpt_labels, _thresholds = ev.load_checkpoint_model(args.checkpoint, device, gpu_ids)
    eval_labels = [label for label in labels_requested if label in ckpt_labels]
    missing_labels = sorted(set(labels_requested) - set(eval_labels))
    if missing_labels:
        raise ValueError(f"Requested labels not present in checkpoint: {missing_labels}; checkpoint labels={ckpt_labels}")
    image_size = train_mod.parse_image_size(cfg.video.image_size)
    num_frames = train_mod.effective_num_frames(cfg)
    normalize_on_cpu = not train_mod.normalize_on_device_enabled(cfg)
    view_mode = str(cfg.model.get("view_fusion", "single"))
    global_image_size = train_mod.parse_image_size(
        cfg.get("spatial_crop", train_mod.ConfigDict()).get("global_image_size", image_size)
    )
    if view_mode != "single":
        raise ValueError("This lightweight rescoring script currently supports single-view checkpoints only")

    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, item in enumerate(items):
        video_id = str(item.get("video_id", "")).strip()
        if not video_id:
            continue
        grouped[video_id].append((index, item))

    rows: list[dict[str, Any]] = []
    rescored: list[dict[str, Any]] = [dict(item) for item in items]
    kept_indices: set[int] = set()
    stats = Counter()
    total_videos = len(grouped)
    with torch.no_grad():
        for video_counter, (video_id, pairs) in enumerate(sorted(grouped.items()), start=1):
            _source, video_path = find_video(video_id, video_roots)
            windows = []
            index_to_manifest: dict[int, tuple[int, dict[str, Any]]] = {}
            for local_idx, (manifest_idx, item) in enumerate(pairs):
                start = float(item.get("start_sec", 0.0) or 0.0)
                end = float(item.get("end_sec", start) or start)
                if end <= start:
                    center = float(item.get("center_sec", start) or start)
                    start = max(center - 5.0, 0.0)
                    end = start + 10.0
                windows.append(ev.WindowRecord(index=local_idx, start_sec=start, end_sec=end))
                index_to_manifest[local_idx] = (manifest_idx, item)
            dataset = ev.SlidingWindowVideoDataset(
                video_path=str(video_path),
                video_id=video_id,
                windows=windows,
                num_frames=num_frames,
                image_size=image_size,
                normalize_on_cpu=normalize_on_cpu,
                decode_strategy=args.decode_strategy,
                video_reader_cache_size=args.video_reader_cache_size,
                window_cropper=None,
                frame_crop_provider=None,
                view_mode=view_mode,
                global_image_size=global_image_size,
            )
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
                persistent_workers=args.num_workers > 0,
                prefetch_factor=2 if args.num_workers > 0 else None,
                collate_fn=ev.collate_windows,
            )
            for batch in loader:
                with train_mod.autocast_context(device, bool(cfg.train.amp), str(cfg.train.amp_dtype)):
                    outputs = train_mod.forward_model_batch(model, batch, device, return_aux=True)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                probs = torch.sigmoid(logits).float().cpu()
                for row_idx, meta in enumerate(batch["meta"]):
                    local_idx = int(meta["index"])
                    manifest_idx, item = index_to_manifest[local_idx]
                    out = dict(rescored[manifest_idx])
                    label_probs = {}
                    for label in eval_labels:
                        prob = float(probs[row_idx, ckpt_labels.index(label)])
                        out[f"target_prob_{label}"] = prob
                        label_probs[label] = prob
                    selected_labels = []
                    if label_probs.get("shot", 0.0) >= args.min_shot_prob:
                        selected_labels.append("shot")
                    if label_probs.get("save", 0.0) >= args.min_save_prob:
                        selected_labels.append("save")
                    out["source_labels"] = list(item.get("labels", [])) if isinstance(item.get("labels", []), list) else item.get("labels", [])
                    out["target_labels"] = selected_labels
                    out["labels"] = selected_labels
                    out["score"] = max((label_probs.get(label, 0.0) for label in selected_labels), default=0.0)
                    out["target_score"] = out["score"]
                    out["target_checkpoint"] = args.checkpoint
                    out["target_rescore_policy"] = {
                        "min_shot_prob": args.min_shot_prob,
                        "min_save_prob": args.min_save_prob,
                    }
                    rescored[manifest_idx] = out
                    rows.append({
                        "manifest_index": manifest_idx,
                        "video_id": video_id,
                        "start_sec": out.get("start_sec"),
                        "end_sec": out.get("end_sec"),
                        "center_sec": out.get("center_sec"),
                        "raw_label": out.get("raw_label"),
                        "source_prob_shot": out.get("prob_shot"),
                        "source_prob_save": out.get("prob_save"),
                        "target_prob_shot": out.get("target_prob_shot"),
                        "target_prob_save": out.get("target_prob_save"),
                        "source_labels": ",".join(str(x) for x in item.get("labels", [])),
                        "target_labels": ",".join(selected_labels),
                        "target_score": out["target_score"],
                    })
                    if selected_labels:
                        kept_indices.add(manifest_idx)
                        for label in selected_labels:
                            stats[f"target_label_{label}"] += 1
                    else:
                        stats["below_target_threshold"] += 1
            print(f"rescored video {video_counter}/{total_videos} {video_id} windows={len(windows)}", flush=True)

    kept = [rescored[index] for index in sorted(kept_indices)]
    manifest = dict(payload)
    manifest["hard_negatives"] = kept
    manifest["rescore"] = {
        "source_manifest": str(args.manifest),
        "target_checkpoint": args.checkpoint,
        "labels": eval_labels,
        "min_shot_prob": args.min_shot_prob,
        "min_save_prob": args.min_save_prob,
        "num_input": len(items),
        "num_kept": len(kept),
        "stats": dict(stats),
    }
    manifest["summary"] = dict(manifest.get("summary", {}))
    kept_label_slots = Counter(label for item in kept for label in item.get("labels", []))
    kept_action_counts = Counter(str(item.get("raw_label", item.get("action_label", "unknown"))) for item in kept)
    manifest["summary"].update({
        "num_hard_negatives_before_target_rescore": len(items),
        "num_hard_negatives": len(kept),
        "label_slots": dict(kept_label_slots),
        "raw_action_counts": dict(kept_action_counts),
        "target_label_slots": {k.removeprefix("target_label_"): v for k, v in stats.items() if k.startswith("target_label_")},
        "target_rescore_checkpoint": args.checkpoint,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    write_csv(args.output.with_suffix(".csv"), rows)
    print(json.dumps(manifest["rescore"], ensure_ascii=False, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
