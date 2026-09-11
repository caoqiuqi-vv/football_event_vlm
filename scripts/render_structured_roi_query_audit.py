#!/usr/bin/env python
"""Render dual-query structured ROI attention on fixed positive events."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train_football_events as football  # noqa: E402
from train_football_events_featuremap_structured_roi import (  # noqa: E402
    load_structured_roi_config,
    load_structured_roi_model,
)
from scripts.render_gt_roi_featuremap_samples import (  # noqa: E402
    SafeRobustClipCropper,
    box_iou,
    crop_panel,
    decode_event_clip,
    heatmap_crop_box,
    labeled_panel,
    select_samples_from_manifest,
    write_gallery,
)


QUERY_COLORS = [
    (255, 210, 30),
    (220, 70, 235),
    (60, 200, 255),
    (255, 120, 40),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-videos", type=int, default=20)
    parser.add_argument("--topk-frames", type=int, default=4)
    parser.add_argument("--random-frames", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clip-duration", type=float, default=10.0)
    parser.add_argument("--heatmap-mass", type=float, default=0.65)
    parser.add_argument("--padding", type=float, default=0.18)
    parser.add_argument("--min-area-ratio", type=float, default=0.08)
    parser.add_argument("--max-area-ratio", type=float, default=0.35)
    return parser.parse_args()


def heatmap_overlay(
    frame_bgr: np.ndarray,
    heatmap: np.ndarray,
    box: Sequence[int],
    color: tuple[int, int, int],
) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    resized = cv2.resize(
        heatmap.astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    )
    lo, hi = float(resized.min()), float(resized.max())
    normalized = ((resized - lo) / max(hi - lo, 1e-8) * 255.0).astype(
        np.uint8
    )
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    overlay = cv2.addWeighted(frame_bgr, 0.52, colored, 0.48, 0.0)
    cv2.rectangle(
        overlay, (box[0], box[1]), (box[2], box[3]), color, 3
    )
    return overlay


def make_composite(
    frame_bgr: np.ndarray,
    query_heatmaps: Sequence[np.ndarray],
    query_boxes: Sequence[Sequence[int]],
    detector_box: Sequence[int] | None,
    *,
    title: str,
) -> np.ndarray:
    marked = frame_bgr.copy()
    for query_index, box in enumerate(query_boxes):
        color = QUERY_COLORS[query_index % len(QUERY_COLORS)]
        cv2.rectangle(
            marked, (box[0], box[1]), (box[2], box[3]), color, 4
        )
        cv2.putText(
            marked,
            f"Q{query_index + 1}",
            (box[0] + 6, max(box[1] + 26, 28)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA,
        )
    if detector_box is not None:
        cv2.rectangle(
            marked,
            (detector_box[0], detector_box[1]),
            (detector_box[2], detector_box[3]),
            (60, 220, 80),
            4,
        )

    panels = [
        labeled_panel(
            marked, "Full image: Q1/Q2 attention boxes; detector=green"
        )
    ]
    for query_index, (heatmap, box) in enumerate(
        zip(query_heatmaps, query_boxes)
    ):
        color = QUERY_COLORS[query_index % len(QUERY_COLORS)]
        panels.extend(
            [
                labeled_panel(
                    heatmap_overlay(frame_bgr, heatmap, box, color),
                    f"Q{query_index + 1} class heatmap + 65% mass box",
                ),
                labeled_panel(
                    crop_panel(frame_bgr, box),
                    f"Q{query_index + 1} attention-derived crop",
                ),
            ]
        )
    panels.append(
        labeled_panel(
            crop_panel(frame_bgr, detector_box), "Existing detector crop"
        )
    )
    row = np.concatenate(panels, axis=1)
    header = np.full((48, row.shape[1], 3), 18, dtype=np.uint8)
    cv2.putText(
        header,
        title,
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    return np.concatenate([header, row], axis=0)


def make_detector_cropper(config_path: Path, max_area_ratio: float):
    detector_cfg = yaml.safe_load(config_path.read_text())
    detector_cfg["spatial_crop"] = {
        "mode": "robust_detector_aware",
        "index_root": "outputs/football_roi_indices/robust_v2",
        "padding": 0.15,
        "min_crop_area_ratio": 0.0,
        "max_crop_area_ratio": max_area_ratio,
        "min_roi_confidence": 0.40,
        "goal_conf": 0.40,
        "person_conf": 0.35,
        "center_circle_conf": 0.35,
        "raw_ball_conf": 0.10,
        "min_goal_frames": 3,
        "min_ball_points": 5,
        "max_people": 10,
        "max_cached_videos": 2,
    }
    return SafeRobustClipCropper(detector_cfg["spatial_crop"])


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    source_manifest = Path(args.source_manifest)
    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "samples"
    thumbs_dir = output_dir / "thumbs"
    samples_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    cfg = load_structured_roi_config(str(config_path), [])
    football.configure_label_schema(cfg)
    saved_init_checkpoint = str(
        cfg.model.featuremap_structured.init_checkpoint
    )
    cfg.model.featuremap_structured.init_checkpoint = str(checkpoint_path)
    model = load_structured_roi_model(cfg, device)
    model.eval()

    lv_cfg = cfg.data.long_video
    root_cfg = lv_cfg.roots[0]
    annotations_dir = Path(root_cfg.annotations_dir)
    videos_dir = Path(root_cfg.videos_dir)
    selected, source_records = select_samples_from_manifest(
        source_manifest,
        annotations_dir,
        videos_dir,
        count=args.num_videos,
    )
    cropper = make_detector_cropper(config_path, args.max_area_ratio)
    input_size = football.parse_image_size(cfg.video.image_size)
    patch_h = input_size[0] // 16
    patch_w = input_size[1] // 16

    records: list[dict[str, Any]] = []
    sample_paths: list[Path] = []
    thumbnail_paths: list[Path] = []
    for sample_index, ((video_id, event, video_path), _) in enumerate(
        zip(selected, source_records), start=1
    ):
        (
            inputs,
            raw_frames,
            frame_indices,
            frame_times,
            clip_start,
            clip_end,
        ) = decode_event_clip(
            video_path,
            event.anchor_time,
            args.clip_duration,
            int(cfg.video.num_frames),
            input_size,
        )
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            outputs = model(inputs.to(device), return_aux=True)

        maps = outputs["spatial_attention_maps"].float().cpu()[0]
        spatial_scores = torch.sigmoid(
            outputs["spatial_frame_event_logits"].float().cpu()[0]
        )
        fused_scores = torch.sigmoid(
            outputs["frame_event_logits"].float().cpu()[0]
        )
        label_index = int(np.argmax(np.asarray(event.labels)))
        event_label = football.LABELS[label_index]
        topk = min(max(args.topk_frames, 0), len(frame_times))
        ranking = torch.argsort(
            spatial_scores[:, label_index], descending=True
        ).tolist()
        selected_slots = [int(slot) for slot in ranking[:topk]]
        remaining = [
            slot for slot in range(len(frame_times))
            if slot not in selected_slots
        ]
        sample_rng = random.Random(args.seed + sample_index * 1009)
        random_count = min(max(args.random_frames, 0), len(remaining))
        selected_slots.extend(sample_rng.sample(remaining, random_count))

        detector_proposal = cropper.get_window_roi(
            video_id,
            max(event.anchor_time - 1.5, 0.0),
            event.anchor_time + 1.5,
            raw_frames[0].shape[1],
            raw_frames[0].shape[0],
            (384, 640),
        )
        detector_box = (
            tuple(detector_proposal.bbox)
            if detector_proposal.valid and detector_proposal.bbox
            else None
        )

        event_records = []
        composites = []
        for rank, slot in enumerate(selected_slots):
            frame = raw_frames[slot]
            height, width = frame.shape[:2]
            query_heatmaps = [
                query_map.reshape(patch_h, patch_w).numpy()
                for query_map in maps[slot, label_index]
            ]
            query_boxes = [
                heatmap_crop_box(
                    heatmap,
                    width,
                    height,
                    mass=args.heatmap_mass,
                    padding=args.padding,
                    min_area_ratio=args.min_area_ratio,
                    max_area_ratio=args.max_area_ratio,
                    target_aspect=640.0 / 384.0,
                )
                for heatmap in query_heatmaps
            ]
            selection_type = "topk" if rank < topk else "random"
            spatial_score = float(spatial_scores[slot, label_index])
            fused_score = float(fused_scores[slot, label_index])
            title = (
                f"{sample_index:02d} | {video_id} | {event_label} | "
                f"GT {event.anchor_time:.3f}s | {selection_type} "
                f"frame {frame_times[slot]:.3f}s | "
                f"ROI score {spatial_score:.3f} | fused {fused_score:.3f}"
            )
            composites.append(
                make_composite(
                    frame,
                    query_heatmaps,
                    query_boxes,
                    detector_box,
                    title=title,
                )
            )
            queries = []
            for query_index, query_box in enumerate(query_boxes):
                area_ratio = (
                    (query_box[2] - query_box[0])
                    * (query_box[3] - query_box[1])
                    / float(width * height)
                )
                queries.append(
                    {
                        "query_index": query_index,
                        "heatmap_box": list(query_box),
                        "heatmap_area_ratio": area_ratio,
                        "heatmap_detector_iou": box_iou(
                            query_box, detector_box
                        ),
                    }
                )
            event_records.append(
                {
                    "selection_type": selection_type,
                    "selection_rank": rank,
                    "frame_slot": slot,
                    "frame_index": frame_indices[slot],
                    "frame_time": frame_times[slot],
                    "frame_offset_sec": (
                        frame_times[slot] - event.anchor_time
                    ),
                    "spatial_frame_score": spatial_score,
                    "fused_frame_score": fused_score,
                    "queries": queries,
                }
            )

        composite = np.concatenate(composites, axis=0)
        sample_path = (
            samples_dir
            / f"{sample_index:02d}_{video_id}_{event_label}.jpg"
        )
        cv2.imwrite(
            str(sample_path), composite, [cv2.IMWRITE_JPEG_QUALITY, 92]
        )
        thumbnail = cv2.resize(
            composite,
            (1440, max(159, int(159 * len(composites)))),
            interpolation=cv2.INTER_AREA,
        )
        thumb_path = thumbs_dir / sample_path.name
        cv2.imwrite(
            str(thumb_path), thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 55]
        )
        sample_paths.append(sample_path)
        thumbnail_paths.append(thumb_path)
        records.append(
            {
                "sample_index": sample_index,
                "video_id": video_id,
                "event_id": event.event_id,
                "event_label": event_label,
                "raw_label": event.raw_label,
                "anchor_time": event.anchor_time,
                "clip_start": clip_start,
                "clip_end": clip_end,
                "detector_box": (
                    list(detector_box) if detector_box else None
                ),
                "detector_valid": bool(detector_proposal.valid),
                "detector_mode": detector_proposal.mode,
                "selected_frames": event_records,
                "image": str(sample_path.resolve()),
            }
        )
        print(
            f"[{sample_index:02d}/{len(selected)}] {video_id} "
            f"{event_label} frames={len(event_records)} -> {sample_path}",
            flush=True,
        )
        del outputs, maps, inputs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    manifest = {
        "seed": args.seed,
        "num_videos": len(records),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "saved_init_checkpoint": saved_init_checkpoint,
        "source_manifest": str(source_manifest.resolve()),
        "frame_policy": {
            "topk_frames": args.topk_frames,
            "random_frames": args.random_frames,
            "ranking_branch": "spatial_frame_event_logits",
        },
        "crop_policy": {
            "heatmap_mass": args.heatmap_mass,
            "padding": args.padding,
            "min_area_ratio": args.min_area_ratio,
            "max_area_ratio": args.max_area_ratio,
            "target_aspect": 640.0 / 384.0,
        },
        "attention_note": (
            "Boxes contain the configured attention mass for visual audit. "
            "The structured model consumes soft attention-weighted spatial "
            "tokens and does not hard-crop these boxes during inference."
        ),
        "samples": records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    gallery_path = output_dir / "gallery-fragment.html"
    write_gallery(gallery_path, records, thumbnail_paths)
    print(f"manifest={manifest_path}", flush=True)
    print(f"gallery={gallery_path}", flush=True)


if __name__ == "__main__":
    main()
