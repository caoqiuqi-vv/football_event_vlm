#!/usr/bin/env python
"""Render the exact dual-query crops consumed by the clip-only ROI experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train_football_events as football  # noqa: E402
from football_structured_roi_crop_clip import (  # noqa: E402
    load_structured_roi_crop_clip_model,
)
from scripts.render_gt_roi_featuremap_samples import (  # noqa: E402
    crop_panel,
    decode_event_clip,
    labeled_panel,
    select_samples_from_manifest,
    write_gallery,
)
from train_football_events_featuremap_structured_roi import (  # noqa: E402
    load_structured_roi_config,
)


QUERY_COLORS = [(255, 210, 30), (220, 70, 235)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-videos", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clip-duration", type=float, default=10.0)
    return parser.parse_args()


def crop_box_from_params(
    params: Sequence[float], width: int, height: int
) -> tuple[int, int, int, int]:
    center_x, center_y, scale_x, scale_y = map(float, params)
    translate_x = (1.0 - scale_x) * max(min(center_x, 1.0), -1.0)
    translate_y = (1.0 - scale_y) * max(min(center_y, 1.0), -1.0)
    x1 = (translate_x - scale_x + 1.0) * 0.5 * width
    x2 = (translate_x + scale_x + 1.0) * 0.5 * width
    y1 = (translate_y - scale_y + 1.0) * 0.5 * height
    y2 = (translate_y + scale_y + 1.0) * 0.5 * height
    return (
        int(round(max(min(x1, width - 1), 0))),
        int(round(max(min(y1, height - 1), 0))),
        int(round(max(min(x2, width), 1))),
        int(round(max(min(y2, height), 1))),
    )


def heatmap_overlay(
    frame: np.ndarray,
    heatmap: np.ndarray,
    box: Sequence[int],
    color: tuple[int, int, int],
) -> np.ndarray:
    height, width = frame.shape[:2]
    resized = cv2.resize(
        heatmap.astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    )
    lo, hi = float(resized.min()), float(resized.max())
    normalized = ((resized - lo) / max(hi - lo, 1e-8) * 255).astype(
        np.uint8
    )
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    overlay = cv2.addWeighted(frame, 0.52, colored, 0.48, 0.0)
    cv2.rectangle(
        overlay, (box[0], box[1]), (box[2], box[3]), color, 3
    )
    return overlay


def make_composite(
    frame: np.ndarray,
    heatmaps: Sequence[np.ndarray],
    boxes: Sequence[Sequence[int]],
    *,
    title: str,
) -> np.ndarray:
    marked = frame.copy()
    for query_index, box in enumerate(boxes):
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
    panels = [
        labeled_panel(marked, "Exact fixed-scale crops used by the model")
    ]
    for query_index, (heatmap, box) in enumerate(zip(heatmaps, boxes)):
        color = QUERY_COLORS[query_index % len(QUERY_COLORS)]
        panels.extend(
            [
                labeled_panel(
                    heatmap_overlay(frame, heatmap, box, color),
                    f"Q{query_index + 1} heatmap + actual 35% crop",
                ),
                labeled_panel(
                    crop_panel(frame, box),
                    f"Q{query_index + 1} pixels re-encoded at 256x448",
                ),
            ]
        )
    row = np.concatenate(panels, axis=1)
    header = np.full((48, row.shape[1], 3), 18, dtype=np.uint8)
    cv2.putText(
        header,
        title,
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    return np.concatenate([header, row], axis=0)


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
    model = load_structured_roi_crop_clip_model(cfg, device)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    state = football.strip_module_prefix(checkpoint["model"])
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch missing={missing} unexpected={unexpected}"
        )
    model.to(device).eval()
    print(
        f"Loaded crop-clip checkpoint={checkpoint_path} "
        f"epoch={checkpoint.get('epoch')} strict=true",
        flush=True,
    )

    root_cfg = cfg.data.long_video.roots[0]
    selected, source_records = select_samples_from_manifest(
        source_manifest,
        Path(root_cfg.annotations_dir),
        Path(root_cfg.videos_dir),
        count=args.num_videos,
    )
    input_size = football.parse_image_size(cfg.video.image_size)
    patch_h, patch_w = input_size[0] // 16, input_size[1] // 16
    topk_frames = int(cfg.model.roi_crop_clip.topk_frames)

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

        label_index = int(np.argmax(np.asarray(event.labels)))
        event_label = football.LABELS[label_index]
        indices = outputs["roi_indices"].cpu()[0, label_index]
        params = outputs["roi_crop_params"].float().cpu()[
            0, label_index
        ]
        attention = outputs["roi_selected_attention"].float().cpu()[
            0, label_index
        ]
        global_frame_scores = torch.sigmoid(
            outputs["global_frame_event_logits"].float().cpu()[
                0, :, label_index
            ]
        )
        global_clip_score = float(
            torch.sigmoid(
                outputs["global_logits"].float().cpu()[0, label_index]
            )
        )
        roi_clip_score = float(
            torch.sigmoid(
                outputs["spatial_clip_logits"].float().cpu()[
                    0, label_index
                ]
            )
        )
        fused_clip_score = float(
            torch.sigmoid(
                outputs["logits"].float().cpu()[0, label_index]
            )
        )

        composites = []
        frame_records = []
        for rank, slot_tensor in enumerate(indices):
            slot = int(slot_tensor)
            frame = raw_frames[slot]
            height, width = frame.shape[:2]
            heatmaps = [
                query.reshape(patch_h, patch_w).numpy()
                for query in attention[rank]
            ]
            boxes = [
                crop_box_from_params(query_params.tolist(), width, height)
                for query_params in params[rank]
            ]
            selection_type = (
                "global_topk" if rank < topk_frames else "exploration"
            )
            frame_score = float(global_frame_scores[slot])
            title = (
                f"{sample_index:02d} | {video_id} | {event_label} | "
                f"GT {event.anchor_time:.3f}s | {selection_type} "
                f"{frame_times[slot]:.3f}s | frame {frame_score:.3f} | "
                f"clip global/ROI/fused "
                f"{global_clip_score:.3f}/{roi_clip_score:.3f}/"
                f"{fused_clip_score:.3f}"
            )
            composites.append(
                make_composite(frame, heatmaps, boxes, title=title)
            )
            frame_records.append(
                {
                    "selection_type": selection_type,
                    "selection_rank": rank,
                    "frame_slot": slot,
                    "frame_index": frame_indices[slot],
                    "frame_time": frame_times[slot],
                    "frame_offset_sec": (
                        frame_times[slot] - event.anchor_time
                    ),
                    "global_frame_score": frame_score,
                    "query_crop_params": params[rank].tolist(),
                    "query_boxes": [list(box) for box in boxes],
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
                "global_clip_score": global_clip_score,
                "roi_clip_score": roi_clip_score,
                "fused_clip_score": fused_clip_score,
                "selected_frames": frame_records,
                "image": str(sample_path.resolve()),
            }
        )
        print(
            f"[{sample_index:02d}/{len(selected)}] {video_id} "
            f"{event_label} -> {sample_path}",
            flush=True,
        )
        del outputs, inputs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    manifest = {
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "source_manifest": str(source_manifest.resolve()),
        "num_videos": len(records),
        "crop_semantics": (
            "These are the exact fixed-scale grid_sample crops consumed by "
            "the crop-clip experiment, not attention-mass audit boxes."
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
