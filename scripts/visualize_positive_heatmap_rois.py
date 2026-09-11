#!/usr/bin/env python
"""Audit class-conditioned heatmap crops on positive football events.

The script intentionally performs no training.  It samples one confirmed
positive event from each of N training videos, runs a trained full-image
spatial-attention model, derives a bounded crop from the GT-class attention
map, and compares it with the existing detector-aware crop.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import random
import sys
from collections import OrderedDict
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
from football_detection_aware import RobustClipCropper  # noqa: E402


LABEL_COLORS = {
    "shot": (42, 92, 240),
    "save": (235, 137, 52),
    "set_piece": (160, 82, 210),
}


class SafeRobustClipCropper(RobustClipCropper):
    """Use tensor-only deserialization for compact ROI indices."""

    def _load(self, video_id: str) -> dict[str, Any]:
        cached = self._cache.pop(video_id, None)
        if cached is not None:
            self._cache[video_id] = cached
            return cached
        path = self.index_root / f"{video_id}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing robust ROI index: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or int(payload.get("version", 0)) != 2:
            raise ValueError(f"Unsupported robust ROI index: {path}")
        self._cache[video_id] = payload
        while len(self._cache) > self.max_cached_videos:
            self._cache.popitem(last=False)
        return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="outputs/football_events/vitl16_spatial_attn_multilayer_readout_16f_hr/config.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/football_events/vitl16_spatial_attn_multilayer_readout_16f_hr/best.pt",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/football_heatmap_roi_audit/train_positive_20",
    )
    parser.add_argument("--num-videos", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clip-duration", type=float, default=10.0)
    parser.add_argument("--heatmap-mass", type=float, default=0.65)
    parser.add_argument("--padding", type=float, default=0.18)
    parser.add_argument("--min-area-ratio", type=float, default=0.08)
    parser.add_argument("--max-area-ratio", type=float, default=0.35)
    return parser.parse_args()


def minimum_mass_interval(values: np.ndarray, mass: float) -> tuple[int, int]:
    values = np.asarray(values, dtype=np.float64)
    values = np.maximum(values, 0.0)
    total = float(values.sum())
    if total <= 0:
        return 0, max(len(values) - 1, 0)
    values /= total
    prefix = np.concatenate([[0.0], np.cumsum(values)])
    best = (0, len(values) - 1)
    for left in range(len(values)):
        target = prefix[left] + mass
        right_exclusive = int(np.searchsorted(prefix, target, side="left"))
        if right_exclusive > len(values):
            break
        right = max(left, right_exclusive - 1)
        if right - left < best[1] - best[0]:
            best = (left, right)
    return best


def fit_box(
    box: Sequence[float],
    width: int,
    height: int,
    *,
    target_aspect: float,
    min_area_ratio: float,
    max_area_ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = map(float, box)
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    box_w, box_h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    if box_w / box_h < target_aspect:
        box_w = box_h * target_aspect
    else:
        box_h = box_w / target_aspect

    frame_area = float(width * height)
    area = box_w * box_h
    min_area = max(min_area_ratio, 0.0) * frame_area
    max_area = max(max_area_ratio, min_area_ratio) * frame_area
    if area < min_area:
        scale = math.sqrt(min_area / max(area, 1.0))
        box_w *= scale
        box_h *= scale
    elif area > max_area:
        scale = math.sqrt(max_area / area)
        box_w *= scale
        box_h *= scale

    box_w = min(box_w, float(width))
    box_h = min(box_h, float(height))
    x1 = min(max(cx - box_w * 0.5, 0.0), width - box_w)
    y1 = min(max(cy - box_h * 0.5, 0.0), height - box_h)
    return (
        int(round(x1)),
        int(round(y1)),
        int(round(x1 + box_w)),
        int(round(y1 + box_h)),
    )


def heatmap_crop_box(
    heatmap: np.ndarray,
    width: int,
    height: int,
    *,
    mass: float,
    padding: float,
    min_area_ratio: float,
    max_area_ratio: float,
    target_aspect: float,
) -> tuple[int, int, int, int]:
    smoothed = cv2.GaussianBlur(heatmap.astype(np.float32), (5, 5), 0)
    x_left, x_right = minimum_mass_interval(smoothed.sum(axis=0), mass)
    y_top, y_bottom = minimum_mass_interval(smoothed.sum(axis=1), mass)
    patch_h, patch_w = smoothed.shape
    x1 = x_left / patch_w * width
    x2 = (x_right + 1) / patch_w * width
    y1 = y_top / patch_h * height
    y2 = (y_bottom + 1) / patch_h * height
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    return fit_box(
        (x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y),
        width,
        height,
        target_aspect=target_aspect,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
    )


def box_iou(a: Sequence[int], b: Sequence[int] | None) -> float | None:
    if b is None:
        return None
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
    area_a = max(a[2] - a[0], 0) * max(a[3] - a[1], 0)
    area_b = max(b[2] - b[0], 0) * max(b[3] - b[1], 0)
    return float(intersection / max(area_a + area_b - intersection, 1))


def resize_panel(image: np.ndarray, size: tuple[int, int] = (480, 270)) -> np.ndarray:
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def crop_panel(image: np.ndarray, box: Sequence[int] | None) -> np.ndarray:
    if box is None:
        panel = np.full_like(image, 36)
        cv2.putText(panel, "invalid detector ROI", (40, panel.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 220), 2)
        return panel
    x1, y1, x2, y2 = box
    return image[y1:y2, x1:x2]


def labeled_panel(image: np.ndarray, label: str) -> np.ndarray:
    panel = resize_panel(image)
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 34), (20, 20, 20), -1)
    cv2.putText(panel, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (245, 245, 245), 2, cv2.LINE_AA)
    return panel


def make_composite(
    frame_bgr: np.ndarray,
    heatmap: np.ndarray,
    heatmap_box: Sequence[int],
    detector_box: Sequence[int] | None,
    *,
    title: str,
) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    marked = frame_bgr.copy()
    cv2.rectangle(marked, (heatmap_box[0], heatmap_box[1]), (heatmap_box[2], heatmap_box[3]), (255, 210, 30), 4)
    if detector_box is not None:
        cv2.rectangle(marked, (detector_box[0], detector_box[1]), (detector_box[2], detector_box[3]), (60, 220, 80), 4)

    resized_heatmap = cv2.resize(heatmap.astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC)
    lo, hi = float(resized_heatmap.min()), float(resized_heatmap.max())
    normalized = ((resized_heatmap - lo) / max(hi - lo, 1e-8) * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    overlay = cv2.addWeighted(frame_bgr, 0.52, colored, 0.48, 0.0)
    cv2.rectangle(overlay, (heatmap_box[0], heatmap_box[1]), (heatmap_box[2], heatmap_box[3]), (255, 255, 255), 3)

    panels = [
        labeled_panel(marked, "Full image: heatmap=yellow, detector=green"),
        labeled_panel(overlay, "GT-class network heatmap + proposed crop"),
        labeled_panel(crop_panel(frame_bgr, heatmap_box), "Heatmap crop (resized only for display)"),
        labeled_panel(crop_panel(frame_bgr, detector_box), "Existing detector crop"),
    ]
    row = np.concatenate(panels, axis=1)
    header = np.full((48, row.shape[1], 3), 18, dtype=np.uint8)
    cv2.putText(header, title, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (245, 245, 245), 2, cv2.LINE_AA)
    return np.concatenate([header, row], axis=0)


def decode_event_clip(
    video_path: Path,
    anchor_time: float,
    clip_duration: float,
    num_frames: int,
    input_size: tuple[int, int],
) -> tuple[torch.Tensor, list[np.ndarray], list[int], list[float], float, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps
    start, end = football.centered_window(anchor_time, clip_duration, duration)
    indices = football.segment_frame_indices(frame_count, fps, num_frames, False, start_sec=start, end_sec=end)
    raw_frames = football._decode_video_frames(cap, indices, "single_seek")
    cap.release()
    if any(frame is None for frame in raw_frames):
        raise RuntimeError(f"Failed to decode all frames from {video_path}")
    frames_bgr = [frame for frame in raw_frames if frame is not None]
    in_h, in_w = input_size
    tensors = []
    for frame in frames_bgr:
        rgb = cv2.cvtColor(cv2.resize(frame, (in_w, in_h), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
        tensors.append(torch.from_numpy(rgb).permute(2, 0, 1))
    times = football.frame_times_from_indices(indices, fps)
    return torch.stack(tensors).unsqueeze(0), frames_bgr, indices, times, start, end


def select_samples(
    split_path: Path,
    annotations_dir: Path,
    videos_dir: Path,
    index_root: Path,
    *,
    count: int,
    seed: int,
) -> list[tuple[str, football.FootballEvent, Path]]:
    video_ids = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
    eligible: list[tuple[str, list[football.FootballEvent], Path]] = []
    for video_id in video_ids:
        annotation = annotations_dir / f"{video_id}.json"
        video_path = football.find_video_path(videos_dir, video_id)
        if not annotation.exists() or video_path is None or not (index_root / f"{video_id}.pt").exists():
            continue
        events = football.load_annotation_events(annotation, "xbotgo_0608", video_id)
        if events:
            eligible.append((video_id, events, video_path))
    rng = random.Random(seed)
    rng.shuffle(eligible)
    selected = []
    for video_id, events, video_path in eligible[:count]:
        selected.append((video_id, rng.choice(events), video_path))
    if len(selected) < count:
        raise RuntimeError(f"Only {len(selected)} eligible videos for requested {count}")
    return selected


def write_gallery(path: Path, records: list[dict[str, Any]], image_paths: list[Path]) -> None:
    cards = []
    for record, image_path in zip(records, image_paths):
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        subtitle = (
            f"{record['video_id']} · {record['event_label']} · "
            f"GT {record['anchor_time']:.3f}s · frame {record['selected_frame_time']:.3f}s"
        )
        cards.append(
            f'<article><div class="meta">{subtitle}</div><img src="data:image/jpeg;base64,{encoded}" alt="{subtitle}"></article>'
        )
    html = """<div id="heatmap-roi-audit">
<style>
#heatmap-roi-audit{font-family:ui-sans-serif,system-ui;color:var(--foreground);display:grid;gap:18px}
#heatmap-roi-audit article{border-bottom:1px solid var(--border);padding-bottom:18px}
#heatmap-roi-audit .meta{font-size:13px;font-weight:650;margin:0 0 8px}
#heatmap-roi-audit img{display:block;width:100%;height:auto;border-radius:6px}
</style>
""" + "\n".join(cards) + "\n</div>\n"
    path.write_text(html)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "samples"
    thumbs_dir = output_dir / "thumbs"
    samples_dir.mkdir(parents=True, exist_ok=True)
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    cfg = football.load_config(str(config_path), [])
    football.configure_label_schema(cfg)
    saved_init_checkpoint = str(cfg.model.get("init_checkpoint", ""))
    cfg.model.init_checkpoint = ""
    cfg.model.spatial_attention.return_attention_maps = True
    cfg.model.freeze_backbone = True
    cfg.model.freeze_global_branch = True

    device = torch.device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = football.make_model(cfg, use_cached_features=False, device=device)
    football.load_model_init_checkpoint(
        model,
        str(checkpoint_path),
        checkpoint=checkpoint,
        expected_backbone=str(cfg.model.backbone),
        strict=True,
    )
    model.to(device).eval()

    lv_cfg = cfg.data.long_video
    root_cfg = lv_cfg.roots[0]
    split_path = Path(lv_cfg.split_files.train[0])
    annotations_dir = Path(root_cfg.annotations_dir)
    videos_dir = Path(root_cfg.videos_dir)
    detector_cfg = yaml.safe_load(config_path.read_text())
    detector_cfg["spatial_crop"] = {
        "mode": "robust_detector_aware",
        "index_root": "outputs/football_roi_indices/robust_v2",
        "padding": 0.15,
        "min_crop_area_ratio": 0.0,
        "max_crop_area_ratio": args.max_area_ratio,
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
    index_root = Path(detector_cfg["spatial_crop"]["index_root"])
    cropper = SafeRobustClipCropper(detector_cfg["spatial_crop"])
    selected = select_samples(
        split_path,
        annotations_dir,
        videos_dir,
        index_root,
        count=args.num_videos,
        seed=args.seed,
    )

    input_size = football.parse_image_size(cfg.video.image_size)
    patch_h = input_size[0] // 16
    patch_w = input_size[1] // 16
    records: list[dict[str, Any]] = []
    sample_paths: list[Path] = []
    thumbnail_paths: list[Path] = []

    for sample_index, (video_id, event, video_path) in enumerate(selected, start=1):
        inputs, raw_frames, frame_indices, frame_times, start, end = decode_event_clip(
            video_path,
            event.anchor_time,
            args.clip_duration,
            int(cfg.video.num_frames),
            input_size,
        )
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outputs = model(inputs.to(device), return_aux=True)
        maps = outputs["spatial_attention_maps"].float().cpu()[0]
        label_index = int(np.argmax(np.asarray(event.labels)))
        nearest = min(range(len(frame_times)), key=lambda index: abs(frame_times[index] - event.anchor_time))
        heatmap = maps[nearest, label_index].mean(dim=0).reshape(patch_h, patch_w).numpy()
        frame = raw_frames[nearest]
        height, width = frame.shape[:2]
        heatmap_box = heatmap_crop_box(
            heatmap,
            width,
            height,
            mass=args.heatmap_mass,
            padding=args.padding,
            min_area_ratio=args.min_area_ratio,
            max_area_ratio=args.max_area_ratio,
            target_aspect=640.0 / 384.0,
        )
        detector_proposal = cropper.get_window_roi(
            video_id,
            max(event.anchor_time - 1.5, 0.0),
            event.anchor_time + 1.5,
            width,
            height,
            (384, 640),
        )
        detector_box = tuple(detector_proposal.bbox) if detector_proposal.valid and detector_proposal.bbox else None
        event_label = football.LABELS[label_index]
        title = (
            f"{sample_index:02d} | video {video_id} | {event_label} | "
            f"GT {event.anchor_time:.3f}s | selected {frame_times[nearest]:.3f}s"
        )
        composite = make_composite(frame, heatmap, heatmap_box, detector_box, title=title)
        sample_path = samples_dir / f"{sample_index:02d}_{video_id}_{event_label}.jpg"
        cv2.imwrite(str(sample_path), composite, [cv2.IMWRITE_JPEG_QUALITY, 92])
        thumbnail = cv2.resize(composite, (960, 159), interpolation=cv2.INTER_AREA)
        thumb_path = thumbs_dir / sample_path.name
        cv2.imwrite(str(thumb_path), thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 48])
        sample_paths.append(sample_path)
        thumbnail_paths.append(thumb_path)

        heatmap_area = (heatmap_box[2] - heatmap_box[0]) * (heatmap_box[3] - heatmap_box[1]) / float(width * height)
        detector_area = None
        if detector_box is not None:
            detector_area = (detector_box[2] - detector_box[0]) * (detector_box[3] - detector_box[1]) / float(width * height)
        record = {
            "sample_index": sample_index,
            "video_id": video_id,
            "event_id": event.event_id,
            "event_label": event_label,
            "raw_label": event.raw_label,
            "anchor_time": event.anchor_time,
            "clip_start": start,
            "clip_end": end,
            "selected_frame_index": frame_indices[nearest],
            "selected_frame_time": frame_times[nearest],
            "selected_frame_offset_sec": frame_times[nearest] - event.anchor_time,
            "heatmap_box": list(heatmap_box),
            "heatmap_area_ratio": heatmap_area,
            "detector_box": list(detector_box) if detector_box is not None else None,
            "detector_valid": bool(detector_proposal.valid),
            "detector_mode": detector_proposal.mode,
            "detector_area_ratio": detector_area,
            "heatmap_detector_iou": box_iou(heatmap_box, detector_box),
            "image": str(sample_path.resolve()),
        }
        records.append(record)
        print(f"[{sample_index:02d}/{len(selected)}] {video_id} {event_label} -> {sample_path}", flush=True)
        del outputs, maps, inputs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    manifest = {
        "seed": args.seed,
        "num_videos": len(records),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "saved_init_checkpoint": saved_init_checkpoint,
        "crop_policy": {
            "heatmap_mass": args.heatmap_mass,
            "padding": args.padding,
            "min_area_ratio": args.min_area_ratio,
            "max_area_ratio": args.max_area_ratio,
            "target_aspect": 640.0 / 384.0,
        },
        "samples": records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    gallery_path = output_dir / "gallery-fragment.html"
    write_gallery(gallery_path, records, thumbnail_paths)
    print(f"manifest={manifest_path}", flush=True)
    print(f"gallery={gallery_path}", flush=True)


if __name__ == "__main__":
    main()
