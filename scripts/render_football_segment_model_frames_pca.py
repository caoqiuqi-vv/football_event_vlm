#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_long_video_checkpoint import load_checkpoint_model


LABELS = ("shot", "save", "set_piece")
SEGMENT_RE = re.compile(
    r"(?P<kind>fp|fn|tp)?_?(?P<label>shot|save|set_piece)_(?P<video_id>\d+)_w(?P<window_index>\d+)_t(?P<start_sec>\d+(?:\.\d+)?)"
)


@dataclass
class Segment:
    raw: str
    video_id: str
    label: str
    start_sec: float
    end_sec: float
    window_index: int | None = None
    kind: str = ""


def parse_size(raw: str) -> tuple[int, int]:
    raw = raw.strip().lower().replace("x", ",")
    h, w = raw.split(",", 1)
    return int(h), int(w)


def parse_segment(raw: str, *, clip_sec: float) -> Segment:
    raw = raw.strip()
    if not raw:
        raise ValueError("empty segment")
    if "," in raw and not SEGMENT_RE.search(raw):
        parts = [item.strip() for item in raw.split(",")]
        if len(parts) < 4:
            raise ValueError(f"CSV segment must be video_id,label,start_sec,end_sec: {raw}")
        return Segment(
            raw=raw,
            video_id=parts[0],
            label=parts[1],
            start_sec=float(parts[2]),
            end_sec=float(parts[3]),
            window_index=int(parts[4]) if len(parts) > 4 and parts[4] else None,
        )
    match = SEGMENT_RE.search(raw)
    if not match:
        raise ValueError(
            "Cannot parse segment. Expected e.g. "
            "fp_shot_2027572095412195330_w00028_t00140.000 "
            "or video_id,label,start_sec,end_sec"
        )
    start = float(match.group("start_sec"))
    return Segment(
        raw=raw,
        video_id=match.group("video_id"),
        label=match.group("label"),
        start_sec=start,
        end_sec=start + float(clip_sec),
        window_index=int(match.group("window_index")),
        kind=match.group("kind") or "",
    )


def load_segments(args: argparse.Namespace) -> list[Segment]:
    raw_items: list[str] = []
    if args.segments:
        for item in args.segments.split(","):
            item = item.strip()
            if item:
                raw_items.append(item)
    if args.segment_file:
        path = Path(args.segment_file)
        if path.suffix.lower() == ".csv":
            with path.open(newline="") as f:
                for row in csv.DictReader(f):
                    video_id = row.get("video_id", "").strip()
                    label = row.get("label", "").strip()
                    start = row.get("start_sec", "").strip()
                    end = row.get("end_sec", "").strip()
                    if video_id and label and start and end:
                        raw_items.append(",".join([video_id, label, start, end, row.get("window_index", "")]))
                    elif row.get("segment"):
                        raw_items.append(row["segment"])
        else:
            raw_items.extend(
                line.strip()
                for line in path.read_text().splitlines()
                if line.strip() and not line.strip().startswith("#")
            )
    if not raw_items:
        raise RuntimeError("No segments provided")
    return [parse_segment(item, clip_sec=args.clip_sec) for item in raw_items]


def find_video(video_root: Path, video_id: str) -> Path:
    for suffix in (".mp4", ".mov", ".mkv", ".avi"):
        path = video_root / f"{video_id}{suffix}"
        if path.exists():
            return path
    matches = sorted(video_root.glob(f"{video_id}.*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Missing video_id={video_id} under {video_root}")


def sample_times(start: float, end: float, num_frames: int) -> np.ndarray:
    if num_frames <= 1:
        return np.asarray([(start + end) * 0.5], dtype=np.float32)
    # Match the usual clip sampling intuition: include both temporal ends of the
    # 10s window, producing exactly the 16 model frames to inspect.
    return np.linspace(start, end, num_frames, dtype=np.float32)


def decode_frames(video_path: Path, times: Sequence[float]) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    frames: list[np.ndarray] = []
    for time_sec in times:
        capture.set(cv2.CAP_PROP_POS_MSEC, max(float(time_sec), 0.0) * 1000.0)
        ok, frame = capture.read()
        if not ok or frame is None:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(frame, f"DECODE FAILED t={time_sec:.2f}", (40, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        frames.append(frame)
    capture.release()
    return frames


def bgr_frames_to_model_uint8(frames: Sequence[np.ndarray], image_hw: tuple[int, int], device: torch.device) -> torch.Tensor:
    h, w = image_hw
    tensors = []
    for frame in frames:
        resized = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensors.append(torch.from_numpy(rgb).permute(2, 0, 1))
    return torch.stack(tensors, dim=0).unsqueeze(0).to(device=device, dtype=torch.uint8)


def codec_candidates(container: str, requested: str) -> list[str]:
    requested = requested.strip()
    if requested:
        base = [requested]
    else:
        base = []
    container = container.lower()
    if container == "avi":
        defaults = ["MJPG", "XVID"]
    else:
        # avc1/H264 are more browser friendly when OpenCV/ffmpeg supports them;
        # mp4v is the broad fallback available in many OpenCV builds.
        defaults = ["avc1", "H264", "mp4v"]
    result: list[str] = []
    for item in base + defaults:
        if item and item not in result:
            result.append(item)
    return result


def save_sample_video(
    path: Path,
    frames: Sequence[np.ndarray],
    times: Sequence[float],
    image_hw: tuple[int, int],
    fps: float,
    *,
    codec: str = "",
    container: str = "mp4",
) -> tuple[Path, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = image_hw
    container = container.lower().lstrip(".") or "mp4"
    path = path.with_suffix("." + container)
    last_error = ""
    for fourcc_name in codec_candidates(container, codec):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc_name), fps, (w, h))
        if not writer.isOpened():
            last_error = f"VideoWriter failed for codec={fourcc_name}"
            continue
        for frame, time_sec in zip(frames, times):
            resized = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            cv2.rectangle(resized, (0, 0), (w, 40), (0, 0, 0), -1)
            cv2.putText(resized, f"sampled frame t={time_sec:.2f}s", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(resized)
        writer.release()
        if path.exists() and path.stat().st_size > 0:
            return path, fourcc_name
        last_error = f"codec={fourcc_name} produced an empty file"
    raise RuntimeError(f"Cannot write sample video {path}: {last_error}")


def infer_patch_grid(num_patches: int, image_hw: tuple[int, int]) -> tuple[int, int]:
    h, w = image_hw
    target_ratio = w / max(h, 1)
    best: tuple[float, int, int] | None = None
    for gh in range(1, int(num_patches ** 0.5) + 2):
        if num_patches % gh:
            continue
        gw = num_patches // gh
        score = abs(gw / max(gh, 1) - target_ratio)
        if best is None or score < best[0]:
            best = (score, gh, gw)
    if best is None:
        side = int(round(num_patches ** 0.5))
        return side, max(num_patches // max(side, 1), 1)
    return best[1], best[2]


def first_component_heatmap(tokens: torch.Tensor, grid_hw: tuple[int, int]) -> np.ndarray:
    x = tokens.float()
    x = x - x.mean(dim=0, keepdim=True)
    try:
        _, _, v = torch.pca_lowrank(x, q=1, center=False)
        direction = v[:, 0]
    except Exception:
        _, _, vh = torch.linalg.svd(x, full_matrices=False)
        direction = vh[0]
    scores = x @ direction
    norms = x.norm(dim=1)
    if scores.numel() > 1:
        corr = torch.corrcoef(torch.stack([scores, norms]))[0, 1]
        if torch.isfinite(corr) and corr < 0:
            scores = -scores
    heat = scores.reshape(grid_hw).detach().cpu().numpy()
    heat -= float(heat.min())
    denom = float(heat.max())
    if denom > 1e-8:
        heat /= denom
    return heat.astype(np.float32)


def overlay_heatmap(frame: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h, w = frame.shape[:2]
    heat = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    color = cv2.applyColorMap(np.uint8(np.clip(heat, 0, 1) * 255), cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 1.0 - alpha, color, alpha, 0)


def draw_bar(image: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 38), (0, 0, 0), -1)
    cv2.putText(image, text[:120], (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)


def make_sheet(title: str, cells: Sequence[tuple[np.ndarray, np.ndarray, str]], cell_hw: tuple[int, int]) -> np.ndarray:
    cell_h, cell_w = cell_hw
    header_h = 62
    sheet = np.zeros((header_h + len(cells) * cell_h, cell_w * 2, 3), dtype=np.uint8)
    sheet[:header_h] = (28, 28, 28)
    cv2.putText(sheet, title[:170], (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    for i, (original, overlay, text) in enumerate(cells):
        y = header_h + i * cell_h
        original = cv2.resize(original, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        overlay = cv2.resize(overlay, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        draw_bar(original, "original  " + text, (240, 240, 240))
        draw_bar(overlay, "DINO PCA  " + text, (0, 255, 255))
        sheet[y : y + cell_h, :cell_w] = original
        sheet[y : y + cell_h, cell_w:] = overlay
    return sheet


def choose_image_size(args: argparse.Namespace, cfg: Any, run_dir: Path | None, first_segment: Segment) -> tuple[int, int]:
    if args.image_size:
        return parse_size(args.image_size)
    if run_dir is not None:
        summary = run_dir / first_segment.video_id / "summary.json"
        if summary.exists():
            data = json.loads(summary.read_text())
            size = data.get("global_image_size") or data.get("image_size")
            if size:
                return int(size[0]), int(size[1])
    spatial = cfg.get("spatial_crop", {})
    if spatial.get("global_image_size"):
        value = spatial["global_image_size"]
        if isinstance(value, str):
            return parse_size(value)
        return int(value[0]), int(value[1])
    value = cfg.video.image_size
    if isinstance(value, str):
        return parse_size(value)
    return int(value[0]), int(value[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Save model-sampled frames as a clip video and render top-4 DINO patch PCA heatmaps.")
    parser.add_argument("--checkpoint", default="", help="Checkpoint path. Defaults to run_config checkpoint when --run-dir is given.")
    parser.add_argument("--run-dir", default="", help="Optional eval run dir; used for checkpoint/image-size defaults only.")
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--segments", default="", help="Comma-separated segment names, e.g. fp_shot_..._w00028_t00140.000")
    parser.add_argument("--segment-file", default="", help="Text list or CSV with segment/video_id,label,start_sec,end_sec.")
    parser.add_argument("--clip-sec", type=float, default=10.0)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--topk-frames", type=int, default=4)
    parser.add_argument("--image-size", default="", help="H,W or HxW. Defaults to run/checkpoint global image size.")
    parser.add_argument("--sheet-cell-size", default="360x630")
    parser.add_argument("--sample-video-fps", type=float, default=2.0, help="FPS for the exported 16-frame sample video. Lower values make frame inspection easier.")
    parser.add_argument("--sample-video-codec", default="", help="Optional fourcc such as avc1, H264, mp4v, MJPG. Empty tries browser-friendly fallbacks.")
    parser.add_argument("--sample-video-container", default="mp4", choices=["mp4", "avi"], help="Use mp4 for browser playback or avi for MJPG fallback.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--jpeg-quality", type=int, default=90)
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else None
    checkpoint = args.checkpoint
    if not checkpoint and run_dir is not None:
        checkpoint = str(json.loads((run_dir / "run_config.json").read_text())["checkpoint"])
    if not checkpoint:
        raise ValueError("--checkpoint is required when --run-dir is not provided")

    segments = load_segments(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, cfg, labels, _ = load_checkpoint_model(checkpoint, device, [])
    model.eval()
    if model.backbone is None:
        raise RuntimeError("Model has no DINO backbone")
    image_hw = choose_image_size(args, cfg, run_dir, segments[0])
    label_to_index = {label: idx for idx, label in enumerate(labels)}

    rows: list[dict[str, Any]] = []
    cards: list[str] = []
    with torch.no_grad():
        for idx, segment in enumerate(segments, start=1):
            if segment.label not in label_to_index:
                raise ValueError(f"Label {segment.label!r} not in checkpoint labels={labels}")
            video_path = find_video(Path(args.video_root), segment.video_id)
            times = sample_times(segment.start_sec, segment.end_sec, args.num_frames)
            frames = decode_frames(video_path, times)
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", segment.raw).strip("_")
            seg_dir = output_dir / safe_name
            seg_dir.mkdir(parents=True, exist_ok=True)
            sample_video = seg_dir / f"{safe_name}_sampled_{args.num_frames}f.{args.sample_video_container}"
            sample_video, sample_video_codec = save_sample_video(
                sample_video,
                frames,
                times,
                image_hw,
                args.sample_video_fps,
                codec=args.sample_video_codec,
                container=args.sample_video_container,
            )

            inputs = bgr_frames_to_model_uint8(frames, image_hw, device)
            features = model.encode_frames(inputs)
            branch_outputs = model._branch_outputs(
                features,
                model.frame_proj,
                model.frame_event_head,
                model.temporal,
                model.head,
            )
            frame_logits = branch_outputs["frame_event_logits"][0]
            frame_probs = torch.sigmoid(frame_logits[:, label_to_index[segment.label]])
            topk = min(max(args.topk_frames, 1), len(frames))
            top_indices = torch.topk(frame_probs, k=topk).indices.detach().cpu().tolist()

            selected_uint8 = inputs[0, top_indices]
            selected_norm = model.preprocess_inputs(selected_uint8)
            patch_features = model.backbone.forward_features(selected_norm)["x_norm_patchtokens"]
            cells: list[tuple[np.ndarray, np.ndarray, str]] = []
            overlay_paths: list[str] = []
            for rank, frame_index in enumerate(top_indices, start=1):
                frame = cv2.resize(frames[frame_index], (image_hw[1], image_hw[0]), interpolation=cv2.INTER_AREA)
                tokens = patch_features[rank - 1]
                grid_hw = infer_patch_grid(int(tokens.shape[0]), image_hw)
                heat = first_component_heatmap(tokens, grid_hw)
                overlay = overlay_heatmap(frame, heat)
                prob = float(frame_probs[frame_index].detach().cpu())
                text = f"rank={rank} frame={frame_index}/{args.num_frames - 1} t={times[frame_index]:.2f}s p={prob:.3f}"
                cells.append((frame.copy(), overlay.copy(), text))
                draw_bar(overlay, f"{segment.raw}  {text}", (0, 255, 255))
                overlay_rel = Path(safe_name) / f"{safe_name}_rank{rank}_frame{frame_index:02d}_t{times[frame_index]:09.3f}_pca.jpg"
                overlay_path = output_dir / overlay_rel
                cv2.imwrite(str(overlay_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                overlay_paths.append(overlay_rel.as_posix())

            title = f"{segment.raw} label={segment.label} [{segment.start_sec:.1f},{segment.end_sec:.1f}]"
            sheet = make_sheet(title, cells, parse_size(args.sheet_cell_size))
            sheet_rel = Path(safe_name) / f"{safe_name}_sheet.jpg"
            sheet_path = output_dir / sheet_rel
            cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
            row = {
                "segment": segment.raw,
                "video_id": segment.video_id,
                "label": segment.label,
                "start_sec": segment.start_sec,
                "end_sec": segment.end_sec,
                "window_index": segment.window_index if segment.window_index is not None else "",
                "sample_video": Path(safe_name, sample_video.name).as_posix(),
                "sample_video_fps": float(args.sample_video_fps),
                "sample_video_codec": sample_video_codec,
                "sheet": sheet_rel.as_posix(),
                "top_frame_indices": json.dumps(top_indices),
                "top_frame_times_sec": json.dumps([float(times[i]) for i in top_indices]),
                "top_frame_probs": json.dumps([float(frame_probs[i].detach().cpu()) for i in top_indices]),
                "overlay_paths": json.dumps(overlay_paths),
            }
            rows.append(row)
            cards.append(
                "<article><h3>{}</h3><video src='{}' controls></video><br><a href='{}'><img src='{}' loading='lazy'></a></article>".format(
                    html.escape(segment.raw),
                    html.escape(row["sample_video"]),
                    html.escape(row["sheet"]),
                    html.escape(row["sheet"]),
                )
            )
            print(f"processed {idx}/{len(segments)} {segment.raw}", flush=True)

    with (output_dir / "index.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    html_doc = (
        "<!doctype html><html><head><meta charset='utf-8'><title>Football segment DINO PCA</title>"
        "<style>body{font-family:Arial;background:#111;color:#eee;margin:18px}"
        "article{background:#222;margin:16px;padding:12px;border-radius:8px}"
        "video{max-width:720px;width:100%;display:block;margin-bottom:10px}"
        "img{max-width:100%;height:auto}</style></head><body>"
        "<h1>Football segment DINO PCA</h1>" + "".join(cards) + "</body></html>"
    )
    (output_dir / "index.html").write_text(html_doc)
    summary = {
        "checkpoint": checkpoint,
        "run_dir": str(run_dir) if run_dir is not None else "",
        "video_root": args.video_root,
        "image_size": list(image_hw),
        "num_frames": args.num_frames,
        "topk_frames": args.topk_frames,
        "num_segments": len(segments),
        "sample_video_fps": float(args.sample_video_fps),
        "sample_video_container": args.sample_video_container,
        "sample_video_codec_requested": args.sample_video_codec,
        "note": "The sampled video contains the exact uniform 16 frames fed to the model-global branch. The sheet shows original/PCA pairs for the top frame-logit frames of the requested label.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
