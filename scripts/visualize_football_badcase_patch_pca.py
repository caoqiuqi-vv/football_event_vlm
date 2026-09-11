#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_long_video_checkpoint import load_checkpoint_model


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def find_video(video_root: Path, video_id: str) -> Path:
    for suffix in (".mp4", ".mov", ".mkv", ".avi"):
        path = video_root / f"{video_id}{suffix}"
        if path.exists():
            return path
    matches = sorted(video_root.glob(f"{video_id}.*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Missing video={video_id} under {video_root}")


def parse_size(raw: str) -> tuple[int, int]:
    parts = raw.lower().replace("x", ",").split(",")
    if len(parts) != 2:
        raise ValueError(f"Invalid size {raw!r}; expected H,W or HxW")
    return int(parts[0]), int(parts[1])


def read_frame(video_path: Path, time_sec: float) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    cap.set(cv2.CAP_PROP_POS_MSEC, max(float(time_sec), 0.0) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to decode {video_path} at {time_sec:.3f}s")
    return frame


def load_frame_logits(video_dir: Path) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in read_csv(video_dir / "frame_event_logits.csv"):
        grouped.setdefault(int(row["window_index"]), []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["frame_index"]))
    return grouped


def select_badcases(index_rows: list[dict[str, str]], max_fp: int, max_fn: int) -> list[dict[str, str]]:
    fp = [row for row in index_rows if row["error_type"] == "FP"]
    fn = [row for row in index_rows if row["error_type"] == "FN"]
    fp.sort(key=lambda row: float(row.get("clip_prob") or 0.0), reverse=True)
    fn.sort(key=lambda row: float(row.get("frame_peak_prob") or row.get("clip_prob") or 0.0), reverse=True)
    if max_fp > 0:
        fp = fp[:max_fp]
    if max_fn > 0:
        fn = fn[:max_fn]
    return fp + fn


def first_component_heatmap(tokens: torch.Tensor, grid_hw: tuple[int, int]) -> np.ndarray:
    tokens = tokens.float()
    tokens = tokens - tokens.mean(dim=0, keepdim=True)
    try:
        _, _, v = torch.pca_lowrank(tokens, q=1, center=False)
        direction = v[:, 0]
    except Exception:
        _, _, vh = torch.linalg.svd(tokens, full_matrices=False)
        direction = vh[0]
    scores = tokens @ direction
    norms = tokens.norm(dim=1)
    if scores.numel() > 1:
        corr = torch.corrcoef(torch.stack([scores, norms]))[0, 1]
        if torch.isfinite(corr) and corr < 0:
            scores = -scores
    scores = scores.reshape(grid_hw).detach().cpu().numpy()
    scores = scores - float(scores.min())
    denom = float(scores.max())
    if denom > 1e-8:
        scores = scores / denom
    return scores.astype(np.float32)


def infer_patch_grid(num_patches: int, image_hw: tuple[int, int]) -> tuple[int, int]:
    height, width = image_hw
    ratio = width / max(height, 1)
    best = None
    for gh in range(1, int(num_patches ** 0.5) + 2):
        if num_patches % gh:
            continue
        gw = num_patches // gh
        score = abs((gw / max(gh, 1)) - ratio)
        if best is None or score < best[0]:
            best = (score, gh, gw)
    if best is None:
        side = int(round(num_patches ** 0.5))
        return side, max(num_patches // max(side, 1), 1)
    return int(best[1]), int(best[2])


def overlay_heatmap(frame_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    resized = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)
    colored = cv2.applyColorMap(np.uint8(np.clip(resized, 0, 1) * 255), cv2.COLORMAP_JET)
    return cv2.addWeighted(frame_bgr, 1.0 - alpha, colored, alpha, 0)




def draw_label(image: np.ndarray, text: str, *, color: tuple[int, int, int] = (255, 255, 255)) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(image, text[:110], (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)


def make_clip_sheet(
    *,
    title: str,
    cells: Sequence[tuple[np.ndarray, np.ndarray, str]],
    cell_hw: tuple[int, int],
) -> np.ndarray:
    cell_h, cell_w = cell_hw
    header_h = 58
    rows = len(cells)
    sheet = np.zeros((header_h + rows * cell_h, cell_w * 2, 3), dtype=np.uint8)
    sheet[:header_h] = (32, 32, 32)
    cv2.putText(sheet, title[:180], (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)
    for row_idx, (original, overlay, label_text) in enumerate(cells):
        y = header_h + row_idx * cell_h
        original = cv2.resize(original, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        overlay = cv2.resize(overlay, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        draw_label(original, "original  " + label_text, color=(230, 230, 230))
        draw_label(overlay, "PCA heatmap  " + label_text, color=(0, 255, 255))
        sheet[y : y + cell_h, 0:cell_w] = original
        sheet[y : y + cell_h, cell_w : cell_w * 2] = overlay
    return sheet

def tensor_from_frame(frame_bgr: np.ndarray, image_hw: tuple[int, int], model: torch.nn.Module, device: torch.device) -> torch.Tensor:
    height, width = image_hw
    resized = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    mean = model.input_mean.to(device).reshape(1, 3, 1, 1)
    std = model.input_std.to(device).reshape(1, 3, 1, 1)
    return (tensor - mean) / std


def main() -> None:
    parser = argparse.ArgumentParser(description="Render patch-token PCA overlays for top-k frames of FP/FN football badcases.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--badcase-index", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", default="", help="H,W or HxW. Defaults to the first video's global_image_size/image_size.")
    parser.add_argument("--max-fp", type=int, default=30)
    parser.add_argument("--max-fn", type=int, default=30)
    parser.add_argument("--topk-frames", type=int, default=4)
    parser.add_argument("--sheet-cell-size", default="360x630", help="H,W for each original/overlay cell in per-clip sheets.")
    parser.add_argument("--jpeg-quality", type=int, default=90)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_rows = read_csv(Path(args.badcase_index))
    selected = select_badcases(index_rows, args.max_fp, args.max_fn)
    if not selected:
        raise RuntimeError("No badcases selected")

    run_config = json.loads((run_dir / "run_config.json").read_text())
    checkpoint = str(run_config["checkpoint"])
    first_summary = json.loads((run_dir / selected[0]["video_id"] / "summary.json").read_text())
    image_hw = parse_size(args.image_size) if args.image_size else tuple(
        int(x) for x in first_summary.get("global_image_size") or first_summary.get("image_size")
    )
    device = torch.device(args.device)
    model, _, labels, _ = load_checkpoint_model(checkpoint, device, [])
    model.eval()
    if model.backbone is None:
        raise RuntimeError("Checkpoint model has no DINO backbone")
    backbone = model.backbone

    frame_cache: dict[str, dict[int, list[dict[str, str]]]] = {}
    rows_out: list[dict[str, Any]] = []
    cards: list[str] = []
    sheet_rows: list[dict[str, Any]] = []
    sheet_cards: list[str] = []
    with torch.no_grad():
        for item_idx, item in enumerate(selected, start=1):
            video_id = item["video_id"]
            label = item["label"]
            window_index = int(item["window_index"])
            frame_cache.setdefault(video_id, load_frame_logits(run_dir / video_id))
            frame_rows = frame_cache[video_id].get(window_index, [])
            if not frame_rows:
                continue
            frame_rows = sorted(
                frame_rows,
                key=lambda row: float(row.get(f"frame_prob_{label}") or 0.0),
                reverse=True,
            )[: max(int(args.topk_frames), 1)]
            video_path = find_video(Path(args.video_root), video_id)
            sheet_cells: list[tuple[np.ndarray, np.ndarray, str]] = []
            for rank, frame_row in enumerate(frame_rows, start=1):
                time_sec = float(frame_row["frame_time_sec"])
                frame = read_frame(video_path, time_sec)
                input_tensor = tensor_from_frame(frame, image_hw, model, device)
                features = backbone.forward_features(input_tensor)
                patch_tokens = features["x_norm_patchtokens"][0]
                grid_hw = infer_patch_grid(int(patch_tokens.shape[0]), image_hw)
                heatmap = first_component_heatmap(patch_tokens, grid_hw)
                resized_frame = cv2.resize(frame, (image_hw[1], image_hw[0]), interpolation=cv2.INTER_AREA)
                overlay = overlay_heatmap(resized_frame, heatmap)
                label_text = (
                    f"rank={rank} t={time_sec:.2f}s "
                    f"frame_p={float(frame_row[f'frame_prob_{label}']):.3f}"
                )
                sheet_cells.append((resized_frame.copy(), overlay.copy(), label_text))
                title = (
                    f"{item['error_type']} {label} {video_id} w={window_index} "
                    f"rank={rank} t={time_sec:.2f}s frame_p={float(frame_row[f'frame_prob_{label}']):.3f} "
                    f"clip_p={float(item['clip_prob']):.3f}"
                )
                cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 42), (0, 0, 0), -1)
                cv2.putText(overlay, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
                rel = Path(item["error_type"].lower()) / label / video_id / (
                    f"{item['error_type'].lower()}_{label}_{video_id}_w{window_index:05d}_rank{rank}_t{time_sec:09.3f}_pca.jpg"
                )
                path = output_dir / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(path), overlay, [cv2.IMWRITE_JPEG_QUALITY, int(args.jpeg_quality)]):
                    raise RuntimeError(f"Failed to write {path}")
                row = {
                    "error_type": item["error_type"],
                    "video_id": video_id,
                    "label": label,
                    "window_index": window_index,
                    "frame_rank": rank,
                    "frame_time_sec": time_sec,
                    "clip_prob": float(item["clip_prob"]),
                    "frame_prob": float(frame_row[f"frame_prob_{label}"]),
                    "image_path": rel.as_posix(),
                }
                rows_out.append(row)
                cards.append(
                    "<article><a href='{0}'><img src='{0}' loading='lazy'></a>"
                    "<p>{1}</p></article>".format(
                        html.escape(rel.as_posix()),
                        html.escape(title),
                    )
                )
            if sheet_cells:
                sheet_title = (
                    f"{item['error_type']} {label} video={video_id} window={window_index} "
                    f"[{float(item['start_sec']):.1f},{float(item['end_sec']):.1f}] "
                    f"clip_p={float(item['clip_prob']):.3f} reason={item.get('reason','')}"
                )
                sheet = make_clip_sheet(
                    title=sheet_title,
                    cells=sheet_cells,
                    cell_hw=parse_size(args.sheet_cell_size),
                )
                sheet_rel = Path("sheets") / item["error_type"].lower() / label / (
                    f"{item['error_type'].lower()}_{label}_{video_id}_w{window_index:05d}_sheet.jpg"
                )
                sheet_path = output_dir / sheet_rel
                sheet_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, int(args.jpeg_quality)]):
                    raise RuntimeError(f"Failed to write {sheet_path}")
                sheet_row = {
                    "error_type": item["error_type"],
                    "video_id": video_id,
                    "label": label,
                    "window_index": window_index,
                    "clip_prob": float(item["clip_prob"]),
                    "frame_peak_prob": float(item.get("frame_peak_prob") or 0.0),
                    "sheet_path": sheet_rel.as_posix(),
                }
                sheet_rows.append(sheet_row)
                sheet_cards.append(
                    "<article><a href='{0}'><img src='{0}' loading='lazy'></a>"
                    "<p>{1}</p></article>".format(
                        html.escape(sheet_rel.as_posix()),
                        html.escape(sheet_title),
                    )
                )
            if item_idx == 1 or item_idx % 10 == 0 or item_idx == len(selected):
                print(f"processed badcases={item_idx}/{len(selected)} overlays={len(rows_out)}", flush=True)

    with (output_dir / "index.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0]) if rows_out else [])
        writer.writeheader()
        writer.writerows(rows_out)
    html_doc = (
        "<!doctype html><html><head><meta charset='utf-8'><title>Patch PCA badcases</title>"
        "<style>body{font-family:Arial;background:#111;color:#eee;margin:18px}"
        "article{background:#222;margin:10px;padding:10px;border-radius:8px}"
        "img{max-width:100%;height:auto}</style></head>"
        "<body><h1>Patch PCA badcases</h1>" + "".join(cards) + "</body></html>"
    )
    (output_dir / "index.html").write_text(html_doc)
    with (output_dir / "sheets.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(sheet_rows[0]) if sheet_rows else [])
        writer.writeheader()
        writer.writerows(sheet_rows)
    sheets_html = (
        "<!doctype html><html><head><meta charset='utf-8'><title>Patch PCA clip sheets</title>"
        "<style>body{font-family:Arial;background:#111;color:#eee;margin:18px}"
        "article{background:#222;margin:14px;padding:10px;border-radius:8px}"
        "img{max-width:100%;height:auto}</style></head>"
        "<body><h1>Patch PCA clip sheets</h1>" + "".join(sheet_cards) + "</body></html>"
    )
    (output_dir / "sheets.html").write_text(sheets_html)
    summary = {
        "run_dir": str(run_dir),
        "badcase_index": args.badcase_index,
        "checkpoint": checkpoint,
        "image_size": list(image_hw),
        "selected_badcases": len(selected),
        "rendered_overlays": len(rows_out),
        "rendered_sheets": len(sheet_rows),
        "topk_frames": args.topk_frames,
        "note": "Heatmaps show the first PCA component of DINO full-image patch tokens for the selected top frame-logit frames.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
