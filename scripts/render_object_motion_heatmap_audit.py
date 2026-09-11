#!/usr/bin/env python
"""Render comparable ball/goal heatmaps from Object Motion checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from football_object_motion.train import forward_motion_batch, object_motion_teacher_from_config  # noqa: E402
import train_football_events as football  # noqa: E402
from scripts.eval_long_video_checkpoint import (  # noqa: E402
    SlidingWindowVideoDataset,
    WindowRecord,
    collate_windows,
    load_checkpoint_model,
    parse_image_size,
)


OBJECTS = ("ball", "goal")
OBJECT_COLORS = {"ball": (255, 235, 40), "goal": (235, 70, 230)}


@dataclass(frozen=True)
class Case:
    name: str
    video_id: str
    start: float
    end: float
    event: str


DEFAULT_CASES = (
    Case("shot-tp", "2044306371297357826", 240.0, 250.0, "shot"),
    Case("shot-setpiece-fn", "2044306371297357826", 520.0, 530.0, "shot + set_piece"),
    Case("shot-save-tp", "2044306371297357826", 780.0, 790.0, "shot + save"),
    Case("setpiece-tp", "2044306371297357826", 820.0, 830.0, "set_piece"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--epochs", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--video-root", default="/mnt/data_16t/football/raw_video_720P")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--teacher-device", default="cuda:6")
    parser.add_argument("--decode-strategy", default="single_seek")
    return parser.parse_args()


def make_dataset(case: Case, video_root: Path, cfg: Any) -> SlidingWindowVideoDataset:
    image_size = parse_image_size(cfg.video.image_size)
    motion_cfg = cfg.model.object_motion
    return SlidingWindowVideoDataset(
        video_path=str(video_root / f"{case.video_id}.mp4"),
        video_id=case.video_id,
        windows=[WindowRecord(index=0, start_sec=case.start, end_sec=case.end)],
        num_frames=int(cfg.video.num_frames),
        image_size=image_size,
        normalize_on_cpu=False,
        decode_strategy="single_seek",
        video_reader_cache_size=0,
        view_mode=str(cfg.model.get("view_fusion", "single")),
        global_image_size=parse_image_size(cfg.get("spatial_crop", {}).get("global_image_size", image_size)),
        dual_sampling=str(cfg.video.get("dual_sampling", "aligned")),
        roi_overlap_frames=int(cfg.video.get("roi_overlap_frames", 8)),
        num_rois=int(cfg.get("spatial_crop", {}).get("num_rois", 1)),
        object_motion_frames_per_segment=int(motion_cfg.frames_per_segment),
        object_motion_image_size=parse_image_size(motion_cfg.image_size),
    )


def raw_frame_bgr(frames: torch.Tensor, index: int) -> np.ndarray:
    frame = frames[index].detach().cpu()
    if frame.dtype != torch.uint8:
        frame = frame.clamp(0, 1).mul(255).to(torch.uint8)
    return cv2.cvtColor(frame.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)


def spatial_metrics(prob: np.ndarray, target: np.ndarray) -> dict[str, float | bool | None]:
    target = np.asarray(target, np.float32)
    prob = np.asarray(prob, np.float32)
    positive = target >= (0.25 if float(target.max()) > 0 else 1.0)
    if not positive.any():
        return {"teacher_present": False, "pointing_hit": None, "mass_lift": None, "target_mean": None, "background_mean": None}
    target_mean = float(prob[positive].mean())
    background_mean = float(prob[~positive].mean()) if (~positive).any() else 0.0
    return {
        "teacher_present": True,
        "pointing_hit": bool(positive[np.unravel_index(int(prob.argmax()), prob.shape)]),
        "mass_lift": float(target_mean / max(float(prob.mean()), 1e-8)),
        "target_mean": target_mean,
        "background_mean": background_mean,
    }


def normalize_heatmap(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, np.float32)
    low, high = np.quantile(value, [0.05, 0.995])
    return np.clip((value - low) / max(float(high - low), 1e-6), 0.0, 1.0)


def draw_target_contour(panel: np.ndarray, target: np.ndarray, color: tuple[int, int, int]) -> None:
    target_u8 = (cv2.resize(target.astype(np.float32), (panel.shape[1], panel.shape[0]), interpolation=cv2.INTER_NEAREST) >= 0.25).astype(np.uint8) * 255
    contours, _ = cv2.findContours(target_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(panel, contours, -1, color, 2, cv2.LINE_AA)


def overlay(frame: np.ndarray, heatmap: np.ndarray, target: np.ndarray, object_name: str) -> np.ndarray:
    normalized = normalize_heatmap(heatmap)
    resized = cv2.resize(normalized, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)
    colored = cv2.applyColorMap((resized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    result = cv2.addWeighted(frame, 0.58, colored, 0.42, 0.0)
    draw_target_contour(result, target, OBJECT_COLORS[object_name])
    return result


def teacher_panel(frame: np.ndarray, targets: dict[str, np.ndarray]) -> np.ndarray:
    result = frame.copy()
    for object_name in OBJECTS:
        draw_target_contour(result, targets[object_name], OBJECT_COLORS[object_name])
    return result


def labeled(image: np.ndarray, title: str, subtitle: str = "", size: tuple[int, int] = (360, 203)) -> np.ndarray:
    panel = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    bar_h = 48
    canvas = np.full((size[1] + bar_h, size[0], 3), 22, dtype=np.uint8)
    canvas[bar_h:] = panel
    cv2.putText(canvas, title, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (248, 248, 248), 1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(canvas, subtitle, (10, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (195, 195, 195), 1, cv2.LINE_AA)
    return canvas


def metric_text(metrics: dict[str, Any]) -> str:
    if not metrics["teacher_present"]:
        return "teacher absent: localization N/A"
    hit = "hit" if metrics["pointing_hit"] else "miss"
    return f"peak {hit} | target lift {metrics['mass_lift']:.2f}x"


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    teacher_device = torch.device(args.teacher_device)
    experiment = Path(args.experiment)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = [(f"epoch {epoch}", experiment / f"epoch_{epoch}.pt") for epoch in args.epochs]
    for _, path in checkpoints:
        if not path.exists():
            raise FileNotFoundError(path)

    cases = list(DEFAULT_CASES)
    cfg = football.load_config(str(experiment / "config.yaml"), [])
    football.configure_label_schema(cfg)
    labels = list(football.LABELS)
    decoded: dict[str, dict[str, Any]] = {}
    predictions: dict[str, dict[str, Any]] = {case.name: {} for case in cases}

    for case in cases:
        item = make_dataset(case, Path(args.video_root), cfg)[0]
        decoded[case.name] = {"batch": collate_windows([item]), "labels": labels}

    teacher_hook = object_motion_teacher_from_config(cfg, teacher_device)
    if teacher_hook is None:
        raise RuntimeError("Online object-motion teacher is disabled")
    teacher_hook.teacher.batch_size = 1
    teacher_data: dict[str, dict[str, torch.Tensor]] = {}
    for case in cases:
        batch = decoded[case.name]["batch"]
        frames = batch["object_motion_inputs"]
        frame_count = int(frames.shape[1])
        grid_h = int(frames.shape[-2]) // int(cfg.model.object_motion.patch_size)
        grid_w = int(frames.shape[-1]) // int(cfg.model.object_motion.patch_size)
        teacher_batch = dict(batch)
        teacher_batch.update(teacher_hook.teacher.empty(1, frame_count, grid_h * grid_w))
        teacher_hook.fill_missing(teacher_batch)
        teacher_data[case.name] = {
            "targets": teacher_batch["object_motion_heatmap_targets"].clone(),
            "confidences": teacher_batch["object_motion_teacher_confidences"].clone(),
        }
        print(f"teacher {case.name}", flush=True)
    del teacher_hook
    gc.collect()
    if teacher_device.type == "cuda":
        with torch.cuda.device(teacher_device):
            torch.cuda.empty_cache()

    for version, checkpoint in checkpoints:
        print(f"loading {version}: {checkpoint}", flush=True)
        model, current_cfg, labels, _ = load_checkpoint_model(str(checkpoint), device, [])
        model.eval()
        for case in cases:
            batch = decoded[case.name]["batch"]
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                outputs = forward_motion_batch(model, batch, device, return_aux=True)
            predictions[case.name][version] = {
                "heatmaps": torch.sigmoid(outputs["object_motion_heatmap_logits"]).float().cpu().numpy()[0],
                "presence": torch.sigmoid(outputs["object_motion_presence_logits"]).float().cpu().numpy()[0],
                "clip_probs": torch.sigmoid(outputs["logits"]).float().cpu().numpy()[0],
                "anchor_probs": torch.sigmoid(outputs["retention_reference_logits"]).float().cpu().numpy()[0],
            }
            print(f"  {case.name}", flush=True)
        del outputs, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report: dict[str, Any] = {"experiment": str(experiment), "versions": [v for v, _ in checkpoints], "cases": []}
    for case in cases:
        batch = decoded[case.name]["batch"]
        frames = batch["object_motion_inputs"]
        frame_count = int(frames.shape[1])
        grid_h = int(frames.shape[-2]) // int(cfg.model.object_motion.patch_size)
        grid_w = int(frames.shape[-1]) // int(cfg.model.object_motion.patch_size)
        targets_tensor = teacher_data[case.name]["targets"][0].reshape(frame_count, grid_h, grid_w, -1)
        confidences = teacher_data[case.name]["confidences"][0]
        ball_score = confidences[:, 0]
        goal_score = confidences[:, 1]
        selection_score = ball_score + 0.6 * goal_score
        if float(selection_score.max()) <= 0:
            frame_index = frame_count // 2
        else:
            frame_index = int(selection_score.argmax())
        frame = raw_frame_bgr(frames[0], frame_index)
        targets = {
            object_name: targets_tensor[frame_index, :, :, object_index].numpy()
            for object_index, object_name in enumerate(OBJECTS)
        }
        time_sec = float(batch["object_motion_times"][0, frame_index])
        first = labeled(
            teacher_panel(frame, targets),
            f"{case.name} | {case.event}",
            f"t={time_sec:.2f}s | cyan=ball teacher, magenta=goal teacher",
        )
        target_viz = np.zeros_like(frame)
        target_viz[:, :, 1] = cv2.resize(targets["ball"], (frame.shape[1], frame.shape[0])) * 255
        target_viz[:, :, 2] = cv2.resize(targets["goal"], (frame.shape[1], frame.shape[0])) * 255
        second = labeled(target_viz, "teacher targets", "green=ball | red=goal")
        top_row = [first]
        bottom_row = [second]
        case_report: dict[str, Any] = {
            "name": case.name,
            "event": case.event,
            "video_id": case.video_id,
            "window": [case.start, case.end],
            "frame_index": frame_index,
            "time_sec": time_sec,
            "teacher_confidence": {"ball": float(confidences[frame_index, 0]), "goal": float(confidences[frame_index, 1])},
            "versions": {},
        }
        for version, _ in checkpoints:
            item = predictions[case.name][version]
            version_report: dict[str, Any] = {}
            panels = []
            for object_index, object_name in enumerate(OBJECTS):
                heatmap = item["heatmaps"][frame_index, :, object_index].reshape(grid_h, grid_w)
                metrics = spatial_metrics(heatmap, targets[object_name])
                version_report[object_name] = metrics
                panels.append(labeled(overlay(frame, heatmap, targets[object_name], object_name), f"{version} | {object_name}", metric_text(metrics)))
            top_row.append(panels[0])
            bottom_row.append(panels[1])
            version_report["event_probs"] = {label: float(value) for label, value in zip(decoded[case.name]["labels"], item["clip_probs"])}
            version_report["anchor_probs"] = {label: float(value) for label, value in zip(decoded[case.name]["labels"], item["anchor_probs"])}
            case_report["versions"][version] = version_report
        composite = np.concatenate((np.concatenate(top_row, axis=1), np.concatenate(bottom_row, axis=1)), axis=0)
        image_path = output_dir / f"{case.name}.jpg"
        cv2.imwrite(str(image_path), composite, [cv2.IMWRITE_JPEG_QUALITY, 91])
        case_report["image"] = str(image_path.resolve())
        report["cases"].append(case_report)
        print(f"wrote {image_path}", flush=True)

    totals: dict[str, Any] = {}
    for version, _ in checkpoints:
        totals[version] = {}
        for object_name in OBJECTS:
            rows = [case["versions"][version][object_name] for case in report["cases"] if case["versions"][version][object_name]["teacher_present"]]
            totals[version][object_name] = {
                "n": len(rows),
                "pointing_accuracy": float(np.mean([row["pointing_hit"] for row in rows])) if rows else None,
                "mean_mass_lift": float(np.mean([row["mass_lift"] for row in rows])) if rows else None,
            }
    report["summary"] = totals
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(totals, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
