#!/usr/bin/env python
"""Evaluate a dense-response (stage-1) checkpoint as a candidate generator.

Stage-1's job is not clip-level F1: it must produce a dense frame-level
response curve that (a) does not miss GT events at low thresholds, (b) peaks
near the event time, and (c) does not flood the stage-2 verifier with
candidates. This script measures exactly those three properties on a set of
long videos:

  1. candidate recall @ threshold grid  -- fraction of GT events with a
     post-NMS candidate (score >= tau) within tolerance
  2. localization error                  -- |peak_time - gt_time| median/p90
     for matched events
  3. candidate density                   -- candidates per video-hour at each
     tau, i.e. the load the downstream verifier must process

The response curve is stitched from per-window frame logits (overlapping
windows merged by max at each timestamp) so the result is a single per-label
peak curve over the whole video; peaks are then extracted with a simple local
maxima detector + score-ordered NMS.

Usage:
  python scripts/evaluate_stage1_dense_response.py \
      --checkpoint outputs/football_events/vitl16_dense_response_stage1_v1/best.pt \
      --split-file configs/football/splits/vitl16_lvd1689m_set_piece_seed42_162videos/candidate_reranker_holdout_6_video_ids.txt \
      --output-dir outputs/football_eval_runs/stage1_dense_response_holdout6 \
      --device cuda:3

Reuses the window decoding machinery from eval_long_video_checkpoint.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch
from torch.utils.data import DataLoader

import train_football_events as train_mod
import eval_long_video_checkpoint as ev

VIDEOS_DIR = Path("/mnt/data_16t/football/raw_video_720P")
GT_DIR = Path("/home/new_users/qiuqi/code/football_events_human_repair")
SOURCE = "xbotgo_0608"


def build_response_curves(
    checkpoint: str,
    video_id: str,
    video_path: str,
    gpu_ids: list[int],
    clip_sec: float,
    stride_sec: float,
    batch_size: int,
    num_workers: int,
    max_windows: int = 0,
    logits_key: str = "auto",
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], float]:
    """Run dense sliding windows, return per-label (times, probs) curves.

    Overlapping windows are merged by taking the max probability at each
    rounded timestamp (recall-oriented: a repeat candidate is harmless for
    stage-1, a missed one is fatal).
    """
    device = torch.device(f"cuda:{gpu_ids[0]}" if gpu_ids else "cpu")
    model, cfg, labels, _ = ev.load_checkpoint_model(checkpoint, device, gpu_ids)
    model.eval()
    image_size = ev.parse_image_size(cfg.video.image_size)
    num_frames = train_mod.effective_num_frames(cfg)
    view_mode = str(cfg.model.get("view_fusion", "single"))
    global_image_size = ev.parse_image_size(
        cfg.get("spatial_crop", ev.ConfigDict()).get("global_image_size", image_size)
    )

    duration = ev.get_video_duration(video_path)
    windows = ev.build_windows(duration, clip_sec, stride_sec, include_tail=True)
    if max_windows > 0:
        windows = windows[:max_windows]

    args = argparse.Namespace(
        spatial_crop_mode=str(cfg.get("spatial_crop", ev.ConfigDict()).get("mode", "none")),
        top_crop_ratio=None,
        adaptive_top_min_ratio=0.05,
        adaptive_top_max_ratio=0.25,
        adaptive_top_fallback_ratio=0.10,
        adaptive_top_person_conf=0.35,
        detector_index_root="",
        roi_temporal_mode="checkpoint",
        roi_dynamic_context_sec=None,
        roi_temporal_smoothing_window_sec=None,
        roi_temporal_max_hold_sec=None,
        roi_temporal_confidence_decay_sec=None,
        detector_manifest_root="",
        detector_ball_conf=0.5,
        detector_goal_conf=0.5,
        detector_person_conf=0.4,
        detector_padding=0.12,
        detector_min_crop_area_ratio=0.15,
        detector_max_crop_area_ratio=0.85,
        detector_max_frame_gap=5,
        detector_window_roi_samples=8,
        detector_use_detection_goals=False,
    )
    top_crop_provider = None
    if args.spatial_crop_mode == "top_fixed":
        configured_ratio = cfg.get("spatial_crop", ev.ConfigDict()).get("top_crop_ratio", 0.0)
        top_crop_provider = train_mod.TopBandCropProvider(configured_ratio)
    window_cropper = ev.RobustWindowCropper.from_args(args, image_size, cfg)

    dataset = ev.SlidingWindowVideoDataset(
        video_path=video_path,
        video_id=video_id,
        windows=windows,
        num_frames=num_frames,
        image_size=image_size,
        normalize_on_cpu=True,
        decode_strategy="single_seek",
        video_reader_cache_size=2,
        window_cropper=window_cropper,
        frame_crop_provider=top_crop_provider,
        view_mode=view_mode,
        global_image_size=global_image_size,
        dual_sampling=str(cfg.video.get("dual_sampling", "aligned")),
        roi_overlap_frames=int(cfg.video.get("roi_overlap_frames", 8)),
        num_rois=int(cfg.get("spatial_crop", ev.ConfigDict()).get("num_rois", 1)),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=ev.collate_windows,
    )

    # per-label accumulation: time bucket (0.1s) -> max prob over windows
    accum: list[dict[int, float]] = [defaultdict(float) for _ in labels]
    start_time = time.time()
    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            with ev.autocast_context(device, bool(cfg.train.amp), str(cfg.train.amp_dtype)):
                outputs = train_mod.forward_model_batch(model, batch, device, return_aux=True)
            selected_logits_key = logits_key
            if selected_logits_key == "auto":
                selected_logits_key = (
                    "response_curve_logits"
                    if "response_curve_logits" in outputs
                    else "frame_event_logits"
                )
            frame_logits = outputs.get(selected_logits_key)
            if frame_logits is None:
                raise RuntimeError(f"checkpoint does not expose {selected_logits_key}")
            frame_logits = frame_logits.float().cpu().numpy()
            frame_probs = 1.0 / (1.0 + np.exp(-frame_logits))
            frame_times = batch["frame_times"].float().cpu().numpy()
            for i in range(frame_times.shape[0]):
                for j in range(frame_times.shape[1]):
                    t = int(round(float(frame_times[i, j]) * 10))  # 0.1s buckets
                    for label_idx in range(len(labels)):
                        p = float(frame_probs[i, j, label_idx])
                        if p > accum[label_idx][t]:
                            accum[label_idx][t] = p
            if step == 1 or step % 50 == 0 or step == len(loader):
                elapsed = time.time() - start_time
                print(
                    f"  step={step}/{len(loader)} elapsed={elapsed:.1f}s",
                    flush=True,
                )

    curves: dict[str, np.ndarray] = {}
    for label_idx, label in enumerate(labels):
        buckets = accum[label_idx]
        if not buckets:
            curves[label] = (np.zeros(0), np.zeros(0))
            continue
        items = sorted(buckets.items())
        times = np.array([t for t, _ in items], dtype=np.float64) / 10.0
        probs = np.array([p for _, p in items], dtype=np.float64)
        curves[label] = (times, probs)
    return curves, dict(zip(labels, labels)), duration


def interval_union_seconds(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted((float(s), float(e)) for s, e in intervals if float(e) > float(s))
    if not ordered:
        return 0.0
    total = 0.0
    cur_s, cur_e = ordered[0]
    for start, end in ordered[1:]:
        if start <= cur_e:
            cur_e = max(cur_e, end)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = start, end
    total += cur_e - cur_s
    return total


def extract_peaks(
    times: np.ndarray,
    probs: np.ndarray,
    nms_radius_sec: float,
) -> list[dict[str, float]]:
    """Local maxima + score-ordered NMS. Returns [{time_sec, score}]."""
    if times.size == 0:
        return []
    maxima = []
    for i in range(times.size):
        left_ok = i == 0 or probs[i] > probs[i - 1] or (
            probs[i] == probs[i - 1] and (i == 0 or probs[i] > probs[i - 1])
        )
        right_ok = i == times.size - 1 or probs[i] >= probs[i + 1]
        if left_ok and right_ok:
            maxima.append((times[i], probs[i]))
    maxima.sort(key=lambda item: (-item[1], item[0]))
    peaks: list[dict[str, float]] = []
    for t, s in maxima:
        if all(abs(t - kept["time_sec"]) > nms_radius_sec for kept in peaks):
            peaks.append({"time_sec": float(t), "score": float(s)})
    return peaks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--videos-dir", default=str(VIDEOS_DIR),
                        help="Directory containing video files (default raw_video_720P).")
    parser.add_argument("--split-file", required=True,
                        help="One video id per line.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-ids", default="")
    parser.add_argument("--clip-sec", type=float, default=10.0)
    parser.add_argument("--stride-sec", type=float, default=2.5,
                        help="Dense stride so response curves are well sampled.")
    parser.add_argument("--tolerance-sec", type=float, default=5.0)
    parser.add_argument("--nms-radius-sec", type=float, default=5.0)
    parser.add_argument("--thresholds", default="0.05,0.1,0.2,0.3,0.4,0.5",
                        help="Comma-separated candidate score thresholds.")
    parser.add_argument("--logits-key", default="auto",
                        choices=("auto", "frame_event_logits", "response_curve_logits"),
                        help="Frame-level logits to stitch. auto prefers response_curve_logits when present.")
    parser.add_argument("--review-clip-sec", type=float, default=10.0,
                        help="Manual-review clip duration centered on each candidate peak for viewing-time estimates.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-windows", type=int, default=0,
                        help="Smoke-test cap on windows per video.")
    parser.add_argument("--max-videos", type=int, default=0)
    args = parser.parse_args()

    gpu_ids = (
        [int(item) for item in args.gpu_ids.split(",") if item.strip()]
        if args.gpu_ids.strip()
        else [int(args.device.split(":")[1])] if args.device.startswith("cuda") else []
    )
    thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = Path(args.videos_dir)
    video_ids = [
        line.strip()
        for line in Path(args.split_file).read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if args.max_videos > 0:
        video_ids = video_ids[: args.max_videos]

    all_metrics: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "videos_dir": str(videos_dir),
        "split": args.split_file,
        "tolerance_sec": args.tolerance_sec,
        "nms_radius_sec": args.nms_radius_sec,
        "thresholds": thresholds,
        "per_label": {},
    }

    for video_id in video_ids:
        video_path = videos_dir / f"{video_id}.mp4"
        annotation_path = GT_DIR / f"{video_id}.json"
        if not video_path.is_file() or not annotation_path.is_file():
            print(f"SKIP missing files: {video_id}", flush=True)
            continue
        print(f"== video {video_id} ==", flush=True)
        curves, _, duration = build_response_curves(
            args.checkpoint,
            video_id,
            str(video_path),
            gpu_ids,
            args.clip_sec,
            args.stride_sec,
            args.batch_size,
            args.num_workers,
            args.max_windows,
            args.logits_key,
        )
        gt_events = ev.load_gt_events(str(annotation_path), SOURCE, video_id)

        per_video: dict[str, Any] = {
            "video_id": video_id,
            "duration_sec": duration,
            "labels": {},
            "overall": {},
        }
        overall_intervals_by_tau: dict[float, list[tuple[float, float]]] = {
            tau: [] for tau in thresholds
        }
        for label, (times, probs) in curves.items():
            peaks = extract_peaks(times, probs, args.nms_radius_sec)
            label_gt = [e for e in gt_events if e["label"] == label]
            matched_peaks: list[dict[str, float]] = []
            for event in label_gt:
                best = None
                for peak in peaks:
                    if abs(peak["time_sec"] - event["time_sec"]) <= args.tolerance_sec:
                        if best is None or peak["score"] > best["score"]:
                            best = peak
                if best is not None:
                    matched_peaks.append({
                        "gt_time_sec": event["time_sec"],
                        "peak_time_sec": best["time_sec"],
                        "peak_score": best["score"],
                        "abs_offset_sec": abs(best["time_sec"] - event["time_sec"]),
                    })
            # per-peak GT matching for precision (a peak is a true positive if
            # it lies within tolerance of ANY GT event of this label).
            peak_is_true: list[bool] = [
                any(abs(p["time_sec"] - e["time_sec"]) <= args.tolerance_sec
                    for e in label_gt)
                for p in peaks
            ]
            label_metrics: dict[str, Any] = {"num_gt": len(label_gt)}
            for tau in thresholds:
                recall = sum(
                    1 for m in matched_peaks if m["peak_score"] >= tau
                ) / max(len(label_gt), 1)
                label_metrics[f"recall@{tau}"] = round(recall, 4)
                cands = sum(1 for p in peaks if p["score"] >= tau)
                true_cands = sum(
                    1 for p, is_true in zip(peaks, peak_is_true)
                    if p["score"] >= tau and is_true
                )
                density = cands / max(duration / 3600.0, 1e-6)
                label_metrics[f"candidates_per_hour@{tau}"] = round(density, 2)
                label_metrics[f"precision@{tau}"] = round(
                    true_cands / max(cands, 1), 4
                )
                label_metrics[f"true_cands@{tau}"] = true_cands
                label_metrics[f"total_cands@{tau}"] = cands
                selected_intervals = [
                    (
                        max(0.0, float(p["time_sec"]) - args.review_clip_sec * 0.5),
                        min(duration, float(p["time_sec"]) + args.review_clip_sec * 0.5),
                    )
                    for p in peaks
                    if p["score"] >= tau
                ]
                viewing_sec = interval_union_seconds(selected_intervals)
                label_metrics[f"viewing_sec@{tau}"] = round(viewing_sec, 2)
                label_metrics[f"viewing_min@{tau}"] = round(viewing_sec / 60.0, 2)
                overall_intervals_by_tau[tau].extend(selected_intervals)
            if matched_peaks:
                offsets = np.array([m["abs_offset_sec"] for m in matched_peaks])
                label_metrics["localization_median_sec"] = round(float(np.median(offsets)), 2)
                label_metrics["localization_p90_sec"] = round(float(np.percentile(offsets, 90)), 2)
                label_metrics["num_matched"] = len(matched_peaks)
            per_video["labels"][label] = label_metrics
        for tau in thresholds:
            viewing_sec = interval_union_seconds(overall_intervals_by_tau[tau])
            per_video["overall"][f"viewing_sec@{tau}"] = round(viewing_sec, 2)
            per_video["overall"][f"viewing_min@{tau}"] = round(viewing_sec / 60.0, 2)
            per_video["overall"][f"participation_ratio@{tau}"] = round(
                viewing_sec / max(duration, 1e-9), 4
            )
            print(
                f"  {label}: gt={len(label_gt)} "
                + " ".join(
                    f"r@{t}={label_metrics.get(f'recall@{t}', 0)}"
                    f"/p@{t}={label_metrics.get(f'precision@{t}', 0)}"
                    for t in thresholds
                )
                + f" loc_med={label_metrics.get('localization_median_sec', '-')}s "
                + f"cand/h={label_metrics.get(f'candidates_per_hour@{thresholds[0]}', '-')}"
            )
        all_metrics.append(per_video)
        (output_dir / f"{video_id}.json").write_text(
            json.dumps(per_video, ensure_ascii=False, indent=1)
        )

    # aggregate
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for per_video in all_metrics:
        for label, m in per_video["labels"].items():
            by_label[label].append(m)
    total_duration_sec = sum(float(v.get("duration_sec", 0.0)) for v in all_metrics)
    summary["total_duration_sec"] = round(total_duration_sec, 2)
    summary["review_clip_sec"] = args.review_clip_sec
    summary["overall"] = {}
    for tau in thresholds:
        total_viewing_sec = sum(
            float(v.get("overall", {}).get(f"viewing_sec@{tau}", 0.0))
            for v in all_metrics
        )
        summary["overall"][f"viewing_sec@{tau}"] = round(total_viewing_sec, 2)
        summary["overall"][f"viewing_min@{tau}"] = round(total_viewing_sec / 60.0, 2)
        summary["overall"][f"participation_ratio@{tau}"] = round(
            total_viewing_sec / max(total_duration_sec, 1e-9), 4
        )

    for label, rows in by_label.items():
        agg: dict[str, Any] = {"num_videos": len(rows)}
        for tau in thresholds:
            gts = [r["num_gt"] for r in rows]
            hits = [
                r.get(f"recall@{tau}", 0) * r["num_gt"]
                for r in rows
            ]
            agg[f"recall@{tau}"] = round(
                sum(hits) / max(sum(gts), 1), 4
            )
            agg[f"candidates_per_hour@{tau}"] = round(
                np.mean([r.get(f"candidates_per_hour@{tau}", 0) for r in rows]), 2
            )
            # weighted precision: total true candidates / total candidates
            # pooled across videos (videos with more candidates weigh more).
            agg[f"precision@{tau}"] = round(
                sum(r.get(f"true_cands@{tau}", 0) for r in rows)
                / max(sum(r.get(f"total_cands@{tau}", 0) for r in rows), 1e-9),
                4,
            )
            total_cands = sum(r.get(f"total_cands@{tau}", 0) for r in rows)
            viewing_sec = sum(float(r.get(f"viewing_sec@{tau}", 0.0)) for r in rows)
            agg[f"total_cands@{tau}"] = int(total_cands)
            agg[f"viewing_sec@{tau}"] = round(viewing_sec, 2)
            agg[f"viewing_min@{tau}"] = round(viewing_sec / 60.0, 2)
            agg[f"participation_ratio@{tau}"] = round(
                viewing_sec / max(total_duration_sec, 1e-9), 4
            )
        offsets = [
            o
            for r in rows
            if "localization_median_sec" in r
            for _ in [None]
            for o in [r["localization_median_sec"]]
        ]
        # recompute medians from raw per-video matched offsets is not possible
        # after aggregation; report median of per-video medians instead.
        if offsets:
            agg["localization_median_of_video_medians_sec"] = round(float(np.median(offsets)), 2)
        summary["per_label"][label] = agg
        print(
            f"\n[AGG {label}] "
            + " ".join(
                f"r@{t}={agg[f'recall@{t}']} p@{t}={agg[f'precision@{t}']} "
                f"cand/h={agg[f'candidates_per_hour@{t}']}"
                for t in thresholds
            )
        )
        print(f"  loc(median of video medians)={agg.get('localization_median_of_video_medians_sec', '-')}s")

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1)
    )
    print(f"\nwrote {output_dir}/summary.json", flush=True)


if __name__ == "__main__":
    main()
