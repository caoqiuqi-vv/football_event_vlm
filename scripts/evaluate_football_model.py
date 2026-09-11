#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from football_eval_semantics import SCORE_SEMANTICS_VERSION, has_current_score_semantics

DENSE_LABELS = ["shot", "save", "set_piece"]
PROPOSAL_LABELS = ["shot"]
VIDEO_EXTENSIONS = (".mp4", ".MP4", ".mov", ".MOV", ".mkv", ".MKV", ".avi", ".AVI")
DEFAULT_GT_DIR = "~/code/football_events_human_repair"
DEFAULT_PROPOSAL_ROOT = "/home/new_users/qiuqi/code/det_and_track/outputs/test_videos/key_event_proposal_v2_20260708"
DEFAULT_VIDEO_ROOTS = [
    "xbotgo_0608=/mnt/data/Datasets/Datasets/Football/xbotgo_football_data_0608/videos",
    "may_xbotgo=/mnt/data/Datasets/Datasets/Football/may/Xbotgo/videos",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a football event checkpoint on dense long videos or detector proposal windows."
    )
    parser.add_argument("--checkpoint", required=True, help="Football model checkpoint, e.g. /mnt/data_16t/football/qiuqi/checkpoints/lora_epoch6.pt")
    parser.add_argument("--mode", required=True, choices=["dense", "proposal"], help="dense evaluates shot/save/set_piece; proposal evaluates shot only")
    parser.add_argument(
        "--video-ids",
        default="",
        help="Comma-separated video ids. If empty, dense mode uses gt-dir JSON names and proposal mode uses proposal-root when available.",
    )
    parser.add_argument("--video-id-file", default="", help="Optional text file with one video id per line.")
    parser.add_argument("--gt-dir", default=DEFAULT_GT_DIR, help="Directory with repaired GT {video_id}.json files.")
    parser.add_argument("--proposal-root", default=DEFAULT_PROPOSAL_ROOT, help="Root containing per-video shot_proposals.json, used by --mode proposal.")
    parser.add_argument("--video-root", action="append", default=[], help="Video root as source=/path/to/videos. Can be repeated.")
    parser.add_argument("--output-root", default="outputs/football_eval_runs", help="Root directory for outputs.")
    parser.add_argument("--run-name", default="", help="Optional run name. Defaults to checkpoint stem + mode.")
    parser.add_argument("--clip-sec", type=float, default=10.0)
    parser.add_argument("--stride-sec", type=float, default=5.0, help="Dense sliding stride seconds.")
    parser.add_argument("--image-size", default="", help="Optional inference image size override as H,W or HxW. Defaults to checkpoint config.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-ids", default="0", help="Comma-separated ids for DataParallel. Use one id for single GPU.")
    parser.add_argument(
        "--thresholds",
        default="0.5",
        help="Default: 0.5. Also accepts checkpoint or a label list like shot=0.4,save=0.5,set_piece=0.6",
    )
    parser.add_argument("--match-tolerance-sec", type=float, default=2.0)
    parser.add_argument(
        "--score-source",
        default="clip",
        choices=["clip", "response", "fusion"],
        help="Window score source for eval_long_video_checkpoint.",
    )
    parser.add_argument(
        "--object-motion-mode",
        default="checkpoint",
        choices=["checkpoint", "anchor-only"],
        help="Use the learned Object Motion residual or force the frozen-anchor ablation.",
    )
    parser.add_argument(
        "--fusion-alpha",
        default="shot=0.4,save=0.4,set_piece=0.8",
        help="Per-class clip-logit weight for --score-source=fusion.",
    )
    parser.add_argument("--clip-temperature", default="1.0")
    parser.add_argument("--response-temperature", default="1.0")
    parser.add_argument(
        "--prediction-postprocess",
        default="window_overlap",
        choices=["window_overlap", "point_nms", "interval_merge"],
        help=(
            "window_overlap keeps every positive sliding window and allows one GT event to match multiple windows; "
            "point_nms is the strict spotting metric; interval_merge reproduces legacy results."
        ),
    )
    parser.add_argument("--nms-radius-sec", type=float, default=5.0)
    parser.add_argument("--merge-gap-sec", type=float, default=2.0, help="Legacy interval_merge only.")
    parser.add_argument("--gt-merge-gap-sec", type=float, default=0.0)
    parser.add_argument("--proposal-dedupe-sec", type=float, default=0.0)
    parser.add_argument("--spatial-crop-mode", default="none", choices=["none", "top_fixed", "adaptive_top_fixed", "detector_aware", "legacy_indexed", "robust_detector_aware"], help="Optional fixed-clip spatial crop.")
    parser.add_argument("--top-crop-ratio", type=float, default=None, help="Top fraction removed in top_fixed mode; defaults to checkpoint config.")
    parser.add_argument("--adaptive-top-min-ratio", type=float, default=0.05)
    parser.add_argument("--adaptive-top-max-ratio", type=float, default=0.25)
    parser.add_argument("--adaptive-top-fallback-ratio", type=float, default=0.10)
    parser.add_argument("--adaptive-top-person-conf", type=float, default=0.35)
    parser.add_argument("--detector-index-root", default="", help="Compact ROI indices for robust_detector_aware; defaults to checkpoint config.")
    parser.add_argument("--roi-temporal-mode", default="checkpoint", choices=["checkpoint", "clip", "dynamic"], help="Override robust ROI temporal mode from checkpoint config.")
    parser.add_argument("--roi-dynamic-context-sec", type=float, default=None)
    parser.add_argument("--roi-temporal-smoothing-window-sec", type=float, default=None)
    parser.add_argument("--roi-temporal-max-hold-sec", type=float, default=None)
    parser.add_argument("--roi-temporal-confidence-decay-sec", type=float, default=None)
    parser.add_argument("--detector-manifest-root", default="", help="Root containing per-video detection_tracking outputs; tracked_objects.json supplies people/ball and detections.json supplies goals.")
    parser.add_argument("--detector-ball-conf", type=float, default=0.5)
    parser.add_argument("--detector-goal-conf", type=float, default=0.5)
    parser.add_argument("--detector-person-conf", type=float, default=0.4)
    parser.add_argument("--detector-padding", type=float, default=0.12)
    parser.add_argument("--detector-min-crop-area-ratio", type=float, default=0.15)
    parser.add_argument("--detector-max-crop-area-ratio", type=float, default=0.85)
    parser.add_argument("--detector-max-frame-gap", type=int, default=5)
    parser.add_argument("--detector-window-roi-samples", type=int, default=8)
    parser.add_argument("--detector-use-detection-goals", action="store_true", help="Also merge cls=2 goal boxes from detections.json. Default uses tracked_objects.json only.")
    parser.add_argument("--save-frame-event-logits", action="store_true", help="Save per-frame event logits and temporal localization metrics.")
    parser.add_argument("--frame-event-topk", type=int, default=8)
    parser.add_argument(
        "--fail-on-zero-object-residual",
        action="store_true",
        help="Fail a video when its enabled checkpoint-backed object residual is inactive.",
    )
    parser.add_argument("--max-windows", type=int, default=0, help="Debug option: limit windows per video; 0 means no limit.")
    parser.add_argument("--force", action="store_true", help="Rerun inference even if metrics.json already exists.")
    return parser.parse_args()


def safe_name(text: str) -> str:
    name = Path(text).stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def thresholds_compatible(raw: str, old_summary: dict[str, Any], labels: list[str]) -> bool:
    requested = raw.strip()
    if requested.lower() in {"", "checkpoint"}:
        return True

    old_thresholds = old_summary.get("thresholds")
    if not isinstance(old_thresholds, dict):
        return False

    try:
        if "=" not in requested:
            value = float(requested)
            expected = {label: value for label in labels}
        else:
            expected = {}
            for item in requested.split(","):
                label, value = item.split("=", 1)
                expected[label.strip()] = float(value)
            # A partial override falls back to checkpoint thresholds. The outer
            # evaluator deliberately does not load the model, so rerun rather
            # than risk accepting an incompatible cached result.
            if set(expected) != set(labels):
                return False
    except (TypeError, ValueError):
        return False

    return all(
        label in old_thresholds
        and abs(float(old_thresholds[label]) - value) <= 1e-12
        for label, value in expected.items()
    )


def label_float_map_compatible(
    raw: str, old_values: Any, labels: list[str]
) -> bool:
    if not isinstance(old_values, dict):
        return False
    requested = str(raw).strip()
    try:
        if "=" not in requested:
            value = float(requested)
            expected = {label: value for label in labels}
        else:
            expected = {}
            for item in requested.split(","):
                label, value = item.split("=", 1)
                expected[label.strip()] = float(value)
            if set(expected) != set(labels):
                return False
    except (TypeError, ValueError):
        return False
    return all(
        label in old_values
        and abs(float(old_values[label]) - value) <= 1e-12
        for label, value in expected.items()
    )


def read_video_ids(args: argparse.Namespace) -> list[str]:
    ids: list[str] = []
    if args.video_ids.strip():
        ids.extend([item.strip() for item in args.video_ids.split(",") if item.strip()])
    if args.video_id_file:
        for line in Path(args.video_id_file).read_text().splitlines():
            item = line.strip()
            if item and not item.startswith("#"):
                ids.append(item)
    if not ids:
        proposal_root = Path(args.proposal_root).expanduser()
        if args.mode == "proposal" and proposal_root.exists():
            ids.extend(sorted(p.name for p in proposal_root.iterdir() if p.is_dir() and (p / "shot_proposals.json").exists()))
        else:
            ids.extend(sorted(p.stem for p in Path(args.gt_dir).expanduser().glob("*.json")))
    seen = set()
    unique = []
    for video_id in ids:
        if video_id not in seen:
            unique.append(video_id)
            seen.add(video_id)
    return unique


def parse_video_roots(values: list[str]) -> list[tuple[str, Path]]:
    items = values or DEFAULT_VIDEO_ROOTS
    roots: list[tuple[str, Path]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --video-root '{item}', expected source=/path/to/videos")
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


def run_eval(video_id: str, args: argparse.Namespace, labels: list[str], run_dir: Path, source: str, video_path: Path, gt_path: Path) -> Path:
    out_dir = run_dir / video_id
    metrics_path = out_dir / "metrics.json"
    annotation_path = str(gt_path.resolve())
    annotation_sha256 = file_sha256(gt_path)
    if metrics_path.exists() and not args.force:
        summary_path = out_dir / "summary.json"
        old_summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        old_postprocess = old_summary.get("prediction_postprocess", "interval_merge")
        compatible = (
            str(old_summary.get("checkpoint", "")) == str(args.checkpoint)
            and float(old_summary.get("clip_sec", -1.0)) == float(args.clip_sec)
            and float(old_summary.get("stride_sec", -1.0)) == float(args.stride_sec)
            and old_postprocess == args.prediction_postprocess
            and float(old_summary.get("match_tolerance_sec", -1.0)) == float(args.match_tolerance_sec)
            and thresholds_compatible(args.thresholds, old_summary, labels)
            and has_current_score_semantics(old_summary)
            and (
                not args.fail_on_zero_object_residual
                or old_summary.get("fail_on_zero_object_residual") is True
            )
            and str(old_summary.get("score_source", "clip")) == args.score_source
            and str(old_summary.get("object_motion", {}).get("mode", "checkpoint"))
            == args.object_motion_mode
            and (
                args.score_source != "fusion"
                or (
                    label_float_map_compatible(
                        args.fusion_alpha, old_summary.get("fusion", {}).get("alpha"), labels
                    )
                    and label_float_map_compatible(
                        args.clip_temperature,
                        old_summary.get("fusion", {}).get("clip_temperature"),
                        labels,
                    )
                    and label_float_map_compatible(
                        args.response_temperature,
                        old_summary.get("fusion", {}).get("response_temperature"),
                        labels,
                    )
                )
            )
            and str(old_summary.get("spatial_crop", {}).get("mode", "none")) == args.spatial_crop_mode
            and str(old_summary.get("annotation_path", "")) == annotation_path
            and str(old_summary.get("annotation_sha256", "")) == annotation_sha256
            and (
                not args.save_frame_event_logits
                or (
                    bool(old_summary.get("frame_event", {}).get("enabled", False))
                    and int(old_summary.get("frame_event", {}).get("topk", -1)) == args.frame_event_topk
                    and out_dir.joinpath("frame_event_analysis.json").exists()
                )
            )
            and (
                args.prediction_postprocess != "point_nms"
                or float(old_summary.get("nms_radius_sec", -1.0)) == float(args.nms_radius_sec)
            )
        )
        if not compatible:
            raise RuntimeError(
                f"Existing output {out_dir} was produced with a different evaluation protocol. "
                "Use a new --run-name or pass --force."
            )
        print(f"skip compatible existing {video_id}: {metrics_path}", flush=True)
        return out_dir

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "eval_long_video_checkpoint.py"),
        "--checkpoint", args.checkpoint,
        "--video-id", video_id,
        "--video-path", str(video_path),
        "--annotation-path", str(gt_path),
        "--source", source,
        "--output-dir", str(out_dir),
        "--eval-labels", ",".join(labels),
        "--clip-sec", str(args.clip_sec),
        "--stride-sec", str(args.stride_sec),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--device", args.device,
        "--gpu-ids", args.gpu_ids,
        "--thresholds", args.thresholds,
        "--match-tolerance-sec", str(args.match_tolerance_sec),
        "--score-source", args.score_source,
        "--object-motion-mode", args.object_motion_mode,
        "--fusion-alpha", args.fusion_alpha,
        "--clip-temperature", args.clip_temperature,
        "--response-temperature", args.response_temperature,
        "--prediction-postprocess", args.prediction_postprocess,
        "--nms-radius-sec", str(args.nms_radius_sec),
        "--merge-gap-sec", str(args.merge_gap_sec),
        "--gt-merge-gap-sec", str(args.gt_merge_gap_sec),
        "--spatial-crop-mode", args.spatial_crop_mode,
    ]
    if args.top_crop_ratio is not None:
        cmd.extend(["--top-crop-ratio", str(args.top_crop_ratio)])
    if args.spatial_crop_mode == "adaptive_top_fixed":
        cmd.extend([
            "--adaptive-top-min-ratio", str(args.adaptive_top_min_ratio),
            "--adaptive-top-max-ratio", str(args.adaptive_top_max_ratio),
            "--adaptive-top-fallback-ratio", str(args.adaptive_top_fallback_ratio),
            "--adaptive-top-person-conf", str(args.adaptive_top_person_conf),
        ])
    if args.save_frame_event_logits:
        cmd.extend(["--save-frame-event-logits", "--frame-event-topk", str(args.frame_event_topk)])
    if args.fail_on_zero_object_residual:
        cmd.append("--fail-on-zero-object-residual")
    if args.image_size:
        cmd.extend(["--image-size", args.image_size])
    if args.roi_temporal_mode != "checkpoint":
        cmd.extend(["--roi-temporal-mode", args.roi_temporal_mode])
    temporal_float_args = {
        "roi_dynamic_context_sec": "--roi-dynamic-context-sec",
        "roi_temporal_smoothing_window_sec": "--roi-temporal-smoothing-window-sec",
        "roi_temporal_max_hold_sec": "--roi-temporal-max-hold-sec",
        "roi_temporal_confidence_decay_sec": "--roi-temporal-confidence-decay-sec",
    }
    for arg_name, flag in temporal_float_args.items():
        value = getattr(args, arg_name)
        if value is not None:
            cmd.extend([flag, str(value)])
    if args.spatial_crop_mode in {"adaptive_top_fixed", "legacy_indexed", "robust_detector_aware"} and args.detector_index_root:
        cmd.extend(["--detector-index-root", args.detector_index_root])
    if args.spatial_crop_mode == "detector_aware":
        cmd.extend([
            "--detector-manifest-root", args.detector_manifest_root,
            "--detector-ball-conf", str(args.detector_ball_conf),
            "--detector-goal-conf", str(args.detector_goal_conf),
            "--detector-person-conf", str(args.detector_person_conf),
            "--detector-padding", str(args.detector_padding),
            "--detector-min-crop-area-ratio", str(args.detector_min_crop_area_ratio),
            "--detector-max-crop-area-ratio", str(args.detector_max_crop_area_ratio),
            "--detector-max-frame-gap", str(args.detector_max_frame_gap),
            "--detector-window-roi-samples", str(args.detector_window_roi_samples),
        ])
        if args.detector_use_detection_goals:
            cmd.append("--detector-use-detection-goals")
    if args.max_windows > 0:
        cmd.extend(["--max-windows", str(args.max_windows)])
    if args.mode == "proposal":
        proposal_dir = Path(args.proposal_root) / video_id
        if not (proposal_dir / "shot_proposals.json").exists():
            raise FileNotFoundError(f"Missing shot_proposals.json for {video_id}: {proposal_dir}")
        cmd.extend(["--proposal-dir", str(proposal_dir), "--proposal-dedupe-sec", str(args.proposal_dedupe_sec)])

    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return out_dir


def metric_zero() -> dict[str, Any]:
    return {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "num_pred": 0,
        "num_gt": 0,
        "num_matched_gt": 0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }


def finalize_counts(item: dict[str, Any], *, allow_many_predictions_per_gt: bool = False) -> dict[str, Any]:
    tp = int(item.get("tp", 0))
    fp = int(item.get("fp", 0))
    fn = int(item.get("fn", 0))
    num_pred = int(item.get("num_pred", tp + fp))
    num_gt = int(item.get("num_gt", tp + fn))
    num_matched_gt = int(item.get("num_matched_gt", tp))
    precision = tp / (tp + fp) if tp + fp else 0.0
    if allow_many_predictions_per_gt:
        recall = num_matched_gt / num_gt if num_gt else 0.0
    else:
        recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    item.update({
        "num_pred": num_pred,
        "num_gt": num_gt,
        "num_matched_gt": num_matched_gt,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    })
    return item


def summarize(run_dir: Path, video_ids: list[str], labels: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    totals = {label: metric_zero() for label in labels}
    micro = metric_zero()
    anchor_totals = {label: metric_zero() for label in labels}
    anchor_micro = metric_zero()
    has_anchor = False
    allow_many_predictions_per_gt = False

    for video_id in video_ids:
        metrics_path = run_dir / video_id / "metrics.json"
        if not metrics_path.exists():
            rows.append({"video_id": video_id, "status": "missing"})
            continue
        metrics = json.loads(metrics_path.read_text())
        allow_many_predictions_per_gt = (
            allow_many_predictions_per_gt
            or bool(metrics.get("allow_many_predictions_per_gt", False))
        )
        for label in labels:
            item = metrics["per_class"].get(label, metric_zero())
            num_matched_gt = int(item.get("num_matched_gt", item.get("tp", 0)))
            row = {
                "video_id": video_id,
                "label": label,
                "tp": int(item["tp"]),
                "fp": int(item["fp"]),
                "fn": int(item["fn"]),
                "num_pred": int(item["num_pred"]),
                "num_gt": int(item["num_gt"]),
                "num_matched_gt": num_matched_gt,
                "precision": float(item["precision"]),
                "recall": float(item["recall"]),
                "f1": float(item["f1"]),
                "status": "done",
            }
            rows.append(row)
            for key in ("tp", "fp", "fn", "num_pred", "num_gt", "num_matched_gt"):
                totals[label][key] += row[key]
                micro[key] += row[key]
            anchor_payload = metrics.get("anchor_only")
            if isinstance(anchor_payload, dict):
                anchor_item = anchor_payload.get("per_class", {}).get(label)
                if isinstance(anchor_item, dict):
                    has_anchor = True
                    for key in (
                        "tp", "fp", "fn", "num_pred", "num_gt", "num_matched_gt"
                    ):
                        value = int(
                            anchor_item.get(
                                key,
                                anchor_item.get("tp", 0)
                                if key == "num_matched_gt"
                                else 0,
                            )
                        )
                        anchor_totals[label][key] += value
                        anchor_micro[key] += value

    totals = {
        label: finalize_counts(
            item,
            allow_many_predictions_per_gt=allow_many_predictions_per_gt,
        )
        for label, item in totals.items()
    }
    micro = finalize_counts(micro, allow_many_predictions_per_gt=allow_many_predictions_per_gt)
    summary = {
        "allow_many_predictions_per_gt": allow_many_predictions_per_gt,
        "per_video": rows,
        "per_class": totals,
        "micro": micro,
    }
    if has_anchor:
        summary["anchor_only"] = {
            "per_class": {
                label: finalize_counts(
                    item,
                    allow_many_predictions_per_gt=allow_many_predictions_per_gt,
                )
                for label, item in anchor_totals.items()
            },
            "micro": finalize_counts(
                anchor_micro,
                allow_many_predictions_per_gt=allow_many_predictions_per_gt,
            ),
        }
    (run_dir / "summary_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    with (run_dir / "summary_metrics.csv").open("w", newline="") as f:
        fieldnames = [
            "video_id",
            "label",
            "tp",
            "fp",
            "fn",
            "num_pred",
            "num_gt",
            "num_matched_gt",
            "precision",
            "recall",
            "f1",
            "status",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return summary


def aggregate_frame_event_outputs(
    run_dir: Path,
    video_ids: list[str],
    labels: list[str],
    topk: int,
) -> dict[str, Any]:
    from scripts.eval_long_video_checkpoint import (
        summarize_frame_localization_samples,
        summarize_frame_window_scores,
    )

    samples: list[dict[str, Any]] = []
    window_scores: list[dict[str, Any]] = []
    per_video: dict[str, Any] = {}
    for video_id in video_ids:
        video_dir = run_dir / video_id
        analysis_path = video_dir / "frame_event_analysis.json"
        if analysis_path.exists():
            per_video[video_id] = json.loads(analysis_path.read_text())
        for path, destination in (
            (video_dir / "frame_event_samples.csv", samples),
            (video_dir / "frame_event_window_scores.csv", window_scores),
        ):
            if not path.exists():
                continue
            with path.open(newline="") as f:
                for row in csv.DictReader(f):
                    destination.append({"video_id": video_id, **row})

    if not samples:
        return {}
    analysis = {
        "topk": int(topk),
        "num_videos": len(per_video),
        "localization": summarize_frame_localization_samples(samples, labels),
        "window_discrimination": summarize_frame_window_scores(window_scores, labels),
        "per_video": per_video,
    }
    (run_dir / "frame_event_analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2)
    )
    for path, rows in (
        (run_dir / "frame_event_samples.csv", samples),
        (run_dir / "frame_event_window_scores.csv", window_scores),
    ):
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return analysis



def print_summary(summary: dict[str, Any], labels: list[str]) -> None:
    show_matched_gt = bool(summary.get("allow_many_predictions_per_gt", False))
    print("\nPer-video per-class:")
    for row in summary["per_video"]:
        if row.get("status") != "done":
            print(f"{row['video_id']}: status={row.get('status', 'missing')}", flush=True)
            continue
        matched_text = (
            f" matched_gt={row.get('num_matched_gt', 0)}/{row['num_gt']}"
            if show_matched_gt
            else ""
        )
        print(
            f"{row['video_id']} {row['label']}: "
            f"P={row['precision']:.4f} R={row['recall']:.4f} F1={row['f1']:.4f} "
            f"TP/FP/FN={row['tp']}/{row['fp']}/{row['fn']}{matched_text}",
            flush=True,
        )
    print("\nPer-class aggregate:")
    for label in labels:
        item = summary["per_class"][label]
        matched_text = (
            f" matched_gt={item.get('num_matched_gt', 0)}/{item['num_gt']}"
            if show_matched_gt
            else ""
        )
        print(
            f"{label}: P={item['precision']:.4f} R={item['recall']:.4f} F1={item['f1']:.4f} "
            f"TP/FP/FN={item['tp']}/{item['fp']}/{item['fn']}{matched_text}",
            flush=True,
        )
    micro = summary["micro"]
    matched_text = (
        f" matched_gt={micro.get('num_matched_gt', 0)}/{micro['num_gt']}"
        if show_matched_gt
        else ""
    )
    print(
        f"micro: P={micro['precision']:.4f} R={micro['recall']:.4f} F1={micro['f1']:.4f} "
        f"TP/FP/FN={micro['tp']}/{micro['fp']}/{micro['fn']}{matched_text}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    gt_dir = Path(args.gt_dir).expanduser().resolve()
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"GT directory does not exist: {gt_dir}")
    args.gt_dir = str(gt_dir)
    labels = DENSE_LABELS if args.mode == "dense" else PROPOSAL_LABELS
    video_ids = read_video_ids(args)
    video_roots = parse_video_roots(args.video_root)
    if args.prediction_postprocess == "window_overlap":
        protocol_tag = f"window_overlap_tol{args.match_tolerance_sec:g}"
    elif args.prediction_postprocess == "point_nms":
        protocol_tag = (
            f"point_nms_nms{args.nms_radius_sec:g}_tol{args.match_tolerance_sec:g}"
        )
    else:
        protocol_tag = (
            f"interval_merge_gap{args.merge_gap_sec:g}_tol{args.match_tolerance_sec:g}"
        )
    threshold_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.thresholds).strip("_") or "default"
    run_name = args.run_name or (
        f"{safe_name(args.checkpoint)}_{args.mode}_"
        f"{'shot' if args.mode == 'proposal' else 'all'}_{protocol_tag}_thr{threshold_tag}"
    )
    run_dir = Path(args.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "checkpoint": args.checkpoint,
        "mode": args.mode,
        "labels": labels,
        "video_ids": video_ids,
        "gt_dir": args.gt_dir,
        "proposal_root": args.proposal_root if args.mode == "proposal" else "",
        "clip_sec": args.clip_sec,
        "stride_sec": args.stride_sec,
        "image_size": args.image_size,
        "match_tolerance_sec": args.match_tolerance_sec,
        "prediction_postprocess": args.prediction_postprocess,
        "nms_radius_sec": args.nms_radius_sec,
        "merge_gap_sec": args.merge_gap_sec,
        "gt_merge_gap_sec": args.gt_merge_gap_sec,
        "thresholds": args.thresholds,
        "score_source": args.score_source,
        "object_motion_mode": args.object_motion_mode,
        "score_semantics_version": SCORE_SEMANTICS_VERSION,
        "fail_on_zero_object_residual": args.fail_on_zero_object_residual,
        "fusion_alpha": args.fusion_alpha,
        "clip_temperature": args.clip_temperature,
        "response_temperature": args.response_temperature,
        "max_windows": args.max_windows,
        "frame_event": {"enabled": args.save_frame_event_logits, "topk": args.frame_event_topk},
        "spatial_crop": {
            "mode": args.spatial_crop_mode,
            "top_crop_ratio": args.top_crop_ratio,
            "adaptive_top_min_ratio": args.adaptive_top_min_ratio,
            "adaptive_top_max_ratio": args.adaptive_top_max_ratio,
            "adaptive_top_fallback_ratio": args.adaptive_top_fallback_ratio,
            "adaptive_top_person_conf": args.adaptive_top_person_conf,
            "detector_index_root": args.detector_index_root,
            "roi_temporal_mode": args.roi_temporal_mode,
            "roi_dynamic_context_sec": args.roi_dynamic_context_sec,
            "roi_temporal_smoothing_window_sec": args.roi_temporal_smoothing_window_sec,
            "roi_temporal_max_hold_sec": args.roi_temporal_max_hold_sec,
            "roi_temporal_confidence_decay_sec": args.roi_temporal_confidence_decay_sec,
            "detector_manifest_root": args.detector_manifest_root,
            "detector_ball_conf": args.detector_ball_conf,
            "detector_goal_conf": args.detector_goal_conf,
            "detector_person_conf": args.detector_person_conf,
            "detector_padding": args.detector_padding,
            "detector_min_crop_area_ratio": args.detector_min_crop_area_ratio,
            "detector_max_crop_area_ratio": args.detector_max_crop_area_ratio,
            "detector_max_frame_gap": args.detector_max_frame_gap,
            "detector_window_roi_samples": args.detector_window_roi_samples,
            "detector_use_detection_goals": args.detector_use_detection_goals,
        },
    }
    (run_dir / "run_config.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    for video_id in video_ids:
        gt_path = gt_dir / f"{video_id}.json"
        if not gt_path.exists():
            raise FileNotFoundError(f"Missing GT for {video_id}: {gt_path}")
        source, video_path = find_video(video_id, video_roots)
        run_eval(video_id, args, labels, run_dir, source, video_path, gt_path)

    summary = summarize(run_dir, video_ids, labels)
    if args.save_frame_event_logits:
        aggregate_frame_event_outputs(run_dir, video_ids, labels, args.frame_event_topk)
    print_summary(summary, labels)
    print(f"\nwrote {run_dir / 'summary_metrics.json'}")
    print(f"wrote {run_dir / 'summary_metrics.csv'}")


if __name__ == "__main__":
    main()
