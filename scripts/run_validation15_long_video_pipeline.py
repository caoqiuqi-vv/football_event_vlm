#!/usr/bin/env python
"""Run the cached 15-long-video validation chain for one football checkpoint."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_long_video_full_pipeline_eval import run_dense_eval, safe_name

DEFAULT_VAL15 = ROOT / "configs/football/splits/thirdparty18_test_long15_val_no_pn_train/internal_val_video_ids.txt"
DEFAULT_GT = Path("/mnt/data_16t/football/football_events_human_repair")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dense-infer the fixed 15-video validation split once, cache every video's "
            "window/frame outputs, then maximize precision subject to recall >= 80%."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--video-id-file", type=Path, default=DEFAULT_VAL15)
    parser.add_argument("--gt-dir", type=Path, default=DEFAULT_GT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/football_long_video_validation15",
    )
    parser.add_argument("--run-name", default="")
    parser.add_argument("--gpu-groups", default="4;5;6;7")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--clip-sec", type=float, default=10.0)
    parser.add_argument("--stride-sec", type=float, default=5.0)
    parser.add_argument("--match-tolerance-sec", type=float, default=5.0)
    parser.add_argument("--recall-floor", type=float, default=0.80)
    parser.add_argument("--grid-size", type=int, default=31)
    parser.add_argument("--max-review-segment-sec", type=float, default=30.0)
    parser.add_argument("--ui-cap-sec", type=float, default=10.0)
    parser.add_argument("--score-source", default="fusion", choices=("clip", "response", "fusion"))
    parser.add_argument("--fusion-alpha", default="shot=0.4,save=0.4,set_piece=0.8")
    parser.add_argument("--clip-temperature", default="1.0")
    parser.add_argument("--response-temperature", default="1.0")
    parser.add_argument(
        "--score-prefixes",
        default="prob,clip_prob,response_prob,frame_max_prob,frame_topk_mean_prob",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    video_file = args.video_id_file.expanduser().resolve()
    gt_dir = args.gt_dir.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not video_file.is_file():
        raise FileNotFoundError(video_file)
    if not gt_dir.is_dir():
        raise FileNotFoundError(gt_dir)

    stat = checkpoint.stat()
    fingerprint = f"{stat.st_size}_{stat.st_mtime_ns}"
    base_name = args.run_name or safe_name(f"{checkpoint.parent.name}_{checkpoint.stem}_{fingerprint}")
    output_dir = args.output_root.expanduser().resolve() / base_name
    dense_root = output_dir / "dense_runs"
    log_dir = output_dir / "logs"
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    dense_args = SimpleNamespace(
        checkpoint=checkpoint,
        gt_dir=gt_dir,
        video_root=[],
        clip_sec=args.clip_sec,
        stride_sec=args.stride_sec,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device="cuda:0",
        match_tolerance_sec=args.match_tolerance_sec,
        score_source=args.score_source,
        fusion_alpha=args.fusion_alpha,
        clip_temperature=args.clip_temperature,
        response_temperature=args.response_temperature,
        prediction_postprocess="window_overlap",
        save_frame_event_logits=True,
        frame_event_topk=8,
        force=args.force,
        gpu_groups=args.gpu_groups,
    )
    dense_run_name = f"{base_name}_val15_{args.score_source}"
    dense_run_dir = run_dense_eval(
        args=dense_args,
        run_name=dense_run_name,
        video_id_file=video_file,
        split_tag="val15",
        output_root=dense_root,
        log_dir=log_dir,
    )

    report_path = report_dir / "precision_at_recall80_ui_workload.json"
    cmd = [
        sys.executable,
        "scripts/select_validation_thresholds_cached.py",
        "--run-dir", str(dense_run_dir),
        "--video-id-file", str(video_file),
        "--score-prefixes", args.score_prefixes,
        "--recall-floor", str(args.recall_floor),
        "--grid-size", str(args.grid_size),
        "--match-tolerance-sec", str(args.match_tolerance_sec),
        "--max-review-segment-sec", str(args.max_review_segment_sec),
        "--ui-cap-sec", str(args.ui_cap_sec),
        "--output", str(report_path),
    ]
    rc = subprocess.call(cmd, cwd=str(ROOT))
    if rc != 0:
        raise RuntimeError(f"threshold selection failed rc={rc}")

    manifest = {
        "protocol": "validation15_cached_full_pipeline_v1",
        "checkpoint": str(checkpoint),
        "checkpoint_fingerprint": fingerprint,
        "video_id_file": str(video_file),
        "dense_run_dir": str(dense_run_dir),
        "report": str(report_path),
        "cache_policy": "checkpoint size+mtime fingerprint; cached per-video dense outputs are reused unless --force",
        "selection": "maximize strict 1:1 window precision subject to micro recall floor; report deduplicated segment workload separately",
        "gpu_groups": args.gpu_groups,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
