#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTERNAL18 = ROOT / "configs/football/splits/thirdparty18_test_long15_val_no_pn_train/thirdparty18_test_video_ids.txt"
DEFAULT_GT_DIR = Path("/mnt/data_16t/football/football_events_human_repair")
DEFAULT_VIDEO_ROOTS = [
    "xbotgo_0608=/mnt/data_16t/football/raw_video_720P",
    "xbotgo_0608=/mnt/data_16t/football/raw_video_hq_720P",
]
DEFAULT_SCORE_PREFIXES = "prob,clip_prob,response_prob,frame_max_prob,frame_topk_mean_prob"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete long-video football evaluation pipeline for one checkpoint: "
            "dense inference once, then threshold clip/response/fusion/frame scores to find "
            "recall targets and deduplicated human review time."
        )
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--video-id-file", type=Path, default=DEFAULT_EXTERNAL18, help="Videos to evaluate. Used as test split when --cal-video-id-file is set; otherwise also used for calibration.")
    parser.add_argument("--cal-video-id-file", type=Path, default=None, help="Optional calibration split for threshold selection. If omitted, thresholds are selected on --video-id-file and marked non-heldout.")
    parser.add_argument("--gt-dir", type=Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--video-root", action="append", default=[], help="source=/path video root. Can be repeated.")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/football_long_video_full_pipeline")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Exact reusable run directory. When set, cached per-video dense outputs are reused unless --force.",
    )
    parser.add_argument("--run-name", default="")
    parser.add_argument("--clip-sec", type=float, default=10.0)
    parser.add_argument("--stride-sec", type=float, default=5.0)
    parser.add_argument("--match-tolerance-sec", type=float, default=5.0)
    parser.add_argument("--prediction-postprocess", default="window_overlap", choices=["window_overlap", "point_nms", "interval_merge"])
    parser.add_argument("--review-mode", default="cap10_peak", choices=["window", "window_tol5", "cap10_peak", "cap15_peak", "cap20_peak"])
    parser.add_argument("--manual-overhead-sec", type=float, default=0.0)
    parser.add_argument("--score-source", default="fusion", choices=["clip", "response", "fusion"], help="Score source used for the primary prob_* columns during inference.")
    parser.add_argument("--fusion-alpha", default="shot=0.4,save=0.4,set_piece=0.8")
    parser.add_argument("--clip-temperature", default="1.0")
    parser.add_argument("--response-temperature", default="1.0")
    parser.add_argument("--score-prefixes", default=DEFAULT_SCORE_PREFIXES, help="Comma-separated window score prefixes to threshold offline.")
    parser.add_argument("--recall-targets", default="0.85,0.88,0.90,0.92,0.95")
    parser.add_argument(
        "--primary-recall-target", type=float, default=0.85,
        help="Calibration recall target used for the concise primary operating-point report.",
    )
    parser.add_argument(
        "--recall-constraint", choices=("per_class", "micro"), default="per_class",
        help="Recall-floor semantics. Default requires every event class to meet the target.",
    )
    parser.add_argument("--budgets", default="10,15,20,25,30,40,50,60,70")
    parser.add_argument("--grid-size", type=int, default=31)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-groups", default="", help="Optional semicolon-separated physical GPU groups, e.g. '2,3;4,5;6,7'. Empty runs one process with current CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--labels", default="shot,save,set_piece")
    parser.add_argument("--save-frame-event-logits", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--frame-event-topk", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "run"


def read_video_ids(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def write_shards(video_ids: Sequence[str], shard_dir: Path, prefix: str, count: int) -> list[Path]:
    shard_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for idx in range(count):
        path = shard_dir / f"{prefix}_shard{idx}.txt"
        path.write_text("\n".join(video_ids[idx::count]) + ("\n" if video_ids[idx::count] else ""), encoding="utf-8")
        paths.append(path)
    return paths


def run_logged(cmd: Sequence[str], log_path: Path, env: dict[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("command=" + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env)
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed rc={rc}; see {log_path}")


def run_dense_eval(
    *,
    args: argparse.Namespace,
    run_name: str,
    video_id_file: Path,
    split_tag: str,
    output_root: Path,
    log_dir: Path,
) -> Path:
    run_dir = output_root / run_name
    video_roots = args.video_root or DEFAULT_VIDEO_ROOTS
    base_cmd = [
        sys.executable,
        "scripts/evaluate_football_model.py",
        "--checkpoint", str(args.checkpoint),
        "--mode", "dense",
        "--gt-dir", str(args.gt_dir),
        "--output-root", str(output_root),
        "--run-name", run_name,
        "--clip-sec", str(args.clip_sec),
        "--stride-sec", str(args.stride_sec),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--device", args.device,
        "--thresholds", "0.5",
        "--match-tolerance-sec", str(args.match_tolerance_sec),
        "--score-source", args.score_source,
        "--fusion-alpha", args.fusion_alpha,
        "--clip-temperature", args.clip_temperature,
        "--response-temperature", args.response_temperature,
        "--prediction-postprocess", args.prediction_postprocess,
    ]
    for root in video_roots:
        base_cmd.extend(["--video-root", root])
    if args.save_frame_event_logits:
        base_cmd.extend(["--save-frame-event-logits", "--frame-event-topk", str(args.frame_event_topk)])
    if args.force:
        base_cmd.append("--force")

    groups = [g.strip() for g in args.gpu_groups.split(";") if g.strip()]
    if not groups:
        cmd = [*base_cmd, "--video-id-file", str(video_id_file), "--gpu-ids", "0"]
        run_logged(cmd, log_dir / f"{split_tag}_dense.log")
        return run_dir

    ids = read_video_ids(video_id_file)
    shard_paths = write_shards(ids, output_root / f"{run_name}_shards", split_tag, len(groups))
    procs: list[tuple[subprocess.Popen[Any], Path]] = []
    shard_run_dirs: list[Path] = []
    for idx, (group, shard) in enumerate(zip(groups, shard_paths)):
        logical_ids = ",".join(str(i) for i, _ in enumerate(group.split(",")))
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = group
        env.setdefault("PYTHONUNBUFFERED", "1")
        shard_run_name = f"{run_name}_shard{idx}"
        shard_run_dirs.append(output_root / shard_run_name)
        cmd = [
            *base_cmd,
            "--run-name", shard_run_name,
            "--video-id-file", str(shard),
            "--gpu-ids", logical_ids,
        ]
        log_path = log_dir / f"{split_tag}_dense_shard{idx}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("w", encoding="utf-8")
        log.write("command=" + " ".join(cmd) + "\n")
        log.write(f"CUDA_VISIBLE_DEVICES={group}\n")
        log.flush()
        procs.append((subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env), log_path))
    failures = []
    for proc, log_path in procs:
        rc = proc.wait()
        if rc != 0:
            failures.append((rc, log_path))
    if failures:
        detail = ", ".join(f"rc={rc} log={path}" for rc, path in failures)
        raise RuntimeError(f"Dense eval shard failed: {detail}")

    # Materialize a single aggregate run directory from independently written shard
    # directories. This avoids concurrent writes to summary_metrics.* and makes
    # downstream threshold selection consume the usual one-run layout.
    run_dir.mkdir(parents=True, exist_ok=True)
    for idx, video_id in enumerate(ids):
        shard_dir = shard_run_dirs[idx % len(shard_run_dirs)]
        src = shard_dir / video_id
        dst = run_dir / video_id
        if not src.is_dir():
            raise FileNotFoundError(f"Missing shard output for {video_id}: {src}")
        if dst.exists():
            if args.force:
                if dst.is_symlink() or dst.is_file():
                    dst.unlink()
                else:
                    shutil.rmtree(dst)
            else:
                continue
        try:
            dst.symlink_to(src, target_is_directory=True)
        except OSError:
            shutil.copytree(src, dst)

    # Re-run the wrapper without --force to aggregate all cached per-video outputs
    # into summary files in run_dir.
    cmd = [*base_cmd, "--video-id-file", str(video_id_file), "--gpu-ids", "0"]
    if "--force" in cmd:
        cmd = [part for part in cmd if part != "--force"]
    run_logged(cmd, log_dir / f"{split_tag}_aggregate_cached.log")
    return run_dir


def run_selector(
    *,
    args: argparse.Namespace,
    score_prefix: str,
    cal_run_dir: Path,
    cal_video_id_file: Path,
    test_run_dir: Path | None,
    test_video_id_file: Path | None,
    report_dir: Path,
    log_dir: Path,
) -> Path:
    output = report_dir / f"threshold_report_{score_prefix}.json"
    cmd = [
        sys.executable,
        "scripts/select_window_overlap_thresholds_by_budget.py",
        "--cal-run-dir", str(cal_run_dir),
        "--cal-video-id-file", str(cal_video_id_file),
        "--labels", args.labels,
        "--budgets", args.budgets,
        "--recall-targets", args.recall_targets,
        "--recall-constraint", args.recall_constraint,
        "--grid-size", str(args.grid_size),
        "--match-tolerance-sec", str(args.match_tolerance_sec),
        "--review-mode", args.review_mode,
        "--manual-overhead-sec", str(args.manual_overhead_sec),
        "--score-prefix", score_prefix,
        "--output", str(output),
    ]
    if test_run_dir is not None and test_video_id_file is not None:
        cmd.extend(["--test-run-dir", str(test_run_dir), "--test-video-id-file", str(test_video_id_file)])
    run_logged(cmd, log_dir / f"select_{score_prefix}.log")
    return output


def flatten_result(score_prefix: str, section: str, key: str, item: dict[str, Any]) -> dict[str, Any]:
    row = {
        "score_prefix": score_prefix,
        "section": section,
        "key": key,
        "precision": item.get("micro", {}).get("precision"),
        "recall": item.get("micro", {}).get("recall"),
        "f1": item.get("micro", {}).get("f1"),
        "tp": item.get("micro", {}).get("tp"),
        "fp": item.get("micro", {}).get("fp"),
        "fn": item.get("micro", {}).get("fn"),
        "num_candidates": item.get("num_candidates"),
        "candidate_view_minutes": item.get("candidate_view_minutes"),
        "raw_candidate_minutes_unmerged": item.get("raw_candidate_minutes_unmerged"),
        "human_minutes_with_overhead": item.get("human_minutes_with_overhead"),
        "participation_pct": item.get("participation_pct"),
        "candidates_per_hour": item.get("candidates_per_hour"),
        "total_video_minutes": item.get("total_video_minutes"),
        "threshold_string": item.get("threshold_string"),
    }
    for label, metrics in item.get("per_class", {}).items():
        row[f"{label}_precision"] = metrics.get("precision")
        row[f"{label}_recall"] = metrics.get("recall")
        row[f"{label}_fp"] = metrics.get("fp")
        row[f"{label}_fn"] = metrics.get("fn")
    return row


def write_summary(report_paths: Sequence[Path], output_json: Path, output_csv: Path) -> None:
    rows: list[dict[str, Any]] = []
    payload: dict[str, Any] = {"reports": [str(path) for path in report_paths], "rows": rows}
    for path in report_paths:
        score_prefix = path.stem.removeprefix("threshold_report_")
        data = json.loads(path.read_text(encoding="utf-8"))
        cal = data.get("calibration", {})
        if cal.get("checkpoint_thresholds"):
            rows.append(flatten_result(score_prefix, "cal_checkpoint", "checkpoint", cal["checkpoint_thresholds"]))
        for section_name, section_key in (("cal_min_cost_by_recall", "min_cost_by_recall"), ("cal_best_by_budget", "best_by_budget")):
            for key, item in cal.get(section_key, {}).items():
                rows.append(flatten_result(score_prefix, section_name, key, item))
        for key, item in data.get("test", {}).get("applied", {}).items():
            rows.append(flatten_result(score_prefix, "test_applied", key, item))
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if rows:
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)



def write_primary_operating_point_summary(
    report_paths: Sequence[Path],
    *,
    recall_target: float,
    calibration_is_test: bool,
    output_json: Path,
    output_csv: Path,
) -> dict[str, Any]:
    """Write a concise, no-per-video report for the requested operating point.

    Thresholds are always selected from calibration metrics. The recommended score
    source is therefore also selected without consulting test performance.
    Human participation is the union duration of review intervals within each video,
    so overlapping classes/windows are reviewed only once.
    """
    target_key = f"{float(recall_target):g}"
    applied_key = f"cal_min_cost_at_recall_{target_key}"
    rows: list[dict[str, Any]] = []
    for path in report_paths:
        score_prefix = path.stem.removeprefix("threshold_report_")
        data = json.loads(path.read_text(encoding="utf-8"))
        cal_item = data.get("calibration", {}).get("min_cost_by_recall", {}).get(target_key)
        test_item = data.get("test", {}).get("applied", {}).get(applied_key)
        if not isinstance(cal_item, dict) or not isinstance(test_item, dict):
            continue
        row: dict[str, Any] = {
            "score_prefix": score_prefix,
            "recall_target": float(recall_target),
            "threshold_string": cal_item.get("threshold_string"),
            "cal_precision": cal_item.get("micro", {}).get("precision"),
            "cal_recall": cal_item.get("micro", {}).get("recall"),
            "cal_dedup_review_minutes": cal_item.get("candidate_view_minutes"),
            "cal_participation_pct": cal_item.get("participation_pct"),
            "test_precision": test_item.get("micro", {}).get("precision"),
            "test_recall": test_item.get("micro", {}).get("recall"),
            "test_f1": test_item.get("micro", {}).get("f1"),
            "test_tp": test_item.get("micro", {}).get("tp"),
            "test_fp": test_item.get("micro", {}).get("fp"),
            "test_fn": test_item.get("micro", {}).get("fn"),
            "test_total_video_minutes": test_item.get("total_video_minutes"),
            "test_dedup_review_minutes": test_item.get("candidate_view_minutes"),
            "test_participation_pct": test_item.get("participation_pct"),
            "test_raw_unmerged_minutes": test_item.get("raw_candidate_minutes_unmerged"),
            "test_num_label_candidates": test_item.get("num_candidates"),
            "review_mode": test_item.get("review_mode"),
        }
        for label, metrics in test_item.get("per_class", {}).items():
            row[f"{label}_precision"] = metrics.get("precision")
            row[f"{label}_recall"] = metrics.get("recall")
            row[f"{label}_f1"] = metrics.get("f1")
            row[f"{label}_tp"] = metrics.get("tp")
            row[f"{label}_fp"] = metrics.get("fp")
            row[f"{label}_fn"] = metrics.get("fn")
        rows.append(row)

    if not rows:
        raise RuntimeError(
            f"No score source produced operating point recall={target_key}; "
            "ensure it is included in --recall-targets"
        )
    recommended = min(
        rows,
        key=lambda row: (
            float(row.get("cal_participation_pct") or float("inf")),
            -float(row.get("cal_precision") or 0.0),
            -float(row.get("cal_recall") or 0.0),
            str(row.get("score_prefix") or ""),
        ),
    )
    payload = {
        "protocol": "football_long_video_primary_operating_point_v1",
        "recall_target": float(recall_target),
        "calibration_is_test": bool(calibration_is_test),
        "recall_constraint": "read from threshold report; interface default is per_class",
        "selection_rule": (
            "For each score source, minimize calibration deduplicated participation "
            "subject to recall target; choose score source by calibration participation, "
            "then calibration precision. Test metrics never select the operating point."
        ),
        "participation_definition": (
            "Per-video union duration of UI review intervals across all labels and "
            "overlapping windows; duplicate footage is counted once."
        ),
        "recommended_score_prefix": recommended["score_prefix"],
        "recommended": recommended,
        "score_source_rows": rows,
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return payload


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    args.checkpoint = checkpoint
    args.gt_dir = args.gt_dir.expanduser().resolve()
    if not args.gt_dir.is_dir():
        raise FileNotFoundError(args.gt_dir)
    test_file = args.video_id_file.expanduser().resolve()
    cal_file = args.cal_video_id_file.expanduser().resolve() if args.cal_video_id_file else test_file
    if not test_file.is_file():
        raise FileNotFoundError(test_file)
    if not cal_file.is_file():
        raise FileNotFoundError(cal_file)
    calibration_is_test = cal_file == test_file

    checkpoint_stat = checkpoint.stat()
    checkpoint_fingerprint = f"{checkpoint_stat.st_size}_{checkpoint_stat.st_mtime_ns}"
    base_name = args.run_name or safe_name(f"{checkpoint.parent.name}_{checkpoint.stem}")
    dense_base_name = safe_name(f"{base_name}_{checkpoint_fingerprint}")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else args.output_root.expanduser().resolve() / f"{base_name}_{stamp}"
    )
    eval_root = output_root / "dense_runs"
    report_dir = output_root / "reports"
    log_dir = output_root / "logs"
    report_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    manifest = vars(args).copy()
    manifest.update({
        "checkpoint": str(checkpoint),
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "cal_video_id_file": str(cal_file),
        "test_video_id_file": str(test_file),
        "calibration_is_test": calibration_is_test,
        "output_root": str(output_root),
    })
    (output_root / "run_config.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    if calibration_is_test:
        run_name = f"{dense_base_name}_all18_{args.score_source}"
        cal_run_dir = run_dense_eval(args=args, run_name=run_name, video_id_file=test_file, split_tag="all18", output_root=eval_root, log_dir=log_dir)
        test_run_dir = cal_run_dir
    else:
        cal_run_dir = run_dense_eval(args=args, run_name=f"{dense_base_name}_cal_{args.score_source}", video_id_file=cal_file, split_tag="cal", output_root=eval_root, log_dir=log_dir)
        test_run_dir = run_dense_eval(args=args, run_name=f"{dense_base_name}_test_{args.score_source}", video_id_file=test_file, split_tag="test", output_root=eval_root, log_dir=log_dir)

    report_paths: list[Path] = []
    for prefix in [item.strip() for item in args.score_prefixes.split(",") if item.strip()]:
        try:
            report_paths.append(
                run_selector(
                    args=args,
                    score_prefix=prefix,
                    cal_run_dir=cal_run_dir,
                    cal_video_id_file=cal_file,
                    test_run_dir=test_run_dir,
                    test_video_id_file=test_file,
                    report_dir=report_dir,
                    log_dir=log_dir,
                )
            )
        except Exception as exc:
            error_path = report_dir / f"threshold_report_{prefix}.error.txt"
            error_path.write_text(str(exc) + "\n", encoding="utf-8")
            print(f"WARN score_prefix={prefix} failed: {exc}", flush=True)

    write_summary(report_paths, output_root / "full_pipeline_summary.json", output_root / "full_pipeline_summary.csv")
    primary = write_primary_operating_point_summary(
        report_paths,
        recall_target=args.primary_recall_target,
        calibration_is_test=calibration_is_test,
        output_json=output_root / "primary_operating_point.json",
        output_csv=output_root / "primary_operating_point.csv",
    )
    print(json.dumps({
        "output_root": str(output_root),
        "summary_json": str(output_root / "full_pipeline_summary.json"),
        "summary_csv": str(output_root / "full_pipeline_summary.csv"),
        "primary_json": str(output_root / "primary_operating_point.json"),
        "primary_csv": str(output_root / "primary_operating_point.csv"),
        "recommended": primary["recommended"],
        "reports": [str(path) for path in report_paths],
        "calibration_is_test": calibration_is_test,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
