"""Run the isolated clean G0 official-vs-supervised-backbone cache ablation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
for path in (PROJECT_ROOT / "src", WORKSPACE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from football_longform_v2.config import load_config  # noqa: E402
from football_longform_v2.schema import read_video_ids  # noqa: E402


CLEAN_G0_IDS = (
    "2044314061738291201",
    "2044314322389118977",
    "2044315055763173378",
    "2044315145319952385",
)
SMOKE_ONLY_ID = "2044306371297357826"


def _scalar(value: np.ndarray) -> str | float:
    item = value.item()
    return item.decode() if isinstance(item, bytes) else item


def summarize(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        context = payload["context"].astype(np.float32)
        motion = payload["motion"].astype(np.float32)
        context_valid = payload["context_valid"].astype(bool)
        motion_valid = payload["motion_valid"].astype(bool)
        return {
            "path": str(path),
            "video_id": path.parent.name,
            "backbone_arch": _scalar(payload["backbone_arch"]),
            "backbone_id": _scalar(payload["backbone_id"]),
            "backbone_weights": _scalar(payload["backbone_weights"]),
            "source_video": _scalar(payload["source_video"]),
            "max_seconds": float(_scalar(payload["max_seconds"])),
            "timeline_shape": list(payload["timestamps"].shape),
            "context_shape": list(context.shape),
            "motion_shape": list(motion.shape),
            "context_valid_ratio": float(context_valid.mean()),
            "motion_valid_ratio": float(motion_valid.mean()),
            "context_finite": bool(np.isfinite(context).all()),
            "motion_finite": bool(np.isfinite(motion).all()),
            "context_mean": float(context.mean()),
            "context_std": float(context.std()),
            "context_l2_mean": float(np.linalg.norm(context, axis=1).mean()),
        }


def compare(official_path: Path, supervised_path: Path) -> dict[str, Any]:
    with np.load(official_path, allow_pickle=False) as official, np.load(supervised_path, allow_pickle=False) as supervised:
        left = official["context"].astype(np.float32)
        right = supervised["context"].astype(np.float32)
        if left.shape != right.shape:
            raise ValueError(f"feature shapes differ: {left.shape} vs {right.shape}")
        dot = np.sum(left * right, axis=1)
        norm = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        cosine = dot / np.clip(norm, 1e-12, None)
        max_abs = float(np.abs(left - right).max())
        identical = bool(np.array_equal(left, right))
        if identical or max_abs == 0.0:
            raise RuntimeError("official and supervised embeddings are exactly identical")
        return {
            "official": str(official_path),
            "supervised": str(supervised_path),
            "cosine_mean": float(cosine.mean()),
            "cosine_min": float(cosine.min()),
            "cosine_max": float(cosine.max()),
            "mean_abs_difference": float(np.abs(left - right).mean()),
            "max_abs_difference": max_abs,
            "identical": identical,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", required=True)
    parser.add_argument("--video-root", default="/mnt/data_16t/football/raw_video_720P")
    parser.add_argument("--max-seconds", type=float, default=120.0)
    parser.add_argument("--context-batch-size", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config_paths = {
        "official": PROJECT_ROOT / "configs/g0_official.yaml",
        "supervised_ema": PROJECT_ROOT / "configs/g0_supervised_ema.yaml",
    }
    calibration_ids = set(read_video_ids((PROJECT_ROOT / ".." / "configs/football/splits/thirdparty18_holdout_all_rest_train/calibration_20_video_ids.txt").resolve()))
    if not set(CLEAN_G0_IDS).issubset(calibration_ids):
        raise RuntimeError("clean G0 ids are not all in the declared calibration split")
    reports: dict[str, Any] = {"scope": {"clean_g0_ids": CLEAN_G0_IDS, "smoke_only_excluded": SMOKE_ONLY_ID, "max_seconds": args.max_seconds, "device": args.device}, "runs": {}, "comparisons": {}}
    builder = PROJECT_ROOT / "scripts/build_rgb_timeline.py"
    for variant, config_path in config_paths.items():
        config = load_config(config_path)
        feature_store = Path(config["paths"]["feature_store"])
        feature_store = feature_store if feature_store.is_absolute() else (PROJECT_ROOT / feature_store).resolve()
        variant_rows = []
        for video_id in CLEAN_G0_IDS:
            output = feature_store / video_id / "timeline.npz"
            command = [sys.executable, str(builder), "--config", str(config_path), "--video-id", video_id, "--video-root", args.video_root, "--device", args.device, "--max-seconds", str(args.max_seconds), "--context-batch-size", str(args.context_batch_size), "--output", str(output)]
            if args.overwrite:
                command.append("--overwrite")
            started = time.monotonic()
            subprocess.run(command, check=True)
            row = summarize(output)
            row["elapsed_seconds"] = round(time.monotonic() - started, 3)
            if not row["context_finite"] or not row["motion_finite"] or row["context_valid_ratio"] < 0.99 or row["motion_valid_ratio"] < 0.99:
                raise RuntimeError(f"cache quality gate failed for {variant}/{video_id}: {row}")
            variant_rows.append(row)
        reports["runs"][variant] = variant_rows
    official_cfg, supervised_cfg = (load_config(config_paths[key]) for key in ("official", "supervised_ema"))
    official_store = (PROJECT_ROOT / official_cfg["paths"]["feature_store"]).resolve()
    supervised_store = (PROJECT_ROOT / supervised_cfg["paths"]["feature_store"]).resolve()
    for video_id in CLEAN_G0_IDS:
        reports["comparisons"][video_id] = compare(official_store / video_id / "timeline.npz", supervised_store / video_id / "timeline.npz")
    report_path = PROJECT_ROOT / "experiments/g0_backbone_ablation/g0_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(reports, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
