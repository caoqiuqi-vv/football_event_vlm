#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

DEFAULT_LABELS = ("shot", "save", "set_piece")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def read_video_ids(paths: Sequence[str]) -> set[str]:
    ids: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(path)
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                ids.add(line)
    return ids


def parse_label_scores(raw: str, labels: Sequence[str], *, default: float) -> dict[str, float]:
    values = {label: float(default) for label in labels}
    if not raw.strip():
        return values
    for item in raw.split(","):
        if not item.strip():
            continue
        label, value = item.split("=", 1)
        label = label.strip()
        if label not in values:
            raise ValueError(f"Unknown label '{label}', expected one of {list(labels)}")
        values[label] = float(value)
    return values


def load_gt_times(path: Path) -> dict[str, list[float]]:
    by_label = {label: [] for label in DEFAULT_LABELS}
    for row in read_csv(path):
        label = row.get("label", "").strip()
        if label in by_label:
            by_label[label].append(float(row["time_sec"]))
    for values in by_label.values():
        values.sort()
    return by_label


def near_gt(
    gt_by_label: dict[str, list[float]],
    *,
    label: str,
    center_sec: float,
    safety_margin_sec: float,
    mode: str,
) -> bool:
    if safety_margin_sec <= 0:
        return False
    if mode == "same_label":
        labels = [label]
    elif mode == "any":
        labels = list(gt_by_label)
    else:
        raise ValueError("safety_label_mode must be same_label or any")
    for item_label in labels:
        for time_sec in gt_by_label.get(item_label, []):
            if abs(float(time_sec) - center_sec) <= safety_margin_sec:
                return True
    return False


def matched_prediction_keys(matches_path: Path) -> set[tuple[str, int, int]]:
    if not matches_path.exists():
        return set()
    keys: set[tuple[str, int, int]] = set()
    for row in read_csv(matches_path):
        label = row.get("label", "").strip()
        if not label:
            continue
        time_sec = float(row.get("pred_time_sec", row.get("time_sec", 0.0)))
        score = float(row.get("pred_score", row.get("score", 0.0)))
        keys.add((label, round(time_sec * 1000), round(score * 1_000_000)))
    return keys


def prediction_key(row: dict[str, str]) -> tuple[str, int, int]:
    return (
        row["label"].strip(),
        round(float(row["time_sec"]) * 1000),
        round(float(row["score"]) * 1_000_000),
    )


def dedupe_candidates(
    candidates: Sequence[dict[str, Any]],
    *,
    gap_sec: float,
    limit: int,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (-float(item["score"]), float(item["center_sec"])),
    ):
        if gap_sec > 0 and any(
            abs(float(candidate["center_sec"]) - float(old["center_sec"])) <= gap_sec
            and str(candidate["label"]) == str(old["label"])
            for old in kept
        ):
            continue
        kept.append(dict(candidate))
        if limit > 0 and len(kept) >= limit:
            break
    return kept


def merge_label_candidates(candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, float], dict[str, Any]] = {}
    for candidate in candidates:
        key = (str(candidate["video_id"]), round(float(candidate["center_sec"]), 3))
        label = str(candidate["label"])
        if key not in merged:
            item = dict(candidate)
            item.pop("label", None)
            item["labels"] = [label]
            item["label_scores"] = {label: float(candidate["score"])}
            merged[key] = item
            continue
        item = merged[key]
        if label not in item["labels"]:
            item["labels"].append(label)
        item["label_scores"][label] = float(candidate["score"])
        item["score"] = max(float(item["score"]), float(candidate["score"]))
    result = list(merged.values())
    for item in result:
        item["labels"].sort()
    return sorted(result, key=lambda item: (-float(item["score"]), float(item["center_sec"])))


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.eval_run_dir)
    labels = [item.strip() for item in args.labels.split(",") if item.strip()]
    min_scores = parse_label_scores(args.min_scores, labels, default=args.min_score)
    max_scores = parse_label_scores(args.max_scores, labels, default=args.max_score)
    reviewed_ids = read_video_ids(args.reviewed_video_ids)
    hard_negatives: list[dict[str, Any]] = []
    per_video_counts: dict[str, int] = {}
    per_label_counts = {label: 0 for label in labels}
    skipped_not_reviewed = 0
    skipped_matched = 0
    skipped_near_gt = 0
    skipped_score = 0

    video_dirs = sorted(
        path
        for path in run_dir.iterdir()
        if path.is_dir()
        and (path / "predicted_events.csv").exists()
        and (path / "gt_events.csv").exists()
    )
    for video_dir in video_dirs:
        video_id = video_dir.name
        if video_id not in reviewed_ids:
            skipped_not_reviewed += 1
            continue
        matched_keys = matched_prediction_keys(video_dir / "matches.csv")
        gt_by_label = load_gt_times(video_dir / "gt_events.csv")
        selected_for_video: list[dict[str, Any]] = []
        by_label: dict[str, list[dict[str, Any]]] = {label: [] for label in labels}
        for row in read_csv(video_dir / "predicted_events.csv"):
            label = row.get("label", "").strip()
            if label not in by_label:
                continue
            score = float(row["score"])
            if score < min_scores[label] or score > max_scores[label]:
                skipped_score += 1
                continue
            if prediction_key(row) in matched_keys:
                skipped_matched += 1
                continue
            center_sec = float(row["time_sec"])
            if near_gt(
                gt_by_label,
                label=label,
                center_sec=center_sec,
                safety_margin_sec=float(args.safety_margin_sec),
                mode=args.safety_label_mode,
            ):
                skipped_near_gt += 1
                continue
            by_label[label].append(
                {
                    "source": args.source,
                    "video_id": video_id,
                    "label": label,
                    "score": score,
                    "start_sec": center_sec,
                    "end_sec": center_sec,
                    "center_sec": center_sec,
                    "support_start_sec": float(row.get("support_start_sec", center_sec) or center_sec),
                    "support_end_sec": float(row.get("support_end_sec", center_sec) or center_sec),
                    "num_windows": int(float(row.get("num_windows", 1) or 1)),
                    "window_indices": row.get("window_indices", ""),
                    "mining_run": run_dir.name,
                    "mining_source": "point_nms_predicted_events",
                    "score_branch": "fused",
                    "mined_from_reviewed_video": True,
                }
            )
        for label in labels:
            selected = dedupe_candidates(
                by_label[label],
                gap_sec=float(args.dedupe_gap_sec),
                limit=int(args.max_per_video_per_label),
            )
            selected_for_video.extend(selected)
            per_label_counts[label] += len(selected)
        selected_for_video = merge_label_candidates(selected_for_video)
        if selected_for_video:
            hard_negatives.extend(selected_for_video)
            per_video_counts[video_id] = len(selected_for_video)

    run_config_path = run_dir / "run_config.json"
    run_config = json.loads(run_config_path.read_text()) if run_config_path.exists() else {}
    return {
        "schema": "football_pointnms_hard_negative_manifest_v1",
        "policy": (
            "Mine false positive events from final predicted_events.csv after PointNMS; "
            "drop matched predictions and predictions near reviewed GT; keep highest-score FPs."
        ),
        "eval_run_dir": str(run_dir),
        "mining_checkpoint": str(run_config.get("checkpoint", "")),
        "source": args.source,
        "labels": labels,
        "min_scores": min_scores,
        "max_scores": max_scores,
        "safety_margin_sec": float(args.safety_margin_sec),
        "safety_label_mode": args.safety_label_mode,
        "dedupe_gap_sec": float(args.dedupe_gap_sec),
        "max_per_video_per_label": int(args.max_per_video_per_label),
        "reviewed_video_id_files": [str(Path(path)) for path in args.reviewed_video_ids],
        "num_videos_scanned": len(video_dirs),
        "num_reviewed_videos_available": len(reviewed_ids),
        "num_videos_skipped_not_reviewed": skipped_not_reviewed,
        "num_videos_with_hard_negatives": len(per_video_counts),
        "num_hard_negatives": len(hard_negatives),
        "per_label_counts": per_label_counts,
        "per_video_counts": per_video_counts,
        "skipped": {
            "matched_predictions": skipped_matched,
            "near_gt": skipped_near_gt,
            "score_filter": skipped_score,
        },
        "hard_negatives": hard_negatives,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build hard negatives from final PointNMS false positive football predictions."
    )
    parser.add_argument("--eval-run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reviewed-video-ids", nargs="+", required=True)
    parser.add_argument("--labels", default="shot,save")
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--max-score", type=float, default=1.0)
    parser.add_argument("--min-scores", default="")
    parser.add_argument("--max-scores", default="")
    parser.add_argument("--safety-margin-sec", type=float, default=8.0)
    parser.add_argument("--safety-label-mode", choices=("same_label", "any"), default="same_label")
    parser.add_argument("--dedupe-gap-sec", type=float, default=10.0)
    parser.add_argument("--max-per-video-per-label", type=int, default=8)
    parser.add_argument("--source", default="xbotgo_0608")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_score < 0 or args.max_score > 1 or args.min_score > args.max_score:
        raise ValueError("score bounds must satisfy 0 <= min <= max <= 1")
    manifest = build_manifest(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(
        f"wrote {output}: negatives={manifest['num_hard_negatives']} "
        f"videos={manifest['num_videos_with_hard_negatives']}/"
        f"{manifest['num_reviewed_videos_available']} per_label={manifest['per_label_counts']} "
        f"skipped={manifest['skipped']}"
    )


if __name__ == "__main__":
    main()
