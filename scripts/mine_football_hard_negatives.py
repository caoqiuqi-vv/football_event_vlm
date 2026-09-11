#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence


DEFAULT_LABELS = ("shot", "save", "set_piece")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine high-confidence background windows from dense football evaluation outputs."
    )
    parser.add_argument("--eval-run-dir", required=True, help="Dense evaluation run containing per-video window_predictions.csv.")
    parser.add_argument("--output", required=True, help="Output hard-negative JSON manifest.")
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--min-score", type=float, default=0.7)
    parser.add_argument(
        "--gt-safety-margin-sec",
        type=float,
        default=2.0,
        help="Reject a candidate when any target GT overlaps the window expanded by this margin.",
    )
    parser.add_argument("--dedupe-gap-sec", type=float, default=5.0)
    parser.add_argument("--max-per-video", type=int, default=20)
    parser.add_argument(
        "--source",
        default="",
        help="Optional source written to every record. Empty source is matched by video_id during training.",
    )
    return parser.parse_args()


def overlaps_target(
    gt_events: Sequence[dict[str, Any]],
    start_sec: float,
    end_sec: float,
    margin_sec: float,
) -> bool:
    expanded_start = start_sec - margin_sec
    expanded_end = end_sec + margin_sec
    for event in gt_events:
        gt_start = float(event.get("start_sec", event.get("time_sec", 0.0)))
        gt_end = float(event.get("end_sec", event.get("time_sec", gt_start)))
        if expanded_end >= gt_start and expanded_start <= gt_end:
            return True
    return False


def mine_video(
    video_dir: Path,
    labels: Sequence[str],
    *,
    min_score: float,
    gt_safety_margin_sec: float,
    dedupe_gap_sec: float,
    max_per_video: int,
    source: str,
) -> list[dict[str, Any]]:
    window_path = video_dir / "window_predictions.csv"
    gt_path = video_dir / "gt_events.json"
    if not window_path.exists() or not gt_path.exists():
        return []

    gt_events = json.loads(gt_path.read_text())
    candidates: list[dict[str, Any]] = []
    with window_path.open(newline="") as f:
        for row in csv.DictReader(f):
            available_labels = [label for label in labels if f"prob_{label}" in row and row[f"prob_{label}"] != ""]
            if not available_labels:
                continue
            probabilities = {label: float(row[f"prob_{label}"]) for label in available_labels}
            predicted_labels = [label for label, score in probabilities.items() if score >= min_score]
            if not predicted_labels:
                continue
            start_sec = float(row["start_sec"])
            end_sec = float(row["end_sec"])
            if overlaps_target(gt_events, start_sec, end_sec, gt_safety_margin_sec):
                continue
            center_sec = (start_sec + end_sec) * 0.5
            candidates.append(
                {
                    "source": source,
                    "video_id": video_dir.name,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "center_sec": center_sec,
                    "score": max(probabilities[label] for label in predicted_labels),
                    "predicted_labels": predicted_labels,
                    "probabilities": probabilities,
                    "window_index": int(row["index"]),
                }
            )

    candidates.sort(key=lambda item: (-float(item["score"]), float(item["center_sec"])))
    kept: list[dict[str, Any]] = []
    for candidate in candidates:
        if dedupe_gap_sec > 0 and any(
            abs(float(candidate["center_sec"]) - float(old["center_sec"])) <= dedupe_gap_sec
            for old in kept
        ):
            continue
        kept.append(candidate)
        if max_per_video > 0 and len(kept) >= max_per_video:
            break
    return kept


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.min_score <= 1.0:
        raise ValueError("--min-score must be in [0, 1]")
    labels = [item.strip() for item in args.labels.split(",") if item.strip()]
    run_dir = Path(args.eval_run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    hard_negatives: list[dict[str, Any]] = []
    per_video: dict[str, int] = {}
    video_dirs = sorted(path for path in run_dir.iterdir() if path.is_dir())
    for video_dir in video_dirs:
        items = mine_video(
            video_dir,
            labels,
            min_score=args.min_score,
            gt_safety_margin_sec=args.gt_safety_margin_sec,
            dedupe_gap_sec=args.dedupe_gap_sec,
            max_per_video=args.max_per_video,
            source=args.source,
        )
        if items:
            per_video[video_dir.name] = len(items)
            hard_negatives.extend(items)

    run_config_path = run_dir / "run_config.json"
    run_config = json.loads(run_config_path.read_text()) if run_config_path.exists() else {}
    payload = {
        "format_version": 1,
        "source_eval_run": str(run_dir),
        "source_checkpoint": run_config.get("checkpoint", ""),
        "labels": labels,
        "min_score": args.min_score,
        "gt_safety_margin_sec": args.gt_safety_margin_sec,
        "dedupe_gap_sec": args.dedupe_gap_sec,
        "max_per_video": args.max_per_video,
        "num_videos_scanned": len(video_dirs),
        "num_videos_with_hard_negatives": len(per_video),
        "num_hard_negatives": len(hard_negatives),
        "per_video_counts": per_video,
        "hard_negatives": hard_negatives,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(
        f"wrote {output_path}: hard_negatives={len(hard_negatives)} "
        f"videos={len(per_video)}/{len(video_dirs)}"
    )


if __name__ == "__main__":
    main()
