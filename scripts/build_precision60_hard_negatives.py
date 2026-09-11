#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence


DEFAULT_LABELS = ("shot", "save")
BRANCH_PREFIXES = {"fused": "", "global": "global_", "local": "local_"}


def parse_label_values(raw: str, labels: Sequence[str]) -> dict[str, float]:
    values: dict[str, float] = {}
    for item in raw.split(","):
        if not item.strip():
            continue
        label, value = item.split("=", 1)
        values[label.strip()] = float(value)
    missing = set(labels) - set(values)
    if missing:
        raise ValueError(f"Missing values for labels: {sorted(missing)}")
    return {label: values[label] for label in labels}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def load_gt_times(path: Path) -> list[float]:
    rows = read_csv(path)
    return sorted(float(row["time_sec"]) for row in rows)


def overlaps_any_gt(
    gt_times: Sequence[float],
    *,
    start_sec: float,
    end_sec: float,
    safety_margin_sec: float,
) -> bool:
    safe_start = start_sec - safety_margin_sec
    safe_end = end_sec + safety_margin_sec
    return any(safe_start <= time_sec <= safe_end for time_sec in gt_times)


def dedupe_candidates(
    candidates: Sequence[dict[str, Any]],
    *,
    gap_sec: float,
    limit: int,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for candidate in sorted(
        candidates, key=lambda item: (-float(item["score"]), float(item["center_sec"]))
    ):
        if gap_sec > 0 and any(
            abs(float(candidate["center_sec"]) - float(old["center_sec"])) <= gap_sec
            for old in kept
        ):
            continue
        kept.append(dict(candidate))
        if limit > 0 and len(kept) >= limit:
            break
    return kept


def merge_label_candidates(
    candidates: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge class-specific supervision for the same decoded clip."""
    merged: dict[tuple[str, float, float], dict[str, Any]] = {}
    for candidate in candidates:
        key = (
            str(candidate["video_id"]),
            float(candidate["start_sec"]),
            float(candidate["end_sec"]),
        )
        if key not in merged:
            item = dict(candidate)
            label = str(item.pop("label"))
            item["labels"] = [label]
            item["label_scores"] = {label: float(item["score"])}
            merged[key] = item
            continue
        item = merged[key]
        label = str(candidate["label"])
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
    run_config_path = run_dir / "run_config.json"
    run_config = json.loads(run_config_path.read_text()) if run_config_path.exists() else {}
    labels = [item.strip() for item in args.labels.split(",") if item.strip()]
    thresholds = parse_label_values(args.min_scores, labels)
    prefix = BRANCH_PREFIXES[args.branch]
    hard_negatives: list[dict[str, Any]] = []
    per_video_counts: dict[str, int] = {}
    per_label_counts = {label: 0 for label in labels}

    video_dirs = sorted(
        path
        for path in run_dir.iterdir()
        if path.is_dir()
        and (path / "window_predictions.csv").exists()
        and (path / "gt_events.csv").exists()
    )
    for video_dir in video_dirs:
        windows = read_csv(video_dir / "window_predictions.csv")
        gt_times = load_gt_times(video_dir / "gt_events.csv")
        selected_for_video: list[dict[str, Any]] = []
        for label in labels:
            column = f"{prefix}prob_{label}"
            if windows and column not in windows[0]:
                raise KeyError(f"Missing score column '{column}' in {video_dir}")
            candidates: list[dict[str, Any]] = []
            for row in windows:
                score = float(row[column])
                if score < thresholds[label]:
                    continue
                start_sec = float(row["start_sec"])
                end_sec = float(row["end_sec"])
                if overlaps_any_gt(
                    gt_times,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    safety_margin_sec=args.safety_margin_sec,
                ):
                    continue
                candidates.append(
                    {
                        "source": args.source,
                        "video_id": video_dir.name,
                        "label": label,
                        "score": score,
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "center_sec": (start_sec + end_sec) * 0.5,
                        "window_index": int(row["index"]),
                        "mining_run": run_dir.name,
                        "score_branch": args.branch,
                    }
                )
            selected = dedupe_candidates(
                candidates,
                gap_sec=args.dedupe_gap_sec,
                limit=args.max_per_video_per_label,
            )
            selected_for_video.extend(selected)
            per_label_counts[label] += len(selected)
        selected_for_video = merge_label_candidates(selected_for_video)
        hard_negatives.extend(selected_for_video)
        per_video_counts[video_dir.name] = len(selected_for_video)

    return {
        "schema": "football_hard_negative_manifest_v1",
        "eval_run_dir": str(run_dir),
        "mining_checkpoint": str(run_config.get("checkpoint", "")),
        "source": args.source,
        "branch": args.branch,
        "min_scores": thresholds,
        "safety_margin_sec": args.safety_margin_sec,
        "dedupe_gap_sec": args.dedupe_gap_sec,
        "max_per_video_per_label": args.max_per_video_per_label,
        "num_videos": len(video_dirs),
        "num_hard_negatives": len(hard_negatives),
        "per_label_counts": per_label_counts,
        "per_video_counts": per_video_counts,
        "hard_negatives": hard_negatives,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a small label-balanced hard-negative manifest from saved dense "
            "football predictions."
        )
    )
    parser.add_argument("--eval-run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--branch", choices=sorted(BRANCH_PREFIXES), default="global")
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--min-scores", default="shot=0.55,save=0.55")
    parser.add_argument("--safety-margin-sec", type=float, default=5.0)
    parser.add_argument("--dedupe-gap-sec", type=float, default=5.0)
    parser.add_argument("--max-per-video-per-label", type=int, default=6)
    parser.add_argument("--source", default="xbotgo_0608")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.safety_margin_sec < 0:
        raise ValueError("--safety-margin-sec must be non-negative")
    if args.dedupe_gap_sec < 0:
        raise ValueError("--dedupe-gap-sec must be non-negative")
    if args.max_per_video_per_label <= 0:
        raise ValueError("--max-per-video-per-label must be positive")
    manifest = build_manifest(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(
        f"wrote {output}: negatives={manifest['num_hard_negatives']} "
        f"videos={manifest['num_videos']} per_label={manifest['per_label_counts']}"
    )


if __name__ == "__main__":
    main()
