#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence


DEFAULT_LABELS = ("shot", "save", "set_piece")
BRANCH_PREFIXES = {"fused": "", "global": "global_", "local": "local_"}


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
    values = {label: default for label in labels}
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


def load_gt_times(path: Path) -> list[float]:
    return sorted(float(row["time_sec"]) for row in read_csv(path))


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
        candidates,
        key=lambda item: (
            abs(float(item["score"]) - float(item["target_score"])),
            -float(item["score"]),
            float(item["center_sec"]),
        ),
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


def merge_label_candidates(candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, float, float], dict[str, Any]] = {}
    for candidate in candidates:
        key = (
            str(candidate["video_id"]),
            float(candidate["start_sec"]),
            float(candidate["end_sec"]),
        )
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
    target_scores = parse_label_scores(args.target_scores, labels, default=args.target_score)
    reviewed_ids = read_video_ids(args.reviewed_video_ids)
    prefix = BRANCH_PREFIXES[args.branch]
    hard_negatives: list[dict[str, Any]] = []
    per_video_counts: dict[str, int] = {}
    per_label_counts = {label: 0 for label in labels}
    skipped_not_reviewed = 0

    video_dirs = sorted(
        path
        for path in run_dir.iterdir()
        if path.is_dir()
        and (path / "window_predictions.csv").exists()
        and (path / "gt_events.csv").exists()
    )
    for video_dir in video_dirs:
        video_id = video_dir.name
        if video_id not in reviewed_ids:
            skipped_not_reviewed += 1
            continue
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
                if score < min_scores[label] or score > max_scores[label]:
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
                        "video_id": video_id,
                        "label": label,
                        "score": score,
                        "target_score": target_scores[label],
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "center_sec": (start_sec + end_sec) * 0.5,
                        "window_index": int(row["index"]),
                        "mining_run": run_dir.name,
                        "score_branch": args.branch,
                        "mined_from_reviewed_video": True,
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
        if selected_for_video:
            hard_negatives.extend(selected_for_video)
            per_video_counts[video_id] = len(selected_for_video)

    run_config_path = run_dir / "run_config.json"
    run_config = json.loads(run_config_path.read_text()) if run_config_path.exists() else {}
    return {
        "schema": "football_reviewed_hard_negative_manifest_v2",
        "policy": (
            "Only mine reviewed train videos; reject windows near any GT; keep "
            "medium/high confidence FP bands and exclude extreme-score hard cases."
        ),
        "eval_run_dir": str(run_dir),
        "mining_checkpoint": str(run_config.get("checkpoint", "")),
        "source": args.source,
        "branch": args.branch,
        "labels": labels,
        "min_scores": min_scores,
        "max_scores": max_scores,
        "target_scores": target_scores,
        "safety_margin_sec": args.safety_margin_sec,
        "dedupe_gap_sec": args.dedupe_gap_sec,
        "max_per_video_per_label": args.max_per_video_per_label,
        "reviewed_video_id_files": [str(Path(path)) for path in args.reviewed_video_ids],
        "num_videos_scanned": len(video_dirs),
        "num_reviewed_videos_available": len(reviewed_ids),
        "num_videos_skipped_not_reviewed": skipped_not_reviewed,
        "num_videos_with_hard_negatives": len(per_video_counts),
        "num_hard_negatives": len(hard_negatives),
        "per_label_counts": per_label_counts,
        "per_video_counts": per_video_counts,
        "hard_negatives": hard_negatives,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build reviewed-video medium-score hard negatives from dense football outputs."
    )
    parser.add_argument("--eval-run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reviewed-video-ids", nargs="+", required=True)
    parser.add_argument("--branch", choices=sorted(BRANCH_PREFIXES), default="global")
    parser.add_argument("--labels", default="shot,save,set_piece")
    parser.add_argument("--min-score", type=float, default=0.35)
    parser.add_argument("--max-score", type=float, default=0.75)
    parser.add_argument("--target-score", type=float, default=0.55)
    parser.add_argument("--min-scores", default="")
    parser.add_argument("--max-scores", default="")
    parser.add_argument("--target-scores", default="")
    parser.add_argument("--safety-margin-sec", type=float, default=8.0)
    parser.add_argument("--dedupe-gap-sec", type=float, default=10.0)
    parser.add_argument("--max-per-video-per-label", type=int, default=12)
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
        f"{manifest['num_reviewed_videos_available']} per_label={manifest['per_label_counts']}"
    )


if __name__ == "__main__":
    main()
