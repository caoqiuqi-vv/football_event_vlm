#!/usr/bin/env python
"""Measure whether coarse long-video retrieval can reduce high-res workload.

This is a proposal-stage analysis, not an event-spotting score.  Dense window
scores are pooled into non-overlapping temporal blocks.  We report the minimum
block coverage reached by the current score ordering at a requested event
recall, together with an oracle lower bound.  No temporal NMS is involved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def merge_duration(intervals: Sequence[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    start, end = ordered[0]
    total = 0.0
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def block_metrics(
    blocks: Sequence[dict[str, Any]],
    gt: dict[str, list[float]],
    durations: dict[str, float],
    selected: np.ndarray,
) -> dict[str, Any]:
    selected_blocks = [block for block, keep in zip(blocks, selected) if keep]
    covered = 0
    per_video: dict[str, dict[str, float | int]] = {}
    for video_id, targets in gt.items():
        intervals = [
            (float(block["start"]), float(block["end"]))
            for block in selected_blocks if block["video_id"] == video_id
        ]
        video_covered = sum(
            any(start <= target < end for start, end in intervals)
            for target in targets
        )
        covered += video_covered
        per_video[video_id] = {
            "gt": len(targets),
            "covered": video_covered,
            "recall": video_covered / max(len(targets), 1),
        }
    selected_duration = sum(float(block["end"] - block["start"]) for block in selected_blocks)
    total_duration = sum(durations.values())
    positive_selected = sum(int(block["gt_count"] > 0) for block in selected_blocks)
    total_gt = sum(len(values) for values in gt.values())
    return {
        "selected_blocks": len(selected_blocks),
        "positive_selected_blocks": positive_selected,
        "block_precision": positive_selected / max(len(selected_blocks), 1),
        "event_recall": covered / max(total_gt, 1),
        "selected_minutes": selected_duration / 60.0,
        "total_minutes": total_duration / 60.0,
        "coverage_ratio": selected_duration / max(total_duration, 1e-12),
        "macro_video_recall": float(np.mean([row["recall"] for row in per_video.values()])),
        "videos_below_80_recall": sum(row["recall"] < 0.8 for row in per_video.values()),
        "per_video": per_video,
    }


def choose_score_threshold(
    blocks: Sequence[dict[str, Any]],
    gt: dict[str, list[float]],
    durations: dict[str, float],
    recall_floor: float,
) -> dict[str, Any]:
    scores = np.asarray([block["score"] for block in blocks], dtype=np.float64)
    candidates = sorted(set(scores.tolist()), reverse=True)
    feasible: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for threshold in candidates:
        metrics = block_metrics(blocks, gt, durations, scores >= threshold)
        metrics["threshold"] = float(threshold)
        all_rows.append(metrics)
        if metrics["event_recall"] + 1e-12 >= recall_floor:
            feasible.append(metrics)
    ceiling = max(row["event_recall"] for row in all_rows)
    pool = feasible or [row for row in all_rows if row["event_recall"] + 1e-12 >= ceiling]
    best = min(
        pool,
        key=lambda row: (
            row["coverage_ratio"], -row["block_precision"],
            -row["macro_video_recall"], -row["threshold"],
        ),
    )
    best["requested_recall_floor"] = recall_floor
    best["recall_floor_reachable"] = ceiling + 1e-12 >= recall_floor
    return best


def oracle(blocks: Sequence[dict[str, Any]], gt: dict[str, list[float]], durations: dict[str, float], recall_floor: float) -> dict[str, Any]:
    order = sorted(
        range(len(blocks)),
        key=lambda index: (-int(blocks[index]["gt_count"]), float(blocks[index]["start"])),
    )
    selected = np.zeros(len(blocks), dtype=bool)
    total_gt = sum(len(values) for values in gt.values())
    covered = 0
    for index in order:
        count = int(blocks[index]["gt_count"])
        if count <= 0 or covered / max(total_gt, 1) + 1e-12 >= recall_floor:
            break
        selected[index] = True
        covered += count
    result = block_metrics(blocks, gt, durations, selected)
    result["requested_recall_floor"] = recall_floor
    return result


def build_blocks(cache: Any, metadata: list[dict[str, Any]], block_sec: float, pooling: str) -> tuple[list[dict[str, Any]], dict[str, list[float]], dict[str, float]]:
    labels = [str(value) for value in cache["labels"]]
    shot_index = labels.index("shot")
    video_ids = cache["video_ids"].astype(str)
    starts = cache["clip_starts"].astype(np.float64)
    ends = cache["clip_ends"].astype(np.float64)
    centres = 0.5 * (starts + ends)
    scores = cache["online_probs"][:, shot_index].astype(np.float64)
    durations = {
        video_id: float(ends[video_ids == video_id].max())
        for video_id in sorted(set(video_ids.tolist()))
    }
    gt: dict[str, list[float]] = {}
    for index, video_id in enumerate(video_ids):
        if video_id not in gt:
            gt[video_id] = sorted(
                float(value) for value in metadata[index]["online_gt_anchors"][shot_index]
            )
    blocks: list[dict[str, Any]] = []
    for video_id, duration in durations.items():
        rows = np.where(video_ids == video_id)[0]
        for start in np.arange(0.0, duration, block_sec):
            end = min(start + block_sec, duration)
            selected = rows[(centres[rows] >= start) & (centres[rows] < end)]
            if not len(selected):
                score = 0.0
            elif pooling == "max":
                score = float(scores[selected].max())
            elif pooling == "top2_mean":
                values = np.sort(scores[selected])[-2:]
                score = float(values.mean())
            elif pooling == "top3_mean":
                values = np.sort(scores[selected])[-3:]
                score = float(values.mean())
            else:
                raise ValueError(pooling)
            targets = [value for value in gt[video_id] if start <= value < end]
            blocks.append({
                "video_id": video_id, "start": float(start), "end": float(end),
                "score": score, "gt_count": len(targets),
            })
    return blocks, gt, durations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--block-sec", default="20,30,60,120,180")
    parser.add_argument("--pooling", default="max,top2_mean,top3_mean")
    parser.add_argument("--recall-floor", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache = np.load(args.cache, allow_pickle=True)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    if len(metadata) != len(cache["video_ids"]):
        raise ValueError("cache and metadata row counts differ")
    report: dict[str, Any] = {
        "protocol": {
            "task": "coarse_temporal_retrieval",
            "not_event_spotting_precision": True,
            "recall_floor": args.recall_floor,
            "no_nms": True,
        },
        "results": [],
    }
    block_sizes = [float(value) for value in args.block_sec.split(",") if value.strip()]
    poolings = [value.strip() for value in args.pooling.split(",") if value.strip()]
    for block_sec in block_sizes:
        oracle_result = None
        for pooling in poolings:
            blocks, gt, durations = build_blocks(cache, metadata, block_sec, pooling)
            if oracle_result is None:
                oracle_result = oracle(blocks, gt, durations, args.recall_floor)
            score_result = choose_score_threshold(blocks, gt, durations, args.recall_floor)
            report["results"].append({
                "block_sec": block_sec,
                "pooling": pooling,
                "score_ordering": score_result,
                "oracle_lower_bound": oracle_result,
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
