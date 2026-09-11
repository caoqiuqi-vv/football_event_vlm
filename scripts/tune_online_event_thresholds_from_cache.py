#!/usr/bin/env python
"""Tune event-level PointNMS thresholds on online Val and rescore test cache."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_football_events as football
import train_football_events_online_simulation as online_sim


def cache_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def exact_metas(cfg: Any, split: str, sample_ids: np.ndarray) -> list[dict[str, Any]]:
    audit = cfg["eval"]["external_audit"]
    audit["enabled"] = True
    audit["split"] = split
    audit["online_mode"]["enabled"] = True
    records, events = online_sim._original_load(cfg, split)
    records = online_sim.build_online_eval_records(records, events, cfg)
    by_id = {record.sample_id: record for record in records}
    missing = [str(value) for value in sample_ids if str(value) not in by_id]
    if missing:
        raise KeyError(f"cache has {len(missing)} sample ids absent from {split}: {missing[:5]}")
    result: list[dict[str, Any]] = []
    for raw_id in sample_ids:
        record = by_id[str(raw_id)]
        result.append(
            {
                "video_id": record.video_id,
                "sample_id": record.sample_id,
                "base_clip_start": float(record.base_clip_start),
                "base_clip_end": float(record.base_clip_end),
                "sampled_clip_start": float(record.base_clip_start),
                "sampled_clip_end": float(record.base_clip_end),
                "online_gt_anchors": record.online_gt_anchors,
            }
        )
    return result


def select_thresholds(
    probs: np.ndarray,
    candidate_times: np.ndarray,
    metas: list[dict[str, Any]],
    floors: dict[str, float],
    nms_radius: float,
    tolerance: float,
    objectives: dict[str, str] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    thresholds: dict[str, float] = {}
    diagnostics: dict[str, Any] = {}
    videos = sorted({str(meta["video_id"]) for meta in metas})
    rows_by_video: dict[str, list[int]] = defaultdict(list)
    for index, meta in enumerate(metas):
        rows_by_video[str(meta["video_id"])].append(index)

    for label_index, label in enumerate(football.LABELS):
        scored_outcomes: list[tuple[float, int]] = []
        support = 0
        for video in videos:
            row_indices = rows_by_video[video]
            gt_values: set[float] = set()
            candidates: list[tuple[float, float]] = []
            for index in row_indices:
                anchors = metas[index].get("online_gt_anchors", ()) or ()
                if len(anchors) == len(football.LABELS):
                    gt_values.update(round(float(value), 4) for value in anchors[label_index])
                candidates.append(
                    (float(probs[index, label_index]), float(candidate_times[index, label_index]))
                )
            support += len(gt_values)
            kept: list[tuple[float, float]] = []
            for score, timestamp in sorted(candidates, reverse=True):
                if any(abs(timestamp - old_time) <= nms_radius for _, old_time in kept):
                    continue
                kept.append((score, timestamp))
            unmatched = set(gt_values)
            for score, timestamp in kept:
                eligible = [value for value in unmatched if abs(timestamp - value) <= tolerance]
                is_tp = 0
                if eligible:
                    matched = min(eligible, key=lambda value: abs(timestamp - value))
                    unmatched.remove(matched)
                    is_tp = 1
                scored_outcomes.append((score, is_tp))

        ordered = sorted(scored_outcomes, reverse=True)
        tp = fp = 0
        objective = (objectives or {}).get(label, "precision")
        if objective not in {"precision", "f1"}:
            raise ValueError(f"unsupported objective for {label}: {objective}")
        best: tuple[float, float, int, float, float, int, int] | None = None
        cursor = 0
        while cursor < len(ordered):
            score = ordered[cursor][0]
            while cursor < len(ordered) and ordered[cursor][0] == score:
                tp += ordered[cursor][1]
                fp += 1 - ordered[cursor][1]
                cursor += 1
            recall = tp / max(support, 1)
            precision = tp / max(tp + fp, 1)
            if recall + 1e-12 < float(floors[label]):
                continue
            f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
            objective_score = f1 if objective == "f1" else precision
            candidate = (objective_score, precision, -(tp + fp), score, recall, tp, fp)
            if best is None or candidate > best:
                best = candidate
        ceiling_tp, ceiling_fp = tp, fp
        ceiling_recall = ceiling_tp / max(support, 1)
        if best is None:
            threshold = 0.0
            tp, fp = ceiling_tp, ceiling_fp
            recall = ceiling_recall
            precision = tp / max(tp + fp, 1)
            feasible = False
        else:
            objective_score, precision, negative_count, threshold, recall, tp, fp = best
            feasible = True
        thresholds[label] = float(threshold)
        diagnostics[label] = {
            "floor": float(floors[label]),
            "objective": objective,
            "feasible": feasible,
            "candidate_recall_ceiling": float(ceiling_recall),
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(support - tp),
            "support": int(support),
        }
    return thresholds, diagnostics


def evaluate(cache: dict[str, np.ndarray], metas: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    threshold_array = np.asarray([thresholds[label] for label in football.LABELS], dtype=np.float32)
    return football.online_event_metrics(
        cache["probs"],
        cache["candidate_times"],
        metas,
        threshold_array,
        nms_radius_sec=5.0,
        tolerance_sec=5.0,
        capped_clip_sec=10.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--val-cache", required=True, type=Path)
    parser.add_argument("--external-cache", required=True, type=Path)
    parser.add_argument("--sampled-anchor-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    cfg = football.load_config(args.config, [])
    football.configure_label_schema(cfg)
    floors = {"shot": 0.85, "save": 0.0, "set_piece": 0.0}
    primary_objectives = {"shot": "precision", "save": "f1", "set_piece": "f1"}
    val_cache = cache_arrays(args.val_cache)
    external_cache = cache_arrays(args.external_cache)
    val_metas = exact_metas(cfg, "val", val_cache["sample_ids"])
    external_metas = exact_metas(cfg, "external", external_cache["sample_ids"])
    online_thresholds, online_search = select_thresholds(
        val_cache["probs"], val_cache["candidate_times"], val_metas,
        floors, 5.0, 5.0, objectives=primary_objectives,
    )
    sampled_report = json.loads(args.sampled_anchor_report.read_text(encoding="utf-8"))
    sampled_thresholds = {
        label: float(sampled_report["thresholds"][label]) for label in football.LABELS
    }
    operating_points: dict[str, Any] = {}
    for target in (0.80, 0.85, 0.90):
        target_floors = {label: target for label in football.LABELS}
        target_thresholds, target_search = select_thresholds(
            val_cache["probs"], val_cache["candidate_times"], val_metas,
            target_floors, 5.0, 5.0,
        )
        operating_points[f"R{int(target * 100)}"] = {
            "thresholds": target_thresholds,
            "search": target_search,
            "online_val15": evaluate(val_cache, val_metas, target_thresholds),
            "external18": evaluate(external_cache, external_metas, target_thresholds),
        }
    ceiling_thresholds = {label: 0.0 for label in football.LABELS}
    result = {
        "protocol": {
            "calibration": "online_val15_exact_stride5",
            "test": "external18_cached_no_redecode",
            "nms": "PointNMS one-to-one radius5 tolerance5",
            "floors": floors,
            "objectives": primary_objectives,
            "primary_policy": "shot precision at event recall>=0.85; save/set_piece exact event F1 optimum without recall floor",
        },
        "sampled_val15_thresholds": sampled_thresholds,
        "online_val15_thresholds": online_thresholds,
        "online_threshold_search": online_search,
        "candidate_recall_ceiling": {
            "thresholds": ceiling_thresholds,
            "online_val15": evaluate(val_cache, val_metas, ceiling_thresholds),
            "external18": evaluate(external_cache, external_metas, ceiling_thresholds),
        },
        "operating_points": operating_points,
        "sampled_threshold_metrics": {
            "online_val15": evaluate(val_cache, val_metas, sampled_thresholds),
            "external18": evaluate(external_cache, external_metas, sampled_thresholds),
        },
        "online_threshold_metrics": {
            "online_val15": evaluate(val_cache, val_metas, online_thresholds),
            "external18": evaluate(external_cache, external_metas, online_thresholds),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result["online_val15_thresholds"], sort_keys=True))
    print(f"report={args.output}")


if __name__ == "__main__":
    main()
