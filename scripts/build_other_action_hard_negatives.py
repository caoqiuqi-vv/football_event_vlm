#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PURE_ACTION_LABELS = {
    "拦截": "interception",
    "抢断": "tackle",
    "解围": "clearance",
    "盘带": "dribble",
    "对抗成功": "duel_won",
}
TARGET_EVENT_LABELS = {"射门", "其他射门类型", "扑救", "角球", "任意球", "点球", "中圈开球"}
TARGET_EVENT_TYPES = {
    "S0199", "B0199", "S0401", "B0401",
    "S0201", "B0201", "S0202", "B0202",
    "S0101", "B0101", "S1004", "S06", "B1004", "B06",
}
SET_PIECE_LABELS = {"角球", "任意球", "点球", "中圈开球"}
SET_PIECE_TYPES = {"S0201", "B0201", "S0202", "B0202", "S0101", "B0101", "S1004", "S06", "B1004", "B06"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build score-filtered other-action hard negatives from reviewed train videos."
    )
    parser.add_argument("--raw-annotations", type=Path, default=Path("/home/new_users/qiuqi/code/football_events_raw"))
    parser.add_argument("--repair-annotations", type=Path, default=Path("/home/new_users/qiuqi/code/football_events_human_repair"))
    parser.add_argument("--reviewed-train-ids", type=Path, default=Path("configs/football/splits/vitl16_lvd1689m_set_piece_seed42_162videos_reviewed/train_video_ids.txt"))
    parser.add_argument("--dense-run-dir", type=Path, default=Path("outputs/football_eval_runs/vitl16_lora_r5_f32_16f_e1_frame_det_reviewed_train_dense_mining"))
    parser.add_argument("--output", type=Path, default=Path("outputs/football_hard_negatives/other_action_reviewed_train_shot_save_score_filtered.json"))
    parser.add_argument("--clip-sec", type=float, default=10.0)
    parser.add_argument("--min-shot-prob", type=float, default=0.10)
    parser.add_argument("--min-save-prob", type=float, default=0.10)
    parser.add_argument("--max-per-video-per-label", type=int, default=10)
    parser.add_argument("--max-total", type=int, default=2200)
    parser.add_argument("--dedupe-sec", type=float, default=2.0)
    parser.add_argument("--positive-safety-sec", type=float, default=8.0)
    parser.add_argument("--set-piece-safety-sec", type=float, default=10.0)
    parser.add_argument("--include-whistle-foul", action="store_true", help="Disabled by default; first version uses only pure duel/action labels.")
    return parser.parse_args()


def parse_time(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        return None if math.isnan(result) or math.isinf(result) else result
    text = str(value).strip()
    if not text:
        return None
    if ":" in text:
        parts = text.split(":")
        try:
            vals = [float(part) for part in parts]
        except ValueError:
            return None
        if len(vals) == 3:
            return vals[0] * 3600.0 + vals[1] * 60.0 + vals[2]
        if len(vals) == 2:
            return vals[0] * 60.0 + vals[1]
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    return None if math.isnan(result) or math.isinf(result) else result


def load_json_items(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    items = raw.get("data", raw) if isinstance(raw, dict) else raw
    return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def event_anchor(item: dict[str, Any]) -> float | None:
    start = parse_time(item.get("startTime", item.get("timestamp")))
    end = parse_time(item.get("endTime"))
    if start is None:
        return None
    if end is None or end < start:
        end = start
    return 0.5 * (start + end) if end > start else start


def item_label(item: dict[str, Any]) -> str:
    return str(item.get("label", ""))


def item_type(item: dict[str, Any]) -> str:
    return str(item.get("eventType", item.get("event_type", "")))


def is_target_event(item: dict[str, Any]) -> bool:
    return item_label(item) in TARGET_EVENT_LABELS or item_type(item) in TARGET_EVENT_TYPES


def is_set_piece(item: dict[str, Any]) -> bool:
    return item_label(item) in SET_PIECE_LABELS or item_type(item) in SET_PIECE_TYPES


def load_repaired_target_anchors(path: Path) -> tuple[list[float], list[float]]:
    targets: list[float] = []
    set_pieces: list[float] = []
    for item in load_json_items(path):
        if item.get("label_correct") is False:
            continue
        anchor = event_anchor(item)
        if anchor is None:
            continue
        if is_target_event(item):
            targets.append(anchor)
        if is_set_piece(item):
            set_pieces.append(anchor)
    return targets, set_pieces


def load_other_actions(path: Path) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for item in load_json_items(path):
        if item.get("label_correct") is False:
            continue
        label = item_label(item)
        if label not in PURE_ACTION_LABELS:
            continue
        anchor = event_anchor(item)
        if anchor is None:
            continue
        actions.append({"time_sec": anchor, "raw_label": label, "action_type": PURE_ACTION_LABELS[label], "raw": item})
    actions.sort(key=lambda item: (float(item["time_sec"]), str(item["raw_label"])))
    return actions


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(result) or math.isinf(result) else result


def nearest_dense_score(rows: list[dict[str, str]], time_sec: float) -> tuple[dict[str, str] | None, float]:
    containing = [
        row for row in rows
        if to_float(row.get("start_sec")) <= time_sec <= to_float(row.get("end_sec"))
    ]
    candidates = containing if containing else rows
    if not candidates:
        return None, float("inf")
    best = min(
        candidates,
        key=lambda row: abs(0.5 * (to_float(row.get("start_sec")) + to_float(row.get("end_sec"))) - time_sec),
    )
    center = 0.5 * (to_float(best.get("start_sec")) + to_float(best.get("end_sec")))
    return best, abs(center - time_sec)


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.strip().startswith("#")]


def main() -> None:
    args = parse_args()
    reviewed_ids = read_ids(args.reviewed_train_ids)
    items: list[dict[str, Any]] = []
    stats = Counter()
    skipped = Counter()

    for video_id in reviewed_ids:
        raw_path = args.raw_annotations / f"{video_id}.json"
        repair_path = args.repair_annotations / f"{video_id}.json"
        dense_path = args.dense_run_dir / video_id / "window_predictions.csv"
        dense_rows = read_csv(dense_path)
        if not dense_rows:
            skipped["missing_dense_scores"] += 1
            continue
        target_anchors, set_piece_anchors = load_repaired_target_anchors(repair_path)
        seen_times_by_label: dict[str, list[float]] = defaultdict(list)
        for action in load_other_actions(raw_path):
            time_sec = float(action["time_sec"])
            if any(abs(time_sec - anchor) <= args.positive_safety_sec for anchor in target_anchors):
                skipped["near_target_gt"] += 1
                continue
            if any(abs(time_sec - anchor) <= args.set_piece_safety_sec for anchor in set_piece_anchors):
                skipped["near_set_piece_gt"] += 1
                continue
            raw_label = str(action["raw_label"])
            if any(abs(time_sec - old) <= args.dedupe_sec for old in seen_times_by_label[raw_label]):
                skipped["dedupe"] += 1
                continue
            row, distance = nearest_dense_score(dense_rows, time_sec)
            if row is None:
                skipped["no_dense_row"] += 1
                continue
            prob_shot = to_float(row.get("prob_shot"))
            prob_save = to_float(row.get("prob_save"))
            labels: list[str] = []
            if prob_shot >= args.min_shot_prob:
                labels.append("shot")
            if prob_save >= args.min_save_prob:
                labels.append("save")
            if not labels:
                skipped["below_score_threshold"] += 1
                continue
            score = max(prob_shot if "shot" in labels else 0.0, prob_save if "save" in labels else 0.0)
            start = max(time_sec - 0.5 * args.clip_sec, 0.0)
            end = start + args.clip_sec
            items.append({
                "video_id": video_id,
                "source": "",
                "center_sec": time_sec,
                "start_sec": start,
                "end_sec": end,
                "labels": labels,
                "score": score,
                "prob_shot": prob_shot,
                "prob_save": prob_save,
                "raw_label": raw_label,
                "action_type": action["action_type"],
                "nearest_dense_center_distance_sec": distance,
                "dense_window_index": int(to_float(row.get("index"), -1)),
                "selection_policy": "pure_actions_only_score_filtered_v1",
            })
            seen_times_by_label[raw_label].append(time_sec)
            stats[raw_label] += 1

    # Per-video/per-label cap, using score ranking. A multi-label item consumes both caps.
    items.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    kept: list[dict[str, Any]] = []
    per_video_label_counts: dict[tuple[str, str], int] = defaultdict(int)
    for item in items:
        labels = [str(label) for label in item.get("labels", [])]
        video_id = str(item["video_id"])
        if args.max_per_video_per_label > 0 and any(
            per_video_label_counts[(video_id, label)] >= args.max_per_video_per_label
            for label in labels
        ):
            skipped["cap_per_video_per_label"] += 1
            continue
        kept.append(item)
        for label in labels:
            per_video_label_counts[(video_id, label)] += 1
        if args.max_total > 0 and len(kept) >= args.max_total:
            skipped["cap_total_after_kept"] += max(0, len(items) - len(kept))
            break

    kept_label_slots = Counter(label for item in kept for label in item.get("labels", []))
    kept_action_counts = Counter(str(item.get("raw_label", "")) for item in kept)
    manifest = {
        "schema_version": "other_action_hard_negative_v1",
        "policy": {
            "reviewed_train_ids": str(args.reviewed_train_ids),
            "dense_run_dir": str(args.dense_run_dir),
            "pure_action_labels": sorted(PURE_ACTION_LABELS),
            "excluded_first_version": ["裁判鸣哨", "犯规", "其他犯规类型"],
            "min_shot_prob": args.min_shot_prob,
            "min_save_prob": args.min_save_prob,
            "positive_safety_sec": args.positive_safety_sec,
            "set_piece_safety_sec": args.set_piece_safety_sec,
            "max_per_video_per_label": args.max_per_video_per_label,
            "max_total": args.max_total,
        },
        "summary": {
            "num_reviewed_videos": len(reviewed_ids),
            "num_candidates_before_cap": len(items),
            "num_hard_negatives": len(kept),
            "num_videos": len({item["video_id"] for item in kept}),
            "label_slots": dict(kept_label_slots),
            "raw_action_counts": dict(kept_action_counts),
            "skipped": dict(skipped),
        },
        "hard_negatives": kept,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
