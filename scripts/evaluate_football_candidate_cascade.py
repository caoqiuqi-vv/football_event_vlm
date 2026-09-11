#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_long_video_checkpoint import compute_event_metrics, window_overlap_predictions


DEFAULT_LABELS = ["shot", "save", "set_piece"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def parse_video_ids(args: argparse.Namespace) -> list[str]:
    ids: list[str] = []
    if args.video_ids:
        ids.extend(item.strip() for item in args.video_ids.split(",") if item.strip())
    if args.video_id_file:
        for line in Path(args.video_id_file).read_text().splitlines():
            item = line.strip()
            if item and not item.startswith("#"):
                ids.append(item)
    if not ids:
        ids = sorted(path.name for path in Path(args.dino_run).iterdir() if path.is_dir())
    seen: set[str] = set()
    result: list[str] = []
    for video_id in ids:
        if video_id not in seen:
            result.append(video_id)
            seen.add(video_id)
    return result


def merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted((float(a), float(b)) for a, b in intervals if float(b) >= float(a)):
        start = max(0.0, start)
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])


def contains(intervals: Sequence[tuple[float, float]], time_sec: float, tolerance_sec: float = 0.0) -> bool:
    event_interval = (float(time_sec) - tolerance_sec, float(time_sec) + tolerance_sec)
    return any(overlap(interval, event_interval) for interval in intervals)


def load_strategy_proposals(
    proposal_root: Path,
    video_id: str,
    label: str,
    *,
    min_confidence: float,
) -> list[dict[str, Any]]:
    path = proposal_root / video_id / f"{label}_proposals.json"
    if not path.exists():
        return []
    return [
        item
        for item in json.loads(path.read_text())
        if float(item.get("confidence", 0.0)) >= min_confidence
    ]


def load_strategy_intervals(
    proposal_root: Path,
    video_id: str,
    labels: Sequence[str],
    *,
    pad_sec: float,
    min_confidence: float,
) -> dict[str, list[tuple[float, float]]]:
    result = {label: [] for label in labels}
    video_dir = proposal_root / video_id
    for label in labels:
        for item in load_strategy_proposals(proposal_root, video_id, label, min_confidence=min_confidence):
            center = float(item.get("time_sec", item.get("center_time_sec", 0.0)))
            start = float(item.get("start_sec", center))
            end = float(item.get("end_sec", center))
            result[label].append((start - pad_sec, end + pad_sec))
    return {label: merge_intervals(items) for label, items in result.items()}


def load_whistle_intervals(
    whistle_root: Path,
    video_id: str,
    *,
    pre_sec: float,
    post_sec: float,
    min_score: float,
) -> list[tuple[float, float]]:
    path = whistle_root / f"{video_id}_whistles.csv"
    if not path.exists():
        return []
    intervals: list[tuple[float, float]] = []
    for row in read_csv(path):
        score = float(row.get("peak_score") or row.get("mean_score") or 0.0)
        if score < min_score:
            continue
        center = float(row.get("center_time_sec") or row.get("peak_time_sec"))
        intervals.append((center - pre_sec, center + post_sec))
    return merge_intervals(intervals)


def combine_candidates(
    strategy: dict[str, list[tuple[float, float]]],
    whistle: list[tuple[float, float]],
    labels: Sequence[str],
    *,
    source_mode: str,
    gate_mode: str,
) -> dict[str, list[tuple[float, float]]]:
    by_label: dict[str, list[tuple[float, float]]] = {label: [] for label in labels}
    if source_mode in ("strategy", "union"):
        for label in labels:
            by_label[label].extend(strategy.get(label, []))
    if source_mode in ("whistle", "union"):
        for label in labels:
            by_label[label].extend(whistle)
    if gate_mode == "class_agnostic":
        all_intervals: list[tuple[float, float]] = []
        for intervals in by_label.values():
            all_intervals.extend(intervals)
        merged = merge_intervals(all_intervals)
        return {label: merged for label in labels}
    return {label: merge_intervals(intervals) for label, intervals in by_label.items()}


def load_thresholds(dino_run: Path, video_ids: Sequence[str], labels: Sequence[str], raw: str) -> dict[str, float]:
    if raw != "checkpoint":
        if "=" not in raw:
            value = float(raw)
            return {label: value for label in labels}
        parsed = dict(item.split("=", 1) for item in raw.split(",") if item.strip())
        return {label: float(parsed[label]) for label in labels}
    thresholds: dict[str, float] = {}
    for label in labels:
        values: set[float] = set()
        for video_id in video_ids:
            summary = json.loads((dino_run / video_id / "summary.json").read_text())
            values.add(float(summary["thresholds"][label]))
        if len(values) != 1:
            raise ValueError(f"inconsistent checkpoint thresholds for {label}: {sorted(values)}")
        thresholds[label] = next(iter(values))
    return thresholds


def zero_metric() -> dict[str, Any]:
    return {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "num_pred": 0,
        "num_gt": 0,
        "num_matched_gt": 0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }


def finalize(item: dict[str, Any]) -> dict[str, Any]:
    tp = int(item.get("tp", 0))
    fp = int(item.get("fp", 0))
    fn = int(item.get("fn", 0))
    num_gt = int(item.get("num_gt", tp + fn))
    matched = int(item.get("num_matched_gt", tp))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = matched / num_gt if num_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    item.update(
        {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "num_pred": int(item.get("num_pred", tp + fp)),
            "num_gt": num_gt,
            "num_matched_gt": matched,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )
    return item


def should_exclude(video_id: str, label: str, excludes: set[tuple[str, str]]) -> bool:
    return (video_id, label) in excludes


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a proposal/whistle candidate gate on existing dense DINO window predictions.")
    parser.add_argument("--dino-run", required=True)
    parser.add_argument("--proposal-root", required=True)
    parser.add_argument("--whistle-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--video-ids", default="")
    parser.add_argument("--video-id-file", default="")
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--thresholds", default="checkpoint")
    parser.add_argument("--source-mode", choices=["strategy", "whistle", "union"], default="union")
    parser.add_argument("--gate-mode", choices=["class_agnostic", "per_class"], default="class_agnostic")
    parser.add_argument("--gate-labels", default="", help="Comma-separated labels to gate. Empty gates all labels; non-gated labels keep all DINO windows.")
    parser.add_argument("--strategy-pad-sec", type=float, default=20.0)
    parser.add_argument("--strategy-min-confidence", type=float, default=0.0)
    parser.add_argument("--whistle-pre-sec", type=float, default=10.0)
    parser.add_argument("--whistle-post-sec", type=float, default=120.0)
    parser.add_argument("--whistle-min-score", type=float, default=0.45)
    parser.add_argument("--save-from-shot-post-sec", type=float, default=0.0, help="If >0, replace save candidates with [shot_strategy_time, shot_strategy_time+N] intervals.")
    parser.add_argument("--match-tolerance-sec", type=float, default=2.0)
    parser.add_argument("--exclude-video-label", action="append", default=[])
    args = parser.parse_args()

    dino_run = Path(args.dino_run)
    proposal_root = Path(args.proposal_root)
    whistle_root = Path(args.whistle_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    video_ids = parse_video_ids(args)
    excludes = {
        tuple(item.split(":", 1))  # type: ignore[misc]
        for item in args.exclude_video_label
        if ":" in item
    }
    thresholds = load_thresholds(dino_run, video_ids, labels, args.thresholds)
    gate_labels = (
        {item.strip() for item in args.gate_labels.split(",") if item.strip()}
        if args.gate_labels.strip()
        else set(labels)
    )
    unknown_gate_labels = sorted(gate_labels.difference(labels))
    if unknown_gate_labels:
        raise ValueError(f"--gate-labels contains labels not in --labels: {unknown_gate_labels}")

    per_video_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    aggregate = {label: zero_metric() for label in labels}
    micro = zero_metric()
    aggregate_ex = {label: zero_metric() for label in labels}
    micro_ex = zero_metric()

    for video_id in video_ids:
        rows = read_csv(dino_run / video_id / "window_predictions.csv")
        gt_events = json.loads((dino_run / video_id / "gt_events.json").read_text())
        strategy = load_strategy_intervals(
            proposal_root,
            video_id,
            labels,
            pad_sec=args.strategy_pad_sec,
            min_confidence=args.strategy_min_confidence,
        )
        whistle = load_whistle_intervals(
            whistle_root,
            video_id,
            pre_sec=args.whistle_pre_sec,
            post_sec=args.whistle_post_sec,
            min_score=args.whistle_min_score,
        )
        candidates = combine_candidates(strategy, whistle, labels, source_mode=args.source_mode, gate_mode=args.gate_mode)
        if args.save_from_shot_post_sec > 0 and "save" in labels:
            shot_proposals = load_strategy_proposals(
                proposal_root,
                video_id,
                "shot",
                min_confidence=args.strategy_min_confidence,
            )
            save_intervals = []
            for item in shot_proposals:
                center = float(item.get("time_sec", item.get("center_time_sec", 0.0)))
                save_intervals.append((center, center + float(args.save_from_shot_post_sec)))
            candidates["save"] = merge_intervals(save_intervals)

        gated_rows: list[dict[str, Any]] = []
        for row in rows:
            gated = dict(row)
            window = (float(row["start_sec"]), float(row["end_sec"]))
            for label in labels:
                if label not in gate_labels:
                    allowed = True
                else:
                    allowed = any(overlap(window, interval) for interval in candidates[label])
                gated[f"candidate_pass_{label}"] = int(allowed)
                if not allowed:
                    gated[f"prob_{label}"] = 0.0
            gated_rows.append(gated)

        predictions = window_overlap_predictions(gated_rows, labels, thresholds)
        metrics = compute_event_metrics(
            predictions,
            gt_events,
            args.match_tolerance_sec,
            matching_mode="window",
            allow_many_predictions_per_gt=True,
        )

        video_dir = output_dir / video_id
        video_dir.mkdir(parents=True, exist_ok=True)
        (video_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
        (video_dir / "candidate_intervals.json").write_text(json.dumps(candidates, ensure_ascii=False, indent=2))
        (video_dir / "predicted_events.json").write_text(json.dumps(predictions, ensure_ascii=False, indent=2))

        for label in labels:
            item = metrics["per_class"].get(label, zero_metric())
            row = {"video_id": video_id, "label": label, **finalize(dict(item))}
            per_video_rows.append(row)
            gt_label = [event for event in gt_events if event.get("label") == label]
            covered = sum(1 for event in gt_label if contains(candidates[label], float(event["time_sec"]), args.match_tolerance_sec))
            coverage = {
                "video_id": video_id,
                "label": label,
                "num_gt": len(gt_label),
                "candidate_matched_gt": covered,
                "candidate_recall": covered / len(gt_label) if gt_label else 0.0,
                "num_intervals": len(candidates[label]),
                "candidate_seconds": sum(end - start for start, end in candidates[label]),
            }
            coverage_rows.append(coverage)
            for key in ("tp", "fp", "fn", "num_pred", "num_gt", "num_matched_gt"):
                aggregate[label][key] += row[key]
                micro[key] += row[key]
                if not should_exclude(video_id, label, excludes):
                    aggregate_ex[label][key] += row[key]
                    micro_ex[key] += row[key]

    per_class = {label: finalize(item) for label, item in aggregate.items()}
    per_class_ex = {label: finalize(item) for label, item in aggregate_ex.items()}
    summary = {
        "config": {
            "dino_run": str(dino_run),
            "proposal_root": str(proposal_root),
            "whistle_root": str(whistle_root),
            "video_ids": video_ids,
            "labels": labels,
            "thresholds": thresholds,
            "source_mode": args.source_mode,
            "gate_mode": args.gate_mode,
            "gate_labels": sorted(gate_labels),
            "strategy_pad_sec": args.strategy_pad_sec,
            "strategy_min_confidence": args.strategy_min_confidence,
            "whistle_pre_sec": args.whistle_pre_sec,
            "whistle_post_sec": args.whistle_post_sec,
            "whistle_min_score": args.whistle_min_score,
            "save_from_shot_post_sec": args.save_from_shot_post_sec,
            "match_tolerance_sec": args.match_tolerance_sec,
            "exclude_video_label": args.exclude_video_label,
        },
        "per_class": per_class,
        "micro": finalize(micro),
        "per_class_exclude": per_class_ex,
        "micro_exclude": finalize(micro_ex),
        "candidate_coverage": coverage_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    write_csv(output_dir / "per_video_per_class.csv", per_video_rows, list(per_video_rows[0]) if per_video_rows else [])
    write_csv(output_dir / "candidate_coverage.csv", coverage_rows, list(coverage_rows[0]) if coverage_rows else [])
    print(json.dumps({"output_dir": str(output_dir), "micro_exclude": summary["micro_exclude"], "per_class_exclude": summary["per_class_exclude"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
