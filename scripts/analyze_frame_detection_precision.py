#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


DEFAULT_LABELS = ("shot", "save", "set_piece")


@dataclass(frozen=True)
class WindowRow:
    video_id: str
    label: str
    clip_prob: float
    frame_prob: float
    matched_gt_indices: tuple[int, ...]


@dataclass(frozen=True)
class LabelData:
    label: str
    windows: tuple[WindowRow, ...]
    gt_counts: dict[str, int]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def parse_exclusions(values: Sequence[str]) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    for value in values:
        video_id, label = value.rsplit(":", 1)
        result.add((video_id.strip(), label.strip()))
    return result


def safe_logit(probability: float) -> float:
    probability = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return math.log(probability / (1.0 - probability))


def metrics_for_mask(data: LabelData, selected: Sequence[bool]) -> dict[str, Any]:
    tp = fp = 0
    matched: dict[str, set[int]] = {video_id: set() for video_id in data.gt_counts}
    for window, keep in zip(data.windows, selected):
        if not keep:
            continue
        if window.matched_gt_indices:
            tp += 1
            matched[window.video_id].update(window.matched_gt_indices)
        else:
            fp += 1
    num_gt = sum(data.gt_counts.values())
    num_matched_gt = sum(len(indices) for indices in matched.values())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = num_matched_gt / num_gt if num_gt else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": num_gt - num_matched_gt,
        "num_pred": tp + fp,
        "num_gt": num_gt,
        "num_matched_gt": num_matched_gt,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def metrics_by_video(
    data: LabelData,
    selected: Sequence[bool],
    *,
    strategy: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for video_id in sorted(data.gt_counts):
        windows = [
            window for window in data.windows if window.video_id == video_id
        ]
        keeps = [
            keep
            for window, keep in zip(data.windows, selected)
            if window.video_id == video_id
        ]
        item = LabelData(data.label, tuple(windows), {video_id: data.gt_counts[video_id]})
        rows.append(
            {
                "strategy": strategy,
                "video_id": video_id,
                "label": data.label,
                **metrics_for_mask(item, keeps),
            }
        )
    return rows


def is_better(candidate: dict[str, Any], best: dict[str, Any] | None) -> bool:
    if best is None:
        return True
    return (
        float(candidate["precision"]),
        float(candidate["recall"]),
        -int(candidate["fp"]),
        float(candidate["f1"]),
    ) > (
        float(best["precision"]),
        float(best["recall"]),
        -int(best["fp"]),
        float(best["f1"]),
    )


def search_threshold(
    data: LabelData,
    scores: Sequence[float],
    *,
    min_matched_gt: int,
    extra_mask: Sequence[bool] | None = None,
) -> tuple[dict[str, Any], list[bool]]:
    ranked = sorted(
        (
            (float(score), index)
            for index, score in enumerate(scores)
            if extra_mask is None or extra_mask[index]
        ),
        reverse=True,
    )
    best: dict[str, Any] | None = None
    tp = fp = 0
    matched: dict[str, set[int]] = {
        video_id: set() for video_id in data.gt_counts
    }
    num_gt = sum(data.gt_counts.values())
    position = 0
    while position < len(ranked):
        threshold = ranked[position][0]
        end = position
        while end < len(ranked) and ranked[end][0] == threshold:
            _, index = ranked[end]
            window = data.windows[index]
            if window.matched_gt_indices:
                tp += 1
                matched[window.video_id].update(window.matched_gt_indices)
            else:
                fp += 1
            end += 1
        num_matched = sum(len(indices) for indices in matched.values())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = num_matched / num_gt if num_gt else 0.0
        metrics = {
            "tp": tp,
            "fp": fp,
            "fn": num_gt - num_matched,
            "num_pred": tp + fp,
            "num_gt": num_gt,
            "num_matched_gt": num_matched,
            "precision": precision,
            "recall": recall,
            "f1": (
                2.0 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            ),
        }
        if int(metrics["num_matched_gt"]) < min_matched_gt:
            position = end
            continue
        candidate = {"threshold": threshold, **metrics}
        if is_better(candidate, best):
            best = candidate
        position = end
    if best is None:
        raise RuntimeError(f"No feasible threshold for label={data.label}")
    best_mask = [
        float(score) >= float(best["threshold"])
        and (extra_mask is None or extra_mask[index])
        for index, score in enumerate(scores)
    ]
    return best, best_mask


def load_data(
    run_dir: Path,
    *,
    labels: Sequence[str],
    tolerance_sec: float,
    exclusions: set[tuple[str, str]],
    requested_video_ids: Sequence[str] | None = None,
    allow_missing: bool = False,
) -> tuple[dict[str, LabelData], list[str]]:
    run_config = json.loads((run_dir / "run_config.json").read_text())
    video_ids = [str(item) for item in run_config["video_ids"]]
    if requested_video_ids is not None:
        requested = {str(item) for item in requested_video_ids}
        video_ids = [video_id for video_id in video_ids if video_id in requested]
    windows_by_label: dict[str, list[WindowRow]] = {label: [] for label in labels}
    gt_counts: dict[str, dict[str, int]] = {label: {} for label in labels}
    loaded_video_ids: list[str] = []
    for video_id in video_ids:
        video_dir = run_dir / video_id
        required = [
            video_dir / "window_predictions.csv",
            video_dir / "frame_event_window_scores.csv",
            video_dir / "gt_events.csv",
        ]
        missing = [path for path in required if not path.exists()]
        if missing and allow_missing:
            continue
        if missing:
            raise FileNotFoundError(f"Missing required eval outputs for video={video_id}: {missing}")
        loaded_video_ids.append(video_id)
        windows = read_csv(video_dir / "window_predictions.csv")
        frame_rows = read_csv(video_dir / "frame_event_window_scores.csv")
        frame_scores = {
            (int(row["window_index"]), row["label"]): float(row["max_frame_prob"])
            for row in frame_rows
            if row.get("branch") == "global"
        }
        gt_rows = read_csv(video_dir / "gt_events.csv")
        gt_by_label = {
            label: sorted(
                float(row["time_sec"])
                for row in gt_rows
                if row["label"] == label
            )
            for label in labels
        }
        for label in labels:
            if (video_id, label) in exclusions:
                continue
            gt_counts[label][video_id] = len(gt_by_label[label])
            for row in windows:
                index = int(row["index"])
                key = (index, label)
                if key not in frame_scores:
                    raise KeyError(f"Missing frame score for {video_id} window={index} label={label}")
                start = float(row["start_sec"]) - tolerance_sec
                end = float(row["end_sec"]) + tolerance_sec
                matches = tuple(
                    gt_index
                    for gt_index, time_sec in enumerate(gt_by_label[label])
                    if start <= time_sec <= end
                )
                windows_by_label[label].append(
                    WindowRow(
                        video_id=video_id,
                        label=label,
                        clip_prob=float(row[f"prob_{label}"]),
                        frame_prob=frame_scores[key],
                        matched_gt_indices=matches,
                    )
                )
    return {
        label: LabelData(label, tuple(windows_by_label[label]), gt_counts[label])
        for label in labels
    }, loaded_video_ids


def load_checkpoint_thresholds(
    run_dir: Path, video_ids: Sequence[str], labels: Sequence[str]
) -> dict[str, float]:
    summary = json.loads((run_dir / video_ids[0] / "summary.json").read_text())
    return {label: float(summary["thresholds"][label]) for label in labels}


def combine(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(int(item["tp"]) for item in items)
    fp = sum(int(item["fp"]) for item in items)
    num_gt = sum(int(item["num_gt"]) for item in items)
    matched = sum(int(item["num_matched_gt"]) for item in items)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = matched / num_gt if num_gt else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": num_gt - matched,
        "num_pred": tp + fp,
        "num_gt": num_gt,
        "num_matched_gt": matched,
        "precision": precision,
        "recall": recall,
        "f1": (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        ),
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure whether saved frame-event scores can raise window-overlap precision without recall loss."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--match-tolerance-sec", type=float, default=5.0)
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--fusion-alphas", default="0.05:0.95:0.05")
    parser.add_argument("--video-ids", default="")
    parser.add_argument("--video-id-file", default="")
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    labels = [item.strip() for item in args.labels.split(",") if item.strip()]
    exclusions = parse_exclusions(args.exclude)
    requested_video_ids: list[str] | None = None
    if args.video_ids or args.video_id_file:
        requested_video_ids = []
        if args.video_ids:
            requested_video_ids.extend(item.strip() for item in args.video_ids.split(",") if item.strip())
        if args.video_id_file:
            for line in Path(args.video_id_file).read_text().splitlines():
                item = line.strip()
                if item and not item.startswith("#"):
                    requested_video_ids.append(item)
    data_by_label, video_ids = load_data(
        run_dir,
        labels=labels,
        tolerance_sec=args.match_tolerance_sec,
        exclusions=exclusions,
        requested_video_ids=requested_video_ids,
        allow_missing=args.allow_missing,
    )
    if not video_ids:
        raise RuntimeError("No completed videos loaded for frame detection analysis")
    checkpoint_thresholds = load_checkpoint_thresholds(run_dir, video_ids, labels)
    start, end, step = (float(item) for item in args.fusion_alphas.split(":"))
    alphas: list[float] = []
    value = start
    while value <= end + 1e-9:
        alphas.append(round(value, 10))
        value += step

    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "protocol": {
            "prediction_postprocess": "window_overlap",
            "match_tolerance_sec": args.match_tolerance_sec,
            "recall_constraint": "num_matched_gt >= checkpoint baseline for every class",
            "excluded_video_label_pairs": [list(item) for item in sorted(exclusions)],
        },
        "checkpoint_thresholds": checkpoint_thresholds,
        "per_class": {},
    }
    detail_rows: list[dict[str, Any]] = []
    strategy_totals: dict[str, list[dict[str, Any]]] = {
        "checkpoint": [],
        "clip_only_oracle": [],
        "frame_hard_gate_oracle": [],
        "frame_logit_fusion_oracle": [],
        "frame_fusion_at_clip_recall_oracle": [],
    }

    for label in labels:
        data = data_by_label[label]
        baseline_mask = [
            window.clip_prob >= checkpoint_thresholds[label]
            for window in data.windows
        ]
        baseline = metrics_for_mask(data, baseline_mask)
        min_matched = int(baseline["num_matched_gt"])

        clip_scores = [window.clip_prob for window in data.windows]
        clip_best, clip_mask = search_threshold(
            data, clip_scores, min_matched_gt=min_matched
        )

        base_clip_mask = baseline_mask
        frame_scores = [window.frame_prob for window in data.windows]
        gate_best, gate_mask = search_threshold(
            data,
            frame_scores,
            min_matched_gt=min_matched,
            extra_mask=base_clip_mask,
        )

        fusion_best: dict[str, Any] | None = None
        fusion_mask: list[bool] = []
        for alpha in alphas:
            fused_scores = [
                (1.0 - alpha) * safe_logit(window.clip_prob)
                + alpha * safe_logit(window.frame_prob)
                for window in data.windows
            ]
            item, mask = search_threshold(
                data, fused_scores, min_matched_gt=min_matched
            )
            item = {"alpha_frame": alpha, **item}
            if is_better(item, fusion_best):
                fusion_best = item
                fusion_mask = mask
        assert fusion_best is not None

        fusion_at_clip_recall: dict[str, Any] | None = None
        fusion_at_clip_mask: list[bool] = []
        for alpha in alphas:
            fused_scores = [
                (1.0 - alpha) * safe_logit(window.clip_prob)
                + alpha * safe_logit(window.frame_prob)
                for window in data.windows
            ]
            item, mask = search_threshold(
                data,
                fused_scores,
                min_matched_gt=int(clip_best["num_matched_gt"]),
            )
            item = {"alpha_frame": alpha, **item}
            if is_better(item, fusion_at_clip_recall):
                fusion_at_clip_recall = item
                fusion_at_clip_mask = mask
        assert fusion_at_clip_recall is not None

        strategies = {
            "checkpoint": (baseline, baseline_mask),
            "clip_only_oracle": (clip_best, clip_mask),
            "frame_hard_gate_oracle": (gate_best, gate_mask),
            "frame_logit_fusion_oracle": (fusion_best, fusion_mask),
            "frame_fusion_at_clip_recall_oracle": (
                fusion_at_clip_recall,
                fusion_at_clip_mask,
            ),
        }
        for strategy, (metrics, mask) in strategies.items():
            strategy_totals[strategy].append(metrics)
            detail_rows.extend(metrics_by_video(data, mask, strategy=strategy))

        summary["per_class"][label] = {
            "checkpoint": baseline,
            "clip_only_oracle": clip_best,
            "frame_hard_gate_oracle": gate_best,
            "frame_logit_fusion_oracle": fusion_best,
            "frame_fusion_at_clip_recall_oracle": fusion_at_clip_recall,
            "frame_gain_over_clip_only_pp": {
                "hard_gate_precision": 100.0
                * (gate_best["precision"] - clip_best["precision"]),
                "logit_fusion_precision": 100.0
                * (fusion_best["precision"] - clip_best["precision"]),
                "hard_gate_recall": 100.0
                * (gate_best["recall"] - clip_best["recall"]),
                "logit_fusion_recall": 100.0
                * (fusion_best["recall"] - clip_best["recall"]),
            },
        }

    summary["micro"] = {
        strategy: combine(items) for strategy, items in strategy_totals.items()
    }
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "frame_precision_no_recall_loss"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    write_csv(output_dir / "per_video_per_class.csv", detail_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
