#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

LABELS = ("shot", "save", "set_piece")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def mixed_score(metrics: dict[str, Any]) -> tuple[float, dict[str, float]]:
    tuned = metrics["tuned"]
    per_class = tuned["per_class"]
    shot = per_class["shot"]
    shot_precision = float(shot["precision"])
    shot_recall = float(shot["recall"])
    shot_score = (
        shot_precision
        if shot_recall >= 0.85
        else shot_precision - 1.0 - (0.85 - shot_recall)
    )
    scores = {
        "shot_precision_at_recall85": shot_score,
        "save_f1": float(per_class["save"]["f1"]),
        "set_piece_f1": float(per_class["set_piece"]["f1"]),
    }
    return sum(scores.values()) / len(scores), scores


def metrics_from_report(path: Path) -> tuple[dict[str, Any], str, int | None]:
    payload = load_json(path)
    metrics = payload.get("metrics", payload)
    return metrics, str(payload.get("checkpoint", path)), None


def best_metrics_from_run(path: Path) -> tuple[dict[str, Any], str, int]:
    candidates = []
    for item in sorted(path.glob("metrics_epoch_*.json")):
        payload = load_json(item)
        selection = payload.get("checkpoint_selection", {})
        metrics = payload.get("ema_metrics") or payload.get("metrics")
        if not metrics or metrics.get("selection_metric_scope") != "online_event_pointnms_strict_1to1":
            continue
        candidates.append((float(selection.get("score", float("-inf"))), payload, metrics))
    if not candidates:
        raise RuntimeError(f"no Online Val15 epoch metrics in {path}")
    _, payload, metrics = max(candidates, key=lambda row: row[0])
    return metrics, str(path / "best.pt"), int(payload["epoch"])


def summarize(metrics: dict[str, Any], checkpoint: str, epoch: int | None) -> dict[str, Any]:
    scope = metrics.get("selection_metric_scope")
    if scope != "online_event_pointnms_strict_1to1":
        raise RuntimeError(f"expected strict online event scope, got {scope!r}")
    score, components = mixed_score(metrics)
    event = metrics["online_event"]
    return {
        "checkpoint": checkpoint,
        "epoch": epoch,
        "selection_score": score,
        "selection_components": components,
        "thresholds": metrics["thresholds"],
        "micro_precision": float(event["micro_precision"]),
        "micro_recall": float(event["micro_recall"]),
        "micro_f1": float(event["micro_f1"]),
        "participation_ratio": float(event["nms_capped_participation_ratio"]),
        "participation_minutes": float(event["nms_capped_union_minutes"]),
        "per_class": event["per_class"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    baseline = parser.add_mutually_exclusive_group(required=True)
    baseline.add_argument("--baseline-report", type=Path)
    baseline.add_argument("--baseline-run", type=Path)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-score-gain", type=float, default=0.002)
    parser.add_argument("--max-participation-increase", type=float, default=0.02)
    args = parser.parse_args()

    if args.baseline_report:
        baseline_metrics, baseline_checkpoint, baseline_epoch = metrics_from_report(args.baseline_report)
    else:
        baseline_metrics, baseline_checkpoint, baseline_epoch = best_metrics_from_run(args.baseline_run)
    candidate_metrics, candidate_checkpoint, candidate_epoch = best_metrics_from_run(args.candidate_run)
    base = summarize(baseline_metrics, baseline_checkpoint, baseline_epoch)
    candidate = summarize(candidate_metrics, candidate_checkpoint, candidate_epoch)
    score_gain = candidate["selection_score"] - base["selection_score"]
    participation_increase = candidate["participation_ratio"] - base["participation_ratio"]
    shot_recall = float(candidate["per_class"]["shot"]["recall"])
    reasons = []
    if score_gain < args.min_score_gain:
        reasons.append(f"selection_gain={score_gain:.6f} < {args.min_score_gain:.6f}")
    if participation_increase > args.max_participation_increase:
        reasons.append(
            f"participation_increase={participation_increase:.6f} > "
            f"{args.max_participation_increase:.6f}"
        )
    if shot_recall < 0.85:
        reasons.append(f"shot_recall={shot_recall:.6f} < 0.85")
    result = {
        "passed": not reasons,
        "reasons": reasons,
        "score_gain": score_gain,
        "participation_increase": participation_increase,
        "baseline": base,
        "candidate": candidate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
