#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


def compare_protocol(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    protocol: str,
    labels: Sequence[str],
    recall_tolerance: float,
) -> dict[str, Any]:
    baseline_metrics = baseline["protocols"][protocol]["per_class"]
    candidate_metrics = candidate["protocols"][protocol]["per_class"]
    per_class: dict[str, Any] = {}
    for label in labels:
        base = baseline_metrics[label]
        current = candidate_metrics[label]
        precision_delta = float(current["precision"]) - float(base["precision"])
        recall_delta = float(current["recall"]) - float(base["recall"])
        f1_delta = float(current["f1"]) - float(base["f1"])
        per_class[label] = {
            "baseline_precision": float(base["precision"]),
            "precision": float(current["precision"]),
            "precision_delta": precision_delta,
            "baseline_recall": float(base["recall"]),
            "recall": float(current["recall"]),
            "recall_delta": recall_delta,
            "baseline_f1": float(base["f1"]),
            "f1": float(current["f1"]),
            "f1_delta": f1_delta,
            "recall_guard_pass": recall_delta >= -recall_tolerance,
        }
    precision_delta_mean = sum(item["precision_delta"] for item in per_class.values()) / len(per_class)
    recall_delta_mean = sum(item["recall_delta"] for item in per_class.values()) / len(per_class)
    return {
        "recall_guard_pass": all(item["recall_guard_pass"] for item in per_class.values()),
        "precision_improved": precision_delta_mean > 0.0,
        "precision_delta_mean": precision_delta_mean,
        "recall_delta_mean": recall_delta_mean,
        "per_class": per_class,
    }


def compare_runs(
    baseline: dict[str, Any],
    candidates: Sequence[tuple[str, dict[str, Any]]],
    protocols: Sequence[str],
    labels: Sequence[str],
    recall_tolerance: float,
) -> dict[str, Any]:
    baseline_labels = set(baseline["labels"])
    missing = [label for label in labels if label not in baseline_labels]
    if missing:
        raise ValueError(f"Baseline is missing labels={missing}")
    results: list[dict[str, Any]] = []
    baseline_thresholds = baseline.get("thresholds", {})
    for path, candidate in candidates:
        candidate_thresholds = candidate.get("thresholds", {})
        protocol_results = {
            protocol: compare_protocol(baseline, candidate, protocol, labels, recall_tolerance)
            for protocol in protocols
        }
        results.append(
            {
                "path": path,
                "thresholds": candidate_thresholds,
                "same_thresholds_as_baseline": candidate_thresholds == baseline_thresholds,
                "protocols": protocol_results,
            }
        )
    return {
        "labels": list(labels),
        "protocols": list(protocols),
        "recall_tolerance_pp": recall_tolerance * 100.0,
        "baseline_thresholds": baseline_thresholds,
        "candidates": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare saved football long-video protocol metrics with a precision-first recall guard."
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--protocols", default="point_nms,window_overlap")
    parser.add_argument("--labels", default="shot,save,set_piece")
    parser.add_argument("--recall-tolerance-pp", type=float, default=1.0)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_path = Path(args.baseline)
    baseline = json.loads(baseline_path.read_text())
    candidate_paths = [Path(path) for path in args.candidates]
    candidates = [(str(path), json.loads(path.read_text())) for path in candidate_paths]
    protocols = [item.strip() for item in args.protocols.split(",") if item.strip()]
    labels = [item.strip() for item in args.labels.split(",") if item.strip()]
    result = compare_runs(
        baseline,
        candidates,
        protocols,
        labels,
        args.recall_tolerance_pp / 100.0,
    )
    result["baseline"] = str(baseline_path)
    for candidate in result["candidates"]:
        print(f"candidate={candidate['path']}")
        for protocol in protocols:
            item = candidate["protocols"][protocol]
            print(
                f"  {protocol}: guard={'PASS' if item['recall_guard_pass'] else 'FAIL'} "
                f"mean_dP={item['precision_delta_mean'] * 100:+.2f}pp "
                f"mean_dR={item['recall_delta_mean'] * 100:+.2f}pp"
            )
            for label in labels:
                cls = item["per_class"][label]
                print(
                    f"    {label}: dP={cls['precision_delta'] * 100:+.2f}pp "
                    f"dR={cls['recall_delta'] * 100:+.2f}pp "
                    f"dF1={cls['f1_delta'] * 100:+.2f}pp"
                )
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
