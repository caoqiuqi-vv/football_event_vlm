#!/usr/bin/env python3
"""Summarize an A0 shared-gradient probe and select a fixed auxiliary weight."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("probe_json", type=Path)
    parser.add_argument("--target-ratio", type=float, default=0.20)
    parser.add_argument("--min-weight", type=float, default=0.02)
    parser.add_argument("--max-weight", type=float, default=1.0)
    parser.add_argument("--round-to", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.probe_json.read_text())
    measurements = payload.get("measurements", [])
    if not measurements:
        raise ValueError("probe contains no gradient measurements")

    raw_ratios = [
        float(row["mechanism_a_object_to_event_grad_ratio"])
        for row in measurements
    ]
    cosines = [float(row["mechanism_a_grad_cosine"]) for row in measurements]
    median_ratio = statistics.median(raw_ratios)
    if median_ratio <= 0:
        raise ValueError("median raw object/event gradient ratio must be positive")
    formula_weight = min(
        max(args.target_ratio / median_ratio, args.min_weight), args.max_weight
    )
    if args.round_to > 0:
        selected_weight = round(formula_weight / args.round_to) * args.round_to
        selected_weight = min(max(selected_weight, args.min_weight), args.max_weight)
    else:
        selected_weight = formula_weight
    selected_weighted_ratios = [selected_weight * value for value in raw_ratios]

    summary = {
        "source": str(args.probe_json),
        "num_measurements": len(measurements),
        "median_raw_object_to_event_grad_ratio": median_ratio,
        "median_gradient_cosine": statistics.median(cosines),
        "target_weighted_ratio": args.target_ratio,
        "formula_weight": formula_weight,
        "selected_weight": selected_weight,
        "selected_median_weighted_ratio": statistics.median(
            selected_weighted_ratios
        ),
        "selected_max_weighted_ratio": max(selected_weighted_ratios),
        "negative_conflict_gate": statistics.median(cosines) < -0.20,
        "all_shared_lora_gradients_nonzero": all(
            float(row["mechanism_a_event_grad_norm"]) > 0
            and float(row["mechanism_a_object_grad_norm"]) > 0
            and float(row["mechanism_a_shared_gradient_tensors"]) > 0
            for row in measurements
        ),
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
