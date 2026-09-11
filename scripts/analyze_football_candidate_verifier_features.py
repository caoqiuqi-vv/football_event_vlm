#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from fit_football_candidate_verifier_from_csv import (
    evaluate,
    fit_oof,
    infer_feature_names,
    load_rows,
    parse_label_floats,
    read_csv,
    select_thresholds,
)


CLIP_FEATURES = {
    "clip_prob",
    "clip_logit",
    "max_other_prob",
    "margin_to_max_other",
    "margin_to_second_class",
    "prob_shot",
    "prob_save",
    "prob_set_piece",
}
FRAME_PREFIXES = (
    "frame_",
    "global_frame_",
    "local_frame_",
    "roi_fused_frame_",
)
ROI_PREFIXES = (
    "crop_",
    "roi_",
)
DUAL_PREFIXES = (
    "global_prob_",
    "local_prob_",
    "roi_gate_",
    "roi_quality_prob_",
    "local_minus_global_prob_",
    "fused_minus_global_prob_",
)


def feature_groups(feature_names: Sequence[str]) -> dict[str, list[str]]:
    features = list(feature_names)
    clip = [name for name in features if name in CLIP_FEATURES]
    frame = [name for name in features if name.startswith(FRAME_PREFIXES)]
    roi = [
        name
        for name in features
        if name.startswith(ROI_PREFIXES)
        and not name.startswith(DUAL_PREFIXES)
        and not name.startswith(FRAME_PREFIXES)
    ]
    dual = [name for name in features if name.startswith(DUAL_PREFIXES)]
    groups = {
        "clip": clip,
        "clip_frame": sorted(set(clip + frame), key=features.index),
        "clip_roi": sorted(set(clip + roi + dual), key=features.index),
        "clip_frame_roi": sorted(set(clip + frame + roi + dual), key=features.index),
        "all": features,
    }
    return {name: values for name, values in groups.items() if values}


def default_thresholds(csv_path: Path, labels: Sequence[str]) -> dict[str, float]:
    rows = read_csv(csv_path)
    result: dict[str, float] = {}
    for label in labels:
        result[label] = next(
            (
                float(row["candidate_threshold"])
                for row in rows
                if row.get("label") == label and row.get("candidate_threshold")
            ),
            0.1,
        )
    return result


def run_group(
    csv_path: Path,
    labels: Sequence[str],
    feature_names: Sequence[str],
    baseline_thresholds: dict[str, float],
    *,
    folds: int,
    regularization_c: float,
    recall_drop_tolerance: float,
    candidates_only: bool,
) -> dict[str, Any]:
    rows, features = load_rows(
        csv_path,
        labels,
        feature_names,
        candidates_only=candidates_only,
    )
    clip_scores = {row.key: row.clip_prob for row in rows}
    baseline = evaluate(rows, clip_scores, baseline_thresholds, labels)
    recall_floors = {
        label: max(
            0.0,
            float(baseline["per_class"][label]["recall"]) - recall_drop_tolerance,
        )
        for label in labels
    }
    oof_scores, _ = fit_oof(
        rows,
        labels,
        folds=folds,
        regularization_c=regularization_c,
    )
    thresholds, selected = select_thresholds(rows, oof_scores, labels, recall_floors)
    verifier = evaluate(rows, oof_scores, thresholds, labels)
    return {
        "features": list(features),
        "num_features": len(features),
        "baseline": baseline,
        "recall_floors": recall_floors,
        "thresholds": thresholds,
        "selected": selected,
        "verifier_oof": verifier,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run feature-group ablations for football candidate verifier CSVs."
    )
    parser.add_argument("--candidate-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--labels", default="shot,save,set_piece")
    parser.add_argument("--groups", default="clip,clip_frame,clip_roi,clip_frame_roi")
    parser.add_argument("--baseline-thresholds", default="")
    parser.add_argument("--recall-drop-tolerance", type=float, default=0.01)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--regularization-c", type=float, default=0.1)
    parser.add_argument("--candidates-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = tuple(label.strip() for label in args.labels.split(",") if label.strip())
    raw_rows = read_csv(args.candidate_csv)
    all_features = infer_feature_names(raw_rows)
    groups = feature_groups(all_features)
    requested_groups = [item.strip() for item in args.groups.split(",") if item.strip()]
    missing = [name for name in requested_groups if name not in groups]
    if missing:
        raise ValueError(f"Unknown or empty feature groups: {missing}; available={sorted(groups)}")
    thresholds = default_thresholds(args.candidate_csv, labels)
    if args.baseline_thresholds:
        thresholds = parse_label_floats(args.baseline_thresholds, labels, 0.1)

    report: dict[str, Any] = {
        "candidate_csv": str(args.candidate_csv),
        "labels": list(labels),
        "all_features": all_features,
        "groups": {},
        "baseline_thresholds": thresholds,
        "recall_drop_tolerance": args.recall_drop_tolerance,
        "candidates_only": bool(args.candidates_only),
    }
    for group_name in requested_groups:
        result = run_group(
            args.candidate_csv,
            labels,
            groups[group_name],
            thresholds,
            folds=args.folds,
            regularization_c=args.regularization_c,
            recall_drop_tolerance=args.recall_drop_tolerance,
            candidates_only=args.candidates_only,
        )
        report["groups"][group_name] = result
        base = result["baseline"]["micro"]
        ver = result["verifier_oof"]["micro"]
        print(
            f"{group_name}: features={result['num_features']} "
            f"baseline P/R={base['precision']:.4f}/{base['recall']:.4f} "
            f"verifier P/R={ver['precision']:.4f}/{ver['recall']:.4f}"
        )
        for label in labels:
            item = result["verifier_oof"]["per_class"][label]
            print(
                f"  {label}: P/R={item['precision']:.4f}/{item['recall']:.4f} "
                f"tp={item['tp']} fp={item['fp']} fn={item['fn']}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
