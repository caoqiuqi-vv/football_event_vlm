#!/usr/bin/env python
"""Multi-protocol dense-window threshold report.

Two families of operating points, both evaluated with the exact window-overlap
protocol (matching dedup + review-interval merging):

1. fixed-threshold rows: same threshold for every label (0.1 ... 0.6) —
   directly shows the model's precision ability without any recall forcing.
2. per-class tuned rows: training-style independent per-label selection
   (pick max precision subject to per-label recall floor), for several floor
   sets — shows how much recall the tuned path buys and at what cost.

Runs in seconds (per-class selection on sampled score axes) instead of the
joint 31^3 grid search (~1-2 h per score prefix).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from select_window_overlap_thresholds_by_budget import evaluate, load_data  # noqa: E402

DEFAULT_FIXED = "0.1,0.2,0.3,0.4,0.5,0.6"
DEFAULT_FLOORS = (
    "shot=0.85,save=0.85,set_piece=0.85;"
    "shot=0.85,save=0.80,set_piece=0.80;"
    "shot=0.80,save=0.75,set_piece=0.75;"
    "shot=0.75,save=0.70,set_piece=0.70"
)


def read_video_ids(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def parse_floor_sets(raw: str, labels: list[str]) -> list[dict[str, float]]:
    floor_sets = []
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            floor_sets.append({label: float(item) for label in labels})
            continue
        floors = {}
        for pair in item.split(","):
            label, value = pair.split("=", 1)
            label = label.strip()
            if label not in labels:
                raise ValueError(f"Unknown label in floors: {label}")
            floors[label] = float(value)
        floor_sets.append(floors)
    return floor_sets


def select_per_class(data, labels, floors, score_prefix: str) -> dict[str, float]:
    """Training-style independent selection on matching-dedup semantics."""
    thresholds: dict[str, float] = {}
    for label in labels:
        scores = []
        for item in data:
            for row in item["rows"]:
                scores.append(float(row.get(f"{score_prefix}_{label}", 0.0) or 0.0))
        values = sorted({s for s in scores if 0.0 <= s <= 1.0}, reverse=True)
        if len(values) > 200:
            idx = sorted({round(i * (len(values) - 1) / 199) for i in range(200)})
            values = [values[i] for i in idx]
        best = None
        for thr in values:
            thr_map = {l: thr if l == label else 2.0 for l in labels}
            result = evaluate(data, labels, thr_map, 5.0, "cap10_peak", 0.0)["per_class"][label]
            if result["recall"] + 1e-12 >= floors[label]:
                cand = (result["precision"], thr)
                if best is None or cand[0] > best[0]:
                    best = cand
        if best is None:
            raise RuntimeError(f"No threshold satisfies recall floor for {label}")
        thresholds[label] = float(best[1])
    return thresholds


def summarize(data, labels, thresholds, tol: float, mode: str) -> dict:
    m = evaluate(data, labels, thresholds, tol, mode, 0.0)
    return {
        "threshold_string": ",".join(f"{l}={thresholds[l]:.4f}" for l in labels),
        "thresholds": thresholds,
        "per_class": {
            l: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in pc.items()}
            for l, pc in m["per_class"].items()
        },
        "micro": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m["micro"].items()},
        "participation_pct": round(m["participation_pct"], 2),
        "candidate_view_minutes": round(m["candidate_view_minutes"], 1),
        "num_candidates": m["num_candidates"],
        "total_video_minutes": round(m["total_video_minutes"], 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--video-id-file", type=Path, required=True)
    parser.add_argument("--labels", default="shot,save,set_piece")
    parser.add_argument("--score-prefix", default="clip_prob")
    parser.add_argument("--fixed-thresholds", default=DEFAULT_FIXED)
    parser.add_argument("--floors", default=DEFAULT_FLOORS)
    parser.add_argument("--match-tolerance-sec", type=float, default=5.0)
    parser.add_argument("--review-mode", default="cap10_peak")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    labels = [item.strip() for item in args.labels.split(",") if item.strip()]
    data = load_data(args.run_dir, read_video_ids(args.video_id_file), labels, args.score_prefix)

    fixed = []
    for value in args.fixed_thresholds.split(","):
        value = value.strip()
        if not value:
            continue
        thr = {label: float(value) for label in labels}
        fixed.append(summarize(data, labels, thr, args.match_tolerance_sec, args.review_mode))

    tuned = []
    for floors in parse_floor_sets(args.floors, labels):
        thr = select_per_class(data, labels, floors, args.score_prefix)
        tuned.append(
            {
                "floors": floors,
                **summarize(data, labels, thr, args.match_tolerance_sec, args.review_mode),
            }
        )

    payload = {
        "protocol": "multi_operating_point_dense_window_v1",
        "run_dir": str(args.run_dir),
        "video_id_file": str(args.video_id_file),
        "labels": labels,
        "score_prefix": args.score_prefix,
        "review_mode": args.review_mode,
        "match_tolerance_sec": args.match_tolerance_sec,
        "fixed_thresholds": fixed,
        "tuned_floors": tuned,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print("=== fixed thresholds (all labels share one threshold) ===")
    for item in fixed:
        print(
            f"thr={item['threshold_string']}  micro P/R/F1="
            f"{item['micro']['precision']}/{item['micro']['recall']}/{item['micro']['f1']}  "
            f"参与度={item['participation_pct']}%  审查={item['candidate_view_minutes']}min"
        )
    print("=== per-class tuned floors ===")
    for item in tuned:
        print(
            f"floors={item['floors']}  thr={item['threshold_string']}  micro P/R/F1="
            f"{item['micro']['precision']}/{item['micro']['recall']}/{item['micro']['f1']}  "
            f"参与度={item['participation_pct']}%  审查={item['candidate_view_minutes']}min"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
