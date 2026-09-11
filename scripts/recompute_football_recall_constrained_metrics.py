#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_football_events as football  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute exact precision-at-recall thresholds from saved logits"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label-schema", default="set_piece")
    parser.add_argument(
        "--min-recalls",
        required=True,
        help="Comma-separated floors, e.g. shot=0.8,save=0.79,set_piece=0.68",
    )
    return parser.parse_args()


def parse_min_recalls(raw: str) -> np.ndarray:
    values: dict[str, float] = {}
    for item in raw.split(","):
        label, value = item.split("=", 1)
        values[label.strip()] = float(value)
    missing = [label for label in football.LABELS if label not in values]
    if missing:
        raise ValueError(f"Missing recall floors for labels: {missing}")
    floors = np.asarray([values[label] for label in football.LABELS], dtype=np.float32)
    if np.any((floors < 0.0) | (floors > 1.0)):
        raise ValueError(f"Recall floors must be in [0, 1], got {floors.tolist()}")
    return floors


def recompute_metrics(
    targets: np.ndarray,
    logits: np.ndarray,
    masks: np.ndarray,
    metas: list[dict[str, Any]],
    floors: np.ndarray,
) -> dict[str, Any]:
    probs = 1.0 / (1.0 + np.exp(-logits))
    thresholds = football.tune_thresholds(
        targets, probs, masks, min_recalls=floors
    )
    return {
        "tuned": football.safe_metrics(targets, probs, thresholds, masks),
        "thresholds": {
            label: float(thresholds[index])
            for index, label in enumerate(football.LABELS)
        },
        "per_video_tuned": football.per_video_metrics(
            targets, probs, masks, metas, thresholds
        ),
    }


def main() -> None:
    args = parse_args()
    football.configure_label_schema(
        football.to_config({"task": {"label_schema": args.label_schema}})
    )
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    raw = payload.get("raw_outputs")
    if not isinstance(raw, dict):
        raise ValueError("Input metrics do not contain raw_outputs")

    targets = np.asarray(raw["targets"], dtype=np.int32)
    masks = np.asarray(raw["masks"], dtype=np.float32)
    metas = raw["meta"]
    floors = parse_min_recalls(args.min_recalls)
    payload.update(
        recompute_metrics(
            targets,
            np.asarray(raw["logits"], dtype=np.float32),
            masks,
            metas,
            floors,
        )
    )

    branch_logits = raw.get("temporal_branch_logits", {})
    for branch_name, logits in branch_logits.items():
        payload.setdefault("temporal_branches", {}).setdefault(
            branch_name, {}
        ).update(
            recompute_metrics(
                targets,
                np.asarray(logits, dtype=np.float32),
                masks,
                metas,
                floors,
            )
        )

    payload["tuned_min_recall"] = {
        label: float(floors[index]) for index, label in enumerate(football.LABELS)
    }
    payload["threshold_search"] = "exact_unique_scores"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "thresholds": payload["thresholds"],
                "event_thresholds": payload.get("temporal_branches", {})
                .get("event", {})
                .get("thresholds"),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
