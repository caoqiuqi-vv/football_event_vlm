from __future__ import annotations

"""Select an A1 epoch without treating unsupported calibration classes as failures."""

import argparse
import json
import math
from pathlib import Path


LABELS = ("shot", "save", "corner", "freekick", "penalty")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--shot-recall", type=float, default=0.90)
    parser.add_argument("--other-recall", type=float, default=0.85)
    args = parser.parse_args()
    experiment = Path(args.experiment).expanduser().resolve()
    rows = []
    for metrics_path in sorted(experiment.glob("metrics_epoch_*.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        per_class = payload["metrics"]["tuned"]["per_class"]
        supported = [label for label in LABELS if int(per_class[label]["support"]) > 0]
        unsupported = [label for label in LABELS if label not in supported]
        targets = {
            label: args.shot_recall if label == "shot" else args.other_recall
            for label in supported
        }
        recalls = {label: float(per_class[label]["recall"]) for label in supported}
        precisions = {label: float(per_class[label]["precision"]) for label in supported}
        aps = {label: float(per_class[label]["ap"]) for label in supported}
        achieved = {label: recalls[label] >= targets[label] for label in supported}
        deficits = {label: max(targets[label] - recalls[label], 0.0) for label in supported}
        other_supported = [label for label in supported if label != "shot"]
        selection = (
            int(achieved.get("shot", False)),
            sum(int(achieved[label]) for label in other_supported),
            -sum(deficits.values()),
            sum(precisions.values()) / max(len(precisions), 1),
            sum(aps.values()) / max(len(aps), 1),
        )
        epoch = int(payload["epoch"])
        checkpoint = experiment / f"epoch_{epoch}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing checkpoint for epoch {epoch}: {checkpoint}")
        rows.append({
            "epoch": epoch,
            "checkpoint": str(checkpoint),
            "selection_tuple": selection,
            "supported_labels": supported,
            "unsupported_labels": unsupported,
            "recall": recalls,
            "precision": precisions,
            "average_precision": aps,
            "target_achieved": achieved,
            "recall_deficit": deficits,
        })
    if not rows:
        raise RuntimeError(f"no completed A1 metrics under {experiment}")
    best = max(rows, key=lambda row: tuple(row["selection_tuple"]))
    report = {
        "schema": "football_longform_v2.a1_supported_checkpoint_selection.v1",
        "selection_rule": (
            "shot_target_first_then_other_targets_then_recall_deficit_"
            "then_precision_then_ap; support=0 is N/A"
        ),
        "best": best,
        "epochs": rows,
    }
    output = experiment / "supported_checkpoint_selection.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output), "best_epoch": best["epoch"],
        "checkpoint": best["checkpoint"], "selection_tuple": best["selection_tuple"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
