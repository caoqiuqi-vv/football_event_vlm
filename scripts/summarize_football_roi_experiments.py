#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml


LABELS = ("shot", "save", "set_piece")
MODES = ("default", "tuned")


def nested(data: dict[str, Any], path: str, default: Any = None) -> Any:
    value: Any = data
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def parse_experiment(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"Expected NAME=OUTPUT_DIR, got: {raw}")
    name, path = raw.split("=", 1)
    if not name.strip() or not path.strip():
        raise ValueError(f"Expected NAME=OUTPUT_DIR, got: {raw}")
    return name.strip(), Path(path).expanduser()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_experiment(name: str, root: Path) -> dict[str, Any]:
    config_path = root / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing {config_path}")
    config = yaml.safe_load(config_path.read_text()) or {}
    epoch_paths = sorted(root.glob("metrics_epoch_*.json"))
    epochs = [load_json(path) for path in epoch_paths]
    best_path = root / "best_metrics.json"
    best = load_json(best_path) if best_path.exists() else None
    return {
        "name": name,
        "root": str(root),
        "config": config,
        "epochs": epochs,
        "best": best,
    }


def runtime_summary(config: dict[str, Any]) -> dict[str, Any]:
    world_size = max(len(config.get("gpu_ids") or []), 1)
    per_gpu_batch = int(nested(config, "train.per_gpu_batch_size", 0) or 0)
    grad_accum = int(nested(config, "train.grad_accum_steps", 1) or 1)
    lr_per_gpu = float(nested(config, "train.lr_per_gpu", 0.0) or 0.0)
    return {
        "world_size": world_size,
        "per_gpu_batch_size": per_gpu_batch,
        "global_batch_size": per_gpu_batch * world_size,
        "grad_accum_steps": grad_accum,
        "effective_batch_size": per_gpu_batch * world_size * grad_accum,
        "lr_per_gpu": lr_per_gpu,
        "global_lr": lr_per_gpu * world_size,
        "workers_per_gpu": int(nested(config, "data.num_workers_per_gpu", 0) or 0),
        "prefetch_factor": int(nested(config, "data.prefetch_factor", 0) or 0),
    }


def config_fingerprint(config: dict[str, Any]) -> dict[str, Any]:
    runtime = runtime_summary(config)
    fields = {
        "seed": config.get("seed"),
        "deterministic": config.get("deterministic"),
        "label_schema": nested(config, "task.label_schema"),
        "train_split": nested(config, "data.long_video.split_files.train"),
        "val_split": nested(config, "data.long_video.split_files.val"),
        "negative_ratio_train": nested(config, "data.long_video.negative_ratio_by_split.train"),
        "negative_ratio_val": nested(config, "data.long_video.negative_ratio_by_split.val"),
        "num_frames": nested(config, "video.num_frames"),
        "clip_duration": nested(config, "video.clip_duration"),
        "temporal_jitter_sec": nested(config, "video.temporal_jitter_sec"),
        "local_image_size": nested(config, "video.image_size"),
        "global_image_size": nested(config, "spatial_crop.global_image_size"),
        "init_checkpoint": nested(config, "model.init_checkpoint"),
        "backbone": nested(config, "model.backbone"),
        "freeze_backbone": nested(config, "model.freeze_backbone"),
        "freeze_loaded_backbone": nested(config, "model.freeze_loaded_backbone"),
        "temporal_fusion": nested(config, "model.temporal_fusion"),
        "hidden_dim": nested(config, "model.hidden_dim"),
        "temporal_layers": nested(config, "model.temporal_layers"),
        "temporal_heads": nested(config, "model.temporal_heads"),
        "pos_weight": nested(config, "train.pos_weight"),
        "frame_det_loss_weight": nested(config, "train.frame_det_loss_weight"),
        "effective_batch_size": runtime["effective_batch_size"],
        "global_lr": runtime["global_lr"],
        "amp_dtype": nested(config, "train.amp_dtype"),
    }
    treatments = {
        "roi_temporal_mode": nested(config, "spatial_crop.temporal_mode"),
        "dynamic_context_sec": nested(config, "spatial_crop.dynamic_context_sec"),
        "smoothing_window_sec": nested(config, "spatial_crop.temporal_smoothing_window_sec"),
        "max_hold_sec": nested(config, "spatial_crop.temporal_max_hold_sec"),
        "confidence_decay_sec": nested(config, "spatial_crop.temporal_confidence_decay_sec"),
        "view_fusion": nested(config, "model.view_fusion"),
        "roi_quality_loss_weight": nested(config, "train.roi_quality_loss_weight", 0.0),
    }
    return {"controlled": fields, "treatments": treatments, "runtime": runtime}


def compare_fingerprints(experiments: list[dict[str, Any]]) -> dict[str, Any]:
    fingerprints = {
        experiment["name"]: config_fingerprint(experiment["config"])
        for experiment in experiments
    }
    reference_name = experiments[0]["name"]
    reference = fingerprints[reference_name]["controlled"]
    mismatches: list[dict[str, Any]] = []
    for name, fingerprint in fingerprints.items():
        if name == reference_name:
            continue
        for field, expected in reference.items():
            actual = fingerprint["controlled"].get(field)
            if actual != expected:
                mismatches.append(
                    {
                        "experiment": name,
                        "field": field,
                        "reference": expected,
                        "actual": actual,
                    }
                )
    return {
        "reference": reference_name,
        "comparable": not mismatches,
        "controlled_mismatches": mismatches,
        "fingerprints": fingerprints,
    }


def metric_rows(
    experiment: str,
    source: str,
    payload: dict[str, Any],
) -> Iterable[dict[str, Any]]:
    epoch = int(payload.get("epoch", 0) or 0)
    selection = payload.get("checkpoint_selection") or {}
    metrics = payload.get("metrics") or {}
    for mode in MODES:
        values = metrics.get(mode) or {}
        aggregate = {
            "experiment": experiment,
            "source": source,
            "epoch": epoch,
            "selection_mode": selection.get("mode", ""),
            "selection_score": selection.get("score", ""),
            "train_loss": payload.get("train_loss", ""),
            "mode": mode,
            "label": "macro",
            "precision": values.get("macro_precision", ""),
            "recall": values.get("macro_recall", ""),
            "f1": values.get("macro_f1", ""),
            "ap": values.get("mAP", ""),
            "auroc": values.get("mAUROC", ""),
            "threshold": "",
            "support": "",
            "tp": "",
            "fp": "",
            "fn": "",
        }
        yield aggregate
        yield {
            **aggregate,
            "label": "micro",
            "precision": values.get("micro_precision", ""),
            "recall": values.get("micro_recall", ""),
            "f1": values.get("micro_f1", ""),
        }
        per_class = values.get("per_class") or {}
        for label in LABELS:
            item = per_class.get(label) or {}
            yield {
                **aggregate,
                "label": label,
                "precision": item.get("precision", ""),
                "recall": item.get("recall", ""),
                "f1": item.get("f1", ""),
                "ap": item.get("ap", ""),
                "auroc": item.get("auroc", ""),
                "threshold": item.get("threshold", ""),
                "support": item.get("support", ""),
                "tp": item.get("tp", ""),
                "fp": item.get("fp", ""),
                "fn": item.get("fn", ""),
            }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percent(value: Any) -> str:
    return "-" if value in ("", None) else f"{100.0 * float(value):.2f}%"


def build_markdown(
    experiments: list[dict[str, Any]],
    comparison: dict[str, Any],
    rows: list[dict[str, Any]],
) -> str:
    lines = ["# Dynamic ROI Experiment Summary", ""]
    lines.append(
        f"Controlled configuration comparable: **{'yes' if comparison['comparable'] else 'no'}**"
    )
    if comparison["controlled_mismatches"]:
        lines.extend(["", "## Controlled Mismatches", "", "| Experiment | Field | Reference | Actual |", "| --- | --- | --- | --- |"])
        for item in comparison["controlled_mismatches"]:
            lines.append(
                f"| {item['experiment']} | `{item['field']}` | `{item['reference']}` | `{item['actual']}` |"
            )
    lines.extend(
        [
            "",
            "## Treatments",
            "",
            "| Experiment | ROI temporal mode | View fusion | Context | Smoothing | Quality loss |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for experiment in experiments:
        treatment = comparison["fingerprints"][experiment["name"]]["treatments"]
        lines.append(
            f"| {experiment['name']} | {treatment['roi_temporal_mode']} | "
            f"{treatment['view_fusion']} | {treatment['dynamic_context_sec']} | "
            f"{treatment['smoothing_window_sec']} | {treatment['roi_quality_loss_weight']} |"
        )
    best_rows = [
        row
        for row in rows
        if row["source"] == "best" and row["mode"] == "tuned" and row["label"] in (*LABELS, "micro")
    ]
    lines.extend(
        [
            "",
            "## Best Tuned Metrics",
            "",
            "| Experiment | Epoch | Label | Precision | Recall | F1 | AP | Threshold |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in best_rows:
        lines.append(
            f"| {row['experiment']} | {row['epoch']} | {row['label']} | "
            f"{percent(row['precision'])} | {percent(row['recall'])} | {percent(row['f1'])} | "
            f"{percent(row['ap'])} | {row['threshold'] or '-'} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize matched fixed/dynamic ROI experiments.")
    parser.add_argument(
        "--experiment",
        action="append",
        required=True,
        metavar="NAME=OUTPUT_DIR",
        help="Repeat for D0, D1, and D2.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/football_roi_experiments/dynamic_roi_comparison",
    )
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiments: list[dict[str, Any]] = []
    for raw in args.experiment:
        name, root = parse_experiment(raw)
        try:
            experiments.append(load_experiment(name, root))
        except FileNotFoundError as exc:
            if not args.allow_missing:
                raise
            print(f"warning: {exc}", file=sys.stderr)
    if not experiments:
        raise ValueError("No experiment outputs were loaded")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = compare_fingerprints(experiments)
    rows: list[dict[str, Any]] = []
    for experiment in experiments:
        for payload in experiment["epochs"]:
            rows.extend(metric_rows(experiment["name"], "epoch", payload))
        if experiment["best"] is not None:
            rows.extend(metric_rows(experiment["name"], "best", experiment["best"]))

    report = {
        "experiments": [
            {
                "name": experiment["name"],
                "root": experiment["root"],
                "num_epochs": len(experiment["epochs"]),
                "best_epoch": (
                    int(experiment["best"].get("epoch", 0))
                    if experiment["best"] is not None
                    else None
                ),
            }
            for experiment in experiments
        ],
        "comparability": comparison,
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    write_csv(output_dir / "metrics_long.csv", rows)
    (output_dir / "summary.md").write_text(build_markdown(experiments, comparison, rows))
    print(f"wrote {output_dir / 'comparison.json'}")
    print(f"wrote {output_dir / 'metrics_long.csv'}")
    print(f"wrote {output_dir / 'summary.md'}")
    if comparison["controlled_mismatches"]:
        print(
            f"warning: found {len(comparison['controlled_mismatches'])} controlled mismatches",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
