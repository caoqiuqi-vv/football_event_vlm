#!/usr/bin/env python
"""Re-evaluate a saved test18 dense run against immutable repaired labels.

The script never modifies the source inference run or the repaired-label export.  It
creates a derived run whose prediction files are symlinks and whose GT CSV files are
a documented coarse-label projection of the final QC export.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from eval_long_video_checkpoint import (  # noqa: E402
    compute_event_metrics,
    point_nms_predictions,
    window_overlap_predictions,
)
from recompute_football_eval_protocols import (  # noqa: E402
    finalize_totals,
    init_totals,
    load_gt_events,
    load_run_metadata,
    normalize_score_columns,
    read_csv,
)

MODEL_LABELS = ("shot", "save", "set_piece")
SET_PIECE_LABELS = {"corner", "free_kick", "kickoff", "set_piece", "penalty"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--final-labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--nms-radius-sec", type=float, default=5.0)
    parser.add_argument("--tolerances", default="3,5")
    return parser.parse_args()


def project_label(event: dict[str, Any]) -> str | None:
    label = str(event.get("semantic_label") or event.get("label") or "").strip()
    if label in {"shot", "save"}:
        return label
    if label in SET_PIECE_LABELS:
        return "set_piece"
    return None


def event_identifier(video_id: str, event: dict[str, Any], index: int) -> str:
    return str(event.get("source_id") or event.get("id") or f"{video_id}_final_{index:05d}")


def project_final_labels(final_labels_path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    payload = json.loads(final_labels_path.read_text())
    videos = payload["videos"]
    if not isinstance(videos, dict):
        raise TypeError("final_labels.videos must be an object keyed by video id")

    projected: dict[str, list[dict[str, Any]]] = {}
    collapsed: list[dict[str, Any]] = []
    excluded = Counter()
    source_semantics = Counter()
    output_counts = Counter()

    for video_id, video_payload in sorted(videos.items()):
        output: list[dict[str, Any]] = []
        exact_parent_groups: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
        for index, event in enumerate(video_payload.get("events", [])):
            semantic = str(event.get("semantic_label") or event.get("label") or "").strip()
            source_semantics[semantic] += 1
            parent = project_label(event)
            if parent is None:
                excluded[semantic] += 1
                continue
            time_sec = float(event["time_sec"])
            case_id = str(event.get("case_id") or "")
            # Only exact duplicates created by coarse-label projection are eligible
            # for collapse. Nearby events remain independent GT instances.
            key = (case_id, parent, round(time_sec, 3))
            exact_parent_groups.setdefault(key, []).append(
                {
                    "label": parent,
                    "time_sec": time_sec,
                    "start_sec": time_sec,
                    "end_sec": time_sec,
                    "raw_label": semantic,
                    "event_type": semantic,
                    "event_id": event_identifier(video_id, event, index),
                    "num_events": 1,
                    "event_ids": event_identifier(video_id, event, index),
                    "case_id": case_id,
                    "lineage_gt_ids": event.get("lineage_gt_ids", []),
                }
            )

        for key, items in exact_parent_groups.items():
            if len(items) == 1:
                output.append(items[0])
                continue
            # Prefer a specific restart subtype over a generic set_piece record.
            selected = sorted(items, key=lambda item: item["raw_label"] == "set_piece")[0]
            output.append(selected)
            collapsed.append(
                {
                    "video_id": video_id,
                    "case_id": key[0],
                    "parent_label": key[1],
                    "time_sec": key[2],
                    "kept": selected["event_id"],
                    "collapsed": [item["event_id"] for item in items if item is not selected],
                    "source_semantics": [item["raw_label"] for item in items],
                }
            )
        output.sort(key=lambda item: (item["time_sec"], item["label"], item["event_id"]))
        projected[video_id] = output
        output_counts.update(item["label"] for item in output)

    audit = {
        "source": str(final_labels_path.resolve()),
        "source_created_at": payload.get("created_at"),
        "source_summary": payload.get("summary"),
        "projection": {
            "shot": ["shot"],
            "save": ["save"],
            "set_piece": sorted(SET_PIECE_LABELS),
            "excluded": "all other semantic labels",
        },
        "rules": {
            "temporal_nms_on_gt": False,
            "collapse_only_exact_same_case_parent_and_millisecond": True,
        },
        "source_semantic_counts": dict(sorted(source_semantics.items())),
        "excluded_counts": dict(sorted(excluded.items())),
        "projected_counts": dict(output_counts),
        "collapsed_exact_projection_duplicates": collapsed,
    }
    return projected, audit


def write_projected_run(
    source_run: Path,
    output_dir: Path,
    projected: dict[str, list[dict[str, Any]]],
    audit: dict[str, Any],
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_video_ids = sorted(
        path.name
        for path in source_run.iterdir()
        if path.is_dir() and (path / "window_predictions.csv").is_file()
    )
    if set(source_video_ids) != set(projected):
        raise ValueError(
            f"Video mismatch: predictions_only={sorted(set(source_video_ids) - set(projected))}, "
            f"labels_only={sorted(set(projected) - set(source_video_ids))}"
        )

    fields = [
        "label", "time_sec", "start_sec", "end_sec", "raw_label", "event_type",
        "event_id", "num_events", "event_ids", "case_id", "lineage_gt_ids",
    ]
    for video_id in source_video_ids:
        source_video = source_run / video_id
        target_video = output_dir / video_id
        target_video.mkdir(parents=True, exist_ok=True)
        for name in ("window_predictions.csv", "summary.json"):
            link = target_video / name
            target = source_video / name
            if link.is_symlink() and link.resolve() == target.resolve():
                pass
            elif link.exists() or link.is_symlink():
                raise FileExistsError(f"Refusing to replace existing path: {link}")
            else:
                link.symlink_to(os.path.relpath(target, target_video))
        with (target_video / "gt_events.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for event in projected[video_id]:
                row = dict(event)
                row["lineage_gt_ids"] = json.dumps(row["lineage_gt_ids"], ensure_ascii=False)
                writer.writerow(row)
        (target_video / "gt_events.json").write_text(
            json.dumps(projected[video_id], ensure_ascii=False, indent=2) + "\n"
        )

    (output_dir / "gt_projection_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
    )
    return source_video_ids


def raw_point_predictions(
    rows: list[dict[str, Any]], labels: Sequence[str], thresholds: dict[str, float]
) -> list[dict[str, Any]]:
    predictions = window_overlap_predictions(rows, labels, thresholds)
    for item in predictions:
        midpoint = (float(item["start_sec"]) + float(item["end_sec"])) * 0.5
        item["time_sec"] = midpoint
        item["start_sec"] = midpoint
        item["end_sec"] = midpoint
    return predictions


def evaluate(
    run_dir: Path,
    video_ids: Sequence[str],
    labels: Sequence[str],
    thresholds: dict[str, float],
    tolerance_sec: float,
    nms_radius_sec: float,
) -> dict[str, Any]:
    protocols = ("raw_no_nms", "point_nms5", "window_coverage")
    totals = {name: init_totals(labels) for name in protocols}
    per_video: list[dict[str, Any]] = []
    matches: dict[str, list[dict[str, Any]]] = {name: [] for name in protocols}

    for video_id in video_ids:
        video_dir = run_dir / video_id
        rows = normalize_score_columns(read_csv(video_dir / "window_predictions.csv"), labels, "prob")
        gt = load_gt_events(video_dir / "gt_events.csv", labels)
        predictions = {
            "raw_no_nms": raw_point_predictions(rows, labels, thresholds),
            "point_nms5": point_nms_predictions(rows, labels, thresholds, nms_radius_sec),
            "window_coverage": window_overlap_predictions(rows, labels, thresholds),
        }
        for protocol, preds in predictions.items():
            allow_many = protocol == "window_coverage"
            metrics = compute_event_metrics(
                preds,
                gt,
                tolerance_sec,
                matching_mode="window" if allow_many else "point",
                allow_many_predictions_per_gt=allow_many,
            )
            matches[protocol].extend({"video_id": video_id, **item} for item in metrics["matches"])
            for label in labels:
                item = metrics["per_class"][label]
                for key in totals[protocol][label]:
                    totals[protocol][label][key] += int(item[key])
                per_video.append(
                    {"video_id": video_id, "protocol": protocol, "label": label, **item}
                )

    result_protocols = {}
    for protocol in protocols:
        allow_many = protocol == "window_coverage"
        per_class, micro = finalize_totals(
            totals[protocol], allow_many_predictions_per_gt=allow_many
        )
        result_protocols[protocol] = {"per_class": per_class, "micro": micro}
    return {"protocols": result_protocols, "per_video": per_video, "matches": matches}


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# test18 dense re-evaluation after final annotation QC",
        "",
        f"- Checkpoint: `{payload['checkpoint']}`",
        f"- Repaired GT: `{payload['repaired_gt_source']}`",
        f"- Frozen thresholds: `{payload['thresholds']}`",
        "- GT temporal NMS: **disabled**",
        "- Point matching: score-ordered, same-class, one-to-one",
        "",
    ]
    for tolerance, comparison in payload["comparisons"].items():
        lines.extend([f"## ±{tolerance}s", ""])
        for protocol in ("point_nms5", "raw_no_nms", "window_coverage"):
            lines.extend([f"### {protocol}", ""])
            lines.append("| GT | Class | Precision old→new | Recall old→new | TP old→new | FP old→new |")
            lines.append("|---|---|---:|---:|---:|---:|")
            old = comparison["old_gt"]["protocols"][protocol]["per_class"]
            new = comparison["repaired_gt"]["protocols"][protocol]["per_class"]
            for label in MODEL_LABELS:
                lines.append(
                    f"| {old[label]['num_gt']}→{new[label]['num_gt']} | {label} | "
                    f"{old[label]['precision']:.4f}→{new[label]['precision']:.4f} | "
                    f"{old[label]['recall']:.4f}→{new[label]['recall']:.4f} | "
                    f"{old[label]['tp']}→{new[label]['tp']} | {old[label]['fp']}→{new[label]['fp']} |"
                )
            lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    source_run = Path(args.source_run).resolve()
    final_labels = Path(args.final_labels).resolve()
    output_dir = Path(args.output_dir).resolve()
    tolerances = [float(item) for item in args.tolerances.split(",") if item.strip()]

    projected, audit = project_final_labels(final_labels)
    video_ids = write_projected_run(source_run, output_dir, projected, audit)
    labels, thresholds = load_run_metadata(source_run, video_ids)
    if tuple(labels) != MODEL_LABELS:
        raise ValueError(f"Expected model labels {MODEL_LABELS}, got {labels}")

    comparisons: dict[str, Any] = {}
    for tolerance in tolerances:
        key = f"{tolerance:g}"
        comparisons[key] = {
            "old_gt": evaluate(source_run, video_ids, labels, thresholds, tolerance, args.nms_radius_sec),
            "repaired_gt": evaluate(output_dir, video_ids, labels, thresholds, tolerance, args.nms_radius_sec),
        }

    first_summary = json.loads((source_run / video_ids[0] / "summary.json").read_text())
    payload = {
        "checkpoint": first_summary.get("checkpoint"),
        "source_run": str(source_run),
        "derived_run": str(output_dir),
        "repaired_gt_source": str(final_labels),
        "video_ids": video_ids,
        "labels": labels,
        "thresholds": thresholds,
        "nms_radius_sec": args.nms_radius_sec,
        "gt_projection": audit,
        "comparisons": comparisons,
    }
    (output_dir / "dense_re_evaluation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    (output_dir / "dense_re_evaluation.md").write_text(markdown_report(payload))
    print(json.dumps({
        "output_dir": str(output_dir),
        "projected_counts": audit["projected_counts"],
        "collapsed": len(audit["collapsed_exact_projection_duplicates"]),
        "thresholds": thresholds,
        "comparisons": {
            tol: {
                protocol: values["repaired_gt"]["protocols"][protocol]
                for protocol in ("point_nms5", "raw_no_nms", "window_coverage")
            }
            for tol, values in comparisons.items()
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
