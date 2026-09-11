#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


PROTOCOL_FILES = {
    "window_overlap": "roi_branch_window_overlap_checkpoint_thr.json",
    "point_nms": "roi_branch_point_nms_checkpoint_thr.json",
}
LABELS = ("micro", "shot", "save", "set_piece")


def parse_spec(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"Expected NAME=RUN_DIR, got: {raw}")
    name, path = raw.split("=", 1)
    if not name.strip() or not path.strip():
        raise ValueError(f"Expected NAME=RUN_DIR, got: {raw}")
    return name.strip(), Path(path).expanduser()


def load_protocol(experiment: str, run_dir: Path, protocol: str) -> dict[str, Any]:
    path = run_dir / PROTOCOL_FILES[protocol]
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    payload = json.loads(path.read_text())
    if payload.get("postprocess") != protocol:
        raise ValueError(
            f"{path} postprocess={payload.get('postprocess')} expected={protocol}"
        )
    return {
        "experiment": experiment,
        "run_dir": str(run_dir),
        "path": str(path),
        "payload": payload,
    }


def summary_by_branch(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["branch"]): item
        for item in payload.get("summaries") or []
    }


def label_metrics(summary: dict[str, Any], label: str) -> dict[str, Any]:
    return summary["micro"] if label == "micro" else summary["per_class"][label]


def metric_rows(loaded: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in loaded:
        payload = item["payload"]
        protocol = str(payload["postprocess"])
        branches = summary_by_branch(payload)
        for branch, summary in branches.items():
            for label in LABELS:
                values = label_metrics(summary, label)
                rows.append(
                    {
                        "experiment": item["experiment"],
                        "protocol": protocol,
                        "branch": branch,
                        "label": label,
                        "precision": values.get("precision", 0.0),
                        "recall": values.get("recall", 0.0),
                        "f1": values.get("f1", 0.0),
                        "tp": values.get("tp", 0),
                        "fp": values.get("fp", 0),
                        "fn": values.get("fn", 0),
                        "num_pred": values.get("num_pred", ""),
                        "num_gt": values.get("num_gt", ""),
                        "num_matched_gt": values.get("num_matched_gt", ""),
                    }
                )
    return rows


def delta_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (row["experiment"], row["protocol"], row["branch"], row["label"]): row
        for row in rows
    }
    deltas: list[dict[str, Any]] = []
    keys = sorted({(row["experiment"], row["protocol"], row["label"]) for row in rows})
    for experiment, protocol, label in keys:
        fused = lookup.get((experiment, protocol, "fused", label))
        if fused is None:
            continue
        for baseline in ("global", "local", "gate_no_confidence"):
            other = lookup.get((experiment, protocol, baseline, label))
            if other is None:
                continue
            deltas.append(
                {
                    "experiment": experiment,
                    "protocol": protocol,
                    "label": label,
                    "comparison": f"fused_minus_{baseline}",
                    "precision_delta": float(fused["precision"]) - float(other["precision"]),
                    "recall_delta": float(fused["recall"]) - float(other["recall"]),
                    "f1_delta": float(fused["f1"]) - float(other["f1"]),
                }
            )
    return deltas


def load_per_video_rows(item: dict[str, Any]) -> list[dict[str, Any]]:
    payload = item["payload"]
    protocol = str(payload["postprocess"])
    prefix = Path(item["path"]).stem
    path = Path(item["run_dir"]) / f"{prefix}_per_video_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return [
        {
            "experiment": item["experiment"],
            "protocol": protocol,
            **row,
        }
        for row in rows
    ]


def comparability(loaded: list[dict[str, Any]]) -> dict[str, Any]:
    reference = loaded[0]
    ref_payload = reference["payload"]
    ref_videos = ref_payload.get("video_ids")
    mismatches: list[dict[str, Any]] = []
    for item in loaded[1:]:
        payload = item["payload"]
        if payload.get("video_ids") != ref_videos:
            mismatches.append(
                {
                    "experiment": item["experiment"],
                    "protocol": payload.get("postprocess"),
                    "field": "video_ids",
                    "reference": ref_videos,
                    "actual": payload.get("video_ids"),
                }
            )
        if float(payload.get("match_tolerance_sec", -1)) != float(
            ref_payload.get("match_tolerance_sec", -1)
        ):
            mismatches.append(
                {
                    "experiment": item["experiment"],
                    "protocol": payload.get("postprocess"),
                    "field": "match_tolerance_sec",
                    "reference": ref_payload.get("match_tolerance_sec"),
                    "actual": payload.get("match_tolerance_sec"),
                }
            )
    return {
        "reference": {
            "experiment": reference["experiment"],
            "protocol": ref_payload.get("postprocess"),
        },
        "comparable": not mismatches,
        "mismatches": mismatches,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pct(value: Any, signed: bool = False) -> str:
    number = 100.0 * float(value)
    return f"{number:+.2f}%" if signed else f"{number:.2f}%"


def build_markdown(
    rows: list[dict[str, Any]],
    deltas: list[dict[str, Any]],
    check: dict[str, Any],
) -> str:
    lines = [
        "# Dynamic ROI Long-Video Summary",
        "",
        f"Evaluation protocol comparable: **{'yes' if check['comparable'] else 'no'}**",
        "",
        "## Fused Results",
        "",
        "| Experiment | Protocol | Label | Precision | Recall | F1 |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row["branch"] != "fused":
            continue
        lines.append(
            f"| {row['experiment']} | {row['protocol']} | {row['label']} | "
            f"{pct(row['precision'])} | {pct(row['recall'])} | {pct(row['f1'])} |"
        )
    lines.extend(
        [
            "",
            "## ROI And Confidence Gain",
            "",
            "| Experiment | Protocol | Label | Comparison | Precision delta | Recall delta | F1 delta |",
            "| --- | --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in deltas:
        if row["comparison"] not in (
            "fused_minus_global",
            "fused_minus_gate_no_confidence",
        ):
            continue
        lines.append(
            f"| {row['experiment']} | {row['protocol']} | {row['label']} | "
            f"{row['comparison']} | {pct(row['precision_delta'], True)} | "
            f"{pct(row['recall_delta'], True)} | {pct(row['f1_delta'], True)} |"
        )
    lines.extend(
        [
            "",
            "## Best Micro-F1 Branch",
            "",
            "| Experiment | Protocol | Branch | Precision | Recall | F1 |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    groups = sorted({(row["experiment"], row["protocol"]) for row in rows})
    for experiment, protocol in groups:
        candidates = [
            row
            for row in rows
            if row["experiment"] == experiment
            and row["protocol"] == protocol
            and row["label"] == "micro"
        ]
        best = max(candidates, key=lambda row: float(row["f1"]))
        lines.append(
            f"| {experiment} | {protocol} | {best['branch']} | "
            f"{pct(best['precision'])} | {pct(best['recall'])} | {pct(best['f1'])} |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Window-overlap, PointNMS, ROI branch, and confidence gains."
    )
    parser.add_argument(
        "--eval-run",
        action="append",
        required=True,
        metavar="NAME=RUN_DIR",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/football_roi_experiments/dynamic_roi_long_eval",
    )
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded: list[dict[str, Any]] = []
    for raw in args.eval_run:
        name, run_dir = parse_spec(raw)
        for protocol in PROTOCOL_FILES:
            try:
                loaded.append(load_protocol(name, run_dir, protocol))
            except FileNotFoundError as exc:
                if not args.allow_missing:
                    raise
                print(f"warning: {exc}", file=sys.stderr)
    if not loaded:
        raise ValueError("No long-video ROI analyses were loaded")

    rows = metric_rows(loaded)
    deltas = delta_rows(rows)
    per_video: list[dict[str, Any]] = []
    for item in loaded:
        per_video.extend(load_per_video_rows(item))
    check = comparability(loaded)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "branch_metrics.csv", rows)
    write_csv(output_dir / "branch_deltas.csv", deltas)
    write_csv(output_dir / "per_video_metrics.csv", per_video)
    (output_dir / "comparison.json").write_text(
        json.dumps(
            {
                "comparability": check,
                "analyses": [
                    {
                        "experiment": item["experiment"],
                        "run_dir": item["run_dir"],
                        "path": item["path"],
                        "protocol": item["payload"]["postprocess"],
                        "thresholds": item["payload"]["thresholds"],
                    }
                    for item in loaded
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    (output_dir / "summary.md").write_text(build_markdown(rows, deltas, check))
    print(f"wrote {output_dir / 'branch_metrics.csv'}")
    print(f"wrote {output_dir / 'branch_deltas.csv'}")
    print(f"wrote {output_dir / 'per_video_metrics.csv'}")
    print(f"wrote {output_dir / 'comparison.json'}")
    print(f"wrote {output_dir / 'summary.md'}")
    if not check["comparable"]:
        print(
            f"warning: found {len(check['mismatches'])} protocol mismatches",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
