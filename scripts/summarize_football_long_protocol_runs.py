#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

LABELS = ("shot", "save", "set_piece")
PROTOCOLS = ("window_overlap", "point_nms")


def parse_named_path(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"Expected NAME=RUN_DIR, got: {raw}")
    name, value = raw.split("=", 1)
    name = name.strip()
    value = value.strip()
    if not name or not value:
        raise ValueError(f"Expected NAME=RUN_DIR, got: {raw}")
    return name, Path(value).expanduser()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def protocol_path(run_dir: Path, prefix: str) -> Path:
    return run_dir / f"{prefix}.json"


def metric_rows(name: str, run_dir: Path, payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    protocols = payload.get("protocols") or {}
    for protocol in PROTOCOLS:
        metrics = protocols.get(protocol) or {}
        per_class = metrics.get("per_class") or {}
        micro = metrics.get("micro") or {}
        if micro:
            rows.append(
                {
                    "experiment": name,
                    "run_dir": str(run_dir),
                    "protocol": protocol,
                    "label": "micro",
                    "precision": micro.get("precision", ""),
                    "recall": micro.get("recall", ""),
                    "f1": micro.get("f1", ""),
                    "tp": micro.get("tp", ""),
                    "fp": micro.get("fp", ""),
                    "fn": micro.get("fn", ""),
                }
            )
        for label in LABELS:
            item = per_class.get(label) or {}
            rows.append(
                {
                    "experiment": name,
                    "run_dir": str(run_dir),
                    "protocol": protocol,
                    "label": label,
                    "precision": item.get("precision", ""),
                    "recall": item.get("recall", ""),
                    "f1": item.get("f1", ""),
                    "tp": item.get("tp", ""),
                    "fp": item.get("fp", ""),
                    "fn": item.get("fn", ""),
                }
            )
    return rows


def compare_rows(name: str, compare_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidates = compare_payload.get("candidates") or []
    if not candidates:
        return rows
    candidate = candidates[0]
    for protocol, values in (candidate.get("protocols") or {}).items():
        rows.append(
            {
                "experiment": name,
                "protocol": protocol,
                "label": "mean",
                "recall_guard_pass": values.get("recall_guard_pass", ""),
                "precision_delta_pp": 100.0 * float(values.get("precision_delta_mean", 0.0)),
                "recall_delta_pp": 100.0 * float(values.get("recall_delta_mean", 0.0)),
                "f1_delta_pp": "",
            }
        )
        for label in LABELS:
            item = (values.get("per_class") or {}).get(label) or {}
            rows.append(
                {
                    "experiment": name,
                    "protocol": protocol,
                    "label": label,
                    "recall_guard_pass": item.get("recall_guard_pass", ""),
                    "precision_delta_pp": 100.0 * float(item.get("precision_delta", 0.0)),
                    "recall_delta_pp": 100.0 * float(item.get("recall_delta", 0.0)),
                    "f1_delta_pp": 100.0 * float(item.get("f1_delta", 0.0)),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pct(value: Any, signed: bool = False) -> str:
    if value in (None, ""):
        return "-"
    number = float(value)
    sign = "+" if signed and number >= 0 else ""
    return f"{sign}{number:.2f}pp" if signed else f"{100.0 * number:.2f}%"


def build_markdown(metric_rows_: list[dict[str, Any]], delta_rows: list[dict[str, Any]]) -> str:
    lines = ["# Football Long-Video Protocol Summary", ""]
    window_rows = [
        row for row in metric_rows_
        if row["protocol"] == "window_overlap" and row["label"] in (*LABELS, "micro")
    ]
    if window_rows:
        lines += [
            "## Window-Overlap Metrics",
            "",
            "| Experiment | Label | Precision | Recall | F1 | TP | FP | FN |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in window_rows:
            lines.append(
                f"| {row['experiment']} | {row['label']} | {pct(row['precision'])} | "
                f"{pct(row['recall'])} | {pct(row['f1'])} | {row['tp']} | {row['fp']} | {row['fn']} |"
            )
    delta_window = [row for row in delta_rows if row["protocol"] == "window_overlap"]
    if delta_window:
        lines += [
            "",
            "## Window-Overlap Delta Vs Baseline",
            "",
            "| Experiment | Label | Guard | dP | dR | dF1 |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for row in delta_window:
            lines.append(
                f"| {row['experiment']} | {row['label']} | {row['recall_guard_pass']} | "
                f"{pct(row['precision_delta_pp'], signed=True)} | "
                f"{pct(row['recall_delta_pp'], signed=True)} | "
                f"{pct(row['f1_delta_pp'], signed=True) if row['f1_delta_pp'] != '' else '-'} |"
            )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize football long-video protocol eval runs.")
    parser.add_argument("--run", action="append", required=True, metavar="NAME=RUN_DIR")
    parser.add_argument("--protocol-prefix", default="protocol_comparison_checkpoint_thr")
    parser.add_argument("--compare-name", default="vs_full_image_e1_recall_guard_1pp")
    parser.add_argument("--output-dir", default="outputs/football_roi_experiments/long_protocol_summary")
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, Any]] = []
    deltas: list[dict[str, Any]] = []
    loaded: list[dict[str, Any]] = []
    for raw in args.run:
        name, run_dir = parse_named_path(raw)
        protocol_json = protocol_path(run_dir, args.protocol_prefix)
        payload = load_json(protocol_json)
        if payload is None:
            if args.allow_missing:
                print(f"warning: missing {protocol_json}")
                continue
            raise FileNotFoundError(protocol_json)
        metrics.extend(metric_rows(name, run_dir, payload))
        compare_payload = load_json(run_dir / f"{args.compare_name}.json")
        if compare_payload is not None:
            deltas.extend(compare_rows(name, compare_payload))
        loaded.append({"name": name, "run_dir": str(run_dir), "protocol_json": str(protocol_json)})
    write_csv(out_dir / "metrics.csv", metrics)
    write_csv(out_dir / "deltas.csv", deltas)
    (out_dir / "summary.md").write_text(build_markdown(metrics, deltas))
    (out_dir / "summary.json").write_text(
        json.dumps({"runs": loaded, "metrics": metrics, "deltas": deltas}, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"wrote {out_dir / 'metrics.csv'}")
    print(f"wrote {out_dir / 'deltas.csv'}")
    print(f"wrote {out_dir / 'summary.md'}")
    print(f"wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
