#!/usr/bin/env python3
"""Build a blinded Astra verifier set directly from frozen DINO dense windows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


LABELS = ("shot", "save", "set_piece")
VIDEOS = ("2041772596969549825", "2042520801973841921", "2042526457476886530")


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-run", type=Path, required=True)
    parser.add_argument("--threshold-report", type=Path, required=True)
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--component-gap-sec", type=float, default=10.01)
    parser.add_argument("--max-core-span-sec", type=float, default=30.0)
    parser.add_argument("--context-sec", type=float, default=6.0)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def frozen_thresholds(report: dict[str, Any]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {video_id: {} for video_id in VIDEOS}
    for label in LABELS:
        rows = report["per_class"][label]["test_frozen_transfer"]["per_video"]
        for row in rows:
            video_id = row["video_id"]
            if video_id in result:
                result[video_id][label] = float(row["effective_probability_threshold"])
    missing = {video_id: sorted(set(LABELS) - set(values)) for video_id, values in result.items()}
    if any(missing.values()):
        raise ValueError(f"Missing frozen thresholds: {missing}")
    return result


def opaque(prefix: str, value: str) -> str:
    return f"{prefix}_{hashlib.sha256(('astra-dino-v1:' + value).encode()).hexdigest()[:20]}"


def split_times(times: list[float], component_gap: float, max_span: float) -> list[list[float]]:
    components: list[list[float]] = []
    for time_sec in times:
        if not components or time_sec - components[-1][-1] > component_gap:
            components.append([time_sec])
        else:
            components[-1].append(time_sec)
    groups: list[list[float]] = []
    for component in components:
        current: list[float] = []
        for time_sec in component:
            if current and time_sec - current[0] > max_span:
                groups.append(current)
                current = []
            current.append(time_sec)
        if current:
            groups.append(current)
    return groups


def main() -> None:
    config = args()
    thresholds = frozen_thresholds(read_json(config.threshold_report))
    public_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    video_stats = []
    for video_id in VIDEOS:
        annotation = read_json(config.annotations_dir / f"{video_id}.json")
        source_video = annotation["video_source"]["video_path"]
        duration = float(annotation["video_source"]["duration_sec"])
        with (config.dense_run / video_id / "window_predictions.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            windows = list(csv.DictReader(handle))
        candidates: dict[float, dict[str, Any]] = {}
        for window in windows:
            time_sec = (float(window["start_sec"]) + float(window["end_sec"])) / 2.0
            probabilities = {label: float(window[f"prob_{label}"]) for label in LABELS}
            active = {
                label: probabilities[label] >= thresholds[video_id][label] for label in LABELS
            }
            if not any(active.values()):
                continue
            candidate_id = opaque("cand", f"{video_id}:{time_sec:.6f}")
            candidates[time_sec] = {
                "candidate_id": candidate_id,
                "video_id": video_id,
                "anchor_time_sec": time_sec,
                "dino_probabilities": probabilities,
                "dino_active_labels": [label for label in LABELS if active[label]],
                "window_start_sec": float(window["start_sec"]),
                "window_end_sec": float(window["end_sec"]),
            }
        groups = split_times(
            sorted(candidates), config.component_gap_sec, config.max_core_span_sec
        )
        for group_index, group in enumerate(groups):
            core_start = max(0.0, group[0] - 2.5)
            core_end = min(duration, group[-1] + 2.5)
            input_start = max(0.0, core_start - config.context_sec)
            input_end = min(duration, core_end + config.context_sec)
            segment_id = opaque("seg", f"{video_id}:{group_index}:{group[0]}:{group[-1]}")
            public_candidates = [
                {
                    "candidate_id": candidates[time_sec]["candidate_id"],
                    "anchor_sec_relative_to_clip": round(time_sec - input_start, 6),
                    "anchor_radius_sec": 3.0,
                }
                for time_sec in group
            ]
            public_rows.append(
                {
                    "segment_id": segment_id,
                    "video_id": video_id,
                    "source_video": source_video,
                    "input_start_sec": round(input_start, 6),
                    "input_end_sec": round(input_end, 6),
                    "source_resolution": [1280, 720],
                    "candidates": public_candidates,
                }
            )
            for time_sec in group:
                private = dict(candidates[time_sec])
                private["segment_id"] = segment_id
                private_rows.append(private)
        active_counts = Counter(
            label for candidate in candidates.values() for label in candidate["dino_active_labels"]
        )
        video_stats.append(
            {
                "video_id": video_id,
                "segments": len(groups),
                "unique_candidate_times": len(candidates),
                "active_candidates_by_label": dict(sorted(active_counts.items())),
                "duration_sec": duration,
            }
        )
    public_rows.sort(key=lambda row: (row["video_id"], row["input_start_sec"]))
    private_rows.sort(key=lambda row: (row["video_id"], row["anchor_time_sec"]))
    write_jsonl(config.output_dir / "blind_segments.jsonl", public_rows)
    write_jsonl(config.output_dir / "private_dino_key.jsonl", private_rows)
    manifest = {
        "schema_version": "football_astra_dino_verifier_v1",
        "dense_run": str(config.dense_run.resolve()),
        "checkpoint": "fromlast_e8/best.pt (epoch 3)",
        "threshold_source": str(config.threshold_report.resolve()),
        "threshold_policy": "validation LOOV recall-guard, frozen transfer to test",
        "frozen_probability_thresholds": thresholds,
        "component_gap_sec": config.component_gap_sec,
        "max_core_span_sec": config.max_core_span_sec,
        "context_sec": config.context_sec,
        "event_nms": False,
        "request_grouping_note": "Nearby candidate anchors share one visual request; every anchor remains an independent scored candidate.",
        "segments": len(public_rows),
        "unique_candidate_times": len(private_rows),
        "videos": video_stats,
        "inference_forbidden": ["private_dino_key.jsonl", "all GT files"],
    }
    write_json(config.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
