from __future__ import annotations

"""Freeze recall-first A2 thresholds only after main calibration and penalty OOF."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from football_longform_v2.canonical import load_canonical_split  # noqa: E402


LABELS = ("shot", "save", "corner", "freekick", "penalty")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def false_budget_threshold(scores: list[float], fp_per_minute: float, minutes: float) -> dict:
    budget = max(int(math.floor(float(fp_per_minute) * float(minutes) + 1e-9)), 0)
    ordered = sorted((float(score) for score in scores), reverse=True)
    if not ordered or budget <= 0:
        return {
            "threshold": 1.000001,
            "false_budget": budget,
            "selected_false_peaks": 0,
        }
    if budget >= len(ordered):
        threshold = min(ordered) - 1e-8
    else:
        upper = ordered[budget - 1]
        lower = ordered[budget]
        threshold = 0.5 * (upper + lower) if upper > lower else upper
    selected = sum(score >= threshold for score in ordered)
    return {
        "threshold": threshold,
        "false_budget": budget,
        "selected_false_peaks": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration-report", required=True)
    parser.add_argument("--penalty-oof", required=True)
    parser.add_argument("--canonical-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    calibration_path = Path(args.calibration_report).expanduser().resolve()
    oof_path = Path(args.penalty_oof).expanduser().resolve()
    canonical_path = Path(args.canonical_manifest).expanduser().resolve()
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    oof = json.loads(oof_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(Path(args.config).expanduser().resolve().read_text(encoding="utf-8"))
    precision_gates = config["deployment_gate"]["minimum_precision_at_target_recall"]
    fp_gates = config["deployment_gate"]["maximum_fp_per_minute"]
    if calibration.get("schema") != "football_longform_v2.sequential_chunk_evaluation.v1":
        raise ValueError("unsupported A2 calibration report")
    if oof.get("schema") != "football_longform_v2.a2_penalty_oof.v1":
        raise ValueError("unsupported A2 penalty OOF report")
    if oof.get("fixed_test_labels_or_predictions_used") is not False:
        raise RuntimeError("penalty OOF does not prove fixed-test isolation")
    canonical = load_canonical_split(canonical_path, "calibration")
    if tuple(calibration["video_ids"]) != canonical.media_ids:
        raise RuntimeError("calibration report does not cover the frozen calibration split exactly")
    tolerance_key = f"{float(calibration['operating_tolerance_seconds']):g}s"
    class_reports = calibration["tolerances"][tolerance_key]
    thresholds = {}
    evidence = {}
    for label in LABELS[:-1]:
        item = class_reports[label]
        operating = item["operating_point"]
        if int(item["support"]) <= 0 or not operating or not operating["target_achieved"]:
            raise RuntimeError(f"cannot freeze {label}: recall target was not achieved")
        if float(operating["precision"] or 0.0) < float(precision_gates[label]):
            raise RuntimeError(
                f"cannot freeze {label}: precision {operating['precision']} "
                f"< gate {precision_gates[label]}"
            )
        if float(operating["fp_per_minute"] or 0.0) > float(fp_gates[label]):
            raise RuntimeError(
                f"cannot freeze {label}: FP/min {operating['fp_per_minute']} "
                f"> gate {fp_gates[label]}"
            )
        thresholds[label] = float(operating["threshold"])
        evidence[label] = {
            "source": "main_calibration",
            "support": int(item["support"]),
            "recall": float(operating["recall"]),
            "precision": float(operating["precision"]),
            "fp_per_minute": float(operating["fp_per_minute"]),
            "average_precision": item["average_precision"],
        }
    penalty_operating = oof.get("operating_point")
    if (
        int(oof.get("penalty_support", 0)) <= 0
        or not penalty_operating
        or not penalty_operating.get("target_achieved")
    ):
        raise RuntimeError("cannot freeze penalty: OOF recall target was not achieved")
    if float(penalty_operating["precision"] or 0.0) < float(precision_gates["penalty"]):
        raise RuntimeError("cannot freeze penalty: OOF precision gate was not achieved")
    if float(penalty_operating["fp_per_minute"] or 0.0) > float(fp_gates["penalty"]):
        raise RuntimeError("cannot freeze penalty: OOF FP/min gate was not achieved")
    penalty_scores = [
        float(score)
        for record in calibration["operating_match_records"]["penalty"]
        for score in record["scores"]
    ]
    transfer = false_budget_threshold(
        penalty_scores,
        float(penalty_operating["fp_per_minute"]),
        float(calibration["total_minutes"]),
    )
    thresholds["penalty"] = float(transfer["threshold"])
    evidence["penalty"] = {
        "source": "five_fold_train_video_oof_fp_budget_transferred_on_main_calibration_background",
        "support": int(oof["penalty_support"]),
        "oof_recall": float(penalty_operating["recall"]),
        "oof_precision": float(penalty_operating["precision"]),
        "oof_fp_per_minute": float(penalty_operating["fp_per_minute"]),
        "oof_recall_wilson_95": oof.get("recall_wilson_95"),
        **transfer,
    }
    payload = {
        "schema": "football_longform_v2.a2_operating_points.v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config": str(Path(args.config).expanduser().resolve()),
        "calibration_report": str(calibration_path),
        "calibration_report_sha256": sha256_file(calibration_path),
        "penalty_oof": str(oof_path),
        "penalty_oof_sha256": sha256_file(oof_path),
        "labels": list(LABELS),
        "thresholds": thresholds,
        "target_recall": calibration["target_recall"],
        "minimum_precision_at_target_recall": precision_gates,
        "maximum_fp_per_minute": fp_gates,
        "operating_tolerance_seconds": calibration["operating_tolerance_seconds"],
        "nms_radius_seconds": None,
        "evidence": evidence,
        "fixed_test_media_opened": False,
        "frozen_before_fixed_test": True,
    }
    # NMS is stored in YAML, while this script intentionally has no dependency
    # on the legacy config loader.
    payload["nms_radius_seconds"] = config["evaluation"]["nms_radius_seconds"]
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "thresholds": thresholds}, ensure_ascii=False))


if __name__ == "__main__":
    main()
