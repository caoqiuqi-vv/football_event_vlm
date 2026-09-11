from __future__ import annotations

"""Video-grouped OOF calibration for penalty when main calibration has no positives."""

import argparse
import concurrent.futures
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from football_longform_v2.evaluation import average_precision, operating_point_at_recall  # noqa: E402


def run_gpu_queue(
    gpu: int,
    folds: list[dict],
    *,
    config: Path,
    output_root: Path,
    epochs: int,
) -> list[dict]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    results = []
    for fold in folds:
        index = int(fold["fold"])
        output = output_root / f"fold_{index}"
        output.mkdir(parents=True, exist_ok=True)
        log_path = output / "train.log"
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts/train_chunk_locator.py"),
            "--config", str(config),
            "--device", "cuda:0",
            "--workers", "3",
            "--epochs", str(epochs),
            "--train-id-file", str(Path(fold["train_id_file"]).resolve()),
            "--validation-id-file", str(Path(fold["validation_id_file"]).resolve()),
            "--validation-canonical-split", "train",
            "--validation-store-split", "train",
            "--output-dir", str(output),
            "--evaluate-only-final",
        ]
        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps({"command": command, "physical_gpu": gpu}) + "\n")
            log.flush()
            completed = subprocess.run(
                command, cwd=REPO_ROOT, env=env, stdout=log,
                stderr=subprocess.STDOUT, check=False,
            )
        if completed.returncode:
            raise RuntimeError(f"A2 penalty OOF fold {index} failed; see {log_path}")
        results.append({"fold": index, "gpu": gpu, "output": str(output), "log": str(log_path)})
    return results


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return [max(center - radius, 0.0), min(center + radius, 1.0)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="football_longform_v2/configs/lf_a2_sequential_locator.yaml"
    )
    parser.add_argument(
        "--fold-manifest",
        default="football_longform_v2/experiments/lf_a0_official_fullscale/penalty_oof_folds.json",
    )
    parser.add_argument(
        "--main-selection",
        default="football_longform_v2/experiments/lf_a2_sequential_locator/checkpoint_selection.json",
    )
    parser.add_argument(
        "--output-root",
        default="football_longform_v2/experiments/lf_a2_penalty_oof",
    )
    parser.add_argument("--gpus", default="0,6,7")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--target-recall", type=float, default=0.85)
    args = parser.parse_args()
    config = (REPO_ROOT / args.config).resolve()
    fold_manifest_path = (REPO_ROOT / args.fold_manifest).resolve()
    main_selection_path = (REPO_ROOT / args.main_selection).resolve()
    output_root = (REPO_ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    folds_payload = json.loads(fold_manifest_path.read_text(encoding="utf-8"))
    if folds_payload.get("schema") != "football_longform_v2.penalty_oof_folds.v1":
        raise ValueError("unsupported penalty OOF fold manifest")
    if folds_payload.get("fixed_test_labels_or_predictions_used") is not False:
        raise RuntimeError("penalty OOF manifest does not guarantee sealed fixed test")
    folds = list(folds_payload["folds"])
    main_selection = json.loads(main_selection_path.read_text(encoding="utf-8"))
    epochs = int(args.epochs or main_selection["best_epoch"])
    if epochs <= 0:
        raise ValueError("fixed OOF epoch budget must be positive")
    gpus = tuple(int(item) for item in args.gpus.split(",") if item.strip())
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError("--gpus must contain unique IDs")
    assignments = {gpu: folds[index::len(gpus)] for index, gpu in enumerate(gpus)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [
            pool.submit(
                run_gpu_queue, gpu, assigned, config=config,
                output_root=output_root, epochs=epochs,
            )
            for gpu, assigned in assignments.items() if assigned
        ]
        workers = [item for future in futures for item in future.result()]

    expected_ids = []
    seen_ids = []
    scores: list[float] = []
    matches: list[bool] = []
    support = 0
    total_minutes = 0.0
    fold_reports = []
    for fold in folds:
        index = int(fold["fold"])
        expected = [str(item) for item in fold["validation_media_ids"]]
        expected_ids.extend(expected)
        report_path = output_root / f"fold_{index}" / f"calibration_epoch_{epochs:03d}.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["video_ids"] != expected:
            raise RuntimeError(f"OOF fold {index} validation IDs disagree with frozen manifest")
        records = report["operating_match_records"]["penalty"]
        fold_ids = [str(item["video_id"]) for item in records]
        if fold_ids != expected:
            raise RuntimeError(f"OOF fold {index} penalty record coverage mismatch")
        seen_ids.extend(fold_ids)
        for item in records:
            support += int(item["support"])
            scores.extend(float(value) for value in item["scores"])
            matches.extend(bool(value) for value in item["matches"])
        total_minutes += float(report["total_minutes"])
        fold_reports.append({
            "fold": index, "report": str(report_path),
            "video_count": report["video_count"],
            "penalty_support": sum(int(item["support"]) for item in records),
        })
    if len(seen_ids) != len(set(seen_ids)) or set(seen_ids) != set(expected_ids):
        raise RuntimeError("OOF validation videos are duplicated or incomplete")
    if support != int(folds_payload["source_penalty_event_count"]):
        raise RuntimeError(f"OOF penalty support {support} != frozen count")
    operating = operating_point_at_recall(
        scores, matches, positive_count=support,
        target_recall=args.target_recall, total_minutes=total_minutes,
    )
    tp = int(operating["tp"] if operating else 0)
    report = {
        "schema": "football_longform_v2.a2_penalty_oof.v1",
        "purpose": "development_only_penalty_threshold_and_fp_budget",
        "fixed_epoch_budget": epochs,
        "fold_count": len(folds),
        "video_count": len(seen_ids),
        "penalty_support": support,
        "average_precision": average_precision(
            torch.tensor(scores), torch.tensor(matches), positive_count=support
        ),
        "operating_point": operating,
        "recall_wilson_95": wilson_interval(tp, support),
        "raw_threshold_portability": "diagnostic_only_use_fp_per_minute_to_transfer_to_final_model",
        "total_minutes": total_minutes,
        "fold_manifest": str(fold_manifest_path),
        "main_selection": str(main_selection_path),
        "folds": fold_reports,
        "workers": workers,
        "fixed_test_labels_or_predictions_used": False,
    }
    output = output_root / "penalty_oof_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output), "support": support,
        "average_precision": report["average_precision"], "operating_point": operating,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
