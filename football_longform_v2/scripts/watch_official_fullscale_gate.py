from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
for path in (PROJECT_ROOT / "src", WORKSPACE_ROOT, PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aggregate_official_feature_store import canonical_run_sha256  # noqa: E402
from build_official_feature_store import resolve, sha256_file  # noqa: E402
from football_longform_v2.config import load_config  # noqa: E402



@contextmanager
def gate_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another gate watcher owns {path}") from error
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def run_logged(
    command: list[str], *, cwd: Path, environment: dict[str, str], log_path: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as handle:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        rendered_command = " ".join(command)
        handle.write(f"\n[{timestamp}] launch={rendered_command}\n")
        handle.flush()
        subprocess.run(
            command, cwd=cwd, env=environment, check=True,
            stdout=handle, stderr=subprocess.STDOUT,
        )


def manifest_contract_error(
    state: dict, *, expected_records: list[tuple[int, str, str]],
    expected_config_sha256: str, expected_weights_sha256: str, required_run_id: str,
    max_age_seconds: float, now: float,
) -> str | None:
    if state.get("aggregation") != "cache_validation_truth":
        return "manifest is not cache-validation aggregation"
    if state.get("run_id") != required_run_id:
        return "run_id mismatch"
    if state.get("config_sha256") != expected_config_sha256:
        return "config_sha256 mismatch"
    if state.get("weights_sha256") != expected_weights_sha256:
        return "weights_sha256 mismatch"
    if state.get("canonical_run_sha256") != canonical_run_sha256([
        (split, video_id) for _, split, video_id in expected_records
    ]):
        return "canonical ID digest mismatch"
    updated = state.get("updated_unix")
    if not isinstance(updated, (float, int)) or updated > now + 60.0 or now - updated > max_age_seconds:
        return "manifest is stale or has invalid updated_unix"
    records = state.get("records")
    if not isinstance(records, list) or len(records) != len(expected_records):
        return "record count mismatch"
    got = [(record.get("global_index"), record.get("split"), record.get("video_id")) for record in records]
    if got != expected_records or len(set(got)) != len(got):
        return "record IDs/order mismatch"
    ready_statuses = {"completed", "skipped_valid"}
    statuses = [record.get("status") for record in records]
    if any(status not in ready_statuses for status in statuses):
        return "non-ready cache record present"
    total = len(expected_records)
    if any(int(state.get(key, -1)) != total for key in ("total", "processed", "completed", "skipped_valid")):
        return "legacy completion counts mismatch"
    if int(state.get("failed", -1)) != 0 or int(state.get("missing", -1)) != 0 or int(state.get("invalid_cache", -1)) != 0:
        return "manifest reports missing, invalid, or failed cache"
    return None


def checkpoint_selection_key(report: dict) -> tuple[float, ...]:
    labels = report.get("labels", {})
    shot = labels.get("shot", {})
    shot_operating = shot.get("operating_point_for_target_recall_at_2s") or {}
    shot_target_met = bool(shot_operating.get("target_achieved"))
    shot_target_precision = float(shot_operating.get("precision") or 0.0)
    shot_recall = float(shot.get("budget_recall_at_2s") or 0.0)
    supported_other = [
        metrics for label, metrics in labels.items()
        if label != "shot" and int(metrics.get("point_support") or 0) > 0
    ]
    other_operating = [
        metrics.get("operating_point_for_target_recall_at_2s") or {}
        for metrics in supported_other
    ]
    other_recalls = [
        float(metrics.get("budget_recall_at_2s") or 0.0)
        for metrics in supported_other
    ]
    other_met_fraction = (
        sum(bool(operating.get("target_achieved")) for operating in other_operating)
        / len(other_operating)
        if other_operating else 0.0
    )
    other_mean_recall = sum(other_recalls) / len(other_recalls) if other_recalls else 0.0
    supported = [metrics for metrics in labels.values() if int(metrics.get("point_support") or 0) > 0]
    aps = [
        float(metrics.get("point_ap_at_tolerance", {}).get("2.0") or 0.0)
        for metrics in supported
    ]
    mean_ap = sum(aps) / len(aps) if aps else 0.0
    other_target_precisions = [
        float(operating.get("precision") or 0.0)
        for operating in other_operating
        if operating.get("target_achieved")
    ]
    other_mean_target_precision = (
        sum(other_target_precisions) / len(other_target_precisions)
        if other_target_precisions else 0.0
    )
    return (
        float(shot_target_met), other_met_fraction, shot_target_precision,
        other_mean_target_precision, shot_recall, other_mean_recall, mean_ap,
    )


REQUIRED_EVENT_LABELS = ("shot", "save", "corner", "penalty", "freekick")


def diagnose_calibration_report(report: dict) -> dict:
    """Turn calibration metrics into an explicit recall-first failure diagnosis."""
    labels = report.get("labels", {})
    diagnosis: dict[str, dict] = {}
    for label in REQUIRED_EVENT_LABELS:
        metrics = labels.get(label, {})
        support = int(metrics.get("point_support") or 0)
        target = 0.90 if label == "shot" else 0.85
        candidate = metrics.get("candidate_ceiling_recall_at_2s")
        budget_recall = metrics.get("budget_recall_at_2s")
        operating = metrics.get("operating_point_for_target_recall_at_2s") or {}
        target_met = bool(operating.get("target_achieved")) if support > 0 else False
        if support <= 0:
            bottleneck = "unverified_no_calibration_support"
            recommendation = "collect_positive_calibration_events_before_claiming_recall"
        elif target_met:
            bottleneck = "target_met_optimize_precision"
            recommendation = "freeze_recall_threshold_then_reduce_false_positives"
        elif candidate is None or float(candidate) < target:
            bottleneck = "candidate_recall_bottleneck"
            recommendation = (
                "upgrade_short_term_rgb_motion_representation"
                if label in {"shot", "save"} else
                "upgrade_multiscale_context_and_restart_state_representation"
            )
        else:
            bottleneck = "ranking_or_budget_bottleneck"
            recommendation = (
                "hard_positive_mining_and_shot_verifier"
                if label == "shot" else
                "conditioned_class_verifier_and_event_balanced_hard_mining"
            )
        diagnosis[label] = {
            "support": support,
            "target_recall": target,
            "target_met": target_met,
            "candidate_ceiling_recall_at_2s": candidate,
            "budget_recall_at_2s": budget_recall,
            "calibration_recall": operating.get("recall"),
            "calibration_precision_at_target": operating.get("precision"),
            "calibration_fp_per_minute_at_target": operating.get("fp_per_minute"),
            "bottleneck": bottleneck,
            "recommended_next_action": recommendation,
        }
    unmet = [label for label, item in diagnosis.items() if not item["target_met"]]
    return {
        "schema": "football_longform_v2.gate_diagnostics.v1",
        "evaluation_split": report.get("evaluation_split"),
        "checkpoint": report.get("checkpoint"),
        "checkpoint_epoch": report.get("checkpoint_epoch"),
        "required_labels": list(REQUIRED_EVENT_LABELS),
        "all_required_recall_targets_verified": not unmet,
        "unmet_or_unverified_labels": unmet,
        "labels": diagnosis,
        "decision": (
            "recall_targets_met_optimize_precision"
            if not unmet else "continue_model_development_before_external_test"
        ),
        "external_test_allowed_for_final_comparison": not unmet,
    }


def launch_gate(
    *, root: Path, args: argparse.Namespace, expected: dict[str, int],
    failure: dict, state_path: Path, lock_path: Path,
) -> None:
    with gate_lock(lock_path):
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(root / "src")
        if args.cuda_visible_devices:
            environment["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        experiment_dir = root / "experiments/lf_a0_official_fullscale"
        log_path = experiment_dir / "gate_watch.log"
        train_command = [
            sys.executable, str(root / "scripts/train_locator.py"),
            "--config", args.config, "--device", args.device, "--epochs", "3",
            "--workers", "4", "--expected-ready-count", str(expected["train"]),
            "--checkpoint-prefix", "official_a_gate", "--positive-alpha", "0.75",
            "--density-balanced-focal", "--class-loss-weight", "0.5",
        ]
        if args.train_device_ids:
            train_command.extend(["--device-ids", args.train_device_ids])
        state_path.write_text(
            json.dumps({"status": "launching_gate", **failure}, indent=2) + "\n",
            encoding="utf-8",
        )
        print("launch=" + " ".join(train_command), flush=True)
        run_logged(train_command, cwd=root, environment=environment, log_path=log_path)
        evaluations: list[dict] = []
        for epoch in range(3):
            checkpoint = experiment_dir / f"official_a_gate_epoch_{epoch:03d}.pt"
            report_path = experiment_dir / f"calibration_epoch_{epoch:03d}.json"
            evaluate_command = [
                sys.executable, str(root / "scripts/evaluate_locator.py"),
                "--config", args.config, "--device", args.device,
                "--checkpoint", str(checkpoint),
                "--expected-video-count", str(expected["calibration"]),
                "--report", str(report_path),
            ]
            print("launch=" + " ".join(evaluate_command), flush=True)
            run_logged(evaluate_command, cwd=root, environment=environment, log_path=log_path)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            evaluations.append({
                "epoch": epoch, "checkpoint": str(checkpoint), "report": str(report_path),
                "selection_key": list(checkpoint_selection_key(report)),
            })
        selected = max(evaluations, key=lambda item: tuple(item["selection_key"]))
        best_checkpoint = experiment_dir / "official_a_gate_best.pt"
        shutil.copy2(selected["checkpoint"], best_checkpoint)
        selection = {
            "schema": "football_longform_v2.checkpoint_selection.v1",
            "priority": [
                "shot_target_met", "other_target_fraction", "shot_target_precision",
                "other_mean_target_precision", "shot_budget_recall",
                "other_mean_budget_recall", "mean_point_ap_2s",
            ],
            "selected": {**selected, "copied_checkpoint": str(best_checkpoint)},
            "evaluations": evaluations,
        }
        selection_path = experiment_dir / "checkpoint_selection.json"
        selection_path.write_text(
            json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        operating_points_path = experiment_dir / "operating_points.json"
        freeze_command = [
            sys.executable, str(root / "scripts/freeze_operating_points.py"),
            "--selection", str(selection_path), "--output", str(operating_points_path),
        ]
        print("launch=" + " ".join(freeze_command), flush=True)
        run_logged(freeze_command, cwd=root, environment=environment, log_path=log_path)
        diagnostics_path = experiment_dir / "gate_diagnostics.json"
        selected_report = json.loads(Path(selected["report"]).read_text(encoding="utf-8"))
        diagnostics = diagnose_calibration_report(selected_report)
        diagnostics["selected_checkpoint"] = str(best_checkpoint)
        diagnostics["operating_points"] = str(operating_points_path)
        diagnostics_path.write_text(
            json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        state_path.write_text(
            json.dumps({
                "status": "complete", **failure,
                "selection": str(selection_path), "best_checkpoint": str(best_checkpoint),
                "operating_points": str(operating_points_path),
                "diagnostics": str(diagnostics_path),
            }, indent=2) + "\n", encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the 3-epoch gate only after canonical cache completion.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--cuda-visible-devices",
        help="Physical CUDA indices exposed to train/eval subprocesses, e.g. 6,7.",
    )
    parser.add_argument(
        "--train-device-ids",
        help="Comma-separated CUDA indices for DataParallel training; evaluation uses --device.",
    )
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-manifest-age-seconds", type=float, default=600.0)
    args = parser.parse_args()
    root = PROJECT_ROOT
    manifest = root / "experiments/lf_a0_official_fullscale/canonical_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"canonical manifest missing: {manifest}")
    canonical = json.loads(manifest.read_text(encoding="utf-8"))
    expected = {"train": len(canonical["train"]), "calibration": len(canonical["calibration"])}
    expected_records = [
        (index, split, str(entry["media_id"]))
        for index, (split, entry) in enumerate(
            [("train", entry) for entry in canonical["train"]] +
            [("calibration", entry) for entry in canonical["calibration"]]
        )
    ]
    config = load_config(args.config)
    config_path = Path(config["_config_path"])
    args.config = str(config_path)
    expected_config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    expected_weights_sha256 = sha256_file(
        resolve(Path(config["_project_root"]), str(config["features"]["context"]["weights"]))
    )
    store_manifest = Path("/mnt/data_16t/football/feature_store_v2/official_lvd1689m_rgbflow_v1/manifest.json")
    state_path = root / "experiments/lf_a0_official_fullscale/gate_watch_state.json"
    lock_path = root / "experiments/lf_a0_official_fullscale/gate_watch.lock"
    while True:
        if not store_manifest.is_file():
            print("waiting: feature manifest not written", flush=True)
            time.sleep(args.poll_seconds)
            continue
        state = json.loads(store_manifest.read_text(encoding="utf-8"))
        contract_error = manifest_contract_error(
            state, expected_records=expected_records,
            expected_config_sha256=expected_config_sha256,
            expected_weights_sha256=expected_weights_sha256, required_run_id=args.run_id,
            max_age_seconds=args.max_manifest_age_seconds, now=time.time(),
        )
        if contract_error:
            print(f"waiting: manifest contract {contract_error}", flush=True)
            time.sleep(args.poll_seconds)
            continue
        if int(state.get("processed", 0)) < int(state.get("total", -1)):
            print(
                f"waiting: {state.get('processed')}/{state.get('total')} "
                f"completed={state.get('completed')} failed={state.get('failed')}", flush=True
            )
            time.sleep(args.poll_seconds)
            continue
        records = state.get("records", [])
        status = Counter(record.get("status") for record in records)
        ready = Counter(
            record.get("split") for record in records
            if record.get("status") in {"completed", "skipped_valid"}
        )
        failure = {
            "feature_manifest": str(store_manifest), "expected": expected,
            "ready": dict(ready), "statuses": dict(status), "failed": state.get("failed"),
        }
        if state.get("failed") or ready != Counter(expected) or status.get("skipped_missing_media", 0):
            failure["status"] = "blocked_incomplete_or_failed_features"
            state_path.write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
            raise RuntimeError(json.dumps(failure))
        break
    launch_gate(
        root=root, args=args, expected=expected, failure=failure,
        state_path=state_path, lock_path=lock_path,
    )


if __name__ == "__main__":
    main()
