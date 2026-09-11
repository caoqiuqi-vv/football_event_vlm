from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from football_longform_v2.config import load_config  # noqa: E402


def resolve(root: Path, raw: str) -> Path:
    value = Path(raw).expanduser()
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def build_commands(
    *, root: Path, config_path: Path, oof_path: Path, selection: dict,
    oof: dict, device: str, workers: int,
) -> tuple[list[dict], list[str], Path]:
    selected = selection.get("selected", {})
    if "epoch" not in selected:
        raise ValueError("main A0 checkpoint selection has no fixed epoch")
    fixed_epoch = int(selected["epoch"])
    epochs = fixed_epoch + 1
    if epochs <= 0:
        raise ValueError("selected epoch is invalid")
    folds = sorted(oof.get("folds", []), key=lambda item: int(item["fold"]))
    if len(folds) != int(oof.get("fold_count", -1)) or not folds:
        raise ValueError("OOF manifest fold count mismatch")
    output_dir = Path(load_config(config_path)["paths"]["output_dir"])
    if not output_dir.is_absolute():
        output_dir = (root / output_dir).resolve()
    jobs = []
    report_paths = []
    for fold in folds:
        fold_index = int(fold["fold"])
        prefix = f"penalty_oof_fold_{fold_index}_fixed_epoch_{fixed_epoch:03d}"
        checkpoint = output_dir / f"{prefix}_epoch_{fixed_epoch:03d}.pt"
        report = output_dir / f"penalty_oof_fold_{fold_index}_report.json"
        train_command = [
            sys.executable, str(root / "scripts/train_locator.py"),
            "--config", str(config_path), "--device", device,
            "--epochs", str(epochs), "--workers", str(workers),
            "--expected-ready-count", str(len(fold["train_media_ids"])),
            "--train-id-file", str(Path(fold["train_id_file"]).resolve()),
            "--checkpoint-prefix", prefix,
            "--positive-alpha", "0.75", "--density-balanced-focal",
            "--class-loss-weight", "0.5",
        ]
        evaluate_command = [
            sys.executable, str(root / "scripts/evaluate_penalty_oof_fold.py"),
            "--config", str(config_path), "--checkpoint", str(checkpoint),
            "--oof-manifest", str(oof_path), "--fold", str(fold_index),
            "--device", device, "--report", str(report),
        ]
        jobs.append({
            "fold": fold_index, "fixed_epoch": fixed_epoch,
            "checkpoint": str(checkpoint), "report": str(report),
            "train_command": train_command, "evaluate_command": evaluate_command,
        })
        report_paths.append(str(report))
    aggregate_output = output_dir / "penalty_oof_aggregate.json"
    aggregate_command = [
        sys.executable, str(root / "scripts/aggregate_penalty_oof.py"),
        "--oof-manifest", str(oof_path), "--reports", *report_paths,
        "--output", str(aggregate_output),
    ]
    return jobs, aggregate_command, aggregate_output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run fixed-epoch five-fold penalty OOF training/evaluation on one GPU."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--oof-manifest", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.device != "cuda:0":
        raise ValueError("current experiment policy requires penalty OOF to use only cuda:0")
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    root = Path(config["_project_root"])
    oof_path = Path(args.oof_manifest).expanduser().resolve()
    selection_path = Path(args.selection).expanduser().resolve()
    oof = json.loads(oof_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if oof.get("schema") != "football_longform_v2.penalty_oof_folds.v1":
        raise ValueError("unsupported OOF manifest schema")
    if selection.get("schema") != "football_longform_v2.checkpoint_selection.v1":
        raise ValueError("unsupported main checkpoint selection schema")
    jobs, aggregate_command, aggregate_output = build_commands(
        root=root, config_path=config_path, oof_path=oof_path,
        selection=selection, oof=oof, device=args.device, workers=args.workers,
    )
    plan = {
        "schema": "football_longform_v2.penalty_oof_run_plan.v1",
        "device": args.device,
        "fixed_epoch_source": str(selection_path),
        "fixed_epoch": jobs[0]["fixed_epoch"],
        "fixed_test_labels_or_predictions_used": False,
        "jobs": jobs,
        "aggregate_command": aggregate_command,
        "aggregate_output": str(aggregate_output),
    }
    plan_path = aggregate_output.with_name("penalty_oof_run_plan.json")
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    existing = [
        path for job in jobs for path in (Path(job["checkpoint"]), Path(job["report"]))
        if path.exists()
    ]
    if aggregate_output.exists():
        existing.append(aggregate_output)
    if existing and not args.force:
        raise FileExistsError(
            "OOF outputs already exist; inspect them or pass --force explicitly: "
            + ", ".join(str(path) for path in existing[:5])
        )
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    state_path = aggregate_output.with_name("penalty_oof_run_state.json")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src") + os.pathsep + str(root.parent)
    completed = []
    for job in jobs:
        state_path.write_text(json.dumps({
            "status": "running", "current_fold": job["fold"], "completed_folds": completed,
            "device": args.device, "fixed_test_labels_or_predictions_used": False,
        }, indent=2) + "\n", encoding="utf-8")
        print("launch=" + " ".join(job["train_command"]), flush=True)
        subprocess.run(job["train_command"], cwd=root, env=environment, check=True)
        print("launch=" + " ".join(job["evaluate_command"]), flush=True)
        subprocess.run(job["evaluate_command"], cwd=root, env=environment, check=True)
        completed.append(job["fold"])
    print("launch=" + " ".join(aggregate_command), flush=True)
    subprocess.run(aggregate_command, cwd=root, env=environment, check=True)
    state_path.write_text(json.dumps({
        "status": "complete", "completed_folds": completed,
        "aggregate_output": str(aggregate_output), "device": args.device,
        "fixed_test_labels_or_predictions_used": False,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
