#!/usr/bin/env python3
"""Generate deterministic five-fold ID files and a watchdog command plan."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def lines(path: str) -> list[str]: return [x.strip() for x in Path(path).read_text().splitlines() if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--config", required=True); p.add_argument("--work", required=True); p.add_argument("--epochs", type=int, default=12); args = p.parse_args()
    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text()); root = Path(args.work); folds = root / "folds"; folds.mkdir(parents=True, exist_ok=True)
    train, calibration = lines(cfg["data"]["train_ids"]), cfg["data"]["calibration_ids"]
    stage: list[dict] = [{"name": "cache_4fps", "command": f"PYTHONPATH=football_e2e_spotter/src python football_e2e_spotter/scripts/build_pixel_audio_store.py --config football_e2e_spotter/configs/set_spotter_v1_cache.yaml --workers 8"}]
    candidate_paths = []
    for fold in range(5):
        valid = train[fold::5]; fit = [item for index, item in enumerate(train) if index % 5 != fold]
        fit_path, valid_path = folds / f"fold{fold}_fit.txt", folds / f"fold{fold}_valid.txt"; fit_path.write_text("\n".join(fit) + "\n"); valid_path.write_text("\n".join(valid) + "\n")
        ckpt_dir, candidates = root / f"stage1_fold{fold}", root / "oof" / f"fold{fold}.jsonl"; candidate_paths.append(str(candidates))
        stage += [
            {"name": f"stage1_fold{fold}", "command": f"PYTHONPATH=football_e2e_spotter/src python football_e2e_spotter/train_set_spotter_ids.py --config {args.config} --ids {fit_path} --output {ckpt_dir} --epochs {args.epochs}"},
            {"name": f"candidates_fold{fold}", "command": f"PYTHONPATH=football_e2e_spotter/src python football_e2e_spotter/run_set_candidates.py --config {args.config} --checkpoint {ckpt_dir}/last.pt --ids {valid_path} --split train --output {candidates} --threshold 0.01"},
        ]
    merged = root / "oof" / "candidates.jsonl"; stage.append({"name": "merge_oof_candidates", "command": f"python football_e2e_spotter/merge_jsonl.py --output {merged} " + " ".join(candidate_paths)})
    stage += [
        {"name": "verifier", "command": f"PYTHONPATH=football_e2e_spotter/src python football_e2e_spotter/train_set_verifier.py --candidates {merged} --annotations {cfg['data']['annotations']} --output {root}/verifier --epochs 6"},
        {"name": "stage1_full", "command": f"PYTHONPATH=football_e2e_spotter/src python football_e2e_spotter/train_set_spotter_ids.py --config {args.config} --ids {cfg['data']['train_ids']} --output {root}/stage1_full --epochs {args.epochs}"},
        {"name": "calibration_candidates", "command": f"PYTHONPATH=football_e2e_spotter/src python football_e2e_spotter/run_set_candidates.py --config {args.config} --checkpoint {root}/stage1_full/last.pt --ids {calibration} --split calibration --output {root}/calibration/candidates.jsonl --threshold 0.01"},
    ]
    plan = {"cwd": str(Path.cwd()), "no_temporal_nms": True, "stages": stage, "pending_final_calibration": True}
    path = root / "watchdog_plan.json"; path.write_text(json.dumps(plan, indent=2) + "\n"); print(path)


if __name__ == "__main__": main()
