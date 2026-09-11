#!/usr/bin/env python
"""Resume v5 from epoch 6 with positive-sparse DDP handling."""

from __future__ import annotations

import os
import subprocess

import watch_goal_retriever_v1 as base


def run_training_v6() -> None:
    base.OUTPUT.mkdir(parents=True, exist_ok=True)
    resume = base.OUTPUT / "last.pt"
    if not resume.is_file():
        raise FileNotFoundError(resume)
    log_path = base.EXPERIMENT / "train_retriever_v6_resume.log"
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in base.GPUS)
    command = [
        "torchrun", "--standalone", f"--nproc_per_node={len(base.GPUS)}",
        str(base.ROOT / "football_e2e_spotter/scripts/train_goal_retriever_v6.py"),
        "--manifest", str(base.MANIFEST), "--feature-root", str(base.FEATURES), "--output", str(base.OUTPUT),
        "--epochs", "20", "--batch-size", "12", "--eval-batch-size", "32", "--workers", "0",
        "--lr", "0.0002", "--hidden-dim", "384", "--shot-recall-floor", "0.97", "--other-recall-floor", "0.94",
        "--resume", str(resume),
    ]
    with log_path.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=base.ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        base.update("retriever_training_v6_resume", pid=process.pid, resume=str(resume), log=str(log_path))
        code = process.wait()
    if code:
        raise RuntimeError(f"retriever v6 resume failed with {code}; see {log_path}")


if __name__ == "__main__":
    base.run_training = run_training_v6
    base.main()

