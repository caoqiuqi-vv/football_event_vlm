#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
LAUNCHER="scripts/launch_football_detection_aware.sh"

bash "${LAUNCHER}" train_gate "${GPU}"
bash "${LAUNCHER}" eval_gate "${GPU}"
bash "${LAUNCHER}" baseline_summary "${GPU}"
