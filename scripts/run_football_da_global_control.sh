#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
LAUNCHER="scripts/launch_football_detection_aware.sh"

bash "${LAUNCHER}" train_global "${GPU}"
bash "${LAUNCHER}" eval_global "${GPU}"
