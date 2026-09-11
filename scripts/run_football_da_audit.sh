#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
LAUNCHER="scripts/launch_football_detection_aware.sh"

bash "${LAUNCHER}" index "${GPU}"
bash "${LAUNCHER}" roi_audit "${GPU}"
bash "${LAUNCHER}" resolution "${GPU}"
bash "${LAUNCHER}" baseline_global "${GPU}"
bash "${LAUNCHER}" baseline_legacy "${GPU}"
bash "${LAUNCHER}" baseline_robust "${GPU}"
bash "${LAUNCHER}" baseline_summary "${GPU}"
