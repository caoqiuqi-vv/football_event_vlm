#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:?Usage: $0 OUTPUT_DIR TRAIN_PID EPOCH BASELINE_BRANCH_METRICS}"
TRAIN_PID="${2:?missing TRAIN_PID}"
EPOCH="${3:?missing EPOCH}"
BASELINE_BRANCH_METRICS="${4:?missing BASELINE_BRANCH_METRICS}"
POLL_SEC="${POLL_SEC:-20}"

metrics="${OUTPUT_DIR}/metrics_epoch_$(printf '%03d' "${EPOCH}").json"
report="${OUTPUT_DIR}/epoch_${EPOCH}_event_vs_reference.json"
markdown="${OUTPUT_DIR}/epoch_${EPOCH}_event_vs_reference.md"
errors="${OUTPUT_DIR}/epoch_${EPOCH}_event_vs_reference.err"
log="${OUTPUT_DIR}/guard.log"

while [[ ! -s "${metrics}" ]]; do
  if ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
    echo "training_exited_before_guard $(date --iso-8601=seconds)" >> "${log}"
    exit 0
  fi
  sleep "${POLL_SEC}"
done

python scripts/compare_football_temporal_branch_metrics.py --baseline "${BASELINE_BRANCH_METRICS}" --candidates "${metrics}" --branch event --labels shot,save,set_piece --recall-tolerance-pp 1 --output "${report}" > "${markdown}" 2> "${errors}"

if jq -e '
  .candidates[0] as $c
  | (
      $c.per_class.shot.recall_guard_pass
      and $c.per_class.save.recall_guard_pass
      and $c.per_class.set_piece.recall_guard_pass
      and (
        (($c.per_class.shot.precision_delta
          + $c.per_class.save.precision_delta) / 2) > 0
      )
    )
' "${report}" >/dev/null; then
  echo "event_guard_pass_continue_epoch2 $(date --iso-8601=seconds)" >> "${log}"
else
  echo "event_guard_fail_stop_after_epoch1 $(date --iso-8601=seconds)" >> "${log}"
  kill -INT "${TRAIN_PID}"
fi
