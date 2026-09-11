#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:?Usage: $0 OUTPUT_DIR TRAIN_PID MAX_EPOCH [BASELINE] [REFERENCE]}"
TRAIN_PID="${2:?missing TRAIN_PID}"
MAX_EPOCH="${3:?missing MAX_EPOCH}"
BASELINE="${4:-/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt}"
REFERENCE="${5:-}"
POLL_SEC="${POLL_SEC:-30}"

mkdir -p "${OUTPUT_DIR}"
LOG="${OUTPUT_DIR}/metric_monitor.log"
echo "monitor_started_at=$(date --iso-8601=seconds) train_pid=${TRAIN_PID}" >> "${LOG}"

for epoch in $(seq 1 "${MAX_EPOCH}"); do
  checkpoint="${OUTPUT_DIR}/epoch_${epoch}.pt"
  while [[ ! -s "${checkpoint}" ]]; do
    if ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
      echo "train_exited_before_epoch=${epoch} at=$(date --iso-8601=seconds)" >> "${LOG}"
      exit 2
    fi
    sleep "${POLL_SEC}"
  done

  candidates=("${checkpoint}")
  if [[ -n "${REFERENCE}" && -s "${REFERENCE}" ]]; then
    candidates=("${REFERENCE}" "${checkpoint}")
  fi
  report="${OUTPUT_DIR}/epoch_${epoch}_vs_e1"
  python scripts/compare_football_checkpoint_metrics.py \
    --baseline "${BASELINE}" \
    --candidates "${candidates[@]}" \
    --labels shot,save,set_piece \
    --recall-tolerance-pp 1 \
    --output "${report}.json" \
    > "${report}.md" 2> "${report}.err"
  echo "compared_epoch=${epoch} at=$(date --iso-8601=seconds)" >> "${LOG}"
done

echo "monitor_completed_at=$(date --iso-8601=seconds)" >> "${LOG}"
