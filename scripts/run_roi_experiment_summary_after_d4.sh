#!/usr/bin/env bash
set -euo pipefail

WAIT_PID="${WAIT_PID:-}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/football_roi_experiments/d3_d4_long_protocol_summary}"
LOG="${OUTPUT_DIR}/summary_watcher.log"
BASELINE_RUN="${BASELINE_RUN:-outputs/football_eval_runs/vitl16_lora_r5_f32_16f_e1_frame_det_last_hr_6videos_window_overlap_checkpoint_thr}"
TOPK_RUN="${TOPK_RUN:-outputs/football_eval_runs/topk_exp_epoch13_6videos_dense_checkpoint_thr}"
D3_RUN="${D3_RUN:-outputs/football_eval_runs/vitl16_robust_dual_16f_e1_dynamic_roi_lora_d3_6videos_window_overlap_checkpoint_thr}"
D4_RUN="${D4_RUN:-outputs/football_eval_runs/vitl16_robust_dual_16f_e1_dynamic_roi_decoupled_lora_d4_6videos_window_overlap_checkpoint_thr}"

mkdir -p "${OUTPUT_DIR}"
echo "summary_watcher_started_at=$(date --iso-8601=seconds) wait_pid=${WAIT_PID}" >> "${LOG}"
if [[ -n "${WAIT_PID}" ]]; then
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep 60
  done
  echo "wait_pid_finished_at=$(date --iso-8601=seconds) wait_pid=${WAIT_PID}" >> "${LOG}"
fi

python scripts/summarize_football_long_protocol_runs.py \
  --run "baseline=${BASELINE_RUN}" \
  --run "topk_epoch13=${TOPK_RUN}" \
  --run "d3_shared_roi_lora=${D3_RUN}" \
  --run "d4_decoupled_roi_lora=${D4_RUN}" \
  --output-dir "${OUTPUT_DIR}" \
  --allow-missing >> "${LOG}" 2>&1

echo "summary_watcher_completed_at=$(date --iso-8601=seconds) output_dir=${OUTPUT_DIR}" >> "${LOG}"
