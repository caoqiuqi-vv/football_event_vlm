#!/usr/bin/env bash
set -euo pipefail

WAIT_PID="${WAIT_PID:-}"
GPU_LIST="${GPU_LIST:-0,1,2,6,7}"
D4_OUTPUT_DIR="${D4_OUTPUT_DIR:-outputs/football_events/vitl16_robust_dual_16f_e1_dynamic_roi_decoupled_lora_d4}"
D4_INIT_CHECKPOINT="${D4_INIT_CHECKPOINT:-/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt}"
D4_TRAIN_EPOCHS="${D4_TRAIN_EPOCHS:-3}"
D4_PER_GPU_BATCH_SIZE="${D4_PER_GPU_BATCH_SIZE:-1}"
D4_GRAD_ACCUM_STEPS="${D4_GRAD_ACCUM_STEPS:-16}"
D4_LR_PER_GPU="${D4_LR_PER_GPU:-0.00003}"
D4_BACKBONE_LR_PER_GPU="${D4_BACKBONE_LR_PER_GPU:-0.000006}"
RUN_NAME="${RUN_NAME:-vitl16_robust_dual_16f_e1_dynamic_roi_decoupled_lora_d4_6videos_window_overlap_checkpoint_thr}"
LOG="${D4_OUTPUT_DIR}/after_d3_console.log"

mkdir -p "${D4_OUTPUT_DIR}"
echo "d4_after_d3_started_at=$(date --iso-8601=seconds) wait_pid=${WAIT_PID} gpu_list=${GPU_LIST}" >> "${LOG}"

if [[ -n "${WAIT_PID}" ]]; then
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep 60
  done
  echo "wait_pid_finished_at=$(date --iso-8601=seconds) wait_pid=${WAIT_PID}" >> "${LOG}"
fi

TRAIN_EPOCHS="${D4_TRAIN_EPOCHS}" \
PER_GPU_BATCH_SIZE="${D4_PER_GPU_BATCH_SIZE}" \
EVAL_PER_GPU_BATCH_SIZE="${D4_PER_GPU_BATCH_SIZE}" \
GRAD_ACCUM_STEPS="${D4_GRAD_ACCUM_STEPS}" \
LR_PER_GPU="${D4_LR_PER_GPU}" \
BACKBONE_LR_PER_GPU="${D4_BACKBONE_LR_PER_GPU}" \
INIT_CHECKPOINT="${D4_INIT_CHECKPOINT}" \
bash scripts/launch_football_detection_aware.sh d4_dynamic_roi_decoupled_lora "${GPU_LIST}" >> "${LOG}" 2>&1

echo "d4_train_finished_at=$(date --iso-8601=seconds)" >> "${LOG}"

OUTPUT_DIR="${D4_OUTPUT_DIR}" \
CHECKPOINT="${D4_OUTPUT_DIR}/best.pt" \
GPU_LIST="${GPU_LIST}" \
RUN_NAME="${RUN_NAME}" \
bash scripts/run_dynamic_roi_lora_d3_post_eval.sh >> "${LOG}" 2>&1

echo "d4_after_d3_completed_at=$(date --iso-8601=seconds)" >> "${LOG}"
