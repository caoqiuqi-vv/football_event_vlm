#!/usr/bin/env bash
set -euo pipefail

EXP_NAME="stage1_peak_spotting_v1_3_curriculum_small384"
DEFAULT_CONFIG="configs/football/dinov3_vitl16_stage1_peak_spotting_v1_3_curriculum_small384_raw720_no_pn_24f.yaml"
CONFIG="${CONFIG:-${DEFAULT_CONFIG}}"
OUTPUT_DIR="outputs/football_events/vitl16_stage1_peak_spotting_v1_3_curriculum_small384_raw720_no_pn_24f"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/${EXP_NAME}_train.log"
GPU_LIST="${GPU_LIST:-2,3,4,5,6,7}"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"
ln -sfn "$(pwd)/${LOG_FILE}" "${OUTPUT_DIR}/train_console.log"

echo "===== ${EXP_NAME} START $(date '+%F_%T') ====="
echo "config=${CONFIG}"
echo "output_dir=${OUTPUT_DIR}"
echo "gpu_list=${GPU_LIST}"
echo "init_checkpoint=outputs/football_events/vitl16_stage1_peak_spotting_v1_1_new_small384_raw720_no_pn_24f/best.pt"
echo "curriculum=stage1:1-5 clip-first; stage2:6-10 dense-ramp; stage3:11-20 high-recall-refine"

CUDA_VISIBLE_DEVICES="${GPU_LIST}" python train_football_events.py --config "${CONFIG}" 2>&1 | tee -a "${LOG_FILE}"

rc=${PIPESTATUS[0]}
echo "===== ${EXP_NAME} EXIT rc=${rc} $(date '+%F_%T') ====="
exit "${rc}"
