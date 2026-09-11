#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/new_users/qiuqi/miniconda3/envs/qiuqi_sam3/bin/python}"
GPU_LIST="${1:-0,1,2,3}"
CONFIG="${CONFIG:-configs/football/dinov3_vitl16_ssl_task_head_transfer_last4_16f_hr.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/football_events/vitl16_ssl4999_task_head_calibration_2e}"

cd "${ROOT_DIR}"
mkdir -p "${OUTPUT_DIR}"
{
  echo "===== SSL task-head-only calibration ====="
  echo "started_at=$(date --iso-8601=seconds)"
  echo "physical_gpus=${GPU_LIST}"
  echo "config=${CONFIG}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "policy=freeze complete SSL/LoRA backbone; adapt migrated task head for 2 epochs"
} | tee -a "${OUTPUT_DIR}/train_console.log"

set +e
CUDA_VISIBLE_DEVICES="${GPU_LIST}" "${PYTHON_BIN}" train_football_events.py \
  --config "${CONFIG}" \
  output_dir="${OUTPUT_DIR}" \
  cache.dir="outputs/football_dino_features/vitl16_ssl4999_task_head_calibration_2e" \
  model.freeze_loaded_backbone=true \
  train.epochs=2 \
  train.resume.enabled=false \
  2>&1 | tee -a "${OUTPUT_DIR}/train_console.log"
status=${PIPESTATUS[0]}
set -e

{
  echo "finished_at=$(date --iso-8601=seconds)"
  echo "status=${status}"
} | tee -a "${OUTPUT_DIR}/train_console.log"
exit "${status}"
