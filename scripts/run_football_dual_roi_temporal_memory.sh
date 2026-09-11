#!/usr/bin/env bash
set -euo pipefail

GPU_LIST="${1:-3,4,6,7}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/football/dinov3_vitl16_dual_roi_temporal_memory_safe_neg_16f_hr.yaml}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-outputs/football_events/vitl16_dual_roi_shared_16f_hr/best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/football_events/vitl16_dual_roi_temporal_memory_safe_neg_16f_hr}"
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-8}"
TARGET_EFFECTIVE_BATCH_SIZE="${TARGET_EFFECTIVE_BATCH_SIZE:-64}"
TARGET_HEAD_LR="${TARGET_HEAD_LR:-0.0001}"
TARGET_BACKBONE_LR="${TARGET_BACKBONE_LR:-0.000008}"
EPOCHS="${EPOCHS:-4}"
NUM_WORKERS_PER_GPU="${NUM_WORKERS_PER_GPU:-1}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"

gpu_count="$("${PYTHON_BIN}" -c 'import sys; print(len([x for x in sys.argv[1].split(",") if x.strip()]))' "${GPU_LIST}")"
local_gpu_ids="$("${PYTHON_BIN}" -c 'import sys; n=int(sys.argv[1]); print("[" + ",".join(map(str, range(n))) + "]")' "${gpu_count}")"
grad_accum_steps="$("${PYTHON_BIN}" -c 'import math,sys; print(max(math.ceil(int(sys.argv[1])/(int(sys.argv[2])*int(sys.argv[3]))),1))' "${TARGET_EFFECTIVE_BATCH_SIZE}" "${PER_GPU_BATCH_SIZE}" "${gpu_count}")"
head_lr_per_gpu="$("${PYTHON_BIN}" -c 'import sys; print(float(sys.argv[1])/int(sys.argv[2]))' "${TARGET_HEAD_LR}" "${gpu_count}")"
backbone_lr_per_gpu="$("${PYTHON_BIN}" -c 'import sys; print(float(sys.argv[1])/int(sys.argv[2]))' "${TARGET_BACKBONE_LR}" "${gpu_count}")"

if [[ ! -f "${INIT_CHECKPOINT}" ]]; then
  echo "Missing completed dual-ROI init checkpoint: ${INIT_CHECKPOINT}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
log_file="${OUTPUT_DIR}/train_console.log"
effective_batch_size=$((PER_GPU_BATCH_SIZE * gpu_count * grad_accum_steps))

{
  echo
  echo "===== staggered dual-ROI temporal-memory fusion experiment ====="
  echo "started_at=$(date --iso-8601=seconds)"
  echo "physical_gpus=${GPU_LIST}"
  echo "config=${CONFIG}"
  echo "init_checkpoint=${INIT_CHECKPOINT}"
  echo "annotations_dir=/home/new_users/qiuqi/code/football_events_human_repair"
  echo "negative_sampling=coverage_balanced hard_negative=false
dual_sampling=multi_staggered fusion=dual_multi_roi_memory"
  echo "per_gpu_batch_size=${PER_GPU_BATCH_SIZE}"
  echo "effective_batch_size=${effective_batch_size}"
  echo "grad_accum_steps=${grad_accum_steps}"
  echo "head_lr_global=${TARGET_HEAD_LR}"
  echo "backbone_lr_global=${TARGET_BACKBONE_LR}"
} | tee -a "${log_file}"

common_args=(
  --config "${CONFIG}"
  "output_dir=${OUTPUT_DIR}"
  "gpu_ids=${local_gpu_ids}"
  "model.init_checkpoint=${INIT_CHECKPOINT}"
  "train.epochs=${EPOCHS}"
  "train.per_gpu_batch_size=${PER_GPU_BATCH_SIZE}"
  "train.grad_accum_steps=${grad_accum_steps}"
  "train.lr_per_gpu=${head_lr_per_gpu}"
  "train.backbone_lr_per_gpu=${backbone_lr_per_gpu}"
  "train.resume.enabled=false"
  "train.resume.load_optimizer=false"
  "train.resume.load_scheduler=false"
  "train.resume.load_scaler=false"
  "eval.per_gpu_batch_size=1"
  "data.num_workers_per_gpu=${NUM_WORKERS_PER_GPU}"
  "data.prefetch_factor=${PREFETCH_FACTOR}"
)

set +e
CUDA_VISIBLE_DEVICES="${GPU_LIST}" PYTHONUNBUFFERED=1 "${PYTHON_BIN}" \
  train_football_events.py "${common_args[@]}" \
  2>&1 | tee -a "${log_file}"
status=${PIPESTATUS[0]}
set -e

echo "finished_at=$(date --iso-8601=seconds) status=${status}" | tee -a "${log_file}"
exit "${status}"
