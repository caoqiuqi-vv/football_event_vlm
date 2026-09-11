#!/usr/bin/env bash
set -euo pipefail

GPU_LIST="${1:-3,4,6,7}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/football/dinov3_vitl16_strong_lora12_mlp_16f_hr_e1.yaml}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/football_events/vitl16_strong_lora12_mlp_16f_hr_e1}"
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-6}"
TARGET_EFFECTIVE_BATCH_SIZE="${TARGET_EFFECTIVE_BATCH_SIZE:-96}"
TARGET_HEAD_LR="${TARGET_HEAD_LR:-0.0002}"
TARGET_BACKBONE_LR="${TARGET_BACKBONE_LR:-0.000015}"
EPOCHS="${EPOCHS:-6}"
NUM_WORKERS_PER_GPU="${NUM_WORKERS_PER_GPU:-2}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
RESUME_TRAINING="${RESUME_TRAINING:-false}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-${OUTPUT_DIR}/last.pt}"
RESUME_STRICT="${RESUME_STRICT:-true}"
RESUME_LOAD_OPTIMIZER="${RESUME_LOAD_OPTIMIZER:-false}"
RESUME_LOAD_SCHEDULER="${RESUME_LOAD_SCHEDULER:-false}"
RESUME_LOAD_SCALER="${RESUME_LOAD_SCALER:-false}"
RESUME_RESET_OPTIMIZER_LR="${RESUME_RESET_OPTIMIZER_LR:-false}"

num_gpus() {
  "${PYTHON_BIN}" -c 'import sys; print(len([x for x in sys.argv[1].split(",") if x.strip()]))' "$1"
}

gpu_count="$(num_gpus "${GPU_LIST}")"
local_gpu_ids="$("${PYTHON_BIN}" -c 'import sys; n=int(sys.argv[1]); print("[" + ",".join(map(str, range(n))) + "]")' "${gpu_count}")"
grad_accum_steps="$("${PYTHON_BIN}" -c 'import math,sys; print(max(math.ceil(int(sys.argv[1])/(int(sys.argv[2])*int(sys.argv[3]))),1))' "${TARGET_EFFECTIVE_BATCH_SIZE}" "${PER_GPU_BATCH_SIZE}" "${gpu_count}")"
head_lr_per_gpu="$("${PYTHON_BIN}" -c 'import sys; print(float(sys.argv[1])/int(sys.argv[2]))' "${TARGET_HEAD_LR}" "${gpu_count}")"
backbone_lr_per_gpu="$("${PYTHON_BIN}" -c 'import sys; print(float(sys.argv[1])/int(sys.argv[2]))' "${TARGET_BACKBONE_LR}" "${gpu_count}")"
effective_batch_size=$((PER_GPU_BATCH_SIZE * gpu_count * grad_accum_steps))

if [[ ! -f "${INIT_CHECKPOINT}" ]]; then
  echo "Missing init checkpoint: ${INIT_CHECKPOINT}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
log_file="${OUTPUT_DIR}/train_console.log"
{
  echo
  echo "===== A strong single-view LoRA last12+MLP 16f 640x1120 ====="
  echo "started_at=$(date --iso-8601=seconds)"
  echo "physical_gpus=${GPU_LIST} local_gpu_ids=${local_gpu_ids}"
  echo "config=${CONFIG}"
  echo "init_checkpoint=${INIT_CHECKPOINT}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "per_gpu_batch_size=${PER_GPU_BATCH_SIZE} effective_batch_size=${effective_batch_size} grad_accum=${grad_accum_steps}"
  echo "head_lr_global=${TARGET_HEAD_LR} backbone_lr_global=${TARGET_BACKBONE_LR}"
  echo "strong_lora=last12,qkv,proj,mlp_fc1,mlp_fc2,train_norm"
  echo "hard_negative=false ranking=false"
  echo "resume=${RESUME_TRAINING} resume_checkpoint=${RESUME_CHECKPOINT}"
  echo "resume_load_optimizer=${RESUME_LOAD_OPTIMIZER} resume_load_scheduler=${RESUME_LOAD_SCHEDULER} resume_reset_optimizer_lr=${RESUME_RESET_OPTIMIZER_LR}"
} | tee -a "${log_file}"

set +e
CUDA_VISIBLE_DEVICES="${GPU_LIST}" PYTHONUNBUFFERED=1 "${PYTHON_BIN}" train_football_events.py \
  --config "${CONFIG}" \
  "output_dir=${OUTPUT_DIR}" \
  "gpu_ids=${local_gpu_ids}" \
  "model.init_checkpoint=${INIT_CHECKPOINT}" \
  "train.epochs=${EPOCHS}" \
  "train.per_gpu_batch_size=${PER_GPU_BATCH_SIZE}" \
  "train.grad_accum_steps=${grad_accum_steps}" \
  "train.lr_per_gpu=${head_lr_per_gpu}" \
  "train.backbone_lr_per_gpu=${backbone_lr_per_gpu}" \
  "train.resume.enabled=${RESUME_TRAINING}" \
  "train.resume.checkpoint=${RESUME_CHECKPOINT}" \
  "train.resume.strict=${RESUME_STRICT}" \
  "train.resume.load_optimizer=${RESUME_LOAD_OPTIMIZER}" \
  "train.resume.load_scheduler=${RESUME_LOAD_SCHEDULER}" \
  "train.resume.load_scaler=${RESUME_LOAD_SCALER}" \
  "train.resume.reset_optimizer_lr=${RESUME_RESET_OPTIMIZER_LR}" \
  "eval.per_gpu_batch_size=${PER_GPU_BATCH_SIZE}" \
  "data.num_workers_per_gpu=${NUM_WORKERS_PER_GPU}" \
  "data.prefetch_factor=${PREFETCH_FACTOR}" \
  2>&1 | tee -a "${log_file}"
status=${PIPESTATUS[0]}
set -e

echo "finished_at=$(date --iso-8601=seconds) status=${status}" | tee -a "${log_file}"
exit "${status}"
