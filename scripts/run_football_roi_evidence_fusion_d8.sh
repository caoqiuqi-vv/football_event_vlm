#!/usr/bin/env bash
set -euo pipefail

GPU_LIST="${1:-0,1,2,3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/football/dinov3_vitl16_d8_roi_evidence_fusion_from_strong_lora.yaml}"
DEFAULT_STRONG_INIT="outputs/football_events/vitl16_strong_lora12_mlp_16f_hr_e1/best.pt"
FALLBACK_INIT="/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt"
if [[ -z "${INIT_CHECKPOINT:-}" ]]; then
  if [[ -f "${DEFAULT_STRONG_INIT}" ]]; then
    INIT_CHECKPOINT="${DEFAULT_STRONG_INIT}"
  else
    INIT_CHECKPOINT="${FALLBACK_INIT}"
  fi
fi
OUTPUT_DIR="${OUTPUT_DIR:-outputs/football_events/vitl16_d8_roi_evidence_fusion_from_strong_lora}"
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-2}"
TARGET_EFFECTIVE_BATCH_SIZE="${TARGET_EFFECTIVE_BATCH_SIZE:-64}"
TARGET_HEAD_LR="${TARGET_HEAD_LR:-0.0001}"
TARGET_GLOBAL_BACKBONE_LR="${TARGET_GLOBAL_BACKBONE_LR:-0.0}"
TARGET_LOCAL_BACKBONE_LR="${TARGET_LOCAL_BACKBONE_LR:-0.000015}"
EPOCHS="${EPOCHS:-5}"
NUM_WORKERS_PER_GPU="${NUM_WORKERS_PER_GPU:-2}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
POSITIVE_RETENTION_LOSS_WEIGHT="${POSITIVE_RETENTION_LOSS_WEIGHT:-0.5}"
FRAME_DET_LOSS_WEIGHT="${FRAME_DET_LOSS_WEIGHT:-0.3}"

num_gpus() { "${PYTHON_BIN}" -c 'import sys; print(len([x for x in sys.argv[1].split(",") if x.strip()]))' "$1"; }
gpu_count="$(num_gpus "${GPU_LIST}")"
local_gpu_ids="$("${PYTHON_BIN}" -c 'import sys; n=int(sys.argv[1]); print("[" + ",".join(map(str, range(n))) + "]")' "${gpu_count}")"
grad_accum_steps="$("${PYTHON_BIN}" -c 'import math,sys; print(max(math.ceil(int(sys.argv[1])/(int(sys.argv[2])*int(sys.argv[3]))),1))' "${TARGET_EFFECTIVE_BATCH_SIZE}" "${PER_GPU_BATCH_SIZE}" "${gpu_count}")"
head_lr_per_gpu="$("${PYTHON_BIN}" -c 'import sys; print(float(sys.argv[1])/int(sys.argv[2]))' "${TARGET_HEAD_LR}" "${gpu_count}")"
global_backbone_lr_per_gpu="$("${PYTHON_BIN}" -c 'import sys; print(float(sys.argv[1])/int(sys.argv[2]))' "${TARGET_GLOBAL_BACKBONE_LR}" "${gpu_count}")"
local_backbone_lr_per_gpu="$("${PYTHON_BIN}" -c 'import sys; print(float(sys.argv[1])/int(sys.argv[2]))' "${TARGET_LOCAL_BACKBONE_LR}" "${gpu_count}")"
effective_batch_size=$((PER_GPU_BATCH_SIZE * gpu_count * grad_accum_steps))

if [[ ! -f "${INIT_CHECKPOINT}" ]]; then
  echo "Missing init checkpoint: ${INIT_CHECKPOINT}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
log_file="${OUTPUT_DIR}/train_console.log"
{
  echo
  echo "===== C/D8 ROI evidence fusion: frozen global + trainable local ROI evidence ====="
  echo "started_at=$(date --iso-8601=seconds)"
  echo "physical_gpus=${GPU_LIST} local_gpu_ids=${local_gpu_ids}"
  echo "config=${CONFIG}"
  echo "init_checkpoint=${INIT_CHECKPOINT}"
  echo "output_dir=${OUTPUT_DIR}"
  echo "per_gpu_batch_size=${PER_GPU_BATCH_SIZE} effective_batch_size=${effective_batch_size} grad_accum=${grad_accum_steps}"
  echo "head_lr_global=${TARGET_HEAD_LR} global_backbone_lr=${TARGET_GLOBAL_BACKBONE_LR} local_backbone_lr=${TARGET_LOCAL_BACKBONE_LR}"
  echo "fusion=dual_cross_attention global_frozen=true local_loss=0 positive_retention=${POSITIVE_RETENTION_LOSS_WEIGHT}"
} | tee -a "${log_file}"

set +e
CUDA_VISIBLE_DEVICES="${GPU_LIST}" PYTHONUNBUFFERED=1 "${PYTHON_BIN}" train_football_events.py \
  --config "${CONFIG}" \
  "output_dir=${OUTPUT_DIR}" \
  "gpu_ids=${local_gpu_ids}" \
  "model.init_checkpoint=${INIT_CHECKPOINT}" \
  "model.freeze_global_branch=true" \
  "model.freeze_loaded_backbone=false" \
  "train.epochs=${EPOCHS}" \
  "train.per_gpu_batch_size=${PER_GPU_BATCH_SIZE}" \
  "train.grad_accum_steps=${grad_accum_steps}" \
  "train.lr_per_gpu=${head_lr_per_gpu}" \
  "train.backbone_lr_per_gpu=${local_backbone_lr_per_gpu}" \
  "train.global_backbone_lr_per_gpu=${global_backbone_lr_per_gpu}" \
  "train.local_backbone_lr_per_gpu=${local_backbone_lr_per_gpu}" \
  "train.local_loss_weight=0.0" \
  "train.frame_det_loss_weight=${FRAME_DET_LOSS_WEIGHT}" \
  "train.positive_retention_loss_weight=${POSITIVE_RETENTION_LOSS_WEIGHT}" \
  "train.resume.enabled=false" \
  "train.resume.load_optimizer=false" \
  "train.resume.load_scheduler=false" \
  "train.resume.load_scaler=false" \
  "eval.per_gpu_batch_size=${PER_GPU_BATCH_SIZE}" \
  "data.num_workers_per_gpu=${NUM_WORKERS_PER_GPU}" \
  "data.prefetch_factor=${PREFETCH_FACTOR}" \
  2>&1 | tee -a "${log_file}"
status=${PIPESTATUS[0]}
set -e

echo "finished_at=$(date --iso-8601=seconds) status=${status}" | tee -a "${log_file}"
exit "${status}"
