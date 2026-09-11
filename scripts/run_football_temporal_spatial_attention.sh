#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
GPU_LIST="${2:-3,4,6,7}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/football/dinov3_vitl16_temporal_spatial_attention_16f_hr.yaml}"
BASE_INIT_CHECKPOINT="${INIT_CHECKPOINT:-outputs/football_events/vitl16_strong_lora12_mlp_16f_hr_e1/best.pt}"
WARMUP_OUTPUT_DIR="${WARMUP_OUTPUT_DIR:-outputs/football_events/vitl16_temporal_spatial_attention_v2_region_only_16f_hr_warmup}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/football_events/vitl16_temporal_spatial_attention_v2_region_only_16f_hr}"
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-6}"
TARGET_EFFECTIVE_BATCH_SIZE="${TARGET_EFFECTIVE_BATCH_SIZE:-48}"
NUM_WORKERS_PER_GPU="${NUM_WORKERS_PER_GPU:-1}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-1}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-5}"
WARMUP_HEAD_LR="${WARMUP_HEAD_LR:-0.0003}"
FINETUNE_HEAD_LR="${FINETUNE_HEAD_LR:-0.00005}"
FINETUNE_BACKBONE_LR="${FINETUNE_BACKBONE_LR:-0.000005}"

case "${MODE}" in
  warmup|finetune|all) ;;
  *)
    echo "Usage: bash $0 [warmup|finetune|all] [gpu_list]" >&2
    exit 2
    ;;
esac

gpu_count="$("${PYTHON_BIN}" -c 'import sys; print(len([x for x in sys.argv[1].split(",") if x.strip()]))' "${GPU_LIST}")"
local_gpu_ids="$("${PYTHON_BIN}" -c 'import sys; n=int(sys.argv[1]); print("[" + ",".join(map(str, range(n))) + "]")' "${gpu_count}")"
grad_accum_steps="$("${PYTHON_BIN}" -c 'import math,sys; print(max(math.ceil(int(sys.argv[1])/(int(sys.argv[2])*int(sys.argv[3]))),1))' "${TARGET_EFFECTIVE_BATCH_SIZE}" "${PER_GPU_BATCH_SIZE}" "${gpu_count}")"
effective_batch_size=$((PER_GPU_BATCH_SIZE * gpu_count * grad_accum_steps))

run_stage() {
  local stage="$1"
  local init_checkpoint="$2"
  local output_dir="$3"
  local epochs="$4"
  local global_head_lr="$5"
  local global_backbone_lr="$6"
  local freeze_loaded_backbone="$7"
  local freeze_global_branch="$8"

  if [[ ! -f "${init_checkpoint}" ]]; then
    echo "Missing init checkpoint for stage ${stage}: ${init_checkpoint}" >&2
    exit 2
  fi

  local head_lr_per_gpu
  local backbone_lr_per_gpu
  head_lr_per_gpu="$("${PYTHON_BIN}" -c 'import sys; print(float(sys.argv[1])/int(sys.argv[2]))' "${global_head_lr}" "${gpu_count}")"
  backbone_lr_per_gpu="$("${PYTHON_BIN}" -c 'import sys; print(float(sys.argv[1])/int(sys.argv[2]))' "${global_backbone_lr}" "${gpu_count}")"

  mkdir -p "${output_dir}"
  local log_file="${output_dir}/train_console.log"
  {
    echo
    echo "===== temporal-conditioned spatial attention: ${stage} ====="
    echo "started_at=$(date --iso-8601=seconds)"
    echo "physical_gpus=${GPU_LIST} local_gpu_ids=${local_gpu_ids}"
    echo "config=${CONFIG}"
    echo "init_checkpoint=${init_checkpoint}"
    echo "output_dir=${output_dir}"
    echo "per_gpu_batch_size=${PER_GPU_BATCH_SIZE} effective_batch_size=${effective_batch_size} grad_accum=${grad_accum_steps}"
    echo "global_head_lr=${global_head_lr} global_backbone_lr=${global_backbone_lr}"
    echo "freeze_loaded_backbone=${freeze_loaded_backbone} freeze_global_branch=${freeze_global_branch}"
    echo "spatial_attention=v2_region_only class_queries=2 residual_init=0 gate_init=0.5 entropy_target=0.85"
  } | tee -a "${log_file}"

  set +e
  CUDA_VISIBLE_DEVICES="${GPU_LIST}" PYTHONUNBUFFERED=1 "${PYTHON_BIN}" train_football_events.py \
    --config "${CONFIG}" \
    "output_dir=${output_dir}" \
    "gpu_ids=${local_gpu_ids}" \
    "model.init_checkpoint=${init_checkpoint}" \
    "model.freeze_loaded_backbone=${freeze_loaded_backbone}" \
    "model.freeze_global_branch=${freeze_global_branch}" \
    "train.epochs=${epochs}" \
    "train.per_gpu_batch_size=${PER_GPU_BATCH_SIZE}" \
    "train.grad_accum_steps=${grad_accum_steps}" \
    "train.lr_per_gpu=${head_lr_per_gpu}" \
    "train.backbone_lr_per_gpu=${backbone_lr_per_gpu}" \
    "train.resume.enabled=false" \
    "eval.per_gpu_batch_size=${PER_GPU_BATCH_SIZE}" \
    "data.num_workers_per_gpu=${NUM_WORKERS_PER_GPU}" \
    "data.prefetch_factor=${PREFETCH_FACTOR}" \
    2>&1 | tee -a "${log_file}"
  local status=${PIPESTATUS[0]}
  set -e
  echo "finished_at=$(date --iso-8601=seconds) status=${status}" | tee -a "${log_file}"
  return "${status}"
}

if [[ "${MODE}" == "warmup" || "${MODE}" == "all" ]]; then
  run_stage \
    warmup \
    "${BASE_INIT_CHECKPOINT}" \
    "${WARMUP_OUTPUT_DIR}" \
    "${WARMUP_EPOCHS}" \
    "${WARMUP_HEAD_LR}" \
    0.0 \
    true \
    true
fi

if [[ "${MODE}" == "finetune" || "${MODE}" == "all" ]]; then
  warmup_checkpoint="${WARMUP_CHECKPOINT:-${WARMUP_OUTPUT_DIR}/best.pt}"
  if [[ ! -f "${warmup_checkpoint}" && -f "${WARMUP_OUTPUT_DIR}/last.pt" ]]; then
    warmup_checkpoint="${WARMUP_OUTPUT_DIR}/last.pt"
  fi
  run_stage \
    finetune \
    "${warmup_checkpoint}" \
    "${OUTPUT_DIR}" \
    "${FINETUNE_EPOCHS}" \
    "${FINETUNE_HEAD_LR}" \
    "${FINETUNE_BACKBONE_LR}" \
    false \
    false
fi

