#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-p1}"
GPU_LIST="${2:-3,4,6,7}"
PYTHON_BIN="${PYTHON_BIN:-python}"
P1_CONFIG="${P1_CONFIG:-configs/football/dinov3_vitl16_spatial_probe_p1_16f_hr.yaml}"
P2_CONFIG="${P2_CONFIG:-configs/football/dinov3_vitl16_spatial_probe_p2_fusion_16f_hr.yaml}"
P3_CONFIG="${P3_CONFIG:-configs/football/dinov3_vitl16_spatial_probe_p3_joint_16f_hr.yaml}"
P1_OUTPUT_DIR="${P1_OUTPUT_DIR:-outputs/football_events/vitl16_spatial_probe_p1_16f_hr}"
P2_OUTPUT_DIR="${P2_OUTPUT_DIR:-outputs/football_events/vitl16_spatial_probe_p2_fusion_16f_hr}"
P3_OUTPUT_DIR="${P3_OUTPUT_DIR:-outputs/football_events/vitl16_spatial_probe_p3_joint_16f_hr}"
P1_INIT="${P1_INIT_CHECKPOINT:-outputs/football_events/vitl16_strong_lora12_mlp_16f_hr_e1/best.pt}"

case "${MODE}" in
  p1|p2|p3|all) ;;
  *)
    echo "Usage: bash $0 [p1|p2|p3|all] [gpu_list]" >&2
    exit 2
    ;;
esac

gpu_count="$(${PYTHON_BIN} -c 'import sys; print(len([x for x in sys.argv[1].split(",") if x.strip()]))' "${GPU_LIST}")"
local_gpu_ids="$(${PYTHON_BIN} -c 'import sys; n=int(sys.argv[1]); print("[" + ",".join(map(str, range(n))) + "]")' "${gpu_count}")"

resolve_checkpoint() {
  local preferred="$1"
  local fallback="$2"
  if [[ -f "${preferred}" ]]; then
    printf '%s\n' "${preferred}"
  elif [[ -f "${fallback}" ]]; then
    printf '%s\n' "${fallback}"
  else
    echo "Missing checkpoint: ${preferred} (fallback ${fallback})" >&2
    return 2
  fi
}

run_stage() {
  local stage="$1"
  local config="$2"
  local init_checkpoint="$3"
  local output_dir="$4"
  local epochs="$5"
  local per_gpu_batch="$6"
  local target_effective_batch="$7"
  local global_head_lr="$8"
  local global_backbone_lr="$9"

  if [[ ! -f "${init_checkpoint}" ]]; then
    echo "Missing init checkpoint for ${stage}: ${init_checkpoint}" >&2
    exit 2
  fi
  local grad_accum
  local effective_batch
  local head_lr_per_gpu
  local backbone_lr_per_gpu
  grad_accum="$(${PYTHON_BIN} -c 'import math,sys; print(max(math.ceil(int(sys.argv[1])/(int(sys.argv[2])*int(sys.argv[3]))),1))' "${target_effective_batch}" "${per_gpu_batch}" "${gpu_count}")"
  effective_batch=$((per_gpu_batch * gpu_count * grad_accum))
  head_lr_per_gpu="$(${PYTHON_BIN} -c 'import sys; print(float(sys.argv[1])/int(sys.argv[2]))' "${global_head_lr}" "${gpu_count}")"
  backbone_lr_per_gpu="$(${PYTHON_BIN} -c 'import sys; print(float(sys.argv[1])/int(sys.argv[2]))' "${global_backbone_lr}" "${gpu_count}")"

  mkdir -p "${output_dir}"
  local log_file="${output_dir}/train_console.log"
  {
    echo
    echo "===== spatial probe pipeline: ${stage} ====="
    echo "started_at=$(date --iso-8601=seconds)"
    echo "physical_gpus=${GPU_LIST} local_gpu_ids=${local_gpu_ids}"
    echo "config=${config} init_checkpoint=${init_checkpoint}"
    echo "output_dir=${output_dir}"
    echo "per_gpu_batch=${per_gpu_batch} effective_batch=${effective_batch} grad_accum=${grad_accum}"
    echo "global_head_lr=${global_head_lr} global_backbone_lr=${global_backbone_lr}"
  } | tee -a "${log_file}"

  set +e
  CUDA_VISIBLE_DEVICES="${GPU_LIST}" PYTHONUNBUFFERED=1 "${PYTHON_BIN}" train_football_events.py \
    --config "${config}" \
    "output_dir=${output_dir}" \
    "gpu_ids=${local_gpu_ids}" \
    "model.init_checkpoint=${init_checkpoint}" \
    "train.epochs=${epochs}" \
    "train.per_gpu_batch_size=${per_gpu_batch}" \
    "train.grad_accum_steps=${grad_accum}" \
    "train.lr_per_gpu=${head_lr_per_gpu}" \
    "train.backbone_lr_per_gpu=${backbone_lr_per_gpu}" \
    "train.resume.enabled=false" \
    "eval.per_gpu_batch_size=${per_gpu_batch}" \
    "data.num_workers_per_gpu=${NUM_WORKERS_PER_GPU:-1}" \
    "data.prefetch_factor=${PREFETCH_FACTOR:-1}" \
    2>&1 | tee -a "${log_file}"
  local status=${PIPESTATUS[0]}
  set -e
  echo "finished_at=$(date --iso-8601=seconds) status=${status}" | tee -a "${log_file}"
  return "${status}"
}

run_p1() {
  run_stage p1 "${P1_CONFIG}" "${P1_INIT}" "${P1_OUTPUT_DIR}" \
    "${P1_EPOCHS:-4}" "${P1_PER_GPU_BATCH_SIZE:-8}" \
    "${P1_TARGET_EFFECTIVE_BATCH_SIZE:-64}" "${P1_HEAD_LR:-0.0001}" 0.0
}

run_p2() {
  local init
  init="$(resolve_checkpoint "${P1_OUTPUT_DIR}/best.pt" "${P1_OUTPUT_DIR}/last.pt")"
  run_stage p2 "${P2_CONFIG}" "${init}" "${P2_OUTPUT_DIR}" \
    "${P2_EPOCHS:-3}" "${P2_PER_GPU_BATCH_SIZE:-8}" \
    "${P2_TARGET_EFFECTIVE_BATCH_SIZE:-64}" "${P2_HEAD_LR:-0.00005}" 0.0
}

run_p3() {
  local init
  init="$(resolve_checkpoint "${P2_OUTPUT_DIR}/best.pt" "${P2_OUTPUT_DIR}/last.pt")"
  run_stage p3 "${P3_CONFIG}" "${init}" "${P3_OUTPUT_DIR}" \
    "${P3_EPOCHS:-6}" "${P3_PER_GPU_BATCH_SIZE:-6}" \
    "${P3_TARGET_EFFECTIVE_BATCH_SIZE:-48}" "${P3_HEAD_LR:-0.00005}" \
    "${P3_BACKBONE_LR:-0.000002}"
}

if [[ "${MODE}" == "p1" || "${MODE}" == "all" ]]; then run_p1; fi
if [[ "${MODE}" == "p2" || "${MODE}" == "all" ]]; then run_p2; fi
if [[ "${MODE}" == "p3" || "${MODE}" == "all" ]]; then run_p3; fi
