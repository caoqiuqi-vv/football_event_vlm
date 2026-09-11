#!/usr/bin/env bash
set -euo pipefail

# ── Plan A: spatial_attention + multi_layer_readout ──
# Phase 1: r=8 for 5 epochs
# Phase 2 (if no improvement): r=16

GPU_LIST="${1:-3,4,6,7}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CONFIG_R8="configs/football/dinov3_vitl16_spatial_attn_multilayer_readout.yaml"
CONFIG_R16="configs/football/dinov3_vitl16_spatial_attn_multilayer_readout_r16.yaml"

PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-6}"
TARGET_EFFECTIVE_BATCH_SIZE="${TARGET_EFFECTIVE_BATCH_SIZE:-48}"
NUM_WORKERS_PER_GPU="${NUM_WORKERS_PER_GPU:-1}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"

gpu_count="$("${PYTHON_BIN}" -c 'import sys; print(len([x for x in sys.argv[1].split(",") if x.strip()]))' "${GPU_LIST}")"
local_gpu_ids="$("${PYTHON_BIN}" -c 'import sys; n=int(sys.argv[1]); print("[" + ",".join(map(str, range(n))) + "]")' "${gpu_count}")"
grad_accum_steps="$("${PYTHON_BIN}" -c 'import math,sys; print(max(math.ceil(int(sys.argv[1])/(int(sys.argv[2])*int(sys.argv[3]))),1))' "${TARGET_EFFECTIVE_BATCH_SIZE}" "${PER_GPU_BATCH_SIZE}" "${gpu_count}")"
effective_batch_size=$((PER_GPU_BATCH_SIZE * gpu_count * grad_accum_steps))

run_plan_a() {
  local config_path="$1"
  local label="$2"
  local config_name
  config_name="$(basename "${config_path}" .yaml)"

  local output_dir
  output_dir="$("${PYTHON_BIN}" -c 'import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))["output_dir"])' "${config_path}")"

  local init_ckpt
  init_ckpt="$("${PYTHON_BIN}" -c 'import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))["model"].get("init_checkpoint","") or "")' "${config_path}")"

  if [[ -n "${init_ckpt}" && ! -f "${init_ckpt}" ]]; then
    echo "ERROR: init_checkpoint not found: ${init_ckpt}" >&2
    exit 2
  fi

  mkdir -p "${output_dir}"
  local log_file="${output_dir}/train_console.log"

  {
    echo
    echo "===== Plan A: spatial_attention + multi_layer_readout : ${label} ====="
    echo "started_at=$(date --iso-8601=seconds)"
    echo "physical_gpus=${GPU_LIST} local_gpu_ids=${local_gpu_ids}"
    echo "config=${config_path}"
    echo "init_checkpoint=${init_ckpt}"
    echo "output_dir=${output_dir}"
    echo "per_gpu_batch_size=${PER_GPU_BATCH_SIZE} effective_batch_size=${effective_batch_size} grad_accum=${grad_accum_steps}"
  } | tee -a "${log_file}"

  set +e
  CUDA_VISIBLE_DEVICES="${GPU_LIST}" PYTHONUNBUFFERED=1 "${PYTHON_BIN}" train_football_events.py \
    --config "${config_path}" \
    "gpu_ids=${local_gpu_ids}" \
    "train.per_gpu_batch_size=${PER_GPU_BATCH_SIZE}" \
    "train.grad_accum_steps=${grad_accum_steps}" \
    "eval.per_gpu_batch_size=${PER_GPU_BATCH_SIZE}" \
    "data.num_workers_per_gpu=${NUM_WORKERS_PER_GPU}" \
    "data.prefetch_factor=${PREFETCH_FACTOR}" \
    2>&1 | tee -a "${log_file}"
  local status=${PIPESTATUS[0]}
  set -e
  echo "finished_at=$(date --iso-8601=seconds) status=${status}" | tee -a "${log_file}"
  return "${status}"
}

echo "=== Plan A Phase 1: spatial_attention + multi_layer_readout r=8 ==="
run_plan_a "${CONFIG_R8}" "r8_5ep"

echo
echo "Phase 1 complete. Check metrics to decide whether to run r=16."
echo "To launch r=16 fallback:"
echo "  bash scripts/run_plan_a_spatial_multilayer.sh r16"
echo

if [[ "${2:-}" == "r16" ]]; then
  echo "=== Plan A Phase 2: spatial_attention + multi_layer_readout r=16 ==="
  run_plan_a "${CONFIG_R16}" "r16_5ep"
fi
