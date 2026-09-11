#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/new_users/qiuqi/code/dinov3-main"
base_config="${repo_dir}/configs/football/dinov3_vitl16_online_simulation_e14_signed_causal_continue4.yaml"
e14_dir="${repo_dir}/outputs/football_events/vitl16_online_simulation_e14_signed_causal_localgrad_from_last6r8_e1"
init_checkpoint="${E14_720_INIT_CHECKPOINT:-${e14_dir}/best.pt}"
output_dir="${E14_720_OUTPUT_DIR:-${repo_dir}/outputs/football_events/vitl16_online_simulation_e14_720p_pilot_from_e14_best}"
gpu_list="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
torchrun_bin="/home/new_users/qiuqi/miniconda3/envs/qiuqi_sam3/bin/torchrun"

if [[ ! -f "${init_checkpoint}" ]]; then
  echo "Missing 720P pilot init checkpoint: ${init_checkpoint}" >&2
  exit 2
fi

mkdir -p "${output_dir}"
cd "${repo_dir}"

export CUDA_VISIBLE_DEVICES="${gpu_list}"
export PYTHONPATH="${repo_dir}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

"${torchrun_bin}" --standalone --nproc-per-node=4 \
  train_football_events_online_simulation_e13.py \
  --config "${base_config}" \
  "output_dir=${output_dir}" \
  "model.init_checkpoint=${init_checkpoint}" \
  "model.init_checkpoint_strict=false" \
  "train.resume.enabled=false" \
  "train.epochs=4" \
  "train.scheduler_epochs=6" \
  "train.early_stopping.enabled=true" \
  "train.early_stopping.min_epoch=2" \
  "train.early_stopping.patience=3" \
  "video.image_size=[720,1280]" \
  "train.per_gpu_batch_size=4" \
  "train.grad_accum_steps=4" \
  "eval.per_gpu_batch_size=2" \
  "data.num_workers_per_gpu=3" \
  2>&1 | tee -a "${output_dir}/train_console.log"

if [[ "${RUN_ONLINE_VAL_AFTER_TRAIN:-1}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${gpu_list}" \
    ONLINE_EVAL_WORLD_SIZE=4 \
    ONLINE_EVAL_PER_GPU_BATCH=2 \
    ONLINE_EVAL_IMAGE_SIZE='[720,1280]' \
    bash "${repo_dir}/scripts/evaluate_online_val15_checkpoint.sh" \
      "${base_config}" "${output_dir}/best.pt" \
      "${output_dir}/online_val15_best.json"
fi
