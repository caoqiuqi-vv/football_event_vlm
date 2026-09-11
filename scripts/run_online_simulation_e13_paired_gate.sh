#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/new_users/qiuqi/code/dinov3-main"
config_path="${repo_dir}/configs/football/dinov3_vitl16_online_simulation_e13_paired_gate_from_last6r8.yaml"
output_dir="${repo_dir}/outputs/football_events/vitl16_online_simulation_e13_paired_gate_from_last6r8_e1"
torchrun_bin="/home/new_users/qiuqi/miniconda3/envs/qiuqi_sam3/bin/torchrun"

mkdir -p "${output_dir}"
cd "${repo_dir}"

export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHONPATH="${repo_dir}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

"${torchrun_bin}" --standalone --nproc-per-node=4 \
  train_football_events_online_simulation_e13.py \
  --config "${config_path}" \
  2>&1 | tee -a "${output_dir}/train_console.log"
