#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/new_users/qiuqi/code/dinov3-main"
config="${repo_dir}/configs/football/vitl16_720p_fromlast_e8_frame_tail_rank_v1.yaml"
output_dir="${OUTPUT_DIR:-${repo_dir}/outputs/football_events/vitl16_d7c_fullimage_720p_fromlast_e8_frame_tail_rank_v1_20260910}"
gpu_list="${CUDA_VISIBLE_DEVICES:-1,4}"
torchrun_bin="${TORCHRUN_BIN:-/home/new_users/qiuqi/miniconda3/envs/qiuqi_sam3/bin/torchrun}"

IFS=',' read -ra gpu_ids <<< "${gpu_list}"
world_size="${#gpu_ids[@]}"
mkdir -p "${output_dir}"
cd "${repo_dir}"
export CUDA_VISIBLE_DEVICES="${gpu_list}" PYTHONPATH="${repo_dir}" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

{
  echo "experiment=frame_tail_rank_v1 (T1: frame-level tail ranking on E16 anchor)"
  echo "config=${config}"
  echo "init_checkpoint=outputs/football_events/vitl16_d7c_fullimage_latest_split_dino_pretrained_lora_ddp_720p_fromlast_e8_20260829/best.pt"
  echo "world_size=${world_size} per_gpu_batch=2 grad_accum=20 effective_batch=80"
  echo "frame_tail_rank: weight=0.1 margin=1.0 tau=0.5 beta=1.0 ramp_epochs=1.0"
} | tee -a "${output_dir}/launch_console.log"

"${torchrun_bin}" --standalone --nproc_per_node="${world_size}" \
  train_football_events.py --config "${config}" \
  2>&1 | tee -a "${output_dir}/launch_console.log"
