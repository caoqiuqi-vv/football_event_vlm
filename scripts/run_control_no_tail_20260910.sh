#!/usr/bin/env bash
# T1 对照组:E16 续训,不加尾部排序损失(frame_tail_rank_loss_weight=0.0)
# 与 T1 唯一差异即损失开关,用于干净的单变量归因。
set -euo pipefail

repo_dir="/home/new_users/qiuqi/code/dinov3-main"
config="${repo_dir}/configs/football/vitl16_720p_fromlast_e8_control_no_tail_20260910.yaml"
output_dir="${OUTPUT_DIR:-${repo_dir}/outputs/football_events/vitl16_d7c_fullimage_720p_fromlast_e8_control_no_tail_20260910}"
gpu_list="${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to 2 free gpu ids}"
torchrun_bin="${TORCHRUN_BIN:-/home/new_users/qiuqi/miniconda3/envs/qiuqi_sam3/bin/torchrun}"

IFS=',' read -ra gpu_ids <<< "${gpu_list}"
world_size="${#gpu_ids[@]}"
mkdir -p "${output_dir}"
cd "${repo_dir}"
export CUDA_VISIBLE_DEVICES="${gpu_list}" PYTHONPATH="${repo_dir}" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

{
  echo "experiment=control_no_tail (T1 control: E16 continuation without tail rank loss)"
  echo "config=${config}"
  echo "init_checkpoint=outputs/football_events/vitl16_d7c_fullimage_latest_split_dino_pretrained_lora_ddp_720p_fromlast_e8_20260829/best.pt"
  echo "world_size=${world_size} per_gpu_batch=2 grad_accum=20 effective_batch=80"
  echo "frame_tail_rank: DISABLED (weight=0.0)"
} | tee -a "${output_dir}/launch_console.log"

"${torchrun_bin}" --standalone --nproc_per_node="${world_size}" \
  train_football_events.py --config "${config}" \
  2>&1 | tee -a "${output_dir}/launch_console.log"
