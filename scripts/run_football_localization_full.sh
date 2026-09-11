#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"
export CUDA_VISIBLE_DEVICES="${STAGE1_FULL_GPUS:-1,3,4,6}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NCCL_ALGO=Ring
python_bin=/home/new_users/qiuqi/miniconda3/envs/qiuqi_sam3/bin/python
IFS=',' read -ra selected_gpus <<< "${CUDA_VISIBLE_DEVICES}"
"${python_bin}" -m torch.distributed.run --standalone --nproc-per-node="${#selected_gpus[@]}" \
 scripts/train_football_localization_full.py \
 --config configs/football/localization_stage1_720p_full_native_20260907.json "$@"
