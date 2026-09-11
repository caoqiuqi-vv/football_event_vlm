#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"
export CUDA_VISIBLE_DEVICES="${STAGE1_GPUS:-0,1,3,4}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
python_bin=/home/new_users/qiuqi/miniconda3/envs/qiuqi_sam3/bin/python
IFS=',' read -ra selected_gpus <<< "${CUDA_VISIBLE_DEVICES}"
"${python_bin}" -m torch.distributed.run --standalone --nproc-per-node="${#selected_gpus[@]}" \
  scripts/train_football_localization_stage1.py \
  --config configs/football/localization_stage1_720p_kl_from720best_20260907.json "$@"
