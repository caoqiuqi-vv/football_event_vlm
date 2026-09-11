#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/new_users/qiuqi/code/dinov3-main"
source_run="${repo_dir}/outputs/football_events/vitl16_d7c_object_teacher_online_allclips_no_bad_media_from_fromlast_e8_20260901"
checkpoint="${source_run}/epoch_2.pt"
python_bin="/home/new_users/qiuqi/miniconda3/envs/qiuqi_sam3/bin/python"
output="${SMOKE_OUTPUT:-${repo_dir}/outputs/football_events/vitl16_d7c_object_motion_v4_causal33_720p_aux025_4gpu_from_object_e2_20260903/smoke_pass.json}"
batch_size="${SMOKE_BATCH_SIZE:-1}"

cd "${repo_dir}"
export CUDA_VISIBLE_DEVICES="${SMOKE_GPU:-1}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
contract_hash="$("${python_bin}" -m football_object_motion.smoke_contract contract \
  --repo "${repo_dir}" \
  --config "${source_run}/config.yaml" \
  --checkpoint "${checkpoint}" \
  | "${python_bin}" -c 'import json,sys; print(json.load(sys.stdin)["contract_hash"])')"
"${python_bin}" -m football_object_motion.smoke \
  --config "${source_run}/config.yaml" \
  --checkpoint "${checkpoint}" \
  --contract-hash "${contract_hash}" \
  --output "${output}" \
  --batch-size "${batch_size}"
