#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_dir}"

python_bin="${PYTHON_BIN:-/home/new_users/qiuqi/miniconda3/envs/qiuqi_sam3/bin/python}"
checkpoint="${PROBE_CHECKPOINT:-${repo_dir}/outputs/football_events/vitl16_d7c_object_motion_v4_causal33_720p_aux025_4gpu_from_object_e2_20260903/epoch_4.pt}"
index_root="${PROBE_INDEX_ROOT:-/mnt/data_7t/qiuqi/football_ball_pseudolabels/yolo_fulltrack_v2_test18heldout/offline_index_v1}"
video_root="${PROBE_VIDEO_ROOT:-/mnt/data_16t/football/raw_video_720P}"
video_ids="${PROBE_VIDEO_IDS:-${repo_dir}/configs/football/splits/object_teacher_online_allclips_no_bad_media/train_media_ids.txt}"
output_dir="${PROBE_OUTPUT_DIR:-${repo_dir}/outputs/football_ball_feature_probe/vitl16_v4e4_observed_conf050_content_holdout_20260907_r2}"
gpus="${PROBE_GPUS:-0,1,4,6}"

mkdir -p "${output_dir}"
export CUDA_VISIBLE_DEVICES="${gpus}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

extra_args=()
if [[ "${PROBE_SMOKE:-0}" == "1" ]]; then
  extra_args+=(--smoke --num-workers 0)
fi

"${python_bin}" -m torch.distributed.run \
  --standalone \
  --nproc-per-node=4 \
  scripts/probe_dino_ball_localization.py \
  --checkpoint "${checkpoint}" \
  --index-root "${index_root}" \
  --video-root "${video_root}" \
  --video-ids "${video_ids}" \
  --output-dir "${output_dir}" \
  --layers 5 11 17 23 \
  --confidence 0.5 \
  --val-video-fraction 0.2 \
  --max-train-frames-per-video 80 \
  --max-val-frames-per-video 60 \
  --epochs 5 \
  --batch-size 2 \
  --num-workers 1 \
  --lr 1e-3 \
  --hidden-dim 256 \
  "${extra_args[@]}" \
  2>&1 | tee -a "${output_dir}/console.log"
