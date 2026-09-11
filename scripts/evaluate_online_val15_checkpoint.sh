#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 CONFIG CHECKPOINT OUTPUT_JSON" >&2
  exit 2
fi

repo_dir="/home/new_users/qiuqi/code/dinov3-main"
config="$1"
checkpoint="$2"
output_json="$3"
gpu_list="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
world_size="${ONLINE_EVAL_WORLD_SIZE:-4}"
eval_batch="${ONLINE_EVAL_PER_GPU_BATCH:-4}"
torchrun_bin="/home/new_users/qiuqi/miniconda3/envs/qiuqi_sam3/bin/torchrun"
eval_dir="$(dirname "${output_json}")"

if [[ ! -f "${config}" ]]; then
  echo "Missing config: ${config}" >&2
  exit 2
fi
if [[ ! -f "${checkpoint}" ]]; then
  echo "Missing checkpoint: ${checkpoint}" >&2
  exit 2
fi

mkdir -p "${eval_dir}"
cd "${repo_dir}"
export CUDA_VISIBLE_DEVICES="${gpu_list}"
export PYTHONPATH="${repo_dir}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

extra_overrides=()
if [[ -n "${ONLINE_EVAL_IMAGE_SIZE:-}" ]]; then
  extra_overrides+=("video.image_size=${ONLINE_EVAL_IMAGE_SIZE}")
fi
if [[ -n "${ONLINE_EVAL_BACKBONE_FRAME_CHUNK_SIZE:-}" ]]; then
  extra_overrides+=("model.backbone_frame_chunk_size=${ONLINE_EVAL_BACKBONE_FRAME_CHUNK_SIZE}")
fi

"${torchrun_bin}" --standalone --nproc-per-node="${world_size}" \
  train_football_events_online_simulation_e13.py \
  --config "${config}" \
  --eval-only \
  --eval-output "${output_json}" \
  "output_dir=${eval_dir}" \
  "model.init_checkpoint=${checkpoint}" \
  "model.init_checkpoint_strict=false" \
  "train.resume.enabled=false" \
  "eval.per_gpu_batch_size=${eval_batch}" \
  "eval.online_validation.enabled=true" \
  "eval.online_validation.window_stride_sec=5.0" \
  "eval.online_validation.nms_radius_sec=5.0" \
  "eval.online_validation.tolerance_sec=5.0" \
  "eval.online_validation.capped_clip_sec=10.0" \
  "eval.online_validation.tune_event_thresholds=true" \
  "eval.online_validation.max_threshold_candidates=401" \
  "eval.online_validation.use_for_checkpoint_selection=true" \
  "eval.online_validation.tuned_objective_by_class.shot=precision_at_recall_floor" \
  "eval.online_validation.tuned_objective_by_class.save=f1" \
  "eval.online_validation.tuned_objective_by_class.set_piece=f1" \
  "eval.online_validation.tuned_min_recall_by_class.shot=0.85" \
  "eval.online_validation.tuned_min_recall_by_class.save=null" \
  "eval.online_validation.tuned_min_recall_by_class.set_piece=null" \
  "eval.external_audit.enabled=false" \
  "${extra_overrides[@]}" \
  2>&1 | tee -a "${eval_dir}/online_val15_console.log"
