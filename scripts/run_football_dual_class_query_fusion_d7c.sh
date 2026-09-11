#!/usr/bin/env bash
set -euo pipefail

GPU_LIST="${1:-2,3,4,6}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_CONFIG="${BASE_CONFIG:-configs/football/dinov3_vitl16_robust_dual_16f_e1_dynamic_roi_decoupled_lora_d4.yaml}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/football_events/vitl16_e1_dual_class_query_fusion_d7c}"
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-4}"
TARGET_EFFECTIVE_BATCH_SIZE="${TARGET_EFFECTIVE_BATCH_SIZE:-80}"
TARGET_HEAD_LR="${TARGET_HEAD_LR:-0.0001}"
TARGET_GLOBAL_BACKBONE_LR="${TARGET_GLOBAL_BACKBONE_LR:-0.0}"
TARGET_LOCAL_BACKBONE_LR="${TARGET_LOCAL_BACKBONE_LR:-0.00002}"
EPOCHS="${EPOCHS:-5}"
NUM_WORKERS_PER_GPU="${NUM_WORKERS_PER_GPU:-2}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
NEGATIVE_RATIO_TRAIN="${NEGATIVE_RATIO_TRAIN:-5.0}"
NEGATIVE_RATIO_VAL="${NEGATIVE_RATIO_VAL:-2.0}"
CLIP_LOSS_TYPE="${CLIP_LOSS_TYPE:-bce}"
CLIP_FOCAL_GAMMA="${CLIP_FOCAL_GAMMA:-2.0}"
CLIP_FOCAL_ALPHA="${CLIP_FOCAL_ALPHA:-shot=0.35,save=0.40,set_piece=0.45}"
VIEW_FUSION_POSITIVE_DELTA_PER_CLASS="${VIEW_FUSION_POSITIVE_DELTA_PER_CLASS:-}"
VIEW_FUSION_NEGATIVE_DELTA_PER_CLASS="${VIEW_FUSION_NEGATIVE_DELTA_PER_CLASS:-}"

gpu_count="$("${PYTHON_BIN}" -c 'import sys; print(len([x for x in sys.argv[1].split(",") if x.strip()]))' "${GPU_LIST}")"
local_gpu_ids="$("${PYTHON_BIN}" -c 'import sys; n=int(sys.argv[1]); print("[" + ",".join(map(str, range(n))) + "]")' "${gpu_count}")"
grad_accum_steps="$("${PYTHON_BIN}" -c 'import math,sys; print(max(math.ceil(int(sys.argv[1])/(int(sys.argv[2])*int(sys.argv[3]))),1))' "${TARGET_EFFECTIVE_BATCH_SIZE}" "${PER_GPU_BATCH_SIZE}" "${gpu_count}")"
head_lr_per_gpu="$("${PYTHON_BIN}" -c 'import sys; print(float(sys.argv[1])/int(sys.argv[2]))' "${TARGET_HEAD_LR}" "${gpu_count}")"
global_backbone_lr_per_gpu="$("${PYTHON_BIN}" -c 'import sys; print(float(sys.argv[1])/int(sys.argv[2]))' "${TARGET_GLOBAL_BACKBONE_LR}" "${gpu_count}")"
local_backbone_lr_per_gpu="$("${PYTHON_BIN}" -c 'import sys; print(float(sys.argv[1])/int(sys.argv[2]))' "${TARGET_LOCAL_BACKBONE_LR}" "${gpu_count}")"

if [[ ! -f "${INIT_CHECKPOINT}" ]]; then
  echo "Missing E1 init checkpoint: ${INIT_CHECKPOINT}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
log_file="${OUTPUT_DIR}/train_console.log"
effective_batch_size=$((PER_GPU_BATCH_SIZE * gpu_count * grad_accum_steps))

{
  echo
  echo "===== D7-C aligned class-query early fusion ====="
  echo "started_at=$(date --iso-8601=seconds)"
  echo "physical_gpus=${GPU_LIST}"
  echo "init_checkpoint=${INIT_CHECKPOINT}"
  echo "base_config=${BASE_CONFIG}"
  echo "global_frames=16 roi_frames=16 sampling=aligned"
  echo "per_gpu_batch_size=${PER_GPU_BATCH_SIZE}"
  echo "effective_batch_size=${effective_batch_size}"
  echo "head_lr_global=${TARGET_HEAD_LR}"
  echo "global_backbone_lr_global=${TARGET_GLOBAL_BACKBONE_LR}"
  echo "num_workers_per_gpu=${NUM_WORKERS_PER_GPU}"
  echo "prefetch_factor=${PREFETCH_FACTOR}"
  echo "local_backbone_lr_global=${TARGET_LOCAL_BACKBONE_LR}"
  echo "local_loss_weight=${LOCAL_LOSS_WEIGHT:-0.0}"
  echo "frame_det_loss_weight=${FRAME_DET_LOSS_WEIGHT:-0.3}"
  echo "roi_quality_loss_weight=${ROI_QUALITY_LOSS_WEIGHT:-0.0}"
  echo "negative_ratio_train=${NEGATIVE_RATIO_TRAIN}"
  echo "negative_ratio_val=${NEGATIVE_RATIO_VAL}"
  echo "clip_loss_type=${CLIP_LOSS_TYPE}"
  echo "clip_focal_gamma=${CLIP_FOCAL_GAMMA}"
  echo "clip_focal_alpha=${CLIP_FOCAL_ALPHA}"
  echo "view_fusion_positive_delta_per_class=${VIEW_FUSION_POSITIVE_DELTA_PER_CLASS}"
  echo "view_fusion_negative_delta_per_class=${VIEW_FUSION_NEGATIVE_DELTA_PER_CLASS}"
} | tee -a "${log_file}"

common_args=(
  --config "${BASE_CONFIG}"
  "output_dir=${OUTPUT_DIR}"
  "gpu_ids=${local_gpu_ids}"
  "model.init_checkpoint=${INIT_CHECKPOINT}"
  "model.view_fusion=dual_class_query_fusion"
  "model.view_fusion_layers=2"
  "model.view_fusion_heads=8"
  "model.view_fusion_positive_delta=0.5"
  "model.view_fusion_negative_delta=2.0"
  "model.separate_local_backbone=true"
  "model.freeze_global_branch=true"
  "model.freeze_loaded_backbone=false"
  "model.gradient_checkpointing=true"
  "video.dual_sampling=aligned"
  "spatial_crop.global_image_size=[640,1120]"
  "train.epochs=${EPOCHS}"
  "train.per_gpu_batch_size=${PER_GPU_BATCH_SIZE}"
  "train.grad_accum_steps=${grad_accum_steps}"
  "train.lr_per_gpu=${head_lr_per_gpu}"
  "train.backbone_lr_per_gpu=${local_backbone_lr_per_gpu}"
  "train.global_backbone_lr_per_gpu=${global_backbone_lr_per_gpu}"
  "train.local_backbone_lr_per_gpu=${local_backbone_lr_per_gpu}"
  "train.local_loss_weight=${LOCAL_LOSS_WEIGHT:-0.0}"
  "data.long_video.negative_ratio=${NEGATIVE_RATIO_TRAIN}"
  "data.long_video.negative_ratio_by_split.train=${NEGATIVE_RATIO_TRAIN}"
  "data.long_video.negative_ratio_by_split.val=${NEGATIVE_RATIO_VAL}"
  "train.clip_loss_type=${CLIP_LOSS_TYPE}"
  "train.clip_focal_gamma=${CLIP_FOCAL_GAMMA}"
  "train.clip_focal_alpha=${CLIP_FOCAL_ALPHA}"
  "train.frame_det_loss_weight=${FRAME_DET_LOSS_WEIGHT:-0.3}"
  "train.roi_quality_loss_weight=${ROI_QUALITY_LOSS_WEIGHT:-0.0}"
  "train.roi_quality_temperature=0.25"
  "train.positive_retention_loss_weight=1.0"
  "train.positive_retention_margin=0.0"
  "train.positive_retention_branch=fused"
  "data.long_video.hard_negative.enabled=false"
  "train.hard_negative_rank_loss_weight=0.0"
  "train.online_hard_negative_loss_weight=0.0"
  "train.online_hard_negative_rank_loss_weight=0.0"
  "train.checkpoint_selection=tuned_macro_precision_at_recall_floor"
  "train.checkpoint_min_recall=0.80"
  "train.resume.enabled=false"
  "train.resume.load_optimizer=false"
  "train.resume.load_scheduler=false"
  "train.resume.load_scaler=false"
  "eval.tuned_min_recall.shot=0.84"
  "eval.tuned_min_recall.save=0.80"
  "eval.tuned_min_recall.set_piece=0.72"
  "eval.per_gpu_batch_size=${PER_GPU_BATCH_SIZE}"
  "data.num_workers_per_gpu=${NUM_WORKERS_PER_GPU}"
  "data.prefetch_factor=${PREFETCH_FACTOR}"
)
if [[ -n "${VIEW_FUSION_POSITIVE_DELTA_PER_CLASS}" ]]; then
  common_args+=("model.view_fusion_positive_delta_per_class=${VIEW_FUSION_POSITIVE_DELTA_PER_CLASS}")
fi
if [[ -n "${VIEW_FUSION_NEGATIVE_DELTA_PER_CLASS}" ]]; then
  common_args+=("model.view_fusion_negative_delta_per_class=${VIEW_FUSION_NEGATIVE_DELTA_PER_CLASS}")
fi

set +e
CUDA_VISIBLE_DEVICES="${GPU_LIST}" PYTHONUNBUFFERED=1 "${PYTHON_BIN}" \
  train_football_events.py "${common_args[@]}" \
  2>&1 | tee -a "${log_file}"
status=${PIPESTATUS[0]}
set -e

echo "finished_at=$(date --iso-8601=seconds) status=${status}" | tee -a "${log_file}"
exit "${status}"
