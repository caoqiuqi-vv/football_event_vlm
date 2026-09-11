#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-rank}"
GPU_LIST="${2:-0,1,2,3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_CONFIG="${BASE_CONFIG:-configs/football/dinov3_vitl16_robust_dual_16f_e1_dynamic_roi_decoupled_lora_d4.yaml}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt}"
HARDNEG_MANIFEST="${HARDNEG_MANIFEST:-outputs/football_hard_negatives/reviewed_mid_score_v2_shot_save_setpiece.json}"
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-4}"
TARGET_EFFECTIVE_BATCH_SIZE="${TARGET_EFFECTIVE_BATCH_SIZE:-80}"
TARGET_HEAD_LR="${TARGET_HEAD_LR:-0.0001}"
TARGET_BACKBONE_LR="${TARGET_BACKBONE_LR:-0.00001}"
EPOCHS="${EPOCHS:-3}"

gpu_count="$("${PYTHON_BIN}" -c 'import sys; print(len([x for x in sys.argv[1].split(",") if x.strip()]))' "${GPU_LIST}")"
local_gpu_ids="$("${PYTHON_BIN}" -c 'import sys; n=int(sys.argv[1]); print("[" + ",".join(map(str, range(n))) + "]")' "${gpu_count}")"
grad_accum_steps="$("${PYTHON_BIN}" -c 'import math,sys; print(max(math.ceil(int(sys.argv[1])/(int(sys.argv[2])*int(sys.argv[3]))),1))' "${TARGET_EFFECTIVE_BATCH_SIZE}" "${PER_GPU_BATCH_SIZE}" "${gpu_count}")"
head_lr_per_gpu="$("${PYTHON_BIN}" -c 'import sys; print(float(sys.argv[1])/int(sys.argv[2]))' "${TARGET_HEAD_LR}" "${gpu_count}")"
backbone_lr_per_gpu="$("${PYTHON_BIN}" -c 'import sys; print(float(sys.argv[1])/int(sys.argv[2]))' "${TARGET_BACKBONE_LR}" "${gpu_count}")"

if [[ ! -f "${INIT_CHECKPOINT}" ]]; then
  echo "Missing E1 init checkpoint: ${INIT_CHECKPOINT}" >&2
  exit 2
fi

case "${MODE}" in
  control)
    output_dir="${OUTPUT_DIR:-outputs/football_events/vitl16_e1_dynamic_roi_veto_d5_control}"
    hard_negative_args=(
      "data.long_video.hard_negative.enabled=false"
      "train.hard_negative_rank_loss_weight=0.0"
    )
    ;;
  rank)
    if [[ ! -f "${HARDNEG_MANIFEST}" ]]; then
      echo "Missing audited hard-negative manifest: ${HARDNEG_MANIFEST}" >&2
      exit 2
    fi
    output_dir="${OUTPUT_DIR:-outputs/football_events/vitl16_e1_dynamic_roi_veto_rank_d5}"
    hard_negative_args=(
      "data.long_video.hard_negative.enabled=true"
      "data.long_video.hard_negative.manifest=${HARDNEG_MANIFEST}"
      "data.long_video.hard_negative.min_score=0.35"
      "data.long_video.hard_negative.max_per_video=30"
      "data.long_video.hard_negative.repeat_factor=3"
      "data.long_video.hard_negative.loss_weight=2.0"
      "data.long_video.hard_negative.temporal_jitter_sec=0.25"
      "data.long_video.hard_negative.safety_margin_sec=5.0"
      "train.hard_negative_rank_loss_weight=0.25"
      "train.hard_negative_rank_branch=fused"
      "train.hard_negative_rank_labels=[shot,save]"
      "train.hard_negative_rank_margin=1.0"
    )
    ;;
  *)
    echo "Usage: bash $0 {control|rank} GPU_LIST" >&2
    exit 2
    ;;
esac

mkdir -p "${output_dir}"
log_file="${output_dir}/train_console.log"
effective_batch_size=$((PER_GPU_BATCH_SIZE * gpu_count * grad_accum_steps))

{
  echo
  echo "===== E1 ROI residual-veto experiment ====="
  echo "started_at=$(date --iso-8601=seconds)"
  echo "mode=${MODE}"
  echo "physical_gpus=${GPU_LIST}"
  echo "init_checkpoint=${INIT_CHECKPOINT}"
  echo "hard_negative_manifest=${HARDNEG_MANIFEST}"
  echo "per_gpu_batch_size=${PER_GPU_BATCH_SIZE}"
  echo "effective_batch_size=${effective_batch_size}"
  echo "global_head_lr=${TARGET_HEAD_LR}"
  echo "global_backbone_lr=${TARGET_BACKBONE_LR}"
} | tee -a "${log_file}"

common_args=(
  --config "${BASE_CONFIG}"
  "output_dir=${output_dir}"
  "gpu_ids=${local_gpu_ids}"
  "model.init_checkpoint=${INIT_CHECKPOINT}"
  "model.view_fusion=dual_verifier"
  "model.roi_verifier_min_confidence=0.60"
  "model.roi_verifier_class_indices=[0,1]"
  "model.separate_local_backbone=true"
  "model.freeze_global_branch=true"
  "model.freeze_loaded_backbone=false"
  "model.gradient_checkpointing=true"
  "spatial_crop.global_image_size=[640,1120]"
  "train.epochs=${EPOCHS}"
  "train.per_gpu_batch_size=${PER_GPU_BATCH_SIZE}"
  "train.grad_accum_steps=${grad_accum_steps}"
  "train.lr_per_gpu=${head_lr_per_gpu}"
  "train.backbone_lr_per_gpu=${backbone_lr_per_gpu}"
  "train.local_loss_weight=0.0"
  "train.frame_det_loss_weight=0.0"
  "train.positive_retention_loss_weight=2.0"
  "train.positive_retention_margin=0.0"
  "train.positive_retention_branch=fused"
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
)

set +e
CUDA_VISIBLE_DEVICES="${GPU_LIST}" PYTHONUNBUFFERED=1 "${PYTHON_BIN}" \
  train_football_events.py "${common_args[@]}" "${hard_negative_args[@]}" \
  2>&1 | tee -a "${log_file}"
status="${PIPESTATUS[0]}"
set -e
echo "finished_at=$(date --iso-8601=seconds) exit_status=${status}" | tee -a "${log_file}"
exit "${status}"
