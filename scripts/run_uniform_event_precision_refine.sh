#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-help}"
GPU_LIST="${2:-0,1,2,6,7}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_CONFIG="${BASE_CONFIG:-outputs/football_events/vitl16_weekend_uniform_event_dual_e1backbone_512x896_frozen_ref/config.yaml}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-outputs/football_events/vitl16_weekend_uniform_event_dual_e1backbone_512x896_frozen_ref/epoch_1.pt}"
HARDNEG_MANIFEST="${HARDNEG_MANIFEST:-outputs/football_hard_negatives/precision60_global_shot_save_cap6.json}"
EPOCHS="${EPOCHS:-2}"
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-4}"
TARGET_EFFECTIVE_BATCH_SIZE="${TARGET_EFFECTIVE_BATCH_SIZE:-80}"

gpu_count="$("${PYTHON_BIN}" -c 'import sys; print(len(sys.argv[1].split(",")))' "${GPU_LIST}")"
local_gpu_ids="$("${PYTHON_BIN}" -c 'import sys; n=int(sys.argv[1]); print("[" + ",".join(map(str, range(n))) + "]")' "${gpu_count}")"
grad_accum_steps="$("${PYTHON_BIN}" -c 'import math,sys; print(max(math.ceil(int(sys.argv[1])/(int(sys.argv[2])*int(sys.argv[3]))),1))' "${TARGET_EFFECTIVE_BATCH_SIZE}" "${PER_GPU_BATCH_SIZE}" "${gpu_count}")"

common_args=(
  --config "${BASE_CONFIG}"
  "gpu_ids=${local_gpu_ids}"
  "model.init_checkpoint=${INIT_CHECKPOINT}"
  "model.uniform_init_checkpoint="
  "model.event_init_checkpoint="
  "data.long_video.hard_negative.enabled=true"
  "data.long_video.hard_negative.manifest=${HARDNEG_MANIFEST}"
  "data.long_video.hard_negative.repeat_factor=1"
  "data.long_video.hard_negative.loss_weight=4.0"
  "data.long_video.hard_negative.temporal_jitter_sec=0.25"
  "data.long_video.hard_negative.safety_margin_sec=5.0"
  "data.long_video.hard_negative.max_per_video=12"
  "train.epochs=${EPOCHS}"
  "train.per_gpu_batch_size=${PER_GPU_BATCH_SIZE}"
  "train.grad_accum_steps=${grad_accum_steps}"
  "train.backbone_lr_per_gpu=0.0"
  "train.resume.enabled=false"
  "eval.per_gpu_batch_size=${PER_GPU_BATCH_SIZE}"
)

case "${MODE}" in
  adaptive_gate)
    output_dir="${OUTPUT_DIR:-outputs/football_events/vitl16_uniform_event_adaptive_gate_refine}"
    extra_args=(
      "model.uniform_event_gate_mode=adaptive"
      "model.uniform_event_gate_hidden=32"
      "model.freeze_event_branch_for_gate=true"
      "model.freeze_uniform_reference=false"
      "train.lr_per_gpu=${GATE_LR_PER_GPU:-0.00006}"
      "train.weight_decay=0.0"
      "train.event_temporal_loss_weight=0.0"
      "train.uniform_temporal_loss_weight=0.0"
      "train.hard_negative_rank_loss_weight=0.0"
      "train.positive_retention_loss_weight=0.5"
      "train.positive_retention_branch=fused"
      "train.temporal_gate_quality_loss_weight=${GATE_QUALITY_LOSS_WEIGHT:-0.5}"
      "train.temporal_gate_quality_temperature=${GATE_QUALITY_TEMPERATURE:-0.25}"
    )
    ;;
  hard_rank)
    output_dir="${OUTPUT_DIR:-outputs/football_events/vitl16_uniform_event_hardneg_rank_refine}"
    extra_args=(
      "model.uniform_event_gate_mode=static"
      "model.freeze_event_branch_for_gate=false"
      "model.freeze_uniform_reference=true"
      "train.lr_per_gpu=${HEAD_LR_PER_GPU:-0.000005}"
      "train.event_temporal_loss_weight=0.5"
      "train.hard_negative_rank_loss_weight=${RANK_LOSS_WEIGHT:-0.25}"
      "train.hard_negative_rank_branch=event"
      "train.hard_negative_rank_margin=${RANK_MARGIN:-1.0}"
      "train.online_hard_negative_loss_weight=0.0"
      "train.positive_retention_loss_weight=0.5"
      "train.positive_retention_branch=event"
      "train.temporal_gate_quality_loss_weight=0.0"
    )
    ;;
  online_ohem)
    output_dir="${OUTPUT_DIR:-outputs/football_events/vitl16_uniform_event_online_ohem_refine}"
    extra_args=(
      "model.uniform_event_gate_mode=static"
      "model.freeze_event_branch_for_gate=false"
      "model.freeze_uniform_reference=true"
      "data.long_video.hard_negative.enabled=false"
      "data.long_video.hard_negative.loss_weight=1.0"
      "train.lr_per_gpu=${HEAD_LR_PER_GPU:-0.000005}"
      "train.event_temporal_loss_weight=0.5"
      "train.hard_negative_rank_loss_weight=0.0"
      "train.online_hard_negative_loss_weight=${ONLINE_HARD_WEIGHT:-0.20}"
      "train.online_hard_negative_rank_loss_weight=0.0"
      "train.online_hard_negative_branch=event"
      "train.online_hard_negative_fraction=${ONLINE_HARD_FRACTION:-0.25}"
      "train.online_hard_negative_min_per_class=${ONLINE_HARD_MIN_PER_CLASS:-2}"
      "train.online_hard_negative_safety_margin_sec=${ONLINE_HARD_SAFETY_MARGIN_SEC:-5.0}"
      "train.online_hard_negative_labels=[shot,save]"
      "train.positive_retention_loss_weight=0.5"
      "train.positive_retention_branch=event"
      "train.temporal_gate_quality_loss_weight=0.0"
    )
    ;;
  duration_ohem)
    output_dir="${OUTPUT_DIR:-outputs/football_events/vitl16_uniform_event_duration_ohem_refine}"
    extra_args=(
      "model.uniform_event_gate_mode=static"
      "model.freeze_event_branch_for_gate=false"
      "model.freeze_uniform_reference=true"
      "data.long_video.hard_negative.enabled=false"
      "data.long_video.hard_negative.loss_weight=1.0"
      "data.long_video.negative_sampling_mode_by_split.train=hybrid"
      "data.long_video.negative_hybrid_event_ratio_by_split.train=${DURATION_EVENT_NEGATIVE_RATIO:-2.5}"
      "data.long_video.negative_per_minute_by_split.train=${NEGATIVE_PER_MINUTE:-1.2}"
      "train.lr_per_gpu=${HEAD_LR_PER_GPU:-0.000005}"
      "train.event_temporal_loss_weight=0.5"
      "train.hard_negative_rank_loss_weight=0.0"
      "train.online_hard_negative_loss_weight=${ONLINE_HARD_WEIGHT:-0.20}"
      "train.online_hard_negative_rank_loss_weight=0.0"
      "train.online_hard_negative_branch=event"
      "train.online_hard_negative_fraction=${ONLINE_HARD_FRACTION:-0.25}"
      "train.online_hard_negative_min_per_class=${ONLINE_HARD_MIN_PER_CLASS:-2}"
      "train.online_hard_negative_safety_margin_sec=${ONLINE_HARD_SAFETY_MARGIN_SEC:-5.0}"
      "train.online_hard_negative_labels=[shot,save]"
      "train.positive_retention_loss_weight=0.5"
      "train.positive_retention_branch=event"
      "train.temporal_gate_quality_loss_weight=0.0"
    )
    ;;
  online_rank)
    output_dir="${OUTPUT_DIR:-outputs/football_events/vitl16_uniform_event_duration_online_rank_refine}"
    extra_args=(
      "model.uniform_event_gate_mode=static"
      "model.freeze_event_branch_for_gate=false"
      "model.freeze_uniform_reference=true"
      "data.long_video.hard_negative.enabled=false"
      "data.long_video.hard_negative.loss_weight=1.0"
      "data.long_video.negative_sampling_mode_by_split.train=hybrid"
      "data.long_video.negative_hybrid_event_ratio_by_split.train=${DURATION_EVENT_NEGATIVE_RATIO:-2.5}"
      "data.long_video.negative_per_minute_by_split.train=${NEGATIVE_PER_MINUTE:-1.2}"
      "train.lr_per_gpu=${HEAD_LR_PER_GPU:-0.000005}"
      "train.event_temporal_loss_weight=0.5"
      "train.hard_negative_rank_loss_weight=0.0"
      "train.online_hard_negative_loss_weight=${ONLINE_HARD_WEIGHT:-0.10}"
      "train.online_hard_negative_rank_loss_weight=${ONLINE_RANK_WEIGHT:-0.15}"
      "train.online_hard_negative_rank_branch=event"
      "train.online_hard_negative_rank_margin=${ONLINE_RANK_MARGIN:-1.0}"
      "train.online_hard_negative_branch=event"
      "train.online_hard_negative_fraction=${ONLINE_HARD_FRACTION:-0.25}"
      "train.online_hard_negative_min_per_class=${ONLINE_HARD_MIN_PER_CLASS:-2}"
      "train.online_hard_negative_safety_margin_sec=${ONLINE_HARD_SAFETY_MARGIN_SEC:-5.0}"
      "train.online_hard_negative_labels=[shot,save]"
      "train.positive_retention_loss_weight=0.5"
      "train.positive_retention_branch=event"
      "train.temporal_gate_quality_loss_weight=0.0"
    )
    ;;
  teacher_guard_hr)
    output_dir="${OUTPUT_DIR:-outputs/football_events/vitl16_uniform_event_teacher_guard_hr16f_refine}"
    teacher_checkpoint="${TEACHER_CHECKPOINT:-/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt}"
    extra_args=(
      "model.init_checkpoint=${teacher_checkpoint}"
      "model.uniform_event_gate_mode=static"
      "model.uniform_event_gate_init=[0.20,0.20,0.10]"
      "model.freeze_event_branch_for_gate=false"
      "model.freeze_uniform_reference=true"
      "video.image_size=[640,1120]"
      "video.num_frames=16"
      "video.candidate_num_frames=16"
      "model.uniform_frames=16"
      "model.event_topk=16"
      "model.context_frames=0"
      "model.event_topk_strategy=shared_max"
      "data.long_video.hard_negative.enabled=false"
      "data.long_video.hard_negative.loss_weight=1.0"
      "data.long_video.negative_sampling_mode_by_split.train=hybrid"
      "data.long_video.negative_hybrid_event_ratio_by_split.train=${DURATION_EVENT_NEGATIVE_RATIO:-2.5}"
      "data.long_video.negative_per_minute_by_split.train=${NEGATIVE_PER_MINUTE:-1.2}"
      "train.epochs=${TEACHER_GUARD_EPOCHS:-1}"
      "train.lr_per_gpu=${HEAD_LR_PER_GPU:-0.000005}"
      "train.event_temporal_loss_weight=0.5"
      "train.hard_negative_rank_loss_weight=0.0"
      "train.online_hard_negative_loss_weight=${ONLINE_HARD_WEIGHT:-0.15}"
      "train.online_hard_negative_rank_loss_weight=0.0"
      "train.online_hard_negative_branch=fused"
      "train.online_hard_negative_fraction=${ONLINE_HARD_FRACTION:-0.25}"
      "train.online_hard_negative_min_per_class=${ONLINE_HARD_MIN_PER_CLASS:-2}"
      "train.online_hard_negative_safety_margin_sec=${ONLINE_HARD_SAFETY_MARGIN_SEC:-5.0}"
      "train.online_hard_negative_labels=[shot,save]"
      "train.positive_retention_loss_weight=${POSITIVE_RETENTION_WEIGHT:-0.75}"
      "train.positive_retention_branch=fused"
      "train.negative_teacher_guard_loss_weight=${TEACHER_GUARD_WEIGHT:-0.50}"
      "train.negative_teacher_guard_margin=${TEACHER_GUARD_MARGIN:-0.0}"
      "train.negative_teacher_guard_branch=fused"
      "train.negative_teacher_guard_labels=[shot,save,set_piece]"
      "train.temporal_gate_quality_loss_weight=0.0"
    )
    ;;
  *)
    echo "Usage: $0 {adaptive_gate|hard_rank|online_ohem|duration_ohem|online_rank|teacher_guard_hr} GPU_LIST" >&2
    exit 2
    ;;
esac

mkdir -p "${output_dir}"
{
  echo "started_at=$(date --iso-8601=seconds)"
  echo "mode=${MODE}"
  echo "physical_gpus=${GPU_LIST}"
  echo "init_checkpoint=${INIT_CHECKPOINT}"
  echo "effective_batch_size=$(( PER_GPU_BATCH_SIZE * gpu_count * grad_accum_steps ))"
} | tee -a "${output_dir}/train_console.log"

CUDA_VISIBLE_DEVICES="${GPU_LIST}" PYTHONUNBUFFERED=1 "${PYTHON_BIN}" train_football_events.py "${common_args[@]}" "output_dir=${output_dir}" "${extra_args[@]}" 2>&1 | tee -a "${output_dir}/train_console.log"
