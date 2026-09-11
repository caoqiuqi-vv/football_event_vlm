#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/new_users/qiuqi/code/dinov3-main"
base_config="${repo_dir}/configs/football/dinov3_vitl16_online_simulation_e14_signed_causal_continue4.yaml"
controlled_root="${repo_dir}/outputs/football_events/vitl16_controlled_online_adaptation_from_last6r8"
original_init="${repo_dir}/outputs/football_events/vitl16_independent_preprojection_evidence_ddp_lora_v1_accelerated_resume_e1/best.pt"
stage1_init="${controlled_root}/stage1_frozen_lora_temporal_heads/best.pt"
init_checkpoint="${HIGHRES_INIT_CHECKPOINT:-}"
output_dir="${HIGHRES_OUTPUT_DIR:-${repo_dir}/outputs/football_events/vitl16_controlled_online_highres720_last6r8}"
gpu_list="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
torchrun_bin="/home/new_users/qiuqi/miniconda3/envs/qiuqi_sam3/bin/torchrun"
frame_chunk="${HIGHRES_BACKBONE_FRAME_CHUNK_SIZE:-12}"
epochs="${HIGHRES_EPOCHS:-4}"

if [[ -z "${init_checkpoint}" ]]; then
  if [[ -f "${stage1_init}" ]]; then
    init_checkpoint="${stage1_init}"
  else
    init_checkpoint="${original_init}"
  fi
fi
if [[ ! -f "${init_checkpoint}" ]]; then
  echo "Missing high-resolution init checkpoint: ${init_checkpoint}" >&2
  exit 2
fi

mkdir -p "${output_dir}"
cd "${repo_dir}"
export CUDA_VISIBLE_DEVICES="${gpu_list}"
export PYTHONPATH="${repo_dir}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

{
  echo "highres_protocol=controlled_single_variable"
  echo "init_checkpoint=${init_checkpoint}"
  echo "image_size=720x1280 patches_per_frame=3600"
  echo "lora=last6/r8/alpha16/qkv+proj"
  echo "logical_batch=4_per_gpu_x4_ddp_x4_accum=64"
  echo "backbone_frame_chunk_size=${frame_chunk}"
  echo "loss_protocol=E16_clean_online_no_signed_causal_no_cross_window_evidence_no_subtype_no_context_span"
} | tee -a "${output_dir}/pipeline.log"

baseline_json="${output_dir}/baseline_init_online_val15_720p.json"
if [[ ! -f "${baseline_json}" ]]; then
  ONLINE_EVAL_WORLD_SIZE=4 \
  ONLINE_EVAL_PER_GPU_BATCH=1 \
  ONLINE_EVAL_IMAGE_SIZE='[720,1280]' \
  ONLINE_EVAL_BACKBONE_FRAME_CHUNK_SIZE="${frame_chunk}" \
    bash scripts/evaluate_online_val15_checkpoint.sh \
      "${base_config}" "${init_checkpoint}" "${baseline_json}"
fi

"${torchrun_bin}" --standalone --nproc-per-node=4 \
  train_football_events_online_simulation_e13.py \
  --config "${base_config}" \
  "output_dir=${output_dir}" \
  "model.init_checkpoint=${init_checkpoint}" \
  "model.init_checkpoint_strict=false" \
  "model.freeze_loaded_backbone=false" \
  "model.controlled_online_train_scope=lora_temporal_heads" \
  "model.backbone_frame_chunk_size=${frame_chunk}" \
  "model.gradient_checkpointing=true" \
  "model.gradient_checkpointing_mode=trainable_blocks" \
  "model.lora.enabled=true" \
  "model.lora.rank=8" \
  "model.lora.alpha=16" \
  "model.lora.target_last_blocks=6" \
  "video.image_size=[720,1280]" \
  "train.resume.enabled=false" \
  "train.epochs=${epochs}" \
  "train.scheduler_epochs=${epochs}" \
  "train.save_epoch_checkpoints=false" \
  "train.per_gpu_batch_size=4" \
  "train.grad_accum_steps=4" \
  "train.backbone_lr_per_gpu=1.25e-7" \
  "train.global_backbone_lr_per_gpu=1.25e-7" \
  "train.local_backbone_lr_per_gpu=1.25e-7" \
  "train.class_evidence_cross_window_weight=0.0" \
  "train.class_evidence_signed_causal_loss_weight=0.0" \
  "train.set_piece_subtype_loss_weight=0.0" \
  "train.raw_context_span_loss_weight=0.0" \
  "train.early_stopping.enabled=true" \
  "train.early_stopping.monitor=selection_score" \
  "train.early_stopping.min_epoch=2" \
  "train.early_stopping.patience=2" \
  "eval.per_gpu_batch_size=1" \
  "eval.external_audit.enabled=false" \
  "eval.online_validation.enabled=true" \
  "eval.online_validation.window_stride_sec=5.0" \
  "eval.online_validation.nms_radius_sec=5.0" \
  "eval.online_validation.tolerance_sec=5.0" \
  "eval.online_validation.use_for_checkpoint_selection=true" \
  "eval.online_validation.tuned_objective_by_class.shot=precision_at_recall_floor" \
  "eval.online_validation.tuned_objective_by_class.save=f1" \
  "eval.online_validation.tuned_objective_by_class.set_piece=f1" \
  "eval.online_validation.tuned_min_recall_by_class.shot=0.85" \
  "eval.online_validation.tuned_min_recall_by_class.save=null" \
  "eval.online_validation.tuned_min_recall_by_class.set_piece=null" \
  "data.num_workers_per_gpu=3" \
  2>&1 | tee -a "${output_dir}/train_console.log"
