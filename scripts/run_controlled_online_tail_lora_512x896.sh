#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/new_users/qiuqi/code/dinov3-main"
base_config="${repo_dir}/configs/football/dinov3_vitl16_online_simulation_e14_signed_causal_continue4.yaml"
init_checkpoint="${INIT_CHECKPOINT:-${repo_dir}/outputs/football_events/vitl16_controlled_online_highres720_headonly_anneal/epoch1_full_consistency/best.pt}"
output_dir="${OUTPUT_DIR:-${repo_dir}/outputs/football_events/vitl16_controlled_online_tail_lora_512x896_e1}"
gpu_list="${CUDA_VISIBLE_DEVICES:-2,3,4,5}"
epochs="${EPOCHS:-2}"
per_gpu_batch_size="${PER_GPU_BATCH_SIZE:-6}"
grad_accum_steps="${GRAD_ACCUM_STEPS:-2}"
num_workers_per_gpu="${NUM_WORKERS_PER_GPU:-2}"
torchrun_bin="${TORCHRUN_BIN:-/home/new_users/qiuqi/miniconda3/envs/qiuqi_sam3/bin/torchrun}"

[[ -f "${init_checkpoint}" ]] || { echo "Missing init checkpoint: ${init_checkpoint}" >&2; exit 2; }
IFS=',' read -ra gpu_ids <<< "${gpu_list}"
world_size="${#gpu_ids[@]}"
mkdir -p "${output_dir}"
cd "${repo_dir}"
export CUDA_VISIBLE_DEVICES="${gpu_list}" PYTHONPATH="${repo_dir}" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

{
  echo "experiment=online_tail_separation_fixed_teacher_last6_lora"
  echo "init_checkpoint=${init_checkpoint}"
  echo "image_size=512x896 frames=24"
  echo "world_size=${world_size} per_gpu_batch=${per_gpu_batch_size} grad_accum=${grad_accum_steps}"
  echo "selection_floors=shot:0.85,save:0.70,set_piece:0.70"
} | tee -a "${output_dir}/pipeline.log"

"${torchrun_bin}" --standalone --nproc-per-node="${world_size}"   train_football_events_online_simulation_e13.py --config "${base_config}"   "output_dir=${output_dir}"   "model.init_checkpoint=${init_checkpoint}" "model.init_checkpoint_strict=false"   "model.freeze_loaded_backbone=false" "model.controlled_online_train_scope=lora_temporal_heads"   "model.gradient_checkpointing=true" "model.gradient_checkpointing_mode=trainable_blocks"   "model.lora.enabled=true" "model.lora.rank=8" "model.lora.alpha=16" "model.lora.target_last_blocks=6"   "video.image_size=[512,896]"   "train.resume.enabled=false" "train.epochs=${epochs}" "train.scheduler_epochs=${epochs}"   "train.warmup.ratio=0.0" "train.per_gpu_batch_size=${per_gpu_batch_size}"   "train.grad_accum_steps=${grad_accum_steps}"   "train.lr_per_gpu=4.6875e-6" "train.temporal_lr_per_gpu=6.25e-6" "train.head_lr_per_gpu=9.375e-6"   "train.class_evidence_temporal_lr_per_gpu=3.125e-6" "train.class_evidence_head_lr_per_gpu=4.6875e-6"   "train.backbone_lr_per_gpu=1.25e-7" "train.global_backbone_lr_per_gpu=1.25e-7"   "train.local_backbone_lr_per_gpu=1.25e-7"   "train.clean_negative_rank_loss_weight=0.0" "train.hard_negative_rank_loss_weight=0.0"   "train.clean_positive_logit_margin_loss_weight=0.0"   "train.tail_separation_loss_weight=0.15" "train.tail_separation_margin=0.15"   "train.tail_separation_positive_quantile=0.25" "train.tail_separation_negative_quantile=0.25"   "train.tail_separation_min_positive=1" "train.tail_separation_min_negative=1"   "train.tail_separation_labels=[shot,save,set_piece]"   "train.ema.enabled=true" "train.ema_positive_retention_teacher=fixed_init"   "train.ema_positive_retention_weight=0.20" "train.ema_positive_retention_margin=0.0"   "train.ema_positive_retention_temperature=0.5"   "train.ema_positive_retention_labels=[shot,save,set_piece]"   "train.online_pair_consistency.clip_weight=0.01"   "train.online_pair_consistency.response_weight=0.002"   "train.early_stopping.enabled=false"   "eval.tuned_objective_by_class.shot=precision_at_recall_floor"   "eval.tuned_objective_by_class.save=precision_at_recall_floor"   "eval.tuned_objective_by_class.set_piece=precision_at_recall_floor"   "eval.tuned_min_recall_by_class.shot=0.85" "eval.tuned_min_recall_by_class.save=0.70"   "eval.tuned_min_recall_by_class.set_piece=0.70"   "eval.online_validation.enabled=true" "eval.online_validation.use_for_checkpoint_selection=true"   "eval.online_validation.tuned_objective_by_class.shot=precision_at_recall_floor"   "eval.online_validation.tuned_objective_by_class.save=precision_at_recall_floor"   "eval.online_validation.tuned_objective_by_class.set_piece=precision_at_recall_floor"   "eval.online_validation.tuned_min_recall_by_class.shot=0.85"   "eval.online_validation.tuned_min_recall_by_class.save=0.70"   "eval.online_validation.tuned_min_recall_by_class.set_piece=0.70"   "eval.external_audit.enabled=false" "data.num_workers_per_gpu=${num_workers_per_gpu}"   2>&1 | tee -a "${output_dir}/train_console.log"
