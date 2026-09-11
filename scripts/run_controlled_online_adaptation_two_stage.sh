#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/new_users/qiuqi/code/dinov3-main"
base_config="${repo_dir}/configs/football/dinov3_vitl16_online_simulation_e14_signed_causal_continue4.yaml"
init_checkpoint="${CONTROLLED_INIT_CHECKPOINT:-${repo_dir}/outputs/football_events/vitl16_independent_preprojection_evidence_ddp_lora_v1_accelerated_resume_e1/best.pt}"
root="${CONTROLLED_OUTPUT_ROOT:-${repo_dir}/outputs/football_events/vitl16_controlled_online_adaptation_from_last6r8}"
stage1_dir="${root}/stage1_frozen_lora_temporal_heads"
stage2_e1_dir="${root}/stage2_low_lr_lora_temporal_heads_e1"
stage2_e2_dir="${root}/stage2_low_lr_lora_temporal_heads_e2"
gpu_list="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
torchrun_bin="/home/new_users/qiuqi/miniconda3/envs/qiuqi_sam3/bin/torchrun"
python_bin="/home/new_users/qiuqi/miniconda3/envs/qiuqi_sam3/bin/python"

mkdir -p "${root}" "${stage1_dir}" "${stage2_e1_dir}" "${stage2_e2_dir}"
cd "${repo_dir}"
export CUDA_VISIBLE_DEVICES="${gpu_list}"
export PYTHONPATH="${repo_dir}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "${root}/pipeline.log"
}

common_overrides=(
  "train.resume.enabled=false"
  "train.epochs=2"
  "train.scheduler_epochs=2"
  "train.save_epoch_checkpoints=false"
  "train.early_stopping.enabled=true"
  "train.early_stopping.monitor=selection_score"
  "train.early_stopping.min_epoch=1"
  "train.early_stopping.patience=1"
  "train.class_evidence_cross_window_weight=0.0"
  "train.class_evidence_signed_causal_loss_weight=0.0"
  "train.set_piece_subtype_loss_weight=0.0"
  "train.raw_context_span_loss_weight=0.0"
  "eval.external_audit.enabled=false"
  "eval.online_validation.enabled=true"
  "eval.online_validation.window_stride_sec=5.0"
  "eval.online_validation.nms_radius_sec=5.0"
  "eval.online_validation.tolerance_sec=5.0"
  "eval.online_validation.capped_clip_sec=10.0"
  "eval.online_validation.tune_event_thresholds=true"
  "eval.online_validation.max_threshold_candidates=401"
  "eval.online_validation.use_for_checkpoint_selection=true"
  "eval.online_validation.tuned_objective_by_class.shot=precision_at_recall_floor"
  "eval.online_validation.tuned_objective_by_class.save=f1"
  "eval.online_validation.tuned_objective_by_class.set_piece=f1"
  "eval.online_validation.tuned_min_recall_by_class.shot=0.85"
  "eval.online_validation.tuned_min_recall_by_class.save=null"
  "eval.online_validation.tuned_min_recall_by_class.set_piece=null"
)

log "baseline Online Val15 start checkpoint=${init_checkpoint}"
ONLINE_EVAL_WORLD_SIZE=4 ONLINE_EVAL_PER_GPU_BATCH=4 \
  bash scripts/evaluate_online_val15_checkpoint.sh \
  "${base_config}" "${init_checkpoint}" "${root}/baseline_online_val15.json"
log "baseline Online Val15 complete"

log "stage1 start scope=temporal_heads freeze_lora=true"
"${torchrun_bin}" --standalone --nproc-per-node=4 \
  train_football_events_online_simulation_e13.py \
  --config "${base_config}" \
  "output_dir=${stage1_dir}" \
  "model.init_checkpoint=${init_checkpoint}" \
  "model.init_checkpoint_strict=false" \
  "model.freeze_loaded_backbone=true" \
  "model.controlled_online_train_scope=temporal_heads" \
  "${common_overrides[@]}" \
  2>&1 | tee -a "${stage1_dir}/train_console.log"

"${python_bin}" scripts/controlled_online_stage_gate.py \
  --baseline-report "${root}/baseline_online_val15.json" \
  --candidate-run "${stage1_dir}" \
  --output "${root}/stage1_gate.json" \
  --min-score-gain 0.002 \
  --max-participation-increase 0.02 \
  2>&1 | tee -a "${root}/pipeline.log"

if ! jq -e '.passed == true' "${root}/stage1_gate.json" >/dev/null; then
  log "stage1 gate failed; stage2 is intentionally not started"
  exit 0
fi

log "stage2 epoch1 start scope=lora_temporal_heads lora_lr_per_gpu=1.25e-7"
"${torchrun_bin}" --standalone --nproc-per-node=4 \
  train_football_events_online_simulation_e13.py \
  --config "${base_config}" \
  "output_dir=${stage2_e1_dir}" \
  "model.init_checkpoint=${stage1_dir}/best.pt" \
  "model.init_checkpoint_strict=false" \
  "model.freeze_loaded_backbone=false" \
  "model.controlled_online_train_scope=lora_temporal_heads" \
  "${common_overrides[@]}" \
  "train.backbone_lr_per_gpu=1.25e-7" \
  "train.epochs=1" \
  "train.scheduler_epochs=1" \
  2>&1 | tee -a "${stage2_e1_dir}/train_console.log"

"${python_bin}" scripts/controlled_online_stage_gate.py \
  --baseline-run "${stage1_dir}" \
  --candidate-run "${stage2_e1_dir}" \
  --output "${root}/stage2_e1_vs_stage1_gate.json" \
  --min-score-gain 0.002 \
  --max-participation-increase 0.02 \
  2>&1 | tee -a "${root}/pipeline.log"

if ! jq -e '.passed == true' "${root}/stage2_e1_vs_stage1_gate.json" >/dev/null; then
  log "stage2 epoch1 gate failed; LoRA epoch2 is intentionally not started"
  exit 0
fi

log "stage2 epoch2 start from positive epoch1 checkpoint"
"${torchrun_bin}" --standalone --nproc-per-node=4 \
  train_football_events_online_simulation_e13.py \
  --config "${base_config}" \
  "output_dir=${stage2_e2_dir}" \
  "model.init_checkpoint=${stage2_e1_dir}/best.pt" \
  "model.init_checkpoint_strict=false" \
  "model.freeze_loaded_backbone=false" \
  "model.controlled_online_train_scope=lora_temporal_heads" \
  "${common_overrides[@]}" \
  "train.backbone_lr_per_gpu=1.25e-7" \
  "train.epochs=1" \
  "train.scheduler_epochs=1" \
  2>&1 | tee -a "${stage2_e2_dir}/train_console.log"

"${python_bin}" scripts/controlled_online_stage_gate.py \
  --baseline-run "${stage2_e1_dir}" \
  --candidate-run "${stage2_e2_dir}" \
  --output "${root}/stage2_e2_vs_e1_gate.json" \
  --min-score-gain 0.0 \
  --max-participation-increase 0.02 \
  2>&1 | tee -a "${root}/pipeline.log"

log "controlled Online adaptation pipeline complete"
