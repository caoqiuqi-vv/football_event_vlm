#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/new_users/qiuqi/code/dinov3-main"
base_config="$repo_dir/configs/football/dinov3_vitl16_d7c_online_e16_reviewguard_720p_from_best_20260831.yaml"
init_checkpoint="$repo_dir/outputs/football_events/vitl16_d7c_fullimage_latest_split_dino_pretrained_lora_ddp_720p_fromlast_e8_20260829/best.pt"
output_dir="$repo_dir/outputs/football_events/vitl16_d7c_online_e16_reviewguard_allgt_720p_from_best_e4_20260831"
run_name="vitl16_d7c_online_e16_reviewguard_allgt_720p_best_thirdparty18_dense_s5"
eval_root="$repo_dir/outputs/football_eval_runs"
run_dir="$eval_root/$run_name"
test_ids="$repo_dir/configs/football/splits/thirdparty18_test_long15_val_no_pn_train/thirdparty18_test_video_ids.txt"
torchrun_bin="/home/new_users/qiuqi/miniconda3/envs/qiuqi_sam3/bin/torchrun"
python_bin="/home/new_users/qiuqi/miniconda3/envs/qiuqi_sam3/bin/python"

mkdir -p "$output_dir"
cd "$repo_dir"
export CUDA_VISIBLE_DEVICES="4,6"
export PYTHONPATH="$repo_dir"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "$output_dir/pipeline.log"
}

if [[ ! -f "$init_checkpoint" ]]; then
  log "ERROR missing init checkpoint: $init_checkpoint"
  exit 2
fi

log "protocol=E16 exact-grid 720p/16f no-NMS dense review guard"
log "init_checkpoint=$init_checkpoint"
log "gpu_list=4,6 world_size=2 effective_batch=10x2x4"
log "event_sampling=cover_all_once (every eligible GT once per epoch)"
log "supervision=grid_clip0.35 grid_frame0.5 edge_clip0.35 edge_frame0.5 frame_det0.15"
log "teacher=fixed_init bidirectional_one_sided_logit_guard weight0.35"
log "checkpoint_selection=shot90 save85 set_piece85 feasible then min raw review union"

if [[ ! -f "$output_dir/best.pt" ]]; then
  "$torchrun_bin" --standalone --nproc-per-node=2 \
    train_football_events_online_simulation_e16.py \
    --config "$base_config" \
    "output_dir=$output_dir" \
    "gpu_ids=[0,1]" \
    "model.init_checkpoint=$init_checkpoint" \
    "model.init_checkpoint_strict=false" \
    "model.freeze_loaded_backbone=false" \
    "model.controlled_online_train_scope=temporal_heads" \
    "model.backbone_frame_chunk_size=8" \
    "video.image_size=[720,1280]" \
    "video.num_frames=16" \
    "data.long_video.online_simulation.grid_clip_loss_weight=0.35" \
    "data.long_video.online_simulation.grid_frame_loss_weight=0.5" \
    "data.long_video.online_simulation.edge_clip_loss_weight=0.35" \
    "data.long_video.online_simulation.edge_frame_loss_weight=0.5" \
    "data.long_video.online_simulation.event_sampling_mode=cover_all_once" \
    "data.long_video.online_simulation.clean_background_windows_per_epoch=1200" \
    "train.resume.enabled=false" \
    "train.epochs=4" \
    "train.scheduler_epochs=4" \
    "train.per_gpu_batch_size=10" \
    "train.grad_accum_steps=4" \
    "train.lr_per_gpu=6.25e-6" \
    "train.temporal_lr_per_gpu=6.25e-6" \
    "train.head_lr_per_gpu=9.375e-6" \
    "train.frame_det_loss_weight=0.15" \
    "train.positive_retention_loss_weight=0.0" \
    "train.ema_positive_retention_teacher=fixed_init" \
    "train.ema_positive_retention_weight=0.0" \
    "train.fixed_teacher_logit_guard_weight=0.35" \
    "train.fixed_teacher_positive_margin=0.0" \
    "train.fixed_teacher_negative_margin=0.0" \
    "train.fixed_teacher_logit_guard_labels=[shot,save,set_piece]" \
    "train.online_pair_consistency.enabled=false" \
    "train.checkpoint_selection=online_recall_floor_min_review" \
    "train.checkpoint_min_recall=0.85" \
    "train.save_epoch_checkpoints=true" \
    "train.early_stopping.enabled=false" \
    "train.ema.enabled=true" \
    "train.ema.evaluate=true" \
    "train.ema.evaluate_raw=false" \
    "train.ema.selection_source=ema" \
    "train.ema.save_best_as_ema=true" \
    "eval.tuned_objective_by_class.shot=precision_at_recall_floor" \
    "eval.tuned_objective_by_class.save=precision_at_recall_floor" \
    "eval.tuned_objective_by_class.set_piece=precision_at_recall_floor" \
    "eval.tuned_min_recall_by_class.shot=0.90" \
    "eval.tuned_min_recall_by_class.save=0.85" \
    "eval.tuned_min_recall_by_class.set_piece=0.85" \
    "eval.online_validation.enabled=true" \
    "eval.online_validation.window_stride_sec=5.0" \
    "eval.online_validation.nms_radius_sec=0.0" \
    "eval.online_validation.tolerance_sec=3.0" \
    "eval.online_validation.capped_clip_sec=10.0" \
    "eval.online_validation.score_fusion=clip" \
    "eval.online_validation.tune_event_thresholds=true" \
    "eval.online_validation.max_threshold_candidates=401" \
    "eval.online_validation.use_for_checkpoint_selection=true" \
    "eval.online_validation.tuned_objective_by_class.shot=precision_at_recall_floor" \
    "eval.online_validation.tuned_objective_by_class.save=precision_at_recall_floor" \
    "eval.online_validation.tuned_objective_by_class.set_piece=precision_at_recall_floor" \
    "eval.online_validation.tuned_min_recall_by_class.shot=0.90" \
    "eval.online_validation.tuned_min_recall_by_class.save=0.85" \
    "eval.online_validation.tuned_min_recall_by_class.set_piece=0.85" \
    "eval.external_audit.enabled=false" \
    "raw_set_piece_supervision.enabled=false" \
    2>&1 | tee -a "$output_dir/train_console.log"
else
  log "training skipped: existing best.pt"
fi

if [[ ! -f "$output_dir/best.pt" ]]; then
  log "ERROR training ended without best.pt"
  exit 3
fi

log "test18 dense inference start checkpoint=$output_dir/best.pt"
"$python_bin" scripts/evaluate_football_model.py \
  --checkpoint "$output_dir/best.pt" \
  --mode dense \
  --video-id-file "$test_ids" \
  --gt-dir /mnt/data_16t/football/football_events_human_repair \
  --video-root xbotgo_0608=/mnt/data_16t/football/raw_video_720P \
  --output-root "$eval_root" \
  --run-name "$run_name" \
  --clip-sec 10 \
  --stride-sec 5 \
  --image-size 720,1280 \
  --batch-size 8 \
  --num-workers 2 \
  --device cuda:0 \
  --gpu-ids 0,1 \
  --thresholds checkpoint \
  --match-tolerance-sec 3 \
  --score-source clip \
  --prediction-postprocess window_overlap \
  --gt-merge-gap-sec 0 \
  --spatial-crop-mode none \
  --force \
  2>&1 | tee -a "$output_dir/test18_dense_console.log"

adaptive_report="$run_dir/video_adaptive_thresholds_loov_tol3.json"
workload_report="$run_dir/adaptive_manual_workload_tol3.json"
log "LOOV adaptive threshold evaluation start"
"$python_bin" scripts/analyze_video_adaptive_dense_thresholds.py \
  --run-dir "$run_dir" \
  --video-id-file "$test_ids" \
  --labels shot,save,set_piece \
  --match-tolerance-sec 3 \
  --recall-floors shot=0.90,save=0.85,set_piece=0.85 \
  --output "$adaptive_report" \
  2>&1 | tee -a "$output_dir/test18_postprocess.log"

"$python_bin" scripts/summarize_adaptive_dense_review_workload.py \
  --run-dir "$run_dir" \
  --adaptive-report "$adaptive_report" \
  --video-id-file "$test_ids" \
  --labels shot,save,set_piece \
  --match-tolerance-sec 3 \
  --merge-gap-sec 0 \
  --manual-overhead-sec 3 \
  --output "$workload_report" \
  2>&1 | tee -a "$output_dir/test18_postprocess.log"

log "pipeline complete adaptive_report=$adaptive_report workload_report=$workload_report"
