#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/new_users/qiuqi/code/dinov3-main"
base_config="$repo_dir/configs/football/dinov3_vitl16_d7c_online_e16_reviewguard_720p_from_best_20260831.yaml"
init_checkpoint="$repo_dir/outputs/football_events/vitl16_d7c_fullimage_latest_split_dino_pretrained_lora_ddp_720p_fromlast_e8_20260829/best.pt"
output_dir="$repo_dir/outputs/football_events/vitl16_d7c_motionguided_highres_roi_residual_allgt_720p_fromlast_e8_e4_20260831"
torchrun_bin="/home/new_users/qiuqi/miniconda3/envs/qiuqi_sam3/bin/torchrun"

mkdir -p "$output_dir"
cd "$repo_dir"
export CUDA_VISIBLE_DEVICES="0,1,6"
export PYTHONPATH="$repo_dir"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

printf '%s\n' \
  "protocol=motionguided-highres-roi anchor=fromlast_e8/best.pt" \
  "gpu_list=0,1,6 world_size=3 effective_batch=10x3x3=90" \
  "anchor=frozen full-image DINO+temporal+head" \
  "roi=2 temporal candidates x4 frames native720p crop384 DINO re-encode" \
  "selector=semantic_query+standardized_two_sided_patch_motion" \
  "fusion=anchor_logits+gated_tanh_residual max_abs_delta=1 zero_init" \
  >> "$output_dir/pipeline.log"

"$torchrun_bin" --standalone --nproc-per-node=3 \
  train_football_events_online_simulation_e16.py \
  --config "$base_config" \
  "output_dir=$output_dir" \
  "gpu_ids=[0,1,2]" \
  "model.init_checkpoint=$init_checkpoint" \
  "model.init_checkpoint_strict=false" \
  "model.freeze_loaded_backbone=true" \
  "model.freeze_global_branch=true" \
  "model.controlled_online_train_scope=highres_residual" \
  "model.backbone_frame_chunk_size=8" \
  "model.gradient_checkpointing=true" \
  "model.spatial_attention.enabled=false" \
  "model.highres_glimpse.enabled=true" \
  "model.highres_glimpse.pool_frames=24" \
  "model.highres_glimpse.pool_image_size=[720,1280]" \
  "model.highres_glimpse.candidates=2" \
  "model.highres_glimpse.frames_per_candidate=4" \
  "model.highres_glimpse.crop_size=384" \
  "model.highres_glimpse.attention_dim=256" \
  "model.highres_glimpse.nms_radius=2" \
  "model.highres_glimpse.min_scale=[0.16,0.22]" \
  "model.highres_glimpse.max_scale=[0.45,0.60]" \
  "model.highres_glimpse.fusion_mode=bounded_residual" \
  "model.highres_glimpse.residual_max_delta=1.0" \
  "model.highres_glimpse.residual_gate_init=0.10" \
  "model.highres_glimpse.motion_prior_weight=1.5" \
  "model.highres_glimpse.attention_temperature=0.70" \
  "model.highres_glimpse.freeze_selector=true" \
  "data.long_video.online_simulation.event_sampling_mode=cover_all_once" \
  "data.long_video.online_simulation.grid_clip_loss_weight=0.35" \
  "data.long_video.online_simulation.grid_frame_loss_weight=0.0" \
  "data.long_video.online_simulation.edge_clip_loss_weight=0.35" \
  "data.long_video.online_simulation.edge_frame_loss_weight=0.0" \
  "data.long_video.online_simulation.clean_background_windows_per_epoch=1200" \
  "data.num_workers_per_gpu=4" \
  "data.prefetch_factor=2" \
  "data.video_reader_cache_size=4" \
  "train.epochs=4" \
  "train.scheduler_epochs=4" \
  "train.per_gpu_batch_size=10" \
  "train.grad_accum_steps=3" \
  "train.ddp_find_unused_parameters=false" \
  "train.lr_per_gpu=1.0e-5" \
  "train.highres_glimpse_lr_per_gpu=2.5e-5" \
  "train.temporal_lr_per_gpu=1.0e-5" \
  "train.head_lr_per_gpu=1.0e-5" \
  "train.frame_det_loss_weight=0.0" \
  "train.frame_heatmap_loss_weight=0.0" \
  "train.frame_mil_loss_weight=0.0" \
  "train.frame_rank_loss_weight=0.0" \
  "train.highres_local_frame_loss_weight=0.20" \
  "train.highres_local_clip_loss_weight=0.50" \
  "train.highres_local_clip_balance=per_class_equal_pos_neg" \
  "train.highres_causal_shuffle_loss_weight=0.15" \
  "train.highres_causal_start_epoch=1" \
  "train.highres_causal_shuffle_margin=0.25" \
  "train.highres_crop_entropy_loss_weight=0.01" \
  "train.highres_crop_entropy_target=0.70" \
  "train.clean_negative_rank_loss_weight=0.15" \
  "train.clean_negative_rank_branch=fused" \
  "train.clean_negative_rank_labels=[shot,save,set_piece]" \
  "train.clean_negative_rank_margin=0.75" \
  "train.clean_negative_rank_pairs_per_positive=4" \
  "train.positive_retention_loss_weight=0.25" \
  "train.positive_retention_margin=0.0" \
  "train.positive_retention_branch=fused" \
  "train.fixed_teacher_logit_guard_weight=0.0" \
  "train.ema.enabled=true" \
  "train.ema.decay=0.995" \
  "train.ema.evaluate=true" \
  "train.ema.evaluate_raw=false" \
  "train.ema.selection_source=ema" \
  "train.ema.save_best_as_ema=true" \
  "train.checkpoint_selection=online_recall_floor_min_review" \
  "train.save_epoch_checkpoints=true" \
  "train.early_stopping.enabled=false" \
  "eval.online_validation.enabled=true" \
  "eval.online_validation.window_stride_sec=5.0" \
  "eval.online_validation.nms_radius_sec=0.0" \
  "eval.online_validation.tolerance_sec=3.0" \
  "eval.online_validation.score_fusion=clip" \
  "eval.online_validation.tune_event_thresholds=true" \
  "eval.online_validation.use_for_checkpoint_selection=true" \
  "eval.online_validation.tuned_min_recall_by_class.shot=0.90" \
  "eval.online_validation.tuned_min_recall_by_class.save=0.85" \
  "eval.online_validation.tuned_min_recall_by_class.set_piece=0.85" \
  "eval.external_audit.enabled=false" \
  "raw_set_piece_supervision.enabled=false" \
  2>&1 | tee -a "$output_dir/train_console.log"
