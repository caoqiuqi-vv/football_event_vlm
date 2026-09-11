#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/new_users/qiuqi/code/dinov3-main"
source_run="${repo_dir}/outputs/football_events/vitl16_d7c_object_teacher_online_allclips_no_bad_media_from_fromlast_e8_20260901"
source_checkpoint="${source_run}/epoch_2.pt"
base_config="${source_run}/config.yaml"
output_dir="${repo_dir}/outputs/football_events/vitl16_d7c_object_motion_v2_antcollapse_dense33_from_object_e2_20260902"
python_bin="/home/new_users/qiuqi/miniconda3/envs/qiuqi_sam3/bin/python"
torchrun_bin="/home/new_users/qiuqi/miniconda3/envs/qiuqi_sam3/bin/torchrun"
gpu_list="${GPU_LIST:-1,4,6}"
per_gpu_batch_size="${PER_GPU_BATCH_SIZE:-1}"
grad_accum_steps="${GRAD_ACCUM_STEPS:-8}"

if [[ ! -s "${source_checkpoint}" ]]; then
  echo "waiting for ${source_checkpoint}" >&2
  exit 75
fi
mkdir -p "${output_dir}"
cd "${repo_dir}"

export CUDA_VISIBLE_DEVICES="${gpu_list}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

"${torchrun_bin}" --standalone --nproc-per-node=3 \
  -m football_object_motion.train \
  --config "${base_config}" \
  "output_dir=${output_dir}" \
  "model.init_checkpoint=${source_checkpoint}" \
  "model.init_checkpoint_strict=false" \
  "model.object_spatial_aux.enabled=false" \
  "model.controlled_online_train_scope=all" \
  "model.freeze_backbone=true" \
  "model.finetune_last_blocks=0" \
  "model.backbone_frame_chunk_size=4" \
  "model.object_motion.enabled=true" \
  "model.object_motion.frames=33" \
  "model.object_motion.duration_sec=4.0" \
  "model.object_motion.image_size=[512,896]" \
  "model.object_motion.patch_size=16" \
  "model.object_motion.hidden_dim=512" \
  "model.object_motion.num_heads=8" \
  "model.object_motion.temporal_layers=2" \
  "model.object_motion.dropout=0.1" \
  "model.object_motion.topk_ratios=[0.01,0.05,0.08]" \
  "model.object_motion.temperature=0.5" \
  "model.object_motion.residual_max_delta=0.25" \
  "model.object_motion.frame_residual_max_delta=0.15" \
  "model.object_motion.clip_residual_max_delta=0.25" \
  "model.object_motion.gate_init=0.10" \
  "model.object_motion.gate_max=0.50" \
  "model.object_motion.evidence_gate_floor=0.02" \
  "model.object_motion.detach_detector_for_event=true" \
  "model.object_motion.learned_gate_budget=0.25" \
  "model.object_motion.residual_saturation_threshold=0.85" \
  "model.object_motion.pairwise_rank_margin=0.05" \
  "model.object_motion.pairwise_rank_temperature=0.10" \
  "model.object_motion.positive_threshold=0.05" \
  "model.object_motion.object_loss_weights=[2.0,1.0,0.25]" \
  "model.object_motion.negative_weights=[0.50,0.25,0.05]" \
  "model.object_motion.presence_negative_weights=[1.0,1.0,0.25]" \
  "model.object_motion.online_teacher.enabled=true" \
  "model.object_motion.online_teacher.ball_checkpoint=/mnt/data_7t/qiuqi/code/soccer/onlysoccer_1920_11s.pt" \
  "model.object_motion.online_teacher.scene_checkpoint=/home/new_users/qiuqi/code/det_and_track/checkpoints/yolo11m_person_goal_fieldLine_1920_7.22.pt" \
  "model.object_motion.online_teacher.ball_confidence=0.10" \
  "model.object_motion.online_teacher.goal_confidence=0.25" \
  "model.object_motion.online_teacher.person_confidence=0.25" \
  "model.object_motion.online_teacher.input_size=[1088,1920]" \
  "model.object_motion.online_teacher.batch_size=16" \
  "model.object_motion.online_teacher.half=true" \
  "video.positive_window_strategy=anchor_range_jitter" \
  "video.positive_anchor_min_sec=0.5" \
  "video.positive_anchor_max_sec=9.5" \
  "video.temporal_jitter_sec=0.0" \
  "data.long_video.negative_ratio=5.0" \
  "data.long_video.negative_ratio_by_split.train=5.0" \
  "data.long_video.online_simulation.enabled=true" \
  "data.long_video.online_simulation.window_stride_sec=5.0" \
  "data.long_video.online_simulation.primary_positive_min_sec=3.0" \
  "data.long_video.online_simulation.primary_positive_max_sec=7.0" \
  "data.long_video.online_simulation.grid_clip_loss_weight=0.35" \
  "data.long_video.online_simulation.grid_frame_loss_weight=0.50" \
  "data.long_video.online_simulation.edge_clip_loss_weight=0.35" \
  "data.long_video.online_simulation.edge_frame_loss_weight=0.50" \
  "data.long_video.online_simulation.event_sampling_mode=cover_all_once" \
  "data.long_video.online_simulation.clean_background_windows_per_epoch=2400" \
  "data.long_video.online_simulation.near_event_ignore_sec=2.0" \
  "data.long_video.online_simulation.near_event_anchor_gap_min_sec=5.0" \
  "data.long_video.online_simulation.near_event_anchor_gap_max_sec=15.0" \
  "data.long_video.split_files.val=[configs/football/splits/object_motion_dense_sentinel3.txt]" \
  "data.num_workers_per_gpu=1" \
  "data.prefetch_factor=1" \
  "train.epochs=6" \
  "train.scheduler_epochs=6" \
  "train.per_gpu_batch_size=${per_gpu_batch_size}" \
  "train.grad_accum_steps=${grad_accum_steps}" \
  "train.lr_per_gpu=2.0e-5" \
  "train.warmup.ratio=0.05" \
  "train.clip_loss_weight=1.0" \
  "train.clip_loss_balance=per_class_equal_pos_neg" \
  "train.object_heatmap_loss_weight=1.0" \
  "train.object_motion_heatmap_loss_weight=0.50" \
  "train.object_motion_distribution_loss_weight=0.50" \
  "train.object_motion_presence_loss_weight=0.25" \
  "train.object_motion_coordinate_loss_weight=0.25" \
  "train.object_motion_consistency_loss_weight=0.20" \
  "train.object_motion_frame_loss_weight=0.10" \
  "train.object_motion_dense_rank_loss_weight=0.15" \
  "train.object_motion_no_evidence_loss_weight=0.10" \
  "train.object_motion_residual_energy_loss_weight=0.50" \
  "train.object_motion_gate_budget_loss_weight=0.20" \
  "train.object_motion_saturation_loss_weight=0.20" \
  "train.object_motion_event_residual_scale=1.0" \
  "train.positive_retention_loss_weight=1.0" \
  "train.positive_retention_margin=0.0" \
  "train.frame_det_loss_weight=0.0" \
  "train.save_epoch_checkpoints=true" \
  "train.checkpoint_selection=online_recall_floor_min_review" \
  "train.curriculum_stages=[{name: detector_distill, start_epoch: 1, end_epoch: 2, overrides: {clip_loss_weight: 1.0, object_motion_event_residual_scale: 0.0, object_motion_frame_loss_weight: 0.0, object_motion_dense_rank_loss_weight: 0.0}}, {name: relation_warmup, start_epoch: 3, end_epoch: 4, overrides: {clip_loss_weight: 1.0, object_motion_event_residual_scale: 0.25, object_motion_frame_loss_weight: 0.05, object_motion_dense_rank_loss_weight: 0.10}}, {name: cautious_fusion, start_epoch: 5, end_epoch: 6, overrides: {clip_loss_weight: 1.0, object_motion_event_residual_scale: 1.0, object_motion_frame_loss_weight: 0.10, object_motion_dense_rank_loss_weight: 0.15}}]" \
  "eval.online_validation.enabled=true" \
  "eval.per_gpu_batch_size=1" \
  "eval.online_validation.window_stride_sec=5.0" \
  "eval.online_validation.nms_radius_sec=5.0" \
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
  2>&1 | tee -a "${output_dir}/train_console.log"
