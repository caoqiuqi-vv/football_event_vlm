#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/new_users/qiuqi/code/dinov3-main"
source_run="${repo_dir}/outputs/football_events/vitl16_d7c_object_teacher_online_allclips_no_bad_media_from_fromlast_e8_20260901"
source_checkpoint="${SOURCE_CHECKPOINT:-${source_run}/epoch_2.pt}"
base_config="${BASE_CONFIG:-${source_run}/config.yaml}"
output_dir="${OUTPUT_DIR:-${repo_dir}/outputs/football_events/vitl16_d7c_object_motion_v3_balllora_fullwindow33_from_object_e2_20260903}"
python_bin="/home/new_users/qiuqi/miniconda3/envs/qiuqi_sam3/bin/python"
torchrun_bin="/home/new_users/qiuqi/miniconda3/envs/qiuqi_sam3/bin/torchrun"
gpu_list="${GPU_LIST:-1,4,6}"
per_gpu_batch_size="${PER_GPU_BATCH_SIZE:-1}"
grad_accum_steps="${GRAD_ACCUM_STEPS:-8}"
motion_frames_per_segment="${MOTION_FRAMES_PER_SEGMENT:-11}"
motion_image_size="${MOTION_IMAGE_SIZE:-[512,896]}"
motion_teacher_image_size="${MOTION_TEACHER_IMAGE_SIZE:-[512,896]}"
motion_class_evidence_floors="${MOTION_CLASS_EVIDENCE_FLOORS:-[0.02,0.02,0.02]}"
motion_teacher_batch_size="${MOTION_TEACHER_BATCH_SIZE:-16}"
motion_online_teacher_enabled="${MOTION_ONLINE_TEACHER_ENABLED:-true}"
motion_online_teacher_objects="${MOTION_ONLINE_TEACHER_OBJECTS:-[ball,goal,person]}"
motion_offline_ball_enabled="${MOTION_OFFLINE_BALL_ENABLED:-false}"
motion_offline_ball_index_root="${MOTION_OFFLINE_BALL_INDEX_ROOT:-}"
motion_offline_ball_max_delta="${MOTION_OFFLINE_BALL_MAX_DELTA_SEC:-0.10}"
motion_offline_ball_cache_videos="${MOTION_OFFLINE_BALL_CACHE_VIDEOS:-4}"
motion_object_loss_weights="${MOTION_OBJECT_LOSS_WEIGHTS:-[2.0,1.0,0.25]}"
motion_negative_weights="${MOTION_NEGATIVE_WEIGHTS:-[0.50,0.25,0.05]}"
motion_presence_negative_weights="${MOTION_PRESENCE_NEGATIVE_WEIGHTS:-[1.0,1.0,0.25]}"
motion_ball_strong_loss_weight="${MOTION_BALL_STRONG_LOSS_WEIGHT:-1.0}"
motion_ball_feature_layers="${MOTION_BALL_FEATURE_LAYERS:-[11,17,20,23]}"
motion_shared_anchor_ball_lora="${MOTION_SHARED_ANCHOR_BALL_LORA:-false}"
motion_object_cross_attention="${MOTION_OBJECT_CROSS_ATTENTION:-false}"
motion_object_cross_attention_max_delta="${MOTION_OBJECT_CROSS_ATTENTION_MAX_DELTA:-0.35}"
motion_object_cross_attention_gate_init="${MOTION_OBJECT_CROSS_ATTENTION_GATE_INIT:-0.25}"
motion_heatmap_patch_size="${MOTION_HEATMAP_PATCH_SIZE:-16}"
motion_heatmap_upsample_factor="${MOTION_HEATMAP_UPSAMPLE_FACTOR:-1}"
motion_ball_fpn_dim="${MOTION_BALL_FPN_DIM:-64}"
motion_legacy_frame_residual_enabled="${MOTION_LEGACY_FRAME_RESIDUAL_ENABLED:-true}"
motion_absence_presence_weights="${MOTION_ABSENCE_PRESENCE_WEIGHTS:-[0.05,0.02,0.02]}"
motion_presence_loss_weight="${MOTION_PRESENCE_LOSS_WEIGHT:-0.25}"
motion_event_residual_scale="${MOTION_EVENT_RESIDUAL_SCALE:-0.0}"
motion_ball_lora_lr="${MOTION_BALL_LORA_LR:-1.0e-6}"
motion_head_weight_decay="${MOTION_HEAD_WEIGHT_DECAY:-0.05}"
if [[ -n "${MOTION_CURRICULUM_STAGES:-}" ]]; then
  motion_curriculum_stages="${MOTION_CURRICULUM_STAGES}"
else
  motion_curriculum_stages='[{name: relation_only, start_epoch: 1, end_epoch: 1, overrides: {object_motion_event_residual_scale: 0.0, object_motion_relation_loss_weight: 1.0, object_motion_dense_rank_loss_weight: 0.15}}, {name: inject_010, start_epoch: 2, end_epoch: 2, overrides: {object_motion_event_residual_scale: 0.10, object_motion_relation_loss_weight: 1.0}}, {name: inject_025, start_epoch: 3, end_epoch: 4, overrides: {object_motion_event_residual_scale: 0.25, object_motion_relation_loss_weight: 1.0}}]'
fi
train_epochs="${TRAIN_EPOCHS:-4}"
train_log_interval="${TRAIN_LOG_INTERVAL:-20}"
IFS=',' read -r -a gpu_ids <<< "${gpu_list}"
gpu_count="${#gpu_ids[@]}"
if (( gpu_count < 1 )); then
  echo "GPU_LIST must contain at least one GPU" >&2
  exit 2
fi
logical_gpu_ids=""
for ((gpu_index = 0; gpu_index < gpu_count; gpu_index++)); do
  logical_gpu_ids+="${logical_gpu_ids:+,}${gpu_index}"
done

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

"${torchrun_bin}" --standalone --nproc-per-node="${gpu_count}" \
  -m football_object_motion.train \
  --config "${base_config}" \
  "output_dir=${output_dir}" \
  "device=cuda:0" \
  "gpu_ids=[${logical_gpu_ids}]" \
  "model.init_checkpoint=${source_checkpoint}" \
  "model.init_checkpoint_strict=false" \
  "model.object_spatial_aux.enabled=false" \
  "model.controlled_online_train_scope=all" \
  "model.freeze_backbone=true" \
  "model.finetune_last_blocks=0" \
  "model.backbone_frame_chunk_size=4" \
  "model.object_motion.enabled=true" \
  "model.object_motion.ball_lora_rank=4" \
  "model.object_motion.ball_lora_alpha=4.0" \
  "model.object_motion.ball_lora_last_blocks=8" \
  "model.object_motion.ball_feature_layers=${motion_ball_feature_layers}" \
  "model.object_motion.ball_layer_weights=[0.15,0.25,0.25,0.35]" \
  "model.object_motion.ball_topk=4" \
  "model.object_motion.ball_temperature=0.25" \
  "model.object_motion.backbone_frame_chunk_size=1" \
  "model.object_motion.checkpoint_trainable_blocks=true" \
  "model.object_motion.frames_per_segment=${motion_frames_per_segment}" \
  "model.object_motion.duration_sec=10.0" \
  "model.object_motion.image_size=${motion_image_size}" \
  "model.object_motion.teacher_image_size=${motion_teacher_image_size}" \
  "model.object_motion.patch_size=16" \
  "model.object_motion.heatmap_patch_size=${motion_heatmap_patch_size}" \
  "model.object_motion.heatmap_upsample_factor=${motion_heatmap_upsample_factor}" \
  "model.object_motion.ball_fpn_dim=${motion_ball_fpn_dim}" \
  "model.object_motion.shared_anchor_ball_lora=${motion_shared_anchor_ball_lora}" \
  "model.object_motion.object_cross_attention_enabled=${motion_object_cross_attention}" \
  "model.object_motion.object_cross_attention_max_delta=${motion_object_cross_attention_max_delta}" \
  "model.object_motion.object_cross_attention_gate_init=${motion_object_cross_attention_gate_init}" \
  "model.object_motion.legacy_frame_residual_enabled=${motion_legacy_frame_residual_enabled}" \
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
  "model.object_motion.gate_max=1.0" \
  "model.object_motion.gate_floor=0.25" \
  "model.object_motion.relation_delta=1.0" \
  "model.object_motion.fusion_delta=0.5" \
  "model.object_motion.evidence_gate_floor=0.02" \
  "model.object_motion.class_evidence_floors=${motion_class_evidence_floors}" \
  "model.object_motion.detach_detector_for_event=true" \
  "model.object_motion.learned_gate_budget=1.0" \
  "model.object_motion.residual_saturation_threshold=0.85" \
  "model.object_motion.pairwise_rank_margin=0.05" \
  "model.object_motion.pairwise_rank_temperature=0.10" \
  "model.object_motion.require_true_pairs=true" \
  "model.object_motion.reviewed_negative_manifests=[${repo_dir}/outputs/football_hard_negatives/reviewed_mid_score_v2_shot_save_setpiece.json,${repo_dir}/outputs/football_hard_negatives/other_action_reviewed_train_shot_save_score_filtered.json]" \
  "model.object_motion.pair_min_gap_sec=5.0" \
  "model.object_motion.negative_guard_margin=0.15" \
  "model.object_motion.positive_threshold=0.05" \
  "model.object_motion.object_loss_weights=${motion_object_loss_weights}" \
  "model.object_motion.negative_weights=${motion_negative_weights}" \
  "model.object_motion.presence_negative_weights=${motion_presence_negative_weights}" \
  "model.object_motion.online_teacher.enabled=${motion_online_teacher_enabled}" \
  "model.object_motion.online_teacher.ball_checkpoint=/mnt/data_7t/qiuqi/code/soccer/onlysoccer_1920_11s.pt" \
  "model.object_motion.online_teacher.scene_checkpoint=/home/new_users/qiuqi/code/det_and_track/checkpoints/yolo11m_person_goal_fieldLine_1920_7.22.pt" \
  "model.object_motion.online_teacher.ball_confidence=0.10" \
  "model.object_motion.online_teacher.goal_confidence=0.25" \
  "model.object_motion.online_teacher.person_confidence=0.25" \
  "model.object_motion.online_teacher.input_size=[1088,1920]" \
  "model.object_motion.online_teacher.batch_size=${motion_teacher_batch_size}" \
  "model.object_motion.online_teacher.half=true" \
  "model.object_motion.online_teacher.absence_presence_weights=${motion_absence_presence_weights}" \
  "model.object_motion.online_teacher.enabled_objects=${motion_online_teacher_objects}" \
  "model.object_motion.offline_ball_teacher.enabled=${motion_offline_ball_enabled}" \
  "model.object_motion.offline_ball_teacher.index_root=${motion_offline_ball_index_root}" \
  "model.object_motion.offline_ball_teacher.max_time_delta_sec=${motion_offline_ball_max_delta}" \
  "model.object_motion.offline_ball_teacher.ball_sigma_patches=1.25" \
  "model.object_motion.offline_ball_teacher.cache_videos=${motion_offline_ball_cache_videos}" \
  "video.positive_window_strategy=anchor_range_jitter" \
  "video.positive_anchor_min_sec=0.5" \
  "video.positive_anchor_max_sec=9.5" \
  "video.temporal_jitter_sec=0.0" \
  "data.long_video.negative_ratio=5.0" \
  "data.long_video.hard_negative.enabled=true" \
  "data.long_video.hard_negative.manifests=[${repo_dir}/outputs/football_hard_negatives/reviewed_mid_score_v2_shot_save_setpiece.json,${repo_dir}/outputs/football_hard_negatives/other_action_reviewed_train_shot_save_score_filtered.json]" \
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
  "train.epochs=${train_epochs}" \
  "train.scheduler_epochs=${train_epochs}" \
  "train.log_interval=${train_log_interval}" \
  "train.per_gpu_batch_size=${per_gpu_batch_size}" \
  "train.grad_accum_steps=${grad_accum_steps}" \
  "train.lr_per_gpu=2.0e-5" \
  "train.warmup.ratio=0.05" \
  "train.clip_loss_weight=1.0" \
  "train.clip_loss_balance=per_class_equal_pos_neg" \
  "train.object_heatmap_loss_weight=${OBJECT_AUX_LOSS_WEIGHT:-0.25}" \
  "train.object_motion_heatmap_loss_weight=0.50" \
  "train.object_motion_distribution_loss_weight=0.50" \
  "train.object_motion_presence_loss_weight=${motion_presence_loss_weight}" \
  "train.object_motion_coordinate_loss_weight=0.25" \
  "train.object_motion_consistency_loss_weight=0.20" \
  "train.object_motion_frame_loss_weight=0.10" \
  "train.object_motion_dense_rank_loss_weight=0.15" \
  "train.object_motion_no_evidence_loss_weight=0.10" \
  "train.object_motion_residual_energy_loss_weight=0.0" \
  "train.object_motion_gate_budget_loss_weight=0.0" \
  "train.object_motion_saturation_loss_weight=0.02" \
  "train.object_motion_ball_strong_loss_weight=${motion_ball_strong_loss_weight}" \
  "train.object_motion_ball_center_loss_weight=0.50" \
  "train.object_motion_ball_contrastive_loss_weight=0.20" \
  "train.object_motion_ball_track_loss_weight=0.10" \
  "train.object_motion_ball_preserve_loss_weight=0.05" \
  "train.object_motion_relation_loss_weight=1.0" \
  "train.object_motion_guard_loss_weight=1.0" \
  "train.ball_lora_lr=${motion_ball_lora_lr}" \
  "train.ball_lora_weight_decay=0.0" \
  "train.ball_lora_grad_clip=0.10" \
  "train.object_motion_head_lr=2.0e-5" \
  "train.object_motion_head_weight_decay=${motion_head_weight_decay}" \
  "train.object_motion_head_grad_clip=1.0" \
  "train.object_motion_event_residual_scale=${motion_event_residual_scale}" \
  "train.positive_retention_loss_weight=1.0" \
  "train.positive_retention_margin=0.0" \
  "train.frame_det_loss_weight=0.0" \
  "train.save_epoch_checkpoints=true" \
  "train.checkpoint_selection=online_recall_floor_min_review" \
  "train.curriculum_stages=${motion_curriculum_stages}" \
  "eval.online_validation.enabled=true" \
  "eval.per_gpu_batch_size=1" \
  "eval.online_validation.window_stride_sec=5.0" \
  "eval.online_validation.nms_radius_sec=5.0" \
  "eval.online_validation.tolerance_sec=3.0" \
  "eval.online_validation.capped_clip_sec=10.0" \
  "eval.online_validation.score_fusion=clip_x_frame_peak" \
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
