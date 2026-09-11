#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/new_users/qiuqi/code/dinov3-main"
v4_dir="${repo_dir}/outputs/football_events/vitl16_d7c_object_motion_v4_causal33_720p_aux025_4gpu_from_object_e2_20260903"

export OUTPUT_DIR="${OUTPUT_DIR:-${repo_dir}/outputs/football_events/vitl16_d7c_object_motion_v6_shared_ball_fpn8_crossattn_4gpu_from_v4e4_20260904}"
export SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${v4_dir}/epoch_4.pt}"
export BASE_CONFIG="${BASE_CONFIG:-${v4_dir}/config.yaml}"
export GPU_LIST="${GPU_LIST:-0,1,4,7}"
export PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-1}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-12}"
export MOTION_FRAMES_PER_SEGMENT="${MOTION_FRAMES_PER_SEGMENT:-11}"
export MOTION_IMAGE_SIZE="${MOTION_IMAGE_SIZE:-[720,1280]}"
export MOTION_TEACHER_IMAGE_SIZE="${MOTION_TEACHER_IMAGE_SIZE:-[720,1280]}"
export MOTION_TEACHER_BATCH_SIZE="${MOTION_TEACHER_BATCH_SIZE:-8}"

# Share the learned BallLoRA with the event anchor.  The frozen no-LoRA anchor
# is still evaluated as a retention reference.
export MOTION_SHARED_ANCHOR_BALL_LORA=true
export MOTION_OBJECT_CROSS_ATTENTION=true
export MOTION_OBJECT_CROSS_ATTENTION_MAX_DELTA=0.35
export MOTION_OBJECT_CROSS_ATTENTION_GATE_INIT=0.25
export MOTION_LEGACY_FRAME_RESIDUAL_ENABLED=false

# Earlier/middle DINO layers feed a light stride-8 FPN.  DINO itself remains
# patch-16, avoiding the quadratic cost of a patch-8 ViT-L backbone.
export MOTION_BALL_FEATURE_LAYERS="[5,11,17,23]"
export MOTION_HEATMAP_PATCH_SIZE=8
export MOTION_HEATMAP_UPSAMPLE_FACTOR=2
export MOTION_BALL_FPN_DIM=64

# Object order is [ball, goal, person]. Person is retained only as negligible
# relation context; it cannot dominate auxiliary optimization.
export MOTION_OBJECT_LOSS_WEIGHTS="[4.0,1.0,0.02]"
export MOTION_NEGATIVE_WEIGHTS="[0.50,0.25,0.0]"
export MOTION_PRESENCE_NEGATIVE_WEIGHTS="[0.0,0.0,0.0]"
export MOTION_ABSENCE_PRESENCE_WEIGHTS="[0.0,0.0,0.0]"
export MOTION_PRESENCE_LOSS_WEIGHT=0.0
export MOTION_BALL_STRONG_LOSS_WEIGHT=1.5

export MOTION_BALL_LORA_LR=5.0e-7
export MOTION_EVENT_RESIDUAL_SCALE=0.25
export MOTION_CURRICULUM_STAGES="[{name: shared_crossattn_025, start_epoch: 1, end_epoch: 1, overrides: {object_motion_event_residual_scale: 0.25, object_motion_relation_loss_weight: 0.5, object_motion_dense_rank_loss_weight: 0.15}}, {name: shared_crossattn_050, start_epoch: 2, end_epoch: 2, overrides: {object_motion_event_residual_scale: 0.50, object_motion_relation_loss_weight: 0.5}}, {name: shared_crossattn_100, start_epoch: 3, end_epoch: 4, overrides: {object_motion_event_residual_scale: 1.0, object_motion_relation_loss_weight: 0.5}}]"

exec "${repo_dir}/scripts/run_object_motion_adapter_v3.sh" "$@"
