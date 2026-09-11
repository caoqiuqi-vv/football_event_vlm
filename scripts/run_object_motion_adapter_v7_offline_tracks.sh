#!/usr/bin/env bash
set -euo pipefail

repo_dir="${FOOTBALL_REPO_DIR:-/home/new_users/qiuqi/code/dinov3-main}"
v4_dir="${repo_dir}/outputs/football_events/vitl16_d7c_object_motion_v4_causal33_720p_aux025_4gpu_from_object_e2_20260903"
offline_index_default="/mnt/data_7t/qiuqi/football_ball_pseudolabels/yolo_fulltrack_v2_test18heldout/offline_index_v1"

export OUTPUT_DIR="${OUTPUT_DIR:-${repo_dir}/outputs/football_events/vitl16_d7c_object_motion_v7_offline_balltrack_inheritedgoal_fpn8_crossattn_4gpu_from_v4e4_20260906_r5}"
export SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${v4_dir}/epoch_4.pt}"
export BASE_CONFIG="${BASE_CONFIG:-${v4_dir}/config.yaml}"
export GPU_LIST="${GPU_LIST:-0,1,4,6}"
export PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-1}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-12}"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-4}"
export TRAIN_LOG_INTERVAL="${TRAIN_LOG_INTERVAL:-10}"
export MOTION_FRAMES_PER_SEGMENT="${MOTION_FRAMES_PER_SEGMENT:-11}"
export MOTION_IMAGE_SIZE="${MOTION_IMAGE_SIZE:-[720,1280]}"
export MOTION_TEACHER_IMAGE_SIZE="${MOTION_TEACHER_IMAGE_SIZE:-[720,1280]}"
export MOTION_TEACHER_BATCH_SIZE="${MOTION_TEACHER_BATCH_SIZE:-8}"

# Tracked YOLO pseudo-labels supervise ball. The V4 goal heatmap readout is
# retained as detached cross-attention evidence; no detector runs in training.
# Person is deliberately absent from auxiliary learning.
export MOTION_OFFLINE_BALL_ENABLED=true
export MOTION_OFFLINE_BALL_INDEX_ROOT="${MOTION_OFFLINE_BALL_INDEX_ROOT:-${offline_index_default}}"
export MOTION_OFFLINE_BALL_MAX_DELTA_SEC="${MOTION_OFFLINE_BALL_MAX_DELTA_SEC:-0.10}"
export MOTION_OFFLINE_BALL_CACHE_VIDEOS="${MOTION_OFFLINE_BALL_CACHE_VIDEOS:-4}"
export MOTION_ONLINE_TEACHER_ENABLED=false
export MOTION_ONLINE_TEACHER_OBJECTS="[goal]"

# The same BallLoRA changes both dense ball features and the main event anchor.
# Detector outputs remain detached at the event-fusion boundary.
export MOTION_SHARED_ANCHOR_BALL_LORA=true
export MOTION_OBJECT_CROSS_ATTENTION=true
export MOTION_OBJECT_CROSS_ATTENTION_MAX_DELTA=0.35
export MOTION_OBJECT_CROSS_ATTENTION_GATE_INIT=0.25
export MOTION_LEGACY_FRAME_RESIDUAL_ENABLED=false

# Stride-8 FPN is initialized as a zero residual over the trained patch-16
# heatmap, preserving V4 behavior on the first forward.
export MOTION_BALL_FEATURE_LAYERS="[11,17,20,23]"
export MOTION_HEATMAP_PATCH_SIZE=8
export MOTION_HEATMAP_UPSAMPLE_FACTOR=2
export MOTION_BALL_FPN_DIM=64

# Object order is [ball, goal, person]. Only offline ball is newly supervised;
# the inherited goal readout remains stable and person is exactly zero.
export MOTION_OBJECT_LOSS_WEIGHTS="[4.0,0.0,0.0]"
export MOTION_NEGATIVE_WEIGHTS="[0.25,0.0,0.0]"
export MOTION_PRESENCE_NEGATIVE_WEIGHTS="[0.0,0.0,0.0]"
export MOTION_ABSENCE_PRESENCE_WEIGHTS="[0.0,0.0,0.0]"
export MOTION_PRESENCE_LOSS_WEIGHT=0.0
export MOTION_BALL_STRONG_LOSS_WEIGHT=1.5
export MOTION_BALL_LORA_LR=5.0e-7
export MOTION_HEAD_WEIGHT_DECAY=0.0

# A gradual residual schedule exposes whether object auxiliary evidence helps
# before the event branch is allowed its full bounded fusion budget.
export MOTION_EVENT_RESIDUAL_SCALE=0.25
export MOTION_CURRICULUM_STAGES='[{name: offline_tracks_025, start_epoch: 1, end_epoch: 1, overrides: {object_motion_event_residual_scale: 0.25, object_motion_relation_loss_weight: 0.5, object_motion_dense_rank_loss_weight: 0.15}}, {name: offline_tracks_050, start_epoch: 2, end_epoch: 2, overrides: {object_motion_event_residual_scale: 0.50, object_motion_relation_loss_weight: 0.5}}, {name: offline_tracks_100, start_epoch: 3, end_epoch: 4, overrides: {object_motion_event_residual_scale: 1.0, object_motion_relation_loss_weight: 0.5}}]'

if [[ ! -s "${MOTION_OFFLINE_BALL_INDEX_ROOT}/manifest.json" ]]; then
  echo "offline ball index missing: ${MOTION_OFFLINE_BALL_INDEX_ROOT}/manifest.json" >&2
  exit 66
fi

exec bash "${repo_dir}/scripts/run_object_motion_adapter_v3.sh" "$@"
