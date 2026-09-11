#!/usr/bin/env bash
set -euo pipefail

# Ball-focused successor to v4. The weighted object reductions are normalized,
# so prioritizing ball does not multiply the whole auxiliary loss scale.
repo_dir="/home/new_users/qiuqi/code/dinov3-main"
export OUTPUT_DIR="${OUTPUT_DIR:-${repo_dir}/outputs/football_events/vitl16_d7c_object_motion_v5_ballfocus_causal33_720p_aux025_4gpu_from_object_e2_20260904}"
export GPU_LIST="${GPU_LIST:-0,1,4,6}"
export MOTION_FRAMES_PER_SEGMENT="${MOTION_FRAMES_PER_SEGMENT:-11}"
export MOTION_IMAGE_SIZE="${MOTION_IMAGE_SIZE:-[720,1280]}"
export MOTION_TEACHER_IMAGE_SIZE="${MOTION_TEACHER_IMAGE_SIZE:-[720,1280]}"
export MOTION_CLASS_EVIDENCE_FLOORS="${MOTION_CLASS_EVIDENCE_FLOORS:-[0.05,0.05,0.35]}"
export MOTION_TEACHER_BATCH_SIZE="${MOTION_TEACHER_BATCH_SIZE:-8}"
export PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-1}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-12}"
export OBJECT_AUX_LOSS_WEIGHT="${OBJECT_AUX_LOSS_WEIGHT:-0.25}"

# Object order is [ball, goal, person]. Person remains weakly supervised for
# relation geometry, but can no longer dominate the shared detector objective.
export MOTION_OBJECT_LOSS_WEIGHTS="${MOTION_OBJECT_LOSS_WEIGHTS:-[4.0,1.0,0.05]}"
export MOTION_NEGATIVE_WEIGHTS="${MOTION_NEGATIVE_WEIGHTS:-[0.50,0.25,0.01]}"
export MOTION_PRESENCE_NEGATIVE_WEIGHTS="${MOTION_PRESENCE_NEGATIVE_WEIGHTS:-[1.0,1.0,0.05]}"
export MOTION_BALL_STRONG_LOSS_WEIGHT="${MOTION_BALL_STRONG_LOSS_WEIGHT:-1.5}"

exec "${repo_dir}/scripts/run_object_motion_adapter_v3.sh" "$@"
