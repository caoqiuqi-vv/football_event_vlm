#!/usr/bin/env bash
set -euo pipefail

# Causal v4: 33 motion frames, higher-resolution teacher decode and
# class-aware evidence floors. Set-piece keeps a larger floor because the ball
# can be occluded/off-screen while the event remains visually identifiable.
repo_dir="/home/new_users/qiuqi/code/dinov3-main"
export OUTPUT_DIR="${OUTPUT_DIR:-${repo_dir}/outputs/football_events/vitl16_d7c_object_motion_v4_causal33_720p_aux025_4gpu_from_object_e2_20260903}"
export GPU_LIST="${GPU_LIST:-0,1,4,6}"
export MOTION_FRAMES_PER_SEGMENT="${MOTION_FRAMES_PER_SEGMENT:-11}"
export MOTION_IMAGE_SIZE="${MOTION_IMAGE_SIZE:-[720,1280]}"
export MOTION_TEACHER_IMAGE_SIZE="${MOTION_TEACHER_IMAGE_SIZE:-[720,1280]}"
export MOTION_CLASS_EVIDENCE_FLOORS="${MOTION_CLASS_EVIDENCE_FLOORS:-[0.05,0.05,0.35]}"
export MOTION_TEACHER_BATCH_SIZE="${MOTION_TEACHER_BATCH_SIZE:-8}"
export PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-1}"
# 4 GPUs x batch 1 x accumulation 12 preserves the prior effective batch 48.
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-12}"
# Keep the auxiliary detector objective below the frozen event-anchor loss.
export OBJECT_AUX_LOSS_WEIGHT="${OBJECT_AUX_LOSS_WEIGHT:-0.25}"

exec "${repo_dir}/scripts/run_object_motion_adapter_v3.sh" "$@"
