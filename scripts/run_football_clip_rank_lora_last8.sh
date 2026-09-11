#!/usr/bin/env bash
set -euo pipefail

export CONFIG="${CONFIG:-configs/football/dinov3_vitl16_lora_r5_f32_16f_hr_single_frame_det_clip_rank_lora_last8.yaml}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/football_events/vitl16_e1_clip_rank_clean_lora_last8}"
export TARGET_HEAD_LR="${TARGET_HEAD_LR:-0.0003}"
export TARGET_BACKBONE_LR="${TARGET_BACKBONE_LR:-0.00003}"
export TARGET_EFFECTIVE_BATCH_SIZE="${TARGET_EFFECTIVE_BATCH_SIZE:-80}"
export PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-4}"

exec bash scripts/run_football_clip_rank_clean.sh "$@"
