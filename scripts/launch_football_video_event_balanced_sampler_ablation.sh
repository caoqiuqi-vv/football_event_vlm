#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/new_users/qiuqi/miniconda3/envs/qiuqi_sam3/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/home/new_users/qiuqi/miniconda3/envs/qiuqi_sam3/bin/torchrun}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
CONFIG="${CONFIG:-configs/football/dinov3_vitl16_independent_preprojection_evidence_ddp_lora_last8_r8_save_setpiece_supervision_v3.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/football_events/vitl16_independent_preprojection_evidence_ddp_lora_last8_r8_v3_video_event_balanced_sampler_ablation}"

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
WORLD_SIZE="${#GPU_ARRAY[@]}"
if [[ "$WORLD_SIZE" -lt 1 ]]; then
  echo "GPU_IDS must contain at least one GPU id" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTHONUNBUFFERED=1

exec "$TORCHRUN_BIN" \
  --standalone \
  --nproc-per-node="$WORLD_SIZE" \
  train_football_events.py \
  --config "$CONFIG" \
  "output_dir=$OUTPUT_DIR" \
  "gpu_ids=[$GPU_IDS]" \
  "train.sampler.mode=video_event_balanced" \
  "train.sampler.positive_fraction=0.25" \
  "train.sampler.unique_videos_per_effective_batch=32" \
  "experiment_notes.sampler_ablation=Hierarchical video-first sampling; each effective batch has 32 unique long videos and a fixed 25 percent event fraction."
