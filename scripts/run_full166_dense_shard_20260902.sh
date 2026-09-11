#!/usr/bin/env bash
set -euo pipefail

gpu_id="$1"
video_ids="$2"
repo_dir=/home/new_users/qiuqi/code/dinov3-main
run_name=full166_fromlast_e8_best_dense_s5_20260902
run_dir="$repo_dir/outputs/football_full_review/$run_name"

cd "$repo_dir"
export CUDA_VISIBLE_DEVICES="$gpu_id"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec /home/new_users/qiuqi/miniconda3/bin/python scripts/evaluate_football_model.py \
  --checkpoint outputs/football_events/vitl16_d7c_fullimage_latest_split_dino_pretrained_lora_ddp_720p_fromlast_e8_20260829/best.pt \
  --mode dense --video-id-file "$video_ids" \
  --gt-dir /mnt/data_16t/football/football_events_human_repair \
  --video-root xbotgo_0608=/mnt/data_16t/football/raw_video_720P \
  --video-root xbotgo_hq=/mnt/data_16t/football/raw_video_hq_720P \
  --output-root outputs/football_full_review --run-name "$run_name" \
  --clip-sec 10 --stride-sec 5 --image-size 720,1280 \
  --batch-size 8 --num-workers 2 --device cuda:0 --gpu-ids 0 \
  --thresholds checkpoint --match-tolerance-sec 3 --score-source clip \
  --prediction-postprocess window_overlap --gt-merge-gap-sec 0 \
  --spatial-crop-mode none \
  >> "$run_dir/shards/gpu${gpu_id}.log" 2>&1
