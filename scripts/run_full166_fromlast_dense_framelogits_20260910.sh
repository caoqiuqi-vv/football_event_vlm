#!/usr/bin/env bash
# V2.1 前置:full166 稠密缓存补落 frame_event_logits(verifier V1 的关键特征来源)
# 与 20260902 的 full166 run 唯一差异:加 --save-frame-event-logits,新 run 目录。
set -euo pipefail

repo_dir=/home/new_users/qiuqi/code/dinov3-main
run_name=full166_fromlast_e8_best_dense_s5_framelogits_20260910
run_dir="$repo_dir/outputs/football_full_review/$run_name"
gpu_list="${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to free gpu ids}"
n_gpus=$(awk -F',' '{print NF}' <<< "$gpu_list")
gpu_ids=$(seq -s, 0 $((n_gpus-1)))
batch_size=$((8 * n_gpus))

mkdir -p "$run_dir"
cp "$repo_dir/outputs/football_full_review/full166_fromlast_e8_best_dense_s5_20260902/all_video_ids.txt" "$run_dir/all_video_ids.txt"

cd "$repo_dir"
export CUDA_VISIBLE_DEVICES="$gpu_list"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec /home/new_users/qiuqi/miniconda3/bin/python scripts/evaluate_football_model.py \
  --checkpoint outputs/football_events/vitl16_d7c_fullimage_latest_split_dino_pretrained_lora_ddp_720p_fromlast_e8_20260829/best.pt \
  --mode dense \
  --video-id-file "$run_dir/all_video_ids.txt" \
  --gt-dir /mnt/data_16t/football/football_events_human_repair \
  --video-root xbotgo_0608=/mnt/data_16t/football/raw_video_720P \
  --video-root xbotgo_hq=/mnt/data_16t/football/raw_video_hq_720P \
  --output-root outputs/football_full_review \
  --run-name "$run_name" \
  --clip-sec 10 \
  --stride-sec 5 \
  --image-size 720,1280 \
  --batch-size "${batch_size}" \
  --num-workers 2 \
  --device cuda:0 \
  --gpu-ids "${gpu_ids}" \
  --thresholds checkpoint \
  --match-tolerance-sec 3 \
  --score-source clip \
  --prediction-postprocess window_overlap \
  --gt-merge-gap-sec 0 \
  --spatial-crop-mode none \
  --save-frame-event-logits \
  >> "$run_dir/dense_console.log" 2>&1
