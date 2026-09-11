#!/usr/bin/env bash
# GPU 空闲监视器(T1 证伪后队列):有空闲卡即启动 full166 frame-logits 稠密缓存
# 用最多 2 张空闲卡(DataParallel)。空闲定义:显存 < 3000MiB 且利用率 < 20%。启动后自动退出。
set -u
repo=/home/new_users/qiuqi/code/dinov3-main
state_dir="$repo/outputs/gpu_queue_20260910"
mkdir -p "$state_dir"
cache_done="$state_dir/cache.launched"

while true; do
  mapfile -t free < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | awk -F', ' '$2 < 3000 && $3 < 20 {print $1}')
  n=${#free[@]}

  if [ ! -f "$cache_done" ] && [ "$n" -ge 1 ]; then
    if [ "$n" -ge 2 ]; then gpus="${free[0]},${free[1]}"; else gpus="${free[0]}"; fi
    echo "$(date '+%F %T') launch frame-logits cache on GPU $gpus" >> "$state_dir/watcher.log"
    CUDA_VISIBLE_DEVICES="$gpus" nohup bash "$repo/scripts/run_full166_fromlast_dense_framelogits_20260910.sh" \
      >> "$state_dir/cache.nohup.log" 2>&1 &
    touch "$cache_done"
    echo "$(date '+%F %T') cache launched, watcher exit" >> "$state_dir/watcher.log"
    exit 0
  fi
  sleep 120
done
