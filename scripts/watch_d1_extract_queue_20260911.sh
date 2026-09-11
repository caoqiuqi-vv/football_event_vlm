#!/usr/bin/env bash
# D1 特征提取排队监视器:有空闲卡就领走下一个 20 视频的分片跑提取,直到 162 个视频全部派出。
# 空闲定义:显存 < 3000MiB 且利用率 < 20%。extract 脚本按视频粒度断点续跑,安全。
set -u
repo=/home/new_users/qiuqi/code/dinov3-main
state_dir="$repo/outputs/gpu_queue_20260910"
d1_dir="$repo/outputs/football_d1_examiner"
ids_file="$d1_dir/d1_video_ids.txt"
assigned_file="$state_dir/d1_assigned_videos.txt"
mkdir -p "$state_dir" "$state_dir/d1_shards"
touch "$assigned_file"
chunk=20

total=$(wc -l < "$ids_file")

while true; do
  assigned_n=$(sort -u "$assigned_file" | grep -c . || true)
  if [ "$assigned_n" -ge "$total" ]; then
    echo "$(date '+%F %T') all $total videos assigned, watcher exit" >> "$state_dir/watcher.log"
    exit 0
  fi

  mapfile -t free < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | awk -F', ' '$2 < 3000 && $3 < 20 {print $1}')

  for gpu in "${free[@]}"; do
    # 该卡上没有我们存活的提取进程才派发
    busy=0
    for pf in "$state_dir"/d1_shards/gpu${gpu}_*.pid; do
      [ -e "$pf" ] || continue
      if kill -0 "$(cat "$pf")" 2>/dev/null; then busy=1; fi
    done
    [ "$busy" = 1 ] && continue

    # 取下一批未分配视频
    mapfile -t batch < <(comm -23 <(sort "$ids_file") <(sort -u "$assigned_file") | head -n "$chunk")
    [ "${#batch[@]}" = 0 ] && break
    ids_csv=$(IFS=,; echo "${batch[*]}")
    tag=$(date +%H%M%S)_gpu${gpu}
    echo "$(date '+%F %T') launch D1 extract shard ($assigned_n assigned) on GPU $gpu: ${#batch[@]} videos" >> "$state_dir/watcher.log"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 nohup /home/new_users/qiuqi/miniconda3/bin/python \
      "$repo/scripts/extract_d1_candidate_features.py" \
      --gpu-ids 0 --batch-size 16 --video-ids "$ids_csv" \
      >> "$state_dir/d1_shards/shard_${tag}.log" 2>&1 &
    echo $! > "$state_dir/d1_shards/gpu${gpu}_${tag}.pid"
    printf '%s\n' "${batch[@]}" >> "$assigned_file"
    sleep 240
  done
  sleep 120
done
