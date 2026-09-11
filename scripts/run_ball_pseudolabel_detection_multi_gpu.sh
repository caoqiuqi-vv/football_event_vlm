#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/new_users/qiuqi/code/dinov3-main"
source_dir="${SOURCE_DIR:-/mnt/data_16t/football/raw_video_720P}"
checkpoint="${BALL_CHECKPOINT:-/mnt/data_7t/qiuqi/code/soccer/onlysoccer_1920_11s.pt}"
output_root="${OUTPUT_ROOT:-/mnt/data_7t/qiuqi/football_ball_pseudolabels/yolo_fulltrack_v2_test18heldout}"
gpu_list="${GPU_LIST:-0,1,2,4,6,7}"
sample_fps="${SAMPLE_FPS:-12}"
max_video_seconds="${MAX_VIDEO_SECONDS:-0}"
tile_image_size="${TILE_IMAGE_SIZE:-640}"
frame_batch_size="${FRAME_BATCH_SIZE:-8}"
tile_batch_size="${TILE_BATCH_SIZE:-32}"
exclude_video_ids="${EXCLUDE_VIDEO_IDS:-${repo_dir}/configs/football/splits/thirdparty18_test_long15_val_no_pn_train/thirdparty18_test_video_ids.txt}"

IFS=',' read -r -a gpus <<< "${gpu_list}"
mkdir -p "${output_root}/logs" "${output_root}/launcher"
printf '%s\n' "${gpu_list}" > "${output_root}/launcher/gpu_list.txt"
printf '%s\n' "${exclude_video_ids}" > "${output_root}/launcher/exclude_video_ids.txt"

for shard_index in "${!gpus[@]}"; do
  gpu="${gpus[$shard_index]}"
  log="${output_root}/logs/detect_gpu${gpu}.log"
  pid_file="${output_root}/launcher/detect_gpu${gpu}.pid"
  session="football_ball_yolo_v2_gpu${gpu}"
  command=(
    /home/new_users/qiuqi/miniconda3/envs/qiuqi_sam3/bin/python \
    "${repo_dir}/scripts/precompute_ball_pseudolabels.py" detect \
    --source "${source_dir}" \
    --checkpoint "${checkpoint}" \
    --output-root "${output_root}" \
    --device "${gpu}" \
    --dynamic-queue \
    --exclude-video-id-file "${exclude_video_ids}" \
    --num-shards "${#gpus[@]}" \
    --shard-index "${shard_index}" \
    --sample-fps "${sample_fps}" \
    --tile-image-size "${tile_image_size}" \
    --frame-batch-size "${frame_batch_size}" \
    --tile-batch-size "${tile_batch_size}" \
    --max-video-seconds "${max_video_seconds}"
  )
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "refusing to replace active tmux session ${session}" >&2
    exit 1
  fi
  printf -v shell_command '%q ' "${command[@]}"
  printf -v quoted_log '%q' "${log}"
  shell_command+=" >> ${quoted_log} 2>&1"
  tmux new-session -d -s "${session}" "${shell_command}"
  pid="$(tmux display-message -p -t "${session}" '#{pane_pid}')"
  printf '%s\n' "${pid}" > "${pid_file}"
  printf '%s\n' "${session}" > "${output_root}/launcher/detect_gpu${gpu}.session"
  echo "started gpu=${gpu} shard=${shard_index}/${#gpus[@]} session=${session} pid=${pid} log=${log}"
done

echo "Detection workers launched. Tracking is intentionally not started until every detection shard is complete."
