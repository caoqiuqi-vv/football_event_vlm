#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/new_users/qiuqi/code/dinov3-main"
output_root="${OUTPUT_ROOT:-/mnt/data_7t/qiuqi/football_ball_pseudolabels/yolo_fulltrack_v2_test18heldout}"
worker_count="${TRACK_WORKERS:-12}"

mkdir -p "${output_root}/tracking_logs" "${output_root}/tracking_launcher"
printf '%s\n' "${worker_count}" > "${output_root}/tracking_launcher/worker_count.txt"

for ((shard_index = 0; shard_index < worker_count; shard_index++)); do
  printf -v shard_name '%02d' "${shard_index}"
  session="football_ball_track_v2_s${shard_name}"
  log="${output_root}/tracking_logs/track_shard${shard_name}.log"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "refusing to replace active tmux session ${session}" >&2
    exit 1
  fi
  command=(
    /home/new_users/qiuqi/miniconda3/envs/qiuqi_sam3/bin/python
    "${repo_dir}/scripts/precompute_ball_pseudolabels.py" track
    --output-root "${output_root}"
    --num-shards "${worker_count}"
    --shard-index "${shard_index}"
    --skip-existing
  )
  printf -v shell_command '%q ' "${command[@]}"
  printf -v quoted_log '%q' "${log}"
  shell_command+=" >> ${quoted_log} 2>&1"
  tmux new-session -d -s "${session}" "${shell_command}"
  pid="$(tmux display-message -p -t "${session}" '#{pane_pid}')"
  printf '%s\n' "${pid}" > "${output_root}/tracking_launcher/track_shard${shard_name}.pid"
  printf '%s\n' "${session}" > "${output_root}/tracking_launcher/track_shard${shard_name}.session"
  echo "started shard=${shard_index}/${worker_count} session=${session} pid=${pid} log=${log}"
done

echo "det_and_track football tracking workers launched"
