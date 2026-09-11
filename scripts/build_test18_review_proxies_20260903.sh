#!/usr/bin/env bash
set -euo pipefail

repo=/home/new_users/qiuqi/code/dinov3-main
manifest="$repo/outputs/football_event_review/test18_full_annotation_repair_20260901/review_manifest.json"
access="$repo/outputs/football_event_review/test18_multiuser_qc_20260903/access_config.json"
proxy_root="$repo/outputs/football_event_review/test18_multiuser_qc_20260903/media_proxy_540p"
log_root="$repo/outputs/football_event_review/test18_multiuser_qc_20260903/proxy_logs"
mkdir -p "$proxy_root" "$log_root"

transcode_user() {
  local user_index="$1"
  while IFS=$'\t' read -r video_id input_path; do
    local output="$proxy_root/${video_id}.mp4"
    local partial="$proxy_root/${video_id}.part.mp4"
    if [[ -s "$output" ]]; then
      continue
    fi
    rm -f "$partial"
    ffmpeg -hide_banner -nostdin -y -i "$input_path" \
      -map 0:v:0 -map '0:a:0?' -vf 'fps=15,scale=-2:540' \
      -c:v libx264 -preset veryfast -threads 4 -b:v 900k -maxrate 1100k -bufsize 2200k \
      -g 15 -keyint_min 15 -sc_threshold 0 -pix_fmt yuv420p \
      -c:a aac -b:a 64k -ac 2 -ar 32000 -movflags +faststart "$partial" \
      > "$log_root/${video_id}.log" 2>&1
    ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$partial" > /dev/null
    mv "$partial" "$output"
  done < <(
    jq -r --argjson i "$user_index" '.users[$i].video_ids[]' "$access" |
    while read -r video_id; do
      jq -r --arg id "$video_id" '.videos[] | select(.video_id==$id) | [.video_id,.video_path] | @tsv' "$manifest"
    done
  )
}

for user_index in 0 1 2 3; do
  transcode_user "$user_index" &
done
wait
