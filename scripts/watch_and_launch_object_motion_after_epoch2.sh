#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/new_users/qiuqi/code/dinov3-main"
source_run="${repo_dir}/outputs/football_events/vitl16_d7c_object_teacher_online_allclips_no_bad_media_from_fromlast_e8_20260901"
checkpoint="${source_run}/epoch_2.pt"
status_file="${repo_dir}/outputs/football_events/object_motion_after_epoch2_watchdog.json"
source_tmux="football_object_teacher_online_20260901"
target_tmux="football_object_motion_dense33_20260902"
monitor_tmux="football_object_motion_monitor_20260902"
target_output="${repo_dir}/outputs/football_events/vitl16_d7c_object_motion_dense33_person_from_object_e2_20260902"

write_status() {
  local state="$1"
  local detail="$2"
  local now
  now="$(date --iso-8601=seconds)"
  printf '{"state":"%s","detail":"%s","updated_at":"%s"}\n' \
    "${state}" "${detail}" "${now}" > "${status_file}"
}

mkdir -p "$(dirname "${status_file}")"
write_status "waiting" "epoch_2 checkpoint"
while [[ ! -s "${checkpoint}" ]]; do
  sleep 20
done
write_status "checkpoint_writing" "waiting for epoch_2 checkpoint size to stabilize"
previous_size=0
stable_checks=0
while (( stable_checks < 2 )); do
  current_size="$(stat -c %s "${checkpoint}")"
  if [[ "${current_size}" == "${previous_size}" && "${current_size}" -gt 100000000 ]]; then
    stable_checks=$((stable_checks + 1))
  else
    stable_checks=0
  fi
  previous_size="${current_size}"
  sleep 10
done
write_status "checkpoint_ready" "stopping source after stable epoch2"

if tmux has-session -t "${source_tmux}" 2>/dev/null; then
  tmux send-keys -t "${source_tmux}" C-c
fi
for _ in $(seq 1 12); do
  if ! tmux has-session -t "${source_tmux}" 2>/dev/null; then
    break
  fi
  sleep 10
done

write_status "smoke_running" "real Teacher + full forward/backward on GPU1"
smoke_log="${target_output}/smoke_console.log"
mkdir -p "${target_output}"
if ! SMOKE_GPU=1 bash "${repo_dir}/scripts/run_object_motion_smoke_after_epoch2.sh" \
  > "${smoke_log}" 2>&1; then
  write_status "smoke_failed" "see ${smoke_log}"
  exit 1
fi
per_gpu_batch_size=1
grad_accum_steps=8
batch2_log="${target_output}/smoke_batch2_console.log"
batch2_json="${target_output}/smoke_batch2_pass.json"
write_status "batch2_probe" "trying complete real batch2 forward/backward"
if SMOKE_GPU=1 SMOKE_BATCH_SIZE=2 SMOKE_OUTPUT="${batch2_json}" \
  bash "${repo_dir}/scripts/run_object_motion_smoke_after_epoch2.sh" \
  > "${batch2_log}" 2>&1; then
  per_gpu_batch_size=2
  grad_accum_steps=4
  write_status "batch2_passed" "launching batch2 x three GPUs"
else
  write_status "batch2_rejected" "using verified batch1; see ${batch2_log}"
fi

if tmux has-session -t "${target_tmux}" 2>/dev/null; then
  write_status "already_running" "target tmux exists"
  exit 0
fi
tmux new-session -d -s "${target_tmux}" \
  "cd '${repo_dir}' && PER_GPU_BATCH_SIZE=${per_gpu_batch_size} GRAD_ACCUM_STEPS=${grad_accum_steps} bash scripts/run_object_motion_adapter_after_epoch2.sh"
tmux new-session -d -s "${monitor_tmux}" \
  "cd '${repo_dir}' && /home/new_users/qiuqi/miniconda3/envs/qiuqi_sam3/bin/python -m football_object_motion.monitor '${target_output}' --interval-sec 30"
write_status "launched" "${target_tmux}"
