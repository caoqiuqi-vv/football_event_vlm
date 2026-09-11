#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/new_users/qiuqi/code/dinov3-main"
launcher="${repo_dir}/scripts/run_object_motion_adapter_v2_antcollapse.sh"
output_dir="${repo_dir}/outputs/football_events/vitl16_d7c_object_motion_v2_antcollapse_dense33_from_object_e2_20260902"
status_file="${output_dir}/gpu_watchdog_status.json"
watchdog_log="${output_dir}/gpu_watchdog.log"
train_session="${TRAIN_SESSION:-football_object_motion_v2_20260902}"

poll_sec="${POLL_SEC:-30}"
idle_max_memory_mib="${IDLE_MAX_MEMORY_MIB:-2048}"
idle_max_utilization="${IDLE_MAX_UTILIZATION:-10}"
min_idle_gpus="${MIN_IDLE_GPUS:-4}"
train_gpu_count="${TRAIN_GPU_COUNT:-3}"
healthy_min_rows="${HEALTHY_MIN_ROWS:-25}"
healthy_confirmations_required="${HEALTHY_CONFIRMATIONS:-3}"
detector_min_metric="${DETECTOR_MIN_METRIC:-0.50}"
goal_top1_min="${GOAL_TOP1_MIN:-0.80}"

mkdir -p "${output_dir}"
cd "${repo_dir}"

idle_count=0
selected_gpus=""
healthy_confirmations=0

write_status() {
  local state="$1"
  local detail="$2"
  local temporary="${status_file}.tmp.$$"
  jq -n \
    --arg state "${state}" \
    --arg detail "${detail}" \
    --arg selected_gpus "${selected_gpus}" \
    --arg updated_at "$(date --iso-8601=seconds)" \
    --argjson idle_gpu_count "${idle_count}" \
    --argjson healthy_confirmations "${healthy_confirmations}" \
    '{
      state: $state,
      detail: $detail,
      idle_gpu_count: $idle_gpu_count,
      selected_gpus: $selected_gpus,
      healthy_confirmations: $healthy_confirmations,
      updated_at: $updated_at
    }' > "${temporary}"
  mv "${temporary}" "${status_file}"
}

discover_idle_gpus() {
  nvidia-smi \
    --query-gpu=index,memory.used,utilization.gpu \
    --format=csv,noheader,nounits |
    awk -F, \
      -v max_memory="${idle_max_memory_mib}" \
      -v max_util="${idle_max_utilization}" \
      '{
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", $3)
        if (($2 + 0) <= max_memory && ($3 + 0) <= max_util) {
          print $1
        }
      }'
}

if [[ ! -x "${launcher}" ]]; then
  write_status "blocked" "launcher is missing or not executable"
  exit 2
fi
if (( train_gpu_count < 1 || min_idle_gpus <= train_gpu_count )); then
  write_status "blocked" "MIN_IDLE_GPUS must be greater than TRAIN_GPU_COUNT"
  exit 2
fi
if tmux has-session -t "${train_session}" 2>/dev/null; then
  write_status "blocked" "training tmux session already exists"
  exit 2
fi
if [[ -e "${output_dir}/train_console.log" ]]; then
  write_status "blocked" "output already contains a training log; refusing overwrite"
  exit 2
fi

write_status "waiting_for_gpu" "need more than three idle GPUs"
while true; do
  mapfile -t idle_gpus < <(discover_idle_gpus)
  idle_count="${#idle_gpus[@]}"
  if (( idle_count >= min_idle_gpus )); then
    selected_gpus="$(IFS=,; echo "${idle_gpus[*]:0:train_gpu_count}")"
    break
  fi
  selected_gpus=""
  write_status "waiting_for_gpu" \
    "idle=${idle_count}, required=${min_idle_gpus}, memory<=${idle_max_memory_mib}MiB, util<=${idle_max_utilization}%"
  sleep "${poll_sec}"
done

write_status "launching" "starting training on GPUs ${selected_gpus}"
tmux new-session -d -s "${train_session}" \
  "cd '${repo_dir}' && GPU_LIST='${selected_gpus}' bash '${launcher}'"
write_status "monitoring" "training launched; waiting for stable health"

while true; do
  if [[ -s "${output_dir}/train_console.log" ]]; then
    /home/new_users/qiuqi/miniconda3/envs/qiuqi_sam3/bin/python \
      -m football_object_motion.monitor "${output_dir}" --once \
      >> "${watchdog_log}" 2>&1 || true

    if [[ -s "${output_dir}/object_motion_learning_status.json" ]]; then
      if jq -e \
        --argjson min_rows "${healthy_min_rows}" \
        --argjson detector_min "${detector_min_metric}" \
        --argjson goal_top1_min "${goal_top1_min}" \
        '
          .healthy == true
          and .rows >= $min_rows
          and (.latest.object_motion_teacher_valid_fraction // 0) >= 0.99
          and (.latest.object_motion_ball_top1_hit // 0) >= $detector_min
          and (.latest.object_motion_goal_top1_hit // 0) >= $goal_top1_min
          and (.latest.object_motion_ball_presence_precision // 0) >= $detector_min
          and (.latest.object_motion_ball_presence_recall // 0) >= $detector_min
          and (.latest.object_motion_goal_presence_precision // 0) >= $detector_min
          and (.latest.object_motion_goal_presence_recall // 0) >= $detector_min
          and (.latest.object_motion_learned_clip_gate_mean // 1) < 0.95
          and (.latest.object_motion_learned_frame_gate_mean // 1) < 0.95
          and (.latest.object_motion_residual_saturation_fraction // 1) <= 0.25
        ' "${output_dir}/object_motion_learning_status.json" >/dev/null
      then
        healthy_confirmations=$((healthy_confirmations + 1))
      else
        healthy_confirmations=0
      fi
      write_status "monitoring" \
        "health confirmations ${healthy_confirmations}/${healthy_confirmations_required}"
      if (( healthy_confirmations >= healthy_confirmations_required )); then
        write_status "healthy" \
          "training is healthy; watchdog stopped and training continues"
        exit 0
      fi
    fi
  fi

  if ! tmux has-session -t "${train_session}" 2>/dev/null; then
    if [[ -e "${output_dir}/training_complete.json" ]]; then
      write_status "completed_before_health_confirmation" \
        "training completed before watchdog health criteria were confirmed"
      exit 1
    fi
    write_status "training_stopped" \
      "training tmux exited before watchdog health criteria were confirmed"
    exit 1
  fi
  sleep "${poll_sec}"
done
