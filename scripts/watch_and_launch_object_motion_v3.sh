#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/new_users/qiuqi/code/dinov3-main"
launcher="${repo_dir}/scripts/run_object_motion_adapter_v3.sh"
output_dir="${repo_dir}/outputs/football_events/vitl16_d7c_object_motion_v3_fullwindow51_from_object_e2_20260903"
status_file="${output_dir}/gpu_watchdog_status.json"
watchdog_log="${output_dir}/gpu_watchdog.log"
train_session="${TRAIN_SESSION:-football_object_motion_v3_20260903}"

poll_sec="${POLL_SEC:-60}"
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
idle_confirmations=0
anchor_ceiling_json="${ANCHOR_CEILING_JSON:-}"
smoke_artifact="${SMOKE_ARTIFACT:-${output_dir}/object_motion_v3_two_step_smoke.json}"
smoke_max_age_sec="${SMOKE_MAX_AGE_SEC:-86400}"
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
  local compute_uuids
  compute_uuids=",$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | tr "\n" ","),"
  while IFS=, read -r index uuid memory utilization; do
    index="${index//[[:space:]]/}"; uuid="${uuid//[[:space:]]/}"
    memory="${memory//[[:space:]]/}"; utilization="${utilization//[[:space:]]/}"
    if (( memory <= idle_max_memory_mib && utilization <= idle_max_utilization )) && [[ "${compute_uuids}" != *",${uuid},"* ]]; then
      echo "${index}"
    fi
  done < <(nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader,nounits)
}

if [[ -z "${anchor_ceiling_json}" || ! -s "${anchor_ceiling_json}" ]]; then
  write_status "blocked_preflight" "ANCHOR_CEILING_JSON is missing; formal training is forbidden"
  exit 78
fi
if ! jq -e '(.candidate_ceiling.shot // 0) >= 0.90 and (.candidate_ceiling.save // 0) >= 0.85 and (.candidate_ceiling.set_piece // 0) >= 0.85' "${anchor_ceiling_json}" >/dev/null; then
  write_status "blocked_preflight" "anchor candidate ceiling below shot=.90/save=.85/set_piece=.85"
  exit 78
fi
if [[ ! -x "${launcher}" ]]; then
  write_status "blocked" "launcher is missing or not executable"
  exit 2
fi
if [[ ! -s "${smoke_artifact}" ]]; then
  write_status "waiting_for_safe_smoke" "missing two-step smoke artifact: ${smoke_artifact}"
  exit 76
fi
launcher_hash="$(sha256sum "${launcher}" | awk '{print $1}')"
now_epoch="$(date +%s)"
if ! jq -e --arg hash "${launcher_hash}" --argjson now "${now_epoch}" --argjson max_age "${smoke_max_age_sec}" '.state == "passed" and .schema == "object_motion_v3_two_step_smoke_v1" and .steps_completed == 2 and .config_sha256 == $hash and (($now - .created_at_unix) >= 0) and (($now - .created_at_unix) <= $max_age) and ((.steps[1].adapter_grad_norms.relation // 0) > 0.00001)' "${smoke_artifact}" >/dev/null; then
  write_status "waiting_for_safe_smoke" "two-step smoke artifact is stale, mismatched, or invalid"
  exit 76
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
    candidate="$(IFS=,; echo "${idle_gpus[*]:0:train_gpu_count}")"
    if [[ "${candidate}" == "${selected_gpus}" ]]; then
      idle_confirmations=$((idle_confirmations + 1))
    else
      selected_gpus="${candidate}"
      idle_confirmations=1
    fi
    write_status "waiting_for_gpu" "idle confirmation ${idle_confirmations}/5 for GPUs ${selected_gpus}"
    if (( idle_confirmations >= 5 )); then break; fi
    sleep "${poll_sec}"
    continue
  fi
  idle_confirmations=0
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
