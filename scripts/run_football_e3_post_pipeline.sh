#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-outputs/football_events/vitl16_weekend_class_query_frame_det}"
BASELINE="${BASELINE:-/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt}"
GPU_LIST="${GPU_LIST:-0,1,2}"
MAX_GPU_MEMORY_MB="${MAX_GPU_MEMORY_MB:-2000}"
MAX_GPU_UTIL="${MAX_GPU_UTIL:-10}"
LOG="${OUTPUT_DIR}/post_training_pipeline.log"
TRAIN_PID="${TRAIN_PID:-}"

mkdir -p "${OUTPUT_DIR}"

wait_for_file() {
  local path="$1"
  while [[ ! -s "${path}" ]]; do
    if [[ -n "${TRAIN_PID}" ]] && ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
      echo "train_exited_before_file=${path} at=$(date --iso-8601=seconds)" >> "${LOG}"
      return 1
    fi
    sleep 30
  done
}

wait_for_gpus() {
  local required=()
  IFS=',' read -r -a required <<< "${GPU_LIST}"
  for _ in $(seq 1 240); do
    local free=1
    local index used util
    while IFS=',' read -r index used util; do
      index="${index//[[:space:]]/}"
      used="${used//[[:space:]]/}"
      util="${util//[[:space:]]/}"
      for required_index in "${required[@]}"; do
        if [[ "${index}" == "${required_index}" ]] && \
           (( used >= MAX_GPU_MEMORY_MB || util >= MAX_GPU_UTIL )); then
          free=0
        fi
      done
    done < <(
      nvidia-smi \
        --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits
    )
    if (( free == 1 )); then
      return 0
    fi
    sleep 30
  done
  return 1
}

echo "pipeline_started_at=$(date --iso-8601=seconds)" >> "${LOG}"
checkpoints=()
for epoch in 1 2 3; do
  checkpoint="${OUTPUT_DIR}/epoch_${epoch}.pt"
  if ! wait_for_file "${checkpoint}"; then
    exit 2
  fi
  checkpoints+=("${checkpoint}")
done

report="${OUTPUT_DIR}/all_epochs_vs_e1"
python scripts/compare_football_checkpoint_metrics.py \
  --baseline "${BASELINE}" \
  --candidates "${checkpoints[@]}" \
  --labels shot,save,set_piece \
  --recall-tolerance-pp 1 \
  --output "${report}.json" \
  > "${report}.md" 2> "${report}.err"

selected="$(
  jq -r '
    ([.candidates[] | select(.recall_guard_pass and .precision_improved)]
      | sort_by([.precision_delta_mean, .mAP_delta])
      | last
      | .path) //
    ([.candidates[]] | sort_by([.mAP, .recall_delta_mean]) | last | .path)
  ' "${report}.json"
)"
if [[ -z "${selected}" || "${selected}" == "null" ]]; then
  echo "no_selected_checkpoint_at=$(date --iso-8601=seconds)" >> "${LOG}"
  exit 2
fi
echo "selected_checkpoint=${selected} at=$(date --iso-8601=seconds)" >> "${LOG}"

if ! wait_for_gpus; then
  echo "gpu_wait_timeout_before_eval_at=$(date --iso-8601=seconds)" >> "${LOG}"
  exit 3
fi

run_name="$(basename "${selected}" .pt)_e3_6videos_point_nms"
CHECKPOINT="${selected}" RUN_NAME="${run_name}" \
  bash scripts/run_football_weekend_a800.sh eval "${GPU_LIST%%,*}" "${selected}" \
  >> "${LOG}" 2>&1
echo "e3_eval_completed_at=$(date --iso-8601=seconds) run_name=${run_name}" >> "${LOG}"

if ! wait_for_gpus; then
  echo "gpu_wait_timeout_before_mining_at=$(date --iso-8601=seconds)" >> "${LOG}"
  exit 4
fi

INIT_CHECKPOINT="${BASELINE}" MINING_PER_GPU_BATCH_SIZE=4 \
  bash scripts/run_football_weekend_a800.sh mine_hardneg_target "${GPU_LIST}" \
  >> "${LOG}" 2>&1
echo "target_hardneg_mining_completed_at=$(date --iso-8601=seconds)" >> "${LOG}"
