#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-outputs/football_events/vitl16_weekend_event_anchor_512x896_neighborhood_st}"
BASELINE="${BASELINE:-/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt}"
GPU_LIST="${GPU_LIST:-0,1,2,6,7}"
MAX_GPU_MEMORY_MB="${MAX_GPU_MEMORY_MB:-2000}"
MAX_GPU_UTIL="${MAX_GPU_UTIL:-10}"
LOG="${OUTPUT_DIR}/post_training_pipeline.log"
TRAIN_PID="${TRAIN_PID:-}"
MAX_EPOCH="${MAX_EPOCH:-4}"
RUN_TAG="${RUN_TAG:-event_anchor}"
VIDEO_IDS="${VIDEO_IDS:-2027564888428580866,2027572095412195330,2027572406738604033,2042125172076392450,2042520971893485569,2042525152494694401}"
GT_DIR="${GT_DIR:-/home/new_users/qiuqi/code/football_events_human_repair}"
VIDEO_ROOT="${VIDEO_ROOT:-/mnt/data_16t/football/raw_video_720P}"
BASELINE_PROTOCOL="${BASELINE_PROTOCOL:-outputs/football_eval_runs/vitl16_lora_r5_f32_16f_e1_frame_det_last_hr_6videos_window_overlap_checkpoint_thr/protocol_comparison_checkpoint_thr.json}"

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
for epoch in $(seq 1 "${MAX_EPOCH}"); do
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

gpu_count="$(python -c 'import sys; print(len(sys.argv[1].split(",")))' "${GPU_LIST}")"
local_gpu_ids="$(python -c 'import sys; n=int(sys.argv[1]); print(",".join(map(str, range(n))))' "${gpu_count}")"
run_name="$(basename "${selected}" .pt)_${RUN_TAG}_6videos_dense_checkpoint_thr"
CUDA_VISIBLE_DEVICES="${GPU_LIST}" PYTHONUNBUFFERED=1 python scripts/evaluate_football_model.py \
  --checkpoint "${selected}" \
  --mode dense \
  --video-ids "${VIDEO_IDS}" \
  --gt-dir "${GT_DIR}" \
  --video-root "xbotgo_0608=${VIDEO_ROOT}" \
  --run-name "${run_name}" \
  --clip-sec 10 \
  --stride-sec 5 \
  --batch-size "${gpu_count}" \
  --num-workers "$(( gpu_count * 2 ))" \
  --device cuda:0 \
  --gpu-ids "${local_gpu_ids}" \
  --thresholds checkpoint \
  --prediction-postprocess point_nms \
  --nms-radius-sec 5 \
  --match-tolerance-sec 5 \
  --save-frame-event-logits \
  --frame-event-topk 8 >> "${LOG}" 2>&1

run_dir="outputs/football_eval_runs/${run_name}"
python scripts/recompute_football_eval_protocols.py \
  --run-dir "${run_dir}" \
  --video-ids "${VIDEO_IDS}" \
  --output-prefix protocol_comparison_checkpoint_thr \
  --nms-radius-sec 5 \
  --match-tolerance-sec 5 >> "${LOG}" 2>&1
python scripts/compare_football_eval_protocols.py \
  --baseline "${BASELINE_PROTOCOL}" \
  --candidates "${run_dir}/protocol_comparison_checkpoint_thr.json" \
  --recall-tolerance-pp 1 \
  --output "${run_dir}/vs_e1_recall_guard_1pp.json" >> "${LOG}" 2>&1
echo "pipeline_completed_at=$(date --iso-8601=seconds) run_dir=${run_dir}" >> "${LOG}"
