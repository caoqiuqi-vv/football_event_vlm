#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MODE="${1:-help}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/football_events/vitl16_e1_precision_focal_neg8_from_e1_5090_fast_b8_g3467}"
CHECKPOINT="${CHECKPOINT:-}"
SELECT_CHECKPOINT="${SELECT_CHECKPOINT:-best}"
GPU_LIST="${GPU_LIST:-3,4,6,7}"
VIDEO_IDS="${VIDEO_IDS:-2027564888428580866,2027572095412195330,2027572406738604033,2042125172076392450,2042520971893485569,2042525152494694401}"
GT_DIR="${GT_DIR:-/home/new_users/qiuqi/code/football_events_human_repair}"
VIDEO_ROOT="${VIDEO_ROOT:-/mnt/data_16t/football/raw_video_720P}"
BASELINE_PROTOCOL="${BASELINE_PROTOCOL:-outputs/football_eval_runs/vitl16_lora_r5_f32_16f_e1_frame_det_last_hr_6videos_window_overlap_checkpoint_thr/protocol_comparison_checkpoint_thr.json}"
MATCH_TOLERANCE_SEC="${MATCH_TOLERANCE_SEC:-5}"
NMS_RADIUS_SEC="${NMS_RADIUS_SEC:-5}"
BATCH_PER_GPU="${BATCH_PER_GPU:-4}"
NUM_WORKERS_PER_GPU="${NUM_WORKERS_PER_GPU:-2}"
WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC:-86400}"
WAIT_INTERVAL_SEC="${WAIT_INTERVAL_SEC:-60}"
FORCE_EVAL="${FORCE_EVAL:-0}"
RUN_NAME="${RUN_NAME:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

local_gpu_ids() {
  "${PYTHON_BIN}" - "$1" <<'PY2'
import sys
n = len([item for item in sys.argv[1].split(',') if item.strip()])
print(','.join(str(i) for i in range(n)))
PY2
}

gpu_count() {
  "${PYTHON_BIN}" - "$1" <<'PY2'
import sys
print(len([item for item in sys.argv[1].split(',') if item.strip()]))
PY2
}

resolve_checkpoint() {
  if [[ -n "${CHECKPOINT}" ]]; then
    echo "${CHECKPOINT}"
    return 0
  fi
  case "${SELECT_CHECKPOINT}" in
    best)
      echo "${OUTPUT_DIR}/best.pt"
      ;;
    last)
      echo "${OUTPUT_DIR}/last.pt"
      ;;
    epoch*)
      local epoch="${SELECT_CHECKPOINT#epoch_}"
      epoch="${epoch#epoch}"
      echo "${OUTPUT_DIR}/epoch_${epoch}.pt"
      ;;
    *)
      echo "${SELECT_CHECKPOINT}"
      ;;
  esac
}

wait_for_checkpoint() {
  local checkpoint="$1"
  local waited=0
  while [[ ! -s "${checkpoint}" ]]; do
    if (( waited >= WAIT_TIMEOUT_SEC )); then
      echo "Timed out waiting for checkpoint: ${checkpoint}" >&2
      return 2
    fi
    echo "waiting_for_checkpoint=${checkpoint} waited_sec=${waited}" >&2
    sleep "${WAIT_INTERVAL_SEC}"
    waited=$((waited + WAIT_INTERVAL_SEC))
  done
}

run_eval() {
  local checkpoint="$1"
  if [[ ! -s "${checkpoint}" ]]; then
    echo "Missing checkpoint: ${checkpoint}" >&2
    exit 2
  fi
  local count local_ids batch_size num_workers run_name run_dir force_args=()
  count="$(gpu_count "${GPU_LIST}")"
  local_ids="$(local_gpu_ids "${GPU_LIST}")"
  batch_size=$((count * BATCH_PER_GPU))
  num_workers=$((count * NUM_WORKERS_PER_GPU))
  if [[ -z "${RUN_NAME}" ]]; then
    run_name="$(basename "${OUTPUT_DIR}")_$(basename "${checkpoint}" .pt)_6videos_dense_checkpoint_thr"
  else
    run_name="${RUN_NAME}"
  fi
  run_dir="outputs/football_eval_runs/${run_name}"
  if [[ "${FORCE_EVAL}" == "1" ]]; then
    force_args=(--force)
  fi

  mkdir -p "${run_dir}"
  {
    echo "===== single precision focal post eval ====="
    echo "started_at=$(date --iso-8601=seconds)"
    echo "checkpoint=${checkpoint}"
    echo "run_dir=${run_dir}"
    echo "gpu_list=${GPU_LIST} local_gpu_ids=${local_ids}"
    echo "batch_size=${batch_size} num_workers=${num_workers}"
  } | tee -a "${run_dir}/post_eval.log"

  CUDA_VISIBLE_DEVICES="${GPU_LIST}" PYTHONUNBUFFERED=1 "${PYTHON_BIN}" scripts/evaluate_football_model.py \
    --checkpoint "${checkpoint}" \
    --mode dense \
    --video-ids "${VIDEO_IDS}" \
    --gt-dir "${GT_DIR}" \
    --video-root "xbotgo_0608=${VIDEO_ROOT}" \
    --run-name "${run_name}" \
    --clip-sec 10 \
    --stride-sec 5 \
    --batch-size "${batch_size}" \
    --num-workers "${num_workers}" \
    --device cuda:0 \
    --gpu-ids "${local_ids}" \
    --thresholds checkpoint \
    --prediction-postprocess point_nms \
    --nms-radius-sec "${NMS_RADIUS_SEC}" \
    --match-tolerance-sec "${MATCH_TOLERANCE_SEC}" \
    --save-frame-event-logits \
    --frame-event-topk 8 \
    "${force_args[@]}" 2>&1 | tee -a "${run_dir}/post_eval.log"

  "${PYTHON_BIN}" scripts/recompute_football_eval_protocols.py \
    --run-dir "${run_dir}" \
    --video-ids "${VIDEO_IDS}" \
    --output-prefix protocol_comparison_checkpoint_thr \
    --nms-radius-sec "${NMS_RADIUS_SEC}" \
    --match-tolerance-sec "${MATCH_TOLERANCE_SEC}" 2>&1 | tee -a "${run_dir}/post_eval.log"

  "${PYTHON_BIN}" scripts/compare_football_eval_protocols.py \
    --baseline "${BASELINE_PROTOCOL}" \
    --candidates "${run_dir}/protocol_comparison_checkpoint_thr.json" \
    --recall-tolerance-pp 1 \
    --output "${run_dir}/vs_e1_recall_guard_1pp.json" 2>&1 | tee -a "${run_dir}/post_eval.log"

  "${PYTHON_BIN}" scripts/analyze_frame_detection_precision.py \
    --run-dir "${run_dir}" \
    --match-tolerance-sec "${MATCH_TOLERANCE_SEC}" \
    --output-dir "${run_dir}/frame_detection_precision" 2>&1 | tee -a "${run_dir}/post_eval.log"

  "${PYTHON_BIN}" scripts/analyze_football_window_fp_distance.py \
    --run-dir "${run_dir}" \
    --protocol-json "${run_dir}/protocol_comparison_checkpoint_thr.json" \
    --output "${run_dir}/window_overlap_fp_distance.json" 2>&1 | tee -a "${run_dir}/post_eval.log"

  "${PYTHON_BIN}" scripts/summarize_football_long_protocol_runs.py \
    --run "precision_focal=${run_dir}" \
    --compare-name vs_e1_recall_guard_1pp \
    --output-dir "${run_dir}/summary" 2>&1 | tee -a "${run_dir}/post_eval.log"

  echo "completed_at=$(date --iso-8601=seconds) run_dir=${run_dir}" | tee -a "${run_dir}/post_eval.log"
}

case "${MODE}" in
  eval)
    checkpoint="$(resolve_checkpoint)"
    run_eval "${checkpoint}"
    ;;
  wait_eval)
    checkpoint="$(resolve_checkpoint)"
    wait_for_checkpoint "${checkpoint}"
    run_eval "${checkpoint}"
    ;;
  checkpoint)
    resolve_checkpoint
    ;;
  help|*)
    cat <<EOF
Usage: $0 {eval|wait_eval|checkpoint}

Examples:
  SELECT_CHECKPOINT=epoch_1 GPU_LIST=3 bash $0 eval
  SELECT_CHECKPOINT=best GPU_LIST=3,4,6,7 bash $0 wait_eval
  CHECKPOINT=/path/to/best.pt RUN_NAME=my_eval bash $0 eval

Key env:
  OUTPUT_DIR=${OUTPUT_DIR}
  SELECT_CHECKPOINT=best|last|epoch_1|/path/to/model.pt
  GPU_LIST=${GPU_LIST}
  FORCE_EVAL=1
  WAIT_TIMEOUT_SEC=${WAIT_TIMEOUT_SEC}
EOF
    ;;
esac
