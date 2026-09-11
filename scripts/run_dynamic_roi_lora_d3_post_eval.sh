#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-outputs/football_events/vitl16_robust_dual_16f_e1_dynamic_roi_lora_d3}"
CHECKPOINT="${CHECKPOINT:-${OUTPUT_DIR}/best.pt}"
TRAIN_PID="${TRAIN_PID:-}"
GPU_LIST="${GPU_LIST:-0,1,2,6,7}"
RUN_NAME="${RUN_NAME:-vitl16_robust_dual_16f_e1_dynamic_roi_lora_d3_6videos_window_overlap_checkpoint_thr}"
VIDEO_IDS="${VIDEO_IDS:-2027564888428580866,2027572095412195330,2027572406738604033,2042125172076392450,2042520971893485569,2042525152494694401}"
GT_DIR="${GT_DIR:-/home/new_users/qiuqi/code/football_events_human_repair}"
VIDEO_ROOT="${VIDEO_ROOT:-/mnt/data_16t/football/raw_video_720P}"
INDEX_ROOT="${INDEX_ROOT:-outputs/football_roi_indices/robust_v2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/football_eval_runs}"
BASELINE_PROTOCOL="${BASELINE_PROTOCOL:-outputs/football_eval_runs/vitl16_lora_r5_f32_16f_e1_frame_det_last_hr_6videos_window_overlap_checkpoint_thr/protocol_comparison_checkpoint_thr.json}"
LOG="${OUTPUT_DIR}/post_eval_console.log"

mkdir -p "${OUTPUT_DIR}"
echo "post_eval_started_at=$(date --iso-8601=seconds) train_pid=${TRAIN_PID} checkpoint=${CHECKPOINT}" >> "${LOG}"

if [[ -n "${TRAIN_PID}" ]]; then
  while kill -0 "${TRAIN_PID}" 2>/dev/null; do
    sleep 60
  done
fi

if [[ ! -s "${CHECKPOINT}" ]]; then
  CHECKPOINT="${OUTPUT_DIR}/last.pt"
fi
if [[ ! -s "${CHECKPOINT}" ]]; then
  echo "missing_checkpoint=${CHECKPOINT} at=$(date --iso-8601=seconds)" >> "${LOG}"
  exit 2
fi

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
gpu_count="${#GPUS[@]}"
local_gpu_ids="$(python -c 'import sys; n=int(sys.argv[1]); print(",".join(map(str, range(n))))' "${gpu_count}")"
batch_size=$(( gpu_count * 2 ))
num_workers=$(( gpu_count * 2 ))
run_dir="${OUTPUT_ROOT}/${RUN_NAME}"

echo "eval_started_at=$(date --iso-8601=seconds) checkpoint=${CHECKPOINT} gpu_list=${GPU_LIST}" >> "${LOG}"
CUDA_VISIBLE_DEVICES="${GPU_LIST}" PYTHONUNBUFFERED=1 python scripts/evaluate_football_model.py \
  --checkpoint "${CHECKPOINT}" \
  --mode dense \
  --video-ids "${VIDEO_IDS}" \
  --gt-dir "${GT_DIR}" \
  --video-root "xbotgo_0608=${VIDEO_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --run-name "${RUN_NAME}" \
  --clip-sec 10 \
  --stride-sec 5 \
  --batch-size "${batch_size}" \
  --num-workers "${num_workers}" \
  --device cuda:0 \
  --gpu-ids "${local_gpu_ids}" \
  --thresholds checkpoint \
  --prediction-postprocess window_overlap \
  --match-tolerance-sec 5 \
  --spatial-crop-mode robust_detector_aware \
  --detector-index-root "${INDEX_ROOT}" \
  --save-frame-event-logits \
  --frame-event-topk 8 >> "${LOG}" 2>&1

python scripts/recompute_football_eval_protocols.py \
  --run-dir "${run_dir}" \
  --video-ids "${VIDEO_IDS}" \
  --output-prefix protocol_comparison_checkpoint_thr \
  --nms-radius-sec 5 \
  --match-tolerance-sec 5 >> "${LOG}" 2>&1

python scripts/analyze_roi_branch_from_eval.py \
  --run-dir "${run_dir}" \
  --output-prefix roi_branch_window_overlap_checkpoint_thr \
  --postprocess window_overlap \
  --match-tolerance-sec 5 >> "${LOG}" 2>&1

if [[ -s "${BASELINE_PROTOCOL}" ]]; then
  python scripts/compare_football_eval_protocols.py \
    --baseline "${BASELINE_PROTOCOL}" \
    --candidates "${run_dir}/protocol_comparison_checkpoint_thr.json" \
    --recall-tolerance-pp 1 \
    --output "${run_dir}/vs_full_image_e1_recall_guard_1pp.json" >> "${LOG}" 2>&1
fi

echo "post_eval_completed_at=$(date --iso-8601=seconds) run_dir=${run_dir}" >> "${LOG}"
