#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
VIDEO_ROOT="${VIDEO_ROOT:-/mnt/data_16t/football/raw_video_720P}"
GT_DIR="${GT_DIR:-/home/new_users/qiuqi/code/football_events_human_repair}"
INDEX_ROOT="${INDEX_ROOT:-outputs/football_roi_indices/robust_v2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/football_eval_runs}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-2}"
ALLOW_MISSING="${ALLOW_MISSING:-0}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.6}"
CONFIDENCE_POWER="${CONFIDENCE_POWER:-1.0}"
VIDEO_IDS="${VIDEO_IDS:-2027564888428580866,2027572095412195330,2027572406738604033,2042125172076392450,2042520971893485569,2042525152494694401}"

D0_CHECKPOINT="${D0_CHECKPOINT:-outputs/football_events/vitl16_robust_dual_16f_e1_fixed_roi_d0/best.pt}"
D1_CHECKPOINT="${D1_CHECKPOINT:-outputs/football_events/vitl16_robust_dual_16f_e1_dynamic_roi_d1_a800_5gpu_colocated/best.pt}"
D2_CHECKPOINT="${D2_CHECKPOINT:-outputs/football_events/vitl16_robust_dual_16f_e1_dynamic_roi_feature_quality_d2_a800_5gpu/best.pt}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONHASHSEED=42

IFS=',' read -r -a VIDEO_ID_ARRAY <<< "${VIDEO_IDS}"

resolve_thresholds() {
  local run_dir="$1"
  local expected=""
  local video_id summary_path current
  for video_id in "${VIDEO_ID_ARRAY[@]}"; do
    summary_path="${run_dir}/${video_id}/summary.json"
    if [[ ! -f "${summary_path}" ]]; then
      echo "Missing evaluation summary: ${summary_path}" >&2
      return 2
    fi
    current="$(jq -c '.thresholds | to_entries | sort_by(.key)' "${summary_path}")"
    if [[ -z "${expected}" ]]; then
      expected="${current}"
    elif [[ "${current}" != "${expected}" ]]; then
      echo "Inconsistent checkpoint thresholds: ${summary_path}" >&2
      echo "expected=${expected}" >&2
      echo "actual=${current}" >&2
      return 2
    fi
  done
  jq -r 'map("\(.key)=\(.value)") | join(",")' <<< "${expected}"
}

evaluate_checkpoint() {
  local name="$1"
  local checkpoint="$2"
  local run_name="dynamic_roi_${name}_6videos_windows_checkpoint_thr"
  local run_dir="${OUTPUT_ROOT}/${run_name}"
  local thresholds

  if [[ ! -f "${checkpoint}" ]]; then
    if [[ "${ALLOW_MISSING}" == "1" ]]; then
      echo "Skip missing ${name} checkpoint: ${checkpoint}" >&2
      return 0
    fi
    echo "Missing ${name} checkpoint: ${checkpoint}" >&2
    return 2
  fi

  echo "Evaluating ${name}: ${checkpoint}"
  "${PYTHON_BIN}" scripts/evaluate_football_model.py \
    --checkpoint "${checkpoint}" \
    --mode dense \
    --video-ids "${VIDEO_IDS}" \
    --gt-dir "${GT_DIR}" \
    --video-root "xbotgo_0608=${VIDEO_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --run-name "${run_name}" \
    --clip-sec 10 \
    --stride-sec 5 \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --num-workers "${EVAL_NUM_WORKERS}" \
    --device cuda:0 \
    --gpu-ids 0 \
    --thresholds checkpoint \
    --prediction-postprocess window_overlap \
    --match-tolerance-sec 5 \
    --spatial-crop-mode robust_detector_aware \
    --detector-index-root "${INDEX_ROOT}"

  thresholds="$(resolve_thresholds "${run_dir}")"
  echo "${name} resolved thresholds: ${thresholds}"

  "${PYTHON_BIN}" scripts/analyze_roi_branch_from_eval.py \
    --run-dir "${run_dir}" \
    --output-prefix roi_branch_window_overlap_checkpoint_thr \
    --postprocess window_overlap \
    --thresholds "${thresholds}" \
    --match-tolerance-sec 5 \
    --confidence-threshold "${CONFIDENCE_THRESHOLD}" \
    --confidence-power "${CONFIDENCE_POWER}"

  "${PYTHON_BIN}" scripts/analyze_roi_branch_from_eval.py \
    --run-dir "${run_dir}" \
    --output-prefix roi_branch_point_nms_checkpoint_thr \
    --postprocess point_nms \
    --thresholds "${thresholds}" \
    --match-tolerance-sec 5 \
    --nms-radius-sec 5 \
    --confidence-threshold "${CONFIDENCE_THRESHOLD}" \
    --confidence-power "${CONFIDENCE_POWER}"
}

evaluate_checkpoint d0 "${D0_CHECKPOINT}"
evaluate_checkpoint d1 "${D1_CHECKPOINT}"
evaluate_checkpoint d2 "${D2_CHECKPOINT}"

echo "Dynamic ROI evaluation complete."
