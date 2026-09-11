#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-outputs/football_events/vitl16_uniform_event_duration_ohem_refine}"
CHECKPOINT="${CHECKPOINT:-${OUTPUT_DIR}/epoch_1.pt}"
CONFIG="${CONFIG:-${OUTPUT_DIR}/config.yaml}"
REFERENCE_BRANCH_METRICS="${REFERENCE_BRANCH_METRICS:-outputs/football_events/vitl16_weekend_uniform_event_dual_e1backbone_512x896_frozen_ref/epoch1_same35_branch_metrics.json}"
BASELINE_PROTOCOL="${BASELINE_PROTOCOL:-outputs/football_eval_runs/vitl16_lora_r5_f32_16f_e1_frame_det_last_hr_6videos_window_overlap_checkpoint_thr/protocol_comparison_checkpoint_thr.json}"
GPU_LIST="${GPU_LIST:-0,1,2,6,7}"
RECALL_TOLERANCE_PP="${RECALL_TOLERANCE_PP:-1.0}"
VIDEO_IDS="${VIDEO_IDS:-2027564888428580866,2027572095412195330,2027572406738604033,2042125172076392450,2042520971893485569,2042525152494694401}"
GT_DIR="${GT_DIR:-/home/new_users/qiuqi/code/football_events_human_repair}"
VIDEO_ROOT="${VIDEO_ROOT:-/mnt/data_16t/football/raw_video_720P}"
WAIT_FOR_LOG="${WAIT_FOR_LOG-${OUTPUT_DIR}/hardneg_post_pipeline.log}"
WAIT_FOR_CHECKPOINT="${WAIT_FOR_CHECKPOINT:-true}"
RUN_NAME="${RUN_NAME:-duration_ohem_epoch1_event_recall_floor_exact_6videos_dense_checkpoint_thr}"
RAW_METRICS_OUTPUT="${RAW_METRICS_OUTPUT:-${OUTPUT_DIR}/recall_constrained_metrics.json}"
METRICS_OUTPUT="${METRICS_OUTPUT:-${OUTPUT_DIR}/recall_constrained_exact_metrics.json}"
EVENT_CHECKPOINT="${EVENT_CHECKPOINT:-${OUTPUT_DIR}/epoch_1_event_recall_floor_exact.pt}"
LOG="${LOG:-${OUTPUT_DIR}/recall_constrained_pipeline.log}"
REUSE_RAW_METRICS="${REUSE_RAW_METRICS:-false}"

mkdir -p "${OUTPUT_DIR}"

wait_for_current_pipeline() {
  if [[ -z "${WAIT_FOR_LOG}" ]]; then
    return 0
  fi
  while ! grep -q '^pipeline_completed_at=' "${WAIT_FOR_LOG}" 2>/dev/null; do
    sleep 30
  done
}

wait_for_checkpoint() {
  if [[ "${WAIT_FOR_CHECKPOINT}" != "true" ]]; then
    return 0
  fi
  while [[ ! -s "${CHECKPOINT}" ]]; do
    sleep 30
  done
}

wait_for_gpus() {
  local required=()
  IFS=',' read -r -a required <<< "${GPU_LIST}"
  for _ in $(seq 1 240); do
    local free=1
    local index used util required_index
    while IFS=',' read -r index used util; do
      index="${index//[[:space:]]/}"
      used="${used//[[:space:]]/}"
      util="${util//[[:space:]]/}"
      for required_index in "${required[@]}"; do
        if [[ "${index}" == "${required_index}" ]] && (( used >= 2000 )); then
          free=0
        fi
      done
    done < <(
      nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
        --format=csv,noheader,nounits
    )
    if (( free == 1 )); then
      return 0
    fi
    sleep 30
  done
  return 1
}

echo "started_at=$(date --iso-8601=seconds)" >> "${LOG}"
wait_for_current_pipeline
wait_for_checkpoint
wait_for_gpus

gpu_count="$(python -c 'import sys; print(len(sys.argv[1].split(",")))' "${GPU_LIST}")"
local_gpu_ids_yaml="$(python -c 'import sys; n=int(sys.argv[1]); print("[" + ",".join(map(str, range(n))) + "]")' "${gpu_count}")"
local_gpu_ids_csv="$(python -c 'import sys; n=int(sys.argv[1]); print(",".join(map(str, range(n))))' "${gpu_count}")"

read -r shot_floor save_floor set_piece_floor < <(
  jq -r --argjson tolerance_pp "${RECALL_TOLERANCE_PP}" '
    (
      .metrics.temporal_branches.event.tuned.per_class //
      .temporal_branches.event.tuned.per_class //
      .tuned.per_class
    )
    | [
        ([.shot.recall - ($tolerance_pp / 100.0), 0] | max),
        ([.save.recall - ($tolerance_pp / 100.0), 0] | max),
        ([.set_piece.recall - ($tolerance_pp / 100.0), 0] | max)
      ]
    | @tsv
  ' "${REFERENCE_BRANCH_METRICS}"
)

{
  echo "recall_floors shot=${shot_floor} save=${save_floor} set_piece=${set_piece_floor}"
  echo "eval_started_at=$(date --iso-8601=seconds) checkpoint=${CHECKPOINT}"
} >> "${LOG}"

if [[ "${REUSE_RAW_METRICS}" != "true" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_LIST}" PYTHONUNBUFFERED=1 python train_football_events.py \
    --config "${CONFIG}" \
    --eval-only \
    --eval-output "${RAW_METRICS_OUTPUT}" \
    "gpu_ids=${local_gpu_ids_yaml}" \
    "model.init_checkpoint=${CHECKPOINT}" \
    "train.resume.enabled=false" \
    "eval.per_gpu_batch_size=4" \
    "eval.save_predictions=true" \
    "eval.tuned_min_recall.shot=${shot_floor}" \
    "eval.tuned_min_recall.save=${save_floor}" \
    "eval.tuned_min_recall.set_piece=${set_piece_floor}" >> "${LOG}" 2>&1
elif [[ ! -s "${RAW_METRICS_OUTPUT}" ]]; then
  echo "Missing RAW_METRICS_OUTPUT=${RAW_METRICS_OUTPUT}" >> "${LOG}"
  exit 2
fi

python scripts/recompute_football_recall_constrained_metrics.py \
  --input "${RAW_METRICS_OUTPUT}" \
  --output "${METRICS_OUTPUT}" \
  --label-schema set_piece \
  --min-recalls \
    "shot=${shot_floor},save=${save_floor},set_piece=${set_piece_floor}" >> "${LOG}" 2>&1

read -r shot_threshold save_threshold set_piece_threshold < <(
  jq -r '.temporal_branches.event.thresholds | [.shot, .save, .set_piece] | @tsv' \
    "${METRICS_OUTPUT}"
)

python scripts/export_temporal_event_branch_checkpoint.py \
  --input "${CHECKPOINT}" \
  --output "${EVENT_CHECKPOINT}" \
  --branch-metrics "${METRICS_OUTPUT}" \
  --thresholds "${shot_threshold}" "${save_threshold}" "${set_piece_threshold}" >> "${LOG}" 2>&1

CUDA_VISIBLE_DEVICES="${GPU_LIST}" PYTHONUNBUFFERED=1 python scripts/evaluate_football_model.py \
  --checkpoint "${EVENT_CHECKPOINT}" \
  --mode dense \
  --video-ids "${VIDEO_IDS}" \
  --gt-dir "${GT_DIR}" \
  --video-root "xbotgo_0608=${VIDEO_ROOT}" \
  --run-name "${RUN_NAME}" \
  --clip-sec 10 \
  --stride-sec 5 \
  --batch-size "$(( gpu_count * 4 ))" \
  --num-workers "$(( gpu_count * 2 ))" \
  --device cuda:0 \
  --gpu-ids "${local_gpu_ids_csv}" \
  --thresholds checkpoint \
  --prediction-postprocess point_nms \
  --nms-radius-sec 5 \
  --match-tolerance-sec 5 \
  --save-frame-event-logits \
  --frame-event-topk 8 >> "${LOG}" 2>&1

run_dir="outputs/football_eval_runs/${RUN_NAME}"
python scripts/recompute_football_eval_protocols.py \
  --run-dir "${run_dir}" \
  --video-ids "${VIDEO_IDS}" \
  --output-prefix protocol_comparison_checkpoint_thr \
  --nms-radius-sec 5 \
  --match-tolerance-sec 5 >> "${LOG}" 2>&1
python scripts/compare_football_eval_protocols.py \
  --baseline "${BASELINE_PROTOCOL}" \
  --candidates "${run_dir}/protocol_comparison_checkpoint_thr.json" \
  --recall-tolerance-pp "${RECALL_TOLERANCE_PP}" \
  --output "${run_dir}/vs_e1_recall_guard_1pp.json" >> "${LOG}" 2>&1
python scripts/analyze_football_window_fp_distance.py \
  --run-dir "${run_dir}" \
  --protocol-json "${run_dir}/protocol_comparison_checkpoint_thr.json" \
  --output "${run_dir}/window_overlap_fp_distance.json" >> "${LOG}" 2>&1

echo "completed_at=$(date --iso-8601=seconds) run_dir=${run_dir}" >> "${LOG}"
