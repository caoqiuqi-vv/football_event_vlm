#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-outputs/football_events/vitl16_uniform_event_epoch1_hardneg_refine}"
REFERENCE_CHECKPOINT="${REFERENCE_CHECKPOINT:-outputs/football_events/vitl16_weekend_uniform_event_dual_e1backbone_512x896_frozen_ref/epoch_1.pt}"
REFERENCE_BRANCH_METRICS="${REFERENCE_BRANCH_METRICS:-outputs/football_events/vitl16_weekend_uniform_event_dual_e1backbone_512x896_frozen_ref/epoch1_same35_branch_metrics.json}"
TRAIN_PID="${TRAIN_PID:-}"
MAX_EPOCH="${MAX_EPOCH:-2}"
GPU_LIST="${GPU_LIST:-0,1,2,6,7}"
RUN_TAG="${RUN_TAG:-hardneg_refine}"
MIN_FUSED_PRECISION_DELTA="${MIN_FUSED_PRECISION_DELTA:-0.002}"
VIDEO_IDS="${VIDEO_IDS:-2027564888428580866,2027572095412195330,2027572406738604033,2042125172076392450,2042520971893485569,2042525152494694401}"
GT_DIR="${GT_DIR:-/home/new_users/qiuqi/code/football_events_human_repair}"
VIDEO_ROOT="${VIDEO_ROOT:-/mnt/data_16t/football/raw_video_720P}"
BASELINE_PROTOCOL="${BASELINE_PROTOCOL:-outputs/football_eval_runs/vitl16_lora_r5_f32_16f_e1_frame_det_last_hr_6videos_window_overlap_checkpoint_thr/protocol_comparison_checkpoint_thr.json}"
LOG="${OUTPUT_DIR}/hardneg_post_pipeline.log"

mkdir -p "${OUTPUT_DIR}"

wait_for_training_exit() {
  if [[ -z "${TRAIN_PID}" ]]; then
    return 0
  fi
  while kill -0 "${TRAIN_PID}" 2>/dev/null; do
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
        if [[ "${index}" == "${required_index}" ]] && \
           (( used >= 2000 || util >= 10 )); then
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

evaluate_checkpoint() {
  local checkpoint="$1"
  local run_name="$2"
  local gpu_count local_gpu_ids run_dir
  gpu_count="$(python -c 'import sys; print(len(sys.argv[1].split(",")))' "${GPU_LIST}")"
  local_gpu_ids="$(python -c 'import sys; n=int(sys.argv[1]); print(",".join(map(str, range(n))))' "${gpu_count}")"
  CUDA_VISIBLE_DEVICES="${GPU_LIST}" PYTHONUNBUFFERED=1 python scripts/evaluate_football_model.py \
    --checkpoint "${checkpoint}" \
    --mode dense \
    --video-ids "${VIDEO_IDS}" \
    --gt-dir "${GT_DIR}" \
    --video-root "xbotgo_0608=${VIDEO_ROOT}" \
    --run-name "${run_name}" \
    --clip-sec 10 \
    --stride-sec 5 \
    --batch-size "$(( gpu_count * 4 ))" \
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
  python scripts/analyze_football_window_fp_distance.py \
    --run-dir "${run_dir}" \
    --protocol-json "${run_dir}/protocol_comparison_checkpoint_thr.json" \
    --output "${run_dir}/window_overlap_fp_distance.json" >> "${LOG}" 2>&1
}

echo "pipeline_started_at=$(date --iso-8601=seconds) train_pid=${TRAIN_PID}" >> "${LOG}"
wait_for_training_exit

checkpoints=()
for epoch in $(seq 1 "${MAX_EPOCH}"); do
  checkpoint="${OUTPUT_DIR}/epoch_${epoch}.pt"
  if [[ -s "${checkpoint}" ]]; then
    checkpoints+=("${checkpoint}")
  fi
done
if (( ${#checkpoints[@]} == 0 )); then
  echo "no_epoch_checkpoints_at=$(date --iso-8601=seconds)" >> "${LOG}"
  exit 2
fi

report="${OUTPUT_DIR}/all_completed_epochs_vs_reference"
python scripts/compare_football_checkpoint_metrics.py \
  --baseline "${REFERENCE_CHECKPOINT}" \
  --candidates "${checkpoints[@]}" \
  --labels shot,save,set_piece \
  --recall-tolerance-pp 1 \
  --output "${report}.json" > "${report}.md" 2> "${report}.err"

selected_fused="$(
  jq --argjson min_delta "${MIN_FUSED_PRECISION_DELTA}" -r '
    [.candidates[]
      | select(
          .per_class.shot.recall_guard_pass and
          .per_class.save.recall_guard_pass and
          .per_class.set_piece.recall_guard_pass and
          (((.per_class.shot.precision_delta + .per_class.save.precision_delta) / 2) > $min_delta)
        )]
    | sort_by([
        ((.per_class.shot.precision_delta + .per_class.save.precision_delta) / 2),
        .mAP_delta
      ])
    | last
    | .path // empty
  ' "${report}.json"
)"
if [[ -z "${selected_fused}" ]]; then
  echo "no_checkpoint_passed_recall_precision_guard_at=$(date --iso-8601=seconds)" >> "${LOG}"
else
  echo "selected_fused_checkpoint=${selected_fused} at=$(date --iso-8601=seconds)" >> "${LOG}"
fi

# Select the event branch independently from fused metrics. A small static
# gate can hide meaningful event-head changes in the fused validation scores.
metric_candidates=()
for checkpoint in "${checkpoints[@]}"; do
  checkpoint_epoch="$(basename "${checkpoint}" .pt | sed 's/^epoch_//')"
  candidate_metrics="${OUTPUT_DIR}/metrics_epoch_$(printf '%03d' "${checkpoint_epoch}").json"
  if [[ -s "${candidate_metrics}" ]]; then
    metric_candidates+=("${candidate_metrics}")
  fi
done
selected_event=""
if [[ -s "${REFERENCE_BRANCH_METRICS}" && ${#metric_candidates[@]} -gt 0 ]]; then
  event_report="${OUTPUT_DIR}/all_completed_epochs_event_vs_reference"
  python scripts/compare_football_temporal_branch_metrics.py \
    --baseline "${REFERENCE_BRANCH_METRICS}" \
    --candidates "${metric_candidates[@]}" \
    --branch event \
    --labels shot,save,set_piece \
    --recall-tolerance-pp 1 \
    --output "${event_report}.json" > "${event_report}.md" 2> "${event_report}.err"
  selected_event_metrics="$(
    jq -r '
      ([.candidates[]
        | select(
            .per_class.shot.recall_guard_pass and
            .per_class.save.recall_guard_pass and
            .per_class.set_piece.recall_guard_pass and
            (((.per_class.shot.precision_delta + .per_class.save.precision_delta) / 2) > 0)
          )]
        | sort_by([
            ((.per_class.shot.precision_delta + .per_class.save.precision_delta) / 2),
            .mAP_delta
          ])
        | last
        | .path) //
      ([.candidates[] | select(.recall_guard_pass)]
        | sort_by([.precision_delta_mean, .mAP_delta])
        | last
        | .path) //
      ([.candidates[]] | sort_by(.mAP) | last | .path) //
      empty
    ' "${event_report}.json"
  )"
  if [[ -n "${selected_event_metrics}" ]]; then
    event_epoch="$(jq -r '.epoch' "${selected_event_metrics}")"
    selected_event="${OUTPUT_DIR}/epoch_${event_epoch}.pt"
  fi
fi
if [[ -z "${selected_event}" ]]; then
  selected_event="${checkpoints[$(( ${#checkpoints[@]} - 1 ))]}"
fi
echo "selected_event_checkpoint=${selected_event} at=$(date --iso-8601=seconds)" >> "${LOG}"

if ! wait_for_gpus; then
  echo "gpu_wait_timeout_at=$(date --iso-8601=seconds)" >> "${LOG}"
  exit 3
fi

epoch="$(basename "${selected_event}" .pt | sed 's/^epoch_//')"
metrics_path="${OUTPUT_DIR}/metrics_epoch_$(printf '%03d' "${epoch}").json"
read -r shot_threshold save_threshold set_piece_threshold < <(
  jq -r '.metrics.temporal_branches.event.thresholds | [.shot, .save, .set_piece] | @tsv' \
    "${metrics_path}"
)
event_checkpoint="${OUTPUT_DIR}/epoch_${epoch}_event_branch.pt"
python scripts/export_temporal_event_branch_checkpoint.py \
  --input "${selected_event}" \
  --output "${event_checkpoint}" \
  --branch-metrics "${metrics_path}" \
  --thresholds "${shot_threshold}" "${save_threshold}" "${set_piece_threshold}" >> "${LOG}" 2>&1

if [[ -n "${selected_fused}" ]]; then
  fused_epoch="$(basename "${selected_fused}" .pt | sed 's/^epoch_//')"
  evaluate_checkpoint "${selected_fused}" "${RUN_TAG}_epoch${fused_epoch}_fused_6videos_dense_checkpoint_thr"
fi
event_run_name="${RUN_TAG}_epoch${epoch}_event_6videos_dense_checkpoint_thr"
evaluate_checkpoint "${event_checkpoint}" "${event_run_name}"
echo "pipeline_completed_at=$(date --iso-8601=seconds) event_run_name=${event_run_name}" >> "${LOG}"
