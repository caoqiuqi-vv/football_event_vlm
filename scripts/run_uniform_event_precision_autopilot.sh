#!/usr/bin/env bash
set -euo pipefail

GPU_LIST="${GPU_LIST:-0,1,2,6,7}"
CURRENT_OUTPUT="${CURRENT_OUTPUT:-outputs/football_events/vitl16_uniform_event_epoch1_hardneg_refine}"
REFERENCE_CHECKPOINT="${REFERENCE_CHECKPOINT:-outputs/football_events/vitl16_weekend_uniform_event_dual_e1backbone_512x896_frozen_ref/epoch_1.pt}"
POLL_SEC="${POLL_SEC:-30}"
NEXT_EPOCHS="${NEXT_EPOCHS:-1}"
EVAL_ROOT="${EVAL_ROOT:-outputs/football_eval_runs}"
DRY_RUN="${DRY_RUN:-false}"
FAIL_MODE="${FAIL_MODE:-online_ohem}"
LOG="${AUTOPILOT_LOG:-outputs/football_events/precision_autopilot.log}"

mkdir -p "$(dirname "${LOG}")"
echo "autopilot_started_at=$(date --iso-8601=seconds)" >> "${LOG}"

while ! grep -q '^pipeline_completed_at=.*event_run_name=' "${CURRENT_OUTPUT}/hardneg_post_pipeline.log" 2>/dev/null; do
  sleep "${POLL_SEC}"
done

completion_line="$(grep '^pipeline_completed_at=.*event_run_name=' "${CURRENT_OUTPUT}/hardneg_post_pipeline.log" | tail -n 1)"
event_run_name="${completion_line##*event_run_name=}"
event_report="${EVAL_ROOT}/${event_run_name}/vs_e1_recall_guard_1pp.json"
while [[ ! -s "${event_report}" ]]; do
  sleep "${POLL_SEC}"
done

selected_event="$(grep '^selected_event_checkpoint=' "${CURRENT_OUTPUT}/hardneg_post_pipeline.log" | tail -n 1 | sed -E 's/^selected_event_checkpoint=([^ ]+).*/\1/')"

if jq -e '
  .candidates[0].protocols.window_overlap
  | (.recall_guard_pass and .precision_improved)
' "${event_report}" >/dev/null; then
  mode="adaptive_gate"
  init_checkpoint="${selected_event}"
  output_dir="outputs/football_events/vitl16_uniform_event_adaptive_gate_refine"
  run_tag="adaptive_gate"
else
  mode="${FAIL_MODE}"
  init_checkpoint="${REFERENCE_CHECKPOINT}"
  case "${mode}" in
    online_ohem)
      output_dir="outputs/football_events/vitl16_uniform_event_online_ohem_refine"
      ;;
    online_rank)
      output_dir="outputs/football_events/vitl16_uniform_event_duration_online_rank_refine"
      ;;
    duration_ohem)
      output_dir="outputs/football_events/vitl16_uniform_event_duration_ohem_refine"
      ;;
    hard_rank)
      output_dir="outputs/football_events/vitl16_uniform_event_hardneg_rank_refine"
      ;;
    *)
      echo "Unsupported FAIL_MODE=${mode}" >> "${LOG}"
      exit 2
      ;;
  esac
  run_tag="${mode}"
fi

{
  echo "decision_at=$(date --iso-8601=seconds)"
  echo "source_event_report=${event_report}"
  echo "selected_mode=${mode}"
  echo "selected_init_checkpoint=${init_checkpoint}"
  echo "selected_output_dir=${output_dir}"
  echo "next_epochs=${NEXT_EPOCHS}"
} >> "${LOG}"

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "autopilot_dry_run_completed_at=$(date --iso-8601=seconds)" >> "${LOG}"
  exit 0
fi

env INIT_CHECKPOINT="${init_checkpoint}" OUTPUT_DIR="${output_dir}" EPOCHS="${NEXT_EPOCHS}" bash scripts/run_uniform_event_precision_refine.sh "${mode}" "${GPU_LIST}" >> "${LOG}" 2>&1

env OUTPUT_DIR="${output_dir}" REFERENCE_CHECKPOINT="${REFERENCE_CHECKPOINT}" GPU_LIST="${GPU_LIST}" RUN_TAG="${run_tag}" MAX_EPOCH="${NEXT_EPOCHS}" bash scripts/run_uniform_event_hardneg_post_pipeline.sh >> "${LOG}" 2>&1

echo "autopilot_completed_at=$(date --iso-8601=seconds)" >> "${LOG}"
