#!/usr/bin/env bash
set -euo pipefail

BASELINE_RUN="${BASELINE_RUN:?Set BASELINE_RUN}"
CANDIDATE_RUN="${CANDIDATE_RUN:?Set CANDIDATE_RUN}"
HOLDOUT_RUN="${HOLDOUT_RUN:?Set HOLDOUT_RUN}"
BASELINE_HOLDOUT_PROTOCOL="${BASELINE_HOLDOUT_PROTOCOL:?Set BASELINE_HOLDOUT_PROTOCOL}"
VIDEO_ID_FILE="${VIDEO_ID_FILE:?Set VIDEO_ID_FILE}"
LOG="${LOG:-${CANDIDATE_RUN}/dense_calibration_pipeline.log}"
RECALL_MARGIN_PP="${RECALL_MARGIN_PP:-0.0}"
POLL_SEC="${POLL_SEC:-30}"

mkdir -p "${CANDIDATE_RUN}"
touch "${LOG}"
while [[ ! -f "${CANDIDATE_RUN}/summary_metrics.json" ]]; do
  if ! pgrep -f "run-name $(basename "${CANDIDATE_RUN}")" >/dev/null; then
    echo "candidate inference stopped before summary_metrics.json" >> "${LOG}"
    exit 1
  fi
  sleep "${POLL_SEC}"
done

calibration_json="${CANDIDATE_RUN}/dense_calibrated_thresholds.json"
python scripts/calibrate_dense_thresholds.py \
  --baseline-run "${BASELINE_RUN}" \
  --candidate-run "${CANDIDATE_RUN}" \
  --video-id-file "${VIDEO_ID_FILE}" \
  --recall-margin-pp "${RECALL_MARGIN_PP}" \
  --output "${calibration_json}" >> "${LOG}" 2>&1

thresholds="$({
  jq -r '.candidate_thresholds | "shot=\(.shot),save=\(.save),set_piece=\(.set_piece)"' \
    "${calibration_json}"
})"
for label in shot save set_piece; do
  baseline_threshold="$(jq -r ".per_class.${label}.baseline_threshold" "${calibration_json}")"
  python scripts/cross_validate_dense_threshold.py \
    --baseline-run "${BASELINE_RUN}" \
    --candidate-run "${CANDIDATE_RUN}" \
    --label "${label}" \
    --baseline-threshold "${baseline_threshold}" \
    --video-id-file "${VIDEO_ID_FILE}" \
    --match-tolerance-sec 5 \
    --recall-tolerance-pp 1 \
    --output "${CANDIDATE_RUN}/dense_calibration_loov_${label}.json" \
    >> "${LOG}" 2>&1
done

python scripts/recompute_football_eval_protocols.py \
  --run-dir "${CANDIDATE_RUN}" \
  --thresholds "${thresholds}" \
  --output-prefix protocol_comparison_dense_calibrated29 \
  >> "${LOG}" 2>&1

python scripts/recompute_football_eval_protocols.py \
  --run-dir "${HOLDOUT_RUN}" \
  --thresholds "${thresholds}" \
  --output-prefix protocol_comparison_dense_calibrated29 \
  >> "${LOG}" 2>&1

python scripts/compare_football_eval_protocols.py \
  --baseline "${BASELINE_HOLDOUT_PROTOCOL}" \
  --candidates "${HOLDOUT_RUN}/protocol_comparison_dense_calibrated29.json" \
  --recall-tolerance-pp 1 \
  --output "${HOLDOUT_RUN}/vs_e1_dense_calibrated29_recall_guard_1pp.json" \
  >> "${LOG}" 2>&1

echo "completed_at=$(date --iso-8601=seconds) thresholds=${thresholds}" >> "${LOG}"
