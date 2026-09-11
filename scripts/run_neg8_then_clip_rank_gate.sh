#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

GPU_LIST="${1:-3,4,6,7}"
NEG8_OUTPUT_DIR="${NEG8_OUTPUT_DIR:-outputs/football_events/vitl16_e1_precision_focal_neg8_from_e1_5090_fast_b8_g3467}"
BASELINE_PROTOCOL="${BASELINE_PROTOCOL:-outputs/football_eval_runs/vitl16_lora_r5_f32_16f_e1_frame_det_last_hr_6videos_window_overlap_checkpoint_thr/protocol_comparison_checkpoint_thr.json}"
WAIT_INTERVAL_SEC="${WAIT_INTERVAL_SEC:-1200}"
WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC:-43200}"
EVAL_GPU_LIST="${EVAL_GPU_LIST:-${GPU_LIST%%,*}}"
RANK_GPU_LIST="${RANK_GPU_LIST:-${GPU_LIST}}"
RANK_OUTPUT_DIR="${RANK_OUTPUT_DIR:-outputs/football_events/vitl16_e1_clip_rank_clean_lora_last8}"
RANK_SCRIPT="${RANK_SCRIPT:-scripts/run_football_clip_rank_lora_last8.sh}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt}"
MANIFEST="${MANIFEST:-outputs/football_hard_negatives/pointnms_fp_clean_shot_save.json}"
WAIT_FOR_MANIFEST_SEC="${WAIT_FOR_MANIFEST_SEC:-21600}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_FILE="${LOG_FILE:-${NEG8_OUTPUT_DIR}/neg8_to_clip_rank_gate.log}"

mkdir -p "$(dirname "${LOG_FILE}")"

log() {
  echo "[$(date --iso-8601=seconds)] $*" | tee -a "${LOG_FILE}"
}

training_running() {
  ps -eo pid=,cmd= \
    | grep -F "train_football_events.py" \
    | grep -F "output_dir=${NEG8_OUTPUT_DIR}" \
    | grep -v grep >/dev/null
}

premining_running() {
  ps -eo pid=,cmd= \
    | grep -E "run_football_clip_rank_clean.sh.*mine_build|vitl16_e1_pointnms_train_fp_mining|build_pointnms_hard_negatives.py" \
    | grep -v grep >/dev/null
}

wait_for_premining() {
  local waited=0
  while premining_running; do
    if (( waited >= WAIT_FOR_MANIFEST_SEC )); then
      log "timeout waiting for pre-mining process; continuing to neg8 evaluation"
      return 0
    fi
    log "pre-mining still running waited_sec=${waited}"
    sleep "${WAIT_INTERVAL_SEC}"
    waited=$((waited + WAIT_INTERVAL_SEC))
  done
}

latest_epoch_checkpoint() {
  "${PYTHON_BIN}" - "${NEG8_OUTPUT_DIR}" <<'PY'
from pathlib import Path
import re
import sys

out = Path(sys.argv[1])
best = None
for path in out.glob("epoch_*.pt"):
    match = re.fullmatch(r"epoch_(\d+)\.pt", path.name)
    if not match:
        continue
    item = (int(match.group(1)), path)
    if best is None or item[0] > best[0]:
        best = item
if best is None:
    raise SystemExit(f"No epoch checkpoint found in {out}")
print(best[1])
PY
}

run_latest_eval() {
  local checkpoint="$1"
  local epoch_name
  epoch_name="$(basename "${checkpoint}" .pt)"
  local run_name="vitl16_e1_precision_focal_neg8_${epoch_name}_6videos_gate"
  log "evaluating checkpoint=${checkpoint} run_name=${run_name} gpu=${EVAL_GPU_LIST}"
  CHECKPOINT="${checkpoint}" \
  RUN_NAME="${run_name}" \
  GPU_LIST="${EVAL_GPU_LIST}" \
  BASELINE_PROTOCOL="${BASELINE_PROTOCOL}" \
  FORCE_EVAL="${FORCE_EVAL:-0}" \
  bash scripts/run_football_single_precision_focal_post_eval.sh eval
  echo "outputs/football_eval_runs/${run_name}"
}

decide_neg8() {
  local compare_json="$1"
  "${PYTHON_BIN}" - "${compare_json}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text())
candidate = payload["candidates"][0]
protocols = candidate["protocols"]

def protocol_pass(name: str) -> bool:
    item = protocols[name]
    if not item["recall_guard_pass"]:
        return False
    if item["precision_delta_mean"] <= 0:
        return False
    for label in ("shot", "save"):
        if item["per_class"][label]["precision_delta"] <= 0:
            return False
    return True

passed = protocol_pass("point_nms") and protocol_pass("window_overlap")
for name in ("point_nms", "window_overlap"):
    item = protocols[name]
    print(
        f"{name}: guard={'PASS' if item['recall_guard_pass'] else 'FAIL'} "
        f"mean_dP={item['precision_delta_mean'] * 100:+.2f}pp "
        f"mean_dR={item['recall_delta_mean'] * 100:+.2f}pp"
    )
    for label in ("shot", "save", "set_piece"):
        cls = item["per_class"][label]
        print(
            f"  {label}: dP={cls['precision_delta'] * 100:+.2f}pp "
            f"dR={cls['recall_delta'] * 100:+.2f}pp "
            f"P={cls['precision']:.4f} R={cls['recall']:.4f}"
        )
print("decision=NEG8_CONTINUE" if passed else "decision=START_CLIP_RANK")
raise SystemExit(0 if passed else 3)
PY
}

wait_for_neg8() {
  local waited=0
  while training_running; do
    if (( waited >= WAIT_TIMEOUT_SEC )); then
      log "timeout waiting for neg8 training output_dir=${NEG8_OUTPUT_DIR}"
      return 2
    fi
    log "neg8 still running waited_sec=${waited}"
    sleep "${WAIT_INTERVAL_SEC}"
    waited=$((waited + WAIT_INTERVAL_SEC))
  done
}

start_clip_rank() {
  local run_mining="${RUN_MINING:-auto}"
  if [[ "${run_mining}" == "auto" ]]; then
    local waited=0
    while [[ ! -s "${MANIFEST}" ]]; do
      if (( waited >= WAIT_FOR_MANIFEST_SEC )); then
        log "manifest still missing after wait; clip-ranking will run mining itself manifest=${MANIFEST}"
        run_mining=1
        break
      fi
      log "waiting_for_manifest=${MANIFEST} waited_sec=${waited}"
      sleep "${WAIT_INTERVAL_SEC}"
      waited=$((waited + WAIT_INTERVAL_SEC))
    done
    if [[ -s "${MANIFEST}" ]]; then
      run_mining=0
    fi
  fi
  log "starting clean clip-ranking experiment script=${RANK_SCRIPT} on gpu=${RANK_GPU_LIST} run_mining=${run_mining} manifest=${MANIFEST}"
  RUN_MINING="${run_mining}" \
  INIT_CHECKPOINT="${INIT_CHECKPOINT}" \
  OUTPUT_DIR="${RANK_OUTPUT_DIR}" \
  MANIFEST="${MANIFEST}" \
  bash "${RANK_SCRIPT}" all "${RANK_GPU_LIST}"
}

main() {
  log "gate started neg8_output=${NEG8_OUTPUT_DIR} gpu_list=${GPU_LIST}"
  wait_for_neg8
  wait_for_premining
  local checkpoint run_dir compare_json
  checkpoint="$(latest_epoch_checkpoint)"
  run_dir="$(run_latest_eval "${checkpoint}" | tail -n 1)"
  compare_json="${run_dir}/vs_e1_recall_guard_1pp.json"
  log "decision based on ${compare_json}"
  set +e
  decide_neg8 "${compare_json}" 2>&1 | tee -a "${LOG_FILE}"
  local decision_code=${PIPESTATUS[0]}
  set -e
  if [[ "${decision_code}" == "0" ]]; then
    log "neg8 passed guard; do not start clip-ranking automatically"
    return 0
  fi
  if [[ "${decision_code}" == "3" ]]; then
    start_clip_rank
    return 0
  fi
  log "decision failed with code=${decision_code}; not starting clip-ranking"
  return "${decision_code}"
}

main "$@"
