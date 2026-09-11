#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/new_users/qiuqi/code/dinov3-main}"
PYTHON_BIN="${PYTHON_BIN:-/home/new_users/qiuqi/miniconda3/bin/python}"
GPU_MIN_FREE_MB="${GPU_MIN_FREE_MB:-30000}"
GPU_MAX_UTIL="${GPU_MAX_UTIL:-5}"
POLL_SEC="${POLL_SEC:-60}"

cd "${ROOT}"

VIDEO_IDS="${VIDEO_IDS:-2027564888428580866,2027572095412195330,2027572406738604033,2042125172076392450,2042520971893485569,2042525152494694401}"
GT_DIR="${GT_DIR:-/home/new_users/qiuqi/code/football_events_human_repair}"
VIDEO_ROOT="${VIDEO_ROOT:-/mnt/data_16t/football/raw_video_720P}"
INDEX_ROOT="${INDEX_ROOT:-outputs/football_roi_indices/robust_v2}"

E1_RUN="${E1_RUN:-outputs/football_eval_runs/vitl16_lora_r5_f32_16f_e1_frame_det_last_hr_6videos_window_overlap_checkpoint_thr}"
E2_RUN_NAME="${E2_RUN_NAME:-e2_attn_pool_epoch3_6videos_dense_ckpt_thr}"
E2_RUN="outputs/football_eval_runs/${E2_RUN_NAME}"
POINT_ROUTE_RUN="${POINT_ROUTE_RUN:-outputs/football_eval_runs/e2_shot_save_e1_set_piece_6videos_point_nms_ckpt_thr}"
WINDOW_ROUTE_RUN="${WINDOW_ROUTE_RUN:-outputs/football_eval_runs/e2_shot_save_e1_set_piece_6videos_window_overlap_ckpt_thr}"

select_gpu() {
  nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits \
    | awk -F, -v min_free="${GPU_MIN_FREE_MB}" -v max_util="${GPU_MAX_UTIL}" '
        {
          idx=$1; free=$2; util=$3;
          gsub(/ /, "", idx); gsub(/ /, "", free); gsub(/ /, "", util);
          if (free >= min_free && util <= max_util) {
            print idx;
            exit 0;
          }
        }'
}

have_e2_windows() {
  local count
  count="$(find "${E2_RUN}" -mindepth 2 -maxdepth 2 -name window_predictions.csv 2>/dev/null | wc -l | tr -d ' ')"
  [[ "${count}" -ge 6 ]]
}

if ! have_e2_windows; then
  echo "Waiting for a GPU with free_mem>=${GPU_MIN_FREE_MB}MB and util<=${GPU_MAX_UTIL}%..."
  GPU_ID=""
  while [[ -z "${GPU_ID}" ]]; do
    candidate="$(select_gpu || true)"
    if [[ -n "${candidate}" ]]; then
      sleep 15
      confirmed="$(select_gpu || true)"
      if [[ "${candidate}" == "${confirmed}" ]]; then
        GPU_ID="${candidate}"
      fi
    fi
    if [[ -z "${GPU_ID}" ]]; then
      date
      nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
      sleep "${POLL_SEC}"
    fi
  done
  echo "Selected GPU ${GPU_ID}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" scripts/evaluate_football_model.py \
    --checkpoint vitl16_lora_r5_f32_16f_e2_attn_pool_epoch3.pt \
    --mode dense \
    --video-ids "${VIDEO_IDS}" \
    --gt-dir "${GT_DIR}" \
    --video-root "xbotgo_0608=${VIDEO_ROOT}" \
    --output-root outputs/football_eval_runs \
    --run-name "${E2_RUN_NAME}" \
    --clip-sec 10 \
    --stride-sec 5 \
    --batch-size 1 \
    --num-workers 2 \
    --device cuda:0 \
    --gpu-ids 0 \
    --thresholds checkpoint \
    --prediction-postprocess point_nms \
    --nms-radius-sec 5 \
    --match-tolerance-sec 5 \
    --spatial-crop-mode robust_detector_aware \
    --detector-index-root "${INDEX_ROOT}"
else
  echo "E2 dense windows already exist: ${E2_RUN}"
fi

"${PYTHON_BIN}" scripts/route_football_eval_runs.py \
  --source-run "e1=${E1_RUN}" \
  --source-run "e2=${E2_RUN}" \
  --label-source shot=e2,save=e2,set_piece=e1 \
  --output-dir "${POINT_ROUTE_RUN}" \
  --thresholds checkpoint \
  --prediction-postprocess point_nms \
  --nms-radius-sec 5 \
  --match-tolerance-sec 5

"${PYTHON_BIN}" scripts/route_football_eval_runs.py \
  --source-run "e1=${E1_RUN}" \
  --source-run "e2=${E2_RUN}" \
  --label-source shot=e2,save=e2,set_piece=e1 \
  --output-dir "${WINDOW_ROUTE_RUN}" \
  --thresholds checkpoint \
  --prediction-postprocess window_overlap \
  --match-tolerance-sec 5

"${PYTHON_BIN}" - <<'PY' "${E1_RUN}" "${E2_RUN}" "${POINT_ROUTE_RUN}" "${WINDOW_ROUTE_RUN}"
import json
import sys
from pathlib import Path

for label, run in zip(("E1_SOURCE", "E2_SOURCE", "ROUTED_POINT_NMS", "ROUTED_WINDOW_OVERLAP"), sys.argv[1:]):
    path = Path(run) / "summary_metrics.json"
    if not path.exists():
        path = Path(run) / "summary.json"
    print(f"\n===== {label}: {run} =====")
    if not path.exists():
        print(f"missing summary: {path}")
        continue
    data = json.loads(path.read_text())
    micro = data.get("micro") or data.get("metrics", {}).get("micro") or {}
    per_class = data.get("per_class") or data.get("metrics", {}).get("per_class") or {}
    print("micro", json.dumps(micro, ensure_ascii=False, sort_keys=True))
    for cls in ("shot", "save", "set_piece"):
        print(cls, json.dumps(per_class.get(cls, {}), ensure_ascii=False, sort_keys=True))
PY
