#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-}"
GPU="${2:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DETECTOR_ROOT="${DETECTOR_ROOT:-/mnt/data_16t/football/detection_and_track_result}"
VIDEO_ROOT="${VIDEO_ROOT:-/mnt/data_16t/football/raw_video_720P}"
INDEX_ROOT="${INDEX_ROOT:-outputs/football_roi_indices/robust_v2}"
SPLIT_ROOT="${SPLIT_ROOT:-configs/football/splits/vitl16_lvd1689m_set_piece_seed42_162videos}"
DEBUG_SPLIT_ROOT="${DEBUG_SPLIT_ROOT:-configs/football/splits/debug_available_roi_indices}"
ACTION_SPLIT_ROOT="${ACTION_SPLIT_ROOT:-configs/football/splits/vitl16_lvd1689m_set_piece_seed42_162videos}"
RESOLUTION_JSON="outputs/football_resolution_check/lora_r2_detector58.json"
EXP4_INIT_CHECKPOINT="${INIT_CHECKPOINT:-/mnt/data_16t/football/qiuqi/checkpoints/lora_r5_best.pt}"
EXP4_DUAL_LORA_CONFIG="configs/football/dinov3_vitl16_robust_dual_16f_exp4_lora.yaml"
EXP4_DUAL_LORA_INIT_CHECKPOINT="${EXP4_DUAL_LORA_INIT_CHECKPOINT:-${INIT_CHECKPOINT:-outputs/football_events/vitl16_robust_dual_16f_exp4/best.pt}}"
EXP4_DUAL_LORA_PER_GPU_BATCH_SIZE="${DUAL_LORA_PER_GPU_BATCH_SIZE:-2}"
EXP4_DUAL_LORA_GRAD_ACCUM_STEPS="${DUAL_LORA_GRAD_ACCUM_STEPS:-4}"
EXP4_DUAL_LORA_LR_PER_GPU="${DUAL_LORA_LR_PER_GPU:-0.00005}"
EXP4_DUAL_LORA_BACKBONE_LR_PER_GPU="${DUAL_LORA_BACKBONE_LR_PER_GPU:-0.00001}"
ACTION_INIT_CHECKPOINT="${ACTION_INIT_CHECKPOINT:-${INIT_CHECKPOINT:-/mnt/data_16t/football/qiuqi/checkpoints/vitl16_robust_dual_16f.pt}}"
LORA_R5_F32_INIT_CHECKPOINT="${LORA_R5_F32_INIT_CHECKPOINT:-${INIT_CHECKPOINT:-/mnt/data_16t/football/qiuqi/checkpoints/lora_r5_last.pt}}"
HR_SINGLE_LORA_CONFIG="configs/football/dinov3_vitl16_lora_r5_f32_16f_hr_single.yaml"
HR_SINGLE_LORA_FRAME_DET_CONFIG="configs/football/dinov3_vitl16_lora_r5_f32_16f_hr_single_frame_det.yaml"
E1_HARD_NEG_CONFIG="configs/football/dinov3_vitl16_robust_dual_16f_e1_frame_det_hard_neg.yaml"
E1_HARD_NEG_INIT_CHECKPOINT="${E1_HARD_NEG_INIT_CHECKPOINT:-${INIT_CHECKPOINT:-outputs/football_events/vitl16_robust_dual_16f_e1_frame_det_exp2_from_vitl16_robust_dual_16f_exp4_hr/best.pt}}"
E1_HARD_NEG_MANIFEST="${E1_HARD_NEG_MANIFEST:-outputs/football_hard_negatives/e1_exp2_best_train_fp_shot_save.json}"
E1_HARD_NEG_MIN_PROBS="${E1_HARD_NEG_MIN_PROBS:-shot=0.30,save=0.40}"
E1_HARD_NEG_REJECT_ANY_LABEL_GT="${E1_HARD_NEG_REJECT_ANY_LABEL_GT:-0}"
DYNAMIC_ROI_D0_CONFIG="configs/football/dinov3_vitl16_robust_dual_16f_e1_fixed_roi_d0.yaml"
DYNAMIC_ROI_D1_CONFIG="configs/football/dinov3_vitl16_robust_dual_16f_e1_dynamic_roi_d1.yaml"
DYNAMIC_ROI_D1_A800_COLOCATED_CONFIG="configs/football/dinov3_vitl16_robust_dual_16f_e1_dynamic_roi_d1_a800_5gpu_colocated.yaml"
DYNAMIC_ROI_D2_CONFIG="configs/football/dinov3_vitl16_robust_dual_16f_e1_dynamic_roi_feature_quality_d2.yaml"
DYNAMIC_ROI_D2_A800_CONFIG="configs/football/dinov3_vitl16_robust_dual_16f_e1_dynamic_roi_feature_quality_d2_a800_5gpu.yaml"
DYNAMIC_ROI_D3_LORA_CONFIG="configs/football/dinov3_vitl16_robust_dual_16f_e1_dynamic_roi_lora_d3.yaml"
DYNAMIC_ROI_D4_DECOUPLED_LORA_CONFIG="configs/football/dinov3_vitl16_robust_dual_16f_e1_dynamic_roi_decoupled_lora_d4.yaml"
DYNAMIC_ROI_D0_INIT_CHECKPOINT="${DYNAMIC_ROI_D0_INIT_CHECKPOINT:-${INIT_CHECKPOINT:-outputs/football_events/vitl16_robust_dual_16f_e1_frame_det_exp2_from_vitl16_robust_dual_16f_exp4_hr/best.pt}}"
DYNAMIC_ROI_D1_INIT_CHECKPOINT="${DYNAMIC_ROI_D1_INIT_CHECKPOINT:-${INIT_CHECKPOINT:-outputs/football_events/vitl16_robust_dual_16f_e1_frame_det_exp2_from_vitl16_robust_dual_16f_exp4_hr/best.pt}}"
DYNAMIC_ROI_D2_INIT_CHECKPOINT="${DYNAMIC_ROI_D2_INIT_CHECKPOINT:-outputs/football_events/vitl16_robust_dual_16f_e1_dynamic_roi_d1/best.pt}"
DYNAMIC_ROI_D3_LORA_INIT_CHECKPOINT="${DYNAMIC_ROI_D3_LORA_INIT_CHECKPOINT:-${INIT_CHECKPOINT:-/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt}}"
DYNAMIC_ROI_D4_DECOUPLED_LORA_INIT_CHECKPOINT="${DYNAMIC_ROI_D4_DECOUPLED_LORA_INIT_CHECKPOINT:-${INIT_CHECKPOINT:-/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt}}"
ROI_DEBUG_CONFIG="${ROI_DEBUG_CONFIG:-configs/football/dinov3_vitl16_robust_dual_16f_exp4.yaml}"
ROI_DEBUG_OUTPUT="${ROI_DEBUG_OUTPUT:-outputs/football_roi_debug/val_positive}"
ROI_DEBUG_LABELS="${ROI_DEBUG_LABELS:-}"
ROI_DEBUG_VIDEO_IDS="${ROI_DEBUG_VIDEO_IDS:-}"
ROI_DEBUG_MAX_SAMPLES="${ROI_DEBUG_MAX_SAMPLES:-0}"
ROI_DEBUG_MAX_PER_VIDEO="${ROI_DEBUG_MAX_PER_VIDEO:-0}"
ROI_DEBUG_OUTPUT_FPS="${ROI_DEBUG_OUTPUT_FPS:-2}"
ROI_DEBUG_FRAME_MODE="${ROI_DEBUG_FRAME_MODE:-model}"
ROI_DEBUG_LAYOUT="${ROI_DEBUG_LAYOUT:-review}"
ROI_COMPARE_OUTPUT="${ROI_COMPARE_OUTPUT:-outputs/football_roi_debug/gt_fixed_vs_dynamic}"
ROI_COMPARE_LABELS="${ROI_COMPARE_LABELS:-shot,save,set_piece}"
ROI_COMPARE_VIDEO_IDS="${ROI_COMPARE_VIDEO_IDS:-}"
ROI_COMPARE_MAX_SAMPLES="${ROI_COMPARE_MAX_SAMPLES:-0}"
ROI_COMPARE_MAX_PER_VIDEO="${ROI_COMPARE_MAX_PER_VIDEO:-0}"
ROI_COMPARE_OUTPUT_FPS="${ROI_COMPARE_OUTPUT_FPS:-1.6}"
VIDEO_IDS="2027564888428580866,2027572095412195330,2027572406738604033,2042125172076392450,2042520971893485569,2042525152494694401"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONHASHSEED=42

usage() {
  echo "Usage: bash scripts/launch_football_detection_aware.sh <index|make_debug_split|roi_crops|roi_crops_debug|val_positive_roi_videos|gt_roi_compare|d0_fixed_roi|debug_d0_fixed_roi|eval_d0_fixed_roi|d1_dynamic_roi|d1_dynamic_roi_a800|debug_d1_dynamic_roi|eval_d1_dynamic_roi|d2_feature_quality|d2_feature_quality_a800|debug_d2_feature_quality|eval_d2_feature_quality|resolution|roi_audit|baseline_global|baseline_legacy|baseline_robust|baseline_summary|train_global|eval_global|train_gate|eval_gate|exp3|eval3|exp4_16f|eval4_16f|exp4_16f_debug|exp4_dual_lora|exp4_dual_lora_debug|eval_exp4_dual_lora|e1_frame_det|eval_e1_frame_det|debug_e1_frame_det|mine_e1_hard_neg|e1_hard_neg|eval_e1_hard_neg|debug_e1_hard_neg|e2_attn_pool|eval_e2_attn_pool|debug_e2_attn_pool|e3_class_query|eval_e3_class_query|debug_e3_class_query|e4_event_topk|eval_e4_event_topk|debug_e4_event_topk|e1_lora_r5_f32|eval_e1_lora_r5_f32|debug_e1_lora_r5_f32|e2_lora_r5_f32|eval_e2_lora_r5_f32|debug_e2_lora_r5_f32|e3_lora_r5_f32|eval_e3_lora_r5_f32|debug_e3_lora_r5_f32|e4_lora_r5_f32|eval_e4_lora_r5_f32|debug_e4_lora_r5_f32> [cuda_visible_devices]"
}

require_index() {
  local count
  count="$(find "${INDEX_ROOT}" -maxdepth 1 -type f -name '*.pt' ! -name 'summary.pt' 2>/dev/null | wc -l)"
  if [[ "${count}" -le 0 ]]; then
    echo "Missing ROI indices in ${INDEX_ROOT}. Run stage 'index' first." >&2
    exit 2
  fi
}

require_split_indices() {
  "${PYTHON_BIN}" - <<'PY_CHECK' "${SPLIT_ROOT}" "${INDEX_ROOT}"
from pathlib import Path
import sys
split_root = Path(sys.argv[1])
index_root = Path(sys.argv[2])
indexed = {path.stem for path in index_root.glob('*.pt')}
missing = {}
for split in ('train', 'val'):
    ids = [line.strip() for line in (split_root / f'{split}_video_ids.txt').read_text().splitlines() if line.strip() and not line.strip().startswith('#')]
    missing[split] = [video_id for video_id in ids if video_id not in indexed]
if missing['train'] or missing['val']:
    print(
        f"Incomplete ROI index for split={split_root}: "
        f"missing_train={len(missing['train'])} missing_val={len(missing['val'])}. "
        "Use stage exp4_16f_debug while data is still transferring.",
        file=sys.stderr,
    )
    for split in ('train', 'val'):
        if missing[split]:
            print(f"missing_{split}_roi_index_ids:", file=sys.stderr)
            for video_id in missing[split]:
                print(video_id, file=sys.stderr)
    raise SystemExit(2)
print(f"ROI index complete for split={split_root}")
PY_CHECK
}

visible_gpu_ids_csv() {
  local count
  count="$(python -c 'import os; print(len([x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if x.strip()]))')"
  python -c 'import sys; n=max(int(sys.argv[1]), 1); print(",".join(str(i) for i in range(n)))' "${count}"
}

gpu_ids_override_arg() {
  local visible="${CUDA_VISIBLE_DEVICES:-0}"
  local count
  count="$(python -c 'import os; print(len([x for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if x.strip()]))')"
  python -c 'import sys; n=max(int(sys.argv[1]), 1); print("gpu_ids=[" + ",".join(str(i) for i in range(n)) + "]")' "${count}"
}

config_output_dir() {
  sed -n "s/^output_dir:[[:space:]]*//p" "$1" | head -n 1
}

run_train_logged() {
  local config_path="$1"
  local output_dir="$2"
  shift 2
  local log_file="${TRAIN_LOG_FILE:-${output_dir}/train_console.log}"
  local status had_errexit=0
  mkdir -p "$(dirname "${log_file}")"
  {
    echo
    echo "===== football training run ====="
    echo "started_at=$(date --iso-8601=seconds)"
    echo "host=$(hostname)"
    echo "cwd=$(pwd)"
    echo "config=${config_path}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
    printf "command="
    printf " %q" "$@"
    echo
    for fingerprint_path in \
      "${config_path}" \
      train_football_events.py \
      football_detection_aware.py \
      football_roi_scoring.py; do
      if [[ -f "${fingerprint_path}" ]]; then
        printf "sha256="
        sha256sum "${fingerprint_path}"
      fi
    done
    echo "log_file=${log_file}"
  } | tee -a "${log_file}"
  [[ "$-" == *e* ]] && had_errexit=1
  set +e
  PYTHONUNBUFFERED=1 "$@" 2>&1 | tee -a "${log_file}"
  status="${PIPESTATUS[0]}"
  [[ "${had_errexit}" -eq 1 ]] && set -e
  echo "finished_at=$(date --iso-8601=seconds) exit_status=${status}" | tee -a "${log_file}"
  return "${status}"
}

make_debug_split() {
  "${PYTHON_BIN}" - "${SPLIT_ROOT}" "${INDEX_ROOT}" "${DEBUG_SPLIT_ROOT}" "${DEBUG_TRAIN_VIDEOS:-4}" "${DEBUG_VAL_VIDEOS:-1}" <<'PY'
from pathlib import Path
import sys
split_root = Path(sys.argv[1])
index_root = Path(sys.argv[2])
out_root = Path(sys.argv[3])
limits = {"train": int(sys.argv[4]), "val": int(sys.argv[5])}
indexed = {path.stem for path in index_root.glob('*.pt')}
out_root.mkdir(parents=True, exist_ok=True)
summary = {}
for split in ('train', 'val'):
    src = split_root / f'{split}_video_ids.txt'
    ids = [line.strip() for line in src.read_text().splitlines() if line.strip() and not line.strip().startswith('#')]
    kept = [video_id for video_id in ids if video_id in indexed][:limits[split]]
    missing = [video_id for video_id in ids if video_id not in indexed]
    (out_root / f'{split}_video_ids.txt').write_text(''.join(f'{video_id}\n' for video_id in kept))
    (out_root / f'{split}_missing_roi_index_video_ids.txt').write_text(''.join(f'{video_id}\n' for video_id in missing))
    summary[split] = {'source': len(ids), 'kept': len(kept), 'missing_roi_index': len(missing)}
if not summary['train']['kept']:
    raise SystemExit(f'No indexed train videos found under {index_root}')
if not summary['val']['kept']:
    raise SystemExit(f'No indexed val videos found under {index_root}')
print(f"debug split={out_root} train={summary['train']['kept']}/{summary['train']['source']} val={summary['val']['kept']}/{summary['val']['source']}")
PY
}

write_action_requested_split_from_config() {
  local config_path="$1"
  "${PYTHON_BIN}" - <<'PY' "${config_path}" "${ACTION_SPLIT_ROOT}"
from pathlib import Path
import sys
import yaml
config_path = Path(sys.argv[1])
out_root = Path(sys.argv[2])
cfg = yaml.safe_load(config_path.read_text())
split_files = ((cfg.get('data') or {}).get('long_video') or {}).get('split_files') or {}
out_root.mkdir(parents=True, exist_ok=True)
for split in ('train', 'val'):
    files = split_files.get(split) or []
    if isinstance(files, (str, Path)):
        files = [files]
    ids = []
    for item in files:
        path = Path(item)
        ids.extend(
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip() and not line.strip().startswith('#')
        )
    deduped = []
    seen = set()
    for video_id in ids:
        if video_id in seen:
            continue
        seen.add(video_id)
        deduped.append(video_id)
    (out_root / f'{split}_requested_video_ids.txt').write_text(''.join(f'{video_id}\n' for video_id in deduped))
    print(f"requested_{split}_ids={len(deduped)} path={out_root / f'{split}_requested_video_ids.txt'}")
PY
}

build_action_indices_from_config() {
  local config_path="$1"
  local request_root="${2:-${ACTION_SPLIT_ROOT}}"
  ACTION_SPLIT_ROOT="${request_root}" write_action_requested_split_from_config "${config_path}"
  "${PYTHON_BIN}" scripts/build_football_roi_indices.py \
    --detector-root "${DETECTOR_ROOT}" \
    --output-root "${INDEX_ROOT}" \
    --split-file "${request_root}/train_requested_video_ids.txt" \
    --split-file "${request_root}/val_requested_video_ids.txt" \
    --sample-fps 2 \
    --ball-track-fps 10 \
    --ball-conf-floor 0.10
}

require_config_indices() {
  local config_path="$1"
  "${PYTHON_BIN}" - <<'PY_CHECK_CONFIG' "${config_path}" "${INDEX_ROOT}"
from pathlib import Path
import sys
import yaml

config_path = Path(sys.argv[1])
index_root = Path(sys.argv[2])
cfg = yaml.safe_load(config_path.read_text())
split_files = ((cfg.get("data") or {}).get("long_video") or {}).get("split_files") or {}
allow_missing = bool((cfg.get("spatial_crop") or {}).get("drop_missing_index_videos", False))
missing = {}
counts = {}
for split in ("train", "val"):
    files = split_files.get(split) or []
    if isinstance(files, (str, Path)):
        files = [files]
    ids = []
    for item in files:
        ids.extend(
            line.strip()
            for line in Path(item).read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    ids = list(dict.fromkeys(ids))
    counts[split] = len(ids)
    missing[split] = [video_id for video_id in ids if not (index_root / f"{video_id}.pt").exists()]

if missing["train"] or missing["val"]:
    print(
        f"Incomplete ROI index for config={config_path}: "
        f"missing_train={len(missing['train'])} missing_val={len(missing['val'])}",
        file=sys.stderr,
    )
    for split in ("train", "val"):
        if missing[split]:
            print(f"missing_{split}_roi_index_ids:", file=sys.stderr)
            for video_id in missing[split]:
                print(video_id, file=sys.stderr)
    if not allow_missing:
        raise SystemExit(2)
    print(
        "WARN spatial_crop.drop_missing_index_videos=true; "
        "training will exclude these videos at runtime.",
        file=sys.stderr,
    )
else:
    print(
        f"ROI index complete for config={config_path} "
        f"train={counts['train']} val={counts['val']}"
    )
PY_CHECK_CONFIG
}

prepare_config_indices() {
  local config_path="$1"
  local config_name
  config_name="$(basename "${config_path}" .yaml)"
  local request_root="outputs/football_roi_index_requests/${config_name}"
  build_action_indices_from_config "${config_path}" "${request_root}"
  require_config_indices "${config_path}"
}

prepare_action_indices_and_split() {
  local config_path="$1"
  build_action_indices_from_config "${config_path}"
  make_action_split_from_config "${config_path}"
}

make_lora_debug_split_from_config() {
  local config_path="$1"
  "${PYTHON_BIN}" - <<'PY_LORA_DEBUG' "${config_path}" "${DEBUG_SPLIT_ROOT}"
from pathlib import Path
import sys
import yaml
config_path = Path(sys.argv[1])
out_root = Path(sys.argv[2])
cfg = yaml.safe_load(config_path.read_text())
split_files = ((cfg.get('data') or {}).get('long_video') or {}).get('split_files') or {}
out_root.mkdir(parents=True, exist_ok=True)
limits = {'train': 4, 'val': 1}
summary = {}
for split in ('train', 'val'):
    files = split_files.get(split) or []
    if isinstance(files, (str, Path)):
        files = [files]
    ids = []
    for item in files:
        path = Path(item)
        ids.extend(
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip() and not line.strip().startswith('#')
        )
    deduped = []
    seen = set()
    for video_id in ids:
        if video_id in seen:
            continue
        seen.add(video_id)
        deduped.append(video_id)
    kept = deduped[:limits[split]]
    if not kept:
        raise SystemExit(f'No {split} videos found from {config_path}')
    (out_root / f'{split}_video_ids.txt').write_text(''.join(f'{video_id}\n' for video_id in kept))
    summary[split] = {'source': len(deduped), 'kept': len(kept)}
print(f"lora debug split={out_root} from_config={config_path} train={summary['train']['kept']}/{summary['train']['source']} val={summary['val']['kept']}/{summary['val']['source']}")
PY_LORA_DEBUG
}

make_action_split_from_config() {
  local config_path="$1"
  "${PYTHON_BIN}" - <<'PY' "${config_path}" "${INDEX_ROOT}" "${ACTION_SPLIT_ROOT}"
from pathlib import Path
import sys
import yaml
config_path = Path(sys.argv[1])
index_root = Path(sys.argv[2])
out_root = Path(sys.argv[3])
cfg = yaml.safe_load(config_path.read_text())
split_files = ((cfg.get('data') or {}).get('long_video') or {}).get('split_files') or {}
indexed = {path.stem for path in index_root.glob('*.pt') if path.name != 'summary.pt'}
out_root.mkdir(parents=True, exist_ok=True)
summary = {}
for split in ('train', 'val'):
    files = split_files.get(split) or []
    if isinstance(files, (str, Path)):
        files = [files]
    ids = []
    for item in files:
        path = Path(item)
        ids.extend(
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip() and not line.strip().startswith('#')
        )
    deduped = []
    seen = set()
    for video_id in ids:
        if video_id in seen:
            continue
        seen.add(video_id)
        deduped.append(video_id)
    kept = [video_id for video_id in deduped if video_id in indexed]
    missing = [video_id for video_id in deduped if video_id not in indexed]
    (out_root / f'{split}_video_ids.txt').write_text(''.join(f'{video_id}\n' for video_id in kept))
    (out_root / f'{split}_missing_roi_index_video_ids.txt').write_text(''.join(f'{video_id}\n' for video_id in missing))
    summary[split] = {'source': len(deduped), 'kept': len(kept), 'missing_roi_index': len(missing)}
if not summary['train']['kept']:
    raise SystemExit(f'No indexed train videos from {config_path} under {index_root}')
if not summary['val']['kept']:
    raise SystemExit(f'No indexed val videos from {config_path} under {index_root}')
print(
    f"action split={out_root} from_config={config_path} "
    f"train={summary['train']['kept']}/{summary['train']['source']} "
    f"val={summary['val']['kept']}/{summary['val']['source']} "
    f"missing={summary['train']['missing_roi_index'] + summary['val']['missing_roi_index']}"
)
for split in ('train', 'val'):
    missing_path = out_root / f'{split}_missing_roi_index_video_ids.txt'
    missing = missing_path.read_text().splitlines()
    if missing:
        print(f"missing_{split}_roi_index_ids_path={missing_path} count={len(missing)}")
        for video_id in missing[:20]:
            print(f"missing_{split}_roi_index_id={video_id}")
        if len(missing) > 20:
            print(f"missing_{split}_roi_index_id=... {len(missing) - 20} more")
PY
}

selected_global_size() {
  if [[ ! -f "${RESOLUTION_JSON}" ]]; then
    echo "Missing resolution decision. Run stage 'resolution' first." >&2
    exit 2
  fi
  "${PYTHON_BIN}" -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["selected_global_image_size"]))' "${RESOLUTION_JSON}"
}

selected_or_override_global_size() {
  if [[ -n "${GLOBAL_IMAGE_SIZE:-}" ]]; then
    echo "${GLOBAL_IMAGE_SIZE}"
  else
    selected_global_size
  fi
}

global_size_override_arg() {
  if [[ -n "${GLOBAL_IMAGE_SIZE:-}" ]]; then
    printf '%s\n' "spatial_crop.global_image_size=${GLOBAL_IMAGE_SIZE}"
  fi
}

dense_eval() {
  local checkpoint="$1"
  local run_name="$2"
  local spatial_mode="${3:-robust_detector_aware}"
  local thresholds="${4:-${DENSE_EVAL_THRESHOLDS:-0.5}}"
  local postprocess="${5:-${DENSE_EVAL_POSTPROCESS:-point_nms}}"
  local match_tolerance_sec="${DENSE_EVAL_MATCH_TOLERANCE_SEC:-5}"
  local nms_radius_sec="${DENSE_EVAL_NMS_RADIUS_SEC:-5}"
  local batch_size="${DENSE_EVAL_BATCH_SIZE:-1}"
  local num_workers="${DENSE_EVAL_NUM_WORKERS:-2}"
  local frame_event_args=()
  if [[ "${DENSE_EVAL_SAVE_FRAME_EVENT_LOGITS:-0}" == "1" ]]; then
    frame_event_args+=(--save-frame-event-logits --frame-event-topk "${DENSE_EVAL_FRAME_EVENT_TOPK:-8}")
  fi
  if [[ "${postprocess}" != "point_nms" && "${postprocess}" != "window_overlap" ]]; then
    echo "DENSE_EVAL_POSTPROCESS must be point_nms or window_overlap, got: ${postprocess}" >&2
    exit 2
  fi
  if [[ "${postprocess}" == "window_overlap" ]]; then
    run_name="${run_name//point_nms/window_overlap}"
  fi
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing checkpoint: ${checkpoint}" >&2
    exit 2
  fi
  "${PYTHON_BIN}" scripts/evaluate_football_model.py \
    --checkpoint "${checkpoint}" \
    --mode dense \
    --video-ids "${VIDEO_IDS}" \
    --gt-dir /home/new_users/qiuqi/code/football_events_human_repair \
    --video-root xbotgo_0608="${VIDEO_ROOT}" \
    --output-root outputs/football_eval_runs \
    --run-name "${run_name}" \
    --clip-sec 10 \
    --stride-sec 5 \
    --batch-size "${batch_size}" \
    --num-workers "${num_workers}" \
    --device cuda:0 \
    --gpu-ids 0 \
    --thresholds "${thresholds}" \
    --prediction-postprocess "${postprocess}" \
    --nms-radius-sec "${nms_radius_sec}" \
    --match-tolerance-sec "${match_tolerance_sec}" \
    --spatial-crop-mode "${spatial_mode}" \
    --detector-index-root "${INDEX_ROOT}" \
    "${frame_event_args[@]}"
}

train_action_config() {
  local config_path="$1"
  shift || true
  local output_dir
  output_dir="$(config_output_dir "${config_path}")"
  require_index
  prepare_action_indices_and_split "${config_path}"
  if [[ ! -f "${ACTION_INIT_CHECKPOINT}" ]]; then
    echo "Missing action experiment init checkpoint: ${ACTION_INIT_CHECKPOINT}" >&2
    echo "Set ACTION_INIT_CHECKPOINT=/path/to/best.pt or INIT_CHECKPOINT=/path/to/best.pt." >&2
    exit 2
  fi
  local args=(
    --config "${config_path}"
    "$(gpu_ids_override_arg)"
    "model.init_checkpoint=${ACTION_INIT_CHECKPOINT}"
    "data.long_video.split_files.train=[${ACTION_SPLIT_ROOT}/train_video_ids.txt]"
    "data.long_video.split_files.val=[${ACTION_SPLIT_ROOT}/val_video_ids.txt]"
  )
  if [[ -n "${GLOBAL_IMAGE_SIZE:-}" ]]; then
    echo "Using overridden global_image_size=${GLOBAL_IMAGE_SIZE}"
    args+=("$(global_size_override_arg)")
  fi
  if [[ -n "${TRAIN_EPOCHS:-}" ]]; then
    args+=("train.epochs=${TRAIN_EPOCHS}")
  fi
  if [[ -n "${PER_GPU_BATCH_SIZE:-}" ]]; then
    args+=("train.per_gpu_batch_size=${PER_GPU_BATCH_SIZE}")
    args+=("eval.per_gpu_batch_size=${EVAL_PER_GPU_BATCH_SIZE:-${PER_GPU_BATCH_SIZE}}")
  elif [[ -n "${EVAL_PER_GPU_BATCH_SIZE:-}" ]]; then
    args+=("eval.per_gpu_batch_size=${EVAL_PER_GPU_BATCH_SIZE}")
  fi
  if [[ -n "${GRAD_ACCUM_STEPS:-}" ]]; then
    args+=("train.grad_accum_steps=${GRAD_ACCUM_STEPS}")
  fi
  if [[ -n "${LR_PER_GPU:-}" ]]; then
    args+=("train.lr_per_gpu=${LR_PER_GPU}")
  fi
  if [[ -n "${BACKBONE_LR_PER_GPU:-}" ]]; then
    args+=("train.backbone_lr_per_gpu=${BACKBONE_LR_PER_GPU}")
  fi
  if [[ -n "${TRAIN_LOG_INTERVAL:-}" ]]; then
    args+=("train.log_interval=${TRAIN_LOG_INTERVAL}")
  fi
  args+=("$@")
  run_train_logged "${config_path}" "${output_dir}" "${PYTHON_BIN}" train_football_events.py "${args[@]}"
}

train_action_debug_config() {
  local config_path="$1"
  shift || true
  local debug_output_dir
  debug_output_dir="${DEBUG_OUTPUT_DIR:-outputs/football_events/$(basename "${config_path}" .yaml)_debug}"
  require_index
  make_debug_split
  if [[ ! -f "${ACTION_INIT_CHECKPOINT}" ]]; then
    echo "Missing action experiment init checkpoint: ${ACTION_INIT_CHECKPOINT}" >&2
    echo "Set ACTION_INIT_CHECKPOINT=/path/to/best.pt or INIT_CHECKPOINT=/path/to/best.pt." >&2
    exit 2
  fi
  local args=(
    --config "${config_path}"
    "output_dir=${debug_output_dir}"
    "$(gpu_ids_override_arg)"
    "model.init_checkpoint=${ACTION_INIT_CHECKPOINT}"
    "data.long_video.split_files.train=[${DEBUG_SPLIT_ROOT}/train_video_ids.txt]"
    "data.long_video.split_files.val=[${DEBUG_SPLIT_ROOT}/val_video_ids.txt]"
    "train.epochs=${DEBUG_EPOCHS:-1}"
    "train.log_interval=1"
  )
  if [[ -n "${GLOBAL_IMAGE_SIZE:-}" ]]; then
    echo "Using overridden global_image_size=${GLOBAL_IMAGE_SIZE}"
    args+=("$(global_size_override_arg)")
  fi
  args+=("$@")
  run_train_logged "${config_path}" "${debug_output_dir}" "${PYTHON_BIN}" train_football_events.py "${args[@]}"
}

train_lora_r5_f32_action_config() {
  local config_path="$1"
  shift || true
  local output_dir
  output_dir="$(config_output_dir "${config_path}")"
  if [[ ! -f "${LORA_R5_F32_INIT_CHECKPOINT}" ]]; then
    echo "Missing lora_r5_f32 init checkpoint: ${LORA_R5_F32_INIT_CHECKPOINT}" >&2
    echo "Set LORA_R5_F32_INIT_CHECKPOINT=/path/to/last.pt or INIT_CHECKPOINT=/path/to/last.pt." >&2
    exit 2
  fi
  local args=(
    --config "${config_path}"
    "$(gpu_ids_override_arg)"
    "model.init_checkpoint=${LORA_R5_F32_INIT_CHECKPOINT}"
  )
  args+=("$@")
  run_train_logged "${config_path}" "${output_dir}" "${PYTHON_BIN}" train_football_events.py "${args[@]}"
}

train_lora_r5_f32_action_debug_config() {
  local config_path="$1"
  shift || true
  local debug_output_dir
  debug_output_dir="${DEBUG_OUTPUT_DIR:-outputs/football_events/$(basename "${config_path}" .yaml)_debug}"
  make_lora_debug_split_from_config "${config_path}"
  if [[ ! -f "${LORA_R5_F32_INIT_CHECKPOINT}" ]]; then
    echo "Missing lora_r5_f32 init checkpoint: ${LORA_R5_F32_INIT_CHECKPOINT}" >&2
    echo "Set LORA_R5_F32_INIT_CHECKPOINT=/path/to/last.pt or INIT_CHECKPOINT=/path/to/last.pt." >&2
    exit 2
  fi
  local args=(
    --config "${config_path}"
    "output_dir=${debug_output_dir}"
    "$(gpu_ids_override_arg)"
    "model.init_checkpoint=${LORA_R5_F32_INIT_CHECKPOINT}"
    "data.long_video.split_files.train=[${DEBUG_SPLIT_ROOT}/train_video_ids.txt]"
    "data.long_video.split_files.val=[${DEBUG_SPLIT_ROOT}/val_video_ids.txt]"
    "train.epochs=${DEBUG_EPOCHS:-1}"
    "train.log_interval=1"
  )
  args+=("$@")
  run_train_logged "${config_path}" "${debug_output_dir}" "${PYTHON_BIN}" train_football_events.py "${args[@]}"
}

mine_e1_hard_negatives() {
  require_index
  prepare_action_indices_and_split "${E1_HARD_NEG_CONFIG}"
  if [[ ! -f "${E1_HARD_NEG_INIT_CHECKPOINT}" ]]; then
    echo "Missing E1 hard-negative mining checkpoint: ${E1_HARD_NEG_INIT_CHECKPOINT}" >&2
    echo "Set E1_HARD_NEG_INIT_CHECKPOINT=/path/to/best.pt or INIT_CHECKPOINT=/path/to/best.pt." >&2
    exit 2
  fi
  local run_name="e1_exp2_best_train_hard_negative_mining"
  "${PYTHON_BIN}" scripts/evaluate_football_model.py \
    --checkpoint "${E1_HARD_NEG_INIT_CHECKPOINT}" \
    --mode dense \
    --video-id-file "${ACTION_SPLIT_ROOT}/train_video_ids.txt" \
    --gt-dir /home/new_users/qiuqi/code/football_events_human_repair \
    --video-root xbotgo_0608="${VIDEO_ROOT}" \
    --output-root outputs/football_eval_runs \
    --run-name "${run_name}" \
    --clip-sec 10 \
    --stride-sec 5 \
    --batch-size "${E1_HARD_NEG_MINE_BATCH_SIZE:-8}" \
    --num-workers "${E1_HARD_NEG_MINE_NUM_WORKERS:-8}" \
    --device cuda:0 \
    --gpu-ids "$(visible_gpu_ids_csv)" \
    --thresholds checkpoint \
    --prediction-postprocess window_overlap \
    --match-tolerance-sec 5 \
    --spatial-crop-mode robust_detector_aware \
    --detector-index-root "${INDEX_ROOT}"
  local reject_any_args=()
  if [[ "${E1_HARD_NEG_REJECT_ANY_LABEL_GT}" == "1" ]]; then
    reject_any_args+=(--reject-any-label-gt)
  fi
  "${PYTHON_BIN}" scripts/build_football_hard_negative_manifest.py \
    --eval-run-dir "outputs/football_eval_runs/${run_name}" \
    --output "${E1_HARD_NEG_MANIFEST}" \
    --source xbotgo_0608 \
    --min-probs "${E1_HARD_NEG_MIN_PROBS}" \
    --tolerance-sec "${E1_HARD_NEG_REJECT_TOLERANCE_SEC:-5}" \
    --max-per-video-per-label "${E1_HARD_NEG_MAX_PER_VIDEO_PER_LABEL:-40}" \
    "${reject_any_args[@]}"
}

train_e1_hard_negative_config() {
  if [[ ! -f "${E1_HARD_NEG_MANIFEST}" ]]; then
    mine_e1_hard_negatives
  fi
  ACTION_INIT_CHECKPOINT="${E1_HARD_NEG_INIT_CHECKPOINT}" \
    train_action_config "${E1_HARD_NEG_CONFIG}" \
      "data.long_video.hard_negative.manifest=${E1_HARD_NEG_MANIFEST}"
}

case "${STAGE}" in
  index)
    "${PYTHON_BIN}" scripts/build_football_roi_indices.py \
      --detector-root "${DETECTOR_ROOT}" \
      --output-root "${INDEX_ROOT}" \
      --split-file "${SPLIT_ROOT}/train_video_ids.txt" \
      --split-file "${SPLIT_ROOT}/val_video_ids.txt" \
      --sample-fps 2 \
      --ball-track-fps 10 \
      --ball-conf-floor 0.10
    ;;
  make_debug_split)
    require_index
    make_debug_split
    ;;
  roi_crops)
    require_index
    "${PYTHON_BIN}" scripts/export_football_roi_crops.py \
      --index-root "${INDEX_ROOT}" \
      --video-root "${VIDEO_ROOT}" \
      --video-id-file "${SPLIT_ROOT}/val_video_ids.txt" \
      --output-dir outputs/football_roi_crops/exp4_val \
      --clip-sec 10 \
      --stride-sec 10 \
      --input-size 384,640
    ;;
  roi_crops_debug)
    require_index
    make_debug_split
    "${PYTHON_BIN}" scripts/export_football_roi_crops.py \
      --index-root "${INDEX_ROOT}" \
      --video-root "${VIDEO_ROOT}" \
      --video-id-file "${DEBUG_SPLIT_ROOT}/val_video_ids.txt" \
      --output-dir outputs/football_roi_crops/exp4_debug_val \
      --clip-sec 10 \
      --stride-sec 10 \
      --input-size 384,640
    ;;
  val_positive_roi_videos)
    require_index
    "${PYTHON_BIN}" scripts/export_val_positive_roi_videos.py \
      --config "${ROI_DEBUG_CONFIG}" \
      --output-dir "${ROI_DEBUG_OUTPUT}" \
      --labels "${ROI_DEBUG_LABELS}" \
      --video-ids "${ROI_DEBUG_VIDEO_IDS}" \
      --max-samples "${ROI_DEBUG_MAX_SAMPLES}" \
      --max-per-video "${ROI_DEBUG_MAX_PER_VIDEO}" \
      --output-fps "${ROI_DEBUG_OUTPUT_FPS}" \
      --frame-mode "${ROI_DEBUG_FRAME_MODE}" \
      --layout "${ROI_DEBUG_LAYOUT}"
    ;;
  gt_roi_compare)
    require_index
    for roi_variant in fixed dynamic; do
      if [[ "${roi_variant}" == "fixed" ]]; then
        roi_config="${DYNAMIC_ROI_D0_CONFIG}"
      else
        roi_config="${DYNAMIC_ROI_D1_CONFIG}"
      fi
      "${PYTHON_BIN}" scripts/export_val_positive_roi_videos.py \
        --config "${roi_config}" \
        --output-dir "${ROI_COMPARE_OUTPUT}/${roi_variant}" \
        --labels "${ROI_COMPARE_LABELS}" \
        --video-ids "${ROI_COMPARE_VIDEO_IDS}" \
        --max-samples "${ROI_COMPARE_MAX_SAMPLES}" \
        --max-per-video "${ROI_COMPARE_MAX_PER_VIDEO}" \
        --output-fps "${ROI_COMPARE_OUTPUT_FPS}" \
        --frame-mode model \
        --layout both
    done
    "${PYTHON_BIN}" scripts/stack_roi_review_videos.py \
      --fixed-root "${ROI_COMPARE_OUTPUT}/fixed" \
      --dynamic-root "${ROI_COMPARE_OUTPUT}/dynamic" \
      --output-dir "${ROI_COMPARE_OUTPUT}/comparison" \
      --force
    ;;
  resolution)
    "${PYTHON_BIN}" scripts/check_football_global_resolution.py \
      --config configs/football/dinov3_vitl16_robust_dual_exp2.yaml \
      --checkpoint /mnt/data_16t/football/qiuqi/checkpoints/lora_r2_best.pt \
      --reference-size 384,640 \
      --candidate-size 192,320 \
      --max-map-drop 0.01 \
      --max-recall-drop 0.01 \
      --max-samples 1024 \
      --device cuda:0 \
      --output "${RESOLUTION_JSON}"
    ;;
  roi_audit)
    require_index
    "${PYTHON_BIN}" scripts/audit_football_robust_roi.py \
      --index-root "${INDEX_ROOT}" \
      --video-id-file "${SPLIT_ROOT}/val_video_ids.txt" \
      --output-dir outputs/football_roi_audit/robust_v3_val11 \
      --clip-sec 10 \
      --stride-sec 5
    ;;
  baseline_global)
    dense_eval \
      /mnt/data_16t/football/qiuqi/checkpoints/lora_r2_last.pt \
      lora_r2_detector47_baseline_global_thr0p5_nms5_tol5 \
      none
    ;;
  baseline_legacy)
    require_index
    dense_eval \
      /mnt/data_16t/football/qiuqi/checkpoints/lora_r2_best.pt \
      lora_r2_detector47_baseline_legacy_indexed_thr0p5_nms5_tol5 \
      legacy_indexed
    ;;
  baseline_robust)
    require_index
    dense_eval \
      /mnt/data_16t/football/qiuqi/checkpoints/lora_r2_best.pt \
      lora_r2_detector47_baseline_robust_crop_thr0p5_nms5_tol5 \
      robust_detector_aware
    ;;
  baseline_summary)
    "${PYTHON_BIN}" scripts/summarize_football_detection_aware.py
    ;;
  train_global)
    GLOBAL_SIZE="$(selected_or_override_global_size)"
    echo "Using global image_size=${GLOBAL_SIZE} from ${RESOLUTION_JSON}"
    "${PYTHON_BIN}" train_football_events.py \
      --config configs/football/dinov3_vitl16_detector47_global_control.yaml \
      "video.image_size=${GLOBAL_SIZE}"
    ;;
  eval_global)
    dense_eval \
      outputs/football_events/vitl16_detector47_global_control/best.pt \
      vitl16_detector47_global_control_dense_thr0p5_nms5_tol5 \
      none
    ;;
  train_gate)
    require_index
    GLOBAL_SIZE="$(selected_or_override_global_size)"
    echo "Using global_image_size=${GLOBAL_SIZE} from ${RESOLUTION_JSON}"
    "${PYTHON_BIN}" train_football_events.py \
      --config configs/football/dinov3_vitl16_robust_dual_exp2.yaml \
      "spatial_crop.global_image_size=${GLOBAL_SIZE}"
    ;;
  eval_gate)
    require_index
    dense_eval \
      outputs/football_events/vitl16_robust_dual_exp2/best.pt \
      vitl16_detector47_robust_gate_dense_thr0p5_nms5_tol5 \
      robust_detector_aware
    ;;
  exp3)
    require_index
    if [[ ! -f outputs/football_events/vitl16_robust_dual_exp2/best.pt ]]; then
      echo "Experiment 3 must start from experiment 2 best.pt." >&2
      exit 2
    fi
    GLOBAL_SIZE="$(selected_or_override_global_size)"
    echo "Using global_image_size=${GLOBAL_SIZE} from ${RESOLUTION_JSON}"
    "${PYTHON_BIN}" train_football_events.py \
      --config configs/football/dinov3_vitl16_robust_dual_precision_exp3.yaml \
      "spatial_crop.global_image_size=${GLOBAL_SIZE}"
    ;;
  eval3)
    require_index
    dense_eval \
      outputs/football_events/vitl16_robust_dual_precision_exp3/best.pt \
      vitl16_robust_dual_precision_exp3_dense_thr0p5_nms5_tol5
    ;;
  exp4_16f)
    prepare_config_indices configs/football/dinov3_vitl16_robust_dual_16f_exp4.yaml
    if [[ ! -f "${EXP4_INIT_CHECKPOINT}" ]]; then
      echo "Missing init checkpoint for exp4_16f: ${EXP4_INIT_CHECKPOINT}" >&2
      echo "Set INIT_CHECKPOINT=/path/to/best.pt if you want to use another checkpoint." >&2
      exit 2
    fi
    if [[ -n "${GLOBAL_IMAGE_SIZE:-}" ]]; then
      echo "Using overridden global_image_size=${GLOBAL_IMAGE_SIZE}"
      "${PYTHON_BIN}" train_football_events.py \
        --config configs/football/dinov3_vitl16_robust_dual_16f_exp4.yaml \
        "$(gpu_ids_override_arg)" \
        "model.init_checkpoint=${EXP4_INIT_CHECKPOINT}" \
        "$(global_size_override_arg)"
    else
      echo "Using global_image_size from configs/football/dinov3_vitl16_robust_dual_16f_exp4.yaml"
      "${PYTHON_BIN}" train_football_events.py \
        --config configs/football/dinov3_vitl16_robust_dual_16f_exp4.yaml \
        "$(gpu_ids_override_arg)" \
        "model.init_checkpoint=${EXP4_INIT_CHECKPOINT}"
    fi
    ;;
  exp4_16f_debug)
    prepare_config_indices configs/football/dinov3_vitl16_robust_dual_16f_exp4.yaml
    make_lora_debug_split_from_config configs/football/dinov3_vitl16_robust_dual_16f_exp4.yaml
    if [[ ! -f "${EXP4_INIT_CHECKPOINT}" ]]; then
      echo "Missing init checkpoint for exp4_16f_debug: ${EXP4_INIT_CHECKPOINT}" >&2
      echo "Set INIT_CHECKPOINT=/path/to/best.pt if you want to use another checkpoint." >&2
      exit 2
    fi
    echo "Using debug split from ${DEBUG_SPLIT_ROOT}"
    "${PYTHON_BIN}" train_football_events.py \
      --config configs/football/dinov3_vitl16_robust_dual_16f_exp4.yaml \
      "$(gpu_ids_override_arg)" \
      "model.init_checkpoint=${EXP4_INIT_CHECKPOINT}" \
      "data.long_video.split_files.train=[${DEBUG_SPLIT_ROOT}/train_video_ids.txt]" \
      "data.long_video.split_files.val=[${DEBUG_SPLIT_ROOT}/val_video_ids.txt]" \
      "train.epochs=${DEBUG_EPOCHS:-1}" \
      "train.log_interval=1"
    ;;
  exp4_dual_lora)
    prepare_config_indices "${EXP4_DUAL_LORA_CONFIG}"
    if [[ ! -f "${EXP4_DUAL_LORA_INIT_CHECKPOINT}" ]]; then
      echo "Missing init checkpoint for exp4_dual_lora: ${EXP4_DUAL_LORA_INIT_CHECKPOINT}" >&2
      echo "Set INIT_CHECKPOINT=/path/to/best.pt to select another initialization checkpoint." >&2
      exit 2
    fi
    "${PYTHON_BIN}" train_football_events.py \
      --config "${EXP4_DUAL_LORA_CONFIG}" \
      "$(gpu_ids_override_arg)" \
      "model.init_checkpoint=${EXP4_DUAL_LORA_INIT_CHECKPOINT}" \
      "train.per_gpu_batch_size=${EXP4_DUAL_LORA_PER_GPU_BATCH_SIZE}" \
      "train.grad_accum_steps=${EXP4_DUAL_LORA_GRAD_ACCUM_STEPS}" \
      "train.lr_per_gpu=${EXP4_DUAL_LORA_LR_PER_GPU}" \
      "train.backbone_lr_per_gpu=${EXP4_DUAL_LORA_BACKBONE_LR_PER_GPU}"
    ;;
  exp4_dual_lora_debug)
    prepare_config_indices "${EXP4_DUAL_LORA_CONFIG}"
    make_lora_debug_split_from_config "${EXP4_DUAL_LORA_CONFIG}"
    if [[ ! -f "${EXP4_DUAL_LORA_INIT_CHECKPOINT}" ]]; then
      echo "Missing init checkpoint for exp4_dual_lora_debug: ${EXP4_DUAL_LORA_INIT_CHECKPOINT}" >&2
      echo "Set INIT_CHECKPOINT=/path/to/best.pt to select another initialization checkpoint." >&2
      exit 2
    fi
    "${PYTHON_BIN}" train_football_events.py \
      --config "${EXP4_DUAL_LORA_CONFIG}" \
      "$(gpu_ids_override_arg)" \
      "model.init_checkpoint=${EXP4_DUAL_LORA_INIT_CHECKPOINT}" \
      "data.long_video.split_files.train=[${DEBUG_SPLIT_ROOT}/train_video_ids.txt]" \
      "data.long_video.split_files.val=[${DEBUG_SPLIT_ROOT}/val_video_ids.txt]" \
      "output_dir=outputs/football_events/vitl16_robust_dual_16f_exp4_lora_debug" \
      "train.per_gpu_batch_size=${EXP4_DUAL_LORA_PER_GPU_BATCH_SIZE}" \
      "train.grad_accum_steps=${EXP4_DUAL_LORA_GRAD_ACCUM_STEPS}" \
      "train.lr_per_gpu=${EXP4_DUAL_LORA_LR_PER_GPU}" \
      "train.backbone_lr_per_gpu=${EXP4_DUAL_LORA_BACKBONE_LR_PER_GPU}" \
      "train.epochs=${DEBUG_EPOCHS:-1}" \
      "train.log_interval=1"
    ;;
  eval_exp4_dual_lora)
    require_index
    dense_eval \
      outputs/football_events/vitl16_robust_dual_16f_exp4_lora/best.pt \
      vitl16_robust_dual_16f_exp4_lora_dense_thr0p5_nms5_tol5
    ;;
  eval4_16f)
    require_index
    dense_eval \
      outputs/football_events/vitl16_robust_dual_16f_exp4/best.pt \
      vitl16_robust_dual_16f_exp4_dense_thr0p5_nms5_tol5
    ;;
  hr_single_lora)
    train_lora_r5_f32_action_config "${HR_SINGLE_LORA_CONFIG}"
    ;;
  debug_hr_single_lora)
    train_lora_r5_f32_action_debug_config "${HR_SINGLE_LORA_CONFIG}"
    ;;
  eval_hr_single_lora)
    dense_eval \
      outputs/football_events/vitl16_lora_r5_f32_16f_hr_single/best.pt \
      vitl16_lora_r5_f32_16f_hr_single_point_nms_checkpoint_thr \
      none \
      checkpoint
    ;;
  hr_single_lora_frame_det)
    train_lora_r5_f32_action_config "${HR_SINGLE_LORA_FRAME_DET_CONFIG}"
    ;;
  debug_hr_single_lora_frame_det)
    train_lora_r5_f32_action_debug_config "${HR_SINGLE_LORA_FRAME_DET_CONFIG}"
    ;;
  eval_hr_single_lora_frame_det)
    DENSE_EVAL_SAVE_FRAME_EVENT_LOGITS=1 dense_eval \
      outputs/football_events/vitl16_lora_r5_f32_16f_hr_single_frame_det/best.pt \
      vitl16_lora_r5_f32_16f_hr_single_frame_det_point_nms_checkpoint_thr \
      none \
      checkpoint
    ;;
  e1_lora_r5_f32)
    train_lora_r5_f32_action_config configs/football/dinov3_vitl16_lora_r5_f32_16f_e1_frame_det.yaml
    ;;
  debug_e1_lora_r5_f32)
    train_lora_r5_f32_action_debug_config configs/football/dinov3_vitl16_lora_r5_f32_16f_e1_frame_det.yaml
    ;;
  eval_e1_lora_r5_f32)
    dense_eval \
      outputs/football_events/vitl16_lora_r5_f32_16f_e1_frame_det/best.pt \
      vitl16_lora_r5_f32_16f_e1_frame_det_point_nms_checkpoint_thr \
      none \
      checkpoint
    ;;
  d0_fixed_roi)
    ACTION_INIT_CHECKPOINT="${DYNAMIC_ROI_D0_INIT_CHECKPOINT}" \
      train_action_config "${DYNAMIC_ROI_D0_CONFIG}"
    ;;
  debug_d0_fixed_roi)
    ACTION_INIT_CHECKPOINT="${DYNAMIC_ROI_D0_INIT_CHECKPOINT}" \
      train_action_debug_config "${DYNAMIC_ROI_D0_CONFIG}"
    ;;
  eval_d0_fixed_roi)
    require_index
    dense_eval \
      outputs/football_events/vitl16_robust_dual_16f_e1_fixed_roi_d0/best.pt \
      vitl16_robust_dual_16f_e1_fixed_roi_d0_point_nms_checkpoint_thr \
      robust_detector_aware \
      checkpoint
    ;;
  d1_dynamic_roi)
    ACTION_INIT_CHECKPOINT="${DYNAMIC_ROI_D1_INIT_CHECKPOINT}" \
      train_action_config "${DYNAMIC_ROI_D1_CONFIG}"
    ;;
  d1_dynamic_roi_a800)
    ACTION_INIT_CHECKPOINT="${DYNAMIC_ROI_D1_INIT_CHECKPOINT}" \
      train_action_config "${DYNAMIC_ROI_D1_A800_COLOCATED_CONFIG}"
    ;;
  debug_d1_dynamic_roi)
    ACTION_INIT_CHECKPOINT="${DYNAMIC_ROI_D1_INIT_CHECKPOINT}" \
      train_action_debug_config "${DYNAMIC_ROI_D1_CONFIG}"
    ;;
  eval_d1_dynamic_roi)
    require_index
    dense_eval \
      outputs/football_events/vitl16_robust_dual_16f_e1_dynamic_roi_d1/best.pt \
      vitl16_robust_dual_16f_e1_dynamic_roi_d1_point_nms_checkpoint_thr \
      robust_detector_aware \
      checkpoint
    ;;
  d2_feature_quality)
    ACTION_INIT_CHECKPOINT="${DYNAMIC_ROI_D2_INIT_CHECKPOINT}" \
      train_action_config "${DYNAMIC_ROI_D2_CONFIG}"
    ;;
  d2_feature_quality_a800)
    ACTION_INIT_CHECKPOINT="${DYNAMIC_ROI_D2_INIT_CHECKPOINT}" \
      train_action_config "${DYNAMIC_ROI_D2_A800_CONFIG}"
    ;;
  d3_dynamic_roi_lora)
    ACTION_INIT_CHECKPOINT="${DYNAMIC_ROI_D3_LORA_INIT_CHECKPOINT}" \
      train_action_config "${DYNAMIC_ROI_D3_LORA_CONFIG}"
    ;;
  debug_d3_dynamic_roi_lora)
    ACTION_INIT_CHECKPOINT="${DYNAMIC_ROI_D3_LORA_INIT_CHECKPOINT}" \
      train_action_debug_config "${DYNAMIC_ROI_D3_LORA_CONFIG}"
    ;;
  eval_d3_dynamic_roi_lora)
    require_index
    dense_eval \
      outputs/football_events/vitl16_robust_dual_16f_e1_dynamic_roi_lora_d3/best.pt \
      vitl16_robust_dual_16f_e1_dynamic_roi_lora_d3_point_nms_checkpoint_thr \
      robust_detector_aware \
      checkpoint
    ;;
  d4_dynamic_roi_decoupled_lora)
    ACTION_INIT_CHECKPOINT="${DYNAMIC_ROI_D4_DECOUPLED_LORA_INIT_CHECKPOINT}" \
      train_action_config "${DYNAMIC_ROI_D4_DECOUPLED_LORA_CONFIG}"
    ;;
  debug_d4_dynamic_roi_decoupled_lora)
    ACTION_INIT_CHECKPOINT="${DYNAMIC_ROI_D4_DECOUPLED_LORA_INIT_CHECKPOINT}" \
      train_action_debug_config "${DYNAMIC_ROI_D4_DECOUPLED_LORA_CONFIG}"
    ;;
  eval_d4_dynamic_roi_decoupled_lora)
    require_index
    dense_eval \
      outputs/football_events/vitl16_robust_dual_16f_e1_dynamic_roi_decoupled_lora_d4/best.pt \
      vitl16_robust_dual_16f_e1_dynamic_roi_decoupled_lora_d4_point_nms_checkpoint_thr \
      robust_detector_aware \
      checkpoint
    ;;
  debug_d2_feature_quality)
    local_init="${DYNAMIC_ROI_D2_INIT_CHECKPOINT}"
    if [[ ! -f "${local_init}" ]]; then
      local_init="${DYNAMIC_ROI_D1_INIT_CHECKPOINT}"
      echo "D1 best.pt not found; debug D2 falls back to ${local_init}"
    fi
    ACTION_INIT_CHECKPOINT="${local_init}" \
      train_action_debug_config "${DYNAMIC_ROI_D2_CONFIG}"
    ;;
  eval_d2_feature_quality)
    require_index
    dense_eval \
      outputs/football_events/vitl16_robust_dual_16f_e1_dynamic_roi_feature_quality_d2/best.pt \
      vitl16_robust_dual_16f_e1_dynamic_roi_feature_quality_d2_point_nms_checkpoint_thr \
      robust_detector_aware \
      checkpoint
    ;;
  e1_frame_det)
    train_action_config configs/football/dinov3_vitl16_robust_dual_16f_e1_frame_det.yaml
    ;;
  debug_e1_frame_det)
    train_action_debug_config configs/football/dinov3_vitl16_robust_dual_16f_e1_frame_det.yaml
    ;;
  eval_e1_frame_det)
    require_index
    dense_eval \
      outputs/football_events/vitl16_robust_dual_16f_e1_frame_det/best.pt \
      vitl16_robust_dual_16f_e1_frame_det_point_nms_checkpoint_thr \
      robust_detector_aware \
      checkpoint
    ;;
  mine_e1_hard_neg)
    mine_e1_hard_negatives
    ;;
  e1_hard_neg)
    train_e1_hard_negative_config
    ;;
  debug_e1_hard_neg)
    ACTION_INIT_CHECKPOINT="${E1_HARD_NEG_INIT_CHECKPOINT}" \
      train_action_debug_config "${E1_HARD_NEG_CONFIG}" \
        "data.long_video.hard_negative.enabled=false"
    ;;
  eval_e1_hard_neg)
    require_index
    dense_eval \
      outputs/football_events/vitl16_robust_dual_16f_e1_frame_det_hard_neg_exp1/best.pt \
      vitl16_robust_dual_16f_e1_hard_neg_exp1_point_nms_checkpoint_thr \
      robust_detector_aware \
      checkpoint
    ;;
  e2_lora_r5_f32)
    train_lora_r5_f32_action_config configs/football/dinov3_vitl16_lora_r5_f32_16f_e2_attn_pool.yaml
    ;;
  debug_e2_lora_r5_f32)
    train_lora_r5_f32_action_debug_config configs/football/dinov3_vitl16_lora_r5_f32_16f_e2_attn_pool.yaml
    ;;
  eval_e2_lora_r5_f32)
    dense_eval \
      outputs/football_events/vitl16_lora_r5_f32_16f_e2_attn_pool/best.pt \
      vitl16_lora_r5_f32_16f_e2_attn_pool_point_nms_checkpoint_thr \
      none \
      checkpoint
    ;;
  e2_attn_pool)
    train_action_config configs/football/dinov3_vitl16_robust_dual_16f_e2_attn_pool.yaml
    ;;
  debug_e2_attn_pool)
    train_action_debug_config configs/football/dinov3_vitl16_robust_dual_16f_e2_attn_pool.yaml
    ;;
  eval_e2_attn_pool)
    require_index
    dense_eval \
      outputs/football_events/vitl16_robust_dual_16f_e2_attn_pool/best.pt \
      vitl16_robust_dual_16f_e2_attn_pool_point_nms_checkpoint_thr \
      robust_detector_aware \
      checkpoint
    ;;
  e3_lora_r5_f32)
    train_lora_r5_f32_action_config configs/football/dinov3_vitl16_lora_r5_f32_16f_e3_class_query.yaml
    ;;
  debug_e3_lora_r5_f32)
    train_lora_r5_f32_action_debug_config configs/football/dinov3_vitl16_lora_r5_f32_16f_e3_class_query.yaml
    ;;
  eval_e3_lora_r5_f32)
    dense_eval \
      outputs/football_events/vitl16_lora_r5_f32_16f_e3_class_query/best.pt \
      vitl16_lora_r5_f32_16f_e3_class_query_point_nms_checkpoint_thr \
      none \
      checkpoint
    ;;
  e3_class_query)
    train_action_config configs/football/dinov3_vitl16_robust_dual_16f_e3_class_query.yaml
    ;;
  debug_e3_class_query)
    train_action_debug_config configs/football/dinov3_vitl16_robust_dual_16f_e3_class_query.yaml
    ;;
  eval_e3_class_query)
    require_index
    dense_eval \
      outputs/football_events/vitl16_robust_dual_16f_e3_class_query/best.pt \
      vitl16_robust_dual_16f_e3_class_query_point_nms_checkpoint_thr \
      robust_detector_aware \
      checkpoint
    ;;
  e4_lora_r5_f32)
    train_lora_r5_f32_action_config configs/football/dinov3_vitl16_lora_r5_f32_16f_e4_event_topk.yaml
    ;;
  debug_e4_lora_r5_f32)
    train_lora_r5_f32_action_debug_config configs/football/dinov3_vitl16_lora_r5_f32_16f_e4_event_topk.yaml
    ;;
  eval_e4_lora_r5_f32)
    dense_eval \
      outputs/football_events/vitl16_lora_r5_f32_16f_e4_event_topk/best.pt \
      vitl16_lora_r5_f32_16f_e4_event_topk_point_nms_checkpoint_thr \
      none \
      checkpoint
    ;;
  e4_event_topk)
    train_action_config configs/football/dinov3_vitl16_robust_dual_16f_e4_event_topk.yaml
    ;;
  debug_e4_event_topk)
    train_action_debug_config configs/football/dinov3_vitl16_robust_dual_16f_e4_event_topk.yaml
    ;;
  eval_e4_event_topk)
    require_index
    dense_eval \
      outputs/football_events/vitl16_robust_dual_16f_e4_event_topk/best.pt \
      vitl16_robust_dual_16f_e4_event_topk_point_nms_checkpoint_thr \
      robust_detector_aware \
      checkpoint
    ;;
  *)
    usage
    exit 2
    ;;
esac
