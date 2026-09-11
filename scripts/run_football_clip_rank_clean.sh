#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MODE="${1:-help}"
GPU_LIST="${2:-2,3}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="${CONFIG:-configs/football/dinov3_vitl16_lora_r5_f32_16f_hr_single_frame_det_clip_rank_clean.yaml}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/football_events/vitl16_e1_clip_rank_clean_lora}"
MANIFEST="${MANIFEST:-outputs/football_hard_negatives/pointnms_fp_clean_shot_save.json}"
MINING_RUN_NAME="${MINING_RUN_NAME:-vitl16_e1_pointnms_train_fp_mining}"
MINING_RUN="${MINING_RUN:-outputs/football_eval_runs/${MINING_RUN_NAME}}"
TRAIN_IDS="${TRAIN_IDS:-configs/football/splits/vitl16_lvd1689m_set_piece_seed42_162videos/train_video_ids.txt}"
REVIEWED_TRAIN_IDS="${REVIEWED_TRAIN_IDS:-${TRAIN_IDS}}"
VIDEO_IDS="${VIDEO_IDS:-2027564888428580866,2027572095412195330,2027572406738604033,2042125172076392450,2042520971893485569,2042525152494694401}"
GT_DIR="${GT_DIR:-/home/new_users/qiuqi/code/football_events_human_repair}"
VIDEO_ROOT="${VIDEO_ROOT:-/mnt/data_16t/football/raw_video_720P}"
PER_GPU_BATCH_SIZE="${PER_GPU_BATCH_SIZE:-4}"
TARGET_EFFECTIVE_BATCH_SIZE="${TARGET_EFFECTIVE_BATCH_SIZE:-80}"
TARGET_HEAD_LR="${TARGET_HEAD_LR:-0.0003}"
TARGET_BACKBONE_LR="${TARGET_BACKBONE_LR:-0.00003}"
EPOCHS="${EPOCHS:-4}"
NUM_WORKERS_PER_GPU="${NUM_WORKERS_PER_GPU:-2}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-1}"
MINING_BATCH_SIZE="${MINING_BATCH_SIZE:-1}"
MINING_NUM_WORKERS="${MINING_NUM_WORKERS:-2}"
EVAL_BATCH_PER_GPU="${EVAL_BATCH_PER_GPU:-4}"
EVAL_NUM_WORKERS_PER_GPU="${EVAL_NUM_WORKERS_PER_GPU:-2}"

comma_count() {
  "${PYTHON_BIN}" -c 'import sys; print(len([x for x in sys.argv[1].split(",") if x.strip()]))' "$1"
}

local_gpu_ids() {
  "${PYTHON_BIN}" -c 'import sys; n=int(sys.argv[1]); print("[" + ",".join(map(str, range(n))) + "]")' "$1"
}

local_gpu_ids_csv() {
  "${PYTHON_BIN}" -c 'import sys; n=int(sys.argv[1]); print(",".join(map(str, range(n))))' "$1"
}

grad_accum_for() {
  "${PYTHON_BIN}" -c 'import math,sys; print(max(math.ceil(int(sys.argv[1])/(int(sys.argv[2])*int(sys.argv[3]))),1))' "$1" "$2" "$3"
}

lr_per_gpu_for() {
  "${PYTHON_BIN}" -c 'import sys; print(float(sys.argv[1])/int(sys.argv[2]))' "$1" "$2"
}

check_init() {
  if [[ ! -s "${INIT_CHECKPOINT}" ]]; then
    echo "Missing init checkpoint: ${INIT_CHECKPOINT}" >&2
    exit 2
  fi
}

mine_pointnms() {
  check_init
  local gpu="${MINING_GPU:-${GPU_LIST%%,*}}"
  local force_args=()
  if [[ "${MINE_FORCE:-0}" == "1" ]]; then
    force_args=(--force)
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" scripts/evaluate_football_model.py \
    --checkpoint "${INIT_CHECKPOINT}" \
    --mode dense \
    --video-id-file "${TRAIN_IDS}" \
    --gt-dir "${GT_DIR}" \
    --video-root "xbotgo_0608=${VIDEO_ROOT}" \
    --output-root outputs/football_eval_runs \
    --run-name "${MINING_RUN_NAME}" \
    --clip-sec 10 \
    --stride-sec 5 \
    --batch-size "${MINING_BATCH_SIZE}" \
    --num-workers "${MINING_NUM_WORKERS}" \
    --device cuda:0 \
    --gpu-ids 0 \
    --thresholds checkpoint \
    --prediction-postprocess point_nms \
    --nms-radius-sec "${NMS_RADIUS_SEC:-5}" \
    --match-tolerance-sec "${MATCH_TOLERANCE_SEC:-5}" \
    "${force_args[@]}"
}

build_manifest() {
  "${PYTHON_BIN}" scripts/build_pointnms_hard_negatives.py \
    --eval-run-dir "${MINING_RUN}" \
    --output "${MANIFEST}" \
    --reviewed-video-ids "${REVIEWED_TRAIN_IDS}" \
    --labels "${HARDNEG_LABELS:-shot,save}" \
    --min-scores "${HARDNEG_MIN_SCORES:-shot=0.25,save=0.30}" \
    --max-scores "${HARDNEG_MAX_SCORES:-shot=1.0,save=1.0}" \
    --safety-margin-sec "${HARDNEG_SAFETY_MARGIN_SEC:-8}" \
    --safety-label-mode "${HARDNEG_SAFETY_LABEL_MODE:-same_label}" \
    --dedupe-gap-sec "${HARDNEG_DEDUPE_GAP_SEC:-10}" \
    --max-per-video-per-label "${HARDNEG_MAX_PER_VIDEO_PER_LABEL:-8}"

  "${PYTHON_BIN}" scripts/audit_football_hard_negatives.py \
    --manifest "${MANIFEST}" \
    --eval-run-dir "${MINING_RUN}" \
    --train-num-clips "${TRAIN_NUM_CLIPS:-48051}" \
    --target-checkpoint "${INIT_CHECKPOINT}" \
    --output "${MANIFEST%.json}_audit.json"
}

train_rank() {
  check_init
  if [[ ! -s "${MANIFEST}" ]]; then
    echo "Missing hard-negative manifest: ${MANIFEST}. Run: bash $0 build ${GPU_LIST}" >&2
    exit 2
  fi
  local gpu_count local_ids grad_accum head_lr_per_gpu backbone_lr_per_gpu effective_batch log_file
  gpu_count="$(comma_count "${GPU_LIST}")"
  local_ids="$(local_gpu_ids "${gpu_count}")"
  grad_accum="$(grad_accum_for "${TARGET_EFFECTIVE_BATCH_SIZE}" "${PER_GPU_BATCH_SIZE}" "${gpu_count}")"
  head_lr_per_gpu="$(lr_per_gpu_for "${TARGET_HEAD_LR}" "${gpu_count}")"
  backbone_lr_per_gpu="$(lr_per_gpu_for "${TARGET_BACKBONE_LR}" "${gpu_count}")"
  effective_batch=$((PER_GPU_BATCH_SIZE * gpu_count * grad_accum))
  mkdir -p "${OUTPUT_DIR}"
  log_file="${OUTPUT_DIR}/train_console.log"
  {
    echo
    echo "===== clean hard-sample clip-ranking experiment ====="
    echo "started_at=$(date --iso-8601=seconds)"
    echo "physical_gpus=${GPU_LIST} local_gpu_ids=${local_ids}"
    echo "config=${CONFIG}"
    echo "init_checkpoint=${INIT_CHECKPOINT}"
    echo "manifest=${MANIFEST}"
    echo "output_dir=${OUTPUT_DIR}"
    echo "per_gpu_batch_size=${PER_GPU_BATCH_SIZE} effective_batch_size=${effective_batch} grad_accum=${grad_accum}"
    echo "head_lr_global=${TARGET_HEAD_LR} backbone_lr_global=${TARGET_BACKBONE_LR}"
    echo "clip_loss=bce frame_rank=0 hard_negative_rank=clip shot/save lora_train=true"
  } | tee -a "${log_file}"

  CUDA_VISIBLE_DEVICES="${GPU_LIST}" PYTHONUNBUFFERED=1 "${PYTHON_BIN}" train_football_events.py \
    --config "${CONFIG}" \
    "output_dir=${OUTPUT_DIR}" \
    "gpu_ids=${local_ids}" \
    "model.init_checkpoint=${INIT_CHECKPOINT}" \
    "model.freeze_backbone=false" \
    "model.freeze_loaded_backbone=false" \
    "data.long_video.hard_negative.manifest=${MANIFEST}" \
    "train.epochs=${EPOCHS}" \
    "train.per_gpu_batch_size=${PER_GPU_BATCH_SIZE}" \
    "train.grad_accum_steps=${grad_accum}" \
    "train.lr_per_gpu=${head_lr_per_gpu}" \
    "train.backbone_lr_per_gpu=${backbone_lr_per_gpu}" \
    "train.resume.enabled=false" \
    "train.resume.load_optimizer=false" \
    "train.resume.load_scheduler=false" \
    "train.resume.load_scaler=false" \
    "eval.per_gpu_batch_size=${PER_GPU_BATCH_SIZE}" \
    "data.num_workers_per_gpu=${NUM_WORKERS_PER_GPU}" \
    "data.prefetch_factor=${PREFETCH_FACTOR}" \
    2>&1 | tee -a "${log_file}"
}

eval_six() {
  local checkpoint="${CHECKPOINT:-${OUTPUT_DIR}/best.pt}"
  local run_name="${RUN_NAME:-$(basename "${OUTPUT_DIR}")_6videos_dense_checkpoint_thr}"
  local gpu_count local_ids_csv batch_size num_workers
  gpu_count="$(comma_count "${EVAL_GPU_LIST:-${GPU_LIST}}")"
  local_ids_csv="$(local_gpu_ids_csv "${gpu_count}")"
  batch_size=$((gpu_count * EVAL_BATCH_PER_GPU))
  num_workers=$((gpu_count * EVAL_NUM_WORKERS_PER_GPU))
  CUDA_VISIBLE_DEVICES="${EVAL_GPU_LIST:-${GPU_LIST}}" "${PYTHON_BIN}" scripts/evaluate_football_model.py \
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
    --gpu-ids "${local_ids_csv}" \
    --thresholds checkpoint \
    --prediction-postprocess point_nms \
    --nms-radius-sec "${NMS_RADIUS_SEC:-5}" \
    --match-tolerance-sec "${MATCH_TOLERANCE_SEC:-5}" \
    --save-frame-event-logits \
    --frame-event-topk 8 \
    ${EVAL_FORCE:+--force}

  "${PYTHON_BIN}" scripts/recompute_football_eval_protocols.py \
    --run-dir "outputs/football_eval_runs/${run_name}" \
    --video-ids "${VIDEO_IDS}" \
    --output-prefix protocol_comparison_checkpoint_thr \
    --nms-radius-sec "${NMS_RADIUS_SEC:-5}" \
    --match-tolerance-sec "${MATCH_TOLERANCE_SEC:-5}"
}

case "${MODE}" in
  mine)
    mine_pointnms
    ;;
  build)
    build_manifest
    ;;
  train)
    train_rank
    ;;
  eval)
    eval_six
    ;;
  mine_build)
    mine_pointnms
    build_manifest
    ;;
  all)
    if [[ "${RUN_MINING:-0}" == "1" ]]; then
      mine_pointnms
    fi
    build_manifest
    train_rank
    eval_six
    ;;
  help|*)
    cat <<EOF
Usage: $0 {mine|build|mine_build|train|eval|all} GPU_LIST

Recommended after neg8 if no gain:
  RUN_MINING=1 INIT_CHECKPOINT=/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt \\
    bash $0 all 2,3

Pre-mine hard negatives before training:
  INIT_CHECKPOINT=/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt bash $0 mine_build 2

Use existing mining run/manifest:
  MINING_RUN=outputs/football_eval_runs/vitl16_e1_pointnms_train_fp_mining bash $0 build
  bash $0 train 2,3

Key env:
  CONFIG=${CONFIG}
  OUTPUT_DIR=${OUTPUT_DIR}
  MANIFEST=${MANIFEST}
  PER_GPU_BATCH_SIZE=${PER_GPU_BATCH_SIZE}
  TARGET_EFFECTIVE_BATCH_SIZE=${TARGET_EFFECTIVE_BATCH_SIZE}
  TARGET_HEAD_LR=${TARGET_HEAD_LR}
  TARGET_BACKBONE_LR=${TARGET_BACKBONE_LR}
EOF
    ;;
esac
