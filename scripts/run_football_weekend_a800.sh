#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MODE="${1:-help}"
GPU_LIST="${2:-0,1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt}"
UNIFORM_INIT_CHECKPOINT="${UNIFORM_INIT_CHECKPOINT:-/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt}"
EVENT_INIT_CHECKPOINT="${EVENT_INIT_CHECKPOINT:-/mnt/data_16t/football/qiuqi/checkpoints/topk_exp_epoch13.pt}"
HR_CONFIG="configs/football/dinov3_vitl16_lora_r5_f32_16f_hr_single_frame_det.yaml"
HARDNEG_CONFIG="configs/football/dinov3_vitl16_stage2_single_hr_hardneg.yaml"
HARDNEG_MANIFEST="${HARDNEG_MANIFEST:-outputs/football_hard_negatives/e1_single_hr_target_shot_save.json}"
HARDNEG_RUN="${HARDNEG_RUN:-outputs/football_eval_runs/e1_single_hr_train_hard_negative_mining}"
HARDNEG_BRANCH="${HARDNEG_BRANCH:-fused}"
HARDNEG_MIN_SCORES="${HARDNEG_MIN_SCORES:-shot=0.25,save=0.30}"
TRAIN_VIDEO_ID_FILE="${TRAIN_VIDEO_ID_FILE:-configs/football/splits/vitl16_lvd1689m_set_piece_seed42_162videos/train_video_ids.txt}"
MINING_PER_GPU_BATCH_SIZE="${MINING_PER_GPU_BATCH_SIZE:-4}"
HARDNEG_REPEAT_FACTOR="${HARDNEG_REPEAT_FACTOR:-1}"
HARDNEG_HEAD_LR="${HARDNEG_HEAD_LR:-0.00005}"
HARDNEG_BACKBONE_LR="${HARDNEG_BACKBONE_LR:-0.000005}"
VIDEO_ROOT="${VIDEO_ROOT:-/mnt/data_16t/football/raw_video_720P}"
GT_DIR="${GT_DIR:-/home/new_users/qiuqi/code/football_events_human_repair}"
VIDEO_IDS="${VIDEO_IDS:-2027564888428580866,2027572095412195330,2027572406738604033,2042125172076392450,2042520971893485569,2042525152494694401}"
EPOCHS="${EPOCHS:-4}"
DRY_RUN="${DRY_RUN:-0}"
TARGET_EFFECTIVE_BATCH_SIZE="${TARGET_EFFECTIVE_BATCH_SIZE:-80}"
BASE_PER_GPU_BATCH_SIZE="${BASE_PER_GPU_BATCH_SIZE:-10}"
E4_PER_GPU_BATCH_SIZE="${E4_PER_GPU_BATCH_SIZE:-4}"
E4_512_PER_GPU_BATCH_SIZE="${E4_512_PER_GPU_BATCH_SIZE:-8}"
E5_ANCHOR_PER_GPU_BATCH_SIZE="${E5_ANCHOR_PER_GPU_BATCH_SIZE:-4}"
E5_ANCHOR_512_PER_GPU_BATCH_SIZE="${E5_ANCHOR_512_PER_GPU_BATCH_SIZE:-8}"
DUAL_TEMPORAL_PER_GPU_BATCH_SIZE="${DUAL_TEMPORAL_PER_GPU_BATCH_SIZE:-8}"
E1_32_PER_GPU_BATCH_SIZE="${E1_32_PER_GPU_BATCH_SIZE:-8}"
HEAD_LR="${HEAD_LR:-0.0003}"
BACKBONE_LR="${BACKBONE_LR:-0.00003}"
RESUME_TRAINING="${RESUME_TRAINING:-false}"
RESUME_LOAD_OPTIMIZER="${RESUME_LOAD_OPTIMIZER:-false}"
RESUME_LOAD_SCHEDULER="${RESUME_LOAD_SCHEDULER:-false}"
RESUME_LOAD_SCALER="${RESUME_LOAD_SCALER:-false}"

gpu_count() {
  "${PYTHON_BIN}" -c 'import sys; print(len([x for x in sys.argv[1].split(",") if x.strip()]))' "${GPU_LIST}"
}

local_gpu_ids() {
  "${PYTHON_BIN}" -c 'import sys; n=int(sys.argv[1]); print("[" + ",".join(map(str, range(n))) + "]")' "$(gpu_count)"
}

local_gpu_csv() {
  "${PYTHON_BIN}" -c 'import sys; n=int(sys.argv[1]); print(",".join(map(str, range(n))))' "$(gpu_count)"
}

per_gpu_lr() {
  "${PYTHON_BIN}" -c 'import sys; print(float(sys.argv[1]) / int(sys.argv[2]))' "$1" "$(gpu_count)"
}

accum_steps() {
  "${PYTHON_BIN}" -c 'import math,sys; target,per_gpu,world=map(int,sys.argv[1:]); print(max(math.ceil(target/(per_gpu*world)),1))' \
    "${TARGET_EFFECTIVE_BATCH_SIZE}" "$1" "$(gpu_count)"
}

run_logged() {
  local output_dir="$1"
  shift
  mkdir -p "${output_dir}"
  {
    echo
    echo "===== weekend A800 experiment ====="
    echo "started_at=$(date --iso-8601=seconds)"
    echo "mode=${MODE}"
    echo "physical_gpus=${GPU_LIST}"
    echo "init_checkpoint=${INIT_CHECKPOINT}"
    printf "command="
    printf " %q" "$@"
    echo
    sha256sum train_football_events.py "$2"
  } | tee -a "${output_dir}/train_console.log"
  set +e
  CUDA_VISIBLE_DEVICES="${GPU_LIST}" PYTHONUNBUFFERED=1 "$@" \
    2>&1 | tee -a "${output_dir}/train_console.log"
  local status="${PIPESTATUS[0]}"
  set -e
  echo "finished_at=$(date --iso-8601=seconds) exit_status=${status}" \
    | tee -a "${output_dir}/train_console.log"
  return "${status}"
}

train_hr_variant() {
  local name="$1"
  local per_gpu_batch="$2"
  shift 2
  local output_dir="outputs/football_events/${name}"
  local dry_run_args=()
  if [[ "${DRY_RUN}" == "1" ]]; then
    dry_run_args+=(--dry-run "data.num_workers_per_gpu=0")
  fi
  run_logged "${output_dir}" \
    "${PYTHON_BIN}" train_football_events.py \
    --config "${HR_CONFIG}" \
    "${dry_run_args[@]}" \
    "output_dir=${output_dir}" \
    "gpu_ids=$(local_gpu_ids)" \
    "model.init_checkpoint=${INIT_CHECKPOINT}" \
    "train.epochs=${EPOCHS}" \
    "train.per_gpu_batch_size=${per_gpu_batch}" \
    "train.grad_accum_steps=$(accum_steps "${per_gpu_batch}")" \
    "train.lr_per_gpu=$(per_gpu_lr "${HEAD_LR}")" \
    "train.backbone_lr_per_gpu=$(per_gpu_lr "${BACKBONE_LR}")" \
    "train.resume.enabled=${RESUME_TRAINING}" \
    "train.resume.load_optimizer=${RESUME_LOAD_OPTIMIZER}" \
    "train.resume.load_scheduler=${RESUME_LOAD_SCHEDULER}" \
    "train.resume.load_scaler=${RESUME_LOAD_SCALER}" \
    "eval.per_gpu_batch_size=${per_gpu_batch}" \
    "$@"
}

train_hard_negative() {
  local enabled="${1:-true}"
  local suffix="${2:-v2_last4}"
  local repeat_factor=1
  if [[ "${enabled}" == "true" ]]; then
    repeat_factor="${HARDNEG_REPEAT_FACTOR}"
  fi
  if [[ "${enabled}" == "true" && ! -f "${HARDNEG_MANIFEST}" ]]; then
    if [[ ! -d "${HARDNEG_RUN}" ]]; then
      echo "Missing hard-negative manifest: ${HARDNEG_MANIFEST}" >&2
      echo "Missing mining run: ${HARDNEG_RUN}" >&2
      echo "Transfer the manifest or generate dense train-video predictions first." >&2
      exit 2
    fi
    echo "Building missing hard-negative manifest from ${HARDNEG_RUN}"
    "${PYTHON_BIN}" scripts/build_precision60_hard_negatives.py \
      --eval-run-dir "${HARDNEG_RUN}" \
      --output "${HARDNEG_MANIFEST}" \
      --branch "${HARDNEG_BRANCH}" \
      --labels shot,save \
      --min-scores "${HARDNEG_MIN_SCORES}" \
      --safety-margin-sec 5 \
      --dedupe-gap-sec 5 \
      --max-per-video-per-label 6
  fi
  local output_dir="outputs/football_events/vitl16_weekend_hardneg_${suffix}"
  run_logged "${output_dir}" \
    "${PYTHON_BIN}" train_football_events.py \
    --config "${HARDNEG_CONFIG}" \
    "output_dir=${output_dir}" \
    "gpu_ids=$(local_gpu_ids)" \
    "model.init_checkpoint=${INIT_CHECKPOINT}" \
    "data.long_video.hard_negative.enabled=${enabled}" \
    "data.long_video.hard_negative.manifest=${HARDNEG_MANIFEST}" \
    "data.long_video.hard_negative.repeat_factor=${repeat_factor}" \
    "train.epochs=${EPOCHS}" \
    "train.per_gpu_batch_size=${BASE_PER_GPU_BATCH_SIZE}" \
    "train.grad_accum_steps=$(accum_steps "${BASE_PER_GPU_BATCH_SIZE}")" \
    "train.lr=${HARDNEG_HEAD_LR}" \
    "train.backbone_lr=${HARDNEG_BACKBONE_LR}" \
    "eval.per_gpu_batch_size=${BASE_PER_GPU_BATCH_SIZE}"
}


mine_target_hard_negatives() {
  if [[ ! -f "${TRAIN_VIDEO_ID_FILE}" ]]; then
    echo "Missing train video id file: ${TRAIN_VIDEO_ID_FILE}" >&2
    exit 2
  fi
  CUDA_VISIBLE_DEVICES="${GPU_LIST}" "${PYTHON_BIN}" scripts/evaluate_football_model.py \
    --checkpoint "${INIT_CHECKPOINT}" \
    --mode dense \
    --video-id-file "${TRAIN_VIDEO_ID_FILE}" \
    --gt-dir "${GT_DIR}" \
    --video-root "xbotgo_0608=${VIDEO_ROOT}" \
    --output-root "$(dirname "${HARDNEG_RUN}")" \
    --run-name "$(basename "${HARDNEG_RUN}")" \
    --clip-sec 10 \
    --stride-sec 5 \
    --batch-size "$(( $(gpu_count) * MINING_PER_GPU_BATCH_SIZE ))" \
    --num-workers "$(gpu_count)" \
    --device cuda:0 \
    --gpu-ids "$(local_gpu_csv)" \
    --thresholds checkpoint \
    --prediction-postprocess point_nms \
    --nms-radius-sec 5 \
    --match-tolerance-sec 5
  "${PYTHON_BIN}" scripts/build_precision60_hard_negatives.py \
    --eval-run-dir "${HARDNEG_RUN}" \
    --output "${HARDNEG_MANIFEST}" \
    --branch "${HARDNEG_BRANCH}" \
    --labels shot,save \
    --min-scores "${HARDNEG_MIN_SCORES}" \
    --safety-margin-sec 5 \
    --dedupe-gap-sec 5 \
    --max-per-video-per-label 6
  "${PYTHON_BIN}" scripts/audit_football_hard_negatives.py \
    --manifest "${HARDNEG_MANIFEST}" \
    --target-checkpoint "${INIT_CHECKPOINT}" \
    --output "${HARDNEG_MANIFEST%.json}_audit.json"
}

evaluate_checkpoint() {
  local checkpoint="${CHECKPOINT:-$3}"
  local run_name="${RUN_NAME:-$(basename "${checkpoint}" .pt)_weekend_6videos_point_nms}"
  CUDA_VISIBLE_DEVICES="${GPU_LIST%%,*}" "${PYTHON_BIN}" scripts/evaluate_football_model.py \
    --checkpoint "${checkpoint}" \
    --mode dense \
    --video-ids "${VIDEO_IDS}" \
    --gt-dir "${GT_DIR}" \
    --video-root "xbotgo_0608=${VIDEO_ROOT}" \
    --run-name "${run_name}" \
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
    --save-frame-event-logits \
    --frame-event-topk 8
  "${PYTHON_BIN}" scripts/recompute_football_eval_protocols.py \
    --run-dir "outputs/football_eval_runs/${run_name}" \
    --video-ids "${VIDEO_IDS}" \
    --output-prefix protocol_comparison_checkpoint_thr \
    --nms-radius-sec 5 \
    --match-tolerance-sec 5
}

case "${MODE}" in
  mine_hardneg_target)
    mine_target_hard_negatives
    ;;
  hardneg)
    train_hard_negative true v2_last4
    ;;
  hardneg_control)
    train_hard_negative false control_v2_last4
    ;;
  lora8)
    train_hr_variant \
      vitl16_weekend_lora8_frame_det "${BASE_PER_GPU_BATCH_SIZE}" \
      "model.temporal_fusion=cls_transformer" \
      "model.lora.target_last_blocks=8"
    ;;
  attn_pool_probe)
    train_hr_variant \
      vitl16_weekend_attn_pool_head_probe "${BASE_PER_GPU_BATCH_SIZE}" \
      "model.temporal_fusion=attn_pool_transformer" \
      "model.freeze_loaded_backbone=true" \
      "model.lora.target_last_blocks=4"
    ;;
  attn_pool)
    train_hr_variant \
      vitl16_weekend_attn_pool_frame_det "${BASE_PER_GPU_BATCH_SIZE}" \
      "model.temporal_fusion=attn_pool_transformer" \
      "model.lora.target_last_blocks=4"
    ;;
  class_query_probe)
    train_hr_variant \
      vitl16_weekend_class_query_head_probe "${BASE_PER_GPU_BATCH_SIZE}" \
      "model.temporal_fusion=class_query_transformer" \
      "model.freeze_loaded_backbone=true" \
      "model.lora.target_last_blocks=4"
    ;;
  class_query)
    train_hr_variant \
      vitl16_weekend_class_query_frame_det "${BASE_PER_GPU_BATCH_SIZE}" \
      "model.temporal_fusion=class_query_transformer" \
      "model.lora.target_last_blocks=4"
    ;;
  e1_32)
    train_hr_variant \
      vitl16_weekend_e1_512x896_32f_all_frame "${E1_32_PER_GPU_BATCH_SIZE}" \
      "video.image_size=[512,896]" \
      "video.num_frames=32" \
      "video.candidate_num_frames=32" \
      "model.temporal_fusion=cls_transformer" \
      "model.lora.target_last_blocks=4"
    ;;
  event_topk_512)
    train_hr_variant \
      vitl16_weekend_event_topk_512x896_class_union_st "${E4_512_PER_GPU_BATCH_SIZE}" \
      "video.image_size=[512,896]" \
      "video.candidate_num_frames=32" \
      "model.temporal_fusion=event_topk_transformer" \
      "model.event_topk=8" \
      "model.context_frames=8" \
      "model.event_topk_strategy=class_union" \
      "model.event_topk_per_class=4" \
      "model.event_topk_class_indices=[0,1]" \
      "model.event_topk_gradient=straight_through" \
      "model.event_topk_temperature=1.0" \
      "model.event_topk_gradient_scale=0.25" \
      "model.lora.target_last_blocks=4"
    ;;
  event_topk)
    train_hr_variant \
      vitl16_weekend_event_topk_class_union_st "${E4_PER_GPU_BATCH_SIZE}" \
      "video.candidate_num_frames=32" \
      "model.temporal_fusion=event_topk_transformer" \
      "model.event_topk=8" \
      "model.context_frames=8" \
      "model.event_topk_strategy=class_union" \
      "model.event_topk_per_class=4" \
      "model.event_topk_class_indices=[0,1]" \
      "model.event_topk_gradient=straight_through" \
      "model.event_topk_temperature=1.0" \
      "model.event_topk_gradient_scale=0.25" \
      "model.lora.target_last_blocks=4"
    ;;
  uniform_event_dual_e1_512)
    train_hr_variant \
      vitl16_weekend_uniform_event_dual_e1backbone_512x896_frozen_ref "${DUAL_TEMPORAL_PER_GPU_BATCH_SIZE}" \
      "video.image_size=[512,896]" \
      "video.candidate_num_frames=32" \
      "model.temporal_fusion=uniform_event_dual_transformer" \
      "model.uniform_init_checkpoint=${UNIFORM_INIT_CHECKPOINT}" \
      "model.event_init_checkpoint=${EVENT_INIT_CHECKPOINT}" \
      "model.freeze_uniform_reference=true" \
      "model.uniform_frames=16" \
      "model.uniform_event_gate_init.shot=0.05" \
      "model.uniform_event_gate_init.save=0.10" \
      "model.uniform_event_gate_init.set_piece=0.02" \
      "model.event_topk=8" \
      "model.context_frames=8" \
      "model.event_topk_strategy=class_union" \
      "model.event_topk_per_class=4" \
      "model.event_topk_class_indices=[0,1]" \
      "model.event_topk_gradient=detached" \
      "model.event_topk_temperature=1.0" \
      "model.event_topk_gradient_scale=0.0" \
      "train.frame_det_loss_weight=0.0" \
      "train.uniform_temporal_loss_weight=0.0" \
      "train.event_temporal_loss_weight=0.50" \
      "train.positive_retention_loss_weight=0.50" \
      "model.lora.target_last_blocks=4"
    ;;
  uniform_event_dual_512)
    train_hr_variant \
      vitl16_weekend_uniform_event_dual_512x896_st "${DUAL_TEMPORAL_PER_GPU_BATCH_SIZE}" \
      "video.image_size=[512,896]" \
      "video.candidate_num_frames=32" \
      "model.temporal_fusion=uniform_event_dual_transformer" \
      "model.uniform_init_checkpoint=${UNIFORM_INIT_CHECKPOINT}" \
      "model.uniform_frames=16" \
      "model.uniform_event_gate_init.shot=0.25" \
      "model.uniform_event_gate_init.save=0.50" \
      "model.uniform_event_gate_init.set_piece=0.10" \
      "model.event_topk=8" \
      "model.context_frames=8" \
      "model.event_topk_strategy=class_union" \
      "model.event_topk_per_class=4" \
      "model.event_topk_class_indices=[0,1]" \
      "model.event_topk_gradient=straight_through" \
      "model.event_topk_temperature=1.0" \
      "model.event_topk_gradient_scale=0.25" \
      "train.uniform_temporal_loss_weight=0.25" \
      "train.event_temporal_loss_weight=0.25" \
      "train.positive_retention_loss_weight=0.50" \
      "model.lora.target_last_blocks=4"
    ;;
  event_anchor_512)
    train_hr_variant \
      vitl16_weekend_event_anchor_512x896_neighborhood_st "${E5_ANCHOR_512_PER_GPU_BATCH_SIZE}" \
      "video.image_size=[512,896]" \
      "video.candidate_num_frames=32" \
      "model.temporal_fusion=event_anchor_transformer" \
      "model.context_frames=4" \
      "model.event_topk_gradient=straight_through" \
      "model.event_topk_temperature=1.0" \
      "model.event_topk_gradient_scale=0.25" \
      "model.event_anchor_topk_per_class.shot=2" \
      "model.event_anchor_topk_per_class.save=2" \
      "model.event_anchor_topk_per_class.set_piece=1" \
      "model.event_anchor_class_indices=[0,1,2]" \
      "model.event_anchor_offsets=[-2,-1,0,1,2]" \
      "model.event_anchor_nms_radius=2" \
      "model.event_anchor_max_frames=16" \
      "model.lora.target_last_blocks=4"
    ;;
  event_anchor_sym20_512)
    train_hr_variant \
      vitl16_weekend_event_anchor_512x896_sym20_st "${E5_ANCHOR_512_PER_GPU_BATCH_SIZE}" \
      "video.image_size=[512,896]" \
      "video.candidate_num_frames=32" \
      "model.temporal_fusion=event_anchor_transformer" \
      "model.context_frames=4" \
      "model.event_topk_gradient=straight_through" \
      "model.event_topk_temperature=1.0" \
      "model.event_topk_gradient_scale=0.25" \
      "model.event_anchor_topk_per_class.shot=2" \
      "model.event_anchor_topk_per_class.save=2" \
      "model.event_anchor_topk_per_class.set_piece=1" \
      "model.event_anchor_class_indices=[0,1,2]" \
      "model.event_anchor_offsets=[-1,0,1]" \
      "model.event_anchor_nms_radius=2" \
      "model.event_anchor_max_frames=20" \
      "model.lora.target_last_blocks=4"
    ;;
  event_anchor)
    train_hr_variant \
      vitl16_weekend_event_anchor_neighborhood_st "${E5_ANCHOR_PER_GPU_BATCH_SIZE}" \
      "video.candidate_num_frames=32" \
      "model.temporal_fusion=event_anchor_transformer" \
      "model.context_frames=4" \
      "model.event_topk_gradient=straight_through" \
      "model.event_topk_temperature=1.0" \
      "model.event_topk_gradient_scale=0.25" \
      "model.event_anchor_topk_per_class.shot=2" \
      "model.event_anchor_topk_per_class.save=2" \
      "model.event_anchor_topk_per_class.set_piece=1" \
      "model.event_anchor_class_indices=[0,1,2]" \
      "model.event_anchor_offsets=[-2,-1,0,1,2]" \
      "model.event_anchor_nms_radius=2" \
      "model.event_anchor_max_frames=16" \
      "model.lora.target_last_blocks=4"
    ;;
  smoke-lora8|smoke-attn_pool_probe|smoke-attn_pool|smoke-class_query_probe|smoke-class_query|smoke-e1_32|smoke-event_topk|smoke-event_topk_512|smoke-event_anchor|smoke-event_anchor_512|smoke-event_anchor_sym20_512|smoke-uniform_event_dual_512|smoke-uniform_event_dual_e1_512)
    variant="${MODE#smoke-}"
    DRY_RUN=1 EPOCHS=1 exec bash "$0" "${variant}" "${GPU_LIST}"
    ;;
  eval)
    if [[ $# -lt 3 ]]; then
      echo "Usage: $0 eval GPU CHECKPOINT" >&2
      exit 2
    fi
    evaluate_checkpoint "$@"
    ;;
  help|*)
    echo "Usage: $0 {mine_hardneg_target|hardneg|hardneg_control|lora8|attn_pool_probe|attn_pool|class_query_probe|class_query|e1_32|event_topk|event_topk_512|event_anchor|event_anchor_512|event_anchor_sym20_512|uniform_event_dual_512|uniform_event_dual_e1_512|smoke-lora8|smoke-attn_pool_probe|smoke-attn_pool|smoke-class_query_probe|smoke-class_query|smoke-e1_32|smoke-event_topk|smoke-event_topk_512|smoke-event_anchor|smoke-event_anchor_512|smoke-event_anchor_sym20_512|smoke-uniform_event_dual_512|smoke-uniform_event_dual_e1_512|eval GPU CHECKPOINT} GPU_LIST"
    echo "Example: INIT_CHECKPOINT=/path/to/best.pt bash $0 event_anchor_512 2,3"
    ;;
esac
