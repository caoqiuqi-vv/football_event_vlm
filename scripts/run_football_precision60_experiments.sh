#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-help}"
BASELINE_RUN="${BASELINE_RUN:-outputs/football_eval_runs/vitl16_lora_r5_f32_16f_e1_frame_det_last_hr_6videos_window_overlap_checkpoint_thr}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-$BASELINE_CHECKPOINT}"
HARDNEG_RUN="${HARDNEG_RUN:-outputs/football_eval_runs/e1_exp2_best_train_hard_negative_mining}"
HARDNEG_MANIFEST="${HARDNEG_MANIFEST:-outputs/football_hard_negatives/precision60_global_shot_save_cap6.json}"
GT_DIR="${GT_DIR:-/home/new_users/qiuqi/code/football_events_human_repair}"
VIDEO_ROOT="${VIDEO_ROOT:-/mnt/data_16t/football/raw_video_720P}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
EVAL_GPU="${EVAL_GPU:-4}"
MATCH_TOLERANCE_SEC="${MATCH_TOLERANCE_SEC:-5}"
VIDEO_IDS="${VIDEO_IDS:-2027564888428580866,2027572095412195330,2027572406738604033,2042125172076392450,2042520971893485569,2042525152494694401}"

run_stage0() {
  python scripts/analyze_dense_pr_ceiling.py \
    --run-dir "$BASELINE_RUN" \
    --match-tolerance-sec 5 \
    --precision-floor 0.60 \
    --recall-target 0.8245 \
    --exclude 2027572406738604033:set_piece
  python scripts/analyze_dense_pr_ceiling.py \
    --run-dir "$BASELINE_RUN" \
    --match-tolerance-sec 2 \
    --precision-floor 0.60 \
    --recall-target 0.8245 \
    --exclude 2027572406738604033:set_piece
}

build_hard_negatives() {
  python scripts/build_precision60_hard_negatives.py \
    --eval-run-dir "$HARDNEG_RUN" \
    --output "$HARDNEG_MANIFEST" \
    --branch global \
    --labels shot,save \
    --min-scores shot=0.55,save=0.55 \
    --safety-margin-sec 5 \
    --dedupe-gap-sec 5 \
    --max-per-video-per-label 6
}

run_stage1() {
  local fusion="$1"
  local suffix="$2"
  CUDA_VISIBLE_DEVICES=0 python train_football_events.py \
    --config configs/football/dinov3_vitl16_stage1_frozen_head_probe.yaml \
    "output_dir=outputs/football_events/vitl16_stage1_frozen_head_probe_${suffix}" \
    "model.temporal_fusion=${fusion}" \
    "gpu_ids=[0]"
}

smoke_stage1() {
  python train_football_events.py \
    --config configs/football/dinov3_vitl16_stage1_frozen_head_probe.yaml \
    --dry-run \
    "train.epochs=1" \
    "data.num_workers_per_gpu=0" \
    "gpu_ids=[0]"
}

run_stage2() {
  local blocks="$1"
  local fusion="${TEMPORAL_FUSION:-cls_transformer}"
  CUDA_VISIBLE_DEVICES="$GPU_LIST" python train_football_events.py \
    --config configs/football/dinov3_vitl16_stage2_single_hr_hardneg.yaml \
    "output_dir=outputs/football_events/vitl16_stage2_single_hr_hardneg_last${blocks}_${fusion}" \
    "model.init_checkpoint=${INIT_CHECKPOINT}" \
    "model.temporal_fusion=${fusion}" \
    "model.lora.target_last_blocks=${blocks}" \
    "gpu_ids=[0,1,2,3]"
}

run_stage3() {
  local fusion="${TEMPORAL_FUSION:-cls_transformer}"
  local blocks="${LORA_BLOCKS:-4}"
  CUDA_VISIBLE_DEVICES="$GPU_LIST" python train_football_events.py \
    --config configs/football/dinov3_vitl16_stage3_roi_verifier.yaml \
    "model.init_checkpoint=${INIT_CHECKPOINT}" \
    "model.temporal_fusion=${fusion}" \
    "model.lora.target_last_blocks=${blocks}" \
    "gpu_ids=[0,1,2,3]"
}

smoke_stage3() {
  CUDA_VISIBLE_DEVICES=0 python train_football_events.py \
    --config configs/football/dinov3_vitl16_stage3_roi_verifier.yaml \
    --dry-run \
    "model.init_checkpoint=${INIT_CHECKPOINT}" \
    "train.epochs=1" \
    "data.num_workers_per_gpu=0" \
    "gpu_ids=[0]"
}

eval_checkpoint() {
  local checkpoint="$1"
  local run_name="$2"
  CUDA_VISIBLE_DEVICES="$EVAL_GPU" python scripts/evaluate_football_model.py \
    --checkpoint "$checkpoint" \
    --mode dense \
    --video-ids "$VIDEO_IDS" \
    --gt-dir "$GT_DIR" \
    --video-root "xbotgo_0608=${VIDEO_ROOT}" \
    --run-name "$run_name" \
    --clip-sec 10 \
    --stride-sec 5 \
    --batch-size 1 \
    --num-workers 2 \
    --device cuda:0 \
    --gpu-ids 0 \
    --thresholds checkpoint \
    --prediction-postprocess window_overlap \
    --match-tolerance-sec "$MATCH_TOLERANCE_SEC"

  python scripts/analyze_dense_pr_ceiling.py \
    --run-dir "outputs/football_eval_runs/${run_name}" \
    --match-tolerance-sec "$MATCH_TOLERANCE_SEC" \
    --precision-floor 0.60 \
    --recall-target 0.8245 \
    --exclude 2027572406738604033:set_piece
}

case "$MODE" in
  stage0)
    run_stage0
    ;;
  stage1-smoke)
    smoke_stage1
    ;;
  stage1-cls)
    run_stage1 cls_transformer cls
    ;;
  stage1-class-query)
    run_stage1 class_query_transformer class_query
    ;;
  stage1-attn-pool)
    run_stage1 attn_pool_transformer attn_pool
    ;;
  hard-negatives)
    build_hard_negatives
    ;;
  stage2-last4)
    run_stage2 4
    ;;
  stage2-last8)
    run_stage2 8
    ;;
  stage3-smoke)
    smoke_stage3
    ;;
  stage3)
    run_stage3
    ;;
  eval)
    if [[ $# -ne 3 ]]; then
      echo "Usage: $0 eval CHECKPOINT RUN_NAME" >&2
      exit 2
    fi
    eval_checkpoint "$2" "$3"
    ;;
  help|*)
    echo "Usage: $0 {stage0|stage1-smoke|stage1-cls|stage1-class-query|stage1-attn-pool|hard-negatives|stage2-last4|stage2-last8|stage3-smoke|stage3|eval CHECKPOINT RUN_NAME}"
    echo "Run stage0 first. Do not launch stage1 or stage2 until the previous stage has been reviewed."
    ;;
esac
