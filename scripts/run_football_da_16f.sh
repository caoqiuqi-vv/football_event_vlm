#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"
GPU="${2:-0,1}"
LAUNCHER="scripts/launch_football_detection_aware.sh"

case "${MODE}" in
  all)
    bash "${LAUNCHER}" exp4_16f "${GPU}"
    bash "${LAUNCHER}" eval4_16f "${GPU}"
    ;;
  debug)
    bash "${LAUNCHER}" exp4_16f_debug "${GPU}"
    ;;
  train)
    bash "${LAUNCHER}" exp4_16f "${GPU}"
    ;;
  dual_lora|exp4_dual_lora)
    bash "${LAUNCHER}" exp4_dual_lora "${GPU}"
    ;;
  dual_lora_debug|exp4_dual_lora_debug)
    bash "${LAUNCHER}" exp4_dual_lora_debug "${GPU}"
    ;;
  dual_lora_eval|exp4_dual_lora_eval)
    bash "${LAUNCHER}" eval_exp4_dual_lora "${GPU}"
    ;;
  eval)
    bash "${LAUNCHER}" eval4_16f "${GPU}"
    ;;
  gt_roi_compare)
    bash "${LAUNCHER}" gt_roi_compare "${GPU}"
    ;;
  hr_single_lora)
    bash "${LAUNCHER}" hr_single_lora "${GPU}"
    ;;
  hr_single_lora_debug)
    bash "${LAUNCHER}" debug_hr_single_lora "${GPU}"
    ;;
  hr_single_lora_eval)
    bash "${LAUNCHER}" eval_hr_single_lora "${GPU}"
    ;;
  hr_single_lora_frame_det)
    bash "${LAUNCHER}" hr_single_lora_frame_det "${GPU}"
    ;;
  hr_single_lora_frame_det_debug)
    bash "${LAUNCHER}" debug_hr_single_lora_frame_det "${GPU}"
    ;;
  hr_single_lora_frame_det_eval)
    bash "${LAUNCHER}" eval_hr_single_lora_frame_det "${GPU}"
    ;;
  e1_lora_r5_f32)
    bash "${LAUNCHER}" e1_lora_r5_f32 "${GPU}"
    ;;
  e1_lora_r5_f32_debug)
    bash "${LAUNCHER}" debug_e1_lora_r5_f32 "${GPU}"
    ;;
  e1_lora_r5_f32_eval)
    bash "${LAUNCHER}" eval_e1_lora_r5_f32 "${GPU}"
    ;;
  d0_fixed_roi)
    bash "${LAUNCHER}" d0_fixed_roi "${GPU}"
    ;;
  d0_fixed_roi_debug)
    bash "${LAUNCHER}" debug_d0_fixed_roi "${GPU}"
    ;;
  d0_fixed_roi_eval)
    bash "${LAUNCHER}" eval_d0_fixed_roi "${GPU}"
    ;;
  d1_dynamic_roi)
    bash "${LAUNCHER}" d1_dynamic_roi "${GPU}"
    ;;
  d1_dynamic_roi_a800)
    bash "${LAUNCHER}" d1_dynamic_roi_a800 "${GPU}"
    ;;
  d1_dynamic_roi_debug)
    bash "${LAUNCHER}" debug_d1_dynamic_roi "${GPU}"
    ;;
  d1_dynamic_roi_eval)
    bash "${LAUNCHER}" eval_d1_dynamic_roi "${GPU}"
    ;;
  d2_feature_quality)
    bash "${LAUNCHER}" d2_feature_quality "${GPU}"
    ;;
  d2_feature_quality_a800)
    bash "${LAUNCHER}" d2_feature_quality_a800 "${GPU}"
    ;;
  d2_feature_quality_debug)
    bash "${LAUNCHER}" debug_d2_feature_quality "${GPU}"
    ;;
  d2_feature_quality_eval)
    bash "${LAUNCHER}" eval_d2_feature_quality "${GPU}"
    ;;
  e1|e1_frame_det)
    bash "${LAUNCHER}" e1_frame_det "${GPU}"
    ;;
  e1_debug)
    bash "${LAUNCHER}" debug_e1_frame_det "${GPU}"
    ;;
  e1_eval)
    bash "${LAUNCHER}" eval_e1_frame_det "${GPU}"
    ;;
  e1_mine_hard_neg|mine_e1_hard_neg)
    bash "${LAUNCHER}" mine_e1_hard_neg "${GPU}"
    ;;
  e1_hard_neg)
    bash "${LAUNCHER}" e1_hard_neg "${GPU}"
    ;;
  e1_hard_neg_debug)
    bash "${LAUNCHER}" debug_e1_hard_neg "${GPU}"
    ;;
  e1_hard_neg_eval)
    bash "${LAUNCHER}" eval_e1_hard_neg "${GPU}"
    ;;
  e2_lora_r5_f32)
    bash "${LAUNCHER}" e2_lora_r5_f32 "${GPU}"
    ;;
  e2_lora_r5_f32_debug)
    bash "${LAUNCHER}" debug_e2_lora_r5_f32 "${GPU}"
    ;;
  e2_lora_r5_f32_eval)
    bash "${LAUNCHER}" eval_e2_lora_r5_f32 "${GPU}"
    ;;
  e2|e2_attn_pool)
    bash "${LAUNCHER}" e2_attn_pool "${GPU}"
    ;;
  e2_debug)
    bash "${LAUNCHER}" debug_e2_attn_pool "${GPU}"
    ;;
  e2_eval)
    bash "${LAUNCHER}" eval_e2_attn_pool "${GPU}"
    ;;
  e3_lora_r5_f32)
    bash "${LAUNCHER}" e3_lora_r5_f32 "${GPU}"
    ;;
  e3_lora_r5_f32_debug)
    bash "${LAUNCHER}" debug_e3_lora_r5_f32 "${GPU}"
    ;;
  e3_lora_r5_f32_eval)
    bash "${LAUNCHER}" eval_e3_lora_r5_f32 "${GPU}"
    ;;
  e3|e3_class_query)
    bash "${LAUNCHER}" e3_class_query "${GPU}"
    ;;
  e3_debug)
    bash "${LAUNCHER}" debug_e3_class_query "${GPU}"
    ;;
  e3_eval)
    bash "${LAUNCHER}" eval_e3_class_query "${GPU}"
    ;;
  e4_lora_r5_f32)
    bash "${LAUNCHER}" e4_lora_r5_f32 "${GPU}"
    ;;
  e4_lora_r5_f32_debug)
    bash "${LAUNCHER}" debug_e4_lora_r5_f32 "${GPU}"
    ;;
  e4_lora_r5_f32_eval)
    bash "${LAUNCHER}" eval_e4_lora_r5_f32 "${GPU}"
    ;;
  e4|e4_event_topk)
    bash "${LAUNCHER}" e4_event_topk "${GPU}"
    ;;
  e4_debug)
    bash "${LAUNCHER}" debug_e4_event_topk "${GPU}"
    ;;
  e4_eval)
    bash "${LAUNCHER}" eval_e4_event_topk "${GPU}"
    ;;
  *)
    echo "Usage: bash scripts/run_football_da_16f.sh [all|debug|train|eval|dual_lora|dual_lora_debug|dual_lora_eval|hr_single_lora|hr_single_lora_debug|hr_single_lora_eval|hr_single_lora_frame_det|hr_single_lora_frame_det_debug|hr_single_lora_frame_det_eval|d0_fixed_roi|d0_fixed_roi_debug|d0_fixed_roi_eval|d1_dynamic_roi|d1_dynamic_roi_a800|d1_dynamic_roi_debug|d1_dynamic_roi_eval|d2_feature_quality|d2_feature_quality_a800|d2_feature_quality_debug|d2_feature_quality_eval|e1|e1_debug|e1_eval|e1_mine_hard_neg|e1_hard_neg|e1_hard_neg_debug|e1_hard_neg_eval|e2|e2_debug|e2_eval|e3|e3_debug|e3_eval|e4|e4_debug|e4_eval|e1_lora_r5_f32|e1_lora_r5_f32_debug|e1_lora_r5_f32_eval|e2_lora_r5_f32|e2_lora_r5_f32_debug|e2_lora_r5_f32_eval|e3_lora_r5_f32|e3_lora_r5_f32_debug|e3_lora_r5_f32_eval|e4_lora_r5_f32|e4_lora_r5_f32_debug|e4_lora_r5_f32_eval] [cuda_visible_devices]" >&2
    exit 2
    ;;
esac
