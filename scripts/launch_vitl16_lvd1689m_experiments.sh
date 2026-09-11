#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-all}"
FREEZE_CFG="configs/football/dinov3_vitl16_lvd1689m_long_set_piece_freeze_temporal.yaml"
LORA_CFG="configs/football/dinov3_vitl16_lvd1689m_long_set_piece_lora_temporal.yaml"
HARD_NEGATIVE_MANIFEST="${HARD_NEGATIVE_MANIFEST:-}"
FOOTBALL_VIDEOS_DIR="${FOOTBALL_VIDEOS_DIR:-}"
FOOTBALL_XBOTGO_VIDEOS_DIR="${FOOTBALL_XBOTGO_VIDEOS_DIR:-}"
FOOTBALL_MAY_VIDEOS_DIR="${FOOTBALL_MAY_VIDEOS_DIR:-}"
FOOTBALL_ANNOTATIONS_DIR="${FOOTBALL_ANNOTATIONS_DIR:-}"

run_train() {
  local cfg="$1"
  shift
  local args=("$@")
  if [[ -n "$FOOTBALL_VIDEOS_DIR" && ( -n "$FOOTBALL_XBOTGO_VIDEOS_DIR" || -n "$FOOTBALL_MAY_VIDEOS_DIR" ) ]]; then
    echo "Use either FOOTBALL_VIDEOS_DIR or the XBOTGO/MAY pair, not both." >&2
    exit 2
  fi
  if [[ -n "$FOOTBALL_VIDEOS_DIR" ]]; then
    if [[ -z "$FOOTBALL_ANNOTATIONS_DIR" ]]; then
      echo "FOOTBALL_ANNOTATIONS_DIR is required with FOOTBALL_VIDEOS_DIR." >&2
      exit 2
    fi
    args+=("data.long_video.roots=[{source: human_repair, videos_dir: $FOOTBALL_VIDEOS_DIR, annotations_dir: $FOOTBALL_ANNOTATIONS_DIR}]")
  elif [[ -n "$FOOTBALL_XBOTGO_VIDEOS_DIR" || -n "$FOOTBALL_MAY_VIDEOS_DIR" || -n "$FOOTBALL_ANNOTATIONS_DIR" ]]; then
    if [[ -z "$FOOTBALL_XBOTGO_VIDEOS_DIR" || -z "$FOOTBALL_MAY_VIDEOS_DIR" || -z "$FOOTBALL_ANNOTATIONS_DIR" ]]; then
      echo "FOOTBALL_XBOTGO_VIDEOS_DIR, FOOTBALL_MAY_VIDEOS_DIR and FOOTBALL_ANNOTATIONS_DIR must be set together." >&2
      exit 2
    fi
    args+=("data.long_video.roots=[{source: xbotgo_0608, videos_dir: $FOOTBALL_XBOTGO_VIDEOS_DIR, annotations_dir: $FOOTBALL_ANNOTATIONS_DIR}, {source: may_xbotgo, videos_dir: $FOOTBALL_MAY_VIDEOS_DIR, annotations_dir: $FOOTBALL_ANNOTATIONS_DIR}]")
  fi
  echo "[$(date '+%F %T')] starting $cfg overrides=${args[*]}"
  python train_football_events.py --config "$cfg" "${args[@]}"
}

run_ablation() {
  local cfg="$1"
  local name="$2"
  local negative_ratio="$3"
  local output_dir="outputs/football_events/${name}"
  local overrides=(
    "output_dir=${output_dir}"
    "seed=42"
    "deterministic=true"
    "data.long_video.negative_ratio=${negative_ratio}"
    "data.long_video.negative_require_all_labels=false"
    "video.num_frames=32"
    "train.batch_size=32"
    "train.grad_accum_steps=2"
    "train.pos_weight=[1.0,1.0,1.0]"
    "train.resume.enabled=false"
    "eval.batch_size=16"
    "eval.threshold=0.5"
  )
  if [[ -n "$HARD_NEGATIVE_MANIFEST" ]]; then
    output_dir="${output_dir}_hardneg"
    overrides[0]="output_dir=${output_dir}"
    overrides+=(
      "data.long_video.hard_negative.enabled=true"
      "data.long_video.hard_negative.manifest=${HARD_NEGATIVE_MANIFEST}"
    )
  fi
  run_train "$cfg" "${overrides[@]}"
}

case "$MODE" in
  freeze)
    run_train "$FREEZE_CFG"
    ;;
  lora)
    run_train "$LORA_CFG"
    ;;
  all)
    run_train "$FREEZE_CFG"
    run_train "$LORA_CFG"
    ;;
  freeze-r2-f32)
    run_ablation "$FREEZE_CFG" "vitl16_freeze_r2_f32" 2
    ;;
  freeze-r5-f32)
    run_ablation "$FREEZE_CFG" "vitl16_freeze_r5_f32" 5
    ;;
  lora-r2-f32)
    run_ablation "$LORA_CFG" "vitl16_lora_r2_f32" 2
    ;;
  lora-r5-f32)
    run_ablation "$LORA_CFG" "vitl16_lora_r5_f32" 5
    ;;
  ablations)
    run_ablation "$FREEZE_CFG" "vitl16_freeze_r2_f32" 2
    run_ablation "$FREEZE_CFG" "vitl16_freeze_r5_f32" 5
    run_ablation "$LORA_CFG" "vitl16_lora_r2_f32" 2
    run_ablation "$LORA_CFG" "vitl16_lora_r5_f32" 5
    ;;
  dry-run)
    run_train "$FREEZE_CFG" --dry-run
    run_train "$LORA_CFG" --dry-run
    ;;
  *)
    echo "Usage: $0 {freeze|lora|all|freeze-r2-f32|freeze-r5-f32|lora-r2-f32|lora-r5-f32|ablations|dry-run}" >&2
    echo "Set HARD_NEGATIVE_MANIFEST=/path/to/hard_negatives.json to enable mined negatives for an ablation." >&2
    exit 2
    ;;
esac
