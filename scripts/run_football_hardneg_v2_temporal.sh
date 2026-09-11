#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-help}"
GPU_LIST="${2:-2,3}"

CONFIG="${CONFIG:-configs/football/dinov3_vitl16_stage2_single_hr_hardneg_v2_temporal.yaml}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt}"
MINING_RUN_NAME="${MINING_RUN_NAME:-vitl16_lora_r5_f32_16f_e1_frame_det_reviewed_train_dense_mining}"
MINING_RUN="${MINING_RUN:-outputs/football_eval_runs/${MINING_RUN_NAME}}"
MANIFEST="${MANIFEST:-outputs/football_hard_negatives/reviewed_mid_score_v2_shot_save_setpiece.json}"
HARDNEG_BRANCH="${HARDNEG_BRANCH:-fused}"
REVIEWED_TRAIN="${REVIEWED_TRAIN:-configs/football/splits/vitl16_lvd1689m_set_piece_seed42_162videos_reviewed/train_video_ids.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/football_events/vitl16_stage2_single_hr_hardneg_v2_temporal}"
VIDEO_IDS="${VIDEO_IDS:-2027564888428580866,2027572095412195330,2027572406738604033,2042125172076392450,2042520971893485569,2042525152494694401}"
GT_DIR="${GT_DIR:-/home/new_users/qiuqi/code/football_events_human_repair}"
VIDEO_ROOT="${VIDEO_ROOT:-/mnt/data_16t/football/raw_video_720P}"

gpu_ids_arg() {
  python - "$1" <<PY2
import sys
n = len([item for item in sys.argv[1].split(",") if item.strip()])
print("[" + ",".join(str(i) for i in range(n)) + "]")
PY2
}

check_manifest_checkpoint() {
  if [[ "${ALLOW_CHECKPOINT_MISMATCH:-0}" == "1" ]]; then
    return
  fi
  python - "$MANIFEST" "$INIT_CHECKPOINT" <<'PY2'
import json
import sys
from pathlib import Path
manifest_path = Path(sys.argv[1])
target = sys.argv[2]
if not manifest_path.exists():
    raise SystemExit(f"Missing hard-negative manifest: {manifest_path}. Run stage 'mine' then 'build' first.")
manifest = json.loads(manifest_path.read_text())
mining = str(manifest.get("mining_checkpoint", ""))
if not mining:
    sys.exit(0)

def same_checkpoint(left: str, right: str) -> bool:
    if left == right:
        return True
    left_path = Path(left)
    right_path = Path(right)
    if left_path.exists() and right_path.exists():
        return left_path.resolve() == right_path.resolve()
    return False

if not same_checkpoint(mining, target):
    raise SystemExit(
        "Hard-negative manifest was mined from a different checkpoint. "
        "Run stage 'mine' and 'build' with the target INIT_CHECKPOINT, or set "
        "ALLOW_CHECKPOINT_MISMATCH=1 for an intentional exploratory run.\n"
        f"manifest={manifest_path}\n"
        f"mining_checkpoint={mining}\n"
        f"target_checkpoint={target}"
    )
PY2
}

sharded_video_id_file() {
  if [[ "${SHARD_COUNT:-1}" == "1" ]]; then
    echo "$REVIEWED_TRAIN"
    return
  fi
  local shard_dir="outputs/football_hard_negatives/shards"
  local shard_file="${shard_dir}/reviewed_train_shard_${SHARD_INDEX:-0}_of_${SHARD_COUNT}.txt"
  mkdir -p "$shard_dir"
  python - "$REVIEWED_TRAIN" "$shard_file" "${SHARD_INDEX:-0}" "$SHARD_COUNT" <<'PY2'
import sys
from pathlib import Path
source = Path(sys.argv[1])
target = Path(sys.argv[2])
index = int(sys.argv[3])
count = int(sys.argv[4])
if count <= 0 or index < 0 or index >= count:
    raise SystemExit(f"Invalid shard index/count: {index}/{count}")
ids = [line.strip() for line in source.read_text().splitlines() if line.strip() and not line.startswith("#")]
shard = [video_id for pos, video_id in enumerate(ids) if pos % count == index]
target.write_text("\n".join(shard) + ("\n" if shard else ""))
print(target)
PY2
}

mine_dense() {
  local extra_args=()
  local force_args=()
  local video_id_file
  video_id_file="$(sharded_video_id_file)"
  if [[ -n "${MINING_EXTRA_ARGS:-}" ]]; then
    read -r -a extra_args <<< "$MINING_EXTRA_ARGS"
  fi
  if [[ "${MINE_FORCE:-0}" == "1" ]]; then
    force_args=(--force)
  fi
  CUDA_VISIBLE_DEVICES="${MINING_GPU:-${GPU_LIST%%,*}}" python scripts/evaluate_football_model.py \
    --checkpoint "$INIT_CHECKPOINT" \
    --mode dense \
    --video-id-file "$video_id_file" \
    --gt-dir "$GT_DIR" \
    --video-root "xbotgo_0608=${VIDEO_ROOT}" \
    --output-root outputs/football_eval_runs \
    --run-name "$MINING_RUN_NAME" \
    --clip-sec 10 \
    --stride-sec 5 \
    --batch-size "${MINING_BATCH_SIZE:-1}" \
    --num-workers "${MINING_NUM_WORKERS:-2}" \
    --device cuda:0 \
    --gpu-ids 0 \
    --thresholds checkpoint \
    --prediction-postprocess window_overlap \
    --match-tolerance-sec 5 \
    "${force_args[@]}" \
    "${extra_args[@]}"
}

build_manifest() {
  python scripts/build_reviewed_hard_negatives_v2.py \
    --eval-run-dir "$MINING_RUN" \
    --output "$MANIFEST" \
    --reviewed-video-ids "$REVIEWED_TRAIN" \
    --branch "$HARDNEG_BRANCH" \
    --labels shot,save,set_piece \
    --min-scores shot=0.35,save=0.35,set_piece=0.35 \
    --max-scores shot=0.75,save=0.75,set_piece=0.70 \
    --target-scores shot=0.55,save=0.55,set_piece=0.50 \
    --safety-margin-sec 8 \
    --dedupe-gap-sec 10 \
    --max-per-video-per-label 12

  python scripts/audit_football_hard_negatives.py \
    --manifest "$MANIFEST" \
    --eval-run-dir "$MINING_RUN" \
    --train-num-clips 32034 \
    --target-checkpoint "$INIT_CHECKPOINT" \
    --output "${MANIFEST%.json}_audit.json"

  if [[ "${ALLOW_CHECKPOINT_MISMATCH:-0}" != "1" ]]; then
    python - "$MANIFEST" <<'PY2'
import json
import sys
from pathlib import Path
manifest = Path(sys.argv[1])
audit = json.loads(manifest.with_name(manifest.stem + "_audit.json").read_text())
if audit.get("checkpoint_matches_target") is False:
    message = (
        "Hard-negative mining checkpoint does not match INIT_CHECKPOINT. "
        "Run the mine stage with the target checkpoint, or set "
        "ALLOW_CHECKPOINT_MISMATCH=1 for an intentional exploratory run.\n"
        f"mining_checkpoint={audit.get('mining_checkpoint')}\n"
        f"target_checkpoint={audit.get('target_checkpoint')}"
    )
    raise SystemExit(message)
PY2
  fi
}

train_temporal() {
  local gpu_ids
  local extra_args=()
  gpu_ids="$(gpu_ids_arg "$GPU_LIST")"
  if [[ -n "${TRAIN_EXTRA_ARGS:-}" ]]; then
    read -r -a extra_args <<< "$TRAIN_EXTRA_ARGS"
  fi
  check_manifest_checkpoint
  CUDA_VISIBLE_DEVICES="$GPU_LIST" PYTHONUNBUFFERED=1 python train_football_events.py \
    --config "$CONFIG" \
    "output_dir=${OUTPUT_DIR}" \
    "model.init_checkpoint=${INIT_CHECKPOINT}" \
    "data.long_video.hard_negative.manifest=${MANIFEST}" \
    "gpu_ids=${gpu_ids}" \
    "train.resume.enabled=false" \
    "${extra_args[@]}"
}

eval_six() {
  local checkpoint="${1:-${OUTPUT_DIR}/best.pt}"
  local run_name="${2:-$(basename "$OUTPUT_DIR")_6videos_window_overlap_checkpoint_thr}"
  local extra_args=()
  if [[ -n "${EVAL_EXTRA_ARGS:-}" ]]; then
    read -r -a extra_args <<< "$EVAL_EXTRA_ARGS"
  fi
  CUDA_VISIBLE_DEVICES="${EVAL_GPU:-${GPU_LIST%%,*}}" python scripts/evaluate_football_model.py \
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
    --match-tolerance-sec 5 \
    "${extra_args[@]}"
}

case "$MODE" in
  mine)
    mine_dense
    ;;
  build)
    build_manifest
    ;;
  train)
    train_temporal
    ;;
  eval)
    eval_six "${CHECKPOINT:-${OUTPUT_DIR}/best.pt}" "${RUN_NAME:-$(basename "$OUTPUT_DIR")_6videos_window_overlap_checkpoint_thr}"
    ;;
  all)
    if [[ "${RUN_MINING:-0}" == "1" ]]; then
      mine_dense
      MINING_RUN="outputs/football_eval_runs/${MINING_RUN_NAME}"
    fi
    build_manifest
    train_temporal
    eval_six "${OUTPUT_DIR}/best.pt" "$(basename "$OUTPUT_DIR")_6videos_window_overlap_checkpoint_thr"
    ;;
  *)
    cat <<EOF
Usage: $0 {mine|build|train|eval|all} GPU_LIST

Examples:
  RUN_MINING=1 bash $0 all 2,3
  bash $0 mine 2
  SHARD_COUNT=4 SHARD_INDEX=0 MINING_GPU=2 bash $0 mine 2
  bash $0 build
  bash $0 train 2,3
  bash $0 eval 2

Key env overrides:
  MINING_RUN=outputs/football_eval_runs/...
  INIT_CHECKPOINT=/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt
  MANIFEST=outputs/football_hard_negatives/reviewed_mid_score_v2_shot_save_setpiece.json
  OUTPUT_DIR=outputs/football_events/vitl16_stage2_single_hr_hardneg_v2_temporal
  TRAIN_EXTRA_ARGS='data.num_workers_per_gpu=0 train.per_gpu_batch_size=2'
  MINE_FORCE=1  # rerun completed videos during mining
EOF
    ;;
esac
