#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MODE="${1:-all}"
MINE_GPU_LIST="${2:-2,3,4,6,7}"
RANK_GPU_LIST="${3:-3,4,6,7}"

PYTHON_BIN="${PYTHON_BIN:-python}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt}"
TRAIN_IDS="${TRAIN_IDS:-configs/football/splits/vitl16_lvd1689m_set_piece_seed42_162videos/train_video_ids.txt}"
REVIEWED_TRAIN_IDS="${REVIEWED_TRAIN_IDS:-${TRAIN_IDS}}"
GT_DIR="${GT_DIR:-/home/new_users/qiuqi/code/football_events_human_repair}"
VIDEO_ROOT="${VIDEO_ROOT:-/mnt/data_16t/football/raw_video_720P}"
MINING_RUN_NAME="${MINING_RUN_NAME:-vitl16_e1_pointnms_train_fp_mining}"
MINING_RUN="${MINING_RUN:-outputs/football_eval_runs/${MINING_RUN_NAME}}"
MANIFEST="${MANIFEST:-outputs/football_hard_negatives/pointnms_fp_clean_shot_save.json}"
SHARD_ROOT="${SHARD_ROOT:-outputs/football_hard_negatives/pointnms_fp_clean_shot_save_parallel}"
MINING_BATCH_SIZE="${MINING_BATCH_SIZE:-6}"
MINING_NUM_WORKERS="${MINING_NUM_WORKERS:-4}"
NMS_RADIUS_SEC="${NMS_RADIUS_SEC:-5}"
MATCH_TOLERANCE_SEC="${MATCH_TOLERANCE_SEC:-5}"

split_csv() {
  "${PYTHON_BIN}" -c 'import sys; print("\n".join([x.strip() for x in sys.argv[1].split(",") if x.strip()]))' "$1"
}

csv_count() {
  "${PYTHON_BIN}" -c 'import sys; print(len([x for x in sys.argv[1].split(",") if x.strip()]))' "$1"
}

check_init() {
  if [[ ! -s "${INIT_CHECKPOINT}" ]]; then
    echo "Missing init checkpoint: ${INIT_CHECKPOINT}" >&2
    exit 2
  fi
}

make_shards() {
  mkdir -p "${SHARD_ROOT}"
  local gpu_count
  gpu_count="$(csv_count "${MINE_GPU_LIST}")"
  "${PYTHON_BIN}" - "${TRAIN_IDS}" "${MINING_RUN}" "${SHARD_ROOT}" "${gpu_count}" <<'PY'
import json
import sys
from pathlib import Path

train_ids = Path(sys.argv[1])
run_dir = Path(sys.argv[2])
shard_root = Path(sys.argv[3])
gpu_count = int(sys.argv[4])

ids = [
    line.strip()
    for line in train_ids.read_text().splitlines()
    if line.strip() and not line.strip().startswith("#")
]
completed = {path.parent.name for path in run_dir.glob("*/metrics.json")}
remaining = [video_id for video_id in ids if video_id not in completed]

for old in shard_root.glob("shard_*.txt"):
    old.unlink()

shards = [[] for _ in range(gpu_count)]
for idx, video_id in enumerate(remaining):
    shards[idx % gpu_count].append(video_id)

summary = {
    "train_ids": str(train_ids),
    "run_dir": str(run_dir),
    "total": len(ids),
    "completed": len(completed & set(ids)),
    "remaining": len(remaining),
    "gpu_count": gpu_count,
    "shards": [],
}
for idx, shard_ids in enumerate(shards):
    path = shard_root / f"shard_{idx}.txt"
    path.write_text("\n".join(shard_ids) + ("\n" if shard_ids else ""))
    summary["shards"].append({"index": idx, "path": str(path), "num_videos": len(shard_ids)})

(shard_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps(summary, ensure_ascii=False))
PY
}

mine_parallel() {
  check_init
  make_shards
  mkdir -p "${SHARD_ROOT}/logs"
  mapfile -t gpus < <(split_csv "${MINE_GPU_LIST}")
  local -a pids=()
  local idx gpu shard log

  for idx in "${!gpus[@]}"; do
    gpu="${gpus[$idx]}"
    shard="${SHARD_ROOT}/shard_${idx}.txt"
    log="${SHARD_ROOT}/logs/mine_gpu${gpu}_shard${idx}.log"
    if [[ ! -s "${shard}" ]]; then
      echo "skip empty shard idx=${idx} gpu=${gpu}"
      continue
    fi
    echo "start mining shard idx=${idx} gpu=${gpu} ids=$(wc -l < "${shard}") log=${log}"
    (
      CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" scripts/evaluate_football_model.py \
        --checkpoint "${INIT_CHECKPOINT}" \
        --mode dense \
        --video-id-file "${shard}" \
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
        --nms-radius-sec "${NMS_RADIUS_SEC}" \
        --match-tolerance-sec "${MATCH_TOLERANCE_SEC}"
    ) >"${log}" 2>&1 &
    pids+=("$!")
  done

  if [[ "${#pids[@]}" -eq 0 ]]; then
    echo "no remaining mining shards"
    return 0
  fi

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "at least one mining shard failed; inspect ${SHARD_ROOT}/logs" >&2
    exit 1
  fi
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
  INIT_CHECKPOINT="${INIT_CHECKPOINT}" \
  MANIFEST="${MANIFEST}" \
  RUN_MINING=0 \
  bash scripts/run_football_clip_rank_lora_last8.sh train "${RANK_GPU_LIST}"
}

eval_rank() {
  MANIFEST="${MANIFEST}" \
  RUN_MINING=0 \
  bash scripts/run_football_clip_rank_lora_last8.sh eval "${RANK_GPU_LIST}"
}

case "${MODE}" in
  shard)
    make_shards
    ;;
  mine)
    mine_parallel
    ;;
  build)
    build_manifest
    ;;
  train)
    train_rank
    ;;
  eval)
    eval_rank
    ;;
  all)
    mine_parallel
    build_manifest
    train_rank
    eval_rank
    ;;
  *)
    cat <<EOF
Usage: $0 {shard|mine|build|train|eval|all} MINE_GPU_LIST RANK_GPU_LIST

Example:
  INIT_CHECKPOINT=/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt \\
    bash $0 all 2,3,4,6,7 3,4,6,7
EOF
    ;;
esac
