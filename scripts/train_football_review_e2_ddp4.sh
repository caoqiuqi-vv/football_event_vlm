#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"
PYTHON_BIN=${PYTHON_BIN:-python}
REVIEW_DATA_DIR=${REVIEW_DATA_DIR:-$REPO_ROOT/outputs/football_review_training/review_e1_e2_20260916}
VIDEO_ROOT=${VIDEO_ROOT:-/mnt/data_16t/football/raw_video_1080P}
INIT_CHECKPOINT=${INIT_CHECKPOINT:-$REPO_ROOT/outputs/football_events/review_clean_e1_e2_20260916/initial_best.pt}
DINO_WEIGHTS=${DINO_WEIGHTS:-$REPO_ROOT/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth}
E2_OUTPUT_DIR=${E2_OUTPUT_DIR:-$REPO_ROOT/outputs/football_events/review_clean_e1_e2_20260916/E2_ddp4}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
"$PYTHON_BIN" scripts/prepare_football_review_e2_ddp4.py \
  --review-data-dir "$REVIEW_DATA_DIR" --video-root "$VIDEO_ROOT" \
  --init-checkpoint "$INIT_CHECKPOINT" --dino-weights "$DINO_WEIGHTS" \
  --output "$E2_OUTPUT_DIR" --gpu-ids "$CUDA_VISIBLE_DEVICES" "$@"
# --check-only verifies the inputs without allocating GPUs or creating output.
if [[ " ${*} " == *" --check-only "* ]]; then exit 0; fi
exec "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=4 \
  train_football_events.py --config "$E2_OUTPUT_DIR/launch_config.yaml"
