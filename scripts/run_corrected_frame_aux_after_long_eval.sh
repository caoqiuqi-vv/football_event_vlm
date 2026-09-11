#!/usr/bin/env bash
set -euo pipefail

while pgrep -f '[e]val_long_video_structured_checkpoint.py' >/dev/null; do
  sleep 30
done
sleep 10

export INIT_CHECKPOINT='outputs/football_events/vitl16_featuremap_dual_roi_pr_correction_from_structured_e2_16f_hr/best.pt'
export OUTPUT_DIR='outputs/football_events/vitl16_featuremap_dual_roi_corrected_frame_aux_from_pr_e2_16f_hr'
export EPOCHS='1'
export PER_GPU_BATCH_SIZE='6'
export TARGET_EFFECTIVE_BATCH_SIZE='60'
export NUM_WORKERS_PER_GPU='2'
export PREFETCH_FACTOR='1'

exec bash scripts/run_football_featuremap_structured_roi_pr.sh '0,3,4,6,7'
