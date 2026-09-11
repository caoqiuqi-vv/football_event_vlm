#!/usr/bin/env bash
# Explicit, observable hand-off for the no-NMS full-image verifier experiment.
set -euo pipefail

work=football_e2e_spotter/experiments/set_spotter_v1_parallel
log="$work/stage2_fullimage_console.log"
mkdir -p "$work"

echo "$(date -Is) stage2_fullimage_watchdog_started" >> "$log"
while ! pgrep -f '[r]un_parallel_stage1.py.*set_spotter_v1_parallel' >/dev/null; do
  echo "$(date -Is) waiting_for_stage1_start" >> "$log"
  sleep 120
done
echo "$(date -Is) stage1_detected" >> "$log"
while pgrep -f '[r]un_parallel_stage1.py.*set_spotter_v1_parallel' >/dev/null; do
  echo "$(date -Is) waiting_for_stage1_completion" >> "$log"
  sleep 120
done
echo "$(date -Is) stage1_complete; merging_oof" >> "$log"

PYTHONPATH=.:football_e2e_spotter/src python football_e2e_spotter/merge_jsonl.py \
  --output "$work/oof/candidates.jsonl" \
  "$work/oof/fold0.jsonl" "$work/oof/fold1.jsonl" "$work/oof/fold2.jsonl" \
  "$work/oof/fold3.jsonl" "$work/oof/fold4.jsonl" >> "$log" 2>&1

echo "$(date -Is) verifier_fullimage_train_start" >> "$log"
CUDA_VISIBLE_DEVICES=0,1,2,7 PYTHONUNBUFFERED=1 PYTHONPATH=.:football_e2e_spotter/src \
  torchrun --standalone --nproc_per_node=4 football_e2e_spotter/train_set_verifier_fullimage_ddp.py \
  --candidates "$work/oof/candidates.jsonl" \
  --annotations /mnt/data_16t/football/football_events_human_repair \
  --output "$work/verifier_fullimage" --epochs 6 >> "$log" 2>&1

echo "$(date -Is) verifier_fullimage_calibration_start" >> "$log"
CUDA_VISIBLE_DEVICES=0,1,2,7 PYTHONUNBUFFERED=1 PYTHONPATH=.:football_e2e_spotter/src \
  torchrun --standalone --nproc_per_node=4 football_e2e_spotter/evaluate_set_pipeline_fullimage_ddp.py \
  --candidates "$work/calibration/candidates.jsonl" \
  --annotations /mnt/data_16t/football/football_events_human_repair \
  --verifier "$work/verifier_fullimage/last.pt" \
  --output "$work/calibration_fullimage" >> "$log" 2>&1
echo "$(date -Is) stage2_fullimage_complete" >> "$log"
