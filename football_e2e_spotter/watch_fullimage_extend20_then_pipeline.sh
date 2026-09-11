#!/usr/bin/env bash
# Finish the active 8-epoch baseline, resume it through epoch 20, then enter
# the verified Stage-1 -> full-image Verifier pipeline.
set -euo pipefail

baseline=/mnt/data_16t/qiuqi/outputs/football_events/vitl16_d7c_fullimage_latest_split_dino_pretrained_lora_ddp_e8_20260828
work=football_e2e_spotter/experiments/set_spotter_v1_parallel
log="$work/fullimage_extend20_watchdog.log"
pattern='[t]orchrun.*vitl16_d7c_fullimage_latest_split_dino_pretrained_lora_ddp_e8_20260828'
mkdir -p "$work"
echo "$(date -Is) extend20_watchdog_started" >> "$log"

while pgrep -f "$pattern" >/dev/null; do
  echo "$(date -Is) initial_8epoch_baseline_running" >> "$log"
  sleep 120
done
if [[ ! -s "$baseline/last.pt" || ! -s "$baseline/best.pt" ]]; then
  echo "$(date -Is) initial_baseline_incomplete: missing last.pt/best.pt" >> "$log"
  exit 10
fi
if ! python -c 'import sys,torch; c=torch.load(sys.argv[1],map_location="cpu",weights_only=False); sys.exit(0 if int(c.get("epoch",0)) >= 8 else 1)' "$baseline/last.pt"; then
  echo "$(date -Is) initial_baseline_incomplete: last checkpoint epoch below 8" >> "$log"
  exit 11
fi

echo "$(date -Is) resume_to_epoch20_start" >> "$log"
CUDA_VISIBLE_DEVICES=0,1,2,7 PYTHONUNBUFFERED=1 \
  torchrun --standalone --nproc_per_node=4 train_football_events.py \
  --config "$baseline/config.yaml" \
  "output_dir=$baseline" \
  "train.epochs=20" \
  "train.save_epoch_checkpoints=true" \
  "train.resume.enabled=true" \
  "train.resume.checkpoint=$baseline/last.pt" \
  "train.resume.strict=true" \
  "train.resume.load_optimizer=false" \
  "train.resume.load_scheduler=false" \
  "train.resume.load_scaler=false" \
  "train.ddp_find_unused_parameters=true" \
  >> "$baseline/resume20_console.log" 2>&1
if ! python -c 'import sys,torch; c=torch.load(sys.argv[1],map_location="cpu",weights_only=False); sys.exit(0 if int(c.get("epoch",0)) >= 20 else 1)' "$baseline/last.pt"; then
  echo "$(date -Is) resume20_incomplete: last checkpoint epoch below 20" >> "$log"
  exit 12
fi
echo "$(date -Is) resume_to_epoch20_verified" >> "$log"

exec bash football_e2e_spotter/watch_fullimage_pipeline_v4.sh
