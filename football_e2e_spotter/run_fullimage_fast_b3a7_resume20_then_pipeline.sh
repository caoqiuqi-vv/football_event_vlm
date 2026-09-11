#!/usr/bin/env bash
# Stable fast profile selected after the batch=4 memory probe OOMed.
set -euo pipefail

baseline=/mnt/data_16t/qiuqi/outputs/football_events/vitl16_d7c_fullimage_latest_split_dino_pretrained_lora_ddp_e8_20260828
work=football_e2e_spotter/experiments/set_spotter_v1_parallel
watch_log="$work/fullimage_fast_resume20_watchdog.log"
run_log="$baseline/resume20_fast_b3a7_no_unused_console.log"
mkdir -p "$work"

echo "$(date -Is) stable_resume_start batch=3 accum=7 find_unused=false target_epoch=20" >> "$watch_log"
CUDA_VISIBLE_DEVICES=0,1,2,7 PYTHONUNBUFFERED=1 \
  torchrun --standalone --nproc_per_node=4 train_football_events.py \
  --config "$baseline/config.yaml" \
  "output_dir=$baseline" \
  "train.epochs=20" \
  "train.per_gpu_batch_size=3" \
  "train.grad_accum_steps=7" \
  "eval.per_gpu_batch_size=4" \
  "train.save_epoch_checkpoints=true" \
  "train.resume.enabled=true" \
  "train.resume.checkpoint=$baseline/last.pt" \
  "train.resume.strict=true" \
  "train.resume.load_optimizer=false" \
  "train.resume.load_scheduler=false" \
  "train.resume.load_scaler=false" \
  "train.ddp_find_unused_parameters=false" \
  >> "$run_log" 2>&1

epoch=$(python -c 'import sys,torch; c=torch.load(sys.argv[1],map_location="cpu",weights_only=False); print(int(c.get("epoch",0)))' "$baseline/last.pt")
if (( epoch < 20 )); then
  echo "$(date -Is) stable_resume_incomplete epoch=$epoch" >> "$watch_log"
  exit 30
fi
echo "$(date -Is) stable_resume_verified epoch=$epoch" >> "$watch_log"

exec bash football_e2e_spotter/watch_fullimage_pipeline_v4.sh
