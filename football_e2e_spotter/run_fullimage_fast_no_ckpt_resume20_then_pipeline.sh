#!/usr/bin/env bash
# Throughput profile: avoid activation recomputation at the original effective
# batch. If it does not fit, automatically restore the proven checkpointed run.
set -euo pipefail

baseline=/mnt/data_16t/qiuqi/outputs/football_events/vitl16_d7c_fullimage_latest_split_dino_pretrained_lora_ddp_e8_20260828
work=football_e2e_spotter/experiments/set_spotter_v1_parallel
watch_log="$work/fullimage_fast_resume20_watchdog.log"
no_ckpt_log="$baseline/resume20_fast_b2a10_no_ckpt_console.log"
fallback_log="$baseline/resume20_stable_b2a10_ckpt_console.log"
mkdir -p "$work"

run_resume() {
  local checkpointing=$1
  local run_log=$2
  echo "$(date -Is) resume_start batch=2 accum=10 gradient_checkpointing=$checkpointing target_epoch=20" >> "$watch_log"
  CUDA_VISIBLE_DEVICES=0,1,2,7 PYTHONUNBUFFERED=1 \
    torchrun --standalone --nproc_per_node=4 train_football_events.py \
    --config "$baseline/config.yaml" \
    "output_dir=$baseline" \
    "model.gradient_checkpointing=$checkpointing" \
    "train.epochs=20" \
    "train.per_gpu_batch_size=2" \
    "train.grad_accum_steps=10" \
    "eval.per_gpu_batch_size=4" \
    "train.save_epoch_checkpoints=true" \
    "train.resume.enabled=true" \
    "train.resume.checkpoint=$baseline/last.pt" \
    "train.resume.strict=true" \
    "train.resume.load_optimizer=false" \
    "train.resume.load_scheduler=false" \
    "train.resume.load_scaler=false" \
    "train.ddp_find_unused_parameters=true" \
    >> "$run_log" 2>&1
}

set +e
run_resume false "$no_ckpt_log"
status=$?
set -e
if (( status != 0 )); then
  if rg -i 'CUDA out of memory|out of memory|CUDNN_STATUS_NOT_SUPPORTED' "$no_ckpt_log" >/dev/null; then
    echo "$(date -Is) no_checkpoint_oom; restoring_checkpointed_batch2" >> "$watch_log"
    run_resume true "$fallback_log"
  else
    echo "$(date -Is) no_checkpoint_failed status=$status; no_unsafe_fallback" >> "$watch_log"
    exit "$status"
  fi
fi

epoch=$(python -c 'import sys,torch; c=torch.load(sys.argv[1],map_location="cpu",weights_only=False); print(int(c.get("epoch",0)))' "$baseline/last.pt")
if (( epoch < 20 )); then
  echo "$(date -Is) resume_incomplete epoch=$epoch" >> "$watch_log"
  exit 40
fi
echo "$(date -Is) resume_verified epoch=$epoch" >> "$watch_log"

exec bash football_e2e_spotter/watch_fullimage_pipeline_v4.sh
