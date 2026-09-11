#!/usr/bin/env bash
# Switch the active full-image baseline at a completed epoch boundary to a
# larger micro-batch, keep the effective global batch approximately constant,
# then continue into the verified Stage-1 -> Verifier pipeline.
set -euo pipefail

baseline=/mnt/data_16t/qiuqi/outputs/football_events/vitl16_d7c_fullimage_latest_split_dino_pretrained_lora_ddp_e8_20260828
work=football_e2e_spotter/experiments/set_spotter_v1_parallel
watch_log="$work/fullimage_fast_resume20_watchdog.log"
fast_log="$baseline/resume20_fast_b4a5_console.log"
fallback_log="$baseline/resume20_fast_b3a7_console.log"
initial_pattern='[t]orchrun.*vitl16_d7c_fullimage_latest_split_dino_pretrained_lora_ddp_e8_20260828.*train\.epochs=8'
mkdir -p "$work"

checkpoint_epoch() {
  python -c 'import sys,torch; c=torch.load(sys.argv[1],map_location="cpu",weights_only=False); print(int(c.get("epoch",0)))' "$1" 2>/dev/null || echo 0
}

echo "$(date -Is) fast_resume_watchdog_started" >> "$watch_log"
while true; do
  if [[ -s "$baseline/last.pt" && -s "$baseline/metrics_epoch_005.json" ]]; then
    epoch=$(checkpoint_epoch "$baseline/last.pt")
    if (( epoch >= 5 )); then
      break
    fi
  fi
  if ! pgrep -f "$initial_pattern" >/dev/null; then
    echo "$(date -Is) initial_training_stopped_before_epoch5_checkpoint" >> "$watch_log"
    exit 20
  fi
  sleep 20
done

echo "$(date -Is) epoch5_checkpoint_verified; stopping_initial_run" >> "$watch_log"
while read -r pgid; do
  [[ -n "$pgid" ]] || continue
  kill -TERM -- "-$pgid" 2>/dev/null || true
done < <(ps -eo pgid=,cmd= | awk '/[t]orchrun/ && /vitl16_d7c_fullimage_latest_split_dino_pretrained_lora_ddp_e8_20260828/ && /train\.epochs=8/ {gsub(/ /,"",$1); print $1}' | sort -u)

for _ in $(seq 1 30); do
  pgrep -f "$initial_pattern" >/dev/null || break
  sleep 2
done
if pgrep -f "$initial_pattern" >/dev/null; then
  echo "$(date -Is) initial_training_did_not_stop_cleanly" >> "$watch_log"
  exit 21
fi

run_resume() {
  local micro_batch=$1
  local accum=$2
  local run_log=$3
  echo "$(date -Is) resume_start batch=$micro_batch accum=$accum" >> "$watch_log"
  CUDA_VISIBLE_DEVICES=0,1,2,7 PYTHONUNBUFFERED=1 \
    torchrun --standalone --nproc_per_node=4 train_football_events.py \
    --config "$baseline/config.yaml" \
    "output_dir=$baseline" \
    "train.epochs=20" \
    "train.per_gpu_batch_size=$micro_batch" \
    "train.grad_accum_steps=$accum" \
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
run_resume 4 5 "$fast_log"
status=$?
set -e
if (( status != 0 )); then
  if rg -i 'CUDA out of memory|out of memory|CUDNN_STATUS_NOT_SUPPORTED' "$fast_log" >/dev/null; then
    echo "$(date -Is) batch4_oom; fallback_batch3_accum7" >> "$watch_log"
    run_resume 3 7 "$fallback_log"
  else
    echo "$(date -Is) batch4_failed status=$status; no_unsafe_fallback" >> "$watch_log"
    exit "$status"
  fi
fi

epoch=$(checkpoint_epoch "$baseline/last.pt")
if (( epoch < 20 )); then
  echo "$(date -Is) resume_incomplete epoch=$epoch" >> "$watch_log"
  exit 22
fi
echo "$(date -Is) resume20_verified epoch=$epoch" >> "$watch_log"

exec bash football_e2e_spotter/watch_fullimage_pipeline_v4.sh
