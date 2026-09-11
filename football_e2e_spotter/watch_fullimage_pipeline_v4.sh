#!/usr/bin/env bash
# Verified full-image/no-ROI pipeline with explicit Stage-1 recall gate.
set -euo pipefail

baseline=/mnt/data_16t/qiuqi/outputs/football_events/vitl16_d7c_fullimage_latest_split_dino_pretrained_lora_ddp_e8_20260828
work=football_e2e_spotter/experiments/set_spotter_v1_parallel
log="$work/fullimage_pipeline_watchdog.log"
pattern='[t]orchrun.*vitl16_d7c_fullimage_latest_split_dino_pretrained_lora_ddp_e8_20260828'
train_ids=football_longform_v2/experiments/lf_a0_official_fullscale/canonical_train_media_ids.txt
mkdir -p "$work"
echo "$(date -Is) fullimage_pipeline_v4_watchdog_started" >> "$log"

while pgrep -f "$pattern" >/dev/null; do
  echo "$(date -Is) baseline_running" >> "$log"
  sleep 120
done
if [[ ! -s "$baseline/last.pt" || ! -s "$baseline/best.pt" ]]; then
  echo "$(date -Is) baseline_failed_or_incomplete: missing last.pt/best.pt" >> "$log"; exit 10
fi
if ! python -c 'import sys,torch; c=torch.load(sys.argv[1],map_location="cpu",weights_only=False); sys.exit(0 if int(c.get("epoch",0)) >= 8 else 1)' "$baseline/last.pt"; then
  echo "$(date -Is) baseline_failed_or_incomplete: last checkpoint epoch below 8" >> "$log"; exit 11
fi
echo "$(date -Is) baseline_verified" >> "$log"

echo "$(date -Is) stage1_start" >> "$log"
PYTHONPATH=.:football_e2e_spotter/src python football_e2e_spotter/run_parallel_stage1.py \
  --config football_e2e_spotter/configs/set_spotter_v1_runtime.yaml \
  --work "$work" --gpus 0,1,2,7 --epochs 12 --wait-cache >> "$log" 2>&1
for fold in 0 1 2 3 4; do
  [[ -s "$work/oof/fold${fold}.jsonl" ]] || { echo "$(date -Is) stage1_missing_fold=${fold}" >> "$log"; exit 20; }
done
[[ -s "$work/calibration/candidates.jsonl" ]] || { echo "$(date -Is) stage1_missing_calibration_candidates" >> "$log"; exit 21; }

PYTHONPATH=.:football_e2e_spotter/src python football_e2e_spotter/merge_jsonl.py \
  --output "$work/oof/candidates.jsonl" \
  "$work/oof/fold0.jsonl" "$work/oof/fold1.jsonl" "$work/oof/fold2.jsonl" \
  "$work/oof/fold3.jsonl" "$work/oof/fold4.jsonl" >> "$log" 2>&1
[[ -s "$work/oof/candidates.jsonl" ]] || { echo "$(date -Is) oof_merge_empty" >> "$log"; exit 22; }

echo "$(date -Is) stage1_candidate_recall_gate" >> "$log"
PYTHONPATH=.:football_e2e_spotter/src python football_e2e_spotter/evaluate_stage1_candidate_recall.py \
  --candidates "$work/oof/candidates.jsonl" \
  --annotations /mnt/data_16t/football/football_events_human_repair \
  --ids "$train_ids" --output "$work/oof/stage1_candidate_recall.json" >> "$log" 2>&1
echo "$(date -Is) stage1_verified_no_nms" >> "$log"

echo "$(date -Is) verifier_start full_image_only checkpointed uncertainty_aware" >> "$log"
CUDA_VISIBLE_DEVICES=0,1,2,7 PYTHONUNBUFFERED=1 PYTHONPATH=.:football_e2e_spotter/src \
  torchrun --standalone --nproc_per_node=4 football_e2e_spotter/train_set_verifier_fullimage_checkpointed_ddp_v4.py \
  --candidates "$work/oof/candidates.jsonl" \
  --annotations /mnt/data_16t/football/football_events_human_repair \
  --output "$work/verifier_fullimage_checkpointed_v4" --epochs 6 --frame-chunk-size 4 >> "$log" 2>&1
[[ -s "$work/verifier_fullimage_checkpointed_v4/last.pt" ]] || { echo "$(date -Is) verifier_missing_checkpoint" >> "$log"; exit 30; }

echo "$(date -Is) calibration_start" >> "$log"
CUDA_VISIBLE_DEVICES=0,1,2,7 PYTHONUNBUFFERED=1 PYTHONPATH=.:football_e2e_spotter/src \
  torchrun --standalone --nproc_per_node=4 football_e2e_spotter/evaluate_set_pipeline_fullimage_checkpointed_ddp.py \
  --candidates "$work/calibration/candidates.jsonl" \
  --annotations /mnt/data_16t/football/football_events_human_repair \
  --verifier "$work/verifier_fullimage_checkpointed_v4/last.pt" \
  --output "$work/calibration_fullimage_checkpointed_v4" >> "$log" 2>&1
[[ -s "$work/calibration_fullimage_checkpointed_v4/calibration_eval.json" ]] || { echo "$(date -Is) missing_calibration_eval" >> "$log"; exit 40; }
echo "$(date -Is) fullimage_pipeline_complete" >> "$log"
