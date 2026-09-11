#!/usr/bin/env bash
set -euo pipefail
cd /home/new_users/qiuqi/code/dinov3-main
LOG=outputs/football_eval_runs/protocol_v2_m0_e1_gpu1.log
mkdir -p outputs/football_eval_runs
exec > >(tee -a "$LOG") 2>&1

echo "[$(date '+%F %T')] start protocol_v2_m0_e1_gpu1 cal29 on GPU 1"
python scripts/evaluate_football_model.py \
  --checkpoint checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt \
  --video-id-file configs/football/splits/vitl16_lvd1689m_set_piece_seed42_162videos/candidate_reranker_calibration_29_video_ids.txt \
  --run-name protocol_v2_m0_e1_cal29_ckptthr \
  --device cuda:1 \
  --gpu-ids 1 \
  --mode dense \
  --gt-dir /home/new_users/qiuqi/code/football_events_human_repair \
  --video-root xbotgo_0608=/mnt/data_16t/football/raw_video_720P \
  --output-root outputs/football_eval_runs \
  --clip-sec 10 \
  --stride-sec 5 \
  --thresholds checkpoint \
  --prediction-postprocess point_nms \
  --nms-radius-sec 5 \
  --match-tolerance-sec 5 \
  --save-frame-event-logits \
  --frame-event-topk 8 \
  --batch-size 8 \
  --num-workers 1 \
  --force


echo "[$(date '+%F %T')] tune cal29 PointNMS thresholds"
THR=$(python scripts/tune_pointnms_thresholds_from_dense_run.py \
  --run-dir outputs/football_eval_runs/protocol_v2_m0_e1_cal29_ckptthr \
  --video-id-file configs/football/splits/vitl16_lvd1689m_set_piece_seed42_162videos/candidate_reranker_calibration_29_video_ids.txt \
  --labels shot,save,set_piece \
  --match-tolerance-sec 5 \
  --nms-radius-sec 5 \
  --output outputs/football_eval_runs/protocol_v2_m0_e1_cal29_thresholds_pointnms_f1.json | head -n 1)
echo "[$(date '+%F %T')] cal29 thresholds: $THR"

echo "[$(date '+%F %T')] start protocol_v2_m0_e1_gpu1 35val with frozen cal29 thresholds"
python scripts/evaluate_football_model.py \
  --checkpoint checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt \
  --video-id-file configs/football/splits/vitl16_lvd1689m_set_piece_seed42_162videos/val_video_ids.txt \
  --run-name protocol_v2_m0_e1_35val_cal29_pointnms_f1thr \
  --device cuda:1 \
  --gpu-ids 1 \
  --thresholds "$THR" \
  --mode dense \
  --gt-dir /home/new_users/qiuqi/code/football_events_human_repair \
  --video-root xbotgo_0608=/mnt/data_16t/football/raw_video_720P \
  --output-root outputs/football_eval_runs \
  --clip-sec 10 \
  --stride-sec 5 \
  --prediction-postprocess point_nms \
  --nms-radius-sec 5 \
  --match-tolerance-sec 5 \
  --save-frame-event-logits \
  --frame-event-topk 8 \
  --batch-size 8 \
  --num-workers 1 \
  --force

echo "[$(date '+%F %T')] complete protocol_v2_m0_e1_gpu1"
