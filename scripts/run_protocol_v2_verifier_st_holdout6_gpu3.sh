#!/usr/bin/env bash
set -euo pipefail
cd /home/new_users/qiuqi/code/dinov3-main
LOG=outputs/football_eval_runs/protocol_v2_verifier_st_holdout6_gpu3.log
mkdir -p outputs/football_eval_runs
exec > >(tee -a "$LOG") 2>&1

echo "[$(date '+%F %T')] start verifier ST holdout6 dense on GPU 3"
python scripts/evaluate_football_model.py   --checkpoint outputs/football_events/vitl16_weekend_event_anchor_512x896_sym20_st/best.pt   --mode dense   --video-id-file configs/football/splits/vitl16_lvd1689m_set_piece_seed42_162videos/candidate_reranker_holdout_6_video_ids.txt   --gt-dir /home/new_users/qiuqi/code/football_events_human_repair   --video-root xbotgo_0608=/mnt/data_16t/football/raw_video_720P   --output-root outputs/football_eval_runs   --run-name protocol_v2_verifier_st_holdout6_ckptthr   --clip-sec 10   --stride-sec 5   --thresholds checkpoint   --prediction-postprocess point_nms   --nms-radius-sec 5   --match-tolerance-sec 5   --save-frame-event-logits   --frame-event-topk 8   --batch-size 8   --num-workers 1   --device cuda:3   --gpu-ids 3   --force

echo "[$(date '+%F %T')] complete verifier ST holdout6"
