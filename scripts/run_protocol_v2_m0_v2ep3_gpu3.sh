#!/usr/bin/env bash
set -euo pipefail
cd /home/new_users/qiuqi/code/dinov3-main
LOG=outputs/football_eval_runs/protocol_v2_m0_v2ep3_gpu3.log
mkdir -p outputs/football_eval_runs
exec > >(tee -a "$LOG") 2>&1

CKPT="outputs/football_events/vitl16_weekend_event_anchor_512x896_sym20_st_xbotgo0807_pos_ft_v2_anchor3_7_rawmissing/epoch_3.pt"
CAL_IDS="configs/football/splits/vitl16_lvd1689m_set_piece_seed42_162videos/candidate_reranker_calibration_29_video_ids.txt"
VAL_IDS="configs/football/splits/vitl16_lvd1689m_set_piece_seed42_162videos/val_video_ids.txt"

if [[ ! -f "$CKPT" ]]; then
  echo "[$(date '+%F %T')] ERROR checkpoint missing: $CKPT"
  exit 2
fi

echo "[$(date '+%F %T')] start protocol_v2_m0_v2ep3_gpu3 cal29 on GPU 3 checkpoint=$CKPT"
python scripts/evaluate_football_model.py   --checkpoint "$CKPT"   --video-id-file "$CAL_IDS"   --run-name protocol_v2_m0_v2ep3_cal29_ckptthr   --device cuda:3   --gpu-ids 3   --mode dense   --gt-dir /home/new_users/qiuqi/code/football_events_human_repair   --video-root xbotgo_0608=/mnt/data_16t/football/raw_video_720P   --output-root outputs/football_eval_runs   --clip-sec 10   --stride-sec 5   --thresholds checkpoint   --prediction-postprocess point_nms   --nms-radius-sec 5   --match-tolerance-sec 5   --save-frame-event-logits   --frame-event-topk 8   --batch-size 8   --num-workers 1   --force


echo "[$(date '+%F %T')] tune v2ep3 cal29 PointNMS-F1 thresholds"
THR=$(python scripts/tune_pointnms_thresholds_from_dense_run.py   --run-dir outputs/football_eval_runs/protocol_v2_m0_v2ep3_cal29_ckptthr   --video-id-file "$CAL_IDS"   --labels shot,save,set_piece   --match-tolerance-sec 5   --nms-radius-sec 5   --output outputs/football_eval_runs/protocol_v2_m0_v2ep3_cal29_thresholds_pointnms_f1.json | head -n 1)
echo "[$(date '+%F %T')] v2ep3 cal29 thresholds: $THR"

echo "[$(date '+%F %T')] start protocol_v2_m0_v2ep3_gpu3 35val with frozen cal29 thresholds"
python scripts/evaluate_football_model.py   --checkpoint "$CKPT"   --video-id-file "$VAL_IDS"   --run-name protocol_v2_m0_v2ep3_35val_cal29_pointnms_f1thr   --device cuda:3   --gpu-ids 3   --thresholds "$THR"   --mode dense   --gt-dir /home/new_users/qiuqi/code/football_events_human_repair   --video-root xbotgo_0608=/mnt/data_16t/football/raw_video_720P   --output-root outputs/football_eval_runs   --clip-sec 10   --stride-sec 5   --prediction-postprocess point_nms   --nms-radius-sec 5   --match-tolerance-sec 5   --save-frame-event-logits   --frame-event-topk 8   --batch-size 8   --num-workers 1   --force

echo "[$(date '+%F %T')] complete protocol_v2_m0_v2ep3_gpu3"
