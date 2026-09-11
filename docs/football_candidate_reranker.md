# Football candidate reranker

This is a deferred precision stage for a strong base DINO event model. It does
not update DINO or the temporal head. It learns a small per-class logistic
reranker from saved long-video clip scores and frame-event scores, then uses the
new score before PointNMS.

Do not use this stage to hide a weak base model. Start it only after the base
model has acceptable PointNMS recall and ranking quality.

## Data isolation

The six commonly reported long videos are part of the 35-video validation
split. The repository therefore fixes:

- 29 videos for reranker training and threshold selection:
  `configs/football/splits/vitl16_lvd1689m_set_piece_seed42_162videos/candidate_reranker_calibration_29_video_ids.txt`
- 6 videos held out from reranker fitting:
  `configs/football/splits/vitl16_lvd1689m_set_piece_seed42_162videos/candidate_reranker_holdout_6_video_ids.txt`

This prevents reranker-level leakage. It is not a fully independent model test:
the base checkpoint may already have used the complete validation split for
checkpoint selection.

## Required inference artifacts

Each video directory must contain:

```text
window_predictions.csv
frame_event_logits.csv
gt_events.csv
```

Generate the 29-video calibration output with the final base checkpoint:

```bash
python scripts/evaluate_football_model.py \
  --checkpoint /path/to/base_model.pt \
  --mode dense \
  --video-id-file configs/football/splits/vitl16_lvd1689m_set_piece_seed42_162videos/candidate_reranker_calibration_29_video_ids.txt \
  --gt-dir /home/new_users/qiuqi/code/football_events_human_repair \
  --video-root xbotgo_0608=/mnt/data_16t/football/raw_video_720P \
  --run-name base_model_candidate_calibration29 \
  --clip-sec 10 \
  --stride-sec 5 \
  --thresholds checkpoint \
  --prediction-postprocess point_nms \
  --nms-radius-sec 5 \
  --match-tolerance-sec 5 \
  --save-frame-event-logits \
  --frame-event-topk 8
```

Generate the six holdout videos with the corresponding holdout ID file and a
different run name.

## Fit and evaluate

```bash
python scripts/fit_football_candidate_reranker.py \
  --calibration-run-dir outputs/football_eval_runs/base_model_candidate_calibration29 \
  --calibration-video-id-file configs/football/splits/vitl16_lvd1689m_set_piece_seed42_162videos/candidate_reranker_calibration_29_video_ids.txt \
  --holdout-run-dir outputs/football_eval_runs/base_model_candidate_holdout6 \
  --holdout-video-id-file configs/football/splits/vitl16_lvd1689m_set_piece_seed42_162videos/candidate_reranker_holdout_6_video_ids.txt \
  --output-dir outputs/football_candidate_rerankers/base_model_v1 \
  --labels shot,save \
  --baseline-thresholds shot=0.15,save=0.20 \
  --match-tolerance-sec 5 \
  --ignore-radius-sec 8 \
  --recall-drop-tolerance 0.01
```

The script uses video-grouped cross-validation. Candidates within 5 seconds of
GT are positive, candidates from 5 to 8 seconds are ignored during fitting, and
farther candidates are negative. It selects a PointNMS threshold that maximizes
precision while keeping recall within one percentage point of the clip-only
baseline.

Outputs:

```text
calibrator.json
candidate_features.csv
oof_predictions.csv
oof_metrics.json
oof_per_video_metrics.csv
holdout_predictions.csv
holdout_metrics.json
holdout_per_video_metrics.csv
```

Accept the reranker only when holdout precision improves by at least two
percentage points or FP falls by at least 10%, while recall falls by no more
than one percentage point. Keep set-piece on the original clip score until its
frame-event discrimination improves.
