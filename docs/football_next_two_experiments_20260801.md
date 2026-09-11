# Football Event Model: Next Two Experiments

Date: 2026-08-01

## Current Evidence

Primary long-video baseline:

- Checkpoint: `checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt`
- Six-video window-overlap, checkpoint thresholds:
  - micro P/R/F1: `0.3538 / 0.8025 / 0.4911`
  - shot P/R: `0.3994 / 0.8836`
  - save P/R: `0.2137 / 0.8406`
  - set_piece P/R: `0.5356 / 0.6566`
- Main issue: precision is dominated by save and shot false positives.

Candidate reranker:

- 512x896 reranker holdout micro degraded from `0.2618 / 0.6977` to
  `0.2439 / 0.6930`.
- Conclusion: current post-hoc candidate features do not separate TP/FP well
  enough; do not keep tuning the same reranker first.

E2 attention-pool temporal head:

- Checkpoint: `/home/new_users/qiuqi/code/dinov3-main/vitl16_lora_r5_f32_16f_e2_attn_pool_epoch3.pt`
- Sampled-val comparison vs E1:
  - tuned shot/save recall guard passes, precision is slightly better.
  - set_piece recall drops heavily in sampled-val, so E2 should not replace all
    classes directly.
- Conclusion: E2 is useful mainly as a shot/save candidate, while set_piece
  should stay on E1 unless long-video evidence says otherwise.

Event-topk:

- The useful root `train_console.log` run is `event_topk_512`.
- Epoch 4/5 are the best sampled-val balance:
  - tuned micro P/R/F1: `0.6500 / 0.7221 / 0.6842`
  - thresholds: shot `0.35`, save `0.45`, set_piece `0.45`
- Epoch 6 increases tuned recall but needs very low thresholds:
  - shot threshold `0.10`, save threshold `0.20`
  - default recall drops strongly.
- Conclusion: event-topk is not the next highest-confidence direction unless its
  epoch 4/5 checkpoints are available for long-video validation. Very low tuned
  thresholds are a warning sign for long-video FP.

Hard-negative continuation:

- Previous hard-negative result improved sampled/short validation precision but
  failed the long-video recall guard.
- Conclusion: hard negatives are still promising, but the next runnable version
  should be conservative and label-specific first. A teacher-retention loss can
  be added later if this cleaner mining setup still loses recall.

## Experiment 1: Class-Routed Temporal Head Evaluation

Hypothesis:

Use E2 attention-pool scores only for shot/save, and keep E1 scores for
set_piece. This targets the classes where E2 looks helpful while avoiding the
set_piece recall drop seen in sampled-val.

Why this is high-probability:

- E2 improves or preserves shot/save sampled-val precision/recall.
- set_piece is known to be harmed by E2, so class routing avoids the most
  obvious failure mode.
- No training is required; this is a fast long-video validation.

Implementation status:

- Added `scripts/route_football_eval_runs.py`.
- The script routes saved per-window probabilities from already-computed dense
  eval runs, then recomputes standard `point_nms`, `window_overlap`, or
  `interval_merge` metrics.
- It does not rerun the model, so the routing step itself is cheap. The expensive
  part is producing both E1 and E2 dense eval outputs.

Runtime cost:

- Offline validation cost is roughly two model eval passes, because E1 and E2
  are currently separate checkpoints.
- If deployed literally as two model instances, online inference cost is also
  roughly 2x backbone cost, which is not ideal.
- If the routed result wins, implement a single trainable routed model with a
  shared DINO/LoRA backbone and class-specific temporal heads:
  - E2/attention-pool head for `shot,save`
  - E1/CLS-transformer head for `set_piece`
  - final logits assembled per label
- That trainable version should add only small temporal-head overhead compared
  with DINO inference. The current code does not yet implement this fused
  trainable routed head; the eval script is the proof step before adding it.

Suggested runs:

```bash
cd /home/new_users/qiuqi/code/dinov3-main

# Copy the new local routing script to the server if it has not been pushed/pulled yet.
scp -F /dev/null -i /Users/caoqiuqi/.ssh/id_rsa \
  /Users/caoqiuqi/Desktop/code/event_classification_vlm/scripts/route_football_eval_runs.py \
  qiuqi@119.147.202.180:/home/new_users/qiuqi/code/dinov3-main/scripts/

# E1 standalone, six-video long eval, same protocol as routed eval.
CUDA_VISIBLE_DEVICES=0 /home/new_users/qiuqi/miniconda3/bin/python scripts/evaluate_football_model.py \
  --checkpoint checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt \
  --mode dense \
  --video-ids 2027564888428580866,2027572095412195330,2027572406738604033,2042125172076392450,2042520971893485569,2042525152494694401 \
  --gt-dir /home/new_users/qiuqi/code/football_events_human_repair \
  --video-root xbotgo_0608=/mnt/data_16t/football/raw_video_720P \
  --output-root outputs/football_eval_runs \
  --run-name e1_frame_det_last_6videos_point_nms_ckpt_thr \
  --clip-sec 10 \
  --stride-sec 5 \
  --batch-size 1 \
  --num-workers 2 \
  --device cuda:0 \
  --gpu-ids 0 \
  --thresholds checkpoint \
  --prediction-postprocess point_nms \
  --nms-radius-sec 5 \
  --match-tolerance-sec 5 \
  --spatial-crop-mode robust_detector_aware \
  --detector-index-root outputs/football_roi_indices/robust_v2

# E2 standalone, six-video long eval.
CUDA_VISIBLE_DEVICES=0 /home/new_users/qiuqi/miniconda3/bin/python scripts/evaluate_football_model.py \
  --checkpoint vitl16_lora_r5_f32_16f_e2_attn_pool_epoch3.pt \
  --mode dense \
  --video-ids 2027564888428580866,2027572095412195330,2027572406738604033,2042125172076392450,2042520971893485569,2042525152494694401 \
  --gt-dir /home/new_users/qiuqi/code/football_events_human_repair \
  --video-root xbotgo_0608=/mnt/data_16t/football/raw_video_720P \
  --output-root outputs/football_eval_runs \
  --run-name e2_attn_pool_epoch3_6videos_point_nms_ckpt_thr \
  --clip-sec 10 \
  --stride-sec 5 \
  --batch-size 1 \
  --num-workers 2 \
  --device cuda:0 \
  --gpu-ids 0 \
  --thresholds checkpoint \
  --prediction-postprocess point_nms \
  --nms-radius-sec 5 \
  --match-tolerance-sec 5 \
  --spatial-crop-mode robust_detector_aware \
  --detector-index-root outputs/football_roi_indices/robust_v2

# Route shot/save from E2 and set_piece from E1.
/home/new_users/qiuqi/miniconda3/bin/python scripts/route_football_eval_runs.py \
  --source-run e1=outputs/football_eval_runs/e1_frame_det_last_6videos_point_nms_ckpt_thr \
  --source-run e2=outputs/football_eval_runs/e2_attn_pool_epoch3_6videos_point_nms_ckpt_thr \
  --label-source shot=e2,save=e2,set_piece=e1 \
  --output-dir outputs/football_eval_runs/e2_shot_save_e1_set_piece_6videos_point_nms_ckpt_thr \
  --thresholds checkpoint \
  --prediction-postprocess point_nms \
  --nms-radius-sec 5 \
  --match-tolerance-sec 5

# Optional diagnostic: same routed probabilities under window-overlap recall accounting.
/home/new_users/qiuqi/miniconda3/bin/python scripts/route_football_eval_runs.py \
  --source-run e1=outputs/football_eval_runs/e1_frame_det_last_6videos_point_nms_ckpt_thr \
  --source-run e2=outputs/football_eval_runs/e2_attn_pool_epoch3_6videos_point_nms_ckpt_thr \
  --label-source shot=e2,save=e2,set_piece=e1 \
  --output-dir outputs/football_eval_runs/e2_shot_save_e1_set_piece_6videos_window_overlap_ckpt_thr \
  --thresholds checkpoint \
  --prediction-postprocess window_overlap \
  --match-tolerance-sec 5
```

Acceptance gate:

- On shot+save: precision improves by at least `+2pp` or FP drops by at least
  `10%`, while recall drop is no worse than `1pp`.
- On all classes: set_piece recall must not drop when using the routed
  E1-for-set_piece variant.
- If E2 standalone fails only because of set_piece, still keep the routed
  variant as a candidate.

Stop condition:

- If shot/save precision does not improve on long-video point_nms, do not spend
  more time on attention-pool training.

## Experiment 2: Recall-Safe Hard-Negative Mining

Hypothesis:

The model needs precision pressure from real long-video false positives. The
previous hard-negative run likely used too many or too loose negatives, so it
could improve sampled validation while hurting long-video recall. Use
label-specific masks and conservative mining to suppress repeated shot/save FP
without converting nearby real events into negatives.

Why this is high-probability:

- Long-video errors are dominated by repeated shot/save FP.
- Training loader already supports label-specific hard-negative masks: an item
  with `"label": "save"` only applies negative pressure to `save`.
- The loader revalidates mined windows against the current repaired annotations
  with `safety_margin_sec`, so stale manifests are less dangerous.
- Previous defaults were aggressive: `shot=0.30,save=0.40`,
  `max_per_video_per_label=40`, and no `reject-any-label-gt` by default.

Training design:

- Initialize from `checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt`.
- Mine hard negatives only for `shot,save` first.
- Use higher mining thresholds and a small cap:
  - primary: `shot=0.55,save=0.55`
  - backup if too few samples: `shot=0.45,save=0.45`
- Use `--reject-any-label-gt` and `tolerance_sec=8` during mining.
- Train only 1-2 epochs with lower LR, then choose checkpoints by long-video
  recall guard, not sampled validation alone.

Local mining commands:

```bash
cd /Users/caoqiuqi/Desktop/code/event_classification_vlm

# Build a conservative manifest from locally synced dense eval outputs.
# This does not need GPU or video files; it only needs window_predictions.csv
# and gt_events.csv under the eval run directory.
mkdir -p outputs/football_hard_negatives
python3 scripts/build_football_hard_negative_manifest.py \
  --eval-run-dir outputs/football_eval_runs/e1_exp2_best_train_hard_negative_mining \
  --output outputs/football_hard_negatives/e1_train_fp_shot_save_recall_safe_055.json \
  --source xbotgo_0608 \
  --min-probs shot=0.55,save=0.55 \
  --tolerance-sec 8 \
  --max-per-video-per-label 6 \
  --reject-any-label-gt

# Quick audit: counts should be concentrated but not dominated by one video.
python3 - <<'PY'
import collections, json
p = "outputs/football_hard_negatives/e1_train_fp_shot_save_recall_safe_055.json"
d = json.load(open(p))
print("num_hard_negatives", d["num_hard_negatives"])
print("per_label_counts", d["per_label_counts"])
print("top_videos", collections.Counter(x["video_id"] for x in d["hard_negatives"]).most_common(10))
PY

# Copy manifest to the server.
ssh -F /dev/null -i /Users/caoqiuqi/.ssh/id_rsa qiuqi@119.147.202.180 \
  'mkdir -p /home/new_users/qiuqi/code/dinov3-main/outputs/football_hard_negatives'
scp -F /dev/null -i /Users/caoqiuqi/.ssh/id_rsa \
  outputs/football_hard_negatives/e1_train_fp_shot_save_recall_safe_055.json \
  qiuqi@119.147.202.180:/home/new_users/qiuqi/code/dinov3-main/outputs/football_hard_negatives/
```

Remote mining fallback if the train dense eval output is not available locally:

```bash
cd /home/new_users/qiuqi/code/dinov3-main

PYTHON_BIN=/home/new_users/qiuqi/miniconda3/bin/python \
E1_HARD_NEG_INIT_CHECKPOINT=checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt \
E1_HARD_NEG_MANIFEST=outputs/football_hard_negatives/e1_train_fp_shot_save_recall_safe_055.json \
E1_HARD_NEG_MIN_PROBS=shot=0.55,save=0.55 \
E1_HARD_NEG_REJECT_ANY_LABEL_GT=1 \
E1_HARD_NEG_REJECT_TOLERANCE_SEC=8 \
E1_HARD_NEG_MAX_PER_VIDEO_PER_LABEL=6 \
bash scripts/launch_football_detection_aware.sh mine_e1_hard_neg 0
```

Training command:

```bash
cd /home/new_users/qiuqi/code/dinov3-main

PYTHON_BIN=/home/new_users/qiuqi/miniconda3/bin/python \
TRAIN_LOG_FILE=outputs/football_events/vitl16_e1_hardneg_recall_safe_055/train_console.log \
E1_HARD_NEG_INIT_CHECKPOINT=checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt \
E1_HARD_NEG_MANIFEST=outputs/football_hard_negatives/e1_train_fp_shot_save_recall_safe_055.json \
bash scripts/launch_football_detection_aware.sh e1_hard_neg 0,1,2,3 \
  output_dir=outputs/football_events/vitl16_e1_hardneg_recall_safe_055 \
  train.epochs=2 \
  train.lr=0.00003 \
  train.backbone_lr=0.0 \
  train.checkpoint_selection=default_precision_at_recall_floor \
  train.checkpoint_min_recall=0.78 \
  data.long_video.hard_negative.max_per_video=12 \
  data.long_video.hard_negative.safety_margin_sec=8.0 \
  data.long_video.negative_ratio_by_split.train=5.0 \
  data.long_video.negative_ratio_by_split.val=2.0
```

Long-video evaluation command:

```bash
cd /home/new_users/qiuqi/code/dinov3-main

CUDA_VISIBLE_DEVICES=0 /home/new_users/qiuqi/miniconda3/bin/python scripts/evaluate_football_model.py \
  --checkpoint outputs/football_events/vitl16_e1_hardneg_recall_safe_055/best.pt \
  --mode dense \
  --video-ids 2027564888428580866,2027572095412195330,2027572406738604033,2042125172076392450,2042520971893485569,2042525152494694401 \
  --gt-dir /home/new_users/qiuqi/code/football_events_human_repair \
  --video-root xbotgo_0608=/mnt/data_16t/football/raw_video_720P \
  --output-root outputs/football_eval_runs \
  --run-name e1_hardneg_recall_safe_055_6videos_point_nms_ckpt_thr \
  --clip-sec 10 \
  --stride-sec 5 \
  --batch-size 1 \
  --num-workers 2 \
  --device cuda:0 \
  --gpu-ids 0 \
  --thresholds checkpoint \
  --prediction-postprocess point_nms \
  --nms-radius-sec 5 \
  --match-tolerance-sec 5 \
  --spatial-crop-mode robust_detector_aware \
  --detector-index-root outputs/football_roi_indices/robust_v2
```

Acceptance gate:

- Six-video point_nms:
  - shot+save FP drops at least `10%`.
  - shot+save recall drop is no worse than `1pp`.
  - all-class micro precision improves at least `+2pp`.
- If sampled-val improves but long-video recall fails, reject the run.

Stop condition:

- Stop after epoch 1 if long-video recall drops more than `2pp`.
- Stop after epoch 2 if FP reduction is below `5%`.
- If the conservative `0.55` manifest has too few examples, retry mining at
  `shot=0.45,save=0.45`, still with `--reject-any-label-gt`,
  `tolerance_sec=8`, and cap `6`.

## Execution Order

1. Run Experiment 1 first because it needs no training and can quickly decide
   whether E2 is useful for shot/save.
2. Run Experiment 2 next. Use the best available temporal head from Experiment 1
   as the student initialization:
   - if routed E2 helps, distill/fine-tune with E2 for shot/save and E1
     teacher retention;
   - otherwise use E1 as both initialization and teacher.
3. Keep event-topk only as a secondary branch. Do not continue it beyond the
   current checkpoint unless long-video epoch 4/5 evidence beats E1/E2 under the
   same point_nms protocol.
