# D5 ROI residual-veto ranking experiment

## Objective

Keep the accepted E1 full-image model as the high-recall reference and use ROI
only to suppress strongly supported false positives. The verifier is not allowed
to create a detection or alter set-piece.

For shot and save:

```text
fused_logit = global_logit - roi_gate * softplus(veto_logit)
```

For invalid or low-confidence ROI, and for set-piece:

```text
fused_logit = global_logit
```

The global E1 branch is frozen. The local ROI branch, verifier gate, and veto
head are trainable.

## Training signal

The rank treatment uses the audited manifest
`outputs/football_hard_negatives/reviewed_mid_score_v2_shot_save_setpiece.json`.
It was mined from the exact E1 checkpoint and contains 2,403 windows. The
launcher repeats these records three times so a global micro-batch is likely to
contain both a positive and a hard negative.

The objective combines:

- fused clip BCE;
- positive retention against frozen E1 logits;
- shot/save pairwise margin ranking against mined hard negatives.

The checkpoint thresholds maximize per-class precision subject to validation
recall floors: shot 0.84, save 0.80, and set-piece 0.72.

## Commands

Use a global micro-batch of at least 16 so the pairwise rank loss is active in
most steps. On two A800 GPUs:

```bash
INIT_CHECKPOINT=/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt \
PER_GPU_BATCH_SIZE=8 \
bash scripts/run_football_roi_veto_rank_d5.sh control 0,1

INIT_CHECKPOINT=/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt \
PER_GPU_BATCH_SIZE=8 \
bash scripts/run_football_roi_veto_rank_d5.sh rank 2,3
```

For the highest-priority rank run on four A800 GPUs:

```bash
INIT_CHECKPOINT=/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt \
PER_GPU_BATCH_SIZE=4 \
bash scripts/run_football_roi_veto_rank_d5.sh rank 0,1,2,3
```

The launcher keeps the global optimizer batch near 80 and divides the target
global learning rates by GPU count.

## Acceptance

Compare every epoch with the same E1 checkpoint.

- shot/save recall drop is at most 1 percentage point;
- shot/save precision improves by at least 2 points, or FP falls by at least 10%;
- mAP/AUROC and `confidence_gap` do not decline;
- `positive_p10 - negative_p90` becomes less negative or positive;
- set-piece fused logits equal global logits;
- improvement holds on the six-video long-video holdout under PointNMS and
  window-overlap.

The training log reports `roi_gate_<label>`, `roi_veto_delta_<label>`,
`hard_negative_rank_loss`, and per-class confidence separation. A rising veto
delta with falling positive retention is a failure mode, even if sampled-val
precision increases.
