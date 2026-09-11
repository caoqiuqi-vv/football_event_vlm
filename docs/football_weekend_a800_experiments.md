# Four-A800 weekend base-model experiments

## Objective

Improve base-model PointNMS precision while preserving or slightly increasing
recall. All experiments are single-view, retain Gaussian frame-event
supervision, and initialize from the same checkpoint:

```text
/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt
```

Candidate reranking and ROI verification are deliberately excluded.

The 16-frame hard-negative, E2, and E3 runs use 640x1120 input. Corrected E4
uses 512x896 input with 32 candidate frames to keep its memory and compute
bounded.

## Controlled runtime

Use two concurrent jobs with two A800 GPUs each. The reference checkpoint used
four GPUs with global micro-batch 40 and gradient accumulation 2: per-GPU
micro-batch 10 and effective optimizer batch 80. The launcher preserves that
optimizer batch:

```text
16-frame job: per-GPU 10, global micro-batch 20, accumulation 4
32-frame E4: per-GPU 4, global micro-batch 8, accumulation 10
global head LR: 3e-4
global LoRA LR: 3e-5
precision: BF16
gradient checkpointing: enabled
```

If E4 has enough memory, use `E4_PER_GPU_BATCH_SIZE=5`; accumulation then
becomes 8 and the effective batch remains 80. If a 16-frame variant cannot fit
10 samples per GPU, set `BASE_PER_GPU_BATCH_SIZE=8`; the launcher calculates
accumulation automatically.

The launcher writes `train_console.log` into each experiment output directory.

## Batch 1: highest-value experiments

Hard negatives target the observed false-positive distribution. This is a
low-LR continuation: head LR `5e-5`, LoRA LR `5e-6`, hard-negative loss
weight `4`, repeat factor `1`, and hard-window temporal jitter `0.25s`.

First mine from the exact target checkpoint. The dense train-video pass is the
expensive part and can be reused by later HN variants:

```bash
INIT_CHECKPOINT=/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt \
bash scripts/run_football_weekend_a800.sh mine_hardneg_target 0,1
```

Run the same-LR control first or in parallel. Without this control, a metric
change cannot be attributed to hard-negative sampling:

```bash
INIT_CHECKPOINT=/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt \
nohup bash scripts/run_football_weekend_a800.sh hardneg_control 0,1 \
  > /tmp/weekend_hardneg_control_launcher.log 2>&1 &

INIT_CHECKPOINT=/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt \
nohup bash scripts/run_football_weekend_a800.sh hardneg 2,3 \
  > /tmp/weekend_hardneg_launcher.log 2>&1 &
```

The mining checkpoint must match `INIT_CHECKPOINT`, or the candidate windows
must first be rescored by `INIT_CHECKPOINT`. The historical manifest was mined
from a dual-view checkpoint and is not an exact target-model manifest.

Corrected E4 uses 512x896 input with 32 candidate frames, takes the union of shot Top4 and save
Top4, adds eight uniform context frames, preserves original candidate-frame
positions, and excludes set-piece from hard frame competition. Its hard forward
selection uses a straight-through soft gate, so clip classification loss also
updates the frame selector without requiring an exact event-time anchor:

```bash
INIT_CHECKPOINT=/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt \
nohup bash scripts/run_football_weekend_a800.sh event_topk_512 2,3 \
  > /tmp/weekend_event_topk_launcher.log 2>&1 &
```

Do not start hard-negative training until the manifest has been manually
audited. Build it with:

```bash
bash scripts/run_football_precision60_experiments.sh hard-negatives
```

## Batch 2: capacity and temporal-head ablations

Run only after Batch 1 completes:

```bash
INIT_CHECKPOINT=/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt \
nohup bash scripts/run_football_weekend_a800.sh lora8 0,1 \
  > /tmp/weekend_lora8_launcher.log 2>&1 &

INIT_CHECKPOINT=/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt \
nohup bash scripts/run_football_weekend_a800.sh class_query 2,3 \
  > /tmp/weekend_class_query_launcher.log 2>&1 &
```

LoRA8 changes only `target_last_blocks: 4 -> 8`. Class-query keeps LoRA at four
blocks and changes only temporal aggregation. Do not add MLP LoRA or raise rank
in this batch.

## Current temporal-head comparison

The remote E2 attention-pool epoch-3 checkpoint is:

```text
vitl16_lora_r5_f32_16f_e2_attn_pool_epoch3.pt
```

It uses 640x1120 input, 16 frames, LoRA on the last four blocks, global
micro-batch 40, accumulation 2, and effective optimizer batch 80. The local E3
run keeps the same model/data settings, uses per-GPU batch 2 on three GPUs and
accumulation 14 (effective batch 84), and runs for three epochs.

Compare checkpoints with a one-recall-point guard:

```bash
python scripts/compare_football_checkpoint_metrics.py \
  --baseline /mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt \
  --candidates vitl16_lora_r5_f32_16f_e2_attn_pool_epoch3.pt \
  --labels shot,save,set_piece \
  --recall-tolerance-pp 1
```

Run both `shot,save` and `shot,save,set_piece` comparisons. A temporal head is
not accepted when it improves the instantaneous classes but violates the
set-piece recall guard.

## Evaluation

Evaluate every saved epoch, not only sampled-validation `best.pt`:

```bash
bash scripts/run_football_weekend_a800.sh eval 0 \
  outputs/football_events/vitl16_weekend_hardneg_last4/epoch_2.pt
```

Primary metric is six-video PointNMS at tolerance 5 seconds. Window-overlap is
diagnostic only. Use checkpoint thresholds for the first comparison; threshold
calibration must use the 29-video calibration split.

Stop an experiment when, after epoch 2:

- shot and save AP both decline;
- frame TopK hit rate declines for two consecutive epochs;
- sampled validation precision falls while recall does not improve;
- loss is flat and gradients/LR are verified healthy.

Accept a model for combination only when PointNMS FP falls by at least 10% and
no class loses more than one recall point, or when macro precision improves by
at least three points with non-decreasing micro recall.

## Combination rule

Do not combine changes during the weekend ablation. After ranking the four
experiments:

1. choose the best architecture between baseline, class-query, and corrected E4;
2. choose LoRA4 or LoRA8 independently;
3. apply audited hard negatives to the selected architecture;
4. train one final combined model from the strongest compatible checkpoint.

This ordering separates data, temporal aggregation, and DINO adaptation gains.
