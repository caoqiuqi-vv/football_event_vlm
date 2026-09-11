# Football Precision Recovery Plan

> Updated: 2026-08-05. Goal: improve long-video precision while preserving high recall for
> `shot`, `save`, and `set_piece`.

## 1. Current Diagnosis

The model is probably not fully at a feature plateau. The stronger diagnosis is:

1. Clip-level validation and long-video event spotting are no longer aligned enough.
   A model can improve sampled val F1/AP but still create more dense long-video false positives.
2. `shot/save` have useful frame-event signals, but simple score multiplication or hard top-k has
   not converted that signal into stable PointNMS precision.
3. ROI contains useful detail, but current dual-path forms often force ROI to behave like a full
   classifier. That makes training unstable when ROI is mislocalized or incomplete.
4. Many failed runs changed several variables at once: ROI strategy, fusion head, LoRA/backbone
   trainability, negative ratio, focal loss, and hard negatives. This makes negative results hard
   to interpret.

So the next phase should stop treating every new architecture as a full replacement. The safer path:

```text
Keep a high-recall E1-style full-image model as the anchor.
Use ROI/frame/hard-FP evidence to improve ranking and veto false positives.
Only accept an experiment if it improves long-video precision under a recall guard.
```

## 2. Evidence So Far

### Stronger Signals

- High-resolution E1/frame-det is still the best stable base among recent single-path experiments.
- Frame-event top-k hit rates are high enough to be useful for `shot/save`.
- Candidate/verifier feature ablations showed that frame and ROI statistics can separate some hard FP.
- Some ROI experiments improved precision, but often by suppressing scores and losing recall.

### Weak Or Negative Signals

- `negative_ratio=8 + focal` epoch1 did not improve 6-video long-video precision and hurt recall.
- Direct ROI dual-path training often plateaued or trained slowly, likely due to feature/gradient
  conflict between full and local crops.
- Offline hard-negative training did not prove ineffective hard FP learning; the previous version was
  confounded by frozen backbone, medium-score mining, frame-rank loss, and online hard negative.
- Pure time-head changes are lower priority until candidate quality/ranking is improved.

## 3. Decision Gates

Every precision-oriented training run should be judged by both protocols:

```text
PointNMS: closer to online event output.
Window-overlap: closer to raw clip classifier quality.
```

Minimum pass condition:

```text
recall >= baseline - 1pp
shot/save precision > baseline
mean precision > baseline
```

If a run only raises sampled-val AP/F1 but loses long-video precision or recall, it is not a useful
mainline improvement.

## 4. Immediate Next Runs

### A. Finish `neg8 + focal`

Status: running locally.

Gate:

- If epoch2/epoch3 still fail long-video recall guard, stop this line.
- Do not add more epochs just because training loss decreases.

Automation:

```bash
bash scripts/run_neg8_then_clip_rank_gate.sh 3,4,6,7
```

The gate evaluates the latest epoch and switches to clip-ranking if `neg8 + focal` fails.

### B. Clean PointNMS Hard-FP Clip Ranking

Purpose: teach the model to rank true event clips above final long-video false positives.

Key controls:

- single full-image E1 architecture
- LoRA/backbone adaptation enabled
- hard negatives mined from final PointNMS FP
- no extra hard-negative BCE weighting; keep `loss_weight=1.0`
- clip-level pairwise rank weight `0.12`, margin `0.5`
- training safety margin `10s` to protect repaired positives
- no frame-rank loss
- no online hard-negative loss
- first version targets `shot/save`

Command:

```bash
RUN_MINING=1 \
INIT_CHECKPOINT=/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt \
OUTPUT_DIR=outputs/football_events/vitl16_e1_clip_rank_clean_lora \
PER_GPU_BATCH_SIZE=4 \
TARGET_EFFECTIVE_BATCH_SIZE=80 \
TARGET_HEAD_LR=0.0003 \
TARGET_BACKBONE_LR=0.00003 \
bash scripts/run_football_clip_rank_clean.sh all 2,3
```

Expected useful outcome:

- `shot/save` precision improves at the same recall floor.
- confidence gap improves, especially positive p10 vs negative p90.
- PointNMS FP count drops without a matching increase in FN.

### C. ROI As Evidence, Not Independent Classifier

Next ROI run should not require ROI to independently classify every event. Preferred formulation:

```text
full-image tokens: high-recall primary evidence
ROI tokens: local detail and anti-FP evidence
fusion: final classifier over combined tokens or bounded residual over full logits
loss: clip BCE + clip-level hard-FP ranking/suppression
```

Avoid in the first controlled run:

- separate `local_loss` as a full event classifier
- large unconstrained ROI residuals
- changing ROI crop strategy and fusion architecture in the same run
- changing full/ROI frame sampling at the same time

## 5. Why This Is Not Yet A Dead Plateau

A true plateau would mean:

```text
positive/negative confidence distributions cannot be separated,
frame-event/ROI diagnostics provide no extra signal,
larger or more trainable models do not improve sampled and long-video metrics,
hard FP visual categories are indistinguishable even after targeted supervision.
```

We do not have that evidence yet. What we have is a training-objective mismatch:

- dense inference produces many hard FP that random negatives do not represent well;
- ROI is useful but too noisy to act as a standalone classifier;
- frame-event evidence is useful but needs a learned fusion/ranking mechanism.

Therefore the next high-value work is not to keep increasing model size or adding losses. It is to
make the training distribution and evaluation distribution match the online failure mode.

## 6. Stop Conditions

Stop an experiment line if:

- two consecutive epochs fail the long-video recall guard;
- precision gain comes only from recall loss;
- `shot/save` do not improve but `set_piece` alone raises macro metrics;
- training loss decreases while confidence gap and dense FP count do not improve.

Promote an experiment if:

- PointNMS and window-overlap both improve precision under recall guard;
- improvement appears on at least `shot` or `save`, not only macro;
- FP distance analysis shows fewer far-from-GT FP;
- frame/ROI diagnostics explain the score change.
