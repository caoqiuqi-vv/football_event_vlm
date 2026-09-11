# Detection-aware football experiments

> 此文件为早期实验草案。当前高优实验矩阵以
> `docs/football_detection_aware_priority_experiments.md` 为准。

## Objective and fixed protocol

- Primary target: dense long-video micro precision greater than 0.30.
- Guardrail: do not obtain precision by suppressing almost every prediction; sampled validation checkpoint selection requires recall >= 0.60 (experiment 3 uses 0.55).
- Classification threshold is fixed at 0.5.
- Dense protocol is fixed at 10 s clips, 5 s stride, point NMS radius 5 s, and match tolerance 5 s.
- Seed, train/validation video IDs, validation negative ratio, `pos_weight=[1.0, 2.5, 4.0]`, and label-mask behavior are fixed across experiments.
- All ROI coordinates are computed once per 10 s clip. No frame-by-frame crop is used.
- Fusion uses global + alpha * (local - global). The minus sign in the draft plan would move the prediction away from local evidence and is therefore treated as a formula error.
- The shared precomputed artifact is a v2 evidence index, not a finite list of window ROIs: temporal jitter creates arbitrary windows, so train/validation/dense call the same deterministic generator over the same index.

## Priority order

### Required pre-check: frozen global resolution

Run the resolution stage once before experiment 2. It evaluates the unchanged
lora_r2_best.pt global model on a balanced paired 1024-sample subset of the same 11-video validation set at
384x640 and 192x320. The low-resolution global branch is selected only when mAP
drops by at most 0.01 and default-threshold recall drops by at most 0.01;
otherwise the launcher automatically keeps 384x640. This is an evaluation gate,
not an additional training experiment.

### Experiment 1: fixed robust ROI, single view

Config: `configs/football/dinov3_vitl16_robust_roi_exp1.yaml`

This is the cheapest causal test of whether removing background tokens improves classification. It initializes from `/mnt/data_16t/football/qiuqi/checkpoints/lora_r2_best.pt`, freezes the complete DINO/LoRA backbone, and updates only the temporal and classification modules for five epochs. Training keeps the baseline `negative_ratio=2`.

Run `index`, `exp1`, then `eval1`. Continue to experiment 2 even if experiment 1 succeeds: ROI-only has no global fail-safe when a plausible but incorrect detector crop passes the confidence filter.

### Experiment 2: global + fixed ROI with class-wise gate

Config: `configs/football/dinov3_vitl16_robust_dual_exp2.yaml`

This is the preferred production design. It initializes both branches from the same baseline, freezes the backbone and global branch, and trains only the local temporal/classification branch plus a class-wise reliability gate. An invalid ROI forces the gate contribution to zero, so the output exactly falls back to the global baseline. Training injects missing/noisy ROI cases, while validation does not.

Run `exp2`, then `eval2`. This costs about twice the DINO forward compute of experiment 1, but uses batch size 1 and gradient accumulation to fit one GPU.

### Experiment 3: conditional precision fine-tune

Config: `configs/football/dinov3_vitl16_robust_dual_precision_exp3.yaml`

Run this only when experiment 2 has dense precision below 0.30 but useful recall (roughly >= 0.55), or when precision is within about five percentage points of the target. It starts from experiment 2 `best.pt`, raises only the training negative ratio from 2 to 4, keeps validation sampling unchanged at 2, lowers the learning rate, and trains for two epochs.

Do not run experiment 3 if experiment 2 precision is already above 0.30. If experiment 2 recall is very low, adding negatives is the wrong direction; inspect ROI coverage and per-class errors first.

## Commands

All stages map the selected physical GPU to logical `cuda:0`:

```bash
mkdir -p logs
bash scripts/launch_football_detection_aware.sh index
bash scripts/launch_football_detection_aware.sh resolution 0

nohup bash scripts/launch_football_detection_aware.sh exp1 0 > logs/football_roi_exp1.log 2>&1 &
bash scripts/launch_football_detection_aware.sh eval1 0

nohup bash scripts/launch_football_detection_aware.sh exp2 0 > logs/football_dual_exp2.log 2>&1 &
bash scripts/launch_football_detection_aware.sh eval2 0

# Conditional only:
nohup bash scripts/launch_football_detection_aware.sh exp3 0 > logs/football_dual_exp3.log 2>&1 &
bash scripts/launch_football_detection_aware.sh eval3 0
```

The dense aggregate result is written to each run's `summary_metrics.json`; per-video, per-class precision/recall is also written to `summary_metrics.csv`. Dual-view `window_predictions.csv` additionally records global probability, ROI probability, and effective ROI gate for each class.

## Decision rules

1. Compare dense results only at threshold 0.5 and the fixed protocol above.
2. A run passes when micro precision > 0.30 and recall remains operationally useful. Also inspect each class: a good aggregate must not hide a class with zero recall.
3. If experiment 1 improves but experiment 2 does not, inspect the learned gate and local/global probabilities before changing ROI thresholds.
4. If both experiments have many `roi_valid=0` windows, the bottleneck is detector/index coverage. If `roi_valid` is high but ROI-only precision falls, visually audit high-confidence false-positive ROIs; do not add more training epochs first.
5. Only after these two experiments should hard-negative mining from dense false positives be considered.
6. After experiment 2 passes, run the detector/index pipeline for the missing 80 training and 24 validation videos, then repeat the selected dual-view configuration on the original 127/35 split. Do not mix indexed and unindexed samples in the first round.
