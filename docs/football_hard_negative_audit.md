# Hard-negative mining audit

## Conclusion

The historical hard-negative run does not prove that hard-example mining is
ineffective. It is confounded by a learning-rate restart, cross-model mining,
and weak hard-example exposure. No label inversion or annotated-GT leakage was
found.

## Historical run result

Compared with `/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt` after
epoch 2:

| Label | Precision delta | Recall delta | AP delta |
|---|---:|---:|---:|
| shot | +9.40 pp | -11.77 pp | -0.57 pp |
| save | +1.19 pp | -3.61 pp | +0.15 pp |

Mean shot/save precision increased by 5.30 pp, but recall fell by 7.69 pp and
validation mAP fell by 1.46 pp. The F1-tuned thresholds moved to 0.5. This is
mainly score suppression and an operating-point trade, not improved ranking.

## Verified issues

1. **Learning-rate restart**: the E1 checkpoint saved head/LoRA LRs of
   `7.34e-6 / 7.34e-7`. The historical HN launcher restarted at
   `3e-4 / 3e-5`, a 40.9x jump. The stage-2 YAML's lower LR was overwritten.
2. **Cross-model mining**: the 590 windows were mined from a 512x896 dual-view
   model's global branch, then used to train a 640x1120 single-view E1 model.
   They were never rescored by the target model.
3. **Weak exposure**: 590 clips were 1.81% of training clips; their 956 trusted
   label slots were only 0.98% of all train label slots. They used normal
   shuffle and no sample/loss weight.
4. **Limited diversity**: 366/590 (62.0%) windows were simultaneously selected
   for shot and save.
5. **Hard-window drift**: the exact false-positive windows inherited the normal
   negative temporal jitter of +/-1.5s.
6. **Missing control**: there was no same-init, same-LR continuation without
   HN, so metric changes could not be causally assigned to HN.

## Logic checks that passed

- All 590 manifest entries map back to a source prediction row.
- Nearest annotated event distance is at least 10.115s; no known GT was turned
  into a negative.
- Label-specific masks are correct: shot/save HNs do not force set_piece to 0.
- 93.2% of shot and 97.3% of save candidates also exceeded 0.55 in the source
  dual model's fused output, so they were not global-branch-only artifacts.
- Manifest entries are deduplicated by time and exact shot/save clips are merged.

## Corrected experiment

The corrected path uses:

- target checkpoint dense mining on the 127 train videos;
- single-view fused scores (`shot>=0.25`, `save>=0.30`), top 6 per video/class;
- head/LoRA LR `5e-5 / 5e-6`;
- hard-negative `loss_weight=4`, `repeat_factor=1`;
- hard-window jitter `+/-0.25s`;
- independent logs for hard-negative sample fraction and clip loss;
- a same-LR no-HN control.

```bash
INIT_CHECKPOINT=/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt \
bash scripts/run_football_weekend_a800.sh mine_hardneg_target 0,1

INIT_CHECKPOINT=/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt \
bash scripts/run_football_weekend_a800.sh hardneg_control 0,1

INIT_CHECKPOINT=/mnt/data_16t/football/qiuqi/checkpoints/vitl16_lora_r5_f32_16f_e1_frame_det_last.pt \
bash scripts/run_football_weekend_a800.sh hardneg 2,3
```

Accept HN only if it improves PointNMS precision at matched recall (recall drop
at most 1 pp), not merely if its independently F1-tuned threshold raises
precision.
