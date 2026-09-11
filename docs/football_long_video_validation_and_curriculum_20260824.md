# 18-video evaluation, Validation-15 pipeline, and V1.3 curriculum handoff

Date: 2026-08-24

## 1. External-18 diagnostic result

Checkpoint:

`outputs/football_events/vitl16_stage1_peak_spotting_v1_1_new_small384_raw720_no_pn_24f/best.pt`

Video set:

`configs/football/splits/thirdparty18_test_long15_val_no_pn_train/thirdparty18_test_video_ids.txt`

The 18 videos contain 1,029.456 minutes and 1,051 labeled events. This run is a
self-calibrated diagnostic because thresholds were searched on these same 18
videos. It must not be cited as untouched external-test performance.

The best score branch in the recall>=0.90 grid is `frame_max_prob`:

- thresholds: shot=0.5, save=0.59375, set_piece=0.1923828125
- strict 1:1 window metric: precision=0.10849, recall=0.90105
- deduplicated review-segment label metric: precision=0.21760, recall=0.88011
- review segments: 1,976

Per-class strict 1:1 window metrics:

| class | precision | recall |
|---|---:|---:|
| shot | 0.1400 | 0.9392 |
| save | 0.0788 | 0.8565 |
| set_piece | 0.0898 | 0.8587 |

Per-class deduplicated review-segment metrics:

| class | precision | recall |
|---|---:|---:|
| shot | 0.3001 | 0.8837 |
| save | 0.1440 | 0.8900 |
| set_piece | 0.1866 | 0.8657 |

### Human workload definitions

`full_segment` merges overlapping positive windows across classes, chunks long
runs into at most 30-second UI segments, and views every second once. It does not
double-count overlapping shot/save/set-piece windows.

- full review time: 655.155 minutes
- participation: 63.641%
- human-visible recall (reviewer may relabel a visible event): 0.96670
- per-class human-visible recall: shot=0.98032, save=0.98565,
  set_piece=0.92580

`capped10_global_peak` views only a 10-second clip around the strongest peak of
each merged segment.

- capped review time: 320.250 minutes
- participation: 31.109%
- human-visible recall: 0.82398
- per-class human-visible recall: shot=0.88014, save=0.84689,
  set_piece=0.69611

Therefore 320.25 minutes is not a recall-preserving workload number. The
recall-preserving operational estimate from this run is 655.15 minutes. The
previous ~74-minute quick number is not used because it did not reproduce the
actual per-video merged UI coverage.

Machine-readable report:

`outputs/football_long_video_full_pipeline/v1_1_new_ep8_best_external18_quick_window_overlap_tol5_20260824_095755/external18_ui_honest_precision_at_recall90.json`

## 2. Validation-15 cached full pipeline

Entry point:

`scripts/run_validation15_long_video_pipeline.py`

Threshold selector:

`scripts/select_validation_thresholds_cached.py`

Protocol:

1. Dense inference on the fixed 15 validation videos.
2. Cache `window_predictions.csv`, `frame_event_logits.csv`, GT, and per-video
   summaries under a checkpoint fingerprint.
3. Search each score branch independently.
4. Select the threshold tuple that maximizes strict 1:1 precision subject to
   micro recall >= 0.80.
5. Report strict window metrics, deduplicated segment-label metrics, full
   segment workload, capped-10 workload, and per-class human-visible recall.

Example (run only when GPUs 4-7 are not occupied by training):

```bash
python scripts/run_validation15_long_video_pipeline.py \
  --checkpoint /absolute/path/to/checkpoint.pt \
  --gpu-groups '4;5;6;7' \
  --batch-size 8 \
  --num-workers 2 \
  --recall-floor 0.80
```

Dense outputs are reused when checkpoint size and mtime are unchanged. Use
`--force` only when intentionally invalidating the cache.

## 3. V1.3 Stage-1 audit and Stage-2 handoff

Initialization was V1.1-new epoch-8 best:

- mAP=0.57723
- frame top-k: shot=0.90114, save=0.88095, set_piece=0.96047

Stage-1 (epochs 1-5, clip-semantic warmup):

- mAP: 0.56322, 0.57568, 0.58630, 0.56538, 0.56108
- tuned micro recall: 0.89655, 0.88630, 0.90401, 0.90028, 0.90774
- epoch-5 frame top-k: shot=0.84030, save=0.81973, set_piece=0.92490

Interpretation: Stage-1 did optimize clip semantics enough to preserve a
high-recall operating point and briefly improved mAP at epoch 3, but the gain
was not sustained and temporal localization degraded. Stage-1 is therefore a
valid completed curriculum phase, not a proven improvement. Stage-2 must be
judged by whether it recovers frame top-k/localization without losing the clip
recall floor.

The true handoff is epoch-5 `last.pt`, not V1.1-new directly. The four-GPU
resume uses batch 6 x 4 GPUs x accumulation 3 = effective batch 72, exactly
matching Stage-1, and restores model, optimizer, scheduler, scaler, and EMA.

Resume config:

`configs/football/dinov3_vitl16_stage1_peak_spotting_v1_3_curriculum_small384_raw720_no_pn_24f_stage2_resume_gpu4567.yaml`

Training log:

`outputs/football_events/vitl16_stage1_peak_spotting_v1_3_curriculum_small384_raw720_no_pn_24f/train_console.log`
