# Football candidate verifier: cal15 -> external18 protocol

## Objective

Use the existing DINO event model as the high-recall generator, then learn a
small per-class verifier from cached clip/frame response statistics.  The
pre-registered product gate is:

- human-visible event recall >= 90%;
- deduplicated full-segment participation < 30%;
- report strict 1:1 PointNMS and window-overlap precision/recall, per class and micro.

The 18-video generator audit shows why both constraints matter.  The current
full-segment chain reaches 96.67% human-visible recall but requires 63.64%
participation.  Capping each segment to 10 seconds reduces participation to
31.11%, but recall falls to 82.40%.  Therefore shortening clips alone does not
solve the problem; the verifier must remove false segments while retaining
event-containing segments.

## Data isolation

The implementation is `scripts/run_football_verifier_protocol_v1.py`.

1. The fixed 15-video internal validation split is used for GroupKFold OOF
   verifier fitting and threshold selection.
2. Per-class thresholds maximize precision subject to the requested recall
   floor.  If PointNMS candidate recall cannot reach the floor, the ceiling and
   shortfall are written explicitly instead of silently relaxing the KPI.
3. A final verifier is fitted on all 15 calibration videos.
4. Only after model and thresholds are frozen are the 18 external videos read.
5. Calibration/external video overlap is a hard error.  Split SHA256 values and
   cache completeness are persisted in `protocol_audit.json`.
6. No external prediction or GT is used for fitting, model selection, or
   threshold selection.

## Verifier V1

V1 is intentionally small and CPU-trainable: one balanced logistic model per
event class.  It consumes seven cached candidate statistics:

- clip logit;
- maximum and top-4 mean frame logits;
- response sharpness;
- frame-peak offset within the 10-second clip;
- neighboring-window clip support and support count.

This tests the strongest low-risk hypothesis first: information already exists
in the DINO clip/frame outputs but is not exposed by one raw threshold.  It does
not modify the backbone or occupy training GPUs.

The workload report merges overlapping windows across all labels before
counting review time.  A time interval is watched once even if both shot and
save fire, so the participation metric reflects the UI review workflow rather
than the sum of per-class durations.  The exact `summary.json:duration_sec` is
used as the denominator.

## Existing evidence and limits

- A historical rich-feature six-video OOF diagnostic improved row-level micro
  precision from 0.2166 to 0.4378 at nearly unchanged recall
  (0.6795 -> 0.6772).  This is useful evidence that model outputs contain
  ranking information, but it is neither a clean external test nor evidence at
  90% recall.
- The historical cal29 -> holdout6 legacy reranker was negative: holdout micro
  precision changed from 0.2618 to 0.2439 while recall changed from 0.6977 to
  0.6930.  Consequently verifier value must not be assumed.
- A CPU regression of the new protocol on the old cal29/holdout6 cache passed
  end-to-end.  This is only a code regression and is not a current-model result.

The current cache audit finds all 18 external videos complete, but the same
checkpoint's 15-video dense cache has not yet been generated.  It is therefore
not scientifically valid to fit on external18, even though doing so would
produce an immediate optimistic number.

## Run after the 15-video cache exists

First run the cached validation pipeline for the exact generator checkpoint:

```bash
python scripts/run_validation15_long_video_pipeline.py \
  --checkpoint /absolute/path/to/generator.pt \
  --gpu-groups '4;5;6;7' \
  --batch-size 8 \
  --num-workers 2 \
  --recall-floor 0.90 \
  --run-name generator_val15
```

Then pass its aggregate dense directory together with the already complete
external18 directory:

```bash
python scripts/run_football_verifier_protocol_v1.py \
  --calibration-run-dir /absolute/path/to/generator_val15_dense_run \
  --calibration-video-id-file configs/football/splits/thirdparty18_test_long15_val_no_pn_train/internal_val_video_ids.txt \
  --external-run-dir outputs/football_long_video_full_pipeline/v1_1_new_ep8_best_external18_quick_window_overlap_tol5_20260824_095755/dense_runs/v1_1_new_ep8_best_external18_quick_window_overlap_tol5_all18_fusion \
  --external-video-id-file configs/football/splits/thirdparty18_test_long15_val_no_pn_train/thirdparty18_test_video_ids.txt \
  --output-dir outputs/football_candidate_verifiers/v1_1_new_ep8_cal15_external18_v1 \
  --recall-floor 0.90 \
  --folds 5 \
  --threshold-grid-size 101
```

Outputs include `calibration_oof_report.json`, `frozen_verifier.json`,
`external18_report.json`, and candidate-level OOF/external CSV predictions.

## Decision rule

- Accept V1 only if external18 full-segment human-visible recall is at least
  90% and participation is below 30%, with no class showing a hidden recall
  collapse.
- If recall is >=90% but participation is 30--40%, move to a segment-level V2
  using richer multi-head curve statistics and event grammar.
- If OOF improves but external18 does not, stop coefficient tuning; the issue is
  cross-video calibration/domain shift.
- If the 15-video OOF ranking gap does not improve, cached DINO outputs are not
  sufficient for a lightweight verifier and the next experiment must add new
  visual evidence, not more threshold tuning.
