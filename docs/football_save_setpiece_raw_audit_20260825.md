# Save / set-piece supervision audit (2026-08-25)

## Decision

Do not switch the training annotation root from `football_events_human_repair`
to `football_events_raw`.  Keep the reviewed timestamp as the spotting anchor.
Raw metadata may be joined as optional interval/context metadata, but only for
review-accepted events and only where `endTime > startTime`.

The pending `savecohort_setpiece_subtype_v2` launch is paused.  Its current
per-microbatch cohort equalization is not valid with per-rank batch size 2.

## Raw versus reviewed set-piece audit

All common raw set-piece entries in the active train/val/external splits are
present in the repair JSON.  The surplus raw entries are exactly the entries
explicitly marked `label_correct=false`, rather than records missing because of
file synchronization.

| split | raw | accepted | rejected | rejected / accepted |
|---|---:|---:|---:|---:|
| train (133 IDs, 132 currently readable) | 1,486 | 1,273 | 213 | 16.7% |
| val15 | 268 | 218 | 50 | 22.9% |
| external18 | 346 | 283 | 63 | 22.3% |

Train subtype counts before/after review:

| subtype | raw | accepted | rejected |
|---|---:|---:|---:|
| corner | 695 | 600 | 95 |
| freekick | 667 | 558 | 109 |
| penalty | 28 | 28 | 0 |
| kickoff | 96 | 87 | 9 |

The accepted repair timestamp matches raw `startTime` (within the JSON time
rounding tolerance).  Directly reading raw annotations would make the current
loader use `(startTime + endTime) / 2`, shifting many train anchors about
1.5 seconds later even though evaluation still uses the reviewed timestamp.

Raw interval completeness is inconsistent:

- In the older train subset, corner/freekick/penalty median duration is about
  3 seconds (90th percentile about 4 seconds).
- Kickoff intervals are zero length.
- In the active val15 and external18 files, set-piece intervals are almost all
  zero length.

Therefore interval metadata is not evidence of a more accurate universal event
center.  It is a partially populated action/context span.

## Safe use of raw annotations

1. Keep reviewed accepted events as strong positives and keep their timestamp
   as the anchor.
2. Join valid raw intervals to accepted events by `(video_id, subtype,
   startTime)` and expose them as optional context metadata.  Do not use their
   midpoint as a new anchor.
3. Treat reviewed-rejected raw events as uncertain until a stratified visual
   audit is complete.  The safe first use is an exclusion/mask zone for
   set-piece negative sampling, not a positive label.
4. Audit the 213 train rejected events by subtype/video.  If the true-positive
   rate is high, repair the GT.  If it is low, these events are valuable
   class-conditional set-piece negatives.  Mixing both cases into weak
   positives would erase that value.
5. Never change val15/external18 GT from raw metadata while using those splits
   for threshold selection or external reporting.

## Corrected save supervision

Literal `save-only` clips are not a reliable semantic cohort: a real save is
causally preceded by a shot and a missing shot annotation can falsely create a
`save-only` label.  The robust cohorts are based on the record's focus event:

- save-anchored positive (shot context may remain visible);
- joint shot-to-save positive;
- trusted shot-only conditional negative (no save in the protected future
  interval and both labels complete).

The current implementation averages only the cohorts present inside each local
microbatch.  With DDP per-rank batch size 2 it cannot represent three cohorts
and produces noisy, rank-dependent weighting.  Replace it with dataset-level
cohort statistics plus fixed per-sample auxiliary weights, or a cohort-aware
sampler.  Gradient accumulation alone does not fix loss normalization that was
already performed per microbatch.

## Set-piece subtype head: expected value and limitation

A corner/freekick/penalty/kickoff auxiliary head is reasonable because it forces
the set-piece evidence token to retain intra-class structure.  It is not by
itself a direct false-positive suppressor: it separates positive subtypes but
does not teach why static players near the box are not a set piece.

Do not increase the 256-dimensional class evidence space yet.  Existing
set-piece frame top-k localization has been strong, so capacity is not the
leading demonstrated failure.  First require:

- subtype train/val AP and support reporting;
- class-balanced subtype loss (penalty has only 28 train examples and no useful
  active-val support);
- aggregate set-piece precision at a fixed recall floor and long-video
  participation time.

## Pre-registered acceptance criteria

Compare against the current last8/r8 best checkpoint using the same val15
threshold protocol and untouched external18 long-video pipeline.

- Save: precision improves at the same recall; recall loss at most 1 pp.
- Set-piece: precision improves at the same recall; no increase in duplicate
  candidates or review time.
- Report the common recall-floor audit in addition to operational per-class
  floors.  Lowering save/set-piece threshold floors to 80% is an operating
  policy, not evidence that the model improved.
- A gain smaller than the video-level bootstrap confidence interval is noise.
