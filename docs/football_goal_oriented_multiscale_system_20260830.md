# Goal-oriented football event system (2026-08-30)

## Primary objective

For the normal/eligible camera cohort, optimize precision and human review time
subject to strict one-to-one shot recall >= 90% at +/-3 seconds.  A small,
predeclared hard/OOD cohort may use a fallback workflow and must be reported
separately.  Predictions are never temporally NMSed or merged.

Calibration selection must target at least 93% shot recall and require the
video-bootstrap 5th percentile to remain >=90%; selecting a threshold whose
point estimate barely reaches 90% is not acceptable.

## Evidence that constrains the design

- A1 candidate-set residual reranking improved precision from 8.74% to 10.73%
  at 90.18% calibration recall, but review time remained 40.61%.  This is useful
  reranking, not a deployable solution.
- At 95.19% shot-region recall, current Stage-1 max-score retrieval covers
  61.97% of video time with 20-second blocks.  The oracle lower bound is 14.22%.
- One-minute output blocks are intrinsically too broad: their oracle coverage
  is already 35.96%, versus 14.22% for 20-second blocks.
- A three-minute score-context OOF probe reduced 20-second coverage only from
  61.97% to 60.83% after identity blending.  Extending already-collapsed logits
  is not the missing capability.
- Frozen/high-capacity patch-token readout experiments did not generalize on
  shot TP-vs-FP (roughly 0.47-0.56 grouped AUC), and football SSL did not add a
  material shot TP-vs-FP gain.  Patch attention or domain SSL alone is not a
  sufficient plan.
- The event-centred clip classifier has useful sampled-data signal (shot AP
  around 0.73-0.76), but that signal has not yet been evaluated as the complete
  temporal model on the exact dense candidate distribution.

## System: multi-rate routed event retrieval

This is one hierarchical system, not two unrelated DINO classifiers.

```text
full video
  -> multi-rate appearance + motion feature bank
  -> long-context event-region retriever (long input, 5-20s outputs)
  -> high-rate candidate action examiner
  -> learned candidate-set resolver (no NMS)
  -> auto-accept / human-review / reject policy
```

### 1. Multi-rate feature bank

- Global state stream: +/-90 seconds at 1-2 FPS.
- Action stream: candidate +/-6 seconds at 8-12 FPS.
- All full-image frames are at least 512x896; no detector is a hard dependency.
- Appearance evidence uses DINOv3 ViT-L/16.
- Motion evidence is independent: dense RGB differences and a
  camera-compensated flow/residual-motion encoder.  This is required because a
  frame DINO plus temporal MLP can represent similar kicks but often lacks
  contact and post-kick trajectory information.
- Patch evidence is compressed with learned spatial resampling, but patch
  attention is not trusted as an independent classifier.
- Audio log-mel/whistle evidence is optional for shot and important for restart
  recovery; missing audio uses a mask and modality dropout.

### 2. Long-context event-region retriever

- Consume 180-second chunks with a 60-second output core.
- Use a factorized temporal U-Net/TCN plus bidirectional sequence encoder over
  global-state tokens; inject motion summaries at higher temporal resolution.
- Predict multiple anchored event slots or two anchors per 2-second cell, with
  sub-cell time offsets.  A 60-second core has an explicit capacity above the
  p99 event density.
- Hungarian supervision makes one GT map to one slot while allowing adjacent
  or simultaneous events in different slots.
- The retriever exports 5-20 second regions, never whole-minute review windows.
- Candidate union includes a conservative existing Stage-1 branch during the
  transition so the new retriever cannot silently lower recall.

Retriever gate on calibration/OOF:

- eligible-cohort shot candidate recall >=97%;
- macro recall >=94%;
- 20-second region coverage <=35% for the first acceptable version, target
  <=25%;
- no more than the predeclared hard/OOD videos below 80% recall.

### 3. High-rate candidate action examiner

The examiner must add information rather than reuse a frozen global embedding.

- Encode 8-12 FPS frames in the local action interval and sparse frames in the
  longer context.
- Fuse DINO appearance tokens with residual-motion tokens using a
  candidate-anchored query.
- Train structured predicates in addition to the final labels:
  kick/contact, rapid ball departure, attacking-direction continuation,
  goalkeeper reaction, restart, camera cut/pan, and confounder type.
- Confounders are explicit labels: cross, long pass, clearance, ordinary pass,
  restart preparation, camera motion, and uncertain/missing annotation.
- `save -> shot` is soft positive semantics; set-piece-like evidence does not
  make shot a negative.
- Fine-tune DINO last blocks/LoRA only after the motion/fusion heads have a
  stable identity-preserving solution.

Examiner gate:

- retain >=95% of retriever-covered shot GT;
- reject >=50% of unmatched candidates in video-grouped OOF;
- demonstrate gain over the complete event-centred clip teacher, not only over
  a frozen-feature baseline.

### 4. Learned set resolver, not temporal NMS

- Group independently scored candidates in a 30-second core plus context.
- Candidate-anchored queries output background/event, time correction, and
  uncertainty.
- Use one-to-one Hungarian loss.  Two nearby shots and simultaneous shot+save
  remain independently representable.
- Keep an identity residual from examiner scores and bounded time correction.

### 5. Selective human workflow

Use two calibrated boundaries rather than one threshold:

- auto-accept: very high precision, no human review;
- review band: required to preserve recall, review the union of display
  intervals only;
- reject: calibrated low-risk background.

Interval union is a UI workload calculation, not prediction NMS.  Shot and save
may share one review interval.

Hard/OOD videos are routed using observable quality features fixed on
calibration data (blur, compression, camera motion, field visibility, feature
distance, and score-distribution instability).  Test GT may not be used to
decide which videos are hard.

## Data protocol

- Generate deployment-like candidates OOF over the 128 training videos.
- Reserve the 18 calibration videos for score/threshold/policy selection and
  the 18 user-specified test videos for one final evaluation.
- Do not train a fold model and apply its raw OOF threshold to a differently
  trained final model.  Use either a fixed formula or a fold ensemble with a
  common calibrated score definition.
- Review a contrast set of approximately 2k-5k dense candidates.  Every item is
  labelled shot, cross, pass, clearance, restart, camera artefact, background,
  or uncertain.  Uncertain/unreviewed high-scoring candidates are not ordinary
  negatives; use masked/PU loss.

## End-to-end acceptance

- Eligible cohort, +/-3 seconds, one-to-one: shot recall >=90%.
- Calibration operating point: point recall >=93% and video-bootstrap lower
  5th percentile >=90%.
- Precision target >=30%, stretch >=40%, while always reporting the full PR
  curve.
- Human review union <=20% of eligible video duration; all-video fallback
  workload reported separately.
- Report macro recall, worst quartile, hard/OOD cohort, candidate ceiling,
  FP/minute, and auto-accept/review/reject counts.

## Experiment order and stop gates

1. Complete-clip-teacher temporal-grid diagnostic on final Stage-1 candidates.
   Stop using a second DINO classifier if it adds no conditional ranking value.
2. OOF contrast-set and annotation-noise audit.  Do not train the examiner
   against unchecked dense false positives.
3. Appearance-only versus appearance+motion examiner.  Promote motion only if
   it improves precision at fixed positive retention on held-out videos.
4. Add raw visual long-state tokens to the retriever.  Reject the line if
   20-second coverage remains above 45% at 97% candidate recall.
5. Add the learned set resolver and selective-review policy.
6. Tune on calibration18 once; evaluate test18 once with frozen choices.
