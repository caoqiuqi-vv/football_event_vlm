# Object Motion Evidence Adapter

This directory isolates the detector-distilled high-frame-rate branch from the
legacy football training implementation.

The inference graph has no YOLO dependency.  YOLO Teachers are used only while
training to supervise ball, goal and person heatmaps, visibility and ball/goal
coordinates.  The student converts DINO patch evidence into explicit motion
and interaction features and returns zero-initialized bounded residuals for
both frame localization and clip classification.

The full-image `fromlast_e8` branch remains the anchor and can be reproduced by
zeroing the object-motion residuals.

The current v3 ball-LoRA branch observes the complete ten-second window as three overlapping four-second segments: `[0,4]`, `[3,7]`, and `[6,10]`, 11 frames each (33 absolute-time-sorted frames; shared endpoints are intentionally retained). Coverage and detector confidence are diagnostics/features, not multiplicative event gates. `monitor.py` writes health state to `object_motion_learning_status.json`.

## Causal v4 (recommended)

Use `scripts/run_object_motion_adapter_v4.sh`. It fixes the experiment contract rather than adding another event head:

- exactly 11 frames per segment (33 total), matching the smoke contract;
- DINO and the YOLO teacher share the same 720x1280 decoded frames, so the small ball is not discarded before patch embedding and no duplicate high/low-resolution batch tensor is retained;
- teacher confidence weights supervision reliability but no longer lowers the positive target amplitude; detector misses are unknown for localization and only weak presence negatives;
- the BallLoRA loss balances rare positive patches against background and adds direct center supervision;
- spatial centers use the same sparse distributions used for routing instead of the background-heavy full sigmoid map;
- learned residual gates are multiplied by class-aware ball-evidence gates; shot/save use a 0.05 fallback while set-piece uses 0.35 for legitimate occlusion/off-screen cases;
- the frozen anchor's observed-window coverage now gates the clip residual;
- legacy ball-head weights initialize all four multi-layer ball readouts rather than an unused generic output row.

Compare v4 against the frozen anchor and a residual-disabled v4 run. Do not compare it directly with the old `fullwindow51` recipe: that launcher used 17 frames per segment while its documentation and smoke test claimed 11.

At patch size 16, 720x1280 produces 3600 tokens per frame versus 1792 at 512x896. Per-frame self-attention is therefore about 4x heavier. The launcher keeps batch size 1 and motion backbone chunk size 1, and the GPU smoke must pass the 28 GiB ceiling before a formal run. If it fails only on memory, set both `MOTION_IMAGE_SIZE` and `MOTION_TEACHER_IMAGE_SIZE` to `[640,1120]`; do not silently fall back to 512p in the same experiment.

## Anti-collapse v2

The event path is deliberately separated from detector distillation:

- event gradients stop at the ball/goal/person evidence boundary;
- effective event gates require predicted ball evidence and have a hard cap;
- frame and clip residuals use separate small maximum deltas;
- curriculum can keep `object_motion_event_residual_scale=0` while the object
  heads learn, then open the event residual gradually;
- normalized spatial-distribution loss emphasizes the small ball target;
- ball and goal losses are weighted above the ubiquitous person target;
- pairwise residual ranking, residual-energy, gate-budget and saturation losses
  prevent the easiest all-positive/all-gates-open solution.

Use `scripts/run_object_motion_adapter_v2_antcollapse.sh` for the guarded
training recipe. The previous launcher is retained only to reproduce the
collapsed baseline.

## Anti-collapse v3 ball-LoRA

- The checkpoint anchor, base DINO, legacy event LoRA, temporal modules and event heads are frozen. A separate rank-4/alpha-4 BallLoRA overlay is trainable only in the last eight DINO qkv/proj modules and only during motion-frame encoding.
- Anchor forward disables BallLoRA under `no_grad`; the motion path enables it with a restoring context manager. A strict trainable-parameter allowlist fails closed.
- Ball readout fuses DINO layers `[11,17,20,23]` with initial convex weights `[0.15,0.25,0.25,0.35]`; no layer may exceed 0.7. Student routing uses topK=4, spatial-softmax temperature 0.25, center, presence, entropy and velocity. Teacher outputs never select relation tokens.
- Weak/missing teacher evidence trains detached readout only. BallLoRA receives strong ball localization, patch contrastive, strong-track and three-frame non-ball feature-preserve losses. Goal/person and event/relation paths cannot update it.
- Production uses 33 motion frames, per-GPU batch 1, accumulation 16, DINO chunk 1, bf16 and teacher batch 8. Adjacent reviewed pairs use a one-sample detached queue; pair order flips each epoch and crossing pair IDs fails.
- BallLoRA uses LR `1e-6`, weight decay 0 and grad clip 0.1. Adapter heads/relation use LR `2e-5`, weight decay 0.05 and grad clip 1. EMA is disabled for the first experiment because the mature trainer cannot yet restrict EMA to BallLoRA plus ball heads.
- Epochs 1-2 keep event injection alpha at zero; epochs 3-4 use 0.10. Promotion to 0.25 is deliberately not automatic and requires a later retention/ball-quality gate.
- The run is **teacher-supervised exploratory** because no independent human ball-validation set exists. It must not be reported as meeting the final model target.
- The watchdog waits for one continuously safe, PID-free GPU and runs its own contract-bound two-step smoke. It requires peak CUDA memory below 28 GiB, then waits for at least four safe GPUs for five 60-second confirmations before choosing three for training.

Reviewed negative sources: `reviewed_mid_score_v2_shot_save_setpiece.json` and `other_action_reviewed_train_shot_save_score_filtered.json`. E1.6 omissions are repaired only by synthesizing windows from these explicit reviews; unreviewed windows are never strong negatives.
