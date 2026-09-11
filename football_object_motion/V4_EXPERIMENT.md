# Object Motion v4 experiment contract

## Required ablation matrix

| Run | Ball supervision | Event residual | Purpose |
|---|---:|---:|---|
| A | off | 0 | frozen anchor reference |
| B | on | 0 | measure ball/goal learning without changing events |
| C | on | 0.10 | test causal auxiliary benefit |
| D | shuffled teacher targets | 0.10 | detect shortcut/non-causal gains |

Promote C only if it improves the pre-declared online metric over A and D does not. Keep thresholds fitted independently per run on validation only.

## Mandatory diagnostics

- ball recall at fixed false positives per frame, split by apparent ball size;
- ball center error on strong-teacher frames and temporal jitter on tracks;
- teacher detection rate and unknown-mask fraction per object;
- clip/frame residual magnitude separately for ball-present and ball-absent frames;
- event precision/recall by class plus anchor-positive retention;
- learned gate, evidence gate and effective gate means (these are distinct quantities).

The YOLO labels are pseudo-labels, so teacher agreement alone cannot establish better football understanding. Before claiming a positive result, manually annotate a small, fixed ball/goal validation subset that is never used for threshold tuning.

## Resolution contract

The primary v4 run uses one 720x1280 tensor for both DINO and YOLO. This isolates the actual question: whether preserving the small ball before DINO patch embedding raises the event ceiling. Do not mix 512p and 720p samples in one run.

If 720p exceeds the smoke memory ceiling, run a separately named 640x1120 fallback. Because token count and optimization dynamics change with resolution, compare each resolution from the same frozen anchor checkpoint and refit validation thresholds independently.
