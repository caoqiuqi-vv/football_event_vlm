# Football ROI Crop Module

This note describes the ROI crop code used by the football event model. It is
intended for engineers who want to inspect, modify, or reuse the crop logic.

## Main Files

Core ROI logic:

- `football_detection_aware.py`
  - `RobustClipCropper`: robust clip/dynamic ROI cropper used by the dual-view model.
  - `ROIProposal`: return object for one sampled frame ROI or an aggregated clip summary.
  - `ROI_META_DIM`: metadata vector size used by the ROI gate.

- `football_roi_scoring.py`
  - Candidate scoring helpers for goal, ball, center-circle, and nearby players.
  - Edit this file when changing how anchors are ranked.

Index builder:

- `scripts/build_football_roi_indices.py`
  - Converts large detection/tracking JSON outputs into compact per-video `.pt`
    files consumed by `RobustClipCropper`.

Training / inference integration:

- `train_football_events.py`
  - `FootballLongVideoDataset.__getitem__`: samples exact frame ids first, then calls
    `RobustClipCropper.get_clip_rois` during training.
  - `read_video_segment_views`: accepts frame-aligned ROI boxes, crops each frame, and
    resizes it before model input.

- `scripts/eval_long_video_checkpoint.py`
  - `RobustWindowCropper`: inference wrapper around `RobustClipCropper`.
  - `SlidingWindowVideoDataset.__getitem__`: calls the cropper during long-video eval.

Visualization / standalone export:

- `scripts/football_roi_crop.py`
  - Standalone ROI crop script for reuse/debug.
  - It only loads `RobustClipCropper`, the compact ROI index, and OpenCV.
  - It does not load the event model, training dataset, or evaluation pipeline.
  - It can save every sampled ROI frame for each window, matching model-style
    `num_frames` sampling, and writes `windows.csv`, `frames.csv`, and `summary.json`.

- `scripts/export_val_positive_roi_videos.py`
  - Loads the actual val positive records from a training config.
  - Default `--frame-mode model` exports the exact model-sampled frames with per-frame
    ROI overlay/crop videos and a manually editable manifest.

- `scripts/stack_roi_review_videos.py`
  - Horizontally stacks matching fixed and dynamic review videos.
  - Use this to inspect the same GT 10-second slice and sampled frames side by side.

## Fixed vs Dynamic GT Review

Export every positive validation sample with the fixed and dynamic ROI policies:

```bash
bash scripts/run_football_da_16f.sh gt_roi_compare 0
```

The default output is:

```text
outputs/football_roi_debug/gt_fixed_vs_dynamic/
  fixed/videos/review/
  fixed/videos/crop/
  fixed/manifest.csv
  dynamic/videos/review/
  dynamic/videos/crop/
  dynamic/manifest.csv
  comparison/videos/
  comparison/comparison_manifest.csv
```

Both variants use the same GT-centered 10-second window and the same 16 model
frame timestamps. At the default playback rate of 1.6 fps, each review video is
10 seconds long. Each comparison frame is laid out as fixed original/ROI crop,
then dynamic original/ROI crop.

For a small check before a full export:

```bash
ROI_COMPARE_OUTPUT=outputs/football_roi_debug/gt_fixed_vs_dynamic_smoke \
ROI_COMPARE_MAX_SAMPLES=6 \
ROI_COMPARE_MAX_PER_VIDEO=1 \
bash scripts/run_football_da_16f.sh gt_roi_compare 0
```

- `scripts/export_football_roi_crops.py`
  - Older static quick-inspection helper.
  - Saves one middle-frame crop per window, not all model input frames.

## Standalone ROI Crop Script

Single video, one explicit 10s window, save 16 frames using the legacy static
window proposal. This helper is useful for crop geometry debugging; use
`scripts/export_val_positive_roi_videos.py` with a dynamic config when exact
per-frame dynamic model inputs are required:

```bash
python scripts/football_roi_crop.py \
  --index-root outputs/football_roi_indices/robust_v2 \
  --video-root /mnt/data_16t/football/raw_video_720P \
  --video-id 2027564888428580866 \
  --start-sec 100 \
  --end-sec 110 \
  --num-frames 16 \
  --target-size 384,640 \
  --output-dir outputs/football_roi_crop_debug/2027564888428580866_100_110
```

If you want to pass one compact ROI index file directly, use `--roi-index`; the
`video_id` is inferred from `{video_id}.pt` when `--video-id` is omitted:

```bash
python scripts/football_roi_crop.py \
  --roi-index outputs/football_roi_indices/robust_v2/2027564888428580866.pt \
  --video-root /mnt/data_16t/football/raw_video_720P \
  --start-sec 100 \
  --end-sec 110 \
  --num-frames 16 \
  --target-size 384,640 \
  --output-dir outputs/football_roi_crop_debug/2027564888428580866_100_110
```

Minimal Python ROI computation:

```python
from scripts.football_roi_crop import compute_roi_from_index

proposal = compute_roi_from_index(
    "outputs/football_roi_indices/robust_v2/2027564888428580866.pt",
    start_sec=100.0,
    end_sec=110.0,
    width=1280,
    height=720,
    target_size=(384, 640),
)
print(proposal.bbox, proposal.to_dict())
```

Multiple videos, sliding 10s windows every 10s:

```bash
python scripts/football_roi_crop.py \
  --index-root outputs/football_roi_indices/robust_v2 \
  --video-root /mnt/data_16t/football/raw_video_720P \
  --video-id-file configs/football/splits/detector_aware_seed42_58videos/val_video_ids.txt \
  --clip-sec 10 \
  --stride-sec 10 \
  --num-frames 16 \
  --target-size 384,640 \
  --save-overlay \
  --save-source-crop \
  --output-dir outputs/football_roi_crop_debug/val_10s
```

Output layout:

```text
{output_dir}/
  resized/{video_id}/window_.../frame_...jpg      # resized to target-size
  source_crop/{video_id}/window_.../frame_...jpg  # optional original-resolution crop
  overlay/{video_id}/window_.../frame_...jpg      # optional full frame with bbox
  windows.csv                                     # one row per ROI window
  frames.csv                                      # one row per saved sampled frame
  summary.json
```

Important behavior:

- The saved `resized/` images are resized to `--target-size`, e.g. `384x640`.
- The bbox in `windows.csv` is still original-video coordinates.
- If ROI is invalid, no image is saved by default. Use
  `--save-invalid-full-frame` to save full-frame fallback crops resized to
  `--target-size`.

## Data Flow

The current robust ROI path has two stages.

1. Build compact ROI indices from detection/tracking output:

```bash
python scripts/build_football_roi_indices.py \
  --detector-root /path/to/detection_tracking_root \
  --output-root outputs/football_roi_indices/robust_v2 \
  --split-file configs/football/splits/.../train_video_ids.txt \
  --split-file configs/football/splits/.../val_video_ids.txt
```

Each video produces:

```text
outputs/football_roi_indices/robust_v2/{video_id}.pt
```

2. At training/eval time, sample the exact source frame ids first and request aligned ROIs:

```python
frame_proposals, aggregate = cropper.get_clip_rois(
    video_id,
    start_sec,
    end_sec,
    frame_times,
    width,
    height,
    target_size,  # e.g. (384, 640)
)
```

Each `frame_proposals[t].bbox` is in original-video pixel coordinates. With
`temporal_mode: clip` every item is the same legacy proposal. With
`temporal_mode: dynamic`, each sampled frame gets a short-window proposal followed
by confidence-aware temporal smoothing. `aggregate` is only clip-level metadata;
it is not reused as the crop for every frame.

## Important Size Semantics

The ROI bbox is not fixed at `384x640` in original video coordinates.

Example current checkpoint:

```yaml
video:
  num_frames: 16
  image_size: [384, 640]          # local ROI branch input

spatial_crop:
  global_image_size: [384, 672]   # full-image/global branch input
```

For a valid ROI:

1. `RobustClipCropper` returns a bbox in source-video coordinates.
2. `read_video_segment_views` crops that bbox from each sampled frame.
3. The cropped image is resized to `video.image_size`, for example `384x640`.

For an invalid ROI in dual-view mode:

1. The global branch still receives the full image resized to `global_image_size`.
2. The local branch falls back to the full frame resized to `video.image_size`.
3. `roi_valid=0` and the ROI gate can down-weight the local branch.

## Current ROI Strategy

The current cropper is `RobustClipCropper` in `football_detection_aware.py`.
`spatial_crop.temporal_mode` selects either the legacy `clip` behavior or the new
`dynamic` behavior. Dynamic mode computes one proposal per actual sampled frame
from a short local context, smooths center/scale with a confidence-weighted window
and Kalman state, holds short detector gaps with confidence decay, and invalidates
long gaps. The semantic anchor priority below is reused inside every local proposal.

Priority order:

1. `goal_ball`
   - Use both goal and ball if both anchors are reliable.

2. `goal_players`
   - Use goal plus nearby stable players when goal is reliable.

3. `ball_players`
   - Use ball plus nearby stable players when ball is reliable.

4. `center_circle_players`
   - Fallback anchor for center-circle scenes.

If `goal_ball` is too large to fit the area constraint, the code falls back to
`ball_players` before `goal_players`. This is intentional because ball position
is more directly tied to event timing.

The ROI confidence is label-agnostic and is stored in `ROIProposal.roi_confidence`.
The dual-view model receives this value through `roi_meta`.

## Key Configuration

Example:

```yaml
spatial_crop:
  mode: robust_detector_aware
  index_root: outputs/football_roi_indices/robust_v2
  require_index: true
  output_view: dual
  global_image_size: [512, 896]
  temporal_mode: dynamic
  dynamic_context_sec: 3.0
  temporal_smoothing_window_sec: 1.5
  temporal_max_hold_sec: 1.5
  temporal_confidence_decay_sec: 1.0
  temporal_process_noise: 0.01
  temporal_measurement_noise: 0.10
  temporal_max_center_jump_ratio: 0.30
  temporal_causal: false

  padding: 0.15
  min_crop_area_ratio: 0.0
  max_crop_area_ratio: 0.35
  min_roi_confidence: 0.40

  goal_conf: 0.40
  person_conf: 0.35
  center_circle_conf: 0.35
  raw_ball_conf: 0.10

  min_goal_frames: 3
  min_ball_points: 3
  max_people: 10
  max_cached_videos: 2
```

Common knobs:

- `padding`: expands the final union box before fitting.
- `min_crop_area_ratio`: minimum source-frame area covered by ROI.
- `max_crop_area_ratio`: maximum source-frame area allowed for ROI.
- `min_roi_confidence`: invalidates low-confidence ROI proposals.
- `min_ball_points`: minimum temporal support for ball-track candidates.
- `min_goal_frames`: minimum temporal support for goal/center-circle candidates.
- `max_people`: maximum nearby players retained in the ROI.

## How ROI Fitting Works

The final bbox is produced by `_fit_roi` in `football_detection_aware.py`.

Important behavior:

- The crop keeps the same aspect ratio as `target_size`.
- The source crop size is an integer multiple of `target_size` when possible.
- The crop must fit inside the source frame.
- The crop must satisfy `min_crop_area_ratio` and `max_crop_area_ratio`.
- If too many nearby players make the crop too large, `_fit_ranked_people_roi`
  drops the lowest-priority people first while keeping semantic anchors.

The cropper first tries the smallest integer multiple of `target_size` that contains the selected anchors. For current 1280x720 input and local `384x640`, scale 2 would require `768x1280` and cannot fit vertically, so every valid ROI is a source `384x640` crop. OpenCV still executes the final resize for a uniform code path, but it does not geometrically downsample that valid crop. Larger source crops are possible only on higher-resolution source video or with a smaller model input.

## Minimal Reuse Example

```python
import cv2

from football_detection_aware import RobustClipCropper
from train_football_events import ConfigDict

cfg = ConfigDict({
    "index_root": "outputs/football_roi_indices/robust_v2",
    "padding": 0.15,
    "min_crop_area_ratio": 0.0,
    "max_crop_area_ratio": 0.35,
    "min_roi_confidence": 0.40,
})

video_id = "2027564888428580866"
video_path = "/path/to/2027564888428580866.mp4"
target_size = (384, 640)  # h, w

cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cropper = RobustClipCropper(cfg)
proposal = cropper.get_window_roi(
    video_id=video_id,
    start_sec=100.0,
    end_sec=110.0,
    width=width,
    height=height,
    target_size=target_size,
)

print(proposal.to_dict())

cap.set(cv2.CAP_PROP_POS_MSEC, 105.0 * 1000.0)
ok, frame = cap.read()
cap.release()

if ok and proposal.valid and proposal.bbox is not None:
    x1, y1, x2, y2 = proposal.bbox
    crop = frame[y1:y2, x1:x2]
else:
    crop = frame

model_input_view = cv2.resize(crop, (target_size[1], target_size[0]))
cv2.imwrite("roi_debug.jpg", model_input_view)
```

## How To Modify The Strategy

Most useful edit points:

- Change anchor priority:
  - `RobustClipCropper.get_window_roi`
  - Look for the mode assignment block:
    `goal_ball`, `goal_players`, `ball_players`, `center_circle_players`.

- Change candidate ranking:
  - `football_roi_scoring.py`
  - `rank_ball_candidates`
  - `rank_object_candidates`
  - `select_ball_candidate`
  - `select_goal_candidate`
  - `stable_nearby_people`

- Change fitting and area behavior:
  - `RobustClipCropper._fit_roi`
  - `RobustClipCropper._fit_ranked_people_roi`

- Change training-time ROI noise:
  - `RobustClipCropper.augment`
  - config under `spatial_crop.noise_augmentation`

- Change model input resize:
  - `video.image_size` for ROI/local branch
  - `spatial_crop.global_image_size` for full-image/global branch

## Debugging Outputs

During inference, `scripts/eval_long_video_checkpoint.py` writes ROI metadata in
`window_predictions.csv`:

- `crop_roi`
- `crop_area_ratio`
- `crop_reason`
- `roi_valid`
- `roi_confidence`
- `roi_proposal_mode`
- `roi_goal_score`
- `roi_ball_score`
- `roi_center_circle_score`
- `roi_person_support`

Quick ROI-frame export:

```bash
python scripts/football_roi_crop.py \
  --index-root outputs/football_roi_indices/robust_v2 \
  --video-root /mnt/data/Datasets/Datasets/Football/xbotgo_football_data_0608/videos \
  --video-id 2027564888428580866 \
  --output-dir outputs/football_roi_crops/debug_2027564888428580866 \
  --clip-sec 10 \
  --stride-sec 5 \
  --num-frames 16 \
  --target-size 384,640 \
  --save-invalid-full-frame \
  --save-overlay
```

For a lightweight one-middle-frame preview only, `scripts/export_football_roi_crops.py`
is still available, but it does not save all sampled model-input ROI frames.

## Common Pitfalls

- Do not compare `crop_roi` bbox size with model input size. The bbox is in
  source-video coordinates; the model input is after resize.

- Keep `target_size` consistent between crop generation and the local branch
  model input. The crop fitting uses the target aspect ratio.

- `RobustClipCropper` requires compact `.pt` indices. If a video has no index,
  `get_window_roi` returns invalid/missing ROI or training can fail when
  `require_index=true`.

- Changing ROI policy can change both recall and precision. Re-run both the
  visual crop audit and model eval after modifying anchor priority or confidence.



## Dynamic ROI Evaluation and Experiments

Runtime crop overrides can be evaluated against an existing checkpoint without changing its saved config:

```bash
python scripts/evaluate_football_model.py \
  --checkpoint outputs/football_events/vitl16_robust_dual_16f_e1_frame_det_exp2_from_vitl16_robust_dual_16f_exp4_hr/best.pt \
  --mode dense \
  --video-ids "$VIDEO_IDS" \
  --prediction-postprocess window_overlap \
  --thresholds checkpoint \
  --spatial-crop-mode robust_detector_aware \
  --roi-temporal-mode dynamic \
  --roi-dynamic-context-sec 3.0 \
  --roi-temporal-smoothing-window-sec 1.5 \
  --roi-temporal-max-hold-sec 1.5 \
  --roi-temporal-confidence-decay-sec 1.0
```

Dynamic mode defaults to `spatial_crop.dynamic_fallback_to_clip: true`. If a short dynamic context has no valid proposal after temporal hold expires, that sampled frame uses the fixed ROI computed from the complete 10-second clip. The proposal remains valid and is marked `dynamic_fixed_fallback` (or `partial_dynamic_fixed_fallback` at clip level); full-frame local input is only the final fallback when the complete clip also has no valid ROI.

`window_predictions.csv` records `crop_frame_rois`, `roi_frame_valids`, and `roi_frame_confidences`; `frame_event_logits.csv` records the aligned ROI bbox, valid flag, and confidence for every sampled frame.

Six-video inference-only ablation using the same fixed-ROI-trained checkpoint and checkpoint thresholds:

| crop at inference | precision | recall | F1 |
| --- | ---: | ---: | ---: |
| fixed 10s clip ROI | 35.06% | 80.25% | 48.80% |
| dynamic, context 3s, smoothing 1.5s | 35.25% | 80.25% | 48.98% |

This proves that the dynamic path is active, but the nearly flat metric result also shows that an old model trained on fixed crops cannot establish the value of dynamic crops. The initialization checkpoint used an older 84-train/27-val split, while current experiments use 127/35, so attribution requires a matched fixed-ROI D0 control and dynamic-ROI D1:

```bash
bash scripts/run_football_da_16f.sh d0_fixed_roi 1,2,3,6,7
# Run D1 separately on the same GPU topology and initialization.
bash scripts/run_football_da_16f.sh d1_dynamic_roi 1,2,3,6,7
```

D1 keeps the loaded backbone frozen and changes the ROI temporal policy only. D0 and D1 must start from the same E1 checkpoint; D1 must not initialize from D0. When five A800 GPUs are already running D0, use the co-located D1 profile to preserve effective batch 240 while reducing per-step memory and decode pressure:

```bash
DYNAMIC_ROI_D1_INIT_CHECKPOINT=outputs/football_events/vitl16_robust_dual_16f_e1_frame_det_exp2_from_vitl16_robust_dual_16f_exp4_hr/best.pt \
  bash scripts/run_football_da_16f.sh d1_dynamic_roi_a800 0,1,2,3,4
```

The co-located profile uses per-GPU batch 8, accumulation 6, one worker per GPU, and prefetch 1. Its output is isolated under `vitl16_robust_dual_16f_e1_dynamic_roi_d1_a800_5gpu_colocated`. Every D0/D1/D2 launcher run appends stdout and stderr to `<output_dir>/train_console.log`, including the resolved command, GPU list, timestamps, and exit status. For offline analysis, transfer `train_console.log`, `config.yaml`, `best_metrics.json`, and `metrics_epoch_*.json`; transfer `best.pt` only when long-video inference will be run locally. D2 must start from the final D1 `best.pt` and adds learned frame-level ROI quality fusion:

```bash
bash scripts/run_football_da_16f.sh d2_feature_quality 1,2,3,6,7
```

After the experiment directories are available on one machine, generate the matched comparison report with:

```bash
python scripts/summarize_football_roi_experiments.py \
  --experiment d0=outputs/football_events/vitl16_robust_dual_16f_e1_fixed_roi_d0 \
  --experiment d1=outputs/football_events/vitl16_robust_dual_16f_e1_dynamic_roi_d1_a800_5gpu_colocated \
  --experiment d2=outputs/football_events/vitl16_robust_dual_16f_e1_dynamic_roi_feature_quality_d2_a800_5gpu \
  --output-dir outputs/football_roi_experiments/dynamic_roi_comparison
```

The report writes `comparison.json`, `metrics_long.csv`, and `summary.md`. It checks controlled fields such as split, initialization, image sizes, effective batch, global learning rate, and loss setup separately from the intentional ROI temporal-policy and fusion-head treatments.

Run long-video inference once per checkpoint and derive both evaluation protocols plus ROI branch ablations offline:

```bash
# Set ALLOW_MISSING=1 while D2 is still training.
ALLOW_MISSING=1 \
  bash scripts/evaluate_dynamic_roi_experiments.sh 0
```

The script evaluates the six fixed long videos with checkpoint-tuned thresholds, verifies that all video summaries resolved the same thresholds, and writes separate Window-overlap and PointNMS analyses for `global`, `local`, `fused`, and confidence-gating alternatives. Remove `ALLOW_MISSING=1` for the final D0/D1/D2 comparison so a missing checkpoint fails the run.

Aggregate the generated D0/D1/D2 long-video analyses with:

```bash
python scripts/summarize_dynamic_roi_long_eval.py \
  --eval-run d0=outputs/football_eval_runs/dynamic_roi_d0_6videos_windows_checkpoint_thr \
  --eval-run d1=outputs/football_eval_runs/dynamic_roi_d1_6videos_windows_checkpoint_thr \
  --eval-run d2=outputs/football_eval_runs/dynamic_roi_d2_6videos_windows_checkpoint_thr \
  --output-dir outputs/football_roi_experiments/dynamic_roi_long_eval
```

This writes fused P/R/F1, per-video metrics, `fused-global` ROI gain, `fused-gate_no_confidence` confidence gain, and the best micro-F1 fusion branch for both protocols.

The six-video confidence ablation supports retaining the current linear confidence gate for D1. Removing confidence or using square-root/squared confidence did not consistently improve all labels; hard confidence switching reduced recall, especially for `set_piece`.

Use the same launcher for both long-video protocols. The checkpoint, six-video list, stride, tolerance, crop mode, and thresholds stay fixed; only post-processing changes:

```bash
# Strict event spotting, closest to online output.
DENSE_EVAL_POSTPROCESS=point_nms \
  bash scripts/run_football_da_16f.sh d1_dynamic_roi_eval 0

# Sliding-window classifier quality.
DENSE_EVAL_POSTPROCESS=window_overlap \
  bash scripts/run_football_da_16f.sh d1_dynamic_roi_eval 0
```

Replace `d1_dynamic_roi_eval` with `d0_fixed_roi_eval` or `d2_feature_quality_eval` for the matched controls. Output run names automatically contain `point_nms` or `window_overlap`, so the two protocols cannot overwrite each other. `DENSE_EVAL_BATCH_SIZE` and `DENSE_EVAL_NUM_WORKERS` may be adjusted for the evaluation host without changing metric semantics.
