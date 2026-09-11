# Football QI ROI Crop Tool Export

Standalone package for exporting QI / `RobustClipCropper` ROI-cropped mp4 clips from GT football events.
This directory is meant to be copied as-is to another machine. It does not import training/eval code from the original repository.

## Files

- `crop_gt_events_qi.py` - CLI entry point
- `football_detection_aware.py` - `RobustClipCropper`
- `football_roi_scoring.py` - scoring helpers used by the cropper
- `build_football_roi_indices.py` - compact ROI index builder, including `build_index`
- `requirements.txt` - minimal Python/runtime dependencies

## Inputs

Required:

- `--video-path`: full source video path
- `--video-id`: video id, used for `{video_id}.pt` and detector lookup
- `--gt-json`: key-event JSON. The usual form is `data[]` with `startTime`, `eventType`, and/or `label`.
- `--pad-sec`: symmetric window in seconds, for example `2` or `5`
- `--out-dir`: output root

ROI index input:

- Existing `{index_root}/{video_id}.pt`, or
- Standard detector layout: `{detector_root}/{video_id}/tracked_objects.json` + `metadata.json`, or
- Part layout: `{detector_root}/{video_id}/detection_tracking/{video_id}_part_XXXX/` containing `tracked_objects.json`, `detections.json`, `trajectory_results.json`, etc.

If `{video_id}.pt` is missing, the tool builds it automatically unless `--no-build-missing-index` is passed.
The `.pt` format written/read here has `version = 2`.

## Example

```bash
python crop_gt_events_qi.py \
  --video-path /path/to/VIDEO.mp4 \
  --video-id VIDEO_ID \
  --gt-json /path/to/gt_key_event.json \
  --detector-root /path/to/detection_root \
  --pad-sec 5 \
  --out-dir /path/to/out \
  --index-root /path/to/roi_indices \
  --letterbox 1280x720 \
  --full-frame-fallback
```

Run again with `--pad-sec 2` for the 2-second windows.

## Outputs

For `--pad-sec 5`, the directory is:

```text
{out_dir}/slices_gt_key_event_5s_QIcrop/
```

For `--pad-sec 2`, the directory is:

```text
{out_dir}/slices_gt_key_event_2s_QIcrop/
```

Each event clip is named like:

```text
event_00_SHOT_01m29.7s_pm5s_QIcrop.mp4
```

Each pad directory contains `manifest.json` with event time, label, window, bbox, `roi_mode`, `roi_valid`, fallback, codec, and output file name.

## Encoding

The CLI writes mp4 through ffmpeg using H.264:

```text
-c:v libx264 -pix_fmt yuv420p
```

Install ffmpeg with libx264 support on the target machine.

## Notes

- `--target-size 384,640` only affects ROI bbox fitting/aspect constraints. It does not force the output mp4 size.
- Without `--letterbox`, output videos are source-resolution ROI crops.
- With `--letterbox 1280x720`, each crop is letterboxed into 1280x720.
- With `--full-frame-fallback`, invalid ROI events still produce full-frame clips and are marked in `manifest.json`.
- With part detector outputs, default `--part-frame-mode offset` concatenates part-local frame ids by part order. Use `--part-frame-mode original` if part files already store original source frame ids.

## Common Failures

- Missing `{video_id}.pt` and no `--detector-root`: pass detector output root or prebuild/copy the ROI index.
- `ffmpeg` not found or no libx264: install ffmpeg with H.264 support, or pass `--ffmpeg-bin /path/to/ffmpeg`.
- No GT events loaded: confirm the JSON has `data[]` entries with `startTime`/`timestamp` and `label`/`eventType`.
