# Five-class NMS-free spotting pipeline

Build the Stage-1 4 FPS cache before training:

```bash
PYTHONPATH=football_e2e_spotter/src python football_e2e_spotter/scripts/build_pixel_audio_store.py \
  --config football_e2e_spotter/configs/set_spotter_v1_cache.yaml --split train
PYTHONPATH=football_e2e_spotter/src python football_e2e_spotter/train_set_spotter.py \
  --config football_e2e_spotter/configs/set_spotter_v1.yaml
```

`TemporalSetSpotter` creates independent slots for contiguous 30-second cores.
`decode_slots` and `infer_video_candidates` deliberately contain no temporal NMS,
merging, or cross-class suppression.  Their JSONL candidate records are the OOF
contract for the DINOv3 ViT-L/16 `DinoVerifier`.

The verifier requires exactly 25 global frames ordered as 17 dense centre frames
followed by 8 context frames, at no less than 512x896. Pass a locally constructed
`dinov3_vitl16` backbone to `DinoVerifier`, then call `configure_dinov3_vitl16`
for the two-epoch warmup / LoRA-plus-last-four-block tuning policy.
