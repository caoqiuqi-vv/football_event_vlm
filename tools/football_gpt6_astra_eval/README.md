# GPT-6 Astra football event benchmark

This folder defines two reproducible test protocols over a fixed subset of the
repaired `test18` holdout. It deliberately keeps media preparation, model calls,
and scoring separate so that prompts or API adapters cannot silently change the
ground truth or evaluation split.

## Pilot set

The default pilot uses three complete videos:

- `2042520801973841921`: high event density;
- `2041772596969549825`: medium density and heavily audited;
- `2042526457476886530`: low event density.

Ground truth defaults to the immutable repaired v5 per-video export. Only
`shot`, `save`, `free_kick`, `corner`, and `kickoff` are scored. Generic
`set_piece` is not silently mapped to a fine-grained class.

## Protocol A: anchored clip classification

Each 12-second clip has an anchor at 6 seconds. The model predicts every target
class occurring within +/-3 seconds of the anchor. Closely timed cross-class
events therefore form one multi-label sample (for example `shot + save`). The
pilot selects 20 positive anchors and 10 background anchors per video by
default. This protocol tests recognition while making temporal search easy.

## Protocol B: chunked full-video localization

Each complete video is divided into non-overlapping 90-second output cores with
15 seconds of context on both sides (up to 120 seconds of input). The model is
asked to return only events whose defining first touch lies inside the core.
This gives every timestamp exactly one owner and avoids temporal NMS or heuristic
deduplication. Predictions are scored with same-class one-to-one matching at
`+/-1`, `+/-3`, and `+/-5` seconds.

## Prepare

Generate manifests and frozen GT snapshots:

```bash
python tools/football_gpt6_astra_eval/prepare_benchmark.py \
  --output-dir outputs/football_gpt6_astra_eval/test18_pilot_v1
```

Add `--materialize clips`, `--materialize chunks`, or `--materialize all` to
create MP4 files. Video is kept at source resolution and encoded with H.264
CRF 18; audio is retained. Preparation never edits source videos or GT.

## Prediction contracts

Protocol A JSONL:

```json
{"sample_id":"...","labels":["shot","save"],"confidence":{"shot":0.91,"save":0.77}}
```

Protocol B JSONL:

```json
{"chunk_id":"...","events":[{"label":"shot","time_sec_relative_to_chunk":51.2,"confidence":0.88}]}
```

Same-time events are separate objects. Duplicate same-class predictions are not
removed and become false positives under one-to-one scoring.

## Score

```bash
python tools/football_gpt6_astra_eval/score_predictions.py \
  --benchmark-dir outputs/football_gpt6_astra_eval/test18_pilot_v1 \
  --clip-predictions /path/to/clip_predictions.jsonl \
  --chunk-predictions /path/to/chunk_predictions.jsonl \
  --output /path/to/report.json
```

The benchmark report includes per-class precision/recall/F1, localization error,
FP/hour, invalid outputs, and coverage. Missing model responses are reported and
are never treated as successful empty predictions.

