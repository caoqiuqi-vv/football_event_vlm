#!/usr/bin/env bash
set -euo pipefail

repo=/home/new_users/qiuqi/code/dinov3-main
run="$repo/outputs/football_full_review/full166_fromlast_e8_best_dense_s5_20260902"
platform="$run/review_platform_remaining148"
ids="$run/remaining148_video_ids.txt"
adaptive="$platform/adaptive_thresholds_loov_90_85_85_safety.json"
queue="$platform/review_queue_gt_union_model_safety.jsonl"
queue_report="$platform/review_queue_report_safety.json"
base_manifest="$platform/review_manifest_without_team.json"
manifest="$platform/review_manifest.json"
inventory="$run/full_review_inventory.json"
clip_plan="$run/team_calibration/clips/team_calibration_clip_plan.json"
detections="$run/team_calibration/detections"
cluster_input="$run/team_calibration/cluster_input_v2"
team_results="$run/team_calibration/team_results_upper"
log="$platform/pipeline.log"
python=/home/new_users/qiuqi/miniconda3/bin/python
team_python=/home/new_users/qiuqi/miniconda3/envs/qiuqi_sam3/bin/python

mkdir -p "$platform" "$team_results"
exec >>"$log" 2>&1
echo "[$(date -Is)] waiting for leakage-free adaptive threshold report"
while [[ ! -s "$adaptive" ]]; do sleep 30; done

if [[ ! -s "$queue" || ! -s "$queue_report" ]]; then
  "$python" tools/football_event_review/build_full_dense_review_queue.py \
    --run-dir "$run" --video-id-file "$ids" --inventory "$inventory" \
    --adaptive-report "$adaptive" --output "$queue" --report "$queue_report" \
    --max-segment-sec 45 --gt-context-sec 5 --match-tolerance-sec 3 \
    --recall-floors shot=0.90,save=0.85,set_piece=0.85
fi

if [[ ! -s "$base_manifest" ]]; then
  "$python" tools/football_event_review/build_annotation_repair_manifest.py \
    --queue-jsonl "$queue" --run-dir "$run" \
    --video-root /mnt/data_16t/football/raw_video_720P \
    --inventory "$inventory" --output "$base_manifest"
fi

echo "[$(date -Is)] waiting for all 148 detector outputs"
while true; do
  ready=$(find "$detections" -name trajectory_results.json -type f | wc -l)
  [[ "$ready" -ge 148 ]] && break
  sleep 30
done

if [[ $(find "$team_results" -name team_assignments.json -type f 2>/dev/null | wc -l) -lt 148 ]]; then
  cd /home/new_users/qiuqi/code/det_and_track
  "$team_python" team_cluster/run_team_cluster.py \
    --input-root "$cluster_input" --out-dir "$team_results" \
    --clusterer hierarchical --cache-mode off --color-region-mode upper \
    --robust-color-v2 --enhance-lowlight-colors --video-decoder opencv
  cd "$repo"
fi

"$python" tools/football_event_review/enrich_team_evidence.py \
  --manifest "$base_manifest" --team-results-root "$team_results" \
  --clip-plan "$clip_plan" --output "$manifest"

"$python" tools/football_event_review/validate_full_review_manifest.py \
  --manifest "$manifest" --run-dir "$run" --queue "$queue" \
  --queue-report "$queue_report" --output "$platform/review_manifest_validation.json"

if [[ ! -s "$platform/access_config.json" || ! -s "$platform/credentials_private.json" ]]; then
  "$python" tools/football_event_review/create_multiuser_access.py \
    --manifest "$manifest" --output-config "$platform/access_config.json" \
    --output-credentials "$platform/credentials_private.json" \
    --base-url https://119.147.202.180:8775
fi

"$python" -c 'import json,sys; from datetime import datetime,timezone; from pathlib import Path; p=Path(sys.argv[1]); payload={"status":"prepared","created_at":datetime.now(timezone.utc).isoformat(),"manifest":sys.argv[2],"database":sys.argv[3],"server":"tools/football_event_review/server_multiuser_v36.py"}; t=p.with_suffix(".tmp"); t.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n"); t.replace(p)' "$platform/PREPARED.json" "$manifest" "$platform/reviews.sqlite3"
echo "[$(date -Is)] remaining148 v36 platform prepared"
