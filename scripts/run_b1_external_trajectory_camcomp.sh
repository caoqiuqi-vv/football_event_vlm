#!/usr/bin/env bash
set -euo pipefail

repo_dir="/home/new_users/qiuqi/code/dinov3-main"
python_bin="/home/new_users/qiuqi/miniconda3/envs/qiuqi_sam3/bin/python"
torchrun_bin="/home/new_users/qiuqi/miniconda3/envs/qiuqi_sam3/bin/torchrun"
config_path="${CONFIG_PATH:-${repo_dir}/configs/football/train_football_events_b1_external_trajectory_camcomp_512.yaml}"
index_dir="${INDEX_DIR:-${repo_dir}/outputs/football_external_trajectory_evidence_v2_20260907}"
output_dir="${OUTPUT_DIR:-${repo_dir}/outputs/football_events/vitl16_b1_external_trajectory_camcomp_symgoal_zerogate_4gpu_from_e16v2_20260907}"
gpu_list="${GPU_LIST:-0,1,4,6}"
index_workers="${INDEX_WORKERS:-4}"

yolo_root="/mnt/data_7t/qiuqi/football_ball_pseudolabels/yolo_fulltrack_v2_test18heldout"
rfdetr_root="/home/new_users/qiuqi/code/SoccerMind-main/runs/rf_detr/inference/rfdetr_medium_p2_sahi160_recheck320_hybrid_idstable_relink_detection_tracking_only_20260827_v6_18videos"
legacy_root="/mnt/data_16t/football/detection_and_track_result"

cd "${repo_dir}"
mkdir -p "${index_dir}" "${output_dir}"

"${python_bin}" scripts/build_external_trajectory_evidence.py \
  --yolo-root "${yolo_root}" \
  --rfdetr-root "${rfdetr_root}" \
  --legacy-root "${legacy_root}" \
  --output-dir "${index_dir}" \
  --workers "${index_workers}"

"${python_bin}" -c '
import json,sys
from pathlib import Path
manifest=json.loads((Path(sys.argv[1])/"manifest.json").read_text())
summary=manifest["summary"]
bad={key:value for key,value in summary.items() if key not in {"built","cached"} and value}
ready=summary.get("built",0)+summary.get("cached",0)
if bad or ready != 176:
    raise SystemExit(f"external evidence index incomplete: ready={ready}/176 bad={bad}")
feature_dim=manifest["feature_dim"]
print(f"external_evidence_index=READY videos={ready} feature_dim={feature_dim}")
' "${index_dir}"

IFS=',' read -r -a gpu_ids <<< "${gpu_list}"
gpu_count="${#gpu_ids[@]}"
logical_gpu_ids=""
for ((index=0; index<gpu_count; index++)); do
  logical_gpu_ids+="${logical_gpu_ids:+,}${index}"
done

export CUDA_VISIBLE_DEVICES="${gpu_list}"
export PYTHONPATH="${repo_dir}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

"${torchrun_bin}" --standalone --nproc-per-node="${gpu_count}" \
  train_football_events_online_simulation_e13.py \
  --config "${config_path}" \
  "output_dir=${output_dir}" \
  "gpu_ids=[${logical_gpu_ids}]" \
  "model.evidence_features.index_dir=${index_dir}" \
  2>&1 | tee -a "${output_dir}/train_console.log"
