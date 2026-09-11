#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
mode="${MOTION_V9_MODE:-dense}"
case "${mode}" in
  control) export MOTION_SAMPLING_MODE=legacy_pairs ;;
  dense) export MOTION_SAMPLING_MODE=dense_mixed ;;
  *) echo 'MOTION_V9_MODE must be control or dense' >&2; exit 64 ;;
esac
# A fixed per-rank budget makes control/dense optimizer-step counts comparable.
# Keep it divisible by the default accumulation of 12 and pair unit of 2.
export MOTION_SAMPLING_BATCHES_PER_RANK="${MOTION_SAMPLING_BATCHES_PER_RANK:-2016}"
export MOTION_NATURAL_WINDOW_FRACTION="${MOTION_NATURAL_WINDOW_FRACTION:-0.5}"
# Lock positive weights across the changed inventories; never silently recompute
# an automatic weight from the much larger dense grid. Use the same value in A/B.
export MOTION_POS_WEIGHT="${MOTION_POS_WEIGHT:-[1.0,1.0,1.0]}"
if [[ "${MOTION_POS_WEIGHT}" == auto ]]; then
  echo 'v9 A/B requires fixed MOTION_POS_WEIGHT, not auto' >&2
  exit 64
fi
export MOTION_V8_MODE="${MOTION_V8_MODE:-grad}"
export OUTPUT_DIR="${OUTPUT_DIR:-${repo_dir}/outputs/football_events/object_motion_v9_${mode}_${MOTION_V8_MODE}_720p_20260911}"
python_bin="${PYTHON_BIN:-/home/new_users/qiuqi/miniconda3/envs/qiuqi_sam3/bin/python}"
cd "${repo_dir}"
"${python_bin}" -m unittest discover -s tests -p test_football_object_motion_v9_sampling.py -v
"${python_bin}" -m pytest -q tests/test_football_object_motion_v9_integration.py
exec bash "${script_dir}/run_object_motion_adapter_v8.sh" "$@"
