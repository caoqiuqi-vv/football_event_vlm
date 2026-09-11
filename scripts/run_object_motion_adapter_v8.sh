#!/usr/bin/env bash
set -euo pipefail

# Same v7 teacher/checkpoint/sampling; each mode is a separate experiment.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Tests and training must use the same checkout, even outside the old server path.
export FOOTBALL_REPO_DIR="$(cd -- "${script_dir}/.." && pwd)"
mode="${MOTION_V8_MODE:-grad}"
case "${mode}" in
  control)
    export MOTION_EVENT_RELATION_GRAD=false
    export MOTION_EVENT_CONTEXT=false
    export MOTION_EVENT_FRAME_FUSION=false
    export MOTION_EVENT_CONTEXT_UNIFORM=false
    ;;
  grad)
    export MOTION_EVENT_RELATION_GRAD=true
    export MOTION_EVENT_CONTEXT=false
    export MOTION_EVENT_FRAME_FUSION=false
    export MOTION_EVENT_CONTEXT_UNIFORM=false
    ;;
  context|uniform)
    export MOTION_EVENT_RELATION_GRAD=true
    export MOTION_EVENT_CONTEXT=true
    export MOTION_EVENT_FRAME_FUSION=true
    export MOTION_EVENT_CONTEXT_UNIFORM=false
    if [[ "${mode}" == uniform ]]; then
      export MOTION_EVENT_CONTEXT_UNIFORM=true
    fi
    ;;
  *) echo "MOTION_V8_MODE must be control, grad, context or uniform" >&2; exit 64 ;;
esac
# First isolate the readout; enable dense-feature event gradients only in a
# separately named follow-up. Shared anchor BallLoRA remains as in v7.
export MOTION_EVENT_CONTEXT_FEATURE_GRAD="${MOTION_EVENT_CONTEXT_FEATURE_GRAD:-false}"
export MOTION_EVENT_RANKING_TARGET="${MOTION_EVENT_RANKING_TARGET:-residual}"
export OUTPUT_DIR="${OUTPUT_DIR:-${FOOTBALL_REPO_DIR}/outputs/football_events/object_motion_v8_${mode}_720p_20260911}"

# Run actual tensor/gradient tests before allowing this launcher to train.
python_bin="${PYTHON_BIN:-/home/new_users/qiuqi/miniconda3/envs/qiuqi_sam3/bin/python}"
cd "${script_dir}/.."
"${python_bin}" -m pytest -q tests/test_football_object_motion_v6.py tests/test_football_object_motion_v8.py tests/test_football_object_motion_v8_forward.py
bash "${script_dir}/run_object_motion_adapter_v7_offline_tracks.sh" "$@"
