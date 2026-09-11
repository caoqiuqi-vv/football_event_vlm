#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="/home/new_users/qiuqi/code/dinov3-main"
CONFIG_PATH="${CONFIG_PATH:-configs/football/train_football_events_object_teacher_online_allclips_from_fromlast_e8.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/football_events/vitl16_d7c_object_teacher_online_allclips_no_bad_media_from_fromlast_e8_20260901}"
GPU_LIST="${GPU_LIST:-1,4,6}"
MASTER_PORT="${MASTER_PORT:-29641}"
PYTHON_BIN="${PYTHON_BIN:-/home/new_users/qiuqi/miniconda3/envs/qiuqi_sam3/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/home/new_users/qiuqi/miniconda3/envs/qiuqi_sam3/bin/torchrun}"

cd "$REPO_ROOT" || exit 2
mkdir -p "$OUTPUT_DIR"

LOG_PATH="$OUTPUT_DIR/train_background.log"
PID_PATH="$OUTPUT_DIR/train_background.pid"
STATUS_PATH="$OUTPUT_DIR/train_background_status.json"

exec >>"$LOG_PATH" 2>&1
echo "[$(date -Is)] background launcher start gpu_list=$GPU_LIST config=$CONFIG_PATH"
echo "$$" >"$PID_PATH"

write_status() {
  local state="$1"
  local exit_code="$2"
  "$PYTHON_BIN" - "$STATUS_PATH" "$state" "$exit_code" "$$" "$GPU_LIST" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path, state, exit_code, pid, gpu_list = sys.argv[1:]
Path(path).write_text(
    json.dumps(
        {
            "state": state,
            "exit_code": int(exit_code),
            "launcher_pid": int(pid),
            "gpu_list": gpu_list,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
        indent=2,
    )
    + "\n"
)
PY
}

write_status running 0
export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export PYTHONUNBUFFERED=1

"$TORCHRUN_BIN" \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=3 \
  --master-port="$MASTER_PORT" \
  train_football_events.py \
  --config "$CONFIG_PATH"
EXIT_CODE=$?

if [[ "$EXIT_CODE" -eq 0 ]]; then
  write_status completed "$EXIT_CODE"
else
  write_status failed "$EXIT_CODE"
fi
echo "[$(date -Is)] background launcher exit_code=$EXIT_CODE"
exit "$EXIT_CODE"
