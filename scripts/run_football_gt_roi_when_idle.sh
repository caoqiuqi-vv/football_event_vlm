#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/new_users/qiuqi/code/dinov3-main}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-outputs/football_roi_debug/gt_fixed_vs_dynamic_h264_20videos/fixed/manifest.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/football_roi_debug/gt_fixed_vs_dynamic_h264_20videos_all_gt}"
CPU_MAX_PERCENT="${CPU_MAX_PERCENT:-35}"
LOAD_PER_CPU_MAX="${LOAD_PER_CPU_MAX:-0.5}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-60}"
REQUIRED_IDLE_CHECKS="${REQUIRED_IDLE_CHECKS:-5}"
NICE_LEVEL="${NICE_LEVEL:-10}"

cd "${ROOT_DIR}"
mkdir -p "${OUTPUT_DIR}"
WAIT_LOG="${OUTPUT_DIR}/idle_wait.log"
RENDER_LOG="${OUTPUT_DIR}/render.log"
STATUS_FILE="${OUTPUT_DIR}/idle_runner.status"
EXIT_FILE="${OUTPUT_DIR}/render.exit_code"
LOCK_FILE="${OUTPUT_DIR}/.idle_runner.lock"

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "Another idle ROI runner already holds ${LOCK_FILE}" >&2
  exit 2
fi

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

set_status() {
  local message="$1"
  printf '%s %s\n' "$(timestamp)" "${message}" | tee -a "${WAIT_LOG}"
  printf '%s\n' "${message}" > "${STATUS_FILE}"
}

read_cpu_counters() {
  local cpu user nice system idle iowait irq softirq steal guest guest_nice
  read -r cpu user nice system idle iowait irq softirq steal guest guest_nice < /proc/stat
  printf '%s %s\n' \
    "$((user + nice + system + idle + iowait + irq + softirq + steal))" \
    "$((idle + iowait))"
}

if [[ ! -f "${SOURCE_MANIFEST}" ]]; then
  set_status "failed missing_source_manifest=${SOURCE_MANIFEST}"
  exit 2
fi

VIDEO_IDS="$(tail -n +2 "${SOURCE_MANIFEST}" | cut -d, -f2 | sort -u | paste -sd, -)"
if [[ -z "${VIDEO_IDS}" ]]; then
  set_status "failed no_video_ids_in=${SOURCE_MANIFEST}"
  exit 2
fi

CPU_COUNT="$(nproc)"
LOAD_MAX="$(awk -v count="${CPU_COUNT}" -v ratio="${LOAD_PER_CPU_MAX}" 'BEGIN { printf "%.2f", count * ratio }')"
consecutive=0
set_status "waiting cpu_max=${CPU_MAX_PERCENT}% load1_max=${LOAD_MAX} required_checks=${REQUIRED_IDLE_CHECKS} interval=${CHECK_INTERVAL_SEC}s"

while (( consecutive < REQUIRED_IDLE_CHECKS )); do
  read -r total_before idle_before < <(read_cpu_counters)
  sleep "${CHECK_INTERVAL_SEC}"
  read -r total_after idle_after < <(read_cpu_counters)
  delta_total=$((total_after - total_before))
  delta_idle=$((idle_after - idle_before))
  cpu_percent="$(awk -v total="${delta_total}" -v idle="${delta_idle}" 'BEGIN { if (total <= 0) print 100; else printf "%.2f", 100 * (total - idle) / total }')"
  load_one="$(awk '{print $1}' /proc/loadavg)"

  if awk -v cpu="${cpu_percent}" -v cpu_max="${CPU_MAX_PERCENT}" -v load="${load_one}" -v load_max="${LOAD_MAX}" 'BEGIN { exit ! (cpu <= cpu_max && load <= load_max) }'; then
    consecutive=$((consecutive + 1))
  else
    consecutive=0
  fi
  set_status "waiting cpu=${cpu_percent}% load1=${load_one} idle_checks=${consecutive}/${REQUIRED_IDLE_CHECKS}"
done

set_status "running output=${OUTPUT_DIR} video_count=20"
export ROI_COMPARE_OUTPUT="${OUTPUT_DIR}"
export ROI_COMPARE_VIDEO_IDS="${VIDEO_IDS}"
export ROI_COMPARE_LABELS="shot,save,set_piece"
export ROI_COMPARE_MAX_SAMPLES=0
export ROI_COMPARE_MAX_PER_VIDEO=0

set +e
if command -v ionice >/dev/null 2>&1; then
  nice -n "${NICE_LEVEL}" ionice -c 2 -n 7 \
    bash scripts/run_football_da_16f.sh gt_roi_compare 0 >> "${RENDER_LOG}" 2>&1
else
  nice -n "${NICE_LEVEL}" \
    bash scripts/run_football_da_16f.sh gt_roi_compare 0 >> "${RENDER_LOG}" 2>&1
fi
exit_code=$?
set -e
printf '%s\n' "${exit_code}" > "${EXIT_FILE}"
if (( exit_code == 0 )); then
  set_status "completed exit_code=0 output=${OUTPUT_DIR}"
else
  set_status "failed exit_code=${exit_code} log=${RENDER_LOG}"
fi
exit "${exit_code}"
