#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
cd "${repo_dir}"

stage="${1:-preflight}"
gpu_list="${GPU_LIST:-0,1,2,3}"
python_bin="${PYTHON_BIN:-/home/new_users/qiuqi/miniconda3/envs/qiuqi_sam3/bin/python}"
config="${CONFIG:-configs/football/dinov3_vitl16_mechanism_a_ball_heatmap_16f_720p.yaml}"
seed="${SEED:-42}"
epochs="${EPOCHS:-2}"
per_gpu_batch="${PER_GPU_BATCH_SIZE:-1}"
target_effective_batch="${TARGET_EFFECTIVE_BATCH_SIZE:-48}"
target_head_lr="${TARGET_HEAD_LR:-8e-5}"
target_backbone_lr="${TARGET_BACKBONE_LR:-1e-5}"
object_loss_weight="${OBJECT_LOSS_WEIGHT:-0.45}"
output_root="${OUTPUT_ROOT:-outputs/football_events/mechanism_a_20260914}"

gpu_count="$(${python_bin} -c 'import sys; print(len([x for x in sys.argv[1].split(",") if x.strip()]))' "${gpu_list}")"
local_gpu_ids="$(${python_bin} -c 'import sys; n=int(sys.argv[1]); print("["+",".join(map(str,range(n)))+"]")' "${gpu_count}")"
grad_accum="$(${python_bin} -c 'import math,sys; print(max(math.ceil(int(sys.argv[1])/(int(sys.argv[2])*int(sys.argv[3]))),1))' "${target_effective_batch}" "${per_gpu_batch}" "${gpu_count}")"
head_lr_per_gpu="$(${python_bin} -c 'import sys; print(float(sys.argv[1])/int(sys.argv[2]))' "${target_head_lr}" "${gpu_count}")"
backbone_lr_per_gpu="$(${python_bin} -c 'import sys; print(float(sys.argv[1])/int(sys.argv[2]))' "${target_backbone_lr}" "${gpu_count}")"

common_overrides=(
  "gpu_ids=${local_gpu_ids}"
  "seed=${seed}"
  "train.epochs=${epochs}"
  "train.per_gpu_batch_size=${per_gpu_batch}"
  "train.grad_accum_steps=${grad_accum}"
  "train.lr_per_gpu=${head_lr_per_gpu}"
  "train.temporal_lr_per_gpu=${head_lr_per_gpu}"
  "train.head_lr_per_gpu=${head_lr_per_gpu}"
  "train.backbone_lr_per_gpu=${backbone_lr_per_gpu}"
  "train.global_backbone_lr_per_gpu=${backbone_lr_per_gpu}"
)

preflight() {
  "${python_bin}" -m py_compile \
    train_football_events.py \
    football_events/mechanism_a.py \
    football_events/tracked_ball_teacher.py
  "${python_bin}" -m unittest discover -s tests \
    -p test_football_mechanism_a.py -v
  "${python_bin}" -m unittest discover -s tests \
    -p test_football_object_spatial_aux.py -v
  "${python_bin}" train_football_events.py --config "${config}" --dry-run \
    "model.mechanism_a.variant=object_aux" \
    "train.object_heatmap_loss_weight=${object_loss_weight}"
  "${python_bin}" train_football_events.py --config "${config}" --dry-run \
    "model.mechanism_a.variant=control" \
    "model.mechanism_a.gradient_diagnostics.enabled=false" \
    "train.object_heatmap_loss_weight=0"
}

run_probe() {
  local output_dir="${output_root}/a0_gradient_probe_seed${seed}"
  mkdir -p "${output_dir}"
  CUDA_VISIBLE_DEVICES="${gpu_list%%,*}" PYTHONUNBUFFERED=1 \
    "${python_bin}" train_football_events.py --config "${config}" \
    "output_dir=${output_dir}" \
    "gpu_ids=[0]" \
    "seed=${seed}" \
    "model.mechanism_a.variant=object_aux" \
    "model.mechanism_a.gradient_diagnostics.enabled=true" \
    "model.mechanism_a.gradient_diagnostics.exit_after_max_measurements=true" \
    "model.mechanism_a.gradient_diagnostics.interval_steps=5" \
    "model.mechanism_a.gradient_diagnostics.max_measurements_per_epoch=4" \
    "train.object_heatmap_loss_weight=${object_loss_weight}" \
    "train.epochs=1" \
    "train.max_steps_per_epoch=20" \
    "train.per_gpu_batch_size=1" \
    "train.grad_accum_steps=4" \
    "train.lr_per_gpu=${target_head_lr}" \
    "train.temporal_lr_per_gpu=${target_head_lr}" \
    "train.head_lr_per_gpu=${target_head_lr}" \
    "train.backbone_lr_per_gpu=${target_backbone_lr}" \
    "train.global_backbone_lr_per_gpu=${target_backbone_lr}" \
    "train.ema.enabled=false" \
    "train.checkpoint_selection=tuned_macro_precision_at_recall_floor" \
    "eval.online_validation.enabled=false" \
    2>&1 | tee "${output_dir}/train_console.log"
}

run_paired_variant() {
  local variant="$1"
  local weight="$2"
  local teacher_offset="$3"
  local suffix="$4"
  local output_dir="${output_root}/${suffix}_seed${seed}"
  mkdir -p "${output_dir}"
  local command=("${python_bin}")
  if [[ "${gpu_count}" -gt 1 ]]; then
    command+=( -m torch.distributed.run --standalone --nproc_per_node="${gpu_count}" )
  fi
  command+=(
    train_football_events.py --config "${config}"
    "output_dir=${output_dir}"
    "model.mechanism_a.variant=${variant}"
    "model.mechanism_a.gradient_diagnostics.enabled=false"
    "model.object_spatial_aux.teacher_time_offset_sec=${teacher_offset}"
    "train.object_heatmap_loss_weight=${weight}"
    "${common_overrides[@]}"
  )
  printf 'stage=%s physical_gpus=%s world=%s effective_batch=%s head_lr=%s backbone_lr=%s\n' \
    "${suffix}" "${gpu_list}" "${gpu_count}" \
    "$((per_gpu_batch * gpu_count * grad_accum))" \
    "${target_head_lr}" "${target_backbone_lr}" | tee "${output_dir}/launch.txt"
  CUDA_VISIBLE_DEVICES="${gpu_list}" PYTHONUNBUFFERED=1 \
    "${command[@]}" 2>&1 | tee "${output_dir}/train_console.log"
}

run_development() {
  local checkpoint="${CHECKPOINT:-}"
  if [[ -z "${checkpoint}" || ! -f "${checkpoint}" ]]; then
    echo "development requires CHECKPOINT=/path/to/best.pt" >&2
    exit 2
  fi
  local checkpoint_name
  checkpoint_name="$(basename "${checkpoint}" .pt)"
  CUDA_VISIBLE_DEVICES="${gpu_list%%,*}" PYTHONUNBUFFERED=1 \
    "${python_bin}" scripts/evaluate_football_model.py \
    --checkpoint "${checkpoint}" \
    --mode dense \
    --video-id-file configs/football/splits/mechanism_a_20260914/development8.txt \
    --gt-dir /mnt/data_16t/football/football_events_human_repair \
    --video-root xbotgo_0608=/mnt/data_16t/football/raw_video_720P \
    --output-root "${output_root}/development8" \
    --run-name "${checkpoint_name}_development8_pointnms" \
    --thresholds checkpoint \
    --clip-sec 10 \
    --stride-sec 5 \
    --image-size 720,1280 \
    --batch-size "${EVAL_BATCH_SIZE:-2}" \
    --num-workers "${EVAL_NUM_WORKERS:-2}" \
    --device cuda:0 \
    --gpu-ids 0 \
    --prediction-postprocess point_nms \
    --nms-radius-sec 5 \
    --match-tolerance-sec 5 \
    --save-frame-event-logits
}

case "${stage}" in
  preflight)
    preflight
    ;;
  probe)
    run_probe
    ;;
  control)
    run_paired_variant control 0 0 a1_control
    ;;
  aux)
    run_paired_variant object_aux "${object_loss_weight}" 0 a2_ball_aux
    ;;
  shifted)
    run_paired_variant object_aux "${object_loss_weight}" 30 a2s_shifted_teacher
    ;;
  pair)
    run_paired_variant control 0 0 a1_control
    run_paired_variant object_aux "${object_loss_weight}" 0 a2_ball_aux
    ;;
  development)
    run_development
    ;;
  *)
    echo "usage: $0 {preflight|probe|control|aux|shifted|pair|development}" >&2
    exit 64
    ;;
esac
