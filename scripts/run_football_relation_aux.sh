#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
cd "${repo_dir}"

stage="${1:-preflight}"
gpu_list="${GPU_LIST:-6,7}"
python_bin="${PYTHON_BIN:-/home/new_users/qiuqi/miniconda3/envs/qiuqi_sam3/bin/python}"
config="${CONFIG:-configs/football/dinov3_vitl16_mechanism_a_ball_heatmap_16f_720p.yaml}"
seed="${SEED:-42}"
epochs="${EPOCHS:-1}"
per_gpu_batch="${PER_GPU_BATCH_SIZE:-2}"
target_effective_batch="${TARGET_EFFECTIVE_BATCH_SIZE:-48}"
target_head_lr="${TARGET_HEAD_LR:-8e-5}"
target_backbone_lr="${TARGET_BACKBONE_LR:-1e-5}"
object_weight="${OBJECT_LOSS_WEIGHT:-0.45}"
relation_weight="${RELATION_LOSS_WEIGHT:-0.20}"
output_root="${OUTPUT_ROOT:-outputs/football_events/mechanism_r_20260915}"
backbone_frame_chunk="${BACKBONE_FRAME_CHUNK_SIZE:-8}"
gradient_checkpointing="${GRADIENT_CHECKPOINTING:-false}"
workers_per_gpu="${NUM_WORKERS_PER_GPU:-2}"
eval_per_gpu_batch="${EVAL_PER_GPU_BATCH_SIZE:-4}"

gpu_count="$(${python_bin} -c 'import sys; print(len([x for x in sys.argv[1].split(",") if x.strip()]))' "${gpu_list}")"
local_gpu_ids="$(${python_bin} -c 'import sys; n=int(sys.argv[1]); print("["+",".join(map(str,range(n)))+"]")' "${gpu_count}")"
grad_accum="$(${python_bin} -c 'import math,sys; print(max(math.ceil(int(sys.argv[1])/(int(sys.argv[2])*int(sys.argv[3]))),1))' "${target_effective_batch}" "${per_gpu_batch}" "${gpu_count}")"
head_lr_per_gpu="$(${python_bin} -c 'import sys; print(float(sys.argv[1])/int(sys.argv[2]))' "${target_head_lr}" "${gpu_count}")"
backbone_lr_per_gpu="$(${python_bin} -c 'import sys; print(float(sys.argv[1])/int(sys.argv[2]))' "${target_backbone_lr}" "${gpu_count}")"

relation_overrides=(
  "model.mechanism_a.variant=relation_aux"
  "model.object_spatial_aux.teacher_format=tracked_ball_goal_relation_v1"
  "model.object_spatial_aux.goal_teacher_index_root=/mnt/data_16t/football/roi_indices/conditional_goal_crowd_v1"
  "model.object_spatial_aux.goal_max_frame_gap_sec=0.12"
  "model.object_spatial_aux.goal_confidence=0.25"
  "model.object_spatial_aux.goal_class_id=2"
  "model.object_spatial_aux.relation_min_ball_confidence=0.20"
  "model.object_spatial_aux.relation_min_ball_quality=0.75"
  "model.object_spatial_aux.goal_negative_weight=0.05"
  "model.object_spatial_aux.negative_weights=[0.02,0.08]"
  "model.ball_goal_relation_aux.enabled=true"
  "model.ball_goal_relation_aux.topk_candidates=3"
  "model.ball_goal_relation_aux.candidate_nms_radius=2"
  "model.ball_goal_relation_aux.context_radii_patches=[1.5,4.0,8.0]"
  "model.ball_goal_relation_aux.context_loss_weight=0.25"
  "train.object_heatmap_loss_weight=${object_weight}"
  "train.ball_goal_relation_loss_weight=${relation_weight}"
  "eval.candidate_time_mode=window_center"
)

case "${stage}" in
  preflight)
    "${python_bin}" -m py_compile train_football_events.py \
      football_object_spatial_aux.py football_events/ball_goal_relation.py \
      football_events/tracked_ball_teacher.py football_events/mechanism_a.py
    "${python_bin}" -m unittest discover -s tests \
      -p test_football_mechanism_a.py -v
    "${python_bin}" train_football_events.py --config "${config}" --dry-run \
      "${relation_overrides[@]}"
    ;;
  probe)
    output_dir="${output_root}/r0_gradient_probe_seed${seed}"
    mkdir -p "${output_dir}"
    CUDA_VISIBLE_DEVICES="${gpu_list%%,*}" PYTHONUNBUFFERED=1 \
      "${python_bin}" train_football_events.py --config "${config}" \
      "output_dir=${output_dir}" "gpu_ids=[0]" "seed=${seed}" \
      "${relation_overrides[@]}" \
      "model.mechanism_a.gradient_diagnostics.enabled=true" \
      "model.mechanism_a.gradient_diagnostics.exit_after_max_measurements=true" \
      "model.mechanism_a.gradient_diagnostics.interval_steps=5" \
      "model.mechanism_a.gradient_diagnostics.max_measurements_per_epoch=4" \
      "train.epochs=1" "train.max_steps_per_epoch=40" \
      "train.per_gpu_batch_size=1" "train.grad_accum_steps=4" \
      "train.lr_per_gpu=${target_head_lr}" \
      "train.temporal_lr_per_gpu=${target_head_lr}" \
      "train.head_lr_per_gpu=${target_head_lr}" \
      "train.backbone_lr_per_gpu=${target_backbone_lr}" \
      "train.global_backbone_lr_per_gpu=${target_backbone_lr}" \
      "train.ema.enabled=false" "eval.online_validation.enabled=false" \
      2>&1 | tee "${output_dir}/train_console.log"
    ;;
  train)
    output_dir="${output_root}/r1_relation_aux_seed${seed}"
    mkdir -p "${output_dir}"
    command=("${python_bin}")
    if [[ "${gpu_count}" -gt 1 ]]; then
      command+=( -m torch.distributed.run --standalone --nproc_per_node="${gpu_count}" )
    fi
    command+=(
      train_football_events.py --config "${config}"
      "output_dir=${output_dir}" "gpu_ids=${local_gpu_ids}" "seed=${seed}"
      "${relation_overrides[@]}"
      "model.mechanism_a.gradient_diagnostics.enabled=false"
      "model.backbone_frame_chunk_size=${backbone_frame_chunk}"
      "model.gradient_checkpointing=${gradient_checkpointing}"
      "data.num_workers_per_gpu=${workers_per_gpu}"
      "train.epochs=${epochs}" "train.per_gpu_batch_size=${per_gpu_batch}"
      "train.grad_accum_steps=${grad_accum}"
      "train.lr_per_gpu=${head_lr_per_gpu}"
      "train.temporal_lr_per_gpu=${head_lr_per_gpu}"
      "train.head_lr_per_gpu=${head_lr_per_gpu}"
      "train.backbone_lr_per_gpu=${backbone_lr_per_gpu}"
      "train.global_backbone_lr_per_gpu=${backbone_lr_per_gpu}"
      "eval.per_gpu_batch_size=${eval_per_gpu_batch}"
    )
    printf 'physical_gpus=%s world=%s effective_batch=%s head_lr=%s backbone_lr=%s relation_weight=%s batch=%s frame_chunk=%s checkpointing=%s workers_per_gpu=%s eval_batch=%s\n' \
      "${gpu_list}" "${gpu_count}" "$((per_gpu_batch * gpu_count * grad_accum))" \
      "${target_head_lr}" "${target_backbone_lr}" "${relation_weight}" \
      "${per_gpu_batch}" "${backbone_frame_chunk}" "${gradient_checkpointing}" \
      "${workers_per_gpu}" "${eval_per_gpu_batch}" \
      | tee "${output_dir}/launch.txt"
    CUDA_VISIBLE_DEVICES="${gpu_list}" PYTHONUNBUFFERED=1 \
      "${command[@]}" 2>&1 | tee "${output_dir}/train_console.log"
    ;;
  *)
    echo "usage: $0 {preflight|probe|train}" >&2
    exit 64
    ;;
esac
